"""Web crawler tool for fetching and parsing web pages."""

import asyncio
import ipaddress
import socket
import time
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
import mlflow
from mlflow.entities import SpanType
from trafilatura import bare_extraction

from src.core.app_config import get_app_config
from src.core.logging_utils import (
    get_logger,
    log_crawl_error,
    log_crawl_request,
    log_crawl_response,
)
from src.core.tracing_constants import (
    ATTR_CRAWL_FAILED,
    ATTR_CRAWL_SUCCESSFUL,
    ATTR_CRAWL_URLS_COUNT,
    list_to_attr,
    tool_span_name,
)
from src.services.search.domain_filter import DomainFilter

logger = get_logger(__name__)

# Maximum content length to fetch (50KB)
MAX_CONTENT_LENGTH = 50000

# Timeout for fetching pages
FETCH_TIMEOUT = 10.0

# User agent for requests
USER_AGENT = (
    "Mozilla/5.0 (compatible; DeepResearchBot/1.0; "
    "+https://databricks.com/deep-research-agent)"
)

# Private IP ranges for SSRF protection
PRIVATE_IP_RANGES = [
    ipaddress.ip_network("127.0.0.0/8"),      # Loopback
    ipaddress.ip_network("10.0.0.0/8"),       # Private Class A
    ipaddress.ip_network("172.16.0.0/12"),    # Private Class B
    ipaddress.ip_network("192.168.0.0/16"),   # Private Class C
    ipaddress.ip_network("169.254.0.0/16"),   # Link-local
    ipaddress.ip_network("::1/128"),          # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),         # IPv6 unique local
    ipaddress.ip_network("fe80::/10"),        # IPv6 link-local
]


def _is_private_ip(hostname: str) -> bool:
    """Check if hostname resolves to a private/internal IP address.

    Args:
        hostname: The hostname to check.

    Returns:
        True if the hostname resolves to a private IP, False otherwise.
    """
    try:
        # Resolve hostname to IP addresses
        addr_info = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC)
        for family, _, _, _, sockaddr in addr_info:
            ip_str = sockaddr[0]
            try:
                ip = ipaddress.ip_address(ip_str)
                # Check if IP is in any private range
                if any(ip in network for network in PRIVATE_IP_RANGES):
                    return True
                # Also check is_private property (catches some edge cases)
                if ip.is_private or ip.is_loopback or ip.is_link_local:
                    return True
            except ValueError:
                continue
        return False
    except socket.gaierror:
        # DNS resolution failed - allow the request to fail naturally
        return False


@dataclass
class CrawlResult:
    """Result from crawling a single URL."""

    url: str
    title: str | None
    content: str
    success: bool
    error: str | None = None


@dataclass
class CrawlOutput:
    """Output from crawling multiple URLs."""

    results: list[CrawlResult]
    successful_count: int
    failed_count: int


class WebCrawler:
    """Async web crawler for fetching and parsing pages."""

    def __init__(self, max_concurrent: int = 5):
        """Initialize web crawler.

        Args:
            max_concurrent: Maximum concurrent requests.
        """
        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._client = httpx.AsyncClient(
            timeout=FETCH_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )

        # Domain filtering (second line of defense after search filtering)
        search_config = get_app_config().search
        self._domain_filter = DomainFilter(search_config.domain_filter)

    async def _fetch_url(self, url: str) -> CrawlResult:
        """Fetch and parse a single URL."""
        async with self._semaphore:
            log_crawl_request(logger, url=url)
            start_time = time.perf_counter()

            try:
                # Domain filter check (second line of defense)
                if self._domain_filter.is_active:
                    match_result = self._domain_filter.is_allowed(url)
                    if not match_result.allowed:
                        logger.warning(
                            "CRAWL_DOMAIN_BLOCKED",
                            url=url[:80],
                            reason=match_result.reason,
                        )
                        return CrawlResult(
                            url=url,
                            title=None,
                            content="",
                            success=False,
                            error=f"Domain not allowed: {match_result.reason}",
                        )

                # Validate URL
                parsed = urlparse(url)
                if parsed.scheme not in ("http", "https"):
                    error = ValueError("Invalid URL scheme")
                    log_crawl_error(logger, url=url, error=error)
                    return CrawlResult(
                        url=url,
                        title=None,
                        content="",
                        success=False,
                        error="Invalid URL scheme",
                    )

                # SSRF protection: reject requests to private/internal IPs
                hostname = parsed.hostname
                if hostname and _is_private_ip(hostname):
                    error = ValueError("Access to private IP ranges is not allowed")
                    log_crawl_error(logger, url=url, error=error)
                    return CrawlResult(
                        url=url,
                        title=None,
                        content="",
                        success=False,
                        error="Access to private IP ranges is not allowed",
                    )

                # Fetch page
                response = await self._client.get(url)
                response.raise_for_status()

                duration_ms = (time.perf_counter() - start_time) * 1000

                # Check content type
                content_type = response.headers.get("content-type", "")
                if "text/html" not in content_type and "text/plain" not in content_type:
                    logger.warning(
                        "CRAWL_UNSUPPORTED_TYPE",
                        url=url[:80],
                        content_type=content_type,
                    )
                    return CrawlResult(
                        url=url,
                        title=None,
                        content="",
                        success=False,
                        error=f"Unsupported content type: {content_type}",
                    )

                # Parse HTML
                html = response.text[:MAX_CONTENT_LENGTH * 2]  # Limit HTML size
                content, title = self._extract_content(html, url)

                # Truncate content if needed
                if len(content) > MAX_CONTENT_LENGTH:
                    content = content[:MAX_CONTENT_LENGTH] + "..."

                # Log successful crawl
                log_crawl_response(
                    logger,
                    url=url,
                    status_code=response.status_code,
                    content_length=len(content),
                    duration_ms=duration_ms,
                )

                return CrawlResult(
                    url=url,
                    title=title,
                    content=content,
                    success=True,
                )

            except httpx.HTTPStatusError as e:
                log_crawl_error(logger, url=url, error=e)
                return CrawlResult(
                    url=url,
                    title=None,
                    content="",
                    success=False,
                    error=f"HTTP {e.response.status_code}",
                )
            except httpx.RequestError as e:
                log_crawl_error(logger, url=url, error=e)
                return CrawlResult(
                    url=url,
                    title=None,
                    content="",
                    success=False,
                    error=str(e),
                )
            except Exception as e:
                log_crawl_error(logger, url=url, error=e)
                return CrawlResult(
                    url=url,
                    title=None,
                    content="",
                    success=False,
                    error=str(e),
                )

    def _extract_content(self, html: str, base_url: str) -> tuple[str, str | None]:
        """Extract text content and title from HTML using trafilatura.

        Trafilatura automatically handles boilerplate removal (headers, footers, ads,
        navigation) using sophisticated heuristics, achieving 0.909 F-Score in
        content extraction benchmarks.

        Args:
            html: HTML content.
            base_url: Base URL for metadata extraction and link resolution.

        Returns:
            Tuple of (content, title).
        """
        # bare_extraction returns a Document object in trafilatura 2.0+
        # With as_dict=False (default), returns Document with .title and .text attributes
        doc = bare_extraction(
            html,
            url=base_url,
            include_comments=False,
            include_tables=True,
            include_links=False,
            with_metadata=True,
            as_dict=False,
        )

        if doc is None:
            return "", None

        # Document has .title and .text attributes directly
        # Type narrowing: as_dict=False guarantees Document, not dict
        return doc.text or "", doc.title  # type: ignore[union-attr]

    async def crawl(self, urls: list[str]) -> CrawlOutput:
        """Crawl multiple URLs concurrently.

        Args:
            urls: List of URLs to crawl.

        Returns:
            CrawlOutput with results.
        """
        # Deduplicate URLs
        unique_urls = list(dict.fromkeys(urls))

        # Fetch all URLs concurrently
        tasks = [self._fetch_url(url) for url in unique_urls]
        results = await asyncio.gather(*tasks)

        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful

        return CrawlOutput(
            results=list(results),
            successful_count=successful,
            failed_count=failed,
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()


async def web_crawl(
    urls: list[str],
    crawler: WebCrawler,
    context: str | None = None,
) -> CrawlOutput:
    """Crawl multiple URLs and extract content.

    Args:
        urls: List of URLs to crawl.
        crawler: WebCrawler instance (injected via DI).
        context: Optional context for span naming (e.g., "step_1", "background").

    Returns:
        CrawlOutput with crawl results.
    """
    span_name = tool_span_name("web_crawl", context)

    with mlflow.start_span(name=span_name, span_type=SpanType.TOOL) as span:
        span.set_attributes({
            ATTR_CRAWL_URLS_COUNT: len(urls),
            "crawl.urls": list_to_attr(urls, max_items=5),
        })

        logger.info(f"Crawling {len(urls)} URLs")

        output = await crawler.crawl(urls)

        logger.info(f"Crawled {output.successful_count} successfully, {output.failed_count} failed")

        # Set output attributes
        span.set_attributes({
            ATTR_CRAWL_SUCCESSFUL: output.successful_count,
            ATTR_CRAWL_FAILED: output.failed_count,
        })

        return output
