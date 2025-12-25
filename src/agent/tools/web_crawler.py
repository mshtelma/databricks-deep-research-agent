"""Web crawler tool for fetching and parsing web pages."""

import asyncio
import time
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
import mlflow
from trafilatura import bare_extraction

from src.core.logging_utils import (
    get_logger,
    log_crawl_error,
    log_crawl_request,
    log_crawl_response,
)

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

    async def _fetch_url(self, url: str) -> CrawlResult:
        """Fetch and parse a single URL."""
        async with self._semaphore:
            log_crawl_request(logger, url=url)
            start_time = time.perf_counter()

            try:
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


@mlflow.trace(name="web_crawl", span_type="TOOL")
async def web_crawl(
    urls: list[str],
    crawler: WebCrawler,
) -> CrawlOutput:
    """Crawl multiple URLs and extract content.

    Args:
        urls: List of URLs to crawl.
        crawler: WebCrawler instance (injected via DI).

    Returns:
        CrawlOutput with crawl results.
    """
    logger.info(f"Crawling {len(urls)} URLs")

    output = await crawler.crawl(urls)

    logger.info(f"Crawled {output.successful_count} successfully, {output.failed_count} failed")

    return output
