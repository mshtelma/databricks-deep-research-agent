"""Web crawl tool — fetches a page and extracts readable text.

Implements the ``ResearchTool`` protocol with constructor DI.  Dependencies
(the actual crawling callable, timeouts, content limits) are injected at
construction time so the tool is fully testable without monkey-patching.

The tool resolves URLs through a ``UrlRegistry`` — the LLM only ever sees
integer indices produced by earlier ``web_search`` calls, preventing
hallucinated-URL injection.

When no custom *crawler* callable is supplied the tool uses **httpx** (from
the ``[web]`` extra) for HTTP fetching and **trafilatura** (from the
``[crawl]`` extra) for HTML-to-text extraction.  Both are optional
dependencies; a clear error is raised at execution time if they are missing.
"""

from __future__ import annotations

import ipaddress
import logging
import random
import socket
from typing import Any, Protocol
from urllib.parse import urlparse

from databricks_deep_research.tools.protocol import (
    SourceInfo,
    ToolContext,
    ToolDefinition,
    ToolResult,
)

__all__ = ["WebCrawlTool"]

logger = logging.getLogger(__name__)

# Rotate user-agents to reduce bot-detection blocking.
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) "
    "Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.2 Safari/605.1.15",
]

_BLOCKED_HOSTNAMES = frozenset({"metadata.google.internal", "metadata.goog"})


def _resolve_and_validate_ip(hostname: str) -> tuple[bool, str | None]:
    """Resolve *hostname* via DNS and validate that all IPs are public.

    Returns ``(is_safe, first_public_ip)`` where *is_safe* is ``True`` only
    when **every** resolved address is a public, routable IP.  The function
    is **fail-closed**: DNS failures, empty results, and mixed public/private
    resolution all return ``(False, None)``.
    """
    if not hostname:
        return False, None
    if hostname in _BLOCKED_HOSTNAMES:
        return False, None
    try:
        addrinfos = socket.getaddrinfo(hostname, None)
        if not addrinfos:
            return False, None
        first_public: str | None = None
        for _family, _type, _proto, _canonname, sockaddr in addrinfos:
            ip = ipaddress.ip_address(sockaddr[0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False, None
            if first_public is None:
                first_public = str(ip)
        return True, first_public
    except (socket.gaierror, ValueError, OSError):
        return False, None  # Fail-closed: unresolvable → blocked


def _is_private_url(hostname: str) -> bool:
    """Check if hostname resolves to a private/internal IP (SSRF protection)."""
    is_safe, _ = _resolve_and_validate_ip(hostname)
    return not is_safe


def _pin_url_to_ip(url: str, resolved_ip: str) -> tuple[str, str]:
    """Rewrite *url* netloc to *resolved_ip*, returning ``(pinned_url, original_host)``.

    Strips userinfo from the netloc for safety (no credential leakage).
    IPv6 addresses are wrapped in brackets per RFC 2732.
    """
    parsed = urlparse(url)
    original_host = parsed.hostname or ""
    port = parsed.port

    ip_host = f"[{resolved_ip}]" if ":" in resolved_ip else resolved_ip
    new_netloc = f"{ip_host}:{port}" if port else ip_host

    pinned = parsed._replace(netloc=new_netloc).geturl()
    return pinned, original_host


_ALLOWED_REDIRECT_SCHEMES = frozenset({"http", "https"})


async def _check_redirect_ssrf(response: Any) -> None:
    """Block HTTP redirects to private/internal IPs and disallowed schemes."""
    if response.is_redirect:
        location = response.headers.get("location", "")
        parsed_loc = urlparse(location)

        # Block non-HTTP(S) schemes (e.g. file://, gopher://)
        if parsed_loc.scheme and parsed_loc.scheme not in _ALLOWED_REDIRECT_SCHEMES:
            import httpx as _httpx

            raise _httpx.TooManyRedirects(
                f"Redirect to disallowed scheme blocked: {parsed_loc.scheme}",
                request=response.request,
            )

        if parsed_loc.hostname and _is_private_url(parsed_loc.hostname):
            import httpx as _httpx

            raise _httpx.TooManyRedirects(
                f"Redirect to private IP blocked: {parsed_loc.hostname}",
                request=response.request,
            )


_NON_RETRYABLE_FAILURES = {
    "http_403",
    "http_404",
    "robots_denied",
    "empty_content",
    "low_quality_content",
}


def _classify_crawl_exception(exc: Exception) -> str:
    """Classify crawl failures into stable buckets for suppression."""
    message = str(exc).lower()
    status_code = getattr(getattr(exc, "response", None), "status_code", None)
    if status_code == 403:
        return "http_403"
    if status_code == 404:
        return "http_404"
    if "robots" in message:
        return "robots_denied"
    if "403" in message and "forbidden" in message:
        return "http_403"
    if "404" in message and "not found" in message:
        return "http_404"
    if "timeout" in message:
        return "timeout"
    if "429" in message:
        return "http_429"
    if "500" in message or "502" in message or "503" in message or "504" in message:
        return "http_5xx"
    return "request_error"


# ---------------------------------------------------------------------------
# Crawler protocol — allows custom implementations via constructor DI
# ---------------------------------------------------------------------------


class ContentCrawler(Protocol):
    """Protocol for the crawling callable injected into ``WebCrawlTool``.

    Implementations receive a URL and return ``(text, title | None)``.
    They may raise on network / parsing errors — the tool catches exceptions.
    """

    async def __call__(self, url: str) -> tuple[str, str | None]:
        """Fetch *url* and return ``(extracted_text, page_title)``."""
        ...


# ---------------------------------------------------------------------------
# Default crawler (httpx + trafilatura)
# ---------------------------------------------------------------------------


async def _default_crawl(
    url: str,
    *,
    timeout: float = 30.0,
    max_content_length: int = 50_000,
    resolved_ip: str | None = None,
) -> tuple[str, str | None]:
    """Fetch *url* with httpx, extract text with trafilatura.

    When *resolved_ip* is provided and the URL scheme is ``http``, the
    request is pinned to the pre-resolved IP to prevent DNS rebinding
    between validation and fetch.

    Raises:
        ImportError: If ``httpx`` or ``trafilatura`` are not installed.
        httpx.HTTPStatusError: On 4xx/5xx responses.
        httpx.RequestError: On connection / timeout errors.
    """
    try:
        import httpx  # noqa: F811  — optional [web] extra
    except ImportError as exc:
        raise ImportError(
            "httpx is required for the default web crawler. "
            "Install it with: pip install 'databricks-deep-research[web]'"
        ) from exc

    user_agent = random.choice(_USER_AGENTS)  # noqa: S311
    headers: dict[str, str] = {"User-Agent": user_agent}
    fetch_url = url
    if resolved_ip and urlparse(url).scheme == "http":
        fetch_url, original_host = _pin_url_to_ip(url, resolved_ip)
        headers["Host"] = original_host

    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        event_hooks={"response": [_check_redirect_ssrf]},
    ) as client:
        response = await client.get(fetch_url, headers=headers)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type and "text/plain" not in content_type:
            raise ValueError(f"Unsupported content type: {content_type}")

        html = response.text[: max_content_length * 2]

    # Extract readable text -------------------------------------------------
    text, title = _extract_with_trafilatura(html, url)

    # Layer 1: Rescue tables from HTML BEFORE trafilatura destroys them.
    # Embed clean markdown tables into the text so Layer 2 (in execute())
    # can detect them regardless of crawler type.
    try:
        from databricks_deep_research.tools.builtins.html_tables import (
            extract_tables_from_html,
            truncate_markdown_table,
        )

        parsed_tables = extract_tables_from_html(html)
        if parsed_tables:
            table_section = "\n\n---\n\n## Tables Found on Page\n\n"
            for i, pt in enumerate(parsed_tables):
                md = truncate_markdown_table(pt.markdown, max_rows=30)
                table_section += (
                    f"### Table {i + 1} ({pt.row_count}\u00d7{pt.col_count})\n\n"
                    f"{md}\n\n"
                )
            text += table_section
    except Exception:
        logger.debug("WEB_CRAWL_TABLE_EXTRACTION_FAILED url=%s", url[:80], exc_info=True)

    if len(text) > max_content_length:
        text = text[:max_content_length] + "..."

    return text, title


def _extract_with_trafilatura(html: str, base_url: str) -> tuple[str, str | None]:
    """Extract text and title using trafilatura (optional ``[crawl]`` extra).

    Falls back to raw HTML truncation when trafilatura is not installed.
    """
    try:
        from trafilatura import bare_extraction
    except ImportError:
        logger.warning(
            "trafilatura not installed — returning raw HTML. "
            "Install with: pip install 'databricks-deep-research[crawl]'"
        )
        # Crude fallback: strip tags and return first chunk.
        import re

        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        return text, None

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
    return doc.text or "", doc.title  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# WebCrawlTool — ResearchTool implementation
# ---------------------------------------------------------------------------


class WebCrawlTool:
    """Fetch and extract readable text from a web page.

    Implements the ``ResearchTool`` protocol.  The LLM provides an integer
    *url_index* (from prior ``web_search`` results) which is resolved via
    the shared ``UrlRegistry`` in the ``ToolContext``.

    Constructor DI Parameters:
        crawler: Optional async callable ``(url) -> (text, title)``.
            When *None*, the built-in httpx + trafilatura pipeline is used.
        timeout: HTTP timeout in seconds (used by the default crawler).
        max_content_length: Maximum extracted text length in characters.
    """

    def __init__(
        self,
        crawler: ContentCrawler | None = None,
        *,
        timeout: float = 30.0,
        max_content_length: int = 50_000,
        extract_tables: bool = True,
    ) -> None:
        self._crawler = crawler
        self._timeout = timeout
        self._max_content_length = max_content_length
        self._extract_tables = extract_tables

        self._definition = ToolDefinition(
            name="web_crawl",
            description=(
                "Fetch full content from a source. Use only an INDEX number "
                "from prior web_search results in this same workflow run "
                "(0, 1, 2, etc.); call web_search first when no valid index "
                "is available. Returns extracted page text for analysis."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url_index": {
                        "type": "integer",
                        "description": (
                            "Index number of the source from search results "
                            "(0, 1, 2, etc.)"
                        ),
                    },
                },
                "required": ["url_index"],
            },
            source_type="web_crawl",
            source_kind="web",
        )

    # -- ResearchTool protocol -----------------------------------------------

    @property
    def definition(self) -> ToolDefinition:
        """Tool definition for LLM function calling."""
        return self._definition

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Validate and clean arguments.

        Returns:
            Dict with a validated ``url_index`` key.

        Raises:
            ValueError: If *url_index* is missing, non-integer, or negative.
        """
        raw_index = arguments.get("url_index")

        if raw_index is None:
            raise ValueError("'url_index' is required")

        # Coerce stringified ints that LLMs sometimes produce.
        if isinstance(raw_index, str):
            try:
                raw_index = int(raw_index)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"'url_index' must be an integer, got {raw_index!r}"
                ) from exc

        if not isinstance(raw_index, int):
            raise ValueError(f"'url_index' must be an integer, got {type(raw_index).__name__}")

        if raw_index < 0:
            raise ValueError(f"'url_index' must be non-negative, got {raw_index}")

        return {"url_index": raw_index}

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Fetch the page at *url_index* and return extracted text.

        Args:
            arguments: Validated dict with ``url_index``.
            context: Execution context carrying the shared ``UrlRegistry``.

        Returns:
            ``ToolResult`` with page text and source metadata.
        """
        url_index: int = arguments["url_index"]

        # -- resolve URL via registry ----------------------------------------
        registry = context.url_registry
        if registry is None:
            return ToolResult(
                content="No URL registry available — cannot resolve url_index.",
                success=False,
                error="URL registry not available",
            )

        url = registry.resolve(url_index)
        if url is None:
            return ToolResult(
                content=(
                    f"No URL found for index {url_index}. "
                    f"Valid indices: 0–{len(registry) - 1}."
                ),
                success=False,
                error=f"Invalid url_index: {url_index}",
            )

        # -- basic URL validation --------------------------------------------
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return ToolResult(
                content=f"Invalid URL scheme: {parsed.scheme}",
                success=False,
                error="Invalid URL scheme",
            )

        # SSRF protection: resolve DNS once, validate, pin HTTP connections
        is_safe, resolved_ip = _resolve_and_validate_ip(parsed.hostname or "")
        if not is_safe:
            return ToolResult(
                content="URL resolves to a private/internal address and cannot be accessed.",
                success=False,
                error="SSRF blocked: private IP",
            )

        suppressed_failure = registry.get_failure(url)
        if suppressed_failure is not None:
            scope = suppressed_failure["scope"]
            failure_class = suppressed_failure["failure_class"]
            logger.info(
                "WEB_CRAWL_SKIP_CACHED_FAILURE url=%s scope=%s failure_class=%s",
                url[:120],
                scope,
                failure_class,
            )
            return ToolResult(
                content=(
                    "Skipping URL because this workflow already observed a repeated "
                    f"non-retryable crawl failure ({failure_class}, scope={scope})."
                ),
                success=False,
                error="Suppressed repeated crawl failure",
                data={
                    "url": url,
                    "url_index": url_index,
                    "suppressed_by_failure_cache": True,
                    "suppression_scope": scope,
                    "failure_class": failure_class,
                },
            )

        # -- fetch and extract -----------------------------------------------
        try:
            if self._crawler is not None:
                text, title = await self._crawler(url)
            else:
                text, title = await _default_crawl(
                    url,
                    timeout=self._timeout,
                    max_content_length=self._max_content_length,
                    resolved_ip=resolved_ip,
                )
        except Exception as exc:
            failure_class = _classify_crawl_exception(exc)
            if failure_class in _NON_RETRYABLE_FAILURES:
                registry.record_non_retryable_failure(url, failure_class)
            logger.warning("WEB_CRAWL_FAILED url=%s error=%s", url[:80], exc)
            return ToolResult(
                content=f"Failed to fetch page: {exc}",
                success=False,
                error=str(exc),
                data={
                    "url": url,
                    "url_index": url_index,
                    "suppressed_by_failure_cache": False,
                    "failure_class": failure_class,
                },
            )

        if not text:
            registry.record_non_retryable_failure(url, "empty_content")
            return ToolResult(
                content="Page fetched but no readable text could be extracted.",
                success=False,
                error="Empty content after extraction",
                data={
                    "url": url,
                    "url_index": url_index,
                    "suppressed_by_failure_cache": False,
                    "failure_class": "empty_content",
                },
            )

        # Minimum content quality gate — reject error pages, CAPTCHAs, etc.
        _MIN_USEFUL_CONTENT_LENGTH = 100
        if len(text.strip()) < _MIN_USEFUL_CONTENT_LENGTH:
            registry.record_non_retryable_failure(url, "low_quality_content")
            return ToolResult(
                content=f"Page content too short ({len(text.strip())} chars) — likely blocked or error page.",
                success=False,
                error="Content below minimum quality threshold",
                data={
                    "url": url,
                    "url_index": url_index,
                    "suppressed_by_failure_cache": False,
                    "failure_class": "low_quality_content",
                },
            )

        registry.clear_failure(url)

        # -- Layer 2: detect markdown tables in text (ALL crawler types) -----
        detected_tables: list[dict[str, Any]] = []
        if self._extract_tables:
            try:
                from databricks_deep_research.tools.builtins.text_utils import (
                    detect_markdown_tables,
                )

                for pt in detect_markdown_tables(text):
                    detected_tables.append({
                        "markdown": pt.markdown,
                        "table_json": pt.table_json,
                        "row_count": pt.row_count,
                        "col_count": pt.col_count,
                    })
            except Exception:
                logger.debug(
                    "WEB_CRAWL_TABLE_DETECT_FAILED url=%s", url[:80], exc_info=True,
                )

        # -- build result ----------------------------------------------------
        title_display = title or "Web Page"
        formatted = f"# {title_display}\n\nURL: {url}\n\n{text}"

        sources = [
            SourceInfo(
                url=url,
                title=title or url,
                snippet=text[:200],
                source_type="web",
            ),
        ]

        data: dict[str, Any] = {
            "url": url,
            "title": title,
            "content_length": len(text),
            "url_index": url_index,
            "suppressed_by_failure_cache": False,
            "failure_class": "",
        }
        if detected_tables:
            # Register tables in shared TableRegistry for structured access
            if context.table_registry is not None:
                for tbl in detected_tables:
                    try:
                        tbl_idx = context.table_registry.register(
                            tbl["table_json"],
                            source_kind="web",
                            source_label=url,
                            markdown=tbl["markdown"],
                        )
                        tbl["table_idx"] = tbl_idx
                    except ValueError:
                        logger.warning(
                            "WEB_CRAWL_TABLE_REGISTER_SKIPPED url=%s reason=capacity",
                            url[:80],
                        )
                        break  # registry full — skip remaining tables
                # Annotate formatted output with table indices
                table_lines = [
                    f"  Table {i + 1} [table_idx={t['table_idx']}] "
                    f"({t['row_count']}x{t['col_count']})"
                    for i, t in enumerate(detected_tables)
                    if "table_idx" in t
                ]
                if table_lines:
                    formatted += (
                        "\n\n---\nDetected tables:\n" + "\n".join(table_lines)
                    )
            data["tables"] = detected_tables
            data["table_count"] = len(detected_tables)

        return ToolResult(
            content=formatted,
            success=True,
            sources=sources,
            data=data,
        )
