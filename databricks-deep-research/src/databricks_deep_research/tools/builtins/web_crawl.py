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

import logging
import random
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
) -> tuple[str, str | None]:
    """Fetch *url* with httpx, extract text with trafilatura.

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

    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
    ) as client:
        response = await client.get(url, headers={"User-Agent": user_agent})
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type and "text/plain" not in content_type:
            raise ValueError(f"Unsupported content type: {content_type}")

        html = response.text[: max_content_length * 2]

    # Extract readable text -------------------------------------------------
    text, title = _extract_with_trafilatura(html, url)

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
    ) -> None:
        self._crawler = crawler
        self._timeout = timeout
        self._max_content_length = max_content_length

        self._definition = ToolDefinition(
            name="web_crawl",
            description=(
                "Fetch full content from a source. Use the INDEX number from "
                "search results (0, 1, 2, etc.). Returns extracted page text "
                "for analysis."
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

        return ToolResult(
            content=formatted,
            success=True,
            sources=sources,
            data={
                "url": url,
                "title": title,
                "content_length": len(text),
                "url_index": url_index,
                "suppressed_by_failure_cache": False,
                "failure_class": "",
            },
        )
