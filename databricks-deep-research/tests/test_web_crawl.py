"""Tests for the web crawl tool."""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest

from databricks_deep_research.tools.builtins.web_crawl import WebCrawlTool
from databricks_deep_research.tools.protocol import ToolContext, UrlRegistry


def _make_context(urls: list[str] | None = None) -> ToolContext:
    registry = UrlRegistry()
    for url in urls or []:
        registry.register(url)
    return ToolContext(url_registry=registry)


def _mock_crawler(text: str, title: str | None = None) -> AsyncMock:
    return AsyncMock(return_value=(text, title))


def _mock_http_error(status_code: int) -> AsyncMock:
    request = httpx.Request("GET", "https://example.com")
    response = httpx.Response(status_code=status_code, request=request)
    return AsyncMock(side_effect=httpx.HTTPStatusError("boom", request=request, response=response))


# ---------------------------------------------------------------------------
# Quality gate: short content rejected
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_short_content_rejected() -> None:
    """Pages with very short content are rejected as low quality."""
    crawler = _mock_crawler("Access Denied", "Error")
    tool = WebCrawlTool(crawler=crawler)
    ctx = _make_context(["https://example.com"])

    result = await tool.execute({"url_index": 0}, ctx)

    assert not result.success
    assert "too short" in result.content
    assert result.sources == []


@pytest.mark.asyncio
async def test_empty_content_rejected() -> None:
    """Pages with empty content are rejected."""
    crawler = _mock_crawler("", None)
    tool = WebCrawlTool(crawler=crawler)
    ctx = _make_context(["https://example.com"])

    result = await tool.execute({"url_index": 0}, ctx)

    assert not result.success
    assert "no readable text" in result.content


@pytest.mark.asyncio
async def test_sufficient_content_accepted() -> None:
    """Pages with enough content pass the quality gate."""
    long_text = "This is a real article with meaningful content. " * 10
    crawler = _mock_crawler(long_text, "Good Article")
    tool = WebCrawlTool(crawler=crawler)
    ctx = _make_context(["https://example.com"])

    result = await tool.execute({"url_index": 0}, ctx)

    assert result.success
    assert len(result.sources) == 1
    assert result.sources[0].url == "https://example.com"


@pytest.mark.asyncio
async def test_whitespace_only_short_content_rejected() -> None:
    """Pages with mostly whitespace and short text are rejected."""
    crawler = _mock_crawler("   Forbidden   \n\n  ", "403")
    tool = WebCrawlTool(crawler=crawler)
    ctx = _make_context(["https://example.com"])

    result = await tool.execute({"url_index": 0}, ctx)

    assert not result.success
    assert "too short" in result.content


@pytest.mark.asyncio
async def test_repeated_non_retryable_url_failure_is_suppressed() -> None:
    crawler = _mock_http_error(403)
    tool = WebCrawlTool(crawler=crawler)
    ctx = _make_context(["https://example.com"])

    first = await tool.execute({"url_index": 0}, ctx)
    second = await tool.execute({"url_index": 0}, ctx)

    assert not first.success
    assert not second.success
    assert second.error == "Suppressed repeated crawl failure"
    assert crawler.await_count == 1


@pytest.mark.asyncio
async def test_domain_suppression_blocks_third_non_retryable_target() -> None:
    crawler = _mock_http_error(403)
    tool = WebCrawlTool(crawler=crawler)
    ctx = _make_context([
        "https://example.com/a",
        "https://example.com/b",
        "https://example.com/c",
    ])

    first = await tool.execute({"url_index": 0}, ctx)
    second = await tool.execute({"url_index": 1}, ctx)
    third = await tool.execute({"url_index": 2}, ctx)

    assert not first.success
    assert not second.success
    assert not third.success
    assert third.error == "Suppressed repeated crawl failure"
    assert third.data["suppression_scope"] == "domain"
    assert crawler.await_count == 2
