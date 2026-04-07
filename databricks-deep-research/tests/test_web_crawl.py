"""Tests for the web crawl tool."""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest

from databricks_deep_research.tools.builtins.web_crawl import WebCrawlTool
from databricks_deep_research.tools.protocol import TableRegistry, ToolContext, UrlRegistry


def _make_context(
    urls: list[str] | None = None,
    *,
    table_registry: TableRegistry | None = None,
) -> ToolContext:
    registry = UrlRegistry()
    for url in urls or []:
        registry.register(url)
    return ToolContext(url_registry=registry, table_registry=table_registry)


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


# ---------------------------------------------------------------------------
# Table extraction in crawled content
# ---------------------------------------------------------------------------

_TABLE_TEXT = (
    "Some intro text about GDP data.\n\n"
    "| Country | GDP |\n"
    "|---|---|\n"
    "| US | 25T |\n"
    "| China | 18T |\n\n"
    "Some trailing text."
)


@pytest.mark.asyncio
async def test_table_detected_and_registered() -> None:
    """Markdown tables in crawled content are detected and registered."""
    crawler = _mock_crawler(_TABLE_TEXT, "GDP Page")
    reg = TableRegistry()
    tool = WebCrawlTool(crawler=crawler)
    ctx = _make_context(["https://example.com"], table_registry=reg)

    result = await tool.execute({"url_index": 0}, ctx)

    assert result.success
    assert "table_idx=" in result.content
    assert result.data.get("table_count", 0) >= 1
    assert len(reg) >= 1

    entry = reg.resolve(0)
    assert entry is not None
    assert entry.source_kind == "web"
    assert "example.com" in entry.source_label


@pytest.mark.asyncio
async def test_extract_tables_disabled() -> None:
    """extract_tables=False suppresses table detection."""
    crawler = _mock_crawler(_TABLE_TEXT, "GDP Page")
    reg = TableRegistry()
    tool = WebCrawlTool(crawler=crawler, extract_tables=False)
    ctx = _make_context(["https://example.com"], table_registry=reg)

    result = await tool.execute({"url_index": 0}, ctx)

    assert result.success
    assert len(reg) == 0
    assert "table_idx" not in result.content


@pytest.mark.asyncio
async def test_table_no_registry_no_crash() -> None:
    """Table detection works even when table_registry is None (no registration)."""
    crawler = _mock_crawler(_TABLE_TEXT, "GDP Page")
    tool = WebCrawlTool(crawler=crawler)
    ctx = _make_context(["https://example.com"])  # no table_registry

    result = await tool.execute({"url_index": 0}, ctx)

    assert result.success
    # Tables detected in data but not registered
    assert result.data.get("table_count", 0) >= 1


@pytest.mark.asyncio
async def test_table_registry_capacity_overflow() -> None:
    """When registry is full, tables are skipped gracefully."""
    crawler = _mock_crawler(_TABLE_TEXT, "GDP Page")
    reg = TableRegistry(max_tables=0)  # already full
    tool = WebCrawlTool(crawler=crawler)
    ctx = _make_context(["https://example.com"], table_registry=reg)

    result = await tool.execute({"url_index": 0}, ctx)

    assert result.success
    assert len(reg) == 0  # nothing registered
