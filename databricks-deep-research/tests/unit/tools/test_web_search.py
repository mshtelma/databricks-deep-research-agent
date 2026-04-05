"""Tests for web_search table detection and registration."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from databricks_deep_research.tools.builtins.web_search import SearchResult, WebSearchTool
from databricks_deep_research.tools.protocol import TableRegistry, ToolContext, UrlRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TABLE_CONTENT = (
    "Economic data summary.\n\n"
    "| Metric | Value |\n"
    "|---|---|\n"
    "| GDP | 25T |\n"
    "| Inflation | 3.2% |\n\n"
    "Source: Bureau of Economic Analysis."
)


@dataclass
class _FakeSearchClient:
    """Minimal SearchClient that returns canned results."""

    results: list[SearchResult]

    async def search(
        self,
        query: str,  # noqa: ARG002
        *,
        count: int = 10,  # noqa: ARG002
        freshness: str | None = None,  # noqa: ARG002
    ) -> list[SearchResult]:
        return self.results


def _make_context(*, table_registry: TableRegistry | None = None) -> ToolContext:
    return ToolContext(url_registry=UrlRegistry(), table_registry=table_registry)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWebSearchTableDetection:
    @pytest.mark.asyncio()
    async def test_tables_detected_and_registered(self) -> None:
        """Jina-style results with full content → tables detected and registered."""
        client = _FakeSearchClient(
            results=[
                SearchResult(
                    url="https://example.com/data",
                    title="Economic Data",
                    snippet="GDP and inflation data",
                    content=_TABLE_CONTENT,
                ),
            ]
        )
        reg = TableRegistry()
        tool = WebSearchTool(client, extract_tables=True)
        ctx = _make_context(table_registry=reg)

        result = await tool.execute(
            tool.validate_arguments({"query": "GDP data"}), ctx,
        )

        assert result.success
        assert len(reg) >= 1
        assert result.data.get("table_count", 0) >= 1
        # Output should contain table_idx annotation
        assert "table_idx=" in result.content

        entry = reg.resolve(0)
        assert entry is not None
        assert entry.source_kind == "web"

    @pytest.mark.asyncio()
    async def test_extract_tables_disabled(self) -> None:
        """extract_tables=False suppresses table detection."""
        client = _FakeSearchClient(
            results=[
                SearchResult(
                    url="https://example.com/data",
                    title="Economic Data",
                    snippet="GDP",
                    content=_TABLE_CONTENT,
                ),
            ]
        )
        reg = TableRegistry()
        tool = WebSearchTool(client, extract_tables=False)
        ctx = _make_context(table_registry=reg)

        result = await tool.execute(
            tool.validate_arguments({"query": "GDP data"}), ctx,
        )

        assert result.success
        assert len(reg) == 0
        assert "table_idx" not in result.content

    @pytest.mark.asyncio()
    async def test_no_tables_when_content_is_none(self) -> None:
        """Brave-style results with no content → no table detection."""
        client = _FakeSearchClient(
            results=[
                SearchResult(
                    url="https://example.com",
                    title="Page",
                    snippet="A snippet",
                    content=None,
                ),
            ]
        )
        reg = TableRegistry()
        tool = WebSearchTool(client, extract_tables=True)
        ctx = _make_context(table_registry=reg)

        result = await tool.execute(
            tool.validate_arguments({"query": "test"}), ctx,
        )

        assert result.success
        assert len(reg) == 0
        assert "table_count" not in result.data

    @pytest.mark.asyncio()
    async def test_no_crash_without_registry(self) -> None:
        """Table detection works even when table_registry is None."""
        client = _FakeSearchClient(
            results=[
                SearchResult(
                    url="https://example.com",
                    title="Page",
                    snippet="A snippet",
                    content=_TABLE_CONTENT,
                ),
            ]
        )
        tool = WebSearchTool(client, extract_tables=True)
        ctx = _make_context(table_registry=None)

        result = await tool.execute(
            tool.validate_arguments({"query": "data"}), ctx,
        )

        assert result.success
        # Tables detected in data but not registered (no registry)
        assert result.data.get("table_count", 0) >= 1

    @pytest.mark.asyncio()
    async def test_capacity_overflow_handled(self) -> None:
        """Registry at capacity → tables skipped gracefully."""
        client = _FakeSearchClient(
            results=[
                SearchResult(
                    url="https://example.com",
                    title="Page",
                    snippet="A snippet",
                    content=_TABLE_CONTENT,
                ),
            ]
        )
        reg = TableRegistry(max_tables=0)
        tool = WebSearchTool(client, extract_tables=True)
        ctx = _make_context(table_registry=reg)

        result = await tool.execute(
            tool.validate_arguments({"query": "data"}), ctx,
        )

        assert result.success
        assert len(reg) == 0  # nothing registered — capacity full
