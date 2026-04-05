"""Unit tests for content-aware WebSearchTool formatting."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from databricks_deep_research.tools.builtins.web_search import SearchResult, WebSearchTool
from databricks_deep_research.tools.protocol import ToolContext, UrlRegistry


def _make_client(results: list[SearchResult]) -> MagicMock:
    client = MagicMock()
    client.search = AsyncMock(return_value=results)
    return client


class TestContentAwareFormatting:
    @pytest.mark.asyncio
    async def test_content_rich_result_shows_full_text(self) -> None:
        results = [
            SearchResult(
                url="https://example.com",
                title="Article",
                snippet="Short snippet",
                content="Full page content here.",
            ),
        ]
        tool = WebSearchTool(_make_client(results))
        ctx = ToolContext(query="test", url_registry=UrlRegistry())

        args = tool.validate_arguments({"query": "test"})
        result = await tool.execute(args, ctx)

        assert result.success
        assert "Full page content here." in result.content
        # Snippet NOT shown when content is available
        assert "Short snippet" not in result.content

    @pytest.mark.asyncio
    async def test_snippet_only_result_shows_snippet(self) -> None:
        results = [
            SearchResult(
                url="https://example.com",
                title="Article",
                snippet="Just a snippet",
            ),
        ]
        tool = WebSearchTool(_make_client(results))
        ctx = ToolContext(query="test", url_registry=UrlRegistry())

        args = tool.validate_arguments({"query": "test"})
        result = await tool.execute(args, ctx)

        assert result.success
        assert "Just a snippet" in result.content

    @pytest.mark.asyncio
    async def test_empty_string_content_treated_as_absent(self) -> None:
        """Empty string content should fall back to snippet."""
        results = [
            SearchResult(
                url="https://example.com",
                title="Article",
                snippet="Fallback snippet",
                content="",
            ),
        ]
        tool = WebSearchTool(_make_client(results))
        ctx = ToolContext(query="test", url_registry=UrlRegistry())

        args = tool.validate_arguments({"query": "test"})
        result = await tool.execute(args, ctx)

        assert "Fallback snippet" in result.content

    @pytest.mark.asyncio
    async def test_max_content_per_result_truncation(self) -> None:
        long_content = "A" * 10_000
        results = [
            SearchResult(
                url="https://example.com",
                title="Long",
                snippet="s",
                content=long_content,
            ),
        ]
        tool = WebSearchTool(_make_client(results), max_content_per_result=100)
        ctx = ToolContext(query="test", url_registry=UrlRegistry())

        args = tool.validate_arguments({"query": "test"})
        result = await tool.execute(args, ctx)

        # Content should be truncated to 100 chars
        assert len(result.content) < 200  # title + 100 chars max

    @pytest.mark.asyncio
    async def test_source_info_content_populated(self) -> None:
        results = [
            SearchResult(
                url="https://example.com",
                title="A",
                snippet="s",
                content="Full content for citation pipeline.",
            ),
        ]
        tool = WebSearchTool(_make_client(results))
        ctx = ToolContext(query="test", url_registry=UrlRegistry())

        args = tool.validate_arguments({"query": "test"})
        result = await tool.execute(args, ctx)

        assert len(result.sources) == 1
        assert result.sources[0].content == "Full content for citation pipeline."

    @pytest.mark.asyncio
    async def test_source_info_content_none_without_content(self) -> None:
        results = [
            SearchResult(url="https://example.com", title="A", snippet="s"),
        ]
        tool = WebSearchTool(_make_client(results))
        ctx = ToolContext(query="test", url_registry=UrlRegistry())

        args = tool.validate_arguments({"query": "test"})
        result = await tool.execute(args, ctx)

        assert len(result.sources) == 1
        assert result.sources[0].content is None

    @pytest.mark.asyncio
    async def test_mixed_results(self) -> None:
        """Mix of content-rich and snippet-only results."""
        results = [
            SearchResult(
                url="https://a.com", title="With Content", snippet="s1",
                content="Full text A",
            ),
            SearchResult(
                url="https://b.com", title="Snippet Only", snippet="Short B",
            ),
        ]
        tool = WebSearchTool(_make_client(results))
        ctx = ToolContext(query="test", url_registry=UrlRegistry())

        args = tool.validate_arguments({"query": "test"})
        result = await tool.execute(args, ctx)

        assert "Full text A" in result.content
        assert "Short B" in result.content
        assert result.sources[0].content == "Full text A"
        assert result.sources[1].content is None

    @pytest.mark.asyncio
    async def test_source_info_content_truncated_same_as_display(self) -> None:
        long_content = "B" * 10_000
        results = [
            SearchResult(
                url="https://example.com", title="A", snippet="s",
                content=long_content,
            ),
        ]
        tool = WebSearchTool(_make_client(results), max_content_per_result=500)
        ctx = ToolContext(query="test", url_registry=UrlRegistry())

        args = tool.validate_arguments({"query": "test"})
        result = await tool.execute(args, ctx)

        assert result.sources[0].content is not None
        assert len(result.sources[0].content) == 500
