"""Unit tests for tool_adapter.py — BraveSearchAdapter, CrawlerAdapter, create_framework_tools."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from deep_research.agent.adapters.tool_adapter import (
    BraveSearchAdapter,
    CrawlerAdapter,
    create_framework_tools,
)
from deep_research.core.app_config import DomainFilterConfig, DomainFilterMode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_search_results(urls: list[str]) -> list[SimpleNamespace]:
    """Build mock search results with the given URLs."""
    return [
        SimpleNamespace(url=url, title=f"Title for {url}", snippet=f"Snippet for {url}")
        for url in urls
    ]


def _make_brave_client(urls: list[str]) -> MagicMock:
    """Build a mock BraveSearchClient that returns results for given URLs."""
    results = _make_search_results(urls)
    client = MagicMock()
    client.search = AsyncMock(
        return_value=SimpleNamespace(results=results),
    )
    return client


# ---------------------------------------------------------------------------
# BraveSearchAdapter
# ---------------------------------------------------------------------------


class TestBraveSearchAdapter:
    """Tests for BraveSearchAdapter (app BraveSearchClient -> framework SearchClient)."""

    @pytest.mark.asyncio
    async def test_returns_results_from_client(self) -> None:
        """Mock BraveSearchClient returns 5 results -> adapter returns list of 5."""
        urls = [f"https://site{i}.com" for i in range(5)]
        client = _make_brave_client(urls)
        adapter = BraveSearchAdapter(client)

        results = await adapter.search("test query")

        assert len(results) == 5
        assert results[0].url == "https://site0.com"
        assert results[4].url == "https://site4.com"
        client.search.assert_awaited_once_with("test query", count=10, freshness=None)

    @pytest.mark.asyncio
    async def test_no_filter_by_default(self) -> None:
        """Adapter created without domain_filter_config -> no filtering applied."""
        urls = [f"https://site{i}.com" for i in range(3)]
        client = _make_brave_client(urls)
        adapter = BraveSearchAdapter(client)

        assert adapter._per_agent_filter is None

        results = await adapter.search("query")
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_per_agent_filter_include_mode_blocks_unmatched(self) -> None:
        """INCLUDE mode with specific domains -> only matching URLs pass."""
        urls = [
            "https://example.com/page1",
            "https://other.com/page2",
            "https://example.com/page3",
        ]
        client = _make_brave_client(urls)
        filter_config = DomainFilterConfig(
            mode=DomainFilterMode.INCLUDE,
            include_domains=["example.com"],
        )
        adapter = BraveSearchAdapter(client, domain_filter_config=filter_config)

        results = await adapter.search("query")

        assert len(results) == 2
        assert all(r.url.startswith("https://example.com") for r in results)

    @pytest.mark.asyncio
    async def test_per_agent_filter_include_mode_empty_blocks_all(self) -> None:
        """INCLUDE mode with empty include_domains -> is_active=False -> all pass through.

        This verifies the DomainFilter.is_active guard: INCLUDE mode with empty
        include list means filtering is NOT active, so all results pass.
        """
        urls = [f"https://site{i}.com" for i in range(5)]
        client = _make_brave_client(urls)
        filter_config = DomainFilterConfig(
            mode=DomainFilterMode.INCLUDE,
            include_domains=[],
        )
        adapter = BraveSearchAdapter(client, domain_filter_config=filter_config)

        # is_active should be False (empty include list)
        assert adapter._per_agent_filter is not None
        assert not adapter._per_agent_filter.is_active

        results = await adapter.search("query")
        # All results pass through because filter is inactive
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_per_agent_filter_exclude_mode_empty_passes_all(self) -> None:
        """EXCLUDE mode with empty exclude_domains -> is_active=False -> all pass."""
        urls = [f"https://site{i}.com" for i in range(4)]
        client = _make_brave_client(urls)
        filter_config = DomainFilterConfig(
            mode=DomainFilterMode.EXCLUDE,
            exclude_domains=[],
        )
        adapter = BraveSearchAdapter(client, domain_filter_config=filter_config)

        assert adapter._per_agent_filter is not None
        assert not adapter._per_agent_filter.is_active

        results = await adapter.search("query")
        assert len(results) == 4

    @pytest.mark.asyncio
    async def test_full_chain_adapter_to_web_search_tool(self) -> None:
        """End-to-end: BraveSearchAdapter -> WebSearchTool -> ToolResult with sources."""
        from databricks_deep_research.tools.builtins.web_search import WebSearchTool
        from databricks_deep_research.tools.protocol import ToolContext

        urls = ["https://a.com", "https://b.com", "https://c.com"]
        client = _make_brave_client(urls)
        adapter = BraveSearchAdapter(client)

        tool = WebSearchTool(search_client=adapter)
        context = ToolContext(query="test")
        result = await tool.execute({"query": "test"}, context)

        assert result.success
        assert len(result.sources) == 3
        assert result.sources[0].url == "https://a.com"
        assert result.sources[2].url == "https://c.com"


# ---------------------------------------------------------------------------
# CrawlerAdapter
# ---------------------------------------------------------------------------


class TestCrawlerAdapter:
    """Tests for CrawlerAdapter (app WebCrawler -> framework ContentCrawler)."""

    @pytest.mark.asyncio
    async def test_successful_crawl(self) -> None:
        """_fetch_url returns success=True -> adapter returns (content, title)."""
        mock_crawler = MagicMock()
        mock_crawler._fetch_url = AsyncMock(
            return_value=SimpleNamespace(
                success=True,
                content="page text",
                title="Page Title",
                error=None,
            ),
        )

        adapter = CrawlerAdapter(mock_crawler)
        content, title = await adapter("https://example.com")

        assert content == "page text"
        assert title == "Page Title"
        mock_crawler._fetch_url.assert_awaited_once_with("https://example.com")

    @pytest.mark.asyncio
    async def test_failed_crawl_raises(self) -> None:
        """_fetch_url returns success=False -> adapter raises RuntimeError."""
        mock_crawler = MagicMock()
        mock_crawler._fetch_url = AsyncMock(
            return_value=SimpleNamespace(
                success=False,
                content="",
                title=None,
                error="404 Not Found",
            ),
        )

        adapter = CrawlerAdapter(mock_crawler)
        with pytest.raises(RuntimeError, match="404 Not Found"):
            await adapter("https://example.com/missing")

    @pytest.mark.asyncio
    async def test_none_title_passthrough(self) -> None:
        """_fetch_url returns title=None -> adapter returns (content, None)."""
        mock_crawler = MagicMock()
        mock_crawler._fetch_url = AsyncMock(
            return_value=SimpleNamespace(
                success=True,
                content="some content",
                title=None,
                error=None,
            ),
        )

        adapter = CrawlerAdapter(mock_crawler)
        content, title = await adapter("https://example.com")

        assert content == "some content"
        assert title is None

    def test_adapter_is_callable(self) -> None:
        """CrawlerAdapter instances satisfy the callable protocol."""
        adapter = CrawlerAdapter(MagicMock())
        assert callable(adapter)


# ---------------------------------------------------------------------------
# create_framework_tools
# ---------------------------------------------------------------------------


class TestCreateFrameworkTools:
    """Tests for the create_framework_tools factory."""

    @pytest.mark.asyncio
    async def test_web_tools_created_with_clients(self) -> None:
        """Both brave_client and crawler provided -> web_search + web_crawl."""
        mock_brave = MagicMock()
        mock_crawler = MagicMock()

        tools = await create_framework_tools(
            brave_client=mock_brave,
            crawler=mock_crawler,
        )

        names = [t.definition.name for t in tools]
        assert "web_search" in names
        assert "web_crawl" in names

    @pytest.mark.asyncio
    async def test_no_clients_no_web_tools(self) -> None:
        """Neither brave_client nor crawler -> empty list."""
        tools = await create_framework_tools()
        assert tools == []

    @pytest.mark.asyncio
    async def test_crawler_gets_adapter_not_raw(self) -> None:
        """WebCrawlTool receives a CrawlerAdapter, not the raw WebCrawler."""
        mock_crawler = MagicMock()

        tools = await create_framework_tools(crawler=mock_crawler)

        # Find the web_crawl tool
        crawl_tools = [t for t in tools if t.definition.name == "web_crawl"]
        assert len(crawl_tools) == 1

        crawl_tool = crawl_tools[0]
        # The internal _crawler should be a CrawlerAdapter, not the raw mock
        assert isinstance(crawl_tool._crawler, CrawlerAdapter)
        # And the adapter should wrap our original mock
        assert crawl_tool._crawler._crawler is mock_crawler
