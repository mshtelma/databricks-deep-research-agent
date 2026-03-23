"""Tests for tool_adapter — BraveSearchAdapter and create_framework_tools."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from deep_research.agent.adapters.tool_adapter import (
    BraveSearchAdapter,
    EnterpriseToolAdapter,
    create_framework_tools,
)
from deep_research.core.app_config import DomainFilterConfig, DomainFilterMode

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass
class FakeSearchResult:
    url: str
    title: str
    snippet: str
    relevance_score: float | None = None


@dataclass
class FakeSearchResponse:
    results: list[FakeSearchResult]
    query: str
    total_results: int


def _make_results(*urls: str) -> list[FakeSearchResult]:
    return [
        FakeSearchResult(url=u, title=f"Title {i}", snippet=f"Snippet {i}")
        for i, u in enumerate(urls)
    ]


def _make_response(*urls: str) -> FakeSearchResponse:
    results = _make_results(*urls)
    return FakeSearchResponse(results=results, query="test", total_results=len(results))


def _make_brave_client(*urls: str) -> AsyncMock:
    client = AsyncMock()
    client.search = AsyncMock(return_value=_make_response(*urls))
    return client


class _FakeEnterpriseTool:
    def __init__(self, tool_name: str) -> None:
        self.definition = SimpleNamespace(
            name=tool_name,
            description=f"Tool {tool_name}",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        )
        self.execute = AsyncMock(return_value=SimpleNamespace(content="ok", success=True))


# ---------------------------------------------------------------------------
# BraveSearchAdapter — result extraction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adapter_extracts_results_from_response() -> None:
    """Adapter unwraps SearchResponse.results into a plain list."""
    client = _make_brave_client(
        "https://example.com/1",
        "https://example.com/2",
    )
    adapter = BraveSearchAdapter(client=client)

    results = await adapter.search("test query", count=5)

    assert isinstance(results, list)
    assert len(results) == 2
    assert results[0].url == "https://example.com/1"
    assert results[1].url == "https://example.com/2"


# ---------------------------------------------------------------------------
# BraveSearchAdapter — domain filtering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adapter_applies_include_domain_filter() -> None:
    """INCLUDE mode: only results matching include_domains pass through."""
    client = _make_brave_client(
        "https://www.cdc.gov/data",
        "https://facebook.com/page",
        "https://www.nasa.gov/mission",
    )
    config = DomainFilterConfig(
        mode=DomainFilterMode.INCLUDE,
        include_domains=["*.gov"],
    )
    adapter = BraveSearchAdapter(client=client, domain_filter_config=config)

    results = await adapter.search("test")

    assert len(results) == 2
    assert all(".gov" in r.url for r in results)


@pytest.mark.asyncio
async def test_adapter_applies_exclude_domain_filter() -> None:
    """EXCLUDE mode: results matching exclude_domains are removed."""
    client = _make_brave_client(
        "https://www.cdc.gov/data",
        "https://facebook.com/page",
        "https://example.com/info",
    )
    config = DomainFilterConfig(
        mode=DomainFilterMode.EXCLUDE,
        exclude_domains=["facebook.com"],
    )
    adapter = BraveSearchAdapter(client=client, domain_filter_config=config)

    results = await adapter.search("test")

    assert len(results) == 2
    urls = [r.url for r in results]
    assert "https://facebook.com/page" not in urls


@pytest.mark.asyncio
async def test_adapter_no_filter_when_config_is_none() -> None:
    """No domain_filter_config means all results pass through."""
    client = _make_brave_client(
        "https://facebook.com/page",
        "https://evil.com/bad",
    )
    adapter = BraveSearchAdapter(client=client, domain_filter_config=None)

    results = await adapter.search("test")

    assert len(results) == 2


@pytest.mark.asyncio
async def test_adapter_stacks_with_global_filter() -> None:
    """Adapter applies per-agent filter on top of what BraveSearchClient returns.

    The global filter runs inside BraveSearchClient.search() (mocked here).
    The adapter adds the per-agent filter as a second layer.
    """
    # Simulate BraveSearchClient already filtered out some results
    # (only .gov and .edu results remain after global filter)
    client = _make_brave_client(
        "https://www.cdc.gov/data",
        "https://www.mit.edu/research",
    )
    # Per-agent filter: also exclude .edu
    config = DomainFilterConfig(
        mode=DomainFilterMode.EXCLUDE,
        exclude_domains=["*.edu"],
    )
    adapter = BraveSearchAdapter(client=client, domain_filter_config=config)

    results = await adapter.search("test")

    assert len(results) == 1
    assert results[0].url == "https://www.cdc.gov/data"


# ---------------------------------------------------------------------------
# EnterpriseToolAdapter — definition passthrough
# ---------------------------------------------------------------------------


def test_enterprise_tool_adapter_preserves_underlying_definition_name() -> None:
    """Wrapped enterprise tools must keep their unique app-level names."""
    adapter = EnterpriseToolAdapter(_FakeEnterpriseTool("search_finance_docs"))

    assert adapter.definition.name == "search_finance_docs"
    assert adapter.definition.description == "Tool search_finance_docs"
    assert adapter.definition.source_type == "enterprise"
    assert adapter.definition.metadata["source_name"] == "search_finance_docs"
    assert adapter.definition.metadata["source_description"] == "Tool search_finance_docs"


@pytest.mark.asyncio
async def test_enterprise_tool_adapter_preserves_structured_sources() -> None:
    """Wrapped tool results must keep sources for framework pool writes."""
    tool = _FakeEnterpriseTool("search_finance_docs")
    tool.execute = AsyncMock(return_value=SimpleNamespace(
        content="Found one result",
        success=True,
        sources=[{
            "type": "vector_search",
            "url": "vs://main.finance_docs/doc-1",
            "title": "Quarterly Earnings",
            "content": "Revenue grew 10 percent year over year.",
            "relevance_score": 0.91,
        }],
        data={"query": "earnings"},
    ))
    adapter = EnterpriseToolAdapter(tool)

    result = await adapter.execute({"query": "earnings"})

    assert result.success is True
    assert result.data == {"query": "earnings"}
    assert len(result.sources) == 1
    source = result.sources[0]
    assert source.url == "vs://main.finance_docs/doc-1"
    assert source.title == "Quarterly Earnings"
    assert source.source_type == "vector_search"
    assert source.content == "Revenue grew 10 percent year over year."
    assert source.snippet == "Revenue grew 10 percent year over year."


@pytest.mark.asyncio
async def test_create_tools_keeps_unique_names_for_same_tool_class() -> None:
    """Multiple enterprise tools of the same Python class must still be unique."""
    tools = await create_framework_tools(
        enterprise_tools=[
            _FakeEnterpriseTool("search_earnings_index"),
            _FakeEnterpriseTool("search_transcripts_index"),
        ],
    )

    assert [tool.definition.name for tool in tools] == [
        "search_earnings_index",
        "search_transcripts_index",
    ]


# ---------------------------------------------------------------------------
# create_framework_tools — integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_tools_with_domain_filter_config() -> None:
    """Passing a DomainFilterConfig creates a WebSearchTool with BraveSearchAdapter."""
    client = _make_brave_client("https://example.com")
    config = DomainFilterConfig(
        mode=DomainFilterMode.EXCLUDE,
        exclude_domains=["evil.com"],
    )

    tools = await create_framework_tools(
        brave_client=client,
        domain_filter_config=config,
    )

    assert len(tools) == 1
    tool = tools[0]
    assert tool.definition.name == "web_search"
    # The tool's internal client should be a BraveSearchAdapter, not the raw client
    assert isinstance(tool._client, BraveSearchAdapter)
    assert tool._client._per_agent_filter is not None


@pytest.mark.asyncio
async def test_create_tools_without_domain_filter() -> None:
    """No domain_filter_config still creates a working WebSearchTool."""
    client = _make_brave_client("https://example.com")

    tools = await create_framework_tools(
        brave_client=client,
        domain_filter_config=None,
    )

    assert len(tools) == 1
    tool = tools[0]
    assert isinstance(tool._client, BraveSearchAdapter)
    assert tool._client._per_agent_filter is None


# ---------------------------------------------------------------------------
# framework_orchestrator call site — no crash
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_domain_filter_config_passed_without_crash() -> None:
    """Verify the orchestrator passes DomainFilterConfig directly (no .to_filter_string())."""
    config = DomainFilterConfig(
        mode=DomainFilterMode.EXCLUDE,
        exclude_domains=["facebook.com", "twitter.com"],
    )

    # The old code would crash here: config.to_filter_string()
    # The fix passes config directly to create_framework_tools
    client = _make_brave_client("https://example.com")

    # This should not raise AttributeError
    tools = await create_framework_tools(
        brave_client=client,
        domain_filter_config=config,
    )

    assert len(tools) == 1
