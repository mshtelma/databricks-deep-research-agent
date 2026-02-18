"""Unit tests for parallel tool execution (007-enterprise-data-sources).

Tests:
- State safety with asyncio.Lock (not threading.RLock)
- Cross-source parallelism (Web + VS + Genie)
- Same-source batching
- Dependency ordering (web_crawl waits for web_search)
- Race condition prevention (no duplicate sources)
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_research.agent.state import ResearchState, SourceInfo
from deep_research.agent.nodes.react_researcher import (
    ReactResearchState,
    ToolBatch,
    _get_execution_batches,
    _group_tools_by_source,
    TOOL_DEPENDENCIES,
    TOOL_SOURCE_MAPPING,
)
from deep_research.services.llm.types import ToolCall


class TestAsyncLockUsage:
    """Tests verifying asyncio.Lock is used (NOT threading.RLock)."""

    def test_research_state_uses_asyncio_lock(self) -> None:
        """Verify ResearchState uses asyncio.Lock for all locks."""
        state = ResearchState(query="test")

        # Check that all locks are asyncio.Lock instances
        assert isinstance(state._sources_lock, asyncio.Lock)
        assert isinstance(state._claims_lock, asyncio.Lock)
        assert isinstance(state._evidence_lock, asyncio.Lock)
        assert isinstance(state._cache_lock, asyncio.Lock)
        assert isinstance(state._phase_lock, asyncio.Lock)
        assert isinstance(state._step_lock, asyncio.Lock)

    def test_react_state_uses_asyncio_lock(self) -> None:
        """Verify ReactResearchState uses asyncio.Lock for all locks."""
        react_state = ReactResearchState()

        # Check that all locks are asyncio.Lock instances
        assert isinstance(react_state._tool_count_lock, asyncio.Lock)
        assert isinstance(react_state._content_lock, asyncio.Lock)
        assert isinstance(react_state._sources_lock, asyncio.Lock)


class TestToolGrouping:
    """Tests for tool grouping by source type."""

    def test_group_single_web_search(self) -> None:
        """Single web_search goes into 'web' group."""
        tool_calls = [
            ToolCall(id="1", name="web_search", arguments={"query": "test"}),
        ]

        groups = _group_tools_by_source(tool_calls)

        assert "web" in groups
        assert len(groups["web"]) == 1
        assert groups["web"][0].name == "web_search"

    def test_group_multiple_sources(self) -> None:
        """Tools from different sources are grouped separately."""
        tool_calls = [
            ToolCall(id="1", name="web_search", arguments={"query": "test"}),
            ToolCall(id="2", name="vector_search", arguments={"query": "test"}),
            ToolCall(id="3", name="genie_query", arguments={"query": "test"}),
        ]

        groups = _group_tools_by_source(tool_calls)

        assert len(groups) == 3
        assert "web" in groups
        assert "vector" in groups
        assert "genie" in groups

    def test_group_same_source_batched(self) -> None:
        """Multiple tools from same source are grouped together."""
        tool_calls = [
            ToolCall(id="1", name="web_search", arguments={"query": "test1"}),
            ToolCall(id="2", name="web_search", arguments={"query": "test2"}),
            ToolCall(id="3", name="web_crawl", arguments={"index": 0}),
        ]

        groups = _group_tools_by_source(tool_calls)

        assert len(groups) == 1  # All web-related
        assert "web" in groups
        assert len(groups["web"]) == 3


class TestDependencyOrdering:
    """Tests for tool dependency ordering."""

    def test_web_crawl_depends_on_web_search(self) -> None:
        """Verify web_crawl is listed as depending on web_search."""
        assert "web_crawl" in TOOL_DEPENDENCIES
        assert "web_search" in TOOL_DEPENDENCIES["web_crawl"]

    def test_execution_batches_no_dependencies(self) -> None:
        """When no dependencies, all tools in one batch."""
        tool_calls = [
            ToolCall(id="1", name="web_search", arguments={"query": "test1"}),
            ToolCall(id="2", name="web_search", arguments={"query": "test2"}),
        ]

        batches = _get_execution_batches(tool_calls)

        assert len(batches) == 1
        assert len(batches[0].tool_calls) == 2

    def test_execution_batches_with_dependency(self) -> None:
        """When web_search and web_crawl present, search runs first."""
        tool_calls = [
            ToolCall(id="1", name="web_search", arguments={"query": "test"}),
            ToolCall(id="2", name="web_crawl", arguments={"index": 0}),
        ]

        batches = _get_execution_batches(tool_calls)

        # Should have 2 batches: search first, then crawl
        assert len(batches) == 2
        assert batches[0].tool_types == ["web_search"]
        assert batches[1].tool_types == ["web_crawl"]

    def test_execution_batches_crawl_only(self) -> None:
        """When only web_crawl (no search), it runs in single batch."""
        tool_calls = [
            ToolCall(id="1", name="web_crawl", arguments={"index": 0}),
            ToolCall(id="2", name="web_crawl", arguments={"index": 1}),
        ]

        batches = _get_execution_batches(tool_calls)

        # No search to wait for, so single batch
        assert len(batches) == 1
        assert "web_crawl" in batches[0].tool_types

    def test_execution_batches_mixed_sources_with_dependency(self) -> None:
        """Mixed sources with dependency splits correctly."""
        tool_calls = [
            ToolCall(id="1", name="web_search", arguments={"query": "test"}),
            ToolCall(id="2", name="web_crawl", arguments={"index": 0}),
            ToolCall(id="3", name="vector_search", arguments={"query": "test"}),
        ]

        batches = _get_execution_batches(tool_calls)

        # Batch 1: web_search
        # Batch 2: web_crawl + vector_search (can run together after search)
        assert len(batches) == 2
        assert batches[0].tool_types == ["web_search"]
        assert len(batches[1].tool_calls) == 2


class TestStateRaceConditionPrevention:
    """Tests verifying race conditions are prevented with async locks."""

    @pytest.mark.asyncio
    async def test_add_source_async_deduplicates(self) -> None:
        """Concurrent add_source_async calls should not create duplicates."""
        state = ResearchState(query="test")
        source = SourceInfo(url="https://example.com", title="Example")

        # Simulate concurrent calls
        async def add_source_many_times() -> None:
            await state.add_source_async(source)

        # Run 10 concurrent additions of the same source
        await asyncio.gather(*[add_source_many_times() for _ in range(10)])

        # Should only have 1 source (deduplicated)
        assert len(state.sources) == 1
        assert state.sources[0].url == "https://example.com"

    @pytest.mark.asyncio
    async def test_add_source_async_preserves_different_sources(self) -> None:
        """Different sources should all be added."""
        state = ResearchState(query="test")

        async def add_unique_source(idx: int) -> None:
            source = SourceInfo(url=f"https://example{idx}.com", title=f"Example {idx}")
            await state.add_source_async(source)

        # Add 5 different sources concurrently
        await asyncio.gather(*[add_unique_source(i) for i in range(5)])

        # Should have all 5 sources
        assert len(state.sources) == 5

    @pytest.mark.asyncio
    async def test_increment_tool_count_atomic(self) -> None:
        """Concurrent increment_tool_count should be atomic."""
        react_state = ReactResearchState()

        async def increment_many_times(count: int) -> list[int]:
            results = []
            for _ in range(count):
                val = await react_state.increment_tool_count()
                results.append(val)
            return results

        # Run 3 coroutines each incrementing 10 times
        results = await asyncio.gather(
            increment_many_times(10),
            increment_many_times(10),
            increment_many_times(10),
        )

        # Flatten and check
        all_values = [v for r in results for v in r]

        # Final count should be 30
        assert react_state.tool_call_count == 30

        # All values from 1-30 should be present (no duplicates, no gaps)
        assert sorted(all_values) == list(range(1, 31))

    @pytest.mark.asyncio
    async def test_add_high_quality_source_deduplicates(self) -> None:
        """Concurrent add_high_quality_source should not create duplicates."""
        react_state = ReactResearchState()

        async def add_source() -> None:
            await react_state.add_high_quality_source("https://example.com")

        # Run 10 concurrent additions
        await asyncio.gather(*[add_source() for _ in range(10)])

        # Should only have 1 source
        assert len(react_state.high_quality_sources) == 1

    @pytest.mark.asyncio
    async def test_add_crawled_content_concurrent(self) -> None:
        """Concurrent add_crawled_content should not lose updates."""
        react_state = ReactResearchState()

        async def add_content(idx: int) -> None:
            await react_state.add_crawled_content(
                f"https://example{idx}.com",
                f"Content {idx}",
            )

        # Add 10 different contents concurrently
        await asyncio.gather(*[add_content(i) for i in range(10)])

        # Should have all 10 contents
        assert len(react_state.crawled_content) == 10


class TestToolBatch:
    """Tests for ToolBatch dataclass."""

    def test_tool_batch_repr(self) -> None:
        """ToolBatch should have informative repr."""
        batch = ToolBatch(
            tool_types=["web_search", "web_crawl"],
            tool_calls=[
                ToolCall(id="1", name="web_search", arguments={}),
                ToolCall(id="2", name="web_crawl", arguments={}),
            ],
        )

        repr_str = repr(batch)

        assert "web_search" in repr_str
        assert "web_crawl" in repr_str
        assert "count=2" in repr_str


class TestSourceMapping:
    """Tests for tool source mapping constants."""

    def test_web_tools_map_to_web(self) -> None:
        """web_search and web_crawl should map to 'web' source."""
        assert TOOL_SOURCE_MAPPING["web_search"] == "web"
        assert TOOL_SOURCE_MAPPING["web_crawl"] == "web"

    def test_vector_search_maps_to_vector(self) -> None:
        """vector_search should map to 'vector' source."""
        assert TOOL_SOURCE_MAPPING["vector_search"] == "vector"

    def test_genie_query_maps_to_genie(self) -> None:
        """genie_query should map to 'genie' source."""
        assert TOOL_SOURCE_MAPPING["genie_query"] == "genie"

    def test_file_search_maps_to_uploaded_file(self) -> None:
        """file_search should map to uploaded_file source."""
        assert TOOL_SOURCE_MAPPING["file_search"] == "uploaded_file"
