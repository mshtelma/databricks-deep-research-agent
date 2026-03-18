"""Tests for PoolState: dedup, capacity, eviction, keyword search."""

from __future__ import annotations

import asyncio

import pytest

from databricks_deep_research.pools.pool_state import PoolConfig, PoolState


class TestPoolConfig:
    def test_defaults(self) -> None:
        cfg = PoolConfig(name="test")
        assert cfg.item_type == "text"
        assert cfg.dedup_key is None
        assert cfg.dedup_content_hash is True
        assert cfg.max_items == 0

    def test_extra_fields_rejected(self) -> None:
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="extra"):
            PoolConfig(name="test", bad_field="x")  # type: ignore[call-arg]


class TestPoolStateAdd:
    def test_add_simple(self) -> None:
        pool = PoolState(PoolConfig(name="obs"))
        assert pool.add("item1") is True
        assert pool.count() == 1

    def test_content_hash_dedup(self) -> None:
        pool = PoolState(PoolConfig(name="obs", dedup_content_hash=True))
        assert pool.add("same") is True
        assert pool.add("same") is False
        assert pool.count() == 1

    def test_key_dedup(self) -> None:
        pool = PoolState(PoolConfig(name="src", dedup_key="url"))
        assert pool.add({"url": "https://a.com", "title": "A"}) is True
        assert pool.add({"url": "https://a.com", "title": "A updated"}) is False
        assert pool.add({"url": "https://b.com", "title": "B"}) is True
        assert pool.count() == 2

    def test_no_dedup(self) -> None:
        pool = PoolState(PoolConfig(name="raw", dedup_content_hash=False))
        assert pool.add("x") is True
        assert pool.add("x") is True
        assert pool.count() == 2


class TestPoolStateCapacity:
    def test_eviction(self) -> None:
        pool = PoolState(PoolConfig(name="small", max_items=3, dedup_content_hash=False))
        for i in range(5):
            pool.add(f"item{i}")
        assert pool.count() == 3
        # Oldest items evicted
        assert pool.items == ["item2", "item3", "item4"]

    def test_unlimited(self) -> None:
        pool = PoolState(PoolConfig(name="big", max_items=0))
        for i in range(100):
            pool.add(f"item{i}")
        assert pool.count() == 100


class TestPoolStateSearch:
    def test_keyword_search(self) -> None:
        pool = PoolState(PoolConfig(name="obs", dedup_content_hash=False))
        pool.add("quantum computing advances")
        pool.add("machine learning models")
        pool.add("quantum entanglement research")
        pool.add("database optimization")

        results = pool.search("quantum research")
        assert len(results) >= 1
        # Items with "quantum" should rank higher
        assert "quantum" in str(results[0]).lower()

    def test_search_dict_items(self) -> None:
        """Dict items are serialized via json.dumps for keyword matching.
        Note: json.dumps adds quotes around values, so tokens include quotes.
        The fallback keyword search is intentionally basic — BM25 handles real use.
        """
        pool = PoolState(PoolConfig(name="src", dedup_content_hash=False))
        pool.add({"content": "quantum computing advances"})
        pool.add({"content": "machine learning optimization"})
        # "computing" appears as a standalone token after json.dumps + split
        results = pool.search("computing")
        assert len(results) >= 1

    def test_search_no_match(self) -> None:
        pool = PoolState(PoolConfig(name="obs"))
        pool.add("hello world")
        results = pool.search("xyznonexistent")
        assert results == []

    def test_search_limit(self) -> None:
        pool = PoolState(PoolConfig(name="obs", dedup_content_hash=False))
        for i in range(20):
            pool.add(f"item with keyword {i}")
        results = pool.search("keyword", limit=5)
        assert len(results) <= 5


class TestPoolStateAccessors:
    def test_get_recent(self) -> None:
        pool = PoolState(PoolConfig(name="obs", dedup_content_hash=False))
        for i in range(10):
            pool.add(f"item{i}")
        recent = pool.get_recent(3)
        assert recent == ["item7", "item8", "item9"]

    def test_get_by_index(self) -> None:
        pool = PoolState(PoolConfig(name="obs"))
        pool.add("a")
        pool.add("b")
        assert pool.get_by_index(0) == "a"
        assert pool.get_by_index(1) == "b"
        assert pool.get_by_index(99) is None
        assert pool.get_by_index(-1) is None

    def test_topics(self) -> None:
        pool = PoolState(PoolConfig(name="obs"))
        pool.add({"topic": "AI", "content": "stuff"})
        pool.add({"topic": "ML", "content": "more"})
        pool.add({"topic": "AI", "content": "other"})
        assert sorted(pool.topics()) == ["AI", "ML"]

    def test_topics_from_title(self) -> None:
        pool = PoolState(PoolConfig(name="src"))
        pool.add({"title": "Paper A"})
        pool.add({"title": "Paper B"})
        assert sorted(pool.topics()) == ["Paper A", "Paper B"]


class TestPoolStateAsync:
    @pytest.mark.asyncio
    async def test_extend_async(self) -> None:
        pool = PoolState(PoolConfig(name="obs"))
        added = await pool.extend_async(["a", "b", "c"])
        assert added == 3
        assert pool.count() == 3

    @pytest.mark.asyncio
    async def test_extend_async_with_dedup(self) -> None:
        pool = PoolState(PoolConfig(name="obs"))
        pool.add("a")
        added = await pool.extend_async(["a", "b", "a", "c"])
        assert added == 2  # only b and c are new
        assert pool.count() == 3

    @pytest.mark.asyncio
    async def test_concurrent_writes(self) -> None:
        pool = PoolState(PoolConfig(name="obs", dedup_content_hash=False))

        async def writer(prefix: str) -> None:
            items = [f"{prefix}_{i}" for i in range(50)]
            await pool.extend_async(items)

        await asyncio.gather(writer("a"), writer("b"), writer("c"))
        assert pool.count() == 150


# ---------------------------------------------------------------------------
# Fix 7: PoolRegistry integration with pool tools
# ---------------------------------------------------------------------------


class TestPoolRegistrySearch:
    @pytest.mark.asyncio
    async def test_pool_search_tool_with_registry(self) -> None:
        """PoolSearchTool uses registry.search() when registry is provided."""
        from databricks_deep_research.pools.pool_registry import PoolRegistry
        from databricks_deep_research.pools.pool_tools import PoolSearchTool
        from databricks_deep_research.tools.protocol import ToolContext

        registry = PoolRegistry()
        registry.initialize_from_configs([{"name": "obs", "dedup_content_hash": False}])
        pool = registry.get("obs")
        pool.add("quantum computing advances")
        pool.add("machine learning models")
        pool.add("quantum entanglement research")

        tool = PoolSearchTool("obs", pool, registry=registry)
        result = await tool.execute(
            {"query": "quantum", "limit": 5},
            ToolContext(),
        )

        import json
        results = json.loads(result.content)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_pool_search_tool_without_registry(self) -> None:
        """PoolSearchTool falls back to PoolState.search() without registry."""
        from databricks_deep_research.pools.pool_tools import PoolSearchTool
        from databricks_deep_research.tools.protocol import ToolContext

        pool = PoolState(PoolConfig(name="obs", dedup_content_hash=False))
        pool.add("quantum computing advances")
        pool.add("machine learning models")

        tool = PoolSearchTool("obs", pool)
        result = await tool.execute(
            {"query": "quantum", "limit": 5},
            ToolContext(),
        )

        import json
        results = json.loads(result.content)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_create_pool_tools_with_registry(self) -> None:
        """create_pool_tools passes registry to PoolSearchTool."""
        from databricks_deep_research.pools.pool_registry import PoolRegistry
        from databricks_deep_research.pools.pool_tools import PoolSearchTool, create_pool_tools

        registry = PoolRegistry()
        registry.initialize_from_configs([{"name": "sources"}])
        pool = registry.get("sources")

        tools = create_pool_tools("sources", pool, registry=registry)
        assert len(tools) == 5

        # First tool should be PoolSearchTool with registry
        search_tool = tools[0]
        assert isinstance(search_tool, PoolSearchTool)
        assert search_tool._registry is registry


class TestPoolStats:
    def test_pool_stats_track_duplicate_rejections(self) -> None:
        pool = PoolState(PoolConfig(name="src", dedup_key="url"))
        assert pool.add({"url": "https://example.com/a", "title": "A"}) is True
        assert pool.add({"url": "https://example.com/a", "title": "A2"}) is False
        assert pool.stats.attempted == 2
        assert pool.stats.added == 1
        assert pool.stats.rejected_duplicate_key == 1

    def test_pool_stats_track_hash_rejections(self) -> None:
        pool = PoolState(PoolConfig(name="obs"))
        assert pool.add("same") is True
        assert pool.add("same") is False
        assert pool.stats.rejected_duplicate_hash == 1
