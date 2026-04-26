"""Unit tests for `deep_research.storage.cold_cache.ColdReadCache`."""

from __future__ import annotations

import asyncio

import pytest

from deep_research.storage.cold_cache import ColdReadCache


class TestColdReadCache:
    def test_miss_returns_none(self) -> None:
        cache = ColdReadCache()
        assert cache.get("t", where={"owner_id": "u"}) is None
        assert cache.misses == 1
        assert cache.hits == 0

    def test_hit_returns_stored_value(self) -> None:
        cache = ColdReadCache()
        cache.put("t", {"owner_id": "u"}, None, None, [1, 2, 3])
        assert cache.get("t", where={"owner_id": "u"}) == [1, 2, 3]
        assert cache.hits == 1

    def test_different_wheres_are_distinct_keys(self) -> None:
        cache = ColdReadCache()
        cache.put("t", {"owner_id": "u1"}, None, None, [1])
        cache.put("t", {"owner_id": "u2"}, None, None, [2])
        assert cache.get("t", {"owner_id": "u1"}) == [1]
        assert cache.get("t", {"owner_id": "u2"}) == [2]

    def test_order_by_and_limit_part_of_key(self) -> None:
        cache = ColdReadCache()
        cache.put("t", None, "name", 10, [1])
        cache.put("t", None, "name", 20, [2])
        assert cache.get("t", None, "name", 10) == [1]
        assert cache.get("t", None, "name", 20) == [2]

    async def test_ttl_expiry(self) -> None:
        cache = ColdReadCache(ttl_sec=0.1)
        cache.put("t", {"o": "u"}, None, None, [42])
        assert cache.get("t", {"o": "u"}) == [42]
        await asyncio.sleep(0.15)
        assert cache.get("t", {"o": "u"}) is None

    def test_invalidate_table_clears_only_that_table(self) -> None:
        cache = ColdReadCache()
        cache.put("templates", {"o": "u"}, None, None, [1])
        cache.put("agents", {"o": "u"}, None, None, [2])
        cache.invalidate_table("templates")
        assert cache.get("templates", {"o": "u"}) is None
        assert cache.get("agents", {"o": "u"}) == [2]

    def test_lru_eviction_removes_least_recent(self) -> None:
        cache = ColdReadCache(max_entries=3)
        cache.put("t", {"k": "a"}, None, None, ["a"])
        cache.put("t", {"k": "b"}, None, None, ["b"])
        cache.put("t", {"k": "c"}, None, None, ["c"])
        # Touch 'a' → becomes MRU
        cache.get("t", {"k": "a"})
        # Insert 'd' → should evict 'b' (LRU)
        cache.put("t", {"k": "d"}, None, None, ["d"])
        assert cache.get("t", {"k": "a"}) == ["a"]
        assert cache.get("t", {"k": "b"}) is None
        assert cache.get("t", {"k": "c"}) == ["c"]
        assert cache.get("t", {"k": "d"}) == ["d"]

    def test_clear_empties_everything(self) -> None:
        cache = ColdReadCache()
        cache.put("t", None, None, None, [1])
        cache.clear()
        assert cache.get("t") is None

    def test_hit_rate_property(self) -> None:
        cache = ColdReadCache()
        cache.put("t", None, None, None, [1])
        cache.get("t")  # hit
        cache.get("t", {"o": "u"})  # miss
        assert cache.hit_rate == pytest.approx(0.5)
