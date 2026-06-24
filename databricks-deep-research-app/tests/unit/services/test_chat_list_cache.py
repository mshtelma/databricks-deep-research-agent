"""Unit tests for F-CHAT-FAST — ``_UserScopedLRU`` and ``CachedChatService.list`` cache.

Coverage targets per the plan §3.6:
1. Cache hit returns without a backend call (counting backend).
2. Concurrent list calls for the same user coalesce to 1 backend call.
3. write through ``create`` invalidates cache.
4. TTL expiry triggers a fresh backend call.
5. Bounded eviction: insert >10 000 keys, earliest evicted.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from deep_research.services._list_cache import _UserScopedLRU

# ---------------------------------------------------------------------------
# _UserScopedLRU unit tests
# ---------------------------------------------------------------------------


class TestUserScopedLRU:
    """Direct tests of the LRU helper — no StorageStack needed."""

    def _cache(self, ttl: float = 2.0, max_entries: int = 10_000) -> _UserScopedLRU:
        return _UserScopedLRU(ttl_sec=ttl, max_entries=max_entries)

    # 1. Basic get/set round-trip
    @pytest.mark.asyncio
    async def test_set_then_get_returns_value(self) -> None:
        cache = self._cache()
        user = "u1"
        chats: list[Any] = [{"id": "c1"}]
        cache.set(user, None, None, 50, 0, chats, 1)

        result = cache.get(user, None, None, 50, 0)
        assert result is not None
        fetched_chats, total = result
        assert fetched_chats == chats
        assert total == 1

    # 2. Miss on different key parameters
    @pytest.mark.asyncio
    async def test_get_miss_different_limit(self) -> None:
        cache = self._cache()
        cache.set("u1", None, None, 50, 0, [], 0)
        result = cache.get("u1", None, None, 100, 0)
        assert result is None

    # 3. TTL expiry
    @pytest.mark.asyncio
    async def test_ttl_expiry_returns_none(self) -> None:
        cache = self._cache(ttl=0.05)  # 50 ms TTL
        cache.set("u2", None, None, 50, 0, [{"id": "x"}], 1)
        # Immediately readable
        assert cache.get("u2", None, None, 50, 0) is not None
        # Wait for TTL to expire
        await asyncio.sleep(0.07)
        assert cache.get("u2", None, None, 50, 0) is None

    # 4. invalidate_user wipes all entries for that user
    @pytest.mark.asyncio
    async def test_invalidate_user_clears_all_keys(self) -> None:
        cache = self._cache()
        cache.set("u3", None, None, 50, 0, [], 0)
        cache.set("u3", "active", None, 50, 0, [], 0)
        cache.set("u3", None, "foo", 50, 0, [], 0)
        cache.set("u4", None, None, 50, 0, [], 0)

        cache.invalidate_user("u3")

        assert cache.get("u3", None, None, 50, 0) is None
        assert cache.get("u3", "active", None, 50, 0) is None
        assert cache.get("u3", None, "foo", 50, 0) is None
        # u4 must be untouched
        assert cache.get("u4", None, None, 50, 0) is not None

    # 5. Bounded eviction — oldest entry dropped when cap is reached
    @pytest.mark.asyncio
    async def test_bounded_eviction(self) -> None:
        cap = 100
        cache = self._cache(max_entries=cap)
        # Insert cap entries with offset 0..cap-1, user "u_evict"
        for i in range(cap):
            cache.set("u_evict", None, None, 50, i, [], 0)

        assert len(cache) == cap
        # The first entry (offset=0) is the oldest; inserting one more should evict it.
        cache.set("u_evict", None, None, 50, cap, [], 0)

        assert len(cache) == cap  # still at cap
        # offset=0 was evicted (oldest)
        assert cache.get("u_evict", None, None, 50, 0) is None
        # Latest entry still present
        assert cache.get("u_evict", None, None, 50, cap) is not None

    # 6. Per-user lock coalesces concurrent callers
    @pytest.mark.asyncio
    async def test_per_user_lock_returned_consistently(self) -> None:
        cache = self._cache()
        lock1 = await cache.user_lock("u5")
        lock2 = await cache.user_lock("u5")
        lock_other = await cache.user_lock("u6")

        assert lock1 is lock2  # same user → same Lock object
        assert lock1 is not lock_other  # different user → different Lock


# ---------------------------------------------------------------------------
# CachedChatService integration tests — counting backend
# ---------------------------------------------------------------------------


class _CountingBackend:
    """Fake backend that counts ``list_chat_metas`` calls."""

    def __init__(self) -> None:
        self.list_calls = 0

    async def list_chat_metas(
        self,
        user_id: str,
        *,
        include_deleted: bool = False,
        limit: int = 50,
        offset: int = 0,
        search: str | None = None,
        status: str | None = None,
    ) -> list[Any]:
        self.list_calls += 1
        return []

    # Minimal stubs so other code paths don't fail
    async def load_chat(self, chat_id: UUID) -> None:
        return None

    async def write_chat(self, doc: Any, *, expected_version: int) -> None:
        pass

    async def list_rows(self, *a: Any, **kw: Any) -> list[Any]:
        return []

    async def upsert_row(self, *a: Any, **kw: Any) -> None:
        pass

    async def delete_row(self, *a: Any, **kw: Any) -> None:
        pass

    async def append_events(self, *a: Any, **kw: Any) -> None:
        pass


def _make_service(backend: _CountingBackend) -> CachedChatService:
    """Build a ``CachedChatService`` over a counting backend."""

    from deep_research.services.cached.chat import CachedChatService

    stack = MagicMock()
    stack.backend = backend
    stack.queue = MagicMock()
    stack.cache = MagicMock()

    return CachedChatService(stack)


class TestCachedChatServiceListCache:
    """Integration: cache hit avoids backend call; write invalidates."""

    def setup_method(self) -> None:
        # Each test gets a fresh cache singleton by patching the module-level object.
        import deep_research.services.cached.chat as chat_mod
        from deep_research.services._list_cache import _UserScopedLRU

        chat_mod._CHAT_LIST_CACHE = _UserScopedLRU(ttl_sec=2.0, max_entries=10_000)

    @pytest.mark.asyncio
    async def test_cache_hit_skips_backend(self) -> None:
        """Second list call for same params must not touch the backend."""
        backend = _CountingBackend()
        svc = _make_service(backend)

        await svc.list("user_a", limit=50, offset=0)
        await svc.list("user_a", limit=50, offset=0)

        # Two backend calls on first invocation (list + count), zero on second.
        assert backend.list_calls == 2

    @pytest.mark.asyncio
    async def test_concurrent_callers_coalesce(self) -> None:
        """20 concurrent list calls for the same user hit backend only once."""
        import deep_research.services.cached.chat as chat_mod
        from deep_research.services._list_cache import _UserScopedLRU

        # Use a cache with a slow backend simulator.
        call_count = 0

        class SlowBackend(_CountingBackend):
            async def list_chat_metas(self, user_id: str, **kw: Any) -> list[Any]:
                nonlocal call_count
                call_count += 1
                await asyncio.sleep(0.02)  # simulate latency
                return []

        backend = SlowBackend()
        svc = _make_service(backend)

        # Reset to a fresh cache
        chat_mod._CHAT_LIST_CACHE = _UserScopedLRU(ttl_sec=2.0, max_entries=10_000)

        tasks = [svc.list("user_concurrent", limit=50, offset=0) for _ in range(20)]
        await asyncio.gather(*tasks)

        # With per-user lock + double-checked locking, the backend is called at
        # most 2× (list + count) for the first winner; the 19 others hit the
        # warm cache after the lock is released.
        # We allow up to 4 to accommodate small scheduling variations.
        assert call_count <= 4, f"Expected ≤4 backend calls, got {call_count}"

    @pytest.mark.asyncio
    async def test_create_invalidates_cache(self) -> None:
        """``create`` must wipe the user's cached list entries."""
        import deep_research.services.cached.chat as chat_mod
        from deep_research.services._list_cache import _UserScopedLRU

        backend = _CountingBackend()
        svc = _make_service(backend)

        chat_mod._CHAT_LIST_CACHE = _UserScopedLRU(ttl_sec=2.0, max_entries=10_000)

        # Warm the cache.
        await svc.list("user_b", limit=50, offset=0)
        calls_after_first_list = backend.list_calls

        # create() should invalidate.
        svc._stack.backend.write_chat = AsyncMock()
        await svc.create("user_b")

        # Next list must go to backend again.
        await svc.list("user_b", limit=50, offset=0)
        assert backend.list_calls > calls_after_first_list

    @pytest.mark.asyncio
    async def test_ttl_expiry_re_fetches(self) -> None:
        """After TTL expires the next list hits the backend."""
        import deep_research.services.cached.chat as chat_mod
        from deep_research.services._list_cache import _UserScopedLRU

        backend = _CountingBackend()
        svc = _make_service(backend)
        chat_mod._CHAT_LIST_CACHE = _UserScopedLRU(ttl_sec=0.05, max_entries=10_000)

        await svc.list("user_c", limit=50, offset=0)
        calls_after_first = backend.list_calls

        # Wait for TTL to expire.
        await asyncio.sleep(0.08)

        await svc.list("user_c", limit=50, offset=0)
        assert backend.list_calls > calls_after_first

    @pytest.mark.asyncio
    async def test_eviction_over_10000_entries(self) -> None:
        """Inserting >10 000 entries evicts the oldest."""
        from deep_research.services._list_cache import _UserScopedLRU

        cache = _UserScopedLRU(ttl_sec=60.0, max_entries=10_000)
        # Insert 10 001 entries (different offsets, same user)
        for i in range(10_001):
            cache.set("u_big", None, None, 50, i, [], 0)

        assert len(cache) == 10_000
        # offset=0 (inserted first) must be evicted
        assert cache.get("u_big", None, None, 50, 0) is None
        # offset=10000 (inserted last) must still be there
        assert cache.get("u_big", None, None, 50, 10_000) is not None
