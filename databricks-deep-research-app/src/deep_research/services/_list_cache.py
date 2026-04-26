"""Per-user LRU cache for chat-list results.

``_UserScopedLRU`` is a process-global singleton (ClassVar on
``_CachedServiceBase``) that provides sub-5 ms p50 on cache hits for the
``CachedChatService.list()`` hot-path.

Design:
* Key: ``(user_id, status_or_empty, search_or_empty, limit, offset)``
* Value: ``(chats: list[Any], total: int, ts: float)``
* TTL: 2.0 s (configurable at construction time)
* Per-user ``asyncio.Lock`` coalesces concurrent list requests for the same
  user so they hit the backend exactly once.
* Bounded: max 10 000 entries total; oldest-insertion entry evicted on set
  when the cap is reached.
* ``invalidate_user(user_id)`` wipes every entry whose key starts with
  that user_id.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from typing import Any


class _UserScopedLRU:
    """Bounded, TTL-aware, per-user-lock LRU cache for chat list results."""

    def __init__(
        self,
        ttl_sec: float = 2.0,
        max_entries: int = 10_000,
    ) -> None:
        self._ttl = ttl_sec
        self._max = max_entries
        # Ordered so we can evict oldest on capacity overflow.
        self._store: OrderedDict[tuple[str, str, str, int, int], tuple[list[Any], int, float]] = (
            OrderedDict()
        )
        # Per-user asyncio.Lock — created on first use.
        self._user_locks: dict[str, asyncio.Lock] = {}
        # Guard for _user_locks dict itself (not for cache entries).
        self._meta_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_key(
        self,
        user_id: str,
        status: str | None,
        search: str | None,
        limit: int,
        offset: int,
    ) -> tuple[str, str, str, int, int]:
        return (user_id, status or "", search or "", limit, offset)

    async def _user_lock(self, user_id: str) -> asyncio.Lock:
        async with self._meta_lock:
            if user_id not in self._user_locks:
                self._user_locks[user_id] = asyncio.Lock()
            return self._user_locks[user_id]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        user_id: str,
        status: str | None,
        search: str | None,
        limit: int,
        offset: int,
    ) -> tuple[list[Any], int] | None:
        """Return cached result or None on miss / expired entry."""
        key = self._make_key(user_id, status, search, limit, offset)
        entry = self._store.get(key)
        if entry is None:
            return None
        chats, total, ts = entry
        if time.monotonic() - ts > self._ttl:
            # Expired — evict and report miss.
            self._store.pop(key, None)
            return None
        # Move to end (most-recently-used).
        self._store.move_to_end(key)
        return chats, total

    def set(
        self,
        user_id: str,
        status: str | None,
        search: str | None,
        limit: int,
        offset: int,
        chats: list[Any],
        total: int,
    ) -> None:
        """Insert or update a cache entry. Evicts oldest if at capacity."""
        key = self._make_key(user_id, status, search, limit, offset)
        self._store[key] = (chats, total, time.monotonic())
        self._store.move_to_end(key)
        while len(self._store) > self._max:
            self._store.popitem(last=False)  # evict oldest

    def invalidate_user(self, user_id: str) -> None:
        """Remove all entries for ``user_id``."""
        to_delete = [k for k in self._store if k[0] == user_id]
        for k in to_delete:
            self._store.pop(k, None)

    async def user_lock(self, user_id: str) -> asyncio.Lock:
        """Return (creating if needed) the per-user asyncio.Lock."""
        return await self._user_lock(user_id)

    def __len__(self) -> int:
        return len(self._store)
