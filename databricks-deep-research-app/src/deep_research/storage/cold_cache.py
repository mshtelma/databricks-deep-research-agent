"""Short-TTL, small-LRU memoization for cold-path list queries.

Backs the settings / template / custom-agent endpoints where the same query
runs many times per minute but the underlying data changes rarely. A 60 s TTL
turns a 1–3 s warehouse hit into a sub-millisecond in-memory lookup.

Invalidation is naive on purpose: any `upsert_row` / `delete_row` to a given
table flushes every cached query for that table. The hit rate stays high
because reads vastly outnumber writes in these tables.
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from typing import Any


def _signature(
    table: str,
    where: dict[str, Any] | None,
    order_by: str | None,
    limit: int | None,
) -> str:
    """Stable key for a query. Deterministic across dict iteration orders."""
    payload = {
        "where": where or {},
        "order_by": order_by,
        "limit": limit,
    }
    return f"{table}::{json.dumps(payload, sort_keys=True, default=str)}"


class ColdReadCache:
    """LRU + TTL memo. Single-event-loop usage only."""

    def __init__(self, *, ttl_sec: float = 60.0, max_entries: int = 1000) -> None:
        self._ttl_sec = ttl_sec
        self._max_entries = max_entries
        # Map key → (expiry_epoch, value). OrderedDict tracks LRU.
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        # Per-table index of keys, for fast table-scoped invalidation.
        self._by_table: dict[str, set[str]] = {}
        self.hits = 0
        self.misses = 0

    def get(
        self,
        table: str,
        where: dict[str, Any] | None = None,
        order_by: str | None = None,
        limit: int | None = None,
    ) -> Any | None:
        key = _signature(table, where, order_by, limit)
        entry = self._store.get(key)
        now = time.monotonic()
        if entry is None or entry[0] < now:
            # Expired entries are lazily evicted on get.
            if entry is not None:
                self._store.pop(key, None)
                self._by_table.get(table, set()).discard(key)
            self.misses += 1
            return None
        # LRU touch.
        self._store.move_to_end(key)
        self.hits += 1
        return entry[1]

    def put(
        self,
        table: str,
        where: dict[str, Any] | None,
        order_by: str | None,
        limit: int | None,
        value: Any,
    ) -> None:
        key = _signature(table, where, order_by, limit)
        self._store[key] = (time.monotonic() + self._ttl_sec, value)
        self._store.move_to_end(key)
        self._by_table.setdefault(table, set()).add(key)
        while len(self._store) > self._max_entries:
            evicted_key, _ = self._store.popitem(last=False)
            for table_keys in self._by_table.values():
                table_keys.discard(evicted_key)

    def invalidate_table(self, table: str) -> None:
        """Flush every cached query for `table` (naive write invalidation)."""
        keys = self._by_table.pop(table, set())
        for key in keys:
            self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()
        self._by_table.clear()

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
