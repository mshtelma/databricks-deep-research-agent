"""Shared helpers for every cache-backed service implementation.

Every `Cached<Service>` inherits (or composes) from `_CachedServiceBase` and
uses its convenience methods to talk to the `StorageStack`. Keeps per-service
boilerplate to ~3–8 lines per method.

Reading contract:
* `_read_chat(chat_id)` returns the live `ChatDocument` from the cache. The
  returned object is NOT a deep copy — callers must treat it as read-only
  and mutate only through `_mutate_chat`.
* `_mutate_chat(chat_id, fn)` takes the per-chat `asyncio.Lock`, applies
  `fn(doc)`, marks the chat dirty, and returns synchronously. Persistence is
  fire-and-forget via the `WriteQueue`.
* `_append_event(table, row)` enqueues one row into an append-only table
  buffer. Flushed on the next queue tick; bounded by buffer cap.
* `_cold_*(...)` helpers wrap the cold-path read cache + backend passthrough
  for list tables (templates, custom_agents, etc.).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any
from uuid import UUID

if TYPE_CHECKING:
    from deep_research.storage.documents import ChatDocument
    from deep_research.storage.factory import StorageStack


logger = logging.getLogger(__name__)


class _CachedServiceBase:
    """Base class for every cache-backed service.

    Concrete services inherit and add their domain methods. Pass a
    `StorageStack` at construction time; lifecycle (start/stop) is the stack's
    responsibility, not the service's.
    """

    def __init__(self, stack: StorageStack) -> None:
        self._stack = stack

    # --- Chat-scoped helpers -----------------------------------------------

    async def _read_chat(
        self,
        chat_id: UUID,
        *,
        user_id: str | None = None,
        title_hint: str = "",
    ) -> ChatDocument:
        """Return the live in-memory document. Not a deep copy."""
        return await self._stack.cache.get(
            chat_id, user_id=user_id, title_hint=title_hint
        )

    async def _mutate_chat(
        self,
        chat_id: UUID,
        fn: Callable[[ChatDocument], None],
        *,
        dirty: str = "both",
    ) -> None:
        """Apply `fn` to the live document under the per-chat lock, then mark
        the chat dirty so the `WriteQueue` flushes on its next tick.
        """
        await self._stack.cache.mutate(chat_id, fn, dirty=dirty)

    # --- Append-only helpers -----------------------------------------------

    def _append_event(self, table: str, row: Mapping[str, Any]) -> None:
        """Enqueue an append-only row. Non-blocking."""
        self._stack.queue.append_event(table, dict(row))

    # --- Cold-path list-table helpers --------------------------------------

    async def _cold_list_rows(
        self,
        table: str,
        where: Mapping[str, Any] | None = None,
        *,
        order_by: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """List rows with memoization. Hits the backend only on cache miss."""
        where_d = dict(where) if where else None
        cached = self._stack.cold_cache.get(
            table, where=where_d, order_by=order_by, limit=limit
        )
        if cached is not None:
            return cached
        rows = await self._stack.backend.list_rows(
            table, where_d or {}, order_by=order_by, limit=limit
        )
        self._stack.cold_cache.put(table, where_d, order_by, limit, rows)
        return rows

    async def _cold_upsert_row(
        self,
        table: str,
        row: Mapping[str, Any],
        *,
        pk: str,
    ) -> None:
        """Write-through upsert; invalidates the cold cache for this table."""
        await self._stack.backend.upsert_row(table, row, pk=pk)
        self._stack.cold_cache.invalidate_table(table)

    async def _cold_delete_row(
        self,
        table: str,
        pk_value: Any,
        *,
        pk: str,
    ) -> None:
        """Delete by PK; invalidates the cold cache."""
        await self._stack.backend.delete_row(table, pk_value, pk=pk)
        self._stack.cold_cache.invalidate_table(table)

    async def commit(self) -> None:
        """No-op for cached impls.

        Cold-path list-table writes (``_cold_upsert_row`` /
        ``_cold_delete_row``) are synchronous against the backend so
        there is nothing left to flush at request-end. Chat-state writes
        flow through the ``WriteQueue`` with documented eventual
        consistency. Provides API parity with legacy services that
        commit a SQLAlchemy session.
        """
        return None
