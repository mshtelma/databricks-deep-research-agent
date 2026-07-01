"""Coalescing, batched, fire-and-forget write pipeline.

`WriteQueue` is the only component that calls the storage backend's `write_*`
and `append_events` methods at runtime. Every mutation initiated by a service
funnels through here:

* Document writes (chat_state, chat_meta, user_documents, prep_job_documents)
  are driven by a *dirty set* — coalesces N per-turn mutations into 1 backend
  call per flush tick.
* Append-only tables (research_events, file_chunks, audit_log,
  message_feedback) use a ring buffer that is drained up to `flush_size` rows
  per tick, yielding 1 bulk INSERT per table per tick.

Behavior guarantees:

1. `mark_*_dirty(...)` and `append_event(...)` never block. Even during
   migration mode (`set_migration_mode(True)`) the enqueue path is free so
   in-flight turns can complete.
2. Flush loops pause during migration mode — they `await` an `asyncio.Event`
   that is clear-in-mode, set-otherwise.
3. Chat-state writes are version-gated. A `ConflictError` triggers a
   controlled re-read + re-enqueue; never silently overwrites.
4. TransientError triggers retry with bounded exponential backoff (100ms,
   500ms, 2s). After 3 attempts the op is dropped and
   `storage_dropped_writes_total` is incremented.
5. `stop(timeout)` drains every dirty set and buffer synchronously before
   returning, so shutdown data-loss is bounded.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from deep_research.storage.backend import (
    ConflictError,
    PermanentError,
    StorageBackend,
    TransientError,
)
from deep_research.storage.cache import ChatStateCache
from deep_research.storage.documents import ChatMeta, PrepJobDocument, UserDocument

logger = logging.getLogger(__name__)

# Default retry schedule in seconds.
_BACKOFFS: tuple[float, ...] = (0.1, 0.5, 2.0)


# --- Small stats record for observability -----------------------------------


@dataclass
class QueueStats:
    chat_state_flushes: int = 0
    events_flushes: int = 0
    conflicts: int = 0
    retries: int = 0
    dropped: int = 0

    # Back-compat alias for the old split-loop variant. `write_chat` flushes
    # meta + state + projection atomically so every increment of
    # `chat_state_flushes` is also a meta flush.
    @property
    def chat_meta_flushes(self) -> int:
        return self.chat_state_flushes


# --- The queue --------------------------------------------------------------


@dataclass
class _UserDocDirty:
    """Dirty user-doc entry held until flush."""

    doc: UserDocument
    enqueued_at: float = field(default_factory=time.monotonic)


@dataclass
class _PrepJobDirty:
    doc: PrepJobDocument
    enqueued_at: float = field(default_factory=time.monotonic)


class WriteQueue:
    """Per-process writer pipeline.

    Not thread-safe. All methods (including the `mark_*` enqueue path) must be
    called from the event loop that the background tasks run on.
    """

    def __init__(
        self,
        backend: StorageBackend,
        cache: ChatStateCache,
        *,
        flush_interval_sec: float = 3.0,
        flush_size: int = 200,
        shutdown_timeout_sec: float = 15.0,
        backoffs: tuple[float, ...] = _BACKOFFS,
    ) -> None:
        self._backend = backend
        self._cache = cache
        self._flush_interval_sec = flush_interval_sec
        self._flush_size = flush_size
        self._shutdown_timeout_sec = shutdown_timeout_sec
        self._backoffs = backoffs

        # Dirty-set state. Sets are used because per-chat coalescing collapses
        # N mutations to 1 flush automatically.
        self._state_dirty: set[UUID] = set()
        self._user_dirty: dict[str, _UserDocDirty] = {}
        self._prep_dirty: dict[UUID, _PrepJobDirty] = {}

        # Append buffers per table.
        self._events: dict[str, deque[dict[str, Any]]] = {}

        # Migration pause — set ⇒ *not* in migration mode (flushes run).
        self._paused_gate = asyncio.Event()
        self._paused_gate.set()

        self._tasks: list[asyncio.Task[None]] = []
        self._stopping = False
        self.stats = QueueStats()

    # -- Subscription glue for ChatStateCache -------------------------

    def notify_dirty(self, chat_id: UUID, _scope: str) -> None:
        """Callback wired into `ChatStateCache.on_dirty`.

        `write_chat` is atomic over meta + state + projection, so we collapse
        both scopes into a single dirty set. The first scope that fires per
        tick wins; subsequent scopes for the same chat coalesce to a no-op.
        """
        self._state_dirty.add(chat_id)

    # -- Direct enqueue (cold-path services) --------------------------

    def mark_chat_dirty(self, chat_id: UUID) -> None:
        self._state_dirty.add(chat_id)

    # Back-compat aliases (a single flush path covers both scopes).
    mark_chat_state_dirty = mark_chat_dirty
    mark_chat_meta_dirty = mark_chat_dirty

    def mark_user_doc_dirty(self, doc: UserDocument) -> None:
        self._user_dirty[doc.user_id] = _UserDocDirty(doc=doc)

    def mark_prep_job_dirty(self, doc: PrepJobDocument) -> None:
        self._prep_dirty[doc.prep_job_id] = _PrepJobDirty(doc=doc)

    def append_event(self, table: str, row: dict[str, Any]) -> None:
        self._events.setdefault(table, deque()).append(row)

    # -- Migration-mode control ---------------------------------------

    def set_migration_mode(self, enabled: bool) -> None:
        if enabled:
            self._paused_gate.clear()
        else:
            self._paused_gate.set()

    # -- Lifecycle ----------------------------------------------------

    def start(self, event_tables: tuple[str, ...] = ()) -> None:
        """Start background flush tasks.

        `event_tables` lists the append-only tables that should get a
        dedicated flush loop. Other tables can still receive events via
        `append_event`; they will flush on shutdown drain at latest.
        """
        self._stopping = False
        self._tasks.append(
            asyncio.create_task(self._run_state_loop(), name="writequeue-state")
        )
        self._tasks.append(
            asyncio.create_task(self._run_user_loop(), name="writequeue-user")
        )
        self._tasks.append(
            asyncio.create_task(self._run_prep_loop(), name="writequeue-prep")
        )
        for table in event_tables:
            self._tasks.append(
                asyncio.create_task(
                    self._run_events_loop(table),
                    name=f"writequeue-events-{table}",
                )
            )

    async def stop(self) -> None:
        """Drain all buffers and stop background tasks.

        Bounded by `shutdown_timeout_sec`: after that, any remaining work is
        logged and dropped. Safe to call more than once.
        """
        self._stopping = True
        # One last pass at full speed — flush everything currently pending.
        try:
            await asyncio.wait_for(
                self._drain_all(), timeout=self._shutdown_timeout_sec
            )
        except TimeoutError:
            logger.warning(
                "WriteQueue shutdown drain timed out after %.1fs; "
                "remaining: state=%d user=%d prep=%d events=%s",
                self._shutdown_timeout_sec,
                len(self._state_dirty),
                len(self._user_dirty),
                len(self._prep_dirty),
                {t: len(q) for t, q in self._events.items()},
            )
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()

    async def flush_chat_now(self, chat_id: UUID) -> None:
        """Synchronously flush a single chat. Used by cache eviction."""
        await self._flush_chat_state_one(chat_id)

    # -- Internal flush loops -----------------------------------------

    async def _run_state_loop(self) -> None:
        await self._tick_loop(self._flush_chat_state_batch)

    async def _run_user_loop(self) -> None:
        await self._tick_loop(self._flush_user_batch)

    async def _run_prep_loop(self) -> None:
        await self._tick_loop(self._flush_prep_batch)

    async def _run_events_loop(self, table: str) -> None:
        await self._tick_loop(lambda: self._flush_events_batch(table))

    async def _tick_loop(self, body: Callable[[], Awaitable[None]]) -> None:
        while not self._stopping:
            try:
                await asyncio.sleep(self._flush_interval_sec)
                # Gate the body, not the sleep — so a tick that began before
                # migration-mode engaged does NOT slip a flush through. Every
                # body invocation re-checks the gate right before firing.
                await self._paused_gate.wait()
                await body()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — loop must never exit
                logger.exception("WriteQueue tick hiccup")

    # -- Internal flush bodies ----------------------------------------

    async def _flush_chat_state_batch(self) -> None:
        if not self._state_dirty:
            return
        ids = list(self._state_dirty)
        self._state_dirty.clear()
        await asyncio.gather(
            *(self._flush_chat_state_one(cid) for cid in ids),
            return_exceptions=True,
        )

    async def _flush_chat_state_one(self, chat_id: UUID) -> None:
        doc = self._cache.snapshot(chat_id)
        if doc is None:
            return
        # Materialize the preview at flush time so list_chats never parses JSON.
        doc.meta.preview = ChatMeta.preview_from_state(doc.state)
        try:
            new_version = await self._attempt(
                lambda: self._backend.write_chat(
                    doc, expected_version=doc.meta.version
                )
            )
        except ConflictError:
            self.stats.conflicts += 1
            # Re-read so the next attempt uses the correct version.
            try:
                fresh = await self._backend.load_chat(chat_id)
                if fresh is not None:
                    self._cache.mark_flushed(
                        chat_id, new_version=fresh.meta.version
                    )
            except Exception:  # noqa: BLE001 — re-read best-effort
                logger.exception("post-conflict re-read failed for %s", chat_id)
            # Re-enqueue so the next tick tries again with the new version.
            self._state_dirty.add(chat_id)
            return
        except PermanentError:
            self.stats.dropped += 1
            logger.exception("dropping chat write for %s", chat_id)
            return
        except Exception:  # noqa: BLE001 — final retry exhausted
            self.stats.dropped += 1
            # Put it back so the next tick retries; the attempt exhaustion was
            # TransientError-driven, not a permanent rejection.
            self._state_dirty.add(chat_id)
            logger.exception("chat write failed for %s; will retry", chat_id)
            return
        self._cache.mark_flushed(chat_id, new_version=new_version)
        self.stats.chat_state_flushes += 1

    async def _flush_user_batch(self) -> None:
        if not self._user_dirty:
            return
        entries = list(self._user_dirty.values())
        self._user_dirty.clear()
        await asyncio.gather(
            *(self._flush_user_one(e.doc) for e in entries),
            return_exceptions=True,
        )

    async def _flush_user_one(self, doc: UserDocument) -> None:
        try:
            await self._attempt(lambda: self._backend.write_user_doc(doc))
        except Exception:  # noqa: BLE001
            self.stats.dropped += 1
            logger.exception("user doc flush failed for %s", doc.user_id)

    async def _flush_prep_batch(self) -> None:
        if not self._prep_dirty:
            return
        entries = list(self._prep_dirty.values())
        self._prep_dirty.clear()
        await asyncio.gather(
            *(self._flush_prep_one(e.doc) for e in entries),
            return_exceptions=True,
        )

    async def _flush_prep_one(self, doc: PrepJobDocument) -> None:
        try:
            await self._attempt(lambda: self._backend.write_prep_job(doc))
        except Exception:  # noqa: BLE001
            self.stats.dropped += 1
            logger.exception("prep_job flush failed for %s", doc.prep_job_id)

    async def _flush_events_batch(self, table: str) -> None:
        buf = self._events.get(table)
        if not buf:
            return
        batch: list[dict[str, Any]] = []
        for _ in range(min(self._flush_size, len(buf))):
            batch.append(buf.popleft())
        if not batch:
            return
        try:
            await self._attempt(
                lambda: self._backend.append_events(table, batch)
            )
        except Exception:  # noqa: BLE001
            # Preserve order; re-prepend in original order.
            for row in reversed(batch):
                buf.appendleft(row)
            self.stats.dropped += len(batch)
            logger.exception("events flush failed for %s", table)
            return
        self.stats.events_flushes += 1

    # -- Retry core ----------------------------------------------------

    async def _attempt(self, op: Callable[[], Awaitable[Any]]) -> Any:
        backoffs = self._backoffs
        for i, delay in enumerate(backoffs):
            try:
                return await op()
            except (ConflictError, PermanentError):
                raise  # caller handles
            except TransientError:
                self.stats.retries += 1
                if i == len(backoffs) - 1:
                    raise
                await asyncio.sleep(delay)
        # Unreachable
        raise RuntimeError("retry loop exited unexpectedly")

    # -- Drain helper --------------------------------------------------

    async def _drain_all(self) -> None:
        while (
            self._state_dirty
            or self._user_dirty
            or self._prep_dirty
            or any(self._events.values())
        ):
            await self._flush_chat_state_batch()
            await self._flush_user_batch()
            await self._flush_prep_batch()
            for table in list(self._events.keys()):
                while self._events.get(table):
                    await self._flush_events_batch(table)
