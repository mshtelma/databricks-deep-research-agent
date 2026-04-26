"""Per-chat in-memory cache + hydrator.

`ChatStateCache` holds the runtime source-of-truth copy of every active chat's
`ChatDocument`. Reads (`get`) await the hydration future; mutations (`mutate`)
are synchronous against the in-memory doc and notify the `WriteQueue` via the
injected `on_dirty` callback.

Hydration is driven by the `Hydrator`, which is triggered as a side effect of
chat-list GET (prefetch) and chat-detail GET / `POST /research` (eager start).
When a request handler calls `cache.get()`, the future has usually resolved;
if not, the await completes within a warehouse round-trip.

Migration mode: calling `cache.set_migration_mode(True)` causes new hydrations
to block briefly and then raise `MigrationInProgressError` (routes translate
to HTTP 503). Already-hydrated entries remain readable — the WriteQueue is
what truly pauses, not the cache.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal
from uuid import UUID

from deep_research.storage.backend import (
    MigrationInProgressError,
    StorageBackend,
)
from deep_research.storage.documents import ChatDocument

logger = logging.getLogger(__name__)

# Typed alias: notification callback the WriteQueue subscribes to. The second
# argument tells the queue which table needs flushing ("state" | "meta").
DirtyCallback = Callable[[UUID, str], None]

# Mutation dirtiness scope.
DirtyScope = Literal["state", "meta", "both"]


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


@dataclass
class _CacheEntry:
    """Per-chat cache record.

    Note: dirty tracking lives in the `WriteQueue`, not here. This entry only
    holds the hydrated doc + synchronization primitives. The queue learns
    about mutations via the `DirtyCallback` passed to `ChatStateCache`.
    """

    hydration: asyncio.Future[None]
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    doc: ChatDocument | None = None
    last_access: datetime = field(default_factory=_utcnow)
    # `user_id` is provided up front so that a brand-new chat (load returns
    # None) can be constructed without a second lookup.
    user_id: str | None = None
    title_hint: str = ""


class ChatStateCache:
    """Per-process chat document cache."""

    def __init__(
        self,
        backend: StorageBackend,
        *,
        idle_ttl_min: int = 30,
        migration_timeout_sec: float = 2.0,
        reaper_interval_sec: float = 60.0,
        on_dirty: DirtyCallback | None = None,
    ) -> None:
        self._backend = backend
        self._idle_ttl_min = idle_ttl_min
        self._migration_timeout_sec = migration_timeout_sec
        self._reaper_interval_sec = reaper_interval_sec
        self._on_dirty = on_dirty

        self._entries: dict[UUID, _CacheEntry] = {}

        # `_migration_mode` starts False → no pause.
        self._migration_mode = False
        # Event is "set" when NOT in migration mode. Hydrations that arrive
        # during migration mode `.wait()` on this with a timeout.
        self._migration_cleared = asyncio.Event()
        self._migration_cleared.set()

        self._reaper_task: asyncio.Task[None] | None = None
        self._stopping = False

    # -- Lifecycle -----------------------------------------------------

    def start_reaper(self) -> None:
        if self._reaper_task is None:
            self._reaper_task = asyncio.create_task(
                self._reaper_loop(), name="cache-reaper"
            )

    async def stop_reaper(self) -> None:
        self._stopping = True
        task = self._reaper_task
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            self._reaper_task = None

    async def evict_all(self) -> None:
        """Drop every entry. Caller must have drained the WriteQueue first."""
        self._entries.clear()

    # -- Migration-mode control ----------------------------------------

    def set_migration_mode(self, enabled: bool) -> None:
        self._migration_mode = enabled
        if enabled:
            self._migration_cleared.clear()
        else:
            self._migration_cleared.set()

    @property
    def migration_mode(self) -> bool:
        return self._migration_mode

    # -- Internal: hydration -------------------------------------------

    def start_hydration(
        self,
        chat_id: UUID,
        *,
        user_id: str | None = None,
        title_hint: str = "",
    ) -> _CacheEntry:
        """Start a hydration task for `chat_id` if none is in flight.

        Idempotent: if an entry already exists, return it unchanged. Called by
        the `Hydrator` and lazily by `get()` as a safety net.
        """
        existing = self._entries.get(chat_id)
        if existing is not None:
            if user_id and existing.user_id is None:
                existing.user_id = user_id
            return existing

        loop = asyncio.get_event_loop()
        entry = _CacheEntry(
            hydration=loop.create_future(),
            user_id=user_id,
            title_hint=title_hint,
        )
        self._entries[chat_id] = entry

        async def _run() -> None:
            try:
                # Brief wait if migration mode engaged — we don't start new
                # hydrations during migration.
                if self._migration_mode:
                    try:
                        await asyncio.wait_for(
                            self._migration_cleared.wait(),
                            timeout=self._migration_timeout_sec,
                        )
                    except asyncio.TimeoutError as exc:
                        raise MigrationInProgressError(
                            f"cannot hydrate chat {chat_id} during migration"
                        ) from exc

                doc = await self._backend.load_chat(chat_id)
                if doc is None:
                    if entry.user_id is None:
                        raise ValueError(
                            f"chat {chat_id} does not exist and no user_id "
                            "was provided for fresh construction"
                        )
                    doc = ChatDocument.new(
                        chat_id, entry.user_id, title=entry.title_hint
                    )
                entry.doc = doc
                entry.hydration.set_result(None)
            except BaseException as exc:  # noqa: BLE001 — surfaces via the future
                if not entry.hydration.done():
                    entry.hydration.set_exception(exc)
                # Drop the entry so a retry creates a fresh future. We do NOT
                # re-raise here — the exception is exposed to awaiters through
                # `entry.hydration`; re-raising would produce a spurious
                # "Task exception was never retrieved" warning since no code
                # awaits the spawned task directly.
                self._entries.pop(chat_id, None)

        asyncio.create_task(_run(), name=f"hydrate-{chat_id}")
        return entry

    # -- Public read / mutate -----------------------------------------

    async def get(
        self,
        chat_id: UUID,
        *,
        user_id: str | None = None,
        title_hint: str = "",
    ) -> ChatDocument:
        """Return the hydrated document; await the existing future if any.

        Does NOT deep-copy — the returned object is the live in-memory doc and
        MUST be treated as read-only. Use `mutate()` to modify.
        """
        entry = self._entries.get(chat_id)
        if entry is None:
            entry = self.start_hydration(
                chat_id, user_id=user_id, title_hint=title_hint
            )
        await entry.hydration
        assert entry.doc is not None  # noqa: S101 — hydration success invariant
        entry.last_access = _utcnow()
        return entry.doc

    async def mutate(
        self,
        chat_id: UUID,
        fn: Callable[[ChatDocument], None],
        *,
        dirty: DirtyScope = "both",
    ) -> None:
        """Apply `fn` to the in-memory doc under the per-chat lock.

        Marks the chat dirty and invokes `on_dirty` so the WriteQueue schedules
        a flush on its next tick.
        """
        entry = self._entries.get(chat_id)
        if entry is None or not entry.hydration.done() or entry.doc is None:
            raise RuntimeError(
                f"chat {chat_id} is not hydrated; call get() before mutate()"
            )
        async with entry.lock:
            fn(entry.doc)
            entry.doc.meta.updated_at = _utcnow()
            entry.last_access = entry.doc.meta.updated_at

        if self._on_dirty is not None:
            if dirty in ("state", "both"):
                self._on_dirty(chat_id, "state")
            if dirty in ("meta", "both"):
                self._on_dirty(chat_id, "meta")

    # -- Internal: snapshot for the WriteQueue -------------------------

    def snapshot(self, chat_id: UUID) -> ChatDocument | None:
        """Return a deep-copy of the current doc, or None if not hydrated.

        Called by the WriteQueue just before flushing. The queue — not the
        cache — owns dirty tracking, so this is a pure read; no flag state
        is modified here.
        """
        entry = self._entries.get(chat_id)
        if entry is None or entry.doc is None or not entry.hydration.done():
            return None
        return entry.doc.model_copy(deep=True)

    def mark_flushed(
        self,
        chat_id: UUID,
        *,
        new_version: int,
    ) -> None:
        """Bump the cache's mirror of `chat_meta.version` after a successful
        flush so the next flush's version gate still matches.
        """
        entry = self._entries.get(chat_id)
        if entry is not None and entry.doc is not None:
            entry.doc.meta.version = new_version

    # -- Reaper --------------------------------------------------------

    async def _reaper_loop(self) -> None:
        cutoff = timedelta(minutes=self._idle_ttl_min)
        while not self._stopping:
            try:
                await asyncio.sleep(self._reaper_interval_sec)
                deadline = _utcnow() - cutoff
                for chat_id in list(self._entries.keys()):
                    entry = self._entries.get(chat_id)
                    # The WriteQueue's dirty set gates eviction externally via
                    # `evict_if_clean`; the reaper defers to that call site so
                    # it never drops a chat with in-flight mutations. Here we
                    # only free entries whose last access is beyond the TTL
                    # and whose hydration has completed.
                    if (
                        entry is not None
                        and entry.last_access < deadline
                        and entry.hydration.done()
                    ):
                        self._entries.pop(chat_id, None)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — reaper never crashes the app
                logger.exception("cache reaper loop hiccup")


class Hydrator:
    """Fire-and-forget hydration trigger."""

    def __init__(self, cache: ChatStateCache, backend: StorageBackend) -> None:
        self._cache = cache
        self._backend = backend

    def start(
        self,
        chat_id: UUID,
        *,
        user_id: str | None = None,
        title_hint: str = "",
    ) -> None:
        """Begin hydration if not already running. Idempotent, non-blocking."""
        self._cache.start_hydration(
            chat_id, user_id=user_id, title_hint=title_hint
        )

    async def prefetch(self, user_id: str, *, top_n: int = 3) -> None:
        """Prefetch the top-N most recent chats for this user.

        Called from the chat-list GET handler as a side effect: by the time
        the user clicks into one of the returned chats, hydration has usually
        already resolved.
        """
        metas = await self._backend.list_chat_metas(user_id, limit=top_n)
        for meta in metas:
            self.start(meta.chat_id, user_id=user_id, title_hint=meta.title)
