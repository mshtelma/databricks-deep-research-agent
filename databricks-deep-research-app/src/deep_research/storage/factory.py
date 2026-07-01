"""Factory: build a `StorageStack` (backend + cache + queue + hydrator + cold cache + cleanup) from settings.

Late-binding resolves the cache ↔ queue cyclic dependency: cache is built with
`on_dirty=None`, queue is built referencing the cache, then
`cache.set_on_dirty(queue.notify_dirty)` wires the signal.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
from dataclasses import dataclass
from typing import TYPE_CHECKING

from deep_research.storage.backend import StorageBackend
from deep_research.storage.cache import ChatStateCache, Hydrator
from deep_research.storage.cold_cache import ColdReadCache
from deep_research.storage.queue import WriteQueue

if TYPE_CHECKING:  # pragma: no cover
    from deep_research.core.config import Settings
    from deep_research.storage.cleanup import CleanupLoop

logger = logging.getLogger(__name__)

# Tables that get a dedicated event-flush loop in the WriteQueue.
_EVENT_TABLES: tuple[str, ...] = (
    "research_events",
    "file_chunks",
    "message_feedback",
    "audit_log",
)


def create_backend(settings: Settings) -> StorageBackend:
    """Return the wire backend selected by `settings.storage_backend`.

    Raises ValueError if the backend name is unknown or misconfigured.
    """
    name = settings.storage_backend
    if name == "lakebase":
        from deep_research.storage.lakebase import LakebaseBackend

        # Pass `storage_schema` so chat-document tables live in their own
        # Postgres schema (e.g. ``deep_research_state``) and don't collide
        # with legacy Alembic tables in ``public``.
        return LakebaseBackend(schema=settings.storage_schema)

    if name == "sql_warehouse":
        if not settings.storage_warehouse_id:
            raise ValueError(
                "STORAGE_BACKEND=sql_warehouse requires STORAGE_WAREHOUSE_ID"
            )
        from deep_research.storage.sql_warehouse import SQLWarehouseBackend

        return SQLWarehouseBackend(
            warehouse_id=settings.storage_warehouse_id,
            catalog=settings.storage_catalog,
            schema=settings.storage_schema,
            timeout_sec=settings.storage_statement_timeout_sec,
        )

    if name == "fake":
        # Deferred import — tests supply this path.
        from tests.fakes.fake_backend import FakeBackend

        return FakeBackend()

    raise ValueError(f"unknown STORAGE_BACKEND={name!r}")


@dataclass
class StorageStack:
    """Bundle of runtime components. Owned by the app lifespan."""

    backend: StorageBackend
    cache: ChatStateCache
    queue: WriteQueue
    hydrator: Hydrator
    cold_cache: ColdReadCache
    cleanup: CleanupLoop | None = None

    _started: bool = False
    _signal_handlers_installed: bool = False

    async def start(self) -> None:
        """Run DDL, start background tasks. Idempotent."""
        if self._started:
            return
        await self.backend.migrate()
        self.cache.start_reaper()
        self.queue.start(event_tables=_EVENT_TABLES)
        if self.cleanup is not None:
            self.cleanup.start()
        self._started = True
        logger.info("StorageStack started", extra={"started": True})

    async def stop(self, *, timeout: float = 15.0) -> None:
        """Drain queue, stop reapers, close backend. Idempotent."""
        if not self._started:
            return
        if self.cleanup is not None:
            try:
                await self.cleanup.stop()
            except Exception:  # noqa: BLE001 — shutdown must not raise
                logger.exception("cleanup stop failed")
        try:
            self.queue._shutdown_timeout_sec = timeout
            await self.queue.stop()
        except Exception:  # noqa: BLE001
            logger.exception("queue stop failed")
        try:
            await self.cache.stop_reaper()
        except Exception:  # noqa: BLE001
            logger.exception("reaper stop failed")
        try:
            await self.backend.close()
        except Exception:  # noqa: BLE001
            logger.exception("backend close failed")
        self._started = False

    def install_signal_handlers(self) -> None:
        """Register SIGTERM handler that drains the queue on shutdown.

        Idempotent. No-op on platforms / contexts without signal support.
        """
        if self._signal_handlers_installed:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop (e.g. under a test harness with asyncio.run).
            logger.debug("no running loop; skipping signal handler install")
            return

        def _on_term() -> None:
            logger.info("SIGTERM received; scheduling storage drain")
            loop.create_task(self.stop(timeout=15.0))

        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(NotImplementedError, ValueError):
                loop.add_signal_handler(sig, _on_term)
        self._signal_handlers_installed = True


def create_storage_stack(settings: Settings) -> StorageStack:
    """Build the full `StorageStack` wired up against the given settings."""
    backend = create_backend(settings)

    cold_cache = ColdReadCache(
        ttl_sec=settings.storage_cold_cache_ttl_sec,
        max_entries=settings.storage_cold_cache_max_entries,
    )

    # Late-binding: cache needs a `notify_dirty` callback; queue needs the
    # cache to snapshot. Resolve by delaying the wire-up until both exist.
    cache = ChatStateCache(
        backend,
        idle_ttl_min=settings.storage_cache_idle_ttl_min,
    )
    queue = WriteQueue(
        backend,
        cache,
        flush_interval_sec=settings.storage_flush_interval_sec,
        flush_size=settings.storage_flush_size,
    )
    # Wire the notification channel (the cache doesn't hold a reference to the
    # queue; the queue doesn't hold a reference to the cache's on_dirty).
    cache._on_dirty = queue.notify_dirty  # noqa: SLF001 — deliberate wire-up

    hydrator = Hydrator(cache, backend)

    cleanup = None
    if settings.storage_cleanup_enabled:
        from deep_research.storage.cleanup import CleanupLoop

        cleanup = CleanupLoop(
            backend,
            interval_sec=settings.storage_cleanup_interval_sec,
            chat_retention_days=settings.storage_chat_retention_days,
        )

    return StorageStack(
        backend=backend,
        cache=cache,
        queue=queue,
        hydrator=hydrator,
        cold_cache=cold_cache,
        cleanup=cleanup,
    )
