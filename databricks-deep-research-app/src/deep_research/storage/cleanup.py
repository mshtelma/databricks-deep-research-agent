"""Hourly cleanup job for soft-deleted chats and orphaned file chunks.

Implemented as a backend-agnostic `CleanupLoop` that runs raw SQL via
`_execute_cleanup_sql` — a small helper the two real backends override to
match their dialect. The `FakeBackend` no-ops because tests don't need to
exercise the DELETE paths (they cover projection correctness directly).

Invariants:

* Never touches chats whose `deleted_at IS NULL`.
* Uses the `chat_deleted_files` projection — never parses JSON.
* Runs on a bounded schedule (`interval_sec`); failure does not crash the
  loop. `storage_cleanup_errors_total` is incremented on failure.
* Cooperative asyncio; obeys cancellation.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from deep_research.storage.backend import StorageBackend
from deep_research.storage.observability import get_sink

logger = logging.getLogger(__name__)


@dataclass
class CleanupStats:
    """Aggregate row counts for one cleanup pass."""

    file_chunks_deleted: int = 0
    chat_state_rows_deleted: int = 0
    chat_deleted_files_rows_deleted: int = 0
    chat_meta_rows_deleted: int = 0
    errors: int = 0


@runtime_checkable
class _CleanupCapable(Protocol):
    """Optional backend extension — real backends implement the raw SQL
    dialect needed for cleanup. Backends that don't (e.g. `FakeBackend`)
    fall back to a no-op via `hasattr` check.
    """

    async def cleanup_soft_deleted(
        self, *, chat_retention_days: int
    ) -> CleanupStats: ...


class CleanupLoop:
    """Hourly background task.

    Lifecycle: `start()` spawns the task; `stop()` cancels and awaits it.
    """

    def __init__(
        self,
        backend: StorageBackend,
        *,
        interval_sec: float = 3600.0,
        chat_retention_days: int = 7,
    ) -> None:
        self._backend = backend
        self._interval_sec = interval_sec
        self._chat_retention_days = chat_retention_days
        self._task: asyncio.Task[None] | None = None
        self._stopping = False

    def start(self) -> None:
        if self._task is not None:
            return
        self._stopping = False
        self._task = asyncio.create_task(self._loop(), name="storage-cleanup")

    async def stop(self) -> None:
        self._stopping = True
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def run_once(self) -> CleanupStats:
        """Synchronously execute one cleanup pass. Used by tests and by the
        periodic loop.
        """
        sink = get_sink()
        if isinstance(self._backend, _CleanupCapable):
            try:
                stats = await self._backend.cleanup_soft_deleted(
                    chat_retention_days=self._chat_retention_days
                )
            except Exception as exc:  # noqa: BLE001 — cleanup must never crash
                logger.exception("cleanup failed: %s", exc)
                stats = CleanupStats(errors=1)
                sink.counter("storage_cleanup_errors_total", 1, backend=_backend_name(self._backend))
        else:
            # Backend doesn't implement the cleanup extension (e.g. FakeBackend).
            stats = CleanupStats()

        sink.counter(
            "storage_cleanup_file_chunks_deleted_total",
            stats.file_chunks_deleted,
            backend=_backend_name(self._backend),
        )
        sink.counter(
            "storage_cleanup_chat_state_deleted_total",
            stats.chat_state_rows_deleted,
            backend=_backend_name(self._backend),
        )
        return stats

    async def _loop(self) -> None:
        # Initial delay so startup doesn't immediately hammer the DB.
        try:
            await asyncio.sleep(self._interval_sec)
        except asyncio.CancelledError:
            raise
        while not self._stopping:
            try:
                await self.run_once()
                await asyncio.sleep(self._interval_sec)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — loop must never exit
                logger.exception("cleanup loop tick failed")
                # Back off briefly to avoid hot-looping on persistent errors.
                try:
                    await asyncio.sleep(min(60.0, self._interval_sec))
                except asyncio.CancelledError:
                    raise


def _backend_name(backend: StorageBackend) -> str:
    return type(backend).__name__.replace("Backend", "").lower() or "unknown"
