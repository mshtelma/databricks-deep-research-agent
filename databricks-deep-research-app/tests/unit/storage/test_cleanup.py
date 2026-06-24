"""Unit tests for `deep_research.storage.cleanup.CleanupLoop`."""

from __future__ import annotations

import asyncio

from deep_research.storage.cleanup import CleanupLoop, CleanupStats


class _StubCleanupBackend:
    """Minimal stub — only implements what `CleanupLoop` touches.

    The real backends implement the `_CleanupCapable` Protocol via
    `cleanup_soft_deleted`. This stub records call args and yields canned
    stats or raises on demand.
    """

    def __init__(
        self,
        *,
        stats: CleanupStats | None = None,
        raise_on_call: BaseException | None = None,
    ) -> None:
        self._stats = stats or CleanupStats(file_chunks_deleted=42, chat_state_rows_deleted=2)
        self._raise = raise_on_call
        self.calls: list[int] = []

    async def cleanup_soft_deleted(self, *, chat_retention_days: int) -> CleanupStats:
        self.calls.append(chat_retention_days)
        if self._raise is not None:
            raise self._raise
        return self._stats


class _NoCleanupBackend:
    """Backend that does NOT implement cleanup — exercises the fallback."""

    async def cleanup_soft_deleted(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("should not be called")


class TestRunOnce:
    async def test_delegates_to_backend(self) -> None:
        backend = _StubCleanupBackend()
        loop = CleanupLoop(backend, chat_retention_days=7)  # type: ignore[arg-type]
        stats = await loop.run_once()
        assert stats.file_chunks_deleted == 42
        assert backend.calls == [7]

    async def test_noop_when_backend_lacks_cleanup(self) -> None:
        class NoExt:
            pass  # No cleanup_soft_deleted method.

        loop = CleanupLoop(NoExt(), chat_retention_days=7)  # type: ignore[arg-type]
        stats = await loop.run_once()
        assert stats.file_chunks_deleted == 0
        assert stats.errors == 0

    async def test_records_error_without_crashing(self) -> None:
        backend = _StubCleanupBackend(raise_on_call=RuntimeError("boom"))
        loop = CleanupLoop(backend, chat_retention_days=7)  # type: ignore[arg-type]
        stats = await loop.run_once()
        assert stats.errors == 1


class TestLifecycle:
    async def test_start_stop_is_idempotent(self) -> None:
        backend = _StubCleanupBackend()
        loop = CleanupLoop(backend, interval_sec=10.0)  # type: ignore[arg-type]
        loop.start()
        loop.start()  # no-op
        await loop.stop()
        await loop.stop()  # no-op

    async def test_loop_invokes_run_once_after_interval(self) -> None:
        backend = _StubCleanupBackend()
        loop = CleanupLoop(
            backend, interval_sec=0.02, chat_retention_days=3
        )  # type: ignore[arg-type]
        loop.start()
        # Initial delay is interval_sec, then run_once, then interval_sec...
        await asyncio.sleep(0.1)
        await loop.stop()
        # At least one call should have landed.
        assert len(backend.calls) >= 1
        assert all(c == 3 for c in backend.calls)
