"""Unit tests for `deep_research.storage.hydration_progress.HydrationProgress`."""

from __future__ import annotations

from uuid import uuid4

import pytest
from tests.fakes.fake_backend import FakeBackend

from deep_research.storage.cache import ChatStateCache
from deep_research.storage.hydration_progress import (
    HydrationProgress,
    HydrationTimeoutError,
)
from deep_research.storage.observability import RecordingSink, use_sink

# Compressed deadlines so tests are fast. Semantics identical to production:
# deadline_1 emits 'context_loading', deadline_2 emits 'context_slow',
# deadline_3 is the hard cap.
_TEST_DEADLINES = (
    (0.05, "context_loading"),
    (0.15, "context_slow"),
    (0.3, None),
)


async def _build_cache(latency_ms: float) -> tuple[FakeBackend, ChatStateCache]:
    backend = FakeBackend(latency_ms=latency_ms)
    await backend.migrate()
    return backend, ChatStateCache(backend)


class TestWarmPath:
    async def test_warm_outcome_no_sse(self) -> None:
        backend, cache = await _build_cache(latency_ms=1)
        events: list[str] = []
        hp = HydrationProgress(
            cache=cache,
            sse_emit=lambda e, d: events.append(e),
            backend_label="fake",
            deadlines=_TEST_DEADLINES,
        )
        sink = RecordingSink()
        with use_sink(sink):
            doc = await hp.hydrate(uuid4(), user_id="u")
        assert doc is not None
        assert events == []
        assert sink.count("storage_first_turn_outcome_total", outcome="warm", backend="fake") == 1


class TestSlowPath:
    async def test_slow_outcome_one_event(self) -> None:
        backend, cache = await _build_cache(latency_ms=80)  # past 0.05, before 0.15
        events: list[str] = []
        hp = HydrationProgress(
            cache=cache,
            sse_emit=lambda e, d: events.append(e),
            backend_label="fake",
            deadlines=_TEST_DEADLINES,
        )
        sink = RecordingSink()
        with use_sink(sink):
            doc = await hp.hydrate(uuid4(), user_id="u")
        assert doc is not None
        assert events == ["context_loading"]
        assert sink.count("storage_first_turn_outcome_total", outcome="slow", backend="fake") == 1


class TestColdPath:
    async def test_cold_outcome_two_events(self) -> None:
        backend, cache = await _build_cache(latency_ms=200)  # past 0.15, before 0.3
        events: list[str] = []
        hp = HydrationProgress(
            cache=cache,
            sse_emit=lambda e, d: events.append(e),
            backend_label="fake",
            deadlines=_TEST_DEADLINES,
        )
        sink = RecordingSink()
        with use_sink(sink):
            doc = await hp.hydrate(uuid4(), user_id="u")
        assert doc is not None
        assert events == ["context_loading", "context_slow"]
        assert sink.count("storage_first_turn_outcome_total", outcome="cold", backend="fake") == 1


class TestCap:
    async def test_cap_raises_and_labels_capped(self) -> None:
        backend, cache = await _build_cache(latency_ms=500)  # past 0.3
        events: list[str] = []
        hp = HydrationProgress(
            cache=cache,
            sse_emit=lambda e, d: events.append(e),
            backend_label="fake",
            deadlines=_TEST_DEADLINES,
        )
        sink = RecordingSink()
        with use_sink(sink), pytest.raises(HydrationTimeoutError) as excinfo:
            await hp.hydrate(uuid4(), user_id="u")
        # Both intermediate events fired before the cap hit.
        assert events == ["context_loading", "context_slow"]
        assert sink.count("storage_first_turn_outcome_total", outcome="capped", backend="fake") == 1
        assert excinfo.value.retry_after == 2


class TestSseEmitFailureIsolation:
    async def test_sse_callback_exception_does_not_abort_hydrate(self) -> None:
        backend, cache = await _build_cache(latency_ms=80)

        def _raises(event: str, data) -> None:
            raise RuntimeError("bad SSE pipe")

        hp = HydrationProgress(
            cache=cache,
            sse_emit=_raises,
            backend_label="fake",
            deadlines=_TEST_DEADLINES,
        )
        # Should complete despite the SSE callback raising.
        doc = await hp.hydrate(uuid4(), user_id="u")
        assert doc is not None


class TestNoSseEmitter:
    async def test_works_without_sse(self) -> None:
        backend, cache = await _build_cache(latency_ms=80)
        hp = HydrationProgress(
            cache=cache,
            sse_emit=None,
            backend_label="fake",
            deadlines=_TEST_DEADLINES,
        )
        doc = await hp.hydrate(uuid4(), user_id="u")
        assert doc is not None
