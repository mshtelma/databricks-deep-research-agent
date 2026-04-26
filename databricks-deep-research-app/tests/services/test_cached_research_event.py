"""Unit tests for `CachedResearchEventService`.

Exercises the cache-backed `IResearchEventService` against a FakeBackend-
powered `StorageStack`. Full legacy/cached parity requires a real SQLAlchemy
session and is deferred to the Wave-7 integration tests.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest

from deep_research.services._protocols import IResearchEventService
from deep_research.services.cached.research_event import (
    CachedResearchEventService,
)
from deep_research.storage.backend import StorageBackend
from deep_research.storage.cache import ChatStateCache, Hydrator
from deep_research.storage.cold_cache import ColdReadCache
from deep_research.storage.factory import StorageStack
from deep_research.storage.queue import WriteQueue
from tests.fakes.fake_backend import FakeBackend


async def _stack() -> StorageStack:
    backend = FakeBackend()
    await backend.migrate()
    cold = ColdReadCache()
    cache = ChatStateCache(backend)
    queue = WriteQueue(
        backend,
        cache,
        flush_interval_sec=0.05,
        flush_size=50,
        backoffs=(0.01, 0.02, 0.05),
    )
    cache._on_dirty = queue.notify_dirty
    hydrator = Hydrator(cache, backend)
    stack = StorageStack(
        backend=backend,
        cache=cache,
        queue=queue,
        hydrator=hydrator,
        cold_cache=cold,
        cleanup=None,
    )
    await stack.start()
    return stack


class TestProtocolConformance:
    async def test_satisfies_protocol(self) -> None:
        stack = await _stack()
        try:
            svc = CachedResearchEventService(stack)
            assert isinstance(svc, IResearchEventService)
        finally:
            await stack.stop(timeout=1.0)


class TestWriteBehavior:
    async def test_save_event_returns_namespace_with_fields(self) -> None:
        stack = await _stack()
        try:
            svc = CachedResearchEventService(stack)
            session_id = uuid4()
            event = await svc.save_event(
                session_id,
                event_type="tool_call",
                payload={"tool": "search", "args": {"q": "acme"}},
            )
            assert isinstance(event, SimpleNamespace)
            assert event.event_type == "tool_call"
            assert event.sequence_number == 1
            assert event.research_session_id == session_id
            assert event.payload == {"tool": "search", "args": {"q": "acme"}}
        finally:
            await stack.stop(timeout=1.0)

    async def test_sequence_increments_per_session(self) -> None:
        stack = await _stack()
        try:
            svc = CachedResearchEventService(stack)
            session_id = uuid4()
            e1 = await svc.save_event(session_id, "t", {})
            e2 = await svc.save_event(session_id, "t", {})
            e3 = await svc.save_event(uuid4(), "t", {})  # new session — counter starts over
            assert (e1.sequence_number, e2.sequence_number) == (1, 2)
            assert e3.sequence_number == 1
        finally:
            await stack.stop(timeout=1.0)

    async def test_save_events_batch_returns_count(self) -> None:
        stack = await _stack()
        try:
            svc = CachedResearchEventService(stack)
            session_id = uuid4()
            batch = [
                {"event_type": "a", "payload": {"i": i}} for i in range(50)
            ]
            count = await svc.save_events_batch(session_id, batch)
            assert count == 50
        finally:
            await stack.stop(timeout=1.0)

    async def test_writes_flush_to_backend_via_queue(self) -> None:
        stack = await _stack()
        try:
            svc = CachedResearchEventService(stack)
            session_id = uuid4()
            for i in range(5):
                await svc.save_event(session_id, "x", {"i": i})
            # Let the queue tick at least once.
            await asyncio.sleep(0.2)
            rows = await stack.backend.list_rows(
                "research_events", {"session_id": session_id}
            )
            assert len(rows) == 5
        finally:
            await stack.stop(timeout=1.0)


class TestReadBehavior:
    async def test_get_events_for_session_returns_ordered(self) -> None:
        stack = await _stack()
        try:
            svc = CachedResearchEventService(stack)
            sid = uuid4()
            for i in range(3):
                await svc.save_event(sid, "t", {"i": i})
            await asyncio.sleep(0.2)
            rows = await svc.get_events_for_session(sid)
            assert [r["sequence_number"] for r in rows] == [1, 2, 3]
        finally:
            await stack.stop(timeout=1.0)

    async def test_get_event_count(self) -> None:
        stack = await _stack()
        try:
            svc = CachedResearchEventService(stack)
            sid = uuid4()
            for i in range(7):
                await svc.save_event(sid, "t", {"i": i})
            await asyncio.sleep(0.2)
            assert await svc.get_event_count(sid) == 7
        finally:
            await stack.stop(timeout=1.0)

    async def test_get_events_since_sequence_filters(self) -> None:
        stack = await _stack()
        try:
            svc = CachedResearchEventService(stack)
            sid = uuid4()
            for i in range(10):
                await svc.save_event(sid, "t", {"i": i})
            await asyncio.sleep(0.2)
            rows = await svc.get_events_since_sequence(sid, since_sequence=4, limit=3)
            seqs = [r["sequence_number"] for r in rows]
            assert seqs == [5, 6, 7]
        finally:
            await stack.stop(timeout=1.0)


class TestSerializationHelpers:
    def test_event_to_dict_from_namespace(self) -> None:
        ns = SimpleNamespace(
            id=uuid4(),
            research_session_id=uuid4(),
            event_type="t",
            payload={"x": 1},
            timestamp=datetime.now(UTC),
            sequence_number=5,
        )
        d = CachedResearchEventService.event_to_dict(ns)
        # camelCase contract — matches legacy `ResearchEventService.event_to_dict`,
        # the SSE payload docstring at `api/v1/jobs.py`, and frontend TS types.
        assert d["eventType"] == "t"
        assert d["payload"] == {"x": 1}
        assert d["sequenceNumber"] == 5
        # Regression: raw UUID / datetime previously raised
        # "TypeError: Object of type UUID is not JSON serializable" when the
        # SSE generator called `json.dumps(event_dict)`.
        assert isinstance(d["id"], str)
        assert isinstance(d["timestamp"], str)
        json.dumps(d)

    def test_events_to_list_passes_through_dicts(self) -> None:
        events = [{"event_type": "a"}, {"event_type": "b"}]
        out = CachedResearchEventService.events_to_list(events)
        assert out == events
