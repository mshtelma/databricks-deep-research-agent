"""Phase 2e-2: consolidate_from_pool also writes coverage rows (topic × status)
with freshness stamps, so the Phase-3a routing gate can read what's covered.
In-memory via the StorageStack double — no Postgres.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from tests.fakes.fake_backend import FakeBackend

from deep_research.services.cached.chat_memory import CachedChatMemoryService
from deep_research.storage.cache import ChatStateCache, Hydrator
from deep_research.storage.cold_cache import ColdReadCache
from deep_research.storage.factory import StorageStack
from deep_research.storage.queue import WriteQueue

pytestmark = pytest.mark.unit


async def _stack() -> StorageStack:
    backend = FakeBackend()
    await backend.migrate()
    cache = ChatStateCache(backend)
    queue = WriteQueue(backend, cache, flush_interval_sec=3600.0, flush_size=10, backoffs=(0.01,))
    cache._on_dirty = queue.notify_dirty
    stack = StorageStack(
        backend=backend, cache=cache, queue=queue,
        hydrator=Hydrator(cache, backend), cold_cache=ColdReadCache(), cleanup=None,
    )
    await stack.start()
    return stack


async def _new_chat(stack: StorageStack):
    cid = uuid4()
    stack.hydrator.start(cid, user_id="u")
    await stack.cache.get(cid, user_id="u")
    return cid


async def test_consolidate_writes_coverage_rows() -> None:
    stack = await _stack()
    try:
        cid = await _new_chat(stack)
        svc = CachedChatMemoryService(stack)
        await svc.hydrate(cid)
        await svc.consolidate_from_pool(
            cid,
            claims=[],
            observations=[],
            research_session_id=uuid4(),
            source_step=1,
            coverage_topics=[{"topic": "Acme revenue", "status": "covered", "depth": "deep"}],
        )
        doc = await stack.cache.get(cid)
        match = [c for c in doc.state.memory.coverage if c.topic == "Acme revenue"]
        assert len(match) == 1
        assert match[0].status == "covered"
        assert match[0].depth == "deep"
        assert match[0].updated_at is not None  # freshness stamp written
    finally:
        await stack.stop(timeout=1.0)


async def test_coverage_topic_upserts_not_duplicated() -> None:
    stack = await _stack()
    try:
        cid = await _new_chat(stack)
        svc = CachedChatMemoryService(stack)
        await svc.hydrate(cid)
        for status in ("gap", "covered"):
            await svc.consolidate_from_pool(
                cid, claims=[], observations=[], research_session_id=uuid4(), source_step=1,
                coverage_topics=[{"topic": "Acme revenue", "status": status}],
            )
        doc = await stack.cache.get(cid)
        match = [c for c in doc.state.memory.coverage if c.topic == "Acme revenue"]
        assert len(match) == 1  # upsert by topic, not duplicated
        assert match[0].status == "covered"  # latest wins
    finally:
        await stack.stop(timeout=1.0)
