"""End-to-end (in-memory) test for the post-run consolidation helper.

`consolidate_turn_knowledge` is what the orchestrator calls after a research
turn: it applies the trust policy to the persisted ``verification_data`` and
writes only the verified claims into durable chat memory. This exercises the
real production (cached) path via the StorageStack double — no Postgres.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from tests.fakes.fake_backend import FakeBackend

from deep_research.agent.turn_consolidation import consolidate_turn_knowledge
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
    queue = WriteQueue(
        backend, cache, flush_interval_sec=3600.0, flush_size=10, backoffs=(0.01,)
    )
    cache._on_dirty = queue.notify_dirty
    stack = StorageStack(
        backend=backend,
        cache=cache,
        queue=queue,
        hydrator=Hydrator(cache, backend),
        cold_cache=ColdReadCache(),
        cleanup=None,
    )
    await stack.start()
    return stack


async def _new_chat(stack: StorageStack):
    cid = uuid4()
    stack.hydrator.start(cid, user_id="u")
    await stack.cache.get(cid, user_id="u")
    return cid


async def test_consolidate_turn_persists_only_verified_claims() -> None:
    stack = await _stack()
    try:
        cid = await _new_chat(stack)
        rsid = uuid4()
        svc = CachedChatMemoryService(stack)
        await svc.hydrate(cid)

        verification_data = {
            "claims": [
                {"claim_text": "Verified fact.", "verification_verdict": "supported", "abstained": False},
                {"claim_text": "Refuted fact.", "verification_verdict": "contradicted", "abstained": False},
                {"claim_text": "Abstained fact.", "verification_verdict": "supported", "abstained": True},
            ],
            "summary": {"total_claims": 3},
        }

        n = await consolidate_turn_knowledge(svc, cid, rsid, verification_data)
        assert n == 1  # only the supported, non-abstained claim

        svc2 = CachedChatMemoryService(stack)
        await svc2.hydrate(cid)
        contents = {f.content for f in svc2.snapshot().findings}
        assert "Verified fact." in contents
        assert "Refuted fact." not in contents
        assert "Abstained fact." not in contents
    finally:
        await stack.stop(timeout=1.0)


async def test_consolidate_turn_empty_verification_is_noop() -> None:
    stack = await _stack()
    try:
        cid = await _new_chat(stack)
        svc = CachedChatMemoryService(stack)
        await svc.hydrate(cid)
        assert await consolidate_turn_knowledge(svc, cid, uuid4(), None) == 0
        assert await consolidate_turn_knowledge(svc, cid, uuid4(), {"claims": []}) == 0
    finally:
        await stack.stop(timeout=1.0)


async def test_consolidate_turn_writes_coverage_even_without_claims() -> None:
    stack = await _stack()
    try:
        cid = await _new_chat(stack)
        svc = CachedChatMemoryService(stack)
        await svc.hydrate(cid)
        # No verified claims, but a coverage topic → coverage still written
        # (a "covered but no new verified claims" turn must still record coverage).
        n = await consolidate_turn_knowledge(
            svc, cid, uuid4(), {"claims": []},
            coverage_topics=[{"topic": "Acme revenue", "status": "covered"}],
        )
        assert n == 0  # no findings
        doc = await stack.cache.get(cid)
        assert any(c.topic == "Acme revenue" for c in doc.state.memory.coverage)
    finally:
        await stack.stop(timeout=1.0)
