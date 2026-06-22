"""Unit tests for cross-turn research consolidation on the cached path.

`CachedChatMemoryService.consolidate_from_pool` is the write half of the unified
chat-memory: at the end of every research/web turn it persists the turn's
verified claims (and quarantined observations) into `ChatState.memory.findings`
so the NEXT turn can read and cite them. Runs fully in-memory via the
StorageStack test double — no Postgres.
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


async def test_consolidate_persists_two_tier_findings() -> None:
    stack = await _stack()
    try:
        cid = await _new_chat(stack)
        rsid = uuid4()
        svc = CachedChatMemoryService(stack)
        await svc.hydrate(cid)

        n = await svc.consolidate_from_pool(
            cid,
            claims=[{"claim_text": "Revenue grew 12% in FY24.", "confidence": "high"}],
            observations=[{"text": "Unverified note about pricing."}],
            research_session_id=rsid,
            source_step=1,
        )
        assert n == 2

        # A FRESH service must see them after re-hydrating from persisted state
        # — i.e. they are durable, available to the next turn.
        svc2 = CachedChatMemoryService(stack)
        await svc2.hydrate(cid)
        snap = svc2.snapshot()
        contents = {f.content for f in snap.findings}
        confidences = {f.confidence for f in snap.findings}
        assert "Revenue grew 12% in FY24." in contents
        assert "high" in confidences and "low" in confidences  # two tiers
    finally:
        await stack.stop(timeout=1.0)


async def test_consolidate_is_idempotent_across_turns() -> None:
    stack = await _stack()
    try:
        cid = await _new_chat(stack)
        rsid = uuid4()
        svc = CachedChatMemoryService(stack)
        await svc.hydrate(cid)

        claim = [{"claim_text": "Same fact.", "confidence": "high"}]
        await svc.consolidate_from_pool(
            cid, claims=claim, observations=[], research_session_id=rsid, source_step=1
        )
        await svc.consolidate_from_pool(
            cid, claims=claim, observations=[], research_session_id=rsid, source_step=2
        )

        svc2 = CachedChatMemoryService(stack)
        await svc2.hydrate(cid)
        matches = [f for f in svc2.snapshot().findings if f.content == "Same fact."]
        assert len(matches) == 1  # content_hash dedup, not duplicated
    finally:
        await stack.stop(timeout=1.0)


async def test_consolidate_empty_is_noop() -> None:
    stack = await _stack()
    try:
        cid = await _new_chat(stack)
        svc = CachedChatMemoryService(stack)
        await svc.hydrate(cid)
        n = await svc.consolidate_from_pool(
            cid, claims=[], observations=[], research_session_id=uuid4(), source_step=1
        )
        assert n == 0
    finally:
        await stack.stop(timeout=1.0)
