"""Regression: migration-vs-cache race (plan §R-7 — schema separation model).

The invariant: when `STORAGE_MIGRATION_MODE=1` is set, the running app's
`WriteQueue` pauses its flush loops (not its enqueue path) — so a concurrent
`migrate_lakebase.py` run into the new schema never interleaves with cache
flushes against the same rows. We prove this in-process using the FakeBackend
and the real `WriteQueue` / `ChatStateCache`: enqueue-during-pause accrues
writes, but no `write_chat` hits the backend until migration mode clears.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from deep_research.storage.cache import ChatStateCache, Hydrator
from deep_research.storage.documents import Message
from deep_research.storage.queue import WriteQueue
from tests.fakes.fake_backend import FakeBackend


async def test_enqueue_during_migration_pause_accrues_without_flushing() -> None:
    backend = FakeBackend()
    await backend.migrate()
    cache = ChatStateCache(backend)
    queue = WriteQueue(
        backend,
        cache,
        flush_interval_sec=0.02,
        flush_size=10,
        backoffs=(0.01, 0.02, 0.05),
    )
    cache._on_dirty = queue.notify_dirty
    hydrator = Hydrator(cache, backend)
    queue.start(event_tables=("research_events",))

    cid = uuid4()
    hydrator.start(cid, user_id="alice")
    await cache.get(cid, user_id="alice")

    # One mutation before migration mode — should land normally.
    await cache.mutate(
        cid, lambda d: d.state.add_message(Message(role="user", content="pre-mig"))
    )
    await asyncio.sleep(0.08)
    flushes_before_mig = queue.stats.chat_state_flushes
    assert flushes_before_mig >= 1

    # Engage migration mode. Enqueue path stays open; flush path pauses.
    queue.set_migration_mode(True)
    await asyncio.sleep(0.05)  # drain any in-flight tick

    for i in range(5):
        await cache.mutate(
            cid,
            lambda d, i=i: d.state.add_message(
                Message(role="user", content=f"mig-{i}")
            ),
        )
    # Give the queue plenty of ticks — but flushes are paused so the count
    # must not increase.
    await asyncio.sleep(0.25)
    assert queue.stats.chat_state_flushes == flushes_before_mig, (
        "WriteQueue flushed during migration mode — pause broken"
    )

    # Clear migration mode; the accumulated mutations flush.
    queue.set_migration_mode(False)
    await asyncio.sleep(0.1)
    assert queue.stats.chat_state_flushes > flushes_before_mig

    await queue.stop()
