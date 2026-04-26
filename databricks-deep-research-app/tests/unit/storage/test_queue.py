"""Unit tests for `deep_research.storage.queue.WriteQueue`."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from deep_research.storage import TransientError
from deep_research.storage.cache import ChatStateCache, Hydrator
from deep_research.storage.documents import (
    ChatState,
    Finding,
    Message,
    PrepJobDocument,
    UserDocument,
)
from deep_research.storage.queue import WriteQueue
from tests.fakes.fake_backend import FakeBackend


async def _wire_up(
    *,
    flush_interval_sec: float = 0.05,
    flush_size: int = 10,
    event_tables: tuple[str, ...] = ("research_events",),
    backoffs: tuple[float, ...] = (0.01, 0.02, 0.05),
) -> tuple[FakeBackend, ChatStateCache, WriteQueue]:
    backend = FakeBackend()
    await backend.migrate()
    cache = ChatStateCache(backend)
    queue = WriteQueue(
        backend,
        cache,
        flush_interval_sec=flush_interval_sec,
        flush_size=flush_size,
        backoffs=backoffs,
    )
    cache._on_dirty = queue.notify_dirty  # plumb the notification channel
    queue.start(event_tables=event_tables)
    return backend, cache, queue


# --- Document flush: coalescing + version bump ---------------------------


class TestChatStateFlush:
    async def test_many_mutations_coalesce_to_one_flush(self) -> None:
        backend, cache, queue = await _wire_up()
        hydrator = Hydrator(cache, backend)
        cid = uuid4()
        hydrator.start(cid, user_id="u")
        await cache.get(cid, user_id="u")
        for i in range(20):
            await cache.mutate(
                cid,
                lambda d, i=i: d.state.add_message(
                    Message(role="user", content=f"m{i}")
                ),
            )
        await asyncio.sleep(0.2)
        await queue.stop()
        persisted = await backend.load_chat(cid)
        assert persisted is not None
        assert len(persisted.state.messages) == 20
        # With 20 back-to-back mutations at 0.05s flush interval, we expect
        # at most a small handful of flushes — the exact count is scheduler-
        # dependent but must be << 20.
        assert queue.stats.chat_state_flushes <= 5

    async def test_preview_materialized_at_flush(self) -> None:
        backend, cache, queue = await _wire_up()
        hydrator = Hydrator(cache, backend)
        cid = uuid4()
        hydrator.start(cid, user_id="u")
        await cache.get(cid, user_id="u")
        await cache.mutate(
            cid,
            lambda d: d.state.add_message(
                Message(role="user", content="Hello world")
            ),
        )
        await asyncio.sleep(0.15)
        await queue.stop()
        persisted = await backend.load_chat(cid)
        assert persisted is not None
        assert persisted.meta.preview == "Hello world"

    async def test_version_gate_recovers_from_conflict(self) -> None:
        backend, cache, queue = await _wire_up()
        hydrator = Hydrator(cache, backend)
        cid = uuid4()
        hydrator.start(cid, user_id="u")
        await cache.get(cid, user_id="u")
        # First mutation + flush to establish the chat in the backend.
        await cache.mutate(
            cid,
            lambda d: d.state.upsert_finding(Finding(content_hash="a", content="x")),
        )
        await asyncio.sleep(0.15)
        # Simulate a concurrent writer bumping the version out from under us.
        backend._chat_meta[cid].version = 999
        await cache.mutate(
            cid,
            lambda d: d.state.upsert_finding(Finding(content_hash="b", content="y")),
        )
        await asyncio.sleep(0.25)  # allow conflict + re-read + retry
        await queue.stop()
        assert queue.stats.conflicts >= 1
        final = await backend.load_chat(cid)
        assert final is not None
        # Final state has both findings; conflict was recovered, not dropped.
        content_hashes = {f.content_hash for f in final.state.memory.findings}
        assert "a" in content_hashes and "b" in content_hashes


# --- Migration-mode pause -----------------------------------------------


class TestMigrationMode:
    async def test_flush_is_paused_but_enqueue_is_free(self) -> None:
        backend, cache, queue = await _wire_up()
        hydrator = Hydrator(cache, backend)
        cid = uuid4()
        hydrator.start(cid, user_id="u")
        await cache.get(cid, user_id="u")

        queue.set_migration_mode(True)
        await asyncio.sleep(0.1)  # drain any pre-engaged tick

        # Enqueue during migration must not raise or block.
        await cache.mutate(
            cid, lambda d: d.state.add_message(Message(role="user", content="A"))
        )
        flushes_before = queue.stats.chat_state_flushes
        await asyncio.sleep(0.2)
        assert queue.stats.chat_state_flushes == flushes_before

        queue.set_migration_mode(False)
        await asyncio.sleep(0.2)
        assert queue.stats.chat_state_flushes > flushes_before
        await queue.stop()


# --- Append-only event flushes ------------------------------------------


class TestEventFlush:
    async def test_events_batch_into_one_insert(self) -> None:
        backend, _, queue = await _wire_up(flush_size=10)
        for i in range(25):
            queue.append_event("research_events", {"i": i})
        await asyncio.sleep(0.3)
        await queue.stop()
        rows = backend.raw_events("research_events")
        assert len(rows) == 25
        # 25 rows / flush_size=10 → at least 3 bulk inserts, at most a handful.
        assert queue.stats.events_flushes >= 3

    async def test_events_respect_ordering_on_failure(self) -> None:
        backend, _, queue = await _wire_up(flush_size=3)
        # Inject a fail hook that raises Transient twice, then succeeds.
        original_append = backend.append_events
        calls = [0]

        async def flaky(table, rows):  # noqa: ANN001 — test shim
            calls[0] += 1
            if calls[0] <= 2:
                raise TransientError("warehouse hiccup")
            await original_append(table, rows)

        backend.append_events = flaky  # type: ignore[assignment]

        for i in range(3):
            queue.append_event("research_events", {"seq": i})

        await asyncio.sleep(0.5)
        await queue.stop()
        rows = backend.raw_events("research_events")
        assert [r["seq"] for r in rows] == [0, 1, 2]


# --- User + PrepJob flush ------------------------------------------------


class TestUserAndPrepFlush:
    async def test_user_doc_flush(self) -> None:
        backend, _, queue = await _wire_up()
        queue.mark_user_doc_dirty(
            UserDocument(user_id="u", preferences={"theme": "dark"})
        )
        await asyncio.sleep(0.2)
        await queue.stop()
        loaded = await backend.load_user_doc("u")
        assert loaded is not None and loaded.preferences == {"theme": "dark"}

    async def test_prep_job_doc_flush(self) -> None:
        backend, _, queue = await _wire_up()
        jid = uuid4()
        queue.mark_prep_job_dirty(
            PrepJobDocument(prep_job_id=jid, account_id="acc", status="queued")
        )
        await asyncio.sleep(0.2)
        await queue.stop()
        loaded = await backend.load_prep_job(jid)
        assert loaded is not None and loaded.status == "queued"


# --- Shutdown drain -------------------------------------------------------


class TestShutdownDrain:
    async def test_stop_drains_pending_state(self) -> None:
        backend, cache, queue = await _wire_up(flush_interval_sec=5.0)
        hydrator = Hydrator(cache, backend)
        cid = uuid4()
        hydrator.start(cid, user_id="u")
        await cache.get(cid, user_id="u")
        # Enqueue mutations; DON'T wait for periodic flush — rely on shutdown.
        await cache.mutate(
            cid, lambda d: d.state.add_message(Message(role="user", content="late"))
        )
        # Enqueue some events too.
        for i in range(5):
            queue.append_event("research_events", {"i": i})
        await queue.stop()
        persisted = await backend.load_chat(cid)
        assert persisted is not None
        assert persisted.state.messages[0].content == "late"
        assert len(backend.raw_events("research_events")) == 5


# --- flush_chat_now -----------------------------------------------------


class TestFlushChatNow:
    async def test_forces_single_chat_to_land(self) -> None:
        backend, cache, queue = await _wire_up(flush_interval_sec=60.0)
        hydrator = Hydrator(cache, backend)
        cid = uuid4()
        hydrator.start(cid, user_id="u")
        await cache.get(cid, user_id="u")
        await cache.mutate(
            cid, lambda d: d.state.add_message(Message(role="user", content="now"))
        )
        # Without flush_chat_now, the 60s interval would delay this.
        await queue.flush_chat_now(cid)
        await queue.stop()
        persisted = await backend.load_chat(cid)
        assert persisted is not None
        assert persisted.state.messages[0].content == "now"
