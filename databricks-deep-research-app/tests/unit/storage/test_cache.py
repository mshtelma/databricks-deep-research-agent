"""Unit tests for `deep_research.storage.cache.ChatStateCache` and `Hydrator`."""

from __future__ import annotations

import asyncio
from uuid import UUID, uuid4

import pytest
from tests.fakes.fake_backend import FakeBackend

from deep_research.storage import MigrationInProgressError
from deep_research.storage.cache import ChatStateCache, Hydrator
from deep_research.storage.documents import (
    ChatDocument,
    Finding,
    Message,
)


async def _hydrated_cache(
    backend: FakeBackend,
) -> tuple[ChatStateCache, Hydrator, list[tuple[UUID, str]]]:
    notifications: list[tuple[UUID, str]] = []
    cache = ChatStateCache(
        backend, on_dirty=lambda cid, scope: notifications.append((cid, scope))
    )
    hydrator = Hydrator(cache, backend)
    return cache, hydrator, notifications


# --- Hydration ------------------------------------------------------------


class TestHydration:
    async def test_get_on_fresh_chat_returns_new_document(self) -> None:
        backend = FakeBackend()
        cache, hydrator, _ = await _hydrated_cache(backend)
        cid = uuid4()
        hydrator.start(cid, user_id="u", title_hint="T")
        doc = await cache.get(cid, user_id="u")
        assert isinstance(doc, ChatDocument)
        assert doc.meta.chat_id == cid
        assert doc.meta.user_id == "u"
        assert doc.meta.version == 0  # unflushed

    async def test_get_without_user_id_on_unknown_chat_fails(self) -> None:
        backend = FakeBackend()
        cache, _, _ = await _hydrated_cache(backend)
        cid = uuid4()
        with pytest.raises(ValueError, match="no user_id"):
            await cache.get(cid)

    async def test_subsequent_get_reuses_future(self) -> None:
        backend = FakeBackend()
        cache, hydrator, _ = await _hydrated_cache(backend)
        cid = uuid4()
        hydrator.start(cid, user_id="u")
        a = await cache.get(cid, user_id="u")
        b = await cache.get(cid, user_id="u")
        assert a is b  # same in-memory doc


# --- Mutation + dirty notification ---------------------------------------


class TestMutationDirtyNotification:
    async def test_mutate_fires_both_dirty_notifications_by_default(self) -> None:
        backend = FakeBackend()
        cache, hydrator, notifications = await _hydrated_cache(backend)
        cid = uuid4()
        hydrator.start(cid, user_id="u")
        await cache.get(cid, user_id="u")
        await cache.mutate(
            cid, lambda d: d.state.add_message(Message(role="user", content="Hi"))
        )
        scopes = [s for _, s in notifications]
        assert "state" in scopes
        assert "meta" in scopes

    async def test_mutate_scoped_state_only(self) -> None:
        backend = FakeBackend()
        cache, hydrator, notifications = await _hydrated_cache(backend)
        cid = uuid4()
        hydrator.start(cid, user_id="u")
        await cache.get(cid, user_id="u")
        await cache.mutate(
            cid,
            lambda d: d.state.upsert_finding(Finding(content_hash="a", content="x")),
            dirty="state",
        )
        assert notifications == [(cid, "state")]

    async def test_mutate_raises_when_not_hydrated(self) -> None:
        backend = FakeBackend()
        cache, _, _ = await _hydrated_cache(backend)
        with pytest.raises(RuntimeError, match="not hydrated"):
            await cache.mutate(uuid4(), lambda d: None)

    async def test_mutate_updates_updated_at(self) -> None:
        backend = FakeBackend()
        cache, hydrator, _ = await _hydrated_cache(backend)
        cid = uuid4()
        hydrator.start(cid, user_id="u")
        doc = await cache.get(cid, user_id="u")
        before = doc.meta.updated_at
        await asyncio.sleep(0.01)
        await cache.mutate(
            cid, lambda d: d.state.add_message(Message(role="user", content="Hi"))
        )
        assert doc.meta.updated_at > before


# --- snapshot + mark_flushed ---------------------------------------------


class TestSnapshotAndFlushTracking:
    async def test_snapshot_returns_deep_copy(self) -> None:
        backend = FakeBackend()
        cache, hydrator, _ = await _hydrated_cache(backend)
        cid = uuid4()
        hydrator.start(cid, user_id="u")
        doc = await cache.get(cid, user_id="u")
        await cache.mutate(
            cid, lambda d: d.state.add_message(Message(role="user", content="a"))
        )
        snap = cache.snapshot(cid)
        assert snap is not None
        assert snap.state.messages[0].content == "a"
        # Mutate the snapshot; live doc must be unaffected.
        snap.state.messages.clear()
        assert len(doc.state.messages) == 1

    async def test_snapshot_returns_none_for_unhydrated(self) -> None:
        backend = FakeBackend()
        cache, _, _ = await _hydrated_cache(backend)
        assert cache.snapshot(uuid4()) is None

    async def test_mark_flushed_updates_version(self) -> None:
        backend = FakeBackend()
        cache, hydrator, _ = await _hydrated_cache(backend)
        cid = uuid4()
        hydrator.start(cid, user_id="u")
        doc = await cache.get(cid, user_id="u")
        cache.mark_flushed(cid, new_version=5)
        assert doc.meta.version == 5


# --- Migration mode -------------------------------------------------------


class TestMigrationMode:
    async def test_new_hydration_is_blocked(self) -> None:
        backend = FakeBackend()
        cache, _, _ = await _hydrated_cache(backend)
        cache._migration_timeout_sec = 0.1
        cache.set_migration_mode(True)
        with pytest.raises(MigrationInProgressError):
            await cache.get(uuid4(), user_id="u")

    async def test_already_hydrated_chat_is_still_readable(self) -> None:
        backend = FakeBackend()
        cache, hydrator, _ = await _hydrated_cache(backend)
        cid = uuid4()
        hydrator.start(cid, user_id="u")
        doc = await cache.get(cid, user_id="u")
        cache.set_migration_mode(True)
        # Still returns the hydrated doc without await'ing any gate.
        same = await cache.get(cid, user_id="u")
        assert same is doc

    async def test_migration_cleared_unblocks_in_progress_hydration(self) -> None:
        backend = FakeBackend()
        cache, _, _ = await _hydrated_cache(backend)
        cache._migration_timeout_sec = 2.0
        cache.set_migration_mode(True)
        cid = uuid4()

        async def _get() -> ChatDocument:
            return await cache.get(cid, user_id="u")

        task = asyncio.create_task(_get())
        await asyncio.sleep(0.05)  # let hydration task start and enter wait
        cache.set_migration_mode(False)
        doc = await task
        assert doc.meta.chat_id == cid


# --- Hydrator.prefetch ----------------------------------------------------


class TestHydratorPrefetch:
    async def test_prefetch_triggers_hydration_of_each_returned_meta(self) -> None:
        backend = FakeBackend()
        cache, hydrator, _ = await _hydrated_cache(backend)
        cid1, cid2 = uuid4(), uuid4()
        for cid in (cid1, cid2):
            await backend.write_chat(
                ChatDocument.new(cid, "alice", title="T"), expected_version=0
            )
        await hydrator.prefetch("alice", top_n=10)
        for cid in (cid1, cid2):
            doc = await cache.get(cid, user_id="alice")
            assert doc.meta.chat_id == cid

    async def test_prefetch_is_idempotent(self) -> None:
        backend = FakeBackend()
        cache, hydrator, _ = await _hydrated_cache(backend)
        cid = uuid4()
        await backend.write_chat(
            ChatDocument.new(cid, "alice", title="T"), expected_version=0
        )
        await hydrator.prefetch("alice", top_n=10)
        entries_after_first = set(cache._entries.keys())
        await hydrator.prefetch("alice", top_n=10)
        entries_after_second = set(cache._entries.keys())
        assert entries_after_first == entries_after_second


# --- Reaper ---------------------------------------------------------------


class TestReaper:
    async def test_reaper_evicts_stale_entries(self) -> None:
        backend = FakeBackend()
        cache = ChatStateCache(
            backend,
            idle_ttl_min=0,  # immediate eviction
            reaper_interval_sec=0.05,
        )
        cid = uuid4()
        cache.start_hydration(cid, user_id="u")
        await cache.get(cid, user_id="u")
        assert cid in cache._entries
        cache.start_reaper()
        # Give reaper multiple ticks to evict.
        for _ in range(10):
            await asyncio.sleep(0.06)
            if cid not in cache._entries:
                break
        await cache.stop_reaper()
        assert cid not in cache._entries

    async def test_stop_reaper_is_safe_when_not_started(self) -> None:
        backend = FakeBackend()
        cache, _, _ = await _hydrated_cache(backend)
        await cache.stop_reaper()  # must not raise


# --- Per-chat lock prevents interleaved mutation -------------------------


class TestPerChatLock:
    async def test_mutate_is_serialized_per_chat(self) -> None:
        backend = FakeBackend()
        cache, hydrator, _ = await _hydrated_cache(backend)
        cid = uuid4()
        hydrator.start(cid, user_id="u")
        await cache.get(cid, user_id="u")

        order: list[str] = []

        def _a(doc: ChatDocument) -> None:
            order.append("a-start")
            order.append("a-end")

        def _b(doc: ChatDocument) -> None:
            order.append("b-start")
            order.append("b-end")

        await asyncio.gather(cache.mutate(cid, _a), cache.mutate(cid, _b))
        # Either a fully completes then b, or vice versa — never interleaved.
        assert order in (
            ["a-start", "a-end", "b-start", "b-end"],
            ["b-start", "b-end", "a-start", "a-end"],
        )
