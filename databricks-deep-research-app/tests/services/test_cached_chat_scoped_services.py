"""Unit tests for the Wave-5a/5b chat-scoped cached services.

Covers `CachedMessageService`, `CachedSourceService`,
`CachedResearchSessionService`. These all mutate `ChatState` under the
per-chat lock via `cache.mutate`; the tests exercise the mutation paths
against a FakeBackend-backed `StorageStack` without any DB round-trip.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from deep_research.services._protocols import (
    IMessageService,
    IResearchSessionService,
    ISourceService,
)
from deep_research.services.cached.message import CachedMessageService
from deep_research.services.cached.research_session import (
    CachedResearchSessionService,
)
from deep_research.services.cached.source import CachedSourceService
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
        backend, cache, flush_interval_sec=0.05, flush_size=20,
        backoffs=(0.01, 0.02, 0.05),
    )
    cache._on_dirty = queue.notify_dirty
    stack = StorageStack(
        backend=backend, cache=cache, queue=queue,
        hydrator=Hydrator(cache, backend), cold_cache=cold, cleanup=None,
    )
    await stack.start()
    return stack


async def _hydrate_empty(stack: StorageStack, cid, user_id="u1") -> None:
    stack.hydrator.start(cid, user_id=user_id)
    await stack.cache.get(cid, user_id=user_id)


# --- CachedMessageService --------------------------------------------------


class TestCachedMessage:
    async def test_protocol(self) -> None:
        stack = await _stack()
        try:
            assert isinstance(CachedMessageService(stack), IMessageService)
        finally:
            await stack.stop(timeout=1.0)

    async def test_create_and_list(self) -> None:
        stack = await _stack()
        try:
            cid = uuid4()
            await _hydrate_empty(stack, cid)
            svc = CachedMessageService(stack)

            a = await svc.create(cid, "user", "hello")
            b = await svc.create(cid, "assistant", "hi there")

            msgs, total = await svc.list_messages(cid)
            assert total == 2
            assert [m.content for m in msgs] == ["hello", "hi there"]
            assert [m.role for m in msgs] == ["user", "assistant"]
            assert msgs[0].id == a.id
            assert msgs[1].id == b.id
        finally:
            await stack.stop(timeout=1.0)

    async def test_list_pagination_and_before(self) -> None:
        stack = await _stack()
        try:
            cid = uuid4()
            await _hydrate_empty(stack, cid)
            svc = CachedMessageService(stack)
            for i in range(5):
                await svc.create(cid, "user", f"msg{i}")
            # list with offset + limit
            page, total = await svc.list_messages(cid, limit=2, offset=1)
            assert total == 5
            assert [m.content for m in page] == ["msg1", "msg2"]

            # before cutoff
            cut = datetime.now(UTC) + timedelta(days=1)
            all_msgs, all_total = await svc.list_messages(cid, before=cut)
            assert all_total == 5
            tiny = datetime.now(UTC) - timedelta(days=1)
            none_msgs, none_total = await svc.list_messages(cid, before=tiny)
            assert none_total == 0
            assert none_msgs == []
        finally:
            await stack.stop(timeout=1.0)

    async def test_get_with_chat(self) -> None:
        stack = await _stack()
        try:
            cid = uuid4()
            await _hydrate_empty(stack, cid)
            svc = CachedMessageService(stack)
            created = await svc.create(cid, "user", "hello")

            got = await svc.get_with_chat(created.id, cid)
            assert got is not None
            assert got.content == "hello"
            assert got.chat_id == cid
            # Wrong chat
            assert await svc.get_with_chat(created.id, uuid4()) is None
            # Wrong msg
            assert await svc.get_with_chat(uuid4(), cid) is None
        finally:
            await stack.stop(timeout=1.0)

    async def test_update_content_and_is_edited(self) -> None:
        stack = await _stack()
        try:
            cid = uuid4()
            await _hydrate_empty(stack, cid)
            svc = CachedMessageService(stack)
            created = await svc.create(cid, "user", "hello")

            updated = await svc.update_content(created.id, "hi edited", chat_id=cid)
            assert updated is not None
            assert updated.content == "hi edited"
            assert updated.is_edited is True
            # Missing msg
            assert await svc.update_content(uuid4(), "x", chat_id=cid) is None
        finally:
            await stack.stop(timeout=1.0)

    async def test_delete_subsequent(self) -> None:
        stack = await _stack()
        try:
            cid = uuid4()
            await _hydrate_empty(stack, cid)
            svc = CachedMessageService(stack)
            m1 = await svc.create(cid, "user", "a")
            m2 = await svc.create(cid, "assistant", "b")
            # delete everything after m1.ts
            cutoff = m1.created_at
            removed = await svc.delete_subsequent(cid, cutoff)
            assert removed == 1
            msgs, total = await svc.list_messages(cid)
            assert total == 1
            assert msgs[0].id == m1.id
            # second call is a no-op
            assert await svc.delete_subsequent(cid, cutoff) == 0
            _ = m2  # silence F841
        finally:
            await stack.stop(timeout=1.0)

    async def test_set_research_session(self) -> None:
        stack = await _stack()
        try:
            cid = uuid4()
            await _hydrate_empty(stack, cid)
            svc = CachedMessageService(stack)
            m = await svc.create(cid, "user", "hello")
            rsid = uuid4()
            updated = await svc.set_research_session(m.id, rsid, chat_id=cid)
            assert updated is not None
            assert updated.research_session_id == str(rsid)
            # missing
            assert await svc.set_research_session(uuid4(), rsid, chat_id=cid) is None
        finally:
            await stack.stop(timeout=1.0)

    async def test_conversation_history_filters_empty(self) -> None:
        stack = await _stack()
        try:
            cid = uuid4()
            await _hydrate_empty(stack, cid)
            svc = CachedMessageService(stack)
            await svc.create(cid, "user", "hello")
            await svc.create(cid, "assistant", "")  # skipped
            await svc.create(cid, "user", "world")

            hist = await svc.get_conversation_history(cid)
            assert hist == [
                {"role": "user", "content": "hello"},
                {"role": "user", "content": "world"},
            ]
        finally:
            await stack.stop(timeout=1.0)


# --- CachedSourceService ---------------------------------------------------


class TestCachedSource:
    async def test_protocol(self) -> None:
        stack = await _stack()
        try:
            assert isinstance(CachedSourceService(stack), ISourceService)
        finally:
            await stack.stop(timeout=1.0)

    async def test_create_and_list_by_session(self) -> None:
        stack = await _stack()
        try:
            cid = uuid4()
            await _hydrate_empty(stack, cid)
            rsid = uuid4()
            svc = CachedSourceService(stack)

            a = await svc.create(
                rsid, "https://a.com", title="A",
                relevance_score=0.9, chat_id=cid,
            )
            b = await svc.create(
                rsid, "https://b.com", title="B",
                relevance_score=0.5, chat_id=cid,
            )

            rows = await svc.list_by_session(rsid, chat_id=cid)
            assert [r.url for r in rows] == ["https://a.com", "https://b.com"]
            assert rows[0].title == "A"
            assert rows[0].relevance_score == 0.9
            assert rows[1].relevance_score == 0.5
            _ = (a, b)
        finally:
            await stack.stop(timeout=1.0)

    async def test_create_many_and_dedup_by_url(self) -> None:
        stack = await _stack()
        try:
            cid = uuid4()
            await _hydrate_empty(stack, cid)
            rsid = uuid4()
            svc = CachedSourceService(stack)
            await svc.create_many(
                rsid,
                [
                    {"url": "https://x.com", "title": "X"},
                    {"url": "https://y.com", "title": "Y"},
                    {"url": "https://x.com", "title": "dup"},  # dedup → first wins
                ],
                chat_id=cid,
            )
            rows = await svc.list_by_session(rsid, chat_id=cid)
            assert len(rows) == 2
            urls = {r.url for r in rows}
            assert urls == {"https://x.com", "https://y.com"}
        finally:
            await stack.stop(timeout=1.0)

    async def test_get_by_url(self) -> None:
        stack = await _stack()
        try:
            cid = uuid4()
            await _hydrate_empty(stack, cid)
            rsid = uuid4()
            svc = CachedSourceService(stack)
            await svc.create(rsid, "https://z.com", chat_id=cid)
            hit = await svc.get_by_url(rsid, "https://z.com", chat_id=cid)
            assert hit is not None
            assert hit.url == "https://z.com"
            miss = await svc.get_by_url(rsid, "https://missing.com", chat_id=cid)
            assert miss is None
        finally:
            await stack.stop(timeout=1.0)

    async def test_update_content_truncates(self) -> None:
        stack = await _stack()
        try:
            cid = uuid4()
            await _hydrate_empty(stack, cid)
            rsid = uuid4()
            svc = CachedSourceService(stack)
            s = await svc.create(rsid, "https://a.com", chat_id=cid)
            long = "x" * 60_000
            upd = await svc.update_content(s.id, long, chat_id=cid)
            assert upd is not None
            assert upd.content is not None
            assert len(upd.content) == 50_000
            # missing
            assert await svc.update_content(uuid4(), "y", chat_id=cid) is None
        finally:
            await stack.stop(timeout=1.0)

    async def test_list_by_session_without_chat_id_returns_empty(self) -> None:
        stack = await _stack()
        try:
            svc = CachedSourceService(stack)
            rows = await svc.list_by_session(uuid4())
            assert rows == []
        finally:
            await stack.stop(timeout=1.0)


# --- CachedResearchSessionService ------------------------------------------


class TestCachedResearchSession:
    async def test_protocol(self) -> None:
        stack = await _stack()
        try:
            assert isinstance(
                CachedResearchSessionService(stack), IResearchSessionService
            )
        finally:
            await stack.stop(timeout=1.0)

    async def test_create_and_get(self) -> None:
        stack = await _stack()
        try:
            cid = uuid4()
            await _hydrate_empty(stack, cid)
            svc = CachedResearchSessionService(stack)
            mid = uuid4()

            s = await svc.create(
                mid, "What is the weather?",
                research_depth="deep",
                query_classification={"intent": "factual"},
                chat_id=cid,
            )
            assert s.query == "What is the weather?"
            assert s.research_depth == "deep"
            assert s.status == "in_progress"

            got = await svc.get(s.id, chat_id=cid)
            assert got is not None
            assert got.id == s.id
            assert got.message_id == mid

            # By message
            by_msg = await svc.get_by_message(mid, chat_id=cid)
            assert by_msg is not None
            assert by_msg.id == s.id
        finally:
            await stack.stop(timeout=1.0)

    async def test_get_active_by_chat_respects_user_id(self) -> None:
        stack = await _stack()
        try:
            cid = uuid4()
            await _hydrate_empty(stack, cid, user_id="alice")
            svc = CachedResearchSessionService(stack)
            await svc.create(uuid4(), "q", chat_id=cid)
            # Right user
            hit = await svc.get_active_session_by_chat(cid, "alice")
            assert hit is not None
            assert hit.status == "in_progress"
            # Wrong user
            assert await svc.get_active_session_by_chat(cid, "mallory") is None
        finally:
            await stack.stop(timeout=1.0)

    async def test_get_active_none_when_all_completed(self) -> None:
        stack = await _stack()
        try:
            cid = uuid4()
            await _hydrate_empty(stack, cid, user_id="alice")
            svc = CachedResearchSessionService(stack)
            s = await svc.create(uuid4(), "q", chat_id=cid)
            await svc.complete(s.id, chat_id=cid)
            assert await svc.get_active_session_by_chat(cid, "alice") is None
        finally:
            await stack.stop(timeout=1.0)

    async def test_plan_observations_reasoning(self) -> None:
        stack = await _stack()
        try:
            cid = uuid4()
            await _hydrate_empty(stack, cid)
            svc = CachedResearchSessionService(stack)
            s = await svc.create(uuid4(), "q", chat_id=cid)

            upd = await svc.update_plan(s.id, {"steps": ["a", "b"]}, chat_id=cid)
            assert upd is not None
            assert upd.plan == {"steps": ["a", "b"]}

            obs = await svc.add_observation(s.id, "found X", 0, chat_id=cid)
            assert obs is not None
            assert obs.observations["items"][0]["observation"] == "found X"

            rs = await svc.add_reasoning_step(
                s.id, "search", "query the web", {"tool": "brave"}, chat_id=cid,
            )
            assert rs is not None
            assert rs.reasoning_steps[0]["type"] == "search"
            assert rs.reasoning_steps[0]["metadata"] == {"tool": "brave"}
        finally:
            await stack.stop(timeout=1.0)

    async def test_complete_success(self) -> None:
        stack = await _stack()
        try:
            cid = uuid4()
            await _hydrate_empty(stack, cid)
            svc = CachedResearchSessionService(stack)
            s = await svc.create(uuid4(), "q", chat_id=cid)
            done = await svc.complete(s.id, chat_id=cid)
            assert done is not None
            assert done.status == "completed"
            assert done.completed_at is not None
            assert done.error_message is None
        finally:
            await stack.stop(timeout=1.0)

    async def test_complete_failure(self) -> None:
        stack = await _stack()
        try:
            cid = uuid4()
            await _hydrate_empty(stack, cid)
            svc = CachedResearchSessionService(stack)
            s = await svc.create(uuid4(), "q", chat_id=cid)
            done = await svc.complete(s.id, error_message="boom", chat_id=cid)
            assert done is not None
            assert done.status == "failed"
            assert done.error_message == "boom"
        finally:
            await stack.stop(timeout=1.0)

    async def test_cancel_in_progress(self) -> None:
        stack = await _stack()
        try:
            cid = uuid4()
            await _hydrate_empty(stack, cid)
            svc = CachedResearchSessionService(stack)
            s = await svc.create(uuid4(), "q", chat_id=cid)
            c = await svc.cancel(s.id, chat_id=cid)
            assert c is not None
            assert c.status == "cancelled"
        finally:
            await stack.stop(timeout=1.0)

    async def test_cancel_noop_when_completed(self) -> None:
        stack = await _stack()
        try:
            cid = uuid4()
            await _hydrate_empty(stack, cid)
            svc = CachedResearchSessionService(stack)
            s = await svc.create(uuid4(), "q", chat_id=cid)
            await svc.complete(s.id, chat_id=cid)
            c = await svc.cancel(s.id, chat_id=cid)
            assert c is not None
            # status unchanged
            assert c.status == "completed"
        finally:
            await stack.stop(timeout=1.0)

    async def test_update_classification(self) -> None:
        stack = await _stack()
        try:
            cid = uuid4()
            await _hydrate_empty(stack, cid)
            svc = CachedResearchSessionService(stack)
            s = await svc.create(uuid4(), "q", chat_id=cid)
            upd = await svc.update_classification(
                s.id, {"intent": "compare"}, chat_id=cid,
            )
            assert upd is not None
            assert upd.query_classification == {"intent": "compare"}
        finally:
            await stack.stop(timeout=1.0)


# --- Factory smoke ----------------------------------------------------------


class TestFactoryDispatch:
    async def test_dispatches_cached_for_all_three(self) -> None:
        from types import SimpleNamespace

        from deep_research.services._impl_factory import (
            make_message_service,
            make_research_session_service,
            make_source_service,
        )

        stack = await _stack()
        try:
            settings = SimpleNamespace(storage_service_impl="cached")
            assert isinstance(
                make_message_service(settings, stack), CachedMessageService
            )
            assert isinstance(
                make_source_service(settings, stack), CachedSourceService
            )
            assert isinstance(
                make_research_session_service(settings, stack),
                CachedResearchSessionService,
            )
        finally:
            await stack.stop(timeout=1.0)

    def test_refuses_cached_without_stack(self) -> None:
        from types import SimpleNamespace

        from deep_research.services._impl_factory import (
            make_message_service,
            make_research_session_service,
            make_source_service,
        )

        settings = SimpleNamespace(storage_service_impl="cached")
        for factory in (
            make_message_service,
            make_source_service,
            make_research_session_service,
        ):
            with pytest.raises(ValueError, match="StorageStack"):
                factory(settings, None)
