"""Behavioural tests for CachedChatService.get_full double-trip.

Regression guard for the v3 plan's BUG 2 fix (architect B1 blocker): the
prior implementation called backend.load_chat DIRECTLY, which meant any
pending cache mutations that hadn't flushed yet were invisible. Swapping
to _read_chat alone would construct a ghost ChatDocument via
cache.get(user_id=X) on backend miss and silently return empty state
instead of None.

The fix: double-trip.
  1. backend.load_chat for existence + ownership (returns None for missing).
  2. _read_chat for cache-fresh state (picks up pending mutations).

See .omc/plans/post-engine-migration-regressions.md (v3 APPROVED).
"""

from __future__ import annotations

import os

os.environ["STORAGE_BACKEND"] = "fake"
os.environ["STORAGE_SERVICE_IMPL"] = "cached"
os.environ.setdefault("SECRET_KEY", "test-secret-key-unit-tests-only")
os.environ.setdefault("DATABRICKS_HOST", "https://test.azuredatabricks.net")
os.environ.setdefault("DATABRICKS_TOKEN", "test-token")
os.environ.setdefault("LLM_ENDPOINT_NAME", "test-endpoint")

from datetime import UTC, datetime  # noqa: E402
from uuid import uuid4  # noqa: E402

import pytest  # noqa: E402
from tests.fakes.fake_backend import FakeBackend  # noqa: E402

from deep_research.core.config import get_settings  # noqa: E402
from deep_research.services.cached.chat import CachedChatService  # noqa: E402
from deep_research.storage.cache import ChatStateCache, Hydrator  # noqa: E402
from deep_research.storage.cold_cache import ColdReadCache  # noqa: E402
from deep_research.storage.documents import (  # noqa: E402
    ChatDocument,
    ChatMeta,
    ChatMetaEmbed,
    ChatState,
    Message,
    ResearchSessionState,
)
from deep_research.storage.factory import StorageStack  # noqa: E402
from deep_research.storage.queue import WriteQueue  # noqa: E402

get_settings.cache_clear()


async def _build_stack() -> StorageStack:
    backend = FakeBackend()
    await backend.migrate()
    cold = ColdReadCache()
    cache = ChatStateCache(backend)
    queue = WriteQueue(
        backend,
        cache,
        flush_interval_sec=0.05,
        flush_size=20,
        backoffs=(0.01, 0.02, 0.05),
    )
    cache._on_dirty = queue.notify_dirty  # noqa: SLF001
    stack = StorageStack(
        backend=backend,
        cache=cache,
        queue=queue,
        hydrator=Hydrator(cache, backend),
        cold_cache=cold,
        cleanup=None,
    )
    await stack.start()
    return stack


def _build_doc(chat_id, user_id, title="seed", deleted=False) -> ChatDocument:
    now = datetime.now(UTC)
    meta = ChatMeta(
        chat_id=chat_id,
        user_id=user_id,
        title=title,
        preview="",
        created_at=now,
        updated_at=now,
        deleted_at=now if deleted else None,
        version=0,
    )
    state = ChatState(chat=ChatMetaEmbed(title=title))
    return ChatDocument(meta=meta, state=state)


@pytest.mark.asyncio
async def test_get_full_returns_none_when_chat_does_not_exist() -> None:
    """When backend has no chat, get_full must return None (not a ghost doc).

    This is the architect B1 blocker regression guard. Using _read_chat
    alone would construct a ghost ChatDocument via cache.get when
    backend.load_chat returned None and a user_id was passed — the fix
    checks existence on the backend FIRST.
    """
    stack = await _build_stack()
    try:
        svc = CachedChatService(stack)
        result = await svc.get_full(uuid4(), user_id="never-seen")
        assert result is None
    finally:
        await stack.stop()


@pytest.mark.asyncio
async def test_get_full_returns_none_for_wrong_user() -> None:
    """Ownership check on the backend doc."""
    stack = await _build_stack()
    try:
        chat_id = uuid4()
        doc = _build_doc(chat_id, user_id="owner")
        await stack.backend.write_chat(doc, expected_version=0)

        svc = CachedChatService(stack)
        result = await svc.get_full(chat_id, user_id="intruder")
        assert result is None
    finally:
        await stack.stop()


@pytest.mark.asyncio
async def test_get_full_returns_none_for_soft_deleted_chat() -> None:
    """Soft-deleted chats must not be returned."""
    stack = await _build_stack()
    try:
        chat_id = uuid4()
        doc = _build_doc(chat_id, user_id="u1", deleted=True)
        await stack.backend.write_chat(doc, expected_version=0)

        svc = CachedChatService(stack)
        result = await svc.get_full(chat_id, user_id="u1")
        assert result is None
    finally:
        await stack.stop()


@pytest.mark.asyncio
async def test_get_full_returns_populated_view_for_existing_chat() -> None:
    """Happy path: chat exists on backend; get_full returns a populated view."""
    stack = await _build_stack()
    try:
        chat_id = uuid4()
        doc = _build_doc(chat_id, user_id="u1", title="My Title")
        await stack.backend.write_chat(doc, expected_version=0)

        svc = CachedChatService(stack)
        view = await svc.get_full(chat_id, user_id="u1")
        assert view is not None
        assert view.title == "My Title"
        assert view.chat.id == chat_id
        # Empty state lists on fresh chat — but they exist as lists, not None.
        assert view.messages == []
        assert view.sources == []
        assert view.research_sessions == []
    finally:
        await stack.stop()


@pytest.mark.asyncio
async def test_get_full_uses_cache_for_fresh_state() -> None:
    """When the cache has been mutated but not yet flushed to the backend,
    get_full must return the CACHE-fresh state (not the stale backend).
    """
    stack = await _build_stack()
    try:
        chat_id = uuid4()
        user_id = "u1"
        doc = _build_doc(chat_id, user_id=user_id, title="OldBackendTitle")
        await stack.backend.write_chat(doc, expected_version=0)

        # Prime the cache and mutate without flushing: the cache will hold
        # "NewCacheTitle" while the backend still holds "OldBackendTitle".
        await stack.cache.get(chat_id, user_id=user_id)

        def _apply(d):
            d.meta.title = "NewCacheTitle"
            d.state.chat.title = "NewCacheTitle"

        # dirty='none' so the WriteQueue does NOT immediately flush — we
        # want to observe the cache-vs-backend divergence.
        await stack.cache.mutate(chat_id, _apply, dirty="none")

        svc = CachedChatService(stack)
        view = await svc.get_full(chat_id, user_id=user_id)
        assert view is not None
        # The fix: return cache-fresh state.
        assert view.title == "NewCacheTitle", (
            f"Expected cache-fresh title; got {view.title!r} "
            "(pre-fix, get_full returned the stale backend doc)"
        )
    finally:
        await stack.stop()


def _build_doc_with_research(
    chat_id,
    user_id,
    *,
    msg_id,
    sessions,
) -> ChatDocument:
    """Build a doc with one agent message and one or more research sessions.

    ``sessions`` is a list of ``ResearchSessionState`` so callers can exercise
    the regeneration / multi-session dedup path.
    """
    now = datetime.now(UTC)
    meta = ChatMeta(
        chat_id=chat_id,
        user_id=user_id,
        title="seed",
        preview="",
        created_at=now,
        updated_at=now,
        deleted_at=None,
        version=0,
    )
    state = ChatState(
        chat=ChatMetaEmbed(title="seed"),
        messages=[Message(id=msg_id, role="agent", content="Report body [1].", ts=now)],
        research_sessions=list(sessions),
    )
    return ChatDocument(meta=meta, state=state)


@pytest.mark.asyncio
async def test_get_full_links_research_session_to_message() -> None:
    """Regression: cached get_full must attach each message's research_session.

    Pre-fix, ``_state_msg_to_view`` returned ``research_session=None`` ("linked
    separately") and ``get_full`` never linked the separate
    ``research_sessions`` list back to the messages. ``chats.py`` reads claims
    EXCLUSIVELY from ``msg.research_session.verification_data``, so every
    message surfaced ``claims=[]`` and ALL citations rendered grey (and the
    frontend never even fired ``/messages/{id}/claims`` because
    ``latestAgentMessageIdForCitations`` requires ``!!m.researchSession``).
    """
    stack = await _build_stack()
    try:
        chat_id = uuid4()
        msg_id = uuid4()
        verification_data = {
            "claims": [
                {"claim_id": str(uuid4()), "text": "A grounded fact.", "citation_key": "1"}
            ],
            "summary": {"total_claims": 1, "supported": 1},
        }
        doc = _build_doc_with_research(
            chat_id,
            "u1",
            msg_id=msg_id,
            sessions=[
                ResearchSessionState(
                    id=uuid4(),
                    message_id=msg_id,
                    status="completed",
                    verification_data=verification_data,
                    completed_at=datetime.now(UTC),
                )
            ],
        )
        await stack.backend.write_chat(doc, expected_version=0)

        svc = CachedChatService(stack)
        view = await svc.get_full(chat_id, user_id="u1")
        assert view is not None

        linked_msgs = [m for m in view.messages if m.id == msg_id]
        assert len(linked_msgs) == 1
        linked = linked_msgs[0].research_session
        assert linked is not None, (
            "message.research_session is None — sessions were not linked to "
            "messages; chats.py emits claims=[] and citations render grey"
        )
        assert linked.verification_data == verification_data
    finally:
        await stack.stop()


@pytest.mark.asyncio
async def test_get_full_prefers_session_with_verification_data() -> None:
    """When a message has multiple sessions (e.g. regeneration), the linked
    session must be the one carrying verification_data so the report's claims
    resolve rather than an earlier empty/in-progress attempt."""
    stack = await _build_stack()
    try:
        chat_id = uuid4()
        msg_id = uuid4()
        earlier_empty = ResearchSessionState(
            id=uuid4(),
            message_id=msg_id,
            status="failed",
            verification_data={},
            started_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        later_with_claims = ResearchSessionState(
            id=uuid4(),
            message_id=msg_id,
            status="completed",
            verification_data={"claims": [{"claim_id": str(uuid4()), "citation_key": "1"}]},
            started_at=datetime(2026, 1, 2, tzinfo=UTC),
        )
        doc = _build_doc_with_research(
            chat_id,
            "u1",
            msg_id=msg_id,
            sessions=[earlier_empty, later_with_claims],
        )
        await stack.backend.write_chat(doc, expected_version=0)

        svc = CachedChatService(stack)
        view = await svc.get_full(chat_id, user_id="u1")
        assert view is not None
        linked = next(m for m in view.messages if m.id == msg_id).research_session
        assert linked is not None
        assert linked.verification_data.get("claims"), (
            "linked the empty session; must prefer the one with verification_data"
        )
    finally:
        await stack.stop()
