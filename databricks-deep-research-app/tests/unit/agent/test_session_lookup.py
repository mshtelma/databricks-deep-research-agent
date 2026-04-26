"""Tests for the storage-aware session lookup helper.

The helper bridges two storage modes:
- ``cached`` (default) — session state lives in the ChatDocument JSONB
  blob inside ``StorageStack``; callers supply ``chat_id`` and
  ``session_id`` from URL path params, so no index table is needed.
- ``sqlalchemy_legacy`` — session state is a row in ``research_sessions``.

Both paths must return a ``SessionControlView`` with equivalent fields
(or ``None`` when the session is not found), so callers downstream of
``load_session_control_view`` (SSE stream validation, cancel verification,
_run_job post-stream lifecycle) don't have to know the mode.
"""

from __future__ import annotations

# --- Must configure env before importing deep_research.* ---
import os

os.environ["STORAGE_BACKEND"] = "fake"
os.environ["STORAGE_SERVICE_IMPL"] = "cached"
os.environ.setdefault("SECRET_KEY", "test-secret-key-unit-tests-only")
os.environ.setdefault("DATABRICKS_HOST", "https://test.azuredatabricks.net")
os.environ.setdefault("DATABRICKS_TOKEN", "test-token")
os.environ.setdefault("LLM_ENDPOINT_NAME", "test-endpoint")

from datetime import UTC, datetime, timedelta  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest.mock import AsyncMock  # noqa: E402
from uuid import uuid4  # noqa: E402

import pytest  # noqa: E402
from tests.fakes.fake_backend import FakeBackend  # noqa: E402

from deep_research.agent.persistence import (  # noqa: E402
    persist_research_session_start_independent,
)
from deep_research.agent.session_lookup import (  # noqa: E402
    SessionControlView,
    load_session_control_view,
)
from deep_research.core.config import Settings, get_settings  # noqa: E402
from deep_research.models.research_session import ResearchStatus  # noqa: E402
from deep_research.storage.cache import ChatStateCache, Hydrator  # noqa: E402
from deep_research.storage.cold_cache import ColdReadCache  # noqa: E402
from deep_research.storage.factory import StorageStack  # noqa: E402
from deep_research.storage.queue import WriteQueue  # noqa: E402

get_settings.cache_clear()


async def _build_stack() -> StorageStack:
    """Minimal StorageStack over FakeBackend — mirrors the cached-mode tests."""
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


def _cached_settings() -> Settings:
    s = get_settings()
    assert s.storage_service_impl == "cached"
    return s


def _legacy_settings() -> Settings:
    return SimpleNamespace(storage_service_impl="sqlalchemy_legacy")  # type: ignore[return-value]


@pytest.mark.asyncio
async def test_cached_returns_view_when_session_in_document() -> None:
    """Chat exists, document contains the session → view with all fields."""
    stack = await _build_stack()
    try:
        chat_id = uuid4()
        user_id = "user-lookup-abc"
        session_id = uuid4()

        await persist_research_session_start_independent(
            chat_id=chat_id,
            user_id=user_id,
            user_query="how many moons does mars have",
            user_message_id=uuid4(),
            agent_message_id=uuid4(),
            research_session_id=session_id,
            research_depth="medium",
            query_mode="deep_research",
            storage_stack=stack,
        )

        view = await load_session_control_view(
            chat_id,
            session_id,
            user_id,
            settings=_cached_settings(),
            storage_stack=stack,
            db=AsyncMock(),  # unused in cached branch
        )
        assert view is not None
        assert isinstance(view, SessionControlView)
        assert view.id == session_id
        assert view.chat_id == chat_id
        assert view.user_id == user_id
        assert view.status == ResearchStatus.IN_PROGRESS
        assert view.query == "how many moons does mars have"
        assert view.query_mode == "deep_research"
    finally:
        await stack.stop(timeout=2.0)


@pytest.mark.asyncio
async def test_cached_returns_none_when_session_not_in_document() -> None:
    """Chat exists but `research_sessions` doesn't contain the given id
    (pruned, corrupted, or wrong id) → None so the caller 404s."""
    stack = await _build_stack()
    try:
        chat_id = uuid4()
        user_id = "user-dangling"
        other_session_id = uuid4()
        unknown_session_id = uuid4()

        await persist_research_session_start_independent(
            chat_id=chat_id,
            user_id=user_id,
            user_query="q",
            user_message_id=uuid4(),
            agent_message_id=uuid4(),
            research_session_id=other_session_id,
            research_depth="light",
            query_mode="simple",
            storage_stack=stack,
        )

        view = await load_session_control_view(
            chat_id,
            unknown_session_id,
            user_id,
            settings=_cached_settings(),
            storage_stack=stack,
            db=AsyncMock(),
        )
        assert view is None
    finally:
        await stack.stop(timeout=2.0)


@pytest.mark.asyncio
async def test_cached_returns_none_when_user_does_not_own_chat() -> None:
    """Ownership enforcement: chat exists and carries the session, but the
    authenticated user_id does not match doc.meta.user_id — refuse."""
    stack = await _build_stack()
    try:
        chat_id = uuid4()
        owner_id = "user-owner"
        attacker_id = "user-attacker"
        session_id = uuid4()

        await persist_research_session_start_independent(
            chat_id=chat_id,
            user_id=owner_id,
            user_query="secret",
            user_message_id=uuid4(),
            agent_message_id=uuid4(),
            research_session_id=session_id,
            research_depth="medium",
            query_mode="deep_research",
            storage_stack=stack,
        )

        view = await load_session_control_view(
            chat_id,
            session_id,
            attacker_id,
            settings=_cached_settings(),
            storage_stack=stack,
            db=AsyncMock(),
        )
        assert view is None
    finally:
        await stack.stop(timeout=2.0)


@pytest.mark.asyncio
async def test_cached_returns_none_when_cache_raises() -> None:
    """If the storage stack raises during cache.get (ChatDocument deleted,
    backend error), the helper must return None rather than propagating —
    otherwise the SSE generator would crash instead of the caller serving
    an honest 404."""
    broken_stack = SimpleNamespace(
        cache=SimpleNamespace(
            get=AsyncMock(side_effect=RuntimeError("document gone")),
        ),
    )

    view = await load_session_control_view(
        uuid4(),
        uuid4(),
        "user-broken",
        settings=_cached_settings(),
        storage_stack=broken_stack,
        db=AsyncMock(),
    )
    assert view is None


@pytest.mark.asyncio
async def test_legacy_returns_view_from_sql_row() -> None:
    """Legacy mode falls through to db.get(ResearchSession, id) and projects
    the SQL row into a view."""
    session_id = uuid4()
    chat_id = uuid4()
    started = datetime.now(UTC) - timedelta(minutes=5)

    fake_row = SimpleNamespace(
        id=session_id,
        user_id="user-legacy",
        chat_id=chat_id,
        status=ResearchStatus.IN_PROGRESS,
        message_id=uuid4(),
        started_at=started,
        completed_at=None,
        error_message=None,
        query="how are you",
        query_mode="deep_research",
        current_step_index=2,
        plan={"steps": [{"idx": 0}, {"idx": 1}, {"idx": 2}, {"idx": 3}]},
    )
    mock_db = AsyncMock()
    mock_db.get = AsyncMock(return_value=fake_row)

    view = await load_session_control_view(
        chat_id,
        session_id,
        "user-legacy",
        settings=_legacy_settings(),
        storage_stack=None,
        db=mock_db,
    )
    assert view is not None
    assert view.id == session_id
    assert view.chat_id == chat_id
    assert view.user_id == "user-legacy"
    assert view.status == ResearchStatus.IN_PROGRESS
    assert view.query == "how are you"
    assert view.query_mode == "deep_research"
    assert view.current_step == 2
    assert view.total_steps == 4


@pytest.mark.asyncio
async def test_legacy_returns_none_when_row_missing() -> None:
    """Legacy mode, random id → db.get returns None → None."""
    mock_db = AsyncMock()
    mock_db.get = AsyncMock(return_value=None)

    view = await load_session_control_view(
        uuid4(),
        uuid4(),
        "user-legacy",
        settings=_legacy_settings(),
        storage_stack=None,
        db=mock_db,
    )
    assert view is None


@pytest.mark.asyncio
async def test_legacy_rejects_wrong_chat_id() -> None:
    """Legacy mode: session row exists but chat_id in URL doesn't match the
    row — ownership enforcement returns None."""
    session_id = uuid4()
    fake_row = SimpleNamespace(
        id=session_id,
        user_id="user-legacy",
        chat_id=uuid4(),  # real owner
        status=ResearchStatus.IN_PROGRESS,
        message_id=uuid4(),
        started_at=datetime.now(UTC),
        completed_at=None,
        error_message=None,
        query="x",
        query_mode="deep_research",
        current_step_index=0,
        plan={},
    )
    mock_db = AsyncMock()
    mock_db.get = AsyncMock(return_value=fake_row)

    view = await load_session_control_view(
        uuid4(),  # a different chat_id supplied by caller
        session_id,
        "user-legacy",
        settings=_legacy_settings(),
        storage_stack=None,
        db=mock_db,
    )
    assert view is None
