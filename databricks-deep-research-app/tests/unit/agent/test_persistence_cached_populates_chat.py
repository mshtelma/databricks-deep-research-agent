"""Behavioural test for the Step 1 storage-path-mismatch fix.

Regression guard for the prod bug where `JobManager.submit_job` did not
thread `storage_stack` into `persist_research_session_start_independent`,
causing the Chat document to be written to Lakebase directly while the
framework later read from the in-process StorageStack cache — which refused
to auto-create without `user_id` and produced
`FWK_EXISTING_SOURCES_CACHE_LOAD_FAILED` + `FWK_PERSISTENCE_FAILED`.

The contract under test: when the caller supplies `storage_stack`, the
helper routes through `_persist_session_start_cached`, which populates the
same in-process ChatDocument cache that `stream_research` later reads from.
After submit returns, `stack.cache.get(chat_id)` (no ``user_id`` argument)
must yield a fully-populated document without raising.
"""

from __future__ import annotations

# --- Must configure env before importing deep_research.* so Settings()
# --- validates and the cached-mode branch is taken.
import os

# Hard assignment because tests/unit/api/conftest.py pins
# STORAGE_SERVICE_IMPL=sqlalchemy_legacy first in a full-suite pass and
# setdefault would silently no-op the cached-mode override.
os.environ["STORAGE_BACKEND"] = "fake"
os.environ["STORAGE_SERVICE_IMPL"] = "cached"
os.environ.setdefault("SECRET_KEY", "test-secret-key-unit-tests-only")
os.environ.setdefault("DATABRICKS_HOST", "https://test.azuredatabricks.net")
os.environ.setdefault("DATABRICKS_TOKEN", "test-token")
os.environ.setdefault("LLM_ENDPOINT_NAME", "test-endpoint")

from uuid import UUID, uuid4  # noqa: E402

import pytest  # noqa: E402
from tests.fakes.fake_backend import FakeBackend  # noqa: E402

from deep_research.agent.persistence import (  # noqa: E402
    persist_research_session_cancelled_independent,
    persist_research_session_start_independent,
)
from deep_research.core.config import get_settings  # noqa: E402
from deep_research.storage.cache import ChatStateCache, Hydrator  # noqa: E402
from deep_research.storage.cold_cache import ColdReadCache  # noqa: E402
from deep_research.storage.factory import StorageStack  # noqa: E402
from deep_research.storage.queue import WriteQueue  # noqa: E402

# Clear any previously-cached Settings so our env overrides take effect
# if another test in the same pytest run loaded Settings first.
get_settings.cache_clear()


async def _build_stack() -> StorageStack:
    """Build a minimal StorageStack over FakeBackend.

    Mirrors the helper in tests/services/test_cached_chat_scoped_services.py
    so we share the same pattern. Flush interval is aggressive so the
    ``flush_chat_now`` call inside `_persist_session_start_cached` is
    exercised deterministically.
    """
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
    cache._on_dirty = queue.notify_dirty  # noqa: SLF001 — deliberate wire-up
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


@pytest.mark.asyncio
async def test_persist_start_populates_cache_when_storage_stack_provided() -> None:
    """After submit, the Chat document must be readable from the cache
    without supplying ``user_id`` — proving the framework's later
    ``cache.get(chat_id)`` calls will not hit the refusal path."""
    stack = await _build_stack()
    try:
        chat_id = uuid4()
        user_id = "user-abc-123"
        query = "What is the capital of nowhere?"
        user_msg_id = uuid4()
        agent_msg_id = uuid4()
        session_id = uuid4()

        await persist_research_session_start_independent(
            chat_id=chat_id,
            user_id=user_id,
            user_query=query,
            user_message_id=user_msg_id,
            agent_message_id=agent_msg_id,
            research_session_id=session_id,
            research_depth="medium",
            query_mode="deep_research",
            storage_stack=stack,
        )

        # Core assertion: subsequent reads from the cache do NOT need
        # user_id. Before Step 1, this raised
        # "chat … does not exist and no user_id was provided".
        doc = await stack.cache.get(chat_id)
        assert doc is not None, "ChatDocument must exist in the cache after submit"

        # Sanity-check the document carries what submit was supposed to write.
        state = doc.state
        assert state.chat.title  # title set from the query
        assert len(state.messages) == 2, "user message + agent placeholder expected"

        roles = {m.role for m in state.messages}
        assert roles == {"user", "agent"}, f"unexpected roles: {roles}"

        assert any(
            UUID(str(rs.id)) == session_id
            and rs.status == "in_progress"
            for rs in state.research_sessions
        ), "research_session with status=in_progress must be upserted"

        # And: the queue flush forces durability to the backend so a
        # different worker hydrating from cold storage also sees it.
        raw = await stack.backend.load_chat(chat_id)
        assert raw is not None, "backend must carry the flushed ChatDocument"
    finally:
        await stack.stop(timeout=2.0)


@pytest.mark.asyncio
async def test_persist_start_cache_is_reusable_by_framework_readers() -> None:
    """Second read of the same chat (simulating stream_research calling
    _load_existing_sources) must succeed without raising — the refusal
    path at storage/cache.py:180 is only hit when user_id is absent AND
    the chat doesn't exist in the backend. After Step 1, the chat always
    exists, so passing no user_id is safe."""
    stack = await _build_stack()
    try:
        chat_id = uuid4()
        user_id = "user-xyz-789"
        await persist_research_session_start_independent(
            chat_id=chat_id,
            user_id=user_id,
            user_query="Query body",
            user_message_id=uuid4(),
            agent_message_id=uuid4(),
            research_session_id=uuid4(),
            research_depth="light",
            query_mode="simple",
            storage_stack=stack,
        )

        # framework's _load_existing_sources does exactly this, with no user_id
        doc1 = await stack.cache.get(chat_id)
        doc2 = await stack.cache.get(chat_id)
        assert doc1 is doc2, "cache.get must return the same live doc"
    finally:
        await stack.stop(timeout=2.0)


@pytest.mark.asyncio
async def test_persist_cancelled_cached_transitions_document_status() -> None:
    """``persist_research_session_cancelled_independent`` routes through
    the cached path when storage_stack + chat_id are provided: the
    ChatDocument's research_session state must flip to
    ``status="cancelled"`` with a ``completed_at``."""
    stack = await _build_stack()
    try:
        chat_id = uuid4()
        user_id = "user-cancel-abc"
        session_id = uuid4()
        await persist_research_session_start_independent(
            chat_id=chat_id,
            user_id=user_id,
            user_query="q",
            user_message_id=uuid4(),
            agent_message_id=uuid4(),
            research_session_id=session_id,
            research_depth="medium",
            query_mode="deep_research",
            storage_stack=stack,
        )

        await persist_research_session_cancelled_independent(
            research_session_id=session_id,
            storage_stack=stack,
            chat_id=chat_id,
        )

        doc = await stack.cache.get(chat_id)
        rs = next(
            (r for r in doc.state.research_sessions if r.id == session_id),
            None,
        )
        assert rs is not None, "research session must survive the transition"
        assert rs.status == "cancelled"
        assert rs.completed_at is not None
    finally:
        await stack.stop(timeout=2.0)


def test_submit_job_source_threads_storage_stack_into_persistence_call() -> None:
    """Guard against regressions where a future refactor drops the
    storage_stack kwarg. We assert on the source because the full
    behavioural path through submit_job requires a live AsyncSession and
    a large fan-out of collaborators (LLM, brave_client, etc.) — the
    repo's convention (see test_job_manager_agent.py) for these guards is
    source inspection.
    """
    import inspect

    from deep_research.services.job_manager import JobManager

    source = inspect.getsource(JobManager.submit_job)

    # Step 1 kwarg must be present in the persistence call.
    assert "storage_stack=self._storage_stack" in source, (
        "submit_job must thread storage_stack into the persistence call; "
        "without it the cached-mode branch is skipped and the chat row is "
        "written to Lakebase without populating the framework's cache."
    )

    # Fail-fast guard must be present for cached mode.
    assert 'storage_service_impl == "cached"' in source
    assert "self._storage_stack is None" in source
    assert "StorageStack attached" in source
