"""Integration test for cross-turn memory consolidation in the cached persist path.

When `CHAT_MEMORY_UNIFIED` is on, completing a research turn must consolidate the
turn's VERIFIED claims into durable `ChatState.memory.findings` (and skip refuted
ones). When off, behavior is unchanged (no findings). Drives the real cached
persist path over the StorageStack/FakeBackend double — no Postgres.
"""

from __future__ import annotations

import os

os.environ["STORAGE_BACKEND"] = "fake"
os.environ["STORAGE_SERVICE_IMPL"] = "cached"
os.environ.setdefault("SECRET_KEY", "test-secret-key-unit-tests-only")
os.environ.setdefault("DATABRICKS_HOST", "https://test.azuredatabricks.net")
os.environ.setdefault("DATABRICKS_TOKEN", "test-token")
os.environ.setdefault("LLM_ENDPOINT_NAME", "test-endpoint")

from uuid import uuid4  # noqa: E402

import pytest  # noqa: E402
from tests.fakes.fake_backend import FakeBackend  # noqa: E402

from deep_research.agent.persistence import (  # noqa: E402
    _persist_session_complete_cached,
    persist_research_session_start_independent,
)
from deep_research.agent.state import ClaimInfo, ResearchState  # noqa: E402
from deep_research.core.config import get_settings  # noqa: E402
from deep_research.storage.cache import ChatStateCache, Hydrator  # noqa: E402
from deep_research.storage.cold_cache import ColdReadCache  # noqa: E402
from deep_research.storage.factory import StorageStack  # noqa: E402
from deep_research.storage.queue import WriteQueue  # noqa: E402

pytestmark = pytest.mark.unit


async def _build_stack() -> StorageStack:
    backend = FakeBackend()
    await backend.migrate()
    cache = ChatStateCache(backend)
    queue = WriteQueue(
        backend, cache, flush_interval_sec=0.05, flush_size=20, backoffs=(0.01, 0.02)
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


async def _seed(stack, chat_id, session_id, agent_msg_id) -> None:
    await persist_research_session_start_independent(
        chat_id=chat_id,
        user_id="u",
        user_query="What is X?",
        user_message_id=uuid4(),
        agent_message_id=agent_msg_id,
        research_session_id=session_id,
        research_depth="medium",
        query_mode="deep_research",
        storage_stack=stack,
    )


def _claim(text: str, verdict: str) -> ClaimInfo:
    return ClaimInfo(
        claim_text=text,
        claim_type="general",
        position_start=0,
        position_end=0,
        verification_verdict=verdict,
        abstained=False,
    )


def _state_with_claims() -> ResearchState:
    state = ResearchState(query="What is X?")
    state.final_report = "Report body."
    state.claims = [
        _claim("Verified fact about X.", "supported"),
        _claim("Refuted fact about X.", "contradicted"),
    ]
    return state


def _set_flag(value: bool) -> None:
    os.environ["CHAT_MEMORY_UNIFIED"] = "true" if value else "false"
    get_settings.cache_clear()


async def test_flag_on_consolidates_only_verified_claims() -> None:
    _set_flag(True)
    stack = await _build_stack()
    try:
        chat_id, session_id, agent_msg_id = uuid4(), uuid4(), uuid4()
        await _seed(stack, chat_id, session_id, agent_msg_id)
        await _persist_session_complete_cached(
            stack=stack,
            chat_id=chat_id,
            research_session_id=session_id,
            agent_message_id=agent_msg_id,
            state=_state_with_claims(),
        )
        doc = await stack.cache.get(chat_id)
        contents = {f.content for f in doc.state.memory.findings}
        assert "Verified fact about X." in contents
        assert "Refuted fact about X." not in contents
    finally:
        _set_flag(False)
        await stack.stop()


async def test_flag_off_writes_no_findings() -> None:
    _set_flag(False)
    stack = await _build_stack()
    try:
        chat_id, session_id, agent_msg_id = uuid4(), uuid4(), uuid4()
        await _seed(stack, chat_id, session_id, agent_msg_id)
        await _persist_session_complete_cached(
            stack=stack,
            chat_id=chat_id,
            research_session_id=session_id,
            agent_message_id=agent_msg_id,
            state=_state_with_claims(),
        )
        doc = await stack.cache.get(chat_id)
        assert doc.state.memory.findings == []
    finally:
        await stack.stop()
