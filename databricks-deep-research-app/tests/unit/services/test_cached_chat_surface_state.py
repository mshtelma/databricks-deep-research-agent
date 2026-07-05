"""Behavioural tests for CachedChatService surface_state_patch support.

Verifies that update_chat with surface_state_patch merges correctly into
ChatState.chat.metadata["surface_state"], and that get_full surfaces the
stored value on the returned ChatFullViewCached.
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


def _build_doc(chat_id, user_id, title="seed") -> ChatDocument:
    now = datetime.now(UTC)
    meta = ChatMeta(
        chat_id=chat_id,
        user_id=user_id,
        title=title,
        preview="",
        created_at=now,
        updated_at=now,
        version=0,
    )
    state = ChatState(chat=ChatMetaEmbed(title=title))
    return ChatDocument(meta=meta, state=state)


@pytest.mark.asyncio
async def test_update_chat_surface_state_patch_stored() -> None:
    """surface_state_patch is merged into ChatState.chat.metadata and persisted."""
    stack = await _build_stack()
    try:
        chat_id = uuid4()
        user_id = "u1"
        doc = _build_doc(chat_id, user_id)
        await stack.backend.write_chat(doc, expected_version=0)

        svc = CachedChatService(stack)
        patch = {
            "agent-1": {
                "data_model": {"key": "val"},
                "surface_etag": "etag-a",
            }
        }
        result = await svc.update_chat(chat_id, user_id, surface_state_patch=patch)

        assert result is not None
        # Read back from cache and verify
        hydrated = await stack.cache.get(chat_id, user_id=user_id)
        ss = hydrated.state.chat.metadata.get("surface_state")
        assert ss is not None
        assert ss["agent-1"]["data_model"] == {"key": "val"}
        assert ss["agent-1"]["surface_etag"] == "etag-a"
    finally:
        await stack.stop()


@pytest.mark.asyncio
async def test_update_chat_surface_state_patch_merges_existing() -> None:
    """A second patch merges rather than replacing the prior surface_state."""
    stack = await _build_stack()
    try:
        chat_id = uuid4()
        user_id = "u1"
        doc = _build_doc(chat_id, user_id)
        await stack.backend.write_chat(doc, expected_version=0)

        svc = CachedChatService(stack)

        # First patch: set agent-1
        await svc.update_chat(
            chat_id, user_id,
            surface_state_patch={"agent-1": {"surface_etag": "etag-1"}},
        )

        # Second patch: add agent-2 (agent-1 must survive)
        await svc.update_chat(
            chat_id, user_id,
            surface_state_patch={"agent-2": {"data_model": {"x": 2}}},
        )

        hydrated = await stack.cache.get(chat_id, user_id=user_id)
        ss = hydrated.state.chat.metadata["surface_state"]
        assert ss["agent-1"]["surface_etag"] == "etag-1"
        assert ss["agent-2"]["data_model"] == {"x": 2}
    finally:
        await stack.stop()


@pytest.mark.asyncio
async def test_get_full_surfaces_surface_state() -> None:
    """ChatFullViewCached.surface_state is populated from stored metadata."""
    stack = await _build_stack()
    try:
        chat_id = uuid4()
        user_id = "u1"
        doc = _build_doc(chat_id, user_id)
        await stack.backend.write_chat(doc, expected_version=0)

        svc = CachedChatService(stack)

        patch = {
            "agent-1": {
                "action_runs": {
                    "run-report": {
                        "session_id": "s1",
                        "message_id": "m1",
                        "status": "completed",
                        "updated_at": "2026-07-01T10:00:00Z",
                    }
                }
            }
        }
        await svc.update_chat(chat_id, user_id, surface_state_patch=patch)

        full = await svc.get_full(chat_id, user_id=user_id)
        assert full is not None
        assert full.surface_state is not None
        assert full.surface_state["agent-1"]["action_runs"]["run-report"]["status"] == "completed"
    finally:
        await stack.stop()


@pytest.mark.asyncio
async def test_get_full_surface_state_none_when_not_set() -> None:
    """ChatFullViewCached.surface_state is None when metadata has no surface_state key."""
    stack = await _build_stack()
    try:
        chat_id = uuid4()
        user_id = "u1"
        doc = _build_doc(chat_id, user_id)
        await stack.backend.write_chat(doc, expected_version=0)

        svc = CachedChatService(stack)
        full = await svc.get_full(chat_id, user_id=user_id)
        assert full is not None
        assert full.surface_state is None
    finally:
        await stack.stop()
