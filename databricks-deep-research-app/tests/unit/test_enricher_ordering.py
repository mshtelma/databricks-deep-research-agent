"""Regression gate for `feedback_render_must_include_plugin_extensions`.

The plugin `ContextEnricher.enrich_research_memory` writes an
`account_brief_markdown` payload via `memory.upsert_plugin_ext(...)` under
the 5 s `asyncio.wait_for` in `framework_orchestrator`. The subsequent
`workflow.start()` step reads the memory via `memory.render(...)` — and
MUST see the plugin's markdown.

This test drives `CachedChatMemoryService` end-to-end: hydrate → upsert via
the plugin path (through cache.mutate + in-memory mirror update) → render,
and asserts the markdown is present before any DB flush tick would have
fired. Mirrors the invariant documented in the plan's Pre-mortem scenario
4 / §R-5.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from tests.fakes.fake_backend import FakeBackend

from deep_research.services.cached.chat_memory import CachedChatMemoryService
from deep_research.storage.cache import ChatStateCache, Hydrator
from deep_research.storage.cold_cache import ColdReadCache
from deep_research.storage.factory import StorageStack
from deep_research.storage.queue import WriteQueue


async def _stack() -> StorageStack:
    backend = FakeBackend()
    await backend.migrate()
    cache = ChatStateCache(backend)
    queue = WriteQueue(
        backend,
        cache,
        # Huge flush interval so the test proves synchrony doesn't rely on
        # DB flush having happened.
        flush_interval_sec=3600.0,
        flush_size=10,
        backoffs=(0.01,),
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


async def test_plugin_ext_visible_to_render_before_db_flush() -> None:
    """Enricher write → render inside the same orchestrator turn is
    synchronously visible, even with the WriteQueue flush interval set to
    3600 s so no flush could possibly have completed.
    """
    stack = await _stack()
    try:
        cid = uuid4()
        # Hydrate a brand-new chat (empty).
        stack.hydrator.start(cid, user_id="u")
        await stack.cache.get(cid, user_id="u")

        svc = CachedChatMemoryService(stack)
        await svc.hydrate(cid)

        # Simulate the plugin enricher under the orchestrator's 5 s window.
        async def _enricher() -> None:
            await svc.upsert_plugin_ext(
                "sapresalesbot",
                {
                    "account_brief": {"account_name": "Acme"},
                    "account_brief_markdown": "# Acme account brief\nStage: Evaluate",
                },
            )

        await asyncio.wait_for(_enricher(), timeout=5.0)

        # The NEXT step is workflow.start(), which calls memory.render().
        # The plugin's markdown must be there even though no DB flush ran.
        rendered = svc.render(agent_type="crm_context")
        assert "# Acme account brief" in rendered, (
            f"plugin_ext markdown missing from render() — got: {rendered!r}"
        )

        # Snapshot must also contain the plugin_extensions entry.
        snap = svc.snapshot()
        assert "sapresalesbot" in snap.plugin_extensions
        assert snap.plugin_extensions["sapresalesbot"]["account_brief_markdown"].startswith(
            "# Acme account brief"
        )
    finally:
        await stack.stop(timeout=1.0)


async def test_plugin_ext_size_cap_enforced() -> None:
    """Oversize payload raises ValueError — legacy-parity invariant."""
    stack = await _stack()
    try:
        cid = uuid4()
        stack.hydrator.start(cid, user_id="u")
        await stack.cache.get(cid, user_id="u")
        svc = CachedChatMemoryService(stack)
        await svc.hydrate(cid)

        huge = "x" * (svc.PAYLOAD_MAX_BYTES + 100)
        with pytest.raises(ValueError, match="exceeds PAYLOAD_MAX_BYTES"):
            await svc.upsert_plugin_ext("p", {"blob": huge})
    finally:
        await stack.stop(timeout=1.0)


async def test_hydrate_loads_existing_plugin_ext() -> None:
    """Re-hydrating after a flush: legacy in-memory mirror is repopulated
    from ChatState.memory.plugin_ext, so render() sees prior briefs.
    """
    stack = await _stack()
    try:
        cid = uuid4()
        stack.hydrator.start(cid, user_id="u")
        await stack.cache.get(cid, user_id="u")

        # Seed an enricher brief.
        svc1 = CachedChatMemoryService(stack)
        await svc1.hydrate(cid)
        await svc1.upsert_plugin_ext(
            "sapresalesbot", {"account_brief_markdown": "# First"}
        )

        # Now construct a fresh service (simulating a fresh turn) and
        # hydrate — should see the in-memory cache's existing state.
        svc2 = CachedChatMemoryService(stack)
        snap = await svc2.hydrate(cid)
        assert "sapresalesbot" in snap.plugin_extensions
        assert (
            snap.plugin_extensions["sapresalesbot"]["account_brief_markdown"]
            == "# First"
        )
    finally:
        await stack.stop(timeout=1.0)


async def test_file_memo_upsert_visible_to_snapshot() -> None:
    """Regression: `_upsert_file_memo` must write to `state.memory.files`
    and update the legacy `self._files` mirror so the CURRENT turn's
    `snapshot().files` sees the memo. Previously, two stacked bugs
    (missing `_read_chat` + wrong-key `state.files`) silently swallowed
    every write, leaving `snapshot.files == []` and causing the
    sapresalesbot plugin's `enrich_research_memory` to short-circuit
    with `CONTEXT_ENRICHER_NOOP reason=no_files`, dropping persona info
    from uploaded docs.
    """
    stack = await _stack()
    try:
        cid = uuid4()
        stack.hydrator.start(cid, user_id="u")
        await stack.cache.get(cid, user_id="u")

        svc = CachedChatMemoryService(stack)
        await svc.hydrate(cid)
        assert svc.snapshot().files == []

        fid = uuid4()
        await svc._upsert_file_memo(
            chat_id=cid,
            file_id=fid,
            filename="roster.pdf",
            content_summary="Attendees: Alice (CTO), Bob (VP Eng)",
            chunk_count=3,
            research_session_id=None,
        )

        snap = svc.snapshot()
        assert len(snap.files) == 1, (
            f"expected 1 file memo in snapshot, got: {snap.files!r}"
        )
        assert snap.files[0].id == fid  # FileRef.id holds the file_id
        # Idempotent: a second upsert with the same file_id is a no-op.
        await svc._upsert_file_memo(
            chat_id=cid,
            file_id=fid,
            filename="roster.pdf",
            content_summary="Attendees: Alice (CTO), Bob (VP Eng)",
            chunk_count=3,
            research_session_id=None,
        )
        assert len(svc.snapshot().files) == 1
    finally:
        await stack.stop(timeout=1.0)


async def test_file_memo_upsert_survives_rehydrate() -> None:
    """After `_upsert_file_memo`, a fresh `CachedChatMemoryService`
    hydrating the same chat must see the memo in `state.memory.files` —
    proves the write went to the canonical path, not a stray attribute.
    """
    stack = await _stack()
    try:
        cid = uuid4()
        stack.hydrator.start(cid, user_id="u")
        await stack.cache.get(cid, user_id="u")

        svc1 = CachedChatMemoryService(stack)
        await svc1.hydrate(cid)
        fid = uuid4()
        await svc1._upsert_file_memo(
            chat_id=cid,
            file_id=fid,
            filename="brief.pdf",
            content_summary="Decision makers identified.",
            chunk_count=1,
            research_session_id=None,
        )

        # Fresh service, same chat_id — hydrate must repopulate from the
        # in-memory ChatState.memory.files (no DB flush has happened).
        svc2 = CachedChatMemoryService(stack)
        snap = await svc2.hydrate(cid)
        assert len(snap.files) == 1
        assert snap.files[0].id == fid  # FileRef.id holds the file_id
    finally:
        await stack.stop(timeout=1.0)
