"""Unit tests for `deep_research.storage.factory`."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from deep_research.storage.backend import StorageBackend
from deep_research.storage.cache import ChatStateCache, Hydrator
from deep_research.storage.cold_cache import ColdReadCache
from deep_research.storage.factory import (
    create_backend,
    create_storage_stack,
)
from deep_research.storage.queue import WriteQueue


def _fake_settings(**overrides) -> SimpleNamespace:
    """Build a Settings-shaped object without going through pydantic.

    Keeps the factory unit test independent of the real Settings model and
    env-var plumbing; relevant attributes live here.
    """
    defaults = {
        "storage_backend": "fake",
        "storage_service_impl": "cached",
        "storage_warehouse_id": None,
        "storage_catalog": "main",
        "storage_schema": "deep_research_state",
        "storage_statement_timeout_sec": 30.0,
        "storage_flush_interval_sec": 0.05,
        "storage_flush_size": 10,
        "storage_cache_idle_ttl_min": 30,
        "storage_cold_cache_ttl_sec": 60.0,
        "storage_cold_cache_max_entries": 1000,
        "storage_max_concurrent_hydrations": 5,
        "storage_event_buffer_cap": 10_000,
        "storage_cleanup_enabled": False,  # keep tests quick
        "storage_cleanup_interval_sec": 3600.0,
        "storage_chat_retention_days": 7,
        "storage_migration_mode": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestCreateBackend:
    def test_fake_backend_by_default(self) -> None:
        b = create_backend(_fake_settings())
        assert isinstance(b, StorageBackend)

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown STORAGE_BACKEND"):
            create_backend(_fake_settings(storage_backend="redis"))

    def test_sql_warehouse_requires_warehouse_id(self) -> None:
        with pytest.raises(ValueError, match="STORAGE_WAREHOUSE_ID"):
            create_backend(_fake_settings(storage_backend="sql_warehouse"))


class TestStorageStack:
    async def test_stack_wires_components(self) -> None:
        stack = create_storage_stack(_fake_settings())
        assert isinstance(stack.backend, StorageBackend)
        assert isinstance(stack.cache, ChatStateCache)
        assert isinstance(stack.queue, WriteQueue)
        assert isinstance(stack.hydrator, Hydrator)
        assert isinstance(stack.cold_cache, ColdReadCache)
        assert stack.cleanup is None  # disabled

    async def test_cache_notifies_queue_on_dirty(self) -> None:
        """Late-binding wire-up: `cache._on_dirty` must point at the queue."""
        stack = create_storage_stack(_fake_settings())
        assert stack.cache._on_dirty == stack.queue.notify_dirty

    async def test_start_stop_is_idempotent(self) -> None:
        stack = create_storage_stack(_fake_settings())
        await stack.start()
        await stack.start()  # no-op
        await stack.stop(timeout=1.0)
        await stack.stop(timeout=1.0)  # no-op

    async def test_start_runs_migrate(self) -> None:
        stack = create_storage_stack(_fake_settings())
        backend = stack.backend
        assert not backend.migrated  # type: ignore[attr-defined]
        await stack.start()
        assert backend.migrated  # type: ignore[attr-defined]
        await stack.stop(timeout=1.0)

    async def test_stack_end_to_end_mutate(self) -> None:
        from deep_research.storage.documents import Message

        stack = create_storage_stack(_fake_settings())
        await stack.start()
        try:
            cid = uuid4()
            stack.hydrator.start(cid, user_id="u")
            await stack.cache.get(cid, user_id="u")
            await stack.cache.mutate(
                cid, lambda d: d.state.add_message(Message(role="user", content="hi"))
            )
            # Wait for one flush tick.
            import asyncio

            await asyncio.sleep(0.15)
            loaded = await stack.backend.load_chat(cid)
            assert loaded is not None
            assert loaded.state.messages[0].content == "hi"
        finally:
            await stack.stop(timeout=2.0)

    async def test_install_signal_handlers_idempotent(self) -> None:
        stack = create_storage_stack(_fake_settings())
        stack.install_signal_handlers()
        stack.install_signal_handlers()  # no-op
        assert stack._signal_handlers_installed is True
        await stack.stop()
