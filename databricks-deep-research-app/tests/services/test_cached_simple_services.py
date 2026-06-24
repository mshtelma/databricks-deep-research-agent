"""Unit tests for the simplest Wave-5a cached services.

Covers `CachedFeedbackService`, `CachedUserService`, `CachedPreferencesService`.
Exercises them against a FakeBackend-backed `StorageStack`. Full parity with
the legacy SQLAlchemy impls is deferred to integration tests.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from tests.fakes.fake_backend import FakeBackend

from deep_research.services._protocols import (
    IFeedbackService,
    IPreferencesService,
    IUserService,
)
from deep_research.services.cached.feedback import CachedFeedbackService
from deep_research.services.cached.preferences import CachedPreferencesService
from deep_research.services.cached.user import CachedUserService
from deep_research.storage.cache import ChatStateCache, Hydrator
from deep_research.storage.cold_cache import ColdReadCache
from deep_research.storage.factory import StorageStack
from deep_research.storage.queue import WriteQueue


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


# --- CachedFeedbackService -------------------------------------------------


class TestCachedFeedback:
    async def test_protocol(self) -> None:
        stack = await _stack()
        try:
            svc = CachedFeedbackService(stack)
            assert isinstance(svc, IFeedbackService)
        finally:
            await stack.stop(timeout=1.0)

    async def test_create_get_update_delete(self) -> None:
        stack = await _stack()
        try:
            svc = CachedFeedbackService(stack)
            mid = uuid4()
            created = await svc.create_feedback(
                mid, "u1", "positive", feedback_text="great",
            )
            assert created.rating == "positive"

            # get_feedback
            loaded = await svc.get_feedback(mid, "u1")
            assert loaded is not None
            assert loaded.rating == "positive"
            fid = loaded.id

            # update_feedback
            updated = await svc.update_feedback(
                fid, "u1", rating="negative", feedback_text="meh",
            )
            assert updated is not None
            assert updated.rating == "negative"
            assert updated.feedback_text == "meh"

            # stats
            stats = await svc.get_message_feedback_stats(mid)
            assert stats == {
                "positive_count": 0, "negative_count": 1, "total": 1,
            }

            # delete
            assert await svc.delete_feedback(fid, "u1") is True
            assert await svc.delete_feedback(fid, "u1") is False
        finally:
            await stack.stop(timeout=1.0)

    async def test_duplicate_feedback_rejected(self) -> None:
        stack = await _stack()
        try:
            svc = CachedFeedbackService(stack)
            mid = uuid4()
            await svc.create_feedback(mid, "u1", "positive")
            with pytest.raises(ValueError, match="already exists"):
                await svc.create_feedback(mid, "u1", "positive")
        finally:
            await stack.stop(timeout=1.0)

    async def test_invalid_rating(self) -> None:
        stack = await _stack()
        try:
            svc = CachedFeedbackService(stack)
            with pytest.raises(ValueError, match="Invalid rating"):
                await svc.create_feedback(uuid4(), "u1", "lukewarm")
        finally:
            await stack.stop(timeout=1.0)

    async def test_update_wrong_owner_returns_none(self) -> None:
        stack = await _stack()
        try:
            svc = CachedFeedbackService(stack)
            mid = uuid4()
            created = await svc.create_feedback(mid, "u1", "positive")
            result = await svc.update_feedback(created.id, "u2", rating="negative")
            assert result is None
        finally:
            await stack.stop(timeout=1.0)


# --- CachedUserService -----------------------------------------------------


class TestCachedUser:
    async def test_protocol(self) -> None:
        stack = await _stack()
        try:
            assert isinstance(CachedUserService(stack), IUserService)
        finally:
            await stack.stop(timeout=1.0)

    async def test_upsert_creates_then_updates(self) -> None:
        stack = await _stack()
        try:
            svc = CachedUserService(stack)
            await svc.upsert("u1", "u1@example.com", "Alice")
            doc = await stack.backend.load_user_doc("u1")
            assert doc is not None
            assert doc.profile["email"] == "u1@example.com"
            assert doc.profile["display_name"] == "Alice"

            # Update
            await svc.upsert("u1", "alice@new.com", "Alice v2")
            doc = await stack.backend.load_user_doc("u1")
            assert doc.profile["email"] == "alice@new.com"
            assert doc.profile["display_name"] == "Alice v2"
        finally:
            await stack.stop(timeout=1.0)

    async def test_resolve_user_ids(self) -> None:
        stack = await _stack()
        try:
            svc = CachedUserService(stack)
            await svc.upsert("u1", "u1@x.com", "Alice")
            await svc.upsert("u2", "u2@x.com", "Bob")
            resolved = await svc.resolve_user_ids(["u1", "u2", "missing"])
            assert resolved == {
                "u1": ("u1@x.com", "Alice"),
                "u2": ("u2@x.com", "Bob"),
            }
        finally:
            await stack.stop(timeout=1.0)

    async def test_resolve_empty(self) -> None:
        stack = await _stack()
        try:
            svc = CachedUserService(stack)
            assert await svc.resolve_user_ids([]) == {}
        finally:
            await stack.stop(timeout=1.0)


# --- CachedPreferencesService ----------------------------------------------


class TestCachedPreferences:
    async def test_protocol(self) -> None:
        stack = await _stack()
        try:
            assert isinstance(CachedPreferencesService(stack), IPreferencesService)
        finally:
            await stack.stop(timeout=1.0)

    async def test_get_preferences_creates_defaults(self) -> None:
        stack = await _stack()
        try:
            svc = CachedPreferencesService(stack)
            prefs = await svc.get_preferences("u1")
            assert prefs.theme == "system"
            assert prefs.notifications_enabled is True
            assert prefs.default_query_mode == "simple"
            assert prefs.system_instructions is None
        finally:
            await stack.stop(timeout=1.0)

    async def test_update_and_reload(self) -> None:
        stack = await _stack()
        try:
            svc = CachedPreferencesService(stack)
            updated = await svc.update_preferences(
                "u1",
                system_instructions="be terse",
                theme="dark",
                notifications_enabled=False,
            )
            assert updated.system_instructions == "be terse"
            assert updated.theme == "dark"
            assert updated.notifications_enabled is False

            fresh = await svc.get_preferences("u1")
            assert fresh.system_instructions == "be terse"
            assert fresh.theme == "dark"
        finally:
            await stack.stop(timeout=1.0)

    async def test_scalar_accessors(self) -> None:
        stack = await _stack()
        try:
            svc = CachedPreferencesService(stack)
            await svc.update_preferences("u1", system_instructions="hello")
            assert await svc.get_system_instructions("u1") == "hello"
            assert await svc.get_default_query_mode("u1") == "simple"
        finally:
            await stack.stop(timeout=1.0)

    async def test_to_dict(self) -> None:
        stack = await _stack()
        try:
            svc = CachedPreferencesService(stack)
            prefs = await svc.update_preferences("u1", theme="light")
            d = svc.to_dict(prefs)
            assert d["theme"] == "light"
            assert d["notifications_enabled"] is True
        finally:
            await stack.stop(timeout=1.0)


# --- Factory smoke -----------------------------------------------------


class TestFactorySmoke:
    async def test_factory_routes_cached_when_flagged(self) -> None:
        from types import SimpleNamespace

        from deep_research.services._impl_factory import (
            make_feedback_service,
            make_preferences_service,
            make_user_service,
        )

        stack = await _stack()
        try:
            settings = SimpleNamespace(storage_service_impl="cached")
            fb = make_feedback_service(settings, stack)
            us = make_user_service(settings, stack)
            pr = make_preferences_service(settings, stack)
            assert isinstance(fb, CachedFeedbackService)
            assert isinstance(us, CachedUserService)
            assert isinstance(pr, CachedPreferencesService)
        finally:
            await stack.stop(timeout=1.0)

    def test_factory_refuses_cached_without_stack(self) -> None:
        from types import SimpleNamespace

        from deep_research.services._impl_factory import make_feedback_service

        settings = SimpleNamespace(storage_service_impl="cached")
        with pytest.raises(ValueError, match="StorageStack"):
            make_feedback_service(settings, None)
