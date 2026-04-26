"""Unit tests for Wave-6 runtime wiring (main.py lifespan + core/deps.py).

Exercises the StorageStack lifecycle bound to an `app.state.storage_stack`
slot, and the `get_storage` / `get_<svc>_service` FastAPI deps against a
`Request`-shaped fake. Full app-under-test integration happens in Wave-8
E2E; this file covers the glue code.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException

from deep_research.core.deps import (
    get_feedback_service,
    get_preferences_service,
    get_research_event_service,
    get_storage,
    get_storage_optional,
    get_user_service,
)
from deep_research.services._protocols import (
    IFeedbackService,
    IPreferencesService,
    IResearchEventService,
    IUserService,
)
from deep_research.storage.factory import StorageStack, create_storage_stack


def _cached_settings():
    """Build settings with `storage_service_impl=cached, backend=fake`."""
    from deep_research.core.config import Settings

    return Settings(
        storage_service_impl="cached",
        storage_backend="fake",
        storage_cleanup_enabled=False,
        storage_flush_interval_sec=0.05,
    )


class TestGetStorage:
    def test_raises_503_when_stack_missing(self) -> None:
        app = FastAPI()
        request = SimpleNamespace(app=app)
        with pytest.raises(HTTPException) as exc:
            get_storage(request)  # type: ignore[arg-type]
        assert exc.value.status_code == 503

    def test_returns_stack_when_present(self) -> None:
        app = FastAPI()
        stack_sentinel = object()
        app.state.storage_stack = stack_sentinel
        request = SimpleNamespace(app=app)
        assert get_storage(request) is stack_sentinel  # type: ignore[arg-type]

    def test_optional_returns_none(self) -> None:
        app = FastAPI()
        request = SimpleNamespace(app=app)
        assert get_storage_optional(request) is None  # type: ignore[arg-type]


class TestCachedDispatch:
    async def test_cached_feedback_service_dispatched(self) -> None:
        settings = _cached_settings()
        stack = create_storage_stack(settings)
        await stack.start()
        try:
            app = FastAPI()
            app.state.storage_stack = stack
            request = SimpleNamespace(app=app, state=SimpleNamespace())

            svc = get_feedback_service(request, settings)  # type: ignore[arg-type]
            assert isinstance(svc, IFeedbackService)

            assert isinstance(
                get_user_service(request, settings), IUserService  # type: ignore[arg-type]
            )
            assert isinstance(
                get_preferences_service(request, settings), IPreferencesService  # type: ignore[arg-type]
            )
            assert isinstance(
                get_research_event_service(request, settings),  # type: ignore[arg-type]
                IResearchEventService,
            )
        finally:
            await stack.stop(timeout=2.0)


class TestStackLifecycle:
    async def test_start_then_stop(self) -> None:
        settings = _cached_settings()
        stack = create_storage_stack(settings)
        await stack.start()
        assert isinstance(stack, StorageStack)
        assert stack.backend is not None
        await stack.stop(timeout=2.0)

    async def test_install_signal_handlers_idempotent(self) -> None:
        settings = _cached_settings()
        stack = create_storage_stack(settings)
        await stack.start()
        try:
            stack.install_signal_handlers()
            stack.install_signal_handlers()  # no-op
            assert stack._signal_handlers_installed is True
        finally:
            await stack.stop(timeout=2.0)

    async def test_make_chat_memory_service_cached(self) -> None:
        from deep_research.services._impl_factory import make_chat_memory_service
        from deep_research.services.cached.chat_memory import (
            CachedChatMemoryService,
        )

        settings = _cached_settings()
        stack = create_storage_stack(settings)
        await stack.start()
        try:
            svc = make_chat_memory_service(settings, stack)
            assert isinstance(svc, CachedChatMemoryService)
        finally:
            await stack.stop(timeout=2.0)
