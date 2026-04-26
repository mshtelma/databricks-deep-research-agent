"""Routing-layer tests for `submit_job`'s tolerant sub-read helpers.

`_load_conversation_history` and `_load_system_instructions` must route
through `make_message_service` / `make_preferences_service` so cached
deployments never touch the legacy SQL tables from this path. They must
also tolerate failures silently and return a safe default, logging the
error with its `error_type` and `sqlstate` (when present).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from deep_research.api.v1.jobs import (
    _load_conversation_history,
    _load_system_instructions,
    _tolerant_tx_guard,
)


def _cached_settings() -> SimpleNamespace:
    return SimpleNamespace(storage_service_impl="cached")


def _legacy_settings() -> SimpleNamespace:
    return SimpleNamespace(storage_service_impl="sqlalchemy_legacy")


class TestFactoryRouting:
    """Both helpers must invoke the factory with (settings, stack, session=db)."""

    @pytest.mark.asyncio
    async def test_preferences_routes_through_factory_cached(self) -> None:
        db = MagicMock(name="db")
        settings = _cached_settings()
        stack = MagicMock(name="stack")
        prefs_impl = MagicMock()
        prefs_impl.get_system_instructions = AsyncMock(return_value="sysinstr")

        with patch(
            "deep_research.api.v1.jobs.make_preferences_service",
            return_value=prefs_impl,
        ) as mk_prefs:
            result = await _load_system_instructions(
                db, settings, stack, "user-42"
            )

        assert result == "sysinstr"
        mk_prefs.assert_called_once()
        args, kwargs = mk_prefs.call_args
        assert args == (settings, stack)
        assert kwargs == {"session": db}
        prefs_impl.get_system_instructions.assert_awaited_once_with("user-42")

    @pytest.mark.asyncio
    async def test_history_routes_through_factory_cached(self) -> None:
        db = MagicMock(name="db")
        settings = _cached_settings()
        stack = MagicMock(name="stack")
        chat_id = uuid4()
        msg_impl = MagicMock()
        msg_impl.get_conversation_history = AsyncMock(
            return_value=[{"role": "user", "content": "hi"}]
        )

        with patch(
            "deep_research.api.v1.jobs.make_message_service",
            return_value=msg_impl,
        ) as mk_msg:
            result = await _load_conversation_history(
                db, settings, stack, chat_id, is_draft=False
            )

        assert result == [{"role": "user", "content": "hi"}]
        args, kwargs = mk_msg.call_args
        assert args == (settings, stack)
        assert kwargs == {"session": db}
        msg_impl.get_conversation_history.assert_awaited_once_with(
            chat_id, limit=10
        )

    @pytest.mark.asyncio
    async def test_preferences_routes_through_factory_legacy(self) -> None:
        db = MagicMock(name="db")
        db.begin_nested = MagicMock()
        db.begin_nested.return_value.__aenter__ = AsyncMock()
        db.begin_nested.return_value.__aexit__ = AsyncMock(return_value=False)
        settings = _legacy_settings()
        stack = None
        prefs_impl = MagicMock()
        prefs_impl.get_system_instructions = AsyncMock(return_value=None)

        with patch(
            "deep_research.api.v1.jobs.make_preferences_service",
            return_value=prefs_impl,
        ) as mk_prefs:
            result = await _load_system_instructions(db, settings, stack, "u")

        assert result is None
        args, kwargs = mk_prefs.call_args
        assert args == (settings, stack)
        assert kwargs == {"session": db}
        # Savepoint must be used in legacy mode.
        db.begin_nested.assert_called_once()

    @pytest.mark.asyncio
    async def test_history_skipped_for_draft_chat(self) -> None:
        """Fresh drafts have no hydrated doc; skip the service call."""
        db = MagicMock(name="db")
        settings = _cached_settings()
        stack = MagicMock()
        with patch(
            "deep_research.api.v1.jobs.make_message_service"
        ) as mk_msg:
            result = await _load_conversation_history(
                db, settings, stack, uuid4(), is_draft=True
            )
        assert result == []
        mk_msg.assert_not_called()


class TestTolerance:
    """Failures in either helper must be swallowed with a safe default."""

    @pytest.mark.asyncio
    async def test_preferences_failure_returns_none(self, caplog) -> None:
        db = MagicMock(name="db")
        settings = _cached_settings()
        stack = MagicMock()
        prefs_impl = MagicMock()
        prefs_impl.get_system_instructions = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        with patch(
            "deep_research.api.v1.jobs.make_preferences_service",
            return_value=prefs_impl,
        ):
            result = await _load_system_instructions(db, settings, stack, "u")
        assert result is None
        assert any("JOB_PREFERENCES_LOAD_FAILED" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_history_failure_returns_empty(self, caplog) -> None:
        db = MagicMock(name="db")
        settings = _cached_settings()
        stack = MagicMock()
        msg_impl = MagicMock()
        msg_impl.get_conversation_history = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        with patch(
            "deep_research.api.v1.jobs.make_message_service",
            return_value=msg_impl,
        ):
            result = await _load_conversation_history(
                db, settings, stack, uuid4(), is_draft=False
            )
        assert result == []
        assert any("JOB_CONVERSATION_HISTORY_FAILED" in r.getMessage() for r in caplog.records)


class TestTolerantTxGuard:
    """`_tolerant_tx_guard` returns a savepoint only in legacy mode."""

    def test_legacy_uses_begin_nested(self) -> None:
        db = MagicMock(name="db")
        sentinel = MagicMock(name="savepoint")
        db.begin_nested.return_value = sentinel
        result = _tolerant_tx_guard(db, _legacy_settings())
        assert result is sentinel
        db.begin_nested.assert_called_once_with()

    def test_cached_uses_nullcontext(self) -> None:
        from contextlib import AbstractAsyncContextManager

        db = MagicMock(name="db")
        result = _tolerant_tx_guard(db, _cached_settings())
        db.begin_nested.assert_not_called()
        assert isinstance(result, AbstractAsyncContextManager)
