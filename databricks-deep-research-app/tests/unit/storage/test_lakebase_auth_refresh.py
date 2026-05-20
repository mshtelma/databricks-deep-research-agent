"""Regression tests for Lakebase auth-failure recovery."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from deep_research.db import session as db_session
from deep_research.storage import lakebase
from deep_research.storage.backend import PermanentError


class TestLakebaseAuthRefresh:
    @pytest.mark.asyncio
    async def test_wrap_error_refreshes_credentials_on_password_auth_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        refresh = AsyncMock()
        monkeypatch.setattr(db_session, "refresh_engine_credentials", refresh)

        wrapped = await lakebase._wrap_error_with_auth_refresh(
            RuntimeError("password authentication failed for user 'svc'")
        )

        assert isinstance(wrapped, PermanentError)
        refresh.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_wrap_error_does_not_refresh_on_unrelated_permanent_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        refresh = AsyncMock()
        monkeypatch.setattr(db_session, "refresh_engine_credentials", refresh)

        wrapped = await lakebase._wrap_error_with_auth_refresh(
            RuntimeError("relation does not exist")
        )

        assert isinstance(wrapped, PermanentError)
        refresh.assert_not_awaited()
