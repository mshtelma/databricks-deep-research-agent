"""Unit tests for PreferencesService."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.dialects import postgresql

from deep_research.services.preferences_service import PreferencesService


@pytest.mark.asyncio
async def test_get_preferences_seeds_user_row_before_default_preferences() -> None:
    """Preferences defaults must not fail if auth sync missed the users row."""
    session = MagicMock()
    preferences = MagicMock()
    select_result = MagicMock()
    select_result.scalar_one.return_value = preferences
    session.execute = AsyncMock(side_effect=[MagicMock(), MagicMock(), select_result])

    result = await PreferencesService(session).get_preferences("5442205634071161")

    assert result is preferences
    assert session.execute.await_count == 3

    user_stmt = session.execute.await_args_list[0].args[0]
    prefs_stmt = session.execute.await_args_list[1].args[0]

    user_sql = str(user_stmt.compile(dialect=postgresql.dialect()))
    prefs_sql = str(prefs_stmt.compile(dialect=postgresql.dialect()))

    assert "INSERT INTO users" in user_sql
    assert "ON CONFLICT (user_id) DO NOTHING" in user_sql
    assert "INSERT INTO user_preferences" in prefs_sql
