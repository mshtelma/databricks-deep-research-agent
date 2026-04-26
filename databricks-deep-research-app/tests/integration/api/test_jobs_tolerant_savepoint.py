"""Savepoint regression test for `submit_job`'s tolerant sub-reads.

Without the savepoint fix, an `IntegrityError` raised inside
`_load_system_instructions` (e.g., FK violation on
`user_preferences.user_id -> users.user_id`) aborts the outer transaction,
and the next query on the same session — in production, the
`_count_user_active_jobs` SELECT inside `JobManager.submit_job` — fails with
``asyncpg.exceptions.InFailedSQLTransactionError:
current transaction is aborted, commands ignored until end of transaction block``.

This test reproduces that exact failure on a real Postgres session,
forces the helper to encounter an `IntegrityError`, and then verifies
that a subsequent COUNT on `research_sessions` succeeds — proving the
savepoint rolled back the tolerant block without poisoning the outer tx.

Gated on `DATABASE_URL` (or Lakebase env) being present, so it only runs
in environments with a real Postgres — matches the gating of
`tests/integration/test_lakebase_user_upsert.py`.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

_HAS_DB = bool(
    os.environ.get("DATABASE_URL")
    or os.environ.get("LAKEBASE_INSTANCE_NAME")
)

pytestmark = pytest.mark.skipif(
    not _HAS_DB,
    reason="requires real Postgres; set DATABASE_URL or LAKEBASE_INSTANCE_NAME",
)

# Defer heavy imports until test bodies run, so collection succeeds when
# the skipif gate fires (which would otherwise load `deep_research.api.v1`
# whose __init__ eagerly validates Settings).
if _HAS_DB:
    from sqlalchemy import func, select, text
    from sqlalchemy.exc import IntegrityError
    from sqlalchemy.ext.asyncio import AsyncSession

    from deep_research.api.v1.jobs import _load_system_instructions
    from deep_research.models.research_session import ResearchSession


def _legacy_settings() -> SimpleNamespace:
    return SimpleNamespace(storage_service_impl="sqlalchemy_legacy")


@pytest.mark.asyncio
async def test_preferences_savepoint_preserves_outer_tx(
    db_session: AsyncSession,
) -> None:
    """The tolerant helper must not poison the outer transaction on failure.

    Uses a `user_id` that does not exist in the `users` table. The
    ``INSERT ... ON CONFLICT DO NOTHING`` inside `PreferencesService.
    get_preferences` (preferences_service.py:43-50) then raises an
    `IntegrityError` with SQLSTATE 23503 (FK violation). Without the
    savepoint, the outer transaction is aborted and the follow-up
    SELECT fails. With the savepoint, the helper returns None and the
    follow-up SELECT succeeds.
    """
    # 1) Autobegin the outer transaction with a benign statement so the
    #    savepoint wraps an existing tx (mirrors the real request flow
    #    where `verify_chat_access` has already issued a SELECT).
    await db_session.execute(text("SELECT 1"))

    # 2) Use a user_id that is guaranteed not to exist in `users`.
    ghost_user = "integ-ghost-user-that-does-not-exist-00000"

    # 3) Load system instructions via the tolerant helper. In legacy mode
    #    this runs the INSERT through the savepoint; an FK violation is
    #    expected (because no `users` row exists for `ghost_user`).
    result = await _load_system_instructions(
        db_session,
        _legacy_settings(),
        None,  # no storage stack — legacy path
        ghost_user,
    )
    # The helper must swallow the failure and return None.
    assert result is None

    # 4) **Regression check** — the exact production query that previously
    #    raised `InFailedSQLTransactionError` must now succeed. This is the
    #    one assertion that cannot be green without the savepoint.
    count = await db_session.scalar(
        select(func.count(ResearchSession.id))
    )
    assert count is not None  # query succeeded; session is not poisoned


@pytest.mark.asyncio
async def test_preferences_savepoint_rolls_back_only_inner(
    db_session: AsyncSession,
) -> None:
    """The savepoint must NOT roll back work done before the tolerant block.

    Issue a plain `SELECT` first, then a failing tolerant call, then
    verify the outer transaction is still alive (a new query succeeds
    and the session has not been closed/rolled back externally).
    """
    pre = await db_session.scalar(text("SELECT 42"))
    assert pre == 42

    ghost_user = "integ-ghost-user-savepoint-test-00001"
    _ = await _load_system_instructions(
        db_session, _legacy_settings(), None, ghost_user
    )

    post = await db_session.scalar(text("SELECT 43"))
    assert post == 43


@pytest.mark.asyncio
async def test_direct_insert_raises_without_savepoint(
    db_session: AsyncSession,
) -> None:
    """Sanity check — without the savepoint, the outer tx would die.

    This proves the test environment actually reproduces the production
    failure mode. We issue the same ``INSERT`` directly and confirm it
    raises `IntegrityError`; the point is that *without* the helper's
    savepoint, a subsequent statement on `db_session` would fail. We
    then explicitly roll back so pytest cleanup does not complain.
    """
    ghost_user = "integ-ghost-user-direct-00002"
    with pytest.raises(IntegrityError):
        await db_session.execute(
            text(
                "INSERT INTO user_preferences (user_id, default_research_depth, "
                "default_query_mode, theme, notifications_enabled) "
                "VALUES (:u, 'AUTO', 'simple', 'system', true) "
                "ON CONFLICT (user_id) DO NOTHING"
            ),
            {"u": ghost_user},
        )
    await db_session.rollback()
