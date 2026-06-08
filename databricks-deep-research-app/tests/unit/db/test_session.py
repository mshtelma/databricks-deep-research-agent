"""Tests for database session management."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_research.db import session


class TestDisposeEngineAsync:
    """Tests for async engine disposal helper."""

    @pytest.mark.asyncio
    async def test_dispose_engine_success(self) -> None:
        """Should dispose engine successfully."""
        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()

        await session._dispose_engine_async(mock_engine)

        mock_engine.dispose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_dispose_engine_handles_error(self) -> None:
        """Should log warning and not raise on disposal error."""
        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock(side_effect=Exception("Disposal failed"))

        # Should not raise
        await session._dispose_engine_async(mock_engine)

        mock_engine.dispose.assert_awaited_once()


class TestEngineDisposalTracking:
    """Tests for engine disposal task lifecycle (fire-and-forget)."""

    def setup_method(self) -> None:
        """Reset module state before each test."""
        session._engine = None
        session._async_session_maker = None
        session._credential_provider = None

    def teardown_method(self) -> None:
        """Clean up module state after each test."""
        session._engine = None
        session._async_session_maker = None
        session._credential_provider = None

    @pytest.mark.asyncio
    async def test_disposal_task_tracked_and_removed(self) -> None:
        """Verify fire-and-forget disposal task completes and disposes engine."""
        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()

        loop = asyncio.get_running_loop()

        task = loop.create_task(session._dispose_engine_async(mock_engine))

        await task

        mock_engine.dispose.assert_awaited_once()
        assert task.done()

    @pytest.mark.asyncio
    async def test_multiple_disposal_tasks_tracked(self) -> None:
        """Verify multiple concurrent disposal tasks all complete."""
        mock_engines = [AsyncMock() for _ in range(3)]
        for engine in mock_engines:
            engine.dispose = AsyncMock()

        loop = asyncio.get_running_loop()

        tasks = []
        for engine in mock_engines:
            task = loop.create_task(session._dispose_engine_async(engine))
            tasks.append(task)

        await asyncio.gather(*tasks)

        for engine in mock_engines:
            engine.dispose.assert_awaited_once()
        for task in tasks:
            assert task.done()


class TestGetEngineTokenRefresh:
    """Tests for token refresh logic in get_engine."""

    def setup_method(self) -> None:
        """Reset module state before each test."""
        session._engine = None
        session._async_session_maker = None
        session._credential_provider = None

    def teardown_method(self) -> None:
        """Clean up module state after each test."""
        session._engine = None
        session._async_session_maker = None
        session._credential_provider = None

    @pytest.mark.asyncio
    async def test_engine_disposal_on_token_expiry(self) -> None:
        """Verify engine is properly disposed when token expires."""
        # Create mock credential provider with expired token
        mock_provider = MagicMock()
        mock_cred = MagicMock()
        mock_cred.is_expired = True
        mock_provider.current_credential = mock_cred
        mock_provider.get_credential = MagicMock()
        mock_provider.build_connection_url = MagicMock(
            return_value="postgresql+asyncpg://user:pass@localhost/db"
        )

        # Create mock engine
        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()

        # Set up module state
        session._credential_provider = mock_provider
        session._engine = mock_engine
        session._async_session_maker = MagicMock()

        # Mock settings for Lakebase
        mock_settings = MagicMock()
        mock_settings.use_lakebase = True
        mock_settings.debug = False
        mock_settings.database_url = None

        # Patch create_async_engine to avoid actual DB connection
        with patch.object(session, "create_async_engine") as mock_create_engine:
            new_mock_engine = MagicMock()
            mock_create_engine.return_value = new_mock_engine

            # Call get_engine which should detect expired token and dispose
            engine = session.get_engine(mock_settings)

            # Verify credential refresh was triggered
            mock_provider.get_credential.assert_called_once_with(force_refresh=True)

            # Verify new engine was created
            mock_create_engine.assert_called_once()
            assert engine == new_mock_engine

            # Old engine should have been scheduled for disposal
            # Wait briefly to allow background task to complete
            await asyncio.sleep(0.1)


class TestResetEngine:
    """Tests for module-level state reset."""

    def setup_method(self) -> None:
        """Reset module state before each test."""
        session._engine = None
        session._async_session_maker = None

    def teardown_method(self) -> None:
        """Clean up module state after each test."""
        session._engine = None
        session._async_session_maker = None

    @pytest.mark.asyncio
    async def test_close_db_disposes_engine(self) -> None:
        """Verify close_db properly disposes the engine."""
        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()

        session._engine = mock_engine
        session._async_session_maker = MagicMock()

        await session.close_db()

        mock_engine.dispose.assert_awaited_once()
        assert session._engine is None
        assert session._async_session_maker is None


class TestGetEnginePgBouncerSafety:
    """Regression: engine must be built with PgBouncer-safe kwargs.

    Prevents a reintroduction of the asyncpg `prepare()` hang described in
    the plan at .claude/plans/implement-in-deep-research-*.md.
    """

    def setup_method(self) -> None:
        session._engine = None
        session._async_session_maker = None
        session._credential_provider = None

    def teardown_method(self) -> None:
        session._engine = None
        session._async_session_maker = None
        session._credential_provider = None

    def _capture_create_engine(self, monkeypatch) -> dict:
        """Monkeypatch create_async_engine and return a dict that receives kwargs."""
        captured: dict = {}

        def fake_create_async_engine(url, **kwargs):
            captured["url"] = url
            captured.update(kwargs)
            engine = MagicMock()
            engine.dispose = AsyncMock()
            return engine

        monkeypatch.setattr(session, "create_async_engine", fake_create_async_engine)
        return captured

    def test_lakebase_engine_has_all_pgbouncer_safe_kwargs(
        self, monkeypatch, force_name_func_supported
    ) -> None:
        """Lakebase engine must pass all PgBouncer-safe connect_args and must NOT
        pass prepared_statement_cache_size at the engine level (the asyncpg
        dialect rejects it with TypeError at create_async_engine() time)."""
        captured = self._capture_create_engine(monkeypatch)

        settings = MagicMock()
        settings.use_lakebase = True
        settings.debug = False
        settings.is_production = True
        settings.db_pool_size = 10
        settings.db_max_overflow = 20
        settings.db_command_timeout = 60.0
        # Force database_url path (no credential provider) to keep the test hermetic.
        settings.database_url = "postgresql+asyncpg://u:p@localhost/db"

        # Patch get_credential_provider to return None so we go through the URL path.
        monkeypatch.setattr(session, "get_credential_provider", lambda s: None)
        monkeypatch.setattr(
            session,
            "get_database_url",
            lambda s: "postgresql+asyncpg://u:p@localhost/db",
        )

        session.get_engine(settings)

        # Regression guard: engine-level prepared_statement_cache_size MUST NOT
        # be passed — the asyncpg dialect rejects it with TypeError at
        # create_async_engine() time. All cache control flows through connect_args.
        assert "prepared_statement_cache_size" not in captured
        # asyncpg-level cache disabled.
        assert captured["connect_args"]["statement_cache_size"] == 0
        # Force unnamed prepared statements — the critical fix for PgBouncer.
        assert captured["connect_args"]["prepared_statement_name_func"]() == ""
        # command_timeout prevents wire-protocol hangs.
        assert captured["connect_args"]["command_timeout"] == 60.0
        # SSL still required.
        assert captured["connect_args"]["ssl"] is True
        # Pool recycle tightened for Lakebase OAuth.
        assert captured["pool_recycle"] == 2700

    def test_non_lakebase_engine_leaves_cache_defaults(self, monkeypatch) -> None:
        """Plain Postgres deployments must not pay the prepare-on-every-query cost."""
        captured = self._capture_create_engine(monkeypatch)

        settings = MagicMock()
        settings.use_lakebase = False
        settings.debug = False
        settings.is_production = False
        settings.db_pool_size = 10
        settings.db_max_overflow = 20
        settings.db_command_timeout = 60.0
        settings.database_url = "postgresql+asyncpg://u:p@localhost/db"

        monkeypatch.setattr(session, "get_credential_provider", lambda s: None)
        monkeypatch.setattr(
            session,
            "get_database_url",
            lambda s: "postgresql+asyncpg://u:p@localhost/db",
        )

        session.get_engine(settings)

        # No cache overrides — asyncpg keeps its default caching behavior.
        assert "prepared_statement_cache_size" not in captured
        assert "statement_cache_size" not in captured["connect_args"]
        assert "prepared_statement_name_func" not in captured["connect_args"]
        assert "command_timeout" not in captured["connect_args"]
        # pool_recycle falls back to 3600 for non-Lakebase.
        assert captured["pool_recycle"] == 3600


class TestIsStaleConnectionError:
    """Tests for the stale-connection detector used by ``get_db`` cleanup."""

    def test_detects_interface_error_connection_closed(self) -> None:
        import sqlalchemy.exc

        err = sqlalchemy.exc.InterfaceError(
            None,
            None,
            Exception(
                "cannot call Transaction.commit(): the underlying connection is closed"
            ),
        )
        assert session._is_stale_connection_error(err) is True

    def test_detects_operational_error_connection_lost(self) -> None:
        import sqlalchemy.exc

        err = sqlalchemy.exc.OperationalError(
            None, None, Exception("server closed the connection is lost")
        )
        assert session._is_stale_connection_error(err) is True

    def test_rejects_unrelated_interface_error(self) -> None:
        import sqlalchemy.exc

        err = sqlalchemy.exc.InterfaceError(
            None, None, Exception("some unrelated DB error")
        )
        assert session._is_stale_connection_error(err) is False

    def test_rejects_non_sqlalchemy_errors(self) -> None:
        # Plain Python exceptions with a matching string must not qualify —
        # the detector only fires for DB driver errors.
        assert session._is_stale_connection_error(ValueError("connection is closed")) is False
        assert session._is_stale_connection_error(RuntimeError()) is False


class TestIsDbAuthError:
    """Tests for Lakebase auth-failure detection."""

    def test_detects_asyncpg_password_auth_failure_message(self) -> None:
        assert (
            session._is_db_auth_error(
                RuntimeError(
                    "password authentication failed for user "
                    "'a22fb8f7-c9db-46b3-9294-5c1f11eefb48'"
                )
            )
            is True
        )

    def test_detects_invalid_password_class_in_cause_chain(self) -> None:
        InvalidPasswordError = type("InvalidPasswordError", (Exception,), {})
        err = RuntimeError("wrapped storage error")
        err.__cause__ = InvalidPasswordError("server rejected credential")

        assert session._is_db_auth_error(err) is True

    def test_rejects_non_auth_error(self) -> None:
        assert session._is_db_auth_error(RuntimeError("connection is closed")) is False


class TestGetDbStaleConnectionCleanup:
    """Verify ``get_db`` distinguishes safe-to-swallow stale-connection
    cleanup from data-loss-bearing stale-connection.

    For SSE-stream endpoints with NO pending writes, swallowing the
    cleanup-time InterfaceError is correct — the request already
    succeeded.

    For mutating endpoints that left pending writes on the session, a
    cleanup-time stale connection means data was lost; we now raise 503
    rather than silently lying to the caller.
    """

    @pytest.mark.asyncio
    async def test_commit_connection_closed_is_swallowed_when_no_pending_writes(self) -> None:
        import sqlalchemy.exc

        stale = sqlalchemy.exc.InterfaceError(
            None, None, Exception("the underlying connection is closed")
        )
        mock_session = MagicMock()
        # No pending writes — typical SSE-stream cleanup case.
        mock_session.new = []
        mock_session.dirty = []
        mock_session.deleted = []
        mock_session.commit = AsyncMock(side_effect=stale)
        mock_session.close = AsyncMock()
        mock_session.rollback = AsyncMock()

        mock_maker = MagicMock()
        mock_maker.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_maker.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch.object(session, "get_session_maker", return_value=mock_maker):
            gen = session.get_db()
            db = await gen.__anext__()
            assert db is mock_session
            # Exiting the generator runs cleanup; stale connection on a
            # session with no pending writes must NOT propagate.
            with pytest.raises(StopAsyncIteration):
                await gen.__anext__()

        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_commit_connection_closed_with_pending_writes_raises_503(self) -> None:
        """Stale connection at cleanup with pending writes = data loss.
        Endpoint that relied on the implicit cleanup commit must surface
        a real failure rather than return a misleading 200."""
        import sqlalchemy.exc
        from fastapi import HTTPException

        stale = sqlalchemy.exc.InterfaceError(
            None, None, Exception("the underlying connection is closed")
        )
        mock_session = MagicMock()
        # Session has pending INSERT — write would be lost on stale conn.
        mock_session.new = [object()]
        mock_session.dirty = []
        mock_session.deleted = []
        mock_session.commit = AsyncMock(side_effect=stale)
        mock_session.close = AsyncMock()
        mock_session.rollback = AsyncMock()

        mock_maker = MagicMock()
        mock_maker.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_maker.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch.object(session, "get_session_maker", return_value=mock_maker):
            gen = session.get_db()
            await gen.__anext__()
            with pytest.raises(HTTPException) as exc_info:
                await gen.__anext__()
            assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_real_commit_error_still_raises(self) -> None:
        import sqlalchemy.exc

        real_err = sqlalchemy.exc.IntegrityError(
            None, None, Exception("unique constraint violation")
        )
        mock_session = MagicMock()
        mock_session.new = []
        mock_session.dirty = []
        mock_session.deleted = []
        mock_session.commit = AsyncMock(side_effect=real_err)
        mock_session.close = AsyncMock()
        mock_session.rollback = AsyncMock()

        mock_maker = MagicMock()
        mock_maker.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_maker.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch.object(session, "get_session_maker", return_value=mock_maker):
            gen = session.get_db()
            await gen.__anext__()
            # Real errors still bubble up so callers can surface them.
            with pytest.raises(sqlalchemy.exc.IntegrityError):
                await gen.__anext__()

        mock_session.rollback.assert_awaited_once()
