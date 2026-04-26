"""Tests for PgBouncer-safe asyncpg configuration helpers."""

from unittest.mock import MagicMock

from deep_research.db.asyncpg_config import (
    _unnamed_prepared_statement,
    lakebase_asyncpg_connect_args,
    lakebase_engine_kwargs,
    lakebase_raw_asyncpg_kwargs,
)


def _lakebase_settings(command_timeout: float | None = 60.0) -> MagicMock:
    """Build a minimal fake Settings where use_lakebase=True.

    We avoid constructing a real Settings() because pydantic-settings reads
    from .env/env-vars and would pick up developer-local config. A stand-in
    object with the two attributes this helper cares about is sufficient.
    """
    settings = MagicMock()
    settings.use_lakebase = True
    settings.db_command_timeout = command_timeout
    return settings


def _non_lakebase_settings() -> MagicMock:
    settings = MagicMock()
    settings.use_lakebase = False
    settings.db_command_timeout = 60.0
    return settings


class TestUnnamedPreparedStatement:
    """The hook that forces PgBouncer-safe unnamed prepared statements."""

    def test_returns_empty_string(self) -> None:
        assert _unnamed_prepared_statement() == ""

    def test_is_stateless_and_deterministic(self) -> None:
        # asyncpg may call it per-connection; must never drift.
        assert _unnamed_prepared_statement() == ""
        assert _unnamed_prepared_statement() == ""


class TestLakebaseAsyncpgConnectArgs:
    """SQLAlchemy `connect_args` for Lakebase-backed engines."""

    def test_non_lakebase_returns_empty_dict(self) -> None:
        assert lakebase_asyncpg_connect_args(_non_lakebase_settings()) == {}

    def test_lakebase_sets_all_pgbouncer_safe_keys(
        self, force_name_func_supported
    ) -> None:
        args = lakebase_asyncpg_connect_args(_lakebase_settings())
        assert args["ssl"] is True
        assert args["statement_cache_size"] == 0
        assert args["command_timeout"] == 60.0
        # Force unnamed prepared statements — this is the critical fix.
        assert args["prepared_statement_name_func"]() == ""

    def test_lakebase_omits_name_func_on_unsupported_asyncpg(
        self, force_name_func_unsupported
    ) -> None:
        # Older asyncpg wheels raise TypeError if prepared_statement_name_func
        # is passed to connect(); the builder must omit it. statement_cache_size=0
        # alone still prevents DuplicatePreparedStatementError under PgBouncer.
        args = lakebase_asyncpg_connect_args(_lakebase_settings())
        assert "prepared_statement_name_func" not in args
        assert args["statement_cache_size"] == 0
        assert args["ssl"] is True

    def test_command_timeout_propagates_from_settings(self) -> None:
        args = lakebase_asyncpg_connect_args(_lakebase_settings(command_timeout=15.0))
        assert args["command_timeout"] == 15.0

    def test_command_timeout_none_passes_through(self) -> None:
        args = lakebase_asyncpg_connect_args(_lakebase_settings(command_timeout=None))
        assert args["command_timeout"] is None


class TestLakebaseEngineKwargs:
    """SQLAlchemy engine-level kwargs (separate cache layer from asyncpg)."""

    def test_non_lakebase_returns_empty_dict(self) -> None:
        assert lakebase_engine_kwargs(_non_lakebase_settings()) == {}

    def test_lakebase_returns_empty_dict(self) -> None:
        # Regression guard: the asyncpg dialect rejects prepared_statement_cache_size
        # at create_async_engine() time with:
        #   TypeError: Invalid argument(s) 'prepared_statement_cache_size' sent
        #   to create_engine(), using configuration PGDialect_asyncpg/...
        # Engine-level cache control is unavailable on this dialect; all cache
        # control flows through connect_args["statement_cache_size"]=0.
        # An empty dict keeps **engine_kwargs splats safe in session.py,
        # migrations/env.py, and bootstrap.py.
        assert lakebase_engine_kwargs(_lakebase_settings()) == {}


class TestLakebaseRawAsyncpgKwargs:
    """Kwargs for raw `asyncpg.connect()` in bootstrap paths."""

    def test_contains_pgbouncer_safe_keys(self, force_name_func_supported) -> None:
        kwargs = lakebase_raw_asyncpg_kwargs()
        assert kwargs["statement_cache_size"] == 0
        assert kwargs["prepared_statement_name_func"]() == ""

    def test_omits_name_func_on_unsupported_asyncpg(
        self, force_name_func_unsupported
    ) -> None:
        kwargs = lakebase_raw_asyncpg_kwargs()
        assert "prepared_statement_name_func" not in kwargs
        assert kwargs["statement_cache_size"] == 0
