"""Tests for the Lakebase connection-birth auth-retry creator.

A freshly-minted Lakebase OAuth credential can be transiently rejected
("password authentication failed") by the PgBouncer/databricks_auth layer until
it propagates. ``session._make_lakebase_async_creator`` retries the SAME token
with bounded backoff instead of minting a new (equally un-propagated) token.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_research.db import session
from deep_research.db.credential_provider import LakebaseCredential

AUTH_ERR = "password authentication failed for user 'a22fb8f7'"


def _provider(token: str = "tok-AAA", expired: bool = False) -> MagicMock:
    cred = MagicMock()
    cred.username = "sp-uuid"
    cred.token = token
    cred.is_expired = expired
    cred.age_s = 0.5
    provider = MagicMock()
    provider.get_credential = MagicMock(return_value=cred)
    provider.current_credential = cred
    provider.get_host = MagicMock(return_value="h")
    provider.get_port = MagicMock(return_value=5432)
    provider.get_database = MagicMock(return_value="deep_research")
    return provider


def _settings(attempts: int = 5, base: float = 0.0, cap: float = 0.0) -> MagicMock:
    s = MagicMock()
    s.lakebase_auth_retry_attempts = attempts
    s.lakebase_auth_retry_base_delay_s = base
    s.lakebase_auth_retry_max_delay_s = cap
    s.db_command_timeout = 60.0
    return s


class TestLakebaseConnectKwargs:
    def test_includes_pgbouncer_safe_options(self) -> None:
        kwargs = session._lakebase_connect_kwargs(_settings(), _provider(token="tok"))
        assert kwargs["host"] == "h"
        assert kwargs["port"] == 5432
        assert kwargs["user"] == "sp-uuid"
        assert kwargs["password"] == "tok"
        assert kwargs["database"] == "deep_research"
        assert kwargs["ssl"] is True
        # PgBouncer transaction-pooling safety must survive the creator path.
        assert kwargs["statement_cache_size"] == 0
        assert kwargs["command_timeout"] == 60.0


class TestLakebaseAsyncCreatorRetry:
    @pytest.mark.asyncio
    async def test_success_first_attempt_no_retry(self) -> None:
        conn = object()
        connect = AsyncMock(return_value=conn)
        with patch.object(session.asyncpg, "connect", new=connect):
            creator = session._make_lakebase_async_creator(_settings(), _provider())
            result = await creator()
        assert result is conn
        assert connect.await_count == 1

    @pytest.mark.asyncio
    async def test_retries_then_succeeds_reusing_same_token(self) -> None:
        conn = object()
        connect = AsyncMock(
            side_effect=[Exception(AUTH_ERR), Exception(AUTH_ERR), conn]
        )
        with (
            patch.object(session.asyncpg, "connect", new=connect),
            patch.object(session, "log_lakebase_auth_failure_diagnostics"),
            patch.object(session, "_maybe_log_endpoint_state"),
        ):
            creator = session._make_lakebase_async_creator(
                _settings(attempts=5), _provider(token="tok-SAME")
            )
            result = await creator()
        assert result is conn
        assert connect.await_count == 3
        # Every attempt presented the SAME token — the whole point of the retry.
        passwords = {call.kwargs["password"] for call in connect.await_args_list}
        assert passwords == {"tok-SAME"}

    @pytest.mark.asyncio
    async def test_diagnostics_and_endpoint_probe_logged_once(self) -> None:
        conn = object()
        connect = AsyncMock(side_effect=[Exception(AUTH_ERR), conn])
        with (
            patch.object(session.asyncpg, "connect", new=connect),
            patch.object(session, "log_lakebase_auth_failure_diagnostics") as diag,
            patch.object(session, "_maybe_log_endpoint_state") as probe,
        ):
            creator = session._make_lakebase_async_creator(_settings(), _provider())
            await creator()
        diag.assert_called_once()
        probe.assert_called_once()

    @pytest.mark.asyncio
    async def test_exhausts_budget_then_raises(self) -> None:
        connect = AsyncMock(side_effect=Exception(AUTH_ERR))
        with (
            patch.object(session.asyncpg, "connect", new=connect),
            patch.object(session, "log_lakebase_auth_failure_diagnostics"),
            patch.object(session, "_maybe_log_endpoint_state"),
        ):
            creator = session._make_lakebase_async_creator(
                _settings(attempts=3), _provider()
            )
            with pytest.raises(Exception) as exc_info:
                await creator()
        assert "authentication failed" in str(exc_info.value).lower()
        assert connect.await_count == 3

    @pytest.mark.asyncio
    async def test_non_auth_error_is_not_retried(self) -> None:
        connect = AsyncMock(side_effect=Exception("connection refused"))
        with patch.object(session.asyncpg, "connect", new=connect):
            creator = session._make_lakebase_async_creator(
                _settings(attempts=5), _provider()
            )
            with pytest.raises(Exception) as exc_info:
                await creator()
        assert "connection refused" in str(exc_info.value)
        assert connect.await_count == 1

    @pytest.mark.asyncio
    async def test_backoff_capped(self) -> None:
        """Delays respect the cap and are bounded by attempts-1 sleeps."""
        conn = object()
        connect = AsyncMock(side_effect=[Exception(AUTH_ERR)] * 3 + [conn])
        sleeps: list[float] = []

        async def fake_sleep(d: float) -> None:
            sleeps.append(d)

        with (
            patch.object(session.asyncpg, "connect", new=connect),
            patch.object(session.asyncio, "sleep", new=fake_sleep),
            patch.object(session, "log_lakebase_auth_failure_diagnostics"),
            patch.object(session, "_maybe_log_endpoint_state"),
        ):
            creator = session._make_lakebase_async_creator(
                _settings(attempts=5, base=0.25, cap=1.0), _provider()
            )
            await creator()
        # 3 failures before success → 3 backoff sleeps: 0.25, 0.5, 1.0 (capped).
        assert sleeps == [0.25, 0.5, 1.0]


class TestLakebaseCredentialAge:
    def test_age_none_without_issued_at(self) -> None:
        cred = LakebaseCredential(
            token="t", username="u", expires_at=datetime.now(UTC) + timedelta(hours=1)
        )
        assert cred.age_s is None

    def test_age_positive_with_issued_at(self) -> None:
        cred = LakebaseCredential(
            token="t",
            username="u",
            expires_at=datetime.now(UTC) + timedelta(hours=1),
            issued_at=datetime.now(UTC) - timedelta(seconds=2),
        )
        age = cred.age_s
        assert age is not None and age >= 1.5
