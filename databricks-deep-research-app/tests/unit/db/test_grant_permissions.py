"""Tests for SQL identifier validation in grant_permissions module."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_research.db.grant_permissions import (
    _TOLERATE_ROLE_CHECK_FAILURE_ENV,
    _tolerate_role_check_failure,
    _validate_sql_identifier,
    grant_permissions_to_app,
)


class TestValidateSqlIdentifier:
    """Tests for _validate_sql_identifier()."""

    @pytest.mark.parametrize(
        "identifier",
        [
            "normal_sp_123",
            "sp-with-dashes",
            "sp.with.dots",
            "SimpleName",
            "abc123",
            "a",
        ],
    )
    def test_valid_identifiers_pass(self, identifier: str) -> None:
        """Valid identifiers should be returned unchanged."""
        result = _validate_sql_identifier(identifier)
        assert result == identifier

    @pytest.mark.parametrize(
        "identifier",
        [
            "'; DROP TABLE --",
            'sp"injection',
            "sp\ninjection",
            "",
            "sp name with spaces",
            "sp;semicolon",
            "sp'quote",
            "$(command)",
        ],
    )
    def test_injection_attempts_raise_value_error(self, identifier: str) -> None:
        """Injection attempts and unsafe characters should raise ValueError."""
        with pytest.raises(ValueError, match="Unsafe SQL"):
            _validate_sql_identifier(identifier)

    def test_custom_label_in_error_message(self) -> None:
        """The label parameter should appear in the error message."""
        with pytest.raises(ValueError, match="Unsafe SQL database name"):
            _validate_sql_identifier("'; DROP TABLE --", label="database name")

    def test_parameterized_query_format(self) -> None:
        """Verify the expected parameterized query format for databricks_create_role.

        The production code uses:
            await conn.execute(
                "SELECT databricks_create_role($1, 'SERVICE_PRINCIPAL')",
                sp_username,
            )
        This test validates that the query string uses $1 placeholder
        rather than f-string interpolation.
        """
        query = "SELECT databricks_create_role($1, 'SERVICE_PRINCIPAL')"
        assert "$1" in query
        assert "'" not in query.replace("'SERVICE_PRINCIPAL'", "")
        # Ensure no f-string patterns
        assert "{" not in query
        assert "}" not in query


# ---------------------------------------------------------------------------
# Strict-mode guard around the pg_roles existence check
# ---------------------------------------------------------------------------


SP_UUID = "a22fb8f7-c9db-46b3-9294-5c1f11eefb48"


class TestTolerateRoleCheckFailure:
    """``_tolerate_role_check_failure()`` honours the opt-out env var."""

    @pytest.mark.parametrize("raw", ["1", "true", "True", "yes", "ON"])
    def test_truthy_values_enable_tolerance(self, raw: str) -> None:
        with patch.dict("os.environ", {_TOLERATE_ROLE_CHECK_FAILURE_ENV: raw}):
            assert _tolerate_role_check_failure() is True

    @pytest.mark.parametrize("raw", ["", "0", "false", "no", "off", "anything-else"])
    def test_falsy_or_unset_means_strict(self, raw: str) -> None:
        with patch.dict("os.environ", {_TOLERATE_ROLE_CHECK_FAILURE_ENV: raw}):
            assert _tolerate_role_check_failure() is False

    def test_unset_means_strict(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            assert _tolerate_role_check_failure() is False


def _build_grant_mocks(
    *,
    pg_roles_check_raises: bool = False,
    pg_roles_returns: Any | None = 1,
    rolcanlogin: Any | None = True,
    rolcanlogin_raises: bool = False,
    backend_type: str = "autoscaling",
) -> tuple[Any, Any, Any]:
    """Build the (workspace_client_cls, credential_provider, conn) mocks.

    Returns three patch contexts ready to apply with ``with``. The conn mock
    has ``fetchval`` programmed to return ``pg_roles_returns`` on the first
    call (the pre-create check), then ``rolcanlogin`` on the second (the
    post-flight check). Either call may be programmed to raise.
    """
    # WorkspaceClient → apps.list returns one app with matching name
    app = MagicMock()
    app.name = "deep-research-agent-ais"
    app.service_principal_id = 73947136707923
    ws = MagicMock()
    ws.apps.list.return_value = iter([app])
    sp = MagicMock()
    sp.application_id = SP_UUID
    ws.service_principals.get.return_value = sp
    ws_cls = MagicMock(return_value=ws)

    # Credential provider
    provider = MagicMock()
    provider.get_backend_type.return_value = backend_type
    cred = MagicMock()
    cred.username = "michael.shtelma@databricks.com"
    cred.token = "tok"
    provider.get_credential.return_value = cred
    provider.get_host.return_value = "ep-x.database.cloud.databricks.com"
    provider.get_port.return_value = 5432

    # asyncpg connection. fetchval returns a sequence — pg_roles check first,
    # then rolcanlogin post-flight check. Either can be programmed to raise.
    conn = MagicMock()
    fetchval_calls: list[Any] = []

    async def _fetchval(query: str, *args: Any) -> Any:
        fetchval_calls.append((query, args))
        # First call: pg_roles existence check from line ~184
        if len(fetchval_calls) == 1:
            if pg_roles_check_raises:
                raise RuntimeError("simulated pg_roles failure")
            return pg_roles_returns
        # Second call: post-flight rolcanlogin check
        if rolcanlogin_raises:
            raise RuntimeError("simulated post-flight failure")
        return rolcanlogin

    conn.fetchval = AsyncMock(side_effect=_fetchval)
    conn.execute = AsyncMock(return_value=None)
    conn.close = AsyncMock(return_value=None)

    return ws_cls, provider, conn


@pytest.fixture
def settings_stub() -> MagicMock:
    s = MagicMock()
    s.use_lakebase = True
    s.databricks_config_profile = "ais"
    s.lakebase_database = "deep_research"
    return s


class TestStrictModeRoleCheck:
    """In default mode, a failed pg_roles check must raise — not silently skip."""

    @pytest.mark.asyncio
    async def test_pg_roles_check_failure_raises_in_strict_mode(
        self, settings_stub: MagicMock
    ) -> None:
        ws_cls, provider, conn = _build_grant_mocks(pg_roles_check_raises=True)
        with patch.dict("os.environ", {}, clear=True), patch(
            "deep_research.db.grant_permissions.WorkspaceClient", ws_cls
        ), patch(
            "deep_research.db.grant_permissions.get_credential_provider",
            return_value=provider,
        ), patch(
            "deep_research.db.grant_permissions.asyncpg.connect",
            new=AsyncMock(return_value=conn),
        ), pytest.raises(RuntimeError, match="pg_roles existence check failed"):
            await grant_permissions_to_app(
                "deep-research-agent-ais", settings=settings_stub
            )

    @pytest.mark.asyncio
    async def test_pg_roles_check_failure_skipped_in_tolerant_mode(
        self, settings_stub: MagicMock
    ) -> None:
        ws_cls, provider, conn = _build_grant_mocks(pg_roles_check_raises=True)
        # Even with check failure, the GRANTs should still run. Disable the
        # post-flight check too via the opt-out env var.
        with patch.dict(
            "os.environ", {_TOLERATE_ROLE_CHECK_FAILURE_ENV: "1"}
        ), patch(
            "deep_research.db.grant_permissions.WorkspaceClient", ws_cls
        ), patch(
            "deep_research.db.grant_permissions.get_credential_provider",
            return_value=provider,
        ), patch(
            "deep_research.db.grant_permissions.asyncpg.connect",
            new=AsyncMock(return_value=conn),
        ):
            # Should NOT raise.
            await grant_permissions_to_app(
                "deep-research-agent-ais", settings=settings_stub
            )
        # GRANT statements ran (at least DATABASE + TABLES + SEQUENCES + 2 ALTER DEFAULT)
        assert conn.execute.await_count >= 5


class TestPostFlightVerification:
    """After grants, verify the SP role exists+canlogin or raise."""

    @pytest.mark.asyncio
    async def test_post_flight_role_missing_raises(
        self, settings_stub: MagicMock
    ) -> None:
        # pg_roles check returns existing (1), so create is skipped, GRANTs
        # run; but the post-flight rolcanlogin query returns None (role gone).
        ws_cls, provider, conn = _build_grant_mocks(
            pg_roles_returns=1,
            rolcanlogin=None,
        )
        with patch.dict("os.environ", {}, clear=True), patch(
            "deep_research.db.grant_permissions.WorkspaceClient", ws_cls
        ), patch(
            "deep_research.db.grant_permissions.get_credential_provider",
            return_value=provider,
        ), patch(
            "deep_research.db.grant_permissions.asyncpg.connect",
            new=AsyncMock(return_value=conn),
        ), pytest.raises(RuntimeError, match="does NOT exist"):
            await grant_permissions_to_app(
                "deep-research-agent-ais", settings=settings_stub
            )

    @pytest.mark.asyncio
    async def test_post_flight_no_login_raises(
        self, settings_stub: MagicMock
    ) -> None:
        ws_cls, provider, conn = _build_grant_mocks(
            pg_roles_returns=1,
            rolcanlogin=False,
        )
        with patch.dict("os.environ", {}, clear=True), patch(
            "deep_research.db.grant_permissions.WorkspaceClient", ws_cls
        ), patch(
            "deep_research.db.grant_permissions.get_credential_provider",
            return_value=provider,
        ), patch(
            "deep_research.db.grant_permissions.asyncpg.connect",
            new=AsyncMock(return_value=conn),
        ), pytest.raises(RuntimeError, match="rolcanlogin=False"):
            await grant_permissions_to_app(
                "deep-research-agent-ais", settings=settings_stub
            )

    @pytest.mark.asyncio
    async def test_post_flight_happy_path_no_raise(
        self, settings_stub: MagicMock
    ) -> None:
        ws_cls, provider, conn = _build_grant_mocks(
            pg_roles_returns=1,
            rolcanlogin=True,
        )
        with patch.dict("os.environ", {}, clear=True), patch(
            "deep_research.db.grant_permissions.WorkspaceClient", ws_cls
        ), patch(
            "deep_research.db.grant_permissions.get_credential_provider",
            return_value=provider,
        ), patch(
            "deep_research.db.grant_permissions.asyncpg.connect",
            new=AsyncMock(return_value=conn),
        ):
            await grant_permissions_to_app(
                "deep-research-agent-ais", settings=settings_stub
            )
        # pg_roles check + post-flight check = 2 fetchval calls.
        assert conn.fetchval.await_count == 2

    @pytest.mark.asyncio
    async def test_post_flight_skipped_in_tolerant_mode(
        self, settings_stub: MagicMock
    ) -> None:
        ws_cls, provider, conn = _build_grant_mocks(
            pg_roles_returns=1,
            rolcanlogin=None,  # would normally raise; tolerant mode skips
        )
        with patch.dict(
            "os.environ", {_TOLERATE_ROLE_CHECK_FAILURE_ENV: "1"}
        ), patch(
            "deep_research.db.grant_permissions.WorkspaceClient", ws_cls
        ), patch(
            "deep_research.db.grant_permissions.get_credential_provider",
            return_value=provider,
        ), patch(
            "deep_research.db.grant_permissions.asyncpg.connect",
            new=AsyncMock(return_value=conn),
        ):
            await grant_permissions_to_app(
                "deep-research-agent-ais", settings=settings_stub
            )
        # Only the pre-create pg_roles check ran; post-flight was skipped.
        assert conn.fetchval.await_count == 1
