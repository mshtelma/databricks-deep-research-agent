"""Unit tests for the reusable OBO/SP WorkspaceClient resolver."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from databricks_deep_research.core import databricks_auth


def _sp(host: str | None = "https://wsp.example.databricks.com") -> MagicMock:
    sp = MagicMock(name="sp_client")
    sp.config.host = host
    return sp


def test_resolve_builds_obo_client_when_token_present() -> None:
    sp = _sp()
    with patch.object(databricks_auth, "WorkspaceClient") as wc:
        out = databricks_auth.resolve_workspace_client(
            sp_client=sp, user_token="user-tok"
        )
    wc.assert_called_once_with(
        host="https://wsp.example.databricks.com",
        token="user-tok",
        auth_type="pat",
    )
    assert out is wc.return_value
    assert out is not sp


def test_resolve_returns_sp_when_no_token() -> None:
    sp = _sp()
    with patch.object(databricks_auth, "WorkspaceClient") as wc:
        out = databricks_auth.resolve_workspace_client(sp_client=sp, user_token=None)
    wc.assert_not_called()
    assert out is sp


def test_resolve_returns_none_when_no_client_and_no_token() -> None:
    assert (
        databricks_auth.resolve_workspace_client(sp_client=None, user_token=None)
        is None
    )


def test_resolve_returns_none_when_no_client_even_with_token() -> None:
    # No SP client → no host to derive → cannot build OBO; returns None.
    assert (
        databricks_auth.resolve_workspace_client(sp_client=None, user_token="tok")
        is None
    )


def test_resolve_falls_back_to_sp_when_host_unresolved() -> None:
    sp = _sp(host=None)
    with patch.object(databricks_auth, "WorkspaceClient") as wc:
        out = databricks_auth.resolve_workspace_client(sp_client=sp, user_token="tok")
    wc.assert_not_called()
    assert out is sp


def test_build_obo_forces_pat_auth() -> None:
    with patch.object(databricks_auth, "WorkspaceClient") as wc:
        out = databricks_auth.build_obo_workspace_client(
            host="https://h", user_token="t"
        )
    wc.assert_called_once_with(host="https://h", token="t", auth_type="pat")
    assert out is wc.return_value
