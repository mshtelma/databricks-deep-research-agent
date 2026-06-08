"""Unit tests for ``core/databricks_auth.get_user_workspace_client`` —
specifically the regression where the Databricks Apps runtime sets
``DATABRICKS_CLIENT_ID``/``DATABRICKS_CLIENT_SECRET`` env vars and the
SDK detected BOTH the explicit PAT kwarg AND env-OAuth, raising
``ValueError: more than one authorization method configured``.

The fix is the ``auth_type="pat"`` kwarg. These tests pin that contract
so the regression cannot return silently.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from deep_research.core.databricks_auth import (
    clear_databricks_auth,
    get_user_workspace_client,
)


def _make_request(obo_header: str | None) -> MagicMock:
    """Build a minimal ``Request``-like mock for the unit under test."""
    request = MagicMock()
    headers = {}
    if obo_header is not None:
        headers["X-Forwarded-Access-Token"] = obo_header
    request.headers = headers
    # request.url is only used when host can't be resolved any other way;
    # we always set DATABRICKS_HOST in tests so this is just a safety stub.
    request.url = MagicMock(scheme="https", netloc="example.cloud.databricks.com")
    return request


@pytest.fixture(autouse=True)
def reset_auth_singleton() -> None:
    """Ensure the singleton DatabricksAuth doesn't leak between tests."""
    clear_databricks_auth()
    yield
    clear_databricks_auth()


@pytest.fixture
def fake_settings(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Return a stub ``Settings`` so the unit-under-test does not have to
    satisfy the real Pydantic Settings validator (which requires Lakebase
    env config to instantiate). The unit only reads ``is_databricks_app``.
    """
    stub = MagicMock()
    stub.is_databricks_app = False
    monkeypatch.setattr(
        "deep_research.core.databricks_auth.get_settings", lambda: stub
    )
    return stub


class TestAppsRuntimeOauthConflict:
    """The Databricks Apps runtime sets OAuth env vars unconditionally.
    Without auth_type="pat" the SDK raises ValueError on Config validation."""

    @patch("deep_research.core.databricks_auth.WorkspaceClient")
    def test_passes_auth_type_pat_to_workspace_client(
        self,
        mock_wc_class: MagicMock,
        fake_settings: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """REGRESSION TEST for T2: when Apps env vars are set, the
        construction call must include ``auth_type='pat'``. Without it,
        the SDK raises ``ValueError: more than one authorization method``.
        """
        fake_settings.is_databricks_app = True
        monkeypatch.setenv("DATABRICKS_HOST", "https://example.cloud.databricks.com")
        monkeypatch.setenv("DATABRICKS_CLIENT_ID", "test-client-id")
        monkeypatch.setenv("DATABRICKS_CLIENT_SECRET", "test-client-secret")

        mock_wc_class.return_value = MagicMock()
        request = _make_request("obo-token-abc")

        get_user_workspace_client(request)

        assert mock_wc_class.call_count == 1
        kwargs = mock_wc_class.call_args.kwargs
        assert kwargs.get("auth_type") == "pat", (
            "auth_type='pat' must be passed to WorkspaceClient. "
            "Without it, the Databricks Apps runtime's OAuth env vars "
            "conflict with the explicit PAT kwarg and the SDK raises."
        )
        assert kwargs.get("token") == "obo-token-abc"
        assert kwargs.get("host") == "https://example.cloud.databricks.com"

    @patch("deep_research.core.databricks_auth.WorkspaceClient")
    def test_construct_does_not_raise_in_apps_env(
        self,
        mock_wc_class: MagicMock,
        fake_settings: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Smoke: construction returns the mocked instance without raising
        when Apps env vars are set."""
        fake_settings.is_databricks_app = True
        monkeypatch.setenv("DATABRICKS_HOST", "https://example.cloud.databricks.com")
        monkeypatch.setenv("DATABRICKS_CLIENT_ID", "id")
        monkeypatch.setenv("DATABRICKS_CLIENT_SECRET", "secret")

        instance = MagicMock()
        mock_wc_class.return_value = instance
        request = _make_request("obo-xyz")
        result = get_user_workspace_client(request)
        assert result is instance


class TestEmptyObOHeaderGuard:
    """Empty string OBO header used to slip past the ``is None`` check and
    reach the SDK with token="". T2's `if not obo_token` guard fixes this."""

    def test_empty_obo_header_in_apps_runtime_raises_401(
        self, fake_settings: MagicMock
    ) -> None:
        fake_settings.is_databricks_app = True
        request = _make_request("")  # header present but empty

        with pytest.raises(HTTPException) as exc_info:
            get_user_workspace_client(request)
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail["error_kind"] == "missing_obo_token"

    def test_missing_obo_header_in_apps_runtime_raises_401(
        self, fake_settings: MagicMock
    ) -> None:
        fake_settings.is_databricks_app = True
        request = _make_request(None)  # no header at all

        with pytest.raises(HTTPException) as exc_info:
            get_user_workspace_client(request)
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail["error_kind"] == "missing_obo_token"


class TestLocalDevFallback:
    """When NOT in Databricks Apps and no OBO header, fall back to SP."""

    @patch("deep_research.core.databricks_auth.get_databricks_auth")
    def test_no_obo_local_dev_falls_back_to_sp_client(
        self,
        mock_get_auth: MagicMock,
        fake_settings: MagicMock,
    ) -> None:
        fake_settings.is_databricks_app = False
        sp_client = MagicMock()
        mock_get_auth.return_value.get_client.return_value = sp_client

        request = _make_request(None)
        result = get_user_workspace_client(request)
        assert result is sp_client

    @patch("deep_research.core.databricks_auth.get_databricks_auth")
    def test_empty_obo_local_dev_falls_back_to_sp_client(
        self,
        mock_get_auth: MagicMock,
        fake_settings: MagicMock,
    ) -> None:
        """Empty-string header in local dev also falls back to SP."""
        fake_settings.is_databricks_app = False
        sp_client = MagicMock()
        mock_get_auth.return_value.get_client.return_value = sp_client

        request = _make_request("")
        result = get_user_workspace_client(request)
        assert result is sp_client
