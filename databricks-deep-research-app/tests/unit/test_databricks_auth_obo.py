"""Unit tests for get_user_workspace_client() in databricks_auth.py (US-401)."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException


class TestGetUserWorkspaceClientWithOboHeader:
    """get_user_workspace_client builds a client from the OBO header."""

    @patch("deep_research.core.databricks_auth.WorkspaceClient")
    @patch("deep_research.core.databricks_auth.os.environ.get")
    def test_builds_client_with_obo_token_and_env_host(
        self,
        mock_env_get: MagicMock,
        mock_workspace_client: MagicMock,
    ) -> None:
        """When X-Forwarded-Access-Token header is present, client uses that token."""
        mock_env_get.return_value = "https://my-workspace.databricks.com"

        mock_request = MagicMock()
        mock_request.headers.get.return_value = "obo-token-abc"

        from deep_research.core.databricks_auth import get_user_workspace_client

        get_user_workspace_client(mock_request)

        mock_workspace_client.assert_called_once_with(
            host="https://my-workspace.databricks.com",
            token="obo-token-abc",
            auth_type="pat",
        )

    @patch("deep_research.core.databricks_auth.WorkspaceClient")
    @patch("deep_research.core.databricks_auth.get_databricks_auth")
    @patch("deep_research.core.databricks_auth.os.environ.get")
    def test_falls_back_to_sp_client_host_when_env_missing(
        self,
        mock_env_get: MagicMock,
        mock_get_auth: MagicMock,
        mock_workspace_client: MagicMock,
    ) -> None:
        """When DATABRICKS_HOST env is absent, host is derived from SP client."""
        mock_env_get.return_value = None  # DATABRICKS_HOST not set

        mock_sp_client = MagicMock()
        mock_sp_client.config.host = "https://sp-derived.databricks.com"
        mock_auth = MagicMock()
        mock_auth.get_client.return_value = mock_sp_client
        mock_get_auth.return_value = mock_auth

        mock_request = MagicMock()
        mock_request.headers.get.return_value = "obo-token-xyz"

        from deep_research.core.databricks_auth import get_user_workspace_client

        get_user_workspace_client(mock_request)

        mock_workspace_client.assert_called_once_with(
            host="https://sp-derived.databricks.com",
            token="obo-token-xyz",
            auth_type="pat",
        )


class TestGetUserWorkspaceClientLocalDevFallback:
    """Without OBO header in local dev, falls back to SP client."""

    @patch("deep_research.core.databricks_auth.get_databricks_auth")
    @patch("deep_research.core.databricks_auth.get_settings")
    def test_fallback_to_sp_client_in_local_dev(
        self,
        mock_get_settings: MagicMock,
        mock_get_auth: MagicMock,
    ) -> None:
        """Missing OBO header in local dev returns SP client (no exception)."""
        mock_settings = MagicMock()
        mock_settings.is_databricks_app = False
        mock_get_settings.return_value = mock_settings

        mock_sp_client = MagicMock()
        mock_auth = MagicMock()
        mock_auth.get_client.return_value = mock_sp_client
        mock_get_auth.return_value = mock_auth

        mock_request = MagicMock()
        mock_request.headers.get.return_value = None  # no OBO header

        from deep_research.core.databricks_auth import get_user_workspace_client

        result = get_user_workspace_client(mock_request)

        assert result is mock_sp_client


class TestGetUserWorkspaceClientDatabricksAppEnv:
    """Inside Databricks Apps, missing OBO header raises HTTPException 401."""

    @patch("deep_research.core.databricks_auth.get_settings")
    def test_raises_401_when_obo_missing_in_databricks_app(
        self,
        mock_get_settings: MagicMock,
    ) -> None:
        """Raises HTTPException 401 with error_kind=missing_obo_token in Databricks Apps."""
        mock_settings = MagicMock()
        mock_settings.is_databricks_app = True
        mock_get_settings.return_value = mock_settings

        mock_request = MagicMock()
        mock_request.headers.get.return_value = None  # OBO header absent

        from deep_research.core.databricks_auth import get_user_workspace_client

        with pytest.raises(HTTPException) as exc_info:
            get_user_workspace_client(mock_request)

        assert exc_info.value.status_code == 401
        detail = exc_info.value.detail
        assert isinstance(detail, dict)
        assert detail["error_kind"] == "missing_obo_token"
