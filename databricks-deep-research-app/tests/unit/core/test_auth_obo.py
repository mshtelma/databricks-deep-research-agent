"""Unit tests for OBO (On-Behalf-Of) authentication functions."""

from unittest.mock import MagicMock, patch

import pytest

from deep_research.core.auth import (
    UserIdentity,
    extract_obo_token,
    get_user_workspace_client,
)


class TestExtractOboToken:
    """Tests for extract_obo_token function."""

    def test_extracts_token_when_present(self) -> None:
        """Token is extracted when header is present."""
        headers = {"x-forwarded-access-token": "user-oauth-token-123"}
        result = extract_obo_token(headers)
        assert result == "user-oauth-token-123"

    def test_returns_none_when_missing(self) -> None:
        """Returns None when header is not present."""
        headers = {"authorization": "Bearer other-token", "content-type": "application/json"}
        result = extract_obo_token(headers)
        assert result is None

    def test_returns_none_for_empty_headers(self) -> None:
        """Returns None for empty headers dict."""
        headers: dict[str, str] = {}
        result = extract_obo_token(headers)
        assert result is None

    def test_header_is_case_sensitive(self) -> None:
        """Header name must be lowercase (HTTP headers normalized by framework)."""
        # FastAPI/Starlette normalize headers to lowercase
        headers = {"X-Forwarded-Access-Token": "token"}
        result = extract_obo_token(headers)
        # Returns None because key is not lowercase
        assert result is None

    def test_extracts_empty_string_token(self) -> None:
        """Extracts empty string if header value is empty."""
        headers = {"x-forwarded-access-token": ""}
        result = extract_obo_token(headers)
        assert result == ""


class TestGetUserWorkspaceClient:
    """Tests for get_user_workspace_client function."""

    @patch("deep_research.core.auth.WorkspaceClient")
    @patch("deep_research.core.auth.get_settings")
    def test_creates_client_with_token_and_host_from_settings(
        self, mock_get_settings: MagicMock, mock_workspace_client: MagicMock
    ) -> None:
        """Creates WorkspaceClient with token and host from settings."""
        mock_settings = MagicMock()
        mock_settings.databricks_host = "https://test.databricks.com"
        mock_get_settings.return_value = mock_settings

        get_user_workspace_client("user-token-abc")

        mock_workspace_client.assert_called_once_with(
            host="https://test.databricks.com", token="user-token-abc", auth_type="pat"
        )

    @patch("deep_research.core.auth.WorkspaceClient")
    @patch("deep_research.core.auth.get_workspace_client")
    @patch("deep_research.core.auth.get_settings")
    def test_derives_host_from_sp_client_when_not_in_settings(
        self,
        mock_get_settings: MagicMock,
        mock_get_sp_client: MagicMock,
        mock_workspace_client: MagicMock,
    ) -> None:
        """Derives host from service principal client when not in settings."""
        mock_settings = MagicMock()
        mock_settings.databricks_host = None
        mock_get_settings.return_value = mock_settings

        mock_sp_client = MagicMock()
        mock_sp_client.config.host = "https://derived-host.databricks.com"
        mock_get_sp_client.return_value = mock_sp_client

        get_user_workspace_client("user-token-xyz")

        mock_workspace_client.assert_called_once_with(
            host="https://derived-host.databricks.com", token="user-token-xyz", auth_type="pat"
        )

    @patch("deep_research.core.auth.WorkspaceClient")
    @patch("deep_research.core.auth.get_settings")
    def test_returns_workspace_client_instance(
        self, mock_get_settings: MagicMock, mock_workspace_client: MagicMock
    ) -> None:
        """Returns the created WorkspaceClient instance."""
        mock_settings = MagicMock()
        mock_settings.databricks_host = "https://test.databricks.com"
        mock_get_settings.return_value = mock_settings

        mock_client_instance = MagicMock()
        mock_workspace_client.return_value = mock_client_instance

        result = get_user_workspace_client("token")

        assert result == mock_client_instance


class TestUserIdentityFromWorkspaceUser:
    """Tests for UserIdentity.from_workspace_user method."""

    def test_creates_identity_from_user_object(self) -> None:
        """Creates UserIdentity from workspace user object."""
        mock_user = MagicMock()
        mock_user.id = 12345
        mock_user.user_name = "test@example.com"
        mock_user.display_name = "Test User"

        identity = UserIdentity.from_workspace_user(mock_user)

        assert identity.user_id == "12345"
        assert identity.email == "test@example.com"
        assert identity.display_name == "Test User"

    def test_uses_username_when_id_is_none(self) -> None:
        """Uses user_name as user_id when id is None."""
        mock_user = MagicMock()
        mock_user.id = None
        mock_user.user_name = "fallback@example.com"
        mock_user.display_name = "Fallback User"

        identity = UserIdentity.from_workspace_user(mock_user)

        assert identity.user_id == "fallback@example.com"

    def test_handles_missing_display_name(self) -> None:
        """Falls back to user_name when display_name is None."""
        mock_user = MagicMock()
        mock_user.id = 999
        mock_user.user_name = "user@example.com"
        mock_user.display_name = None

        identity = UserIdentity.from_workspace_user(mock_user)

        assert identity.display_name == "user@example.com"


class TestUserIdentityAnonymous:
    """Tests for UserIdentity.anonymous method."""

    def test_creates_anonymous_identity(self) -> None:
        """Creates anonymous user identity for development."""
        identity = UserIdentity.anonymous()

        assert identity.user_id == "anonymous"
        assert identity.email == "anonymous@local.dev"
        assert identity.display_name == "Anonymous User"

    def test_anonymous_identity_is_frozen(self) -> None:
        """Anonymous identity is immutable (frozen dataclass)."""
        identity = UserIdentity.anonymous()

        with pytest.raises(AttributeError):
            identity.user_id = "changed"  # type: ignore[misc]
