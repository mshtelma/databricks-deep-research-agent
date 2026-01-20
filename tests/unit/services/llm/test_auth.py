"""Unit tests for Databricks OAuth credential provider.

Tests the centralized DatabricksAuth module and backwards-compatible
LLMCredentialProvider wrapper.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from deep_research.core.databricks_auth import (
    DatabricksAuth,
    OAuthCredential,
    TOKEN_LIFETIME,
    TOKEN_REFRESH_BUFFER,
    clear_databricks_auth,
)


class TestOAuthCredential:
    """Tests for OAuthCredential dataclass."""

    def test_is_expired_false_when_fresh(self) -> None:
        """Token should not be expired when just created."""
        credential = OAuthCredential(
            token="test-token",
            expires_at=datetime.now(UTC) + TOKEN_LIFETIME,
        )
        assert not credential.is_expired

    def test_is_expired_true_when_within_buffer(self) -> None:
        """Token should be expired when within refresh buffer."""
        credential = OAuthCredential(
            token="test-token",
            expires_at=datetime.now(UTC) + TOKEN_REFRESH_BUFFER - timedelta(seconds=1),
        )
        assert credential.is_expired

    def test_is_expired_true_when_past_expiry(self) -> None:
        """Token should be expired when past expiry time."""
        credential = OAuthCredential(
            token="test-token",
            expires_at=datetime.now(UTC) - timedelta(minutes=1),
        )
        assert credential.is_expired

    def test_is_expired_false_just_outside_buffer(self) -> None:
        """Token should not be expired when just outside refresh buffer."""
        credential = OAuthCredential(
            token="test-token",
            expires_at=datetime.now(UTC) + TOKEN_REFRESH_BUFFER + timedelta(seconds=10),
        )
        assert not credential.is_expired


class TestDatabricksAuth:
    """Tests for centralized DatabricksAuth."""

    def teardown_method(self) -> None:
        """Clear singleton after each test."""
        clear_databricks_auth()

    @patch("deep_research.core.databricks_auth.get_settings")
    @patch("deep_research.core.databricks_auth.WorkspaceClient")
    def test_profile_auth_mode(
        self, mock_wc_class: MagicMock, mock_get_settings: MagicMock
    ) -> None:
        """Should use profile auth when DATABRICKS_CONFIG_PROFILE is set."""
        mock_settings = MagicMock()
        mock_settings.databricks_token = None
        mock_settings.databricks_config_profile = "test-profile"
        mock_settings.is_databricks_app = False
        mock_get_settings.return_value = mock_settings

        mock_wc = MagicMock()
        mock_wc.config.host = "https://test.databricks.com"
        # authenticate() returns headers dict with Bearer token
        mock_wc.config.authenticate.return_value = {"Authorization": "Bearer test-token"}
        mock_wc_class.return_value = mock_wc

        auth = DatabricksAuth()

        assert auth.auth_mode == "profile"
        assert auth.is_oauth

    @patch("deep_research.core.databricks_auth.get_settings")
    @patch("deep_research.core.databricks_auth.WorkspaceClient")
    def test_automatic_auth_mode(
        self, mock_wc_class: MagicMock, mock_get_settings: MagicMock
    ) -> None:
        """Should use automatic auth when running as Databricks App."""
        mock_settings = MagicMock()
        mock_settings.databricks_token = None
        mock_settings.databricks_config_profile = None
        mock_settings.is_databricks_app = True
        mock_get_settings.return_value = mock_settings

        mock_wc = MagicMock()
        mock_wc.config.host = "https://test.databricks.com"
        mock_wc_class.return_value = mock_wc

        auth = DatabricksAuth()

        assert auth.auth_mode == "automatic"
        assert auth.is_oauth

        # Trigger client creation (lazy init)
        auth.get_client()
        mock_wc_class.assert_called_once_with()  # No args for automatic auth

    @patch("deep_research.core.databricks_auth.get_settings")
    def test_direct_token_auth_mode(self, mock_get_settings: MagicMock) -> None:
        """Should use direct token when DATABRICKS_TOKEN is set."""
        mock_settings = MagicMock()
        mock_settings.databricks_token = "direct-token"
        mock_settings.databricks_host = "https://test.databricks.com"
        mock_get_settings.return_value = mock_settings

        auth = DatabricksAuth()

        assert auth.auth_mode == "direct_token"
        assert not auth.is_oauth
        assert auth.get_token() == "direct-token"

    @patch("deep_research.core.databricks_auth.get_settings")
    def test_no_auth_raises_error(self, mock_get_settings: MagicMock) -> None:
        """Should raise error when no auth is configured."""
        mock_settings = MagicMock()
        mock_settings.databricks_token = None
        mock_settings.databricks_config_profile = None
        mock_settings.is_databricks_app = False
        mock_get_settings.return_value = mock_settings

        with pytest.raises(ValueError, match="No Databricks auth configured"):
            DatabricksAuth()

    @patch("deep_research.core.databricks_auth.get_settings")
    @patch("deep_research.core.databricks_auth.WorkspaceClient")
    def test_get_token_generates_on_first_call(
        self, mock_wc_class: MagicMock, mock_get_settings: MagicMock
    ) -> None:
        """Should generate credential on first call."""
        mock_settings = MagicMock()
        mock_settings.databricks_token = None
        mock_settings.databricks_config_profile = "test-profile"
        mock_settings.is_databricks_app = False
        mock_get_settings.return_value = mock_settings

        mock_wc = MagicMock()
        mock_wc.config.host = "https://test.databricks.com"
        # authenticate() returns headers dict with Bearer token
        mock_wc.config.authenticate.return_value = {"Authorization": "Bearer oauth-token"}
        mock_wc_class.return_value = mock_wc

        auth = DatabricksAuth()
        token = auth.get_token()

        assert token == "oauth-token"
        mock_wc.config.authenticate.assert_called_once()

    @patch("deep_research.core.databricks_auth.get_settings")
    @patch("deep_research.core.databricks_auth.WorkspaceClient")
    def test_get_token_caches_when_valid(
        self, mock_wc_class: MagicMock, mock_get_settings: MagicMock
    ) -> None:
        """Should return cached credential when not expired."""
        mock_settings = MagicMock()
        mock_settings.databricks_token = None
        mock_settings.databricks_config_profile = "test-profile"
        mock_settings.is_databricks_app = False
        mock_get_settings.return_value = mock_settings

        mock_wc = MagicMock()
        mock_wc.config.host = "https://test.databricks.com"
        # authenticate() returns headers dict with Bearer token
        mock_wc.config.authenticate.return_value = {"Authorization": "Bearer oauth-token"}
        mock_wc_class.return_value = mock_wc

        auth = DatabricksAuth()
        token1 = auth.get_token()
        token2 = auth.get_token()

        assert token1 == token2
        # Should only authenticate once
        assert mock_wc.config.authenticate.call_count == 1

    @patch("deep_research.core.databricks_auth.get_settings")
    @patch("deep_research.core.databricks_auth.WorkspaceClient")
    def test_get_token_force_refresh(
        self, mock_wc_class: MagicMock, mock_get_settings: MagicMock
    ) -> None:
        """Should refresh credential when force_refresh=True."""
        mock_settings = MagicMock()
        mock_settings.databricks_token = None
        mock_settings.databricks_config_profile = "test-profile"
        mock_settings.is_databricks_app = False
        mock_get_settings.return_value = mock_settings

        mock_wc = MagicMock()
        mock_wc.config.host = "https://test.databricks.com"
        # authenticate() returns headers dict with Bearer token
        mock_wc.config.authenticate.return_value = {"Authorization": "Bearer oauth-token"}
        mock_wc_class.return_value = mock_wc

        auth = DatabricksAuth()
        auth.get_token()
        auth.get_token(force_refresh=True)

        # Should authenticate twice
        assert mock_wc.config.authenticate.call_count == 2

    @patch("deep_research.core.databricks_auth.get_settings")
    @patch("deep_research.core.databricks_auth.WorkspaceClient")
    def test_get_base_url_from_profile(
        self, mock_wc_class: MagicMock, mock_get_settings: MagicMock
    ) -> None:
        """Should derive base URL from WorkspaceClient for profile auth."""
        mock_settings = MagicMock()
        mock_settings.databricks_token = None
        mock_settings.databricks_config_profile = "test-profile"
        mock_settings.is_databricks_app = False
        mock_get_settings.return_value = mock_settings

        mock_wc = MagicMock()
        mock_wc.config.host = "https://my-workspace.databricks.com"
        mock_wc_class.return_value = mock_wc

        auth = DatabricksAuth()
        base_url = auth.get_base_url()

        assert base_url == "https://my-workspace.databricks.com/serving-endpoints"

    @patch("deep_research.core.databricks_auth.get_settings")
    def test_get_base_url_from_direct_token(
        self, mock_get_settings: MagicMock
    ) -> None:
        """Should use DATABRICKS_HOST for direct token auth."""
        mock_settings = MagicMock()
        mock_settings.databricks_token = "direct-token"
        mock_settings.databricks_host = "https://direct.databricks.com"
        mock_get_settings.return_value = mock_settings

        auth = DatabricksAuth()
        base_url = auth.get_base_url()

        assert base_url == "https://direct.databricks.com/serving-endpoints"

    @patch("deep_research.core.databricks_auth.get_settings")
    @patch("deep_research.core.databricks_auth.WorkspaceClient")
    def test_workspace_client_reused(
        self, mock_wc_class: MagicMock, mock_get_settings: MagicMock
    ) -> None:
        """Should reuse WorkspaceClient instance."""
        mock_settings = MagicMock()
        mock_settings.databricks_token = None
        mock_settings.databricks_config_profile = "test-profile"
        mock_settings.is_databricks_app = False
        mock_get_settings.return_value = mock_settings

        mock_wc = MagicMock()
        mock_wc.config.host = "https://test.databricks.com"
        # authenticate() returns headers dict with Bearer token
        mock_wc.config.authenticate.return_value = {"Authorization": "Bearer oauth-token"}
        mock_wc_class.return_value = mock_wc

        auth = DatabricksAuth()

        # Multiple operations should only create one client
        auth.get_token()
        auth.get_base_url()
        auth.get_token(force_refresh=True)
        auth.get_client()

        # WorkspaceClient should only be instantiated once
        assert mock_wc_class.call_count == 1


# Backwards compatibility tests
class TestLLMCredential:
    """Tests for LLMCredential alias (backwards compatibility)."""

    def test_llm_credential_is_oauth_credential(self) -> None:
        """LLMCredential should be an alias for OAuthCredential."""
        from deep_research.services.llm.auth import LLMCredential

        assert LLMCredential is OAuthCredential
