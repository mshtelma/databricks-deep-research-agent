"""Unit tests for LLMClient OAuth integration."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from deep_research.core.databricks_auth import TOKEN_LIFETIME, clear_databricks_auth


class TestLLMClientTokenRefresh:
    """Tests for LLMClient token refresh behavior."""

    def teardown_method(self) -> None:
        """Clear singleton after each test."""
        clear_databricks_auth()

    @patch("deep_research.core.databricks_auth.get_settings")
    @patch("deep_research.services.llm.client.AsyncOpenAI")
    @patch("deep_research.services.llm.client.ModelConfig")
    def test_direct_token_no_oauth(
        self,
        mock_model_config: MagicMock,
        mock_openai: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Direct token auth should not use OAuth."""
        mock_settings.return_value.databricks_token = "direct-token"
        mock_settings.return_value.databricks_host = "https://test.databricks.com"
        mock_settings.return_value.databricks_config_profile = None
        mock_settings.return_value.is_databricks_app = False

        from deep_research.services.llm.client import LLMClient

        client = LLMClient()

        assert not client._auth.is_oauth
        assert client._current_token == "direct-token"
        mock_openai.assert_called_once_with(
            api_key="direct-token",
            base_url="https://test.databricks.com/serving-endpoints",
        )

    @patch("deep_research.core.databricks_auth.get_settings")
    @patch("deep_research.core.databricks_auth.WorkspaceClient")
    @patch("deep_research.services.llm.client.AsyncOpenAI")
    @patch("deep_research.services.llm.client.ModelConfig")
    def test_profile_auth_uses_oauth(
        self,
        mock_model_config: MagicMock,
        mock_openai: MagicMock,
        mock_wc_class: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Profile auth should use OAuth."""
        mock_settings.return_value.databricks_token = None
        mock_settings.return_value.databricks_host = "https://test.databricks.com"
        mock_settings.return_value.databricks_config_profile = "test-profile"
        mock_settings.return_value.is_databricks_app = False

        mock_wc = MagicMock()
        mock_wc.config.host = "https://workspace.databricks.com"
        # authenticate() returns headers dict with Bearer token
        mock_wc.config.authenticate.return_value = {"Authorization": "Bearer oauth-token"}
        mock_wc_class.return_value = mock_wc

        from deep_research.services.llm.client import LLMClient

        client = LLMClient()

        assert client._auth.is_oauth
        assert client._current_token == "oauth-token"
        mock_openai.assert_called_once_with(
            api_key="oauth-token",
            base_url="https://workspace.databricks.com/serving-endpoints",
        )

    @patch("deep_research.core.databricks_auth.get_settings")
    @patch("deep_research.services.llm.client.AsyncOpenAI")
    @patch("deep_research.services.llm.client.ModelConfig")
    def test_ensure_fresh_client_noop_for_direct_token(
        self,
        mock_model_config: MagicMock,
        mock_openai: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """_ensure_fresh_client should be no-op for direct token auth."""
        mock_settings.return_value.databricks_token = "direct-token"
        mock_settings.return_value.databricks_host = "https://test.databricks.com"
        mock_settings.return_value.databricks_config_profile = None
        mock_settings.return_value.is_databricks_app = False

        from deep_research.services.llm.client import LLMClient

        client = LLMClient()
        original_client = client._client

        # Should not recreate client
        client._ensure_fresh_client()

        assert client._client is original_client
        # OpenAI client should only be created once (in __init__)
        assert mock_openai.call_count == 1

    @patch("deep_research.core.databricks_auth.get_settings")
    @patch("deep_research.core.databricks_auth.WorkspaceClient")
    @patch("deep_research.services.llm.client.AsyncOpenAI")
    @patch("deep_research.services.llm.client.ModelConfig")
    def test_ensure_fresh_client_recreates_on_token_change(
        self,
        mock_model_config: MagicMock,
        mock_openai: MagicMock,
        mock_wc_class: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Should recreate client when token changes."""
        mock_settings.return_value.databricks_token = None
        mock_settings.return_value.databricks_host = "https://test.databricks.com"
        mock_settings.return_value.databricks_config_profile = "test-profile"
        mock_settings.return_value.is_databricks_app = False

        mock_wc = MagicMock()
        mock_wc.config.host = "https://workspace.databricks.com"
        # authenticate() returns headers dict with Bearer token
        mock_wc.config.authenticate.return_value = {"Authorization": "Bearer token-1"}
        mock_wc_class.return_value = mock_wc

        from deep_research.services.llm.client import LLMClient

        client = LLMClient()
        assert client._current_token == "token-1"
        assert mock_openai.call_count == 1

        # Verify initial call was with token-1
        mock_openai.assert_called_with(
            api_key="token-1",
            base_url="https://workspace.databricks.com/serving-endpoints",
        )

        # Simulate token refresh by expiring the credential and changing token
        client._auth._credential.expires_at = datetime.now(
            UTC
        ) - timedelta(minutes=1)
        mock_wc.config.authenticate.return_value = {"Authorization": "Bearer token-2"}

        # This should detect token change and recreate client
        client._ensure_fresh_client()

        assert client._current_token == "token-2"
        # OpenAI client should be created twice (init + refresh)
        assert mock_openai.call_count == 2
        # Verify second call was with token-2
        mock_openai.assert_called_with(
            api_key="token-2",
            base_url="https://workspace.databricks.com/serving-endpoints",
        )

    @patch("deep_research.core.databricks_auth.get_settings")
    @patch("deep_research.core.databricks_auth.WorkspaceClient")
    @patch("deep_research.services.llm.client.AsyncOpenAI")
    @patch("deep_research.services.llm.client.ModelConfig")
    def test_ensure_fresh_client_no_recreate_when_token_same(
        self,
        mock_model_config: MagicMock,
        mock_openai: MagicMock,
        mock_wc_class: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Should not recreate client when token is the same."""
        mock_settings.return_value.databricks_token = None
        mock_settings.return_value.databricks_host = "https://test.databricks.com"
        mock_settings.return_value.databricks_config_profile = "test-profile"
        mock_settings.return_value.is_databricks_app = False

        mock_wc = MagicMock()
        mock_wc.config.host = "https://workspace.databricks.com"
        # authenticate() returns headers dict with Bearer token
        mock_wc.config.authenticate.return_value = {"Authorization": "Bearer same-token"}
        mock_wc_class.return_value = mock_wc

        from deep_research.services.llm.client import LLMClient

        client = LLMClient()
        original_client = client._client

        # Multiple calls should not recreate client
        client._ensure_fresh_client()
        client._ensure_fresh_client()
        client._ensure_fresh_client()

        assert client._client is original_client
        # OpenAI client should only be created once
        assert mock_openai.call_count == 1

    @patch("deep_research.core.databricks_auth.get_settings")
    @patch("deep_research.services.llm.client.AsyncOpenAI")
    @patch("deep_research.services.llm.client.ModelConfig")
    def test_raises_without_any_auth(
        self,
        mock_model_config: MagicMock,
        mock_openai: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Should raise ValueError when no auth is configured."""
        mock_settings.return_value.databricks_token = None
        mock_settings.return_value.databricks_host = "https://test.databricks.com"
        mock_settings.return_value.databricks_config_profile = None
        mock_settings.return_value.is_databricks_app = False

        from deep_research.services.llm.client import LLMClient

        with pytest.raises(ValueError, match="No Databricks auth configured"):
            LLMClient()

    @patch("deep_research.core.databricks_auth.get_settings")
    @patch("deep_research.core.databricks_auth.WorkspaceClient")
    @patch("deep_research.services.llm.client.AsyncOpenAI")
    @patch("deep_research.services.llm.client.ModelConfig")
    def test_databricks_app_uses_automatic_auth(
        self,
        mock_model_config: MagicMock,
        mock_openai: MagicMock,
        mock_wc_class: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Databricks App environment should use automatic OAuth."""
        mock_settings.return_value.databricks_token = None
        mock_settings.return_value.databricks_host = None
        mock_settings.return_value.databricks_config_profile = None
        mock_settings.return_value.is_databricks_app = True

        mock_wc = MagicMock()
        mock_wc.config.host = "https://app.databricks.com"
        # authenticate() returns headers dict with Bearer token
        mock_wc.config.authenticate.return_value = {"Authorization": "Bearer app-token"}
        mock_wc_class.return_value = mock_wc

        from deep_research.services.llm.client import LLMClient

        client = LLMClient()

        assert client._auth.is_oauth
        assert client._auth.auth_mode == "automatic"
        assert client._current_token == "app-token"
        # WorkspaceClient should be created with no args
        mock_wc_class.assert_called_once_with()
