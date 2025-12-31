"""Unit tests for LLMClient OAuth integration."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.services.llm.auth import TOKEN_LIFETIME


class TestLLMClientTokenRefresh:
    """Tests for LLMClient token refresh behavior."""

    @patch("src.services.llm.client.get_settings")
    @patch("src.services.llm.client.AsyncOpenAI")
    @patch("src.services.llm.client.ModelConfig")
    def test_direct_token_no_credential_provider(
        self,
        mock_model_config: MagicMock,
        mock_openai: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Direct token auth should not create credential provider."""
        mock_settings.return_value.databricks_token = "direct-token"
        mock_settings.return_value.databricks_host = "https://test.databricks.com"
        mock_settings.return_value.databricks_config_profile = None

        from src.services.llm.client import LLMClient

        client = LLMClient()

        assert client._credential_provider is None
        assert client._current_token == "direct-token"
        mock_openai.assert_called_once_with(
            api_key="direct-token",
            base_url="https://test.databricks.com/serving-endpoints",
        )

    @patch("src.services.llm.client.get_settings")
    @patch("src.services.llm.client.AsyncOpenAI")
    @patch("src.services.llm.client.ModelConfig")
    @patch("src.services.llm.auth.WorkspaceClient")
    def test_profile_auth_creates_credential_provider(
        self,
        mock_wc_class: MagicMock,
        mock_model_config: MagicMock,
        mock_openai: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Profile auth should create credential provider."""
        mock_settings.return_value.databricks_token = None
        mock_settings.return_value.databricks_host = "https://test.databricks.com"
        mock_settings.return_value.databricks_config_profile = "test-profile"

        mock_wc = MagicMock()
        mock_wc.config.host = "https://workspace.databricks.com"
        mock_wc.config.oauth_token.return_value.access_token = "oauth-token"
        mock_wc_class.return_value = mock_wc

        from src.services.llm.client import LLMClient

        client = LLMClient()

        assert client._credential_provider is not None
        assert client._current_token == "oauth-token"
        mock_openai.assert_called_once_with(
            api_key="oauth-token",
            base_url="https://workspace.databricks.com/serving-endpoints",
        )

    @patch("src.services.llm.client.get_settings")
    @patch("src.services.llm.client.AsyncOpenAI")
    @patch("src.services.llm.client.ModelConfig")
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

        from src.services.llm.client import LLMClient

        client = LLMClient()
        original_client = client._client

        # Should not recreate client
        client._ensure_fresh_client()

        assert client._client is original_client
        # OpenAI client should only be created once (in __init__)
        assert mock_openai.call_count == 1

    @patch("src.services.llm.client.get_settings")
    @patch("src.services.llm.client.AsyncOpenAI")
    @patch("src.services.llm.client.ModelConfig")
    @patch("src.services.llm.auth.WorkspaceClient")
    def test_ensure_fresh_client_recreates_on_token_change(
        self,
        mock_wc_class: MagicMock,
        mock_model_config: MagicMock,
        mock_openai: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Should recreate client when token changes."""
        mock_settings.return_value.databricks_token = None
        mock_settings.return_value.databricks_host = "https://test.databricks.com"
        mock_settings.return_value.databricks_config_profile = "test-profile"

        mock_wc = MagicMock()
        mock_wc.config.host = "https://workspace.databricks.com"
        mock_wc.config.oauth_token.return_value.access_token = "token-1"
        mock_wc_class.return_value = mock_wc

        from src.services.llm.client import LLMClient

        client = LLMClient()
        assert client._current_token == "token-1"
        assert mock_openai.call_count == 1

        # Verify initial call was with token-1
        mock_openai.assert_called_with(
            api_key="token-1",
            base_url="https://workspace.databricks.com/serving-endpoints",
        )

        # Simulate token refresh by expiring the credential and changing token
        client._credential_provider._credential.expires_at = datetime.now(
            UTC
        ) - timedelta(minutes=1)
        mock_wc.config.oauth_token.return_value.access_token = "token-2"

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

    @patch("src.services.llm.client.get_settings")
    @patch("src.services.llm.client.AsyncOpenAI")
    @patch("src.services.llm.client.ModelConfig")
    @patch("src.services.llm.auth.WorkspaceClient")
    def test_ensure_fresh_client_no_recreate_when_token_same(
        self,
        mock_wc_class: MagicMock,
        mock_model_config: MagicMock,
        mock_openai: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Should not recreate client when token is the same."""
        mock_settings.return_value.databricks_token = None
        mock_settings.return_value.databricks_host = "https://test.databricks.com"
        mock_settings.return_value.databricks_config_profile = "test-profile"

        mock_wc = MagicMock()
        mock_wc.config.host = "https://workspace.databricks.com"
        mock_wc.config.oauth_token.return_value.access_token = "same-token"
        mock_wc_class.return_value = mock_wc

        from src.services.llm.client import LLMClient

        client = LLMClient()
        original_client = client._client

        # Multiple calls should not recreate client
        client._ensure_fresh_client()
        client._ensure_fresh_client()
        client._ensure_fresh_client()

        assert client._client is original_client
        # OpenAI client should only be created once
        assert mock_openai.call_count == 1

    @patch("src.services.llm.client.get_settings")
    @patch("src.services.llm.client.AsyncOpenAI")
    @patch("src.services.llm.client.ModelConfig")
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

        from src.services.llm.client import LLMClient

        with pytest.raises(ValueError, match="No Databricks token available"):
            LLMClient()
