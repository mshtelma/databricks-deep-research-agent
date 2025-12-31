"""Unit tests for LLM OAuth credential provider."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.services.llm.auth import (
    LLMCredential,
    LLMCredentialProvider,
    TOKEN_LIFETIME,
    TOKEN_REFRESH_BUFFER,
)


class TestLLMCredential:
    """Tests for LLMCredential dataclass."""

    def test_is_expired_false_when_fresh(self) -> None:
        """Token should not be expired when just created."""
        credential = LLMCredential(
            token="test-token",
            expires_at=datetime.now(UTC) + TOKEN_LIFETIME,
        )
        assert not credential.is_expired

    def test_is_expired_true_when_within_buffer(self) -> None:
        """Token should be expired when within refresh buffer."""
        credential = LLMCredential(
            token="test-token",
            expires_at=datetime.now(UTC) + TOKEN_REFRESH_BUFFER - timedelta(seconds=1),
        )
        assert credential.is_expired

    def test_is_expired_true_when_past_expiry(self) -> None:
        """Token should be expired when past expiry time."""
        credential = LLMCredential(
            token="test-token",
            expires_at=datetime.now(UTC) - timedelta(minutes=1),
        )
        assert credential.is_expired

    def test_is_expired_false_just_outside_buffer(self) -> None:
        """Token should not be expired when just outside refresh buffer."""
        credential = LLMCredential(
            token="test-token",
            expires_at=datetime.now(UTC) + TOKEN_REFRESH_BUFFER + timedelta(seconds=10),
        )
        assert not credential.is_expired


class TestLLMCredentialProvider:
    """Tests for LLMCredentialProvider."""

    @patch("src.services.llm.auth.WorkspaceClient")
    def test_get_credential_generates_on_first_call(
        self, mock_wc_class: MagicMock
    ) -> None:
        """Should generate credential on first call."""
        mock_wc = MagicMock()
        mock_wc.config.host = "https://test.databricks.com"
        mock_wc.config.oauth_token.return_value.access_token = "test-token"
        mock_wc_class.return_value = mock_wc

        provider = LLMCredentialProvider(profile="test-profile")
        credential = provider.get_credential()

        assert credential.token == "test-token"
        mock_wc.config.authenticate.assert_called_once()

    @patch("src.services.llm.auth.WorkspaceClient")
    def test_get_credential_caches_when_valid(self, mock_wc_class: MagicMock) -> None:
        """Should return cached credential when not expired."""
        mock_wc = MagicMock()
        mock_wc.config.host = "https://test.databricks.com"
        mock_wc.config.oauth_token.return_value.access_token = "test-token"
        mock_wc_class.return_value = mock_wc

        provider = LLMCredentialProvider(profile="test-profile")
        cred1 = provider.get_credential()
        cred2 = provider.get_credential()

        assert cred1 is cred2
        assert mock_wc.config.authenticate.call_count == 1

    @patch("src.services.llm.auth.WorkspaceClient")
    def test_get_credential_refreshes_when_expired(
        self, mock_wc_class: MagicMock
    ) -> None:
        """Should refresh credential when expired."""
        mock_wc = MagicMock()
        mock_wc.config.host = "https://test.databricks.com"
        mock_wc.config.oauth_token.return_value.access_token = "test-token"
        mock_wc_class.return_value = mock_wc

        provider = LLMCredentialProvider(profile="test-profile")

        # Get initial credential
        cred1 = provider.get_credential()

        # Manually expire it
        cred1.expires_at = datetime.now(UTC) - timedelta(minutes=1)

        # Should get a new one
        cred2 = provider.get_credential()

        assert cred1 is not cred2
        assert mock_wc.config.authenticate.call_count == 2

    @patch("src.services.llm.auth.WorkspaceClient")
    def test_force_refresh(self, mock_wc_class: MagicMock) -> None:
        """Should refresh credential when force_refresh=True."""
        mock_wc = MagicMock()
        mock_wc.config.host = "https://test.databricks.com"
        mock_wc.config.oauth_token.return_value.access_token = "test-token"
        mock_wc_class.return_value = mock_wc

        provider = LLMCredentialProvider(profile="test-profile")

        cred1 = provider.get_credential()
        cred2 = provider.get_credential(force_refresh=True)

        assert cred1 is not cred2
        assert mock_wc.config.authenticate.call_count == 2

    @patch("src.services.llm.auth.WorkspaceClient")
    def test_get_base_url(self, mock_wc_class: MagicMock) -> None:
        """Should return correct base URL from workspace config."""
        mock_wc = MagicMock()
        mock_wc.config.host = "https://my-workspace.databricks.com"
        mock_wc_class.return_value = mock_wc

        provider = LLMCredentialProvider(profile="test-profile")
        base_url = provider.get_base_url()

        assert base_url == "https://my-workspace.databricks.com/serving-endpoints"

    @patch("src.services.llm.auth.WorkspaceClient")
    def test_workspace_client_reused(self, mock_wc_class: MagicMock) -> None:
        """Should reuse WorkspaceClient instance."""
        mock_wc = MagicMock()
        mock_wc.config.host = "https://test.databricks.com"
        mock_wc.config.oauth_token.return_value.access_token = "test-token"
        mock_wc_class.return_value = mock_wc

        provider = LLMCredentialProvider(profile="test-profile")

        # Multiple operations should only create one client
        provider.get_credential()
        provider.get_base_url()
        provider.get_credential(force_refresh=True)

        # WorkspaceClient should only be instantiated once
        assert mock_wc_class.call_count == 1

    @patch("src.services.llm.auth.WorkspaceClient")
    def test_credential_expiry_set_correctly(self, mock_wc_class: MagicMock) -> None:
        """Should set credential expiry to TOKEN_LIFETIME from now."""
        mock_wc = MagicMock()
        mock_wc.config.host = "https://test.databricks.com"
        mock_wc.config.oauth_token.return_value.access_token = "test-token"
        mock_wc_class.return_value = mock_wc

        before = datetime.now(UTC)
        provider = LLMCredentialProvider(profile="test-profile")
        credential = provider.get_credential()
        after = datetime.now(UTC)

        # Expiry should be approximately TOKEN_LIFETIME from now
        expected_min = before + TOKEN_LIFETIME
        expected_max = after + TOKEN_LIFETIME

        assert expected_min <= credential.expires_at <= expected_max
