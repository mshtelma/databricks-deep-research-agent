"""Tests for Autoscaling credential provider."""

import base64
import json
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from deep_research.db.autoscaling_auth import AutoscalingCredentialProvider


def _make_jwt(sub: str = "test-user@example.com") -> str:
    """Create a minimal JWT token for testing."""
    header = base64.urlsafe_b64encode(json.dumps({"alg": "none"}).encode()).rstrip(b"=").decode()
    payload = base64.urlsafe_b64encode(json.dumps({"sub": sub}).encode()).rstrip(b"=").decode()
    return f"{header}.{payload}.signature"


class TestAutoscalingCredentialProviderInit:
    """Tests for AutoscalingCredentialProvider initialization."""

    def test_requires_endpoint_name(self) -> None:
        """Raises ValueError if ENDPOINT_NAME is not set."""
        settings = MagicMock()
        settings.endpoint_name = None

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="ENDPOINT_NAME is required"):
                AutoscalingCredentialProvider(settings)

    def test_accepts_endpoint_from_settings(self) -> None:
        """Creates provider when endpoint_name is in settings."""
        settings = MagicMock()
        settings.endpoint_name = "projects/abc/branches/def/endpoints/ghi"

        provider = AutoscalingCredentialProvider(settings)

        assert provider._endpoint_name == "projects/abc/branches/def/endpoints/ghi"

    def test_accepts_endpoint_from_env(self) -> None:
        """Creates provider when ENDPOINT_NAME is in env."""
        settings = MagicMock()
        settings.endpoint_name = None

        with patch.dict("os.environ", {"ENDPOINT_NAME": "projects/x/branches/y/endpoints/z"}):
            provider = AutoscalingCredentialProvider(settings)

        assert provider._endpoint_name == "projects/x/branches/y/endpoints/z"


class TestAutoscalingCredentialProviderMethods:
    """Tests for AutoscalingCredentialProvider interface methods."""

    def _make_provider(self) -> AutoscalingCredentialProvider:
        """Create a provider with mocked settings."""
        settings = MagicMock()
        settings.endpoint_name = "projects/abc/branches/def/endpoints/ghi"
        settings.lakebase_database = "deep_research"
        return AutoscalingCredentialProvider(settings)

    def test_get_backend_type(self) -> None:
        """Returns 'autoscaling'."""
        provider = self._make_provider()
        assert provider.get_backend_type() == "autoscaling"

    def test_get_host_from_env(self) -> None:
        """Gets host from PGHOST env var."""
        provider = self._make_provider()

        with patch.dict("os.environ", {"PGHOST": "ep-abc.database.us-west-2.cloud.databricks.com"}):
            host = provider.get_host()

        assert host == "ep-abc.database.us-west-2.cloud.databricks.com"

    def test_get_host_missing_raises(self) -> None:
        """Raises ValueError when PGHOST is not set."""
        provider = self._make_provider()

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="PGHOST is required"):
                provider.get_host()

    def test_get_port_default(self) -> None:
        """Default port is 5432."""
        provider = self._make_provider()

        with patch.dict("os.environ", {}, clear=True):
            assert provider.get_port() == 5432

    def test_get_port_from_env(self) -> None:
        """Port from PGPORT env var."""
        provider = self._make_provider()

        with patch.dict("os.environ", {"PGPORT": "5433"}):
            assert provider.get_port() == 5433

    def test_get_database_from_settings(self) -> None:
        """Database from settings when PGDATABASE is not set."""
        provider = self._make_provider()

        with patch.dict("os.environ", {}, clear=True):
            assert provider.get_database() == "deep_research"

    def test_get_database_from_env(self) -> None:
        """Database from PGDATABASE env var overrides settings."""
        provider = self._make_provider()

        with patch.dict("os.environ", {"PGDATABASE": "custom_db"}):
            assert provider.get_database() == "custom_db"

    @patch("deep_research.core.databricks_auth.get_databricks_auth")
    def test_get_credential(self, mock_auth: MagicMock) -> None:
        """Generates credential via postgres.generate_database_credential()."""
        provider = self._make_provider()

        mock_client = MagicMock()
        mock_auth.return_value.get_client.return_value = mock_client

        token = _make_jwt("sp-user@example.com")
        mock_client.postgres.generate_database_credential.return_value = MagicMock(token=token)

        cred = provider.get_credential()

        mock_client.postgres.generate_database_credential.assert_called_once_with(
            endpoint="projects/abc/branches/def/endpoints/ghi",
        )
        assert cred.token == token
        assert cred.username == "sp-user@example.com"
        assert cred.expires_at > datetime.now(UTC)

    @patch("deep_research.core.databricks_auth.get_databricks_auth")
    def test_credential_caching(self, mock_auth: MagicMock) -> None:
        """Second call returns cached credential without API call."""
        provider = self._make_provider()

        mock_client = MagicMock()
        mock_auth.return_value.get_client.return_value = mock_client

        token = _make_jwt()
        mock_client.postgres.generate_database_credential.return_value = MagicMock(token=token)

        cred1 = provider.get_credential()
        cred2 = provider.get_credential()

        assert cred1 is cred2
        mock_client.postgres.generate_database_credential.assert_called_once()

    def test_current_credential_before_generation(self) -> None:
        """current_credential is None before first get_credential()."""
        provider = self._make_provider()
        assert provider.current_credential is None

    @patch("deep_research.core.databricks_auth.get_databricks_auth")
    def test_build_connection_url(self, mock_auth: MagicMock) -> None:
        """Builds correct asyncpg connection URL."""
        provider = self._make_provider()

        mock_client = MagicMock()
        mock_auth.return_value.get_client.return_value = mock_client

        token = _make_jwt("user@example.com")
        mock_client.postgres.generate_database_credential.return_value = MagicMock(token=token)

        with patch.dict("os.environ", {
            "PGHOST": "ep-abc.database.cloud.databricks.com",
            "PGPORT": "5432",
            "PGDATABASE": "my_db",
        }):
            url = provider.build_connection_url()

        assert url.startswith("postgresql+asyncpg://")
        assert "ep-abc.database.cloud.databricks.com" in url
        assert "5432" in url
        assert "my_db" in url
