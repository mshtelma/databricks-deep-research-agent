"""Tests for Lakebase connection utilities."""

import base64
import json
import os
from unittest.mock import MagicMock, patch

import pytest

from deep_research.deployment.lakebase_connection import (
    LakebaseConnectionInfo,
    extract_username_from_token,
    get_lakebase_connection_info,
    get_lakebase_host,
)


class TestGetLakebaseHost:
    """Tests for get_lakebase_host function."""

    def test_pghost_env_variable_takes_priority(self) -> None:
        """PGHOST environment variable should be used when set."""
        with patch.dict(os.environ, {"PGHOST": "instance-abc123.database.cloud.databricks.com"}):
            host = get_lakebase_host("my-instance", workspace_client=MagicMock())
            assert host == "instance-abc123.database.cloud.databricks.com"

    def test_api_lookup_when_pghost_not_set(self) -> None:
        """Should use API lookup when PGHOST is not set."""
        mock_client = MagicMock()
        mock_instance = MagicMock()
        mock_instance.read_write_dns = "instance-xyz789.database.cloud.databricks.com"
        mock_client.database.get_database_instance.return_value = mock_instance

        # Ensure PGHOST is not set
        with patch.dict(os.environ, {}, clear=True):
            # Also patch out any existing PGHOST
            os.environ.pop("PGHOST", None)
            host = get_lakebase_host("my-instance", workspace_client=mock_client)

        assert host == "instance-xyz789.database.cloud.databricks.com"
        mock_client.database.get_database_instance.assert_called_once_with(name="my-instance")

    def test_raises_when_no_read_write_dns(self) -> None:
        """Should raise ValueError when instance has no read_write_dns."""
        mock_client = MagicMock()
        mock_instance = MagicMock()
        mock_instance.read_write_dns = None
        mock_client.database.get_database_instance.return_value = mock_instance

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("PGHOST", None)
            with pytest.raises(ValueError, match="has no read_write_dns"):
                get_lakebase_host("my-instance", workspace_client=mock_client)

    def test_raises_when_api_lookup_fails(self) -> None:
        """Should raise ValueError when API lookup fails."""
        mock_client = MagicMock()
        mock_client.database.get_database_instance.side_effect = Exception("API error")

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("PGHOST", None)
            with pytest.raises(ValueError, match="Failed to get Lakebase instance"):
                get_lakebase_host("my-instance", workspace_client=mock_client)


class TestExtractUsernameFromToken:
    """Tests for extract_username_from_token function."""

    def _create_jwt_token(self, sub: str) -> str:
        """Helper to create a JWT token with a given sub claim."""
        header = base64.urlsafe_b64encode(b'{"alg":"RS256","typ":"JWT"}').rstrip(b"=").decode()
        payload = base64.urlsafe_b64encode(json.dumps({"sub": sub}).encode()).rstrip(b"=").decode()
        signature = base64.urlsafe_b64encode(b"fake_signature").rstrip(b"=").decode()
        return f"{header}.{payload}.{signature}"

    def test_pguser_env_variable_takes_priority(self) -> None:
        """PGUSER environment variable should be used when set."""
        token = self._create_jwt_token("token_user@example.com")
        with patch.dict(os.environ, {"PGUSER": "env_user@example.com"}):
            username = extract_username_from_token(token)
            assert username == "env_user@example.com"

    def test_extracts_from_jwt_sub_claim(self) -> None:
        """Should extract username from JWT 'sub' claim when PGUSER not set."""
        token = self._create_jwt_token("user@example.com")
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("PGUSER", None)
            username = extract_username_from_token(token)
            assert username == "user@example.com"

    def test_extracts_service_principal_id(self) -> None:
        """Should extract service principal ID from JWT 'sub' claim."""
        sp_id = "12345678-1234-1234-1234-123456789012"
        token = self._create_jwt_token(sp_id)
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("PGUSER", None)
            username = extract_username_from_token(token)
            assert username == sp_id

    def test_raises_when_no_sub_claim(self) -> None:
        """Should raise ValueError when JWT has no 'sub' claim."""
        header = base64.urlsafe_b64encode(b'{"alg":"RS256","typ":"JWT"}').rstrip(b"=").decode()
        payload = base64.urlsafe_b64encode(b'{"foo":"bar"}').rstrip(b"=").decode()
        signature = base64.urlsafe_b64encode(b"fake_signature").rstrip(b"=").decode()
        token = f"{header}.{payload}.{signature}"

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("PGUSER", None)
            with pytest.raises(ValueError, match="Could not determine username"):
                extract_username_from_token(token)

    def test_raises_when_invalid_token(self) -> None:
        """Should raise ValueError when token is invalid."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("PGUSER", None)
            with pytest.raises(ValueError, match="Could not determine username"):
                extract_username_from_token("invalid_token")


class TestGetLakebaseConnectionInfo:
    """Tests for get_lakebase_connection_info function."""

    def _create_jwt_token(self, sub: str) -> str:
        """Helper to create a JWT token with a given sub claim."""
        header = base64.urlsafe_b64encode(b'{"alg":"RS256","typ":"JWT"}').rstrip(b"=").decode()
        payload = base64.urlsafe_b64encode(json.dumps({"sub": sub}).encode()).rstrip(b"=").decode()
        signature = base64.urlsafe_b64encode(b"fake_signature").rstrip(b"=").decode()
        return f"{header}.{payload}.{signature}"

    def test_returns_connection_info(self) -> None:
        """Should return complete connection info."""
        mock_client = MagicMock()
        mock_instance = MagicMock()
        mock_instance.read_write_dns = "instance-abc.database.cloud.databricks.com"
        mock_client.database.get_database_instance.return_value = mock_instance

        mock_cred = MagicMock()
        mock_cred.token = self._create_jwt_token("user@example.com")
        mock_client.database.generate_database_credential.return_value = mock_cred

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("PGHOST", None)
            os.environ.pop("PGPORT", None)
            os.environ.pop("PGUSER", None)

            info = get_lakebase_connection_info("my-instance", workspace_client=mock_client)

            assert isinstance(info, LakebaseConnectionInfo)
            assert info.host == "instance-abc.database.cloud.databricks.com"
            assert info.port == 5432
            assert info.username == "user@example.com"
            assert info.token == mock_cred.token

    def test_uses_pgport_env_variable(self) -> None:
        """Should use PGPORT environment variable when set."""
        mock_client = MagicMock()
        mock_instance = MagicMock()
        mock_instance.read_write_dns = "instance-abc.database.cloud.databricks.com"
        mock_client.database.get_database_instance.return_value = mock_instance

        mock_cred = MagicMock()
        mock_cred.token = self._create_jwt_token("user@example.com")
        mock_client.database.generate_database_credential.return_value = mock_cred

        with patch.dict(os.environ, {"PGPORT": "15432"}, clear=True):
            os.environ.pop("PGHOST", None)
            os.environ.pop("PGUSER", None)

            info = get_lakebase_connection_info("my-instance", workspace_client=mock_client)

            assert info.port == 15432

    def test_raises_when_no_token_returned(self) -> None:
        """Should raise ValueError when no token is returned."""
        mock_client = MagicMock()
        mock_instance = MagicMock()
        mock_instance.read_write_dns = "instance-abc.database.cloud.databricks.com"
        mock_client.database.get_database_instance.return_value = mock_instance

        mock_cred = MagicMock()
        mock_cred.token = None
        mock_client.database.generate_database_credential.return_value = mock_cred

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("PGHOST", None)
            os.environ.pop("PGUSER", None)

            with pytest.raises(ValueError, match="No token returned"):
                get_lakebase_connection_info("my-instance", workspace_client=mock_client)


class TestLakebaseConnectionInfo:
    """Tests for LakebaseConnectionInfo dataclass."""

    def test_dataclass_attributes(self) -> None:
        """Should store all connection attributes."""
        info = LakebaseConnectionInfo(
            host="instance-abc.database.cloud.databricks.com",
            port=5432,
            username="user@example.com",
            token="my-token",
        )

        assert info.host == "instance-abc.database.cloud.databricks.com"
        assert info.port == 5432
        assert info.username == "user@example.com"
        assert info.token == "my-token"