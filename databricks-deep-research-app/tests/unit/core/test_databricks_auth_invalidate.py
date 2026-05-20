"""Unit tests for DatabricksAuth.invalidate().

Verifies that ``invalidate()`` drops all cached state so the next call
rebuilds from scratch — the defence against a poisoned SDK-side token
cache that landed in the "Invalid Token" 403 fix.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from deep_research.core.databricks_auth import (
    DatabricksAuth,
    clear_databricks_auth,
)


class TestDatabricksAuthInvalidate:
    """Tests for DatabricksAuth.invalidate()."""

    def teardown_method(self) -> None:
        """Clear singleton so each test gets a fresh DatabricksAuth."""
        clear_databricks_auth()

    @patch("deep_research.core.databricks_auth.get_settings")
    def test_invalidate_clears_all_cached_state_direct_token(
        self, mock_settings: MagicMock
    ) -> None:
        """direct_token mode: invalidate clears _credential / _client / _base_url."""
        mock_settings.return_value.databricks_token = "tok"
        mock_settings.return_value.databricks_host = "https://h.databricks.com"
        mock_settings.return_value.databricks_config_profile = None
        mock_settings.return_value.is_databricks_app = False

        auth = DatabricksAuth()
        # Direct token has no _credential, but get_base_url caches _base_url
        _ = auth.get_base_url()
        assert auth._base_url is not None

        auth.invalidate()

        assert auth._credential is None
        assert auth._client is None
        assert auth._base_url is None

    @patch("deep_research.core.databricks_auth.get_settings")
    @patch("deep_research.core.databricks_auth.WorkspaceClient")
    def test_invalidate_clears_all_cached_state_oauth(
        self, mock_wc_class: MagicMock, mock_settings: MagicMock
    ) -> None:
        """profile (OAuth) mode: invalidate clears every populated field."""
        mock_settings.return_value.databricks_token = None
        mock_settings.return_value.databricks_host = None
        mock_settings.return_value.databricks_config_profile = "p"
        mock_settings.return_value.is_databricks_app = False

        mock_wc = MagicMock()
        mock_wc.config.host = "https://w.databricks.com"
        mock_wc.config.authenticate.return_value = {"Authorization": "Bearer tok"}
        mock_wc_class.return_value = mock_wc

        auth = DatabricksAuth()
        _ = auth.get_client()
        _ = auth.get_token()
        _ = auth.get_base_url()
        assert auth._client is not None
        assert auth._credential is not None
        assert auth._base_url is not None

        auth.invalidate()

        assert auth._credential is None
        assert auth._client is None
        assert auth._base_url is None

    @patch("deep_research.core.databricks_auth.get_settings")
    @patch("deep_research.core.databricks_auth.WorkspaceClient")
    def test_invalidate_then_get_token_mints_new_credential(
        self, mock_wc_class: MagicMock, mock_settings: MagicMock
    ) -> None:
        """After invalidate, the next get_token rebuilds via authenticate()."""
        mock_settings.return_value.databricks_token = None
        mock_settings.return_value.databricks_host = None
        mock_settings.return_value.databricks_config_profile = "p"
        mock_settings.return_value.is_databricks_app = False

        mock_wc = MagicMock()
        mock_wc.config.host = "https://w.databricks.com"
        mock_wc.config.authenticate.return_value = {"Authorization": "Bearer tok-A"}
        mock_wc_class.return_value = mock_wc

        auth = DatabricksAuth()
        assert auth.get_token() == "tok-A"

        # Rotate the SDK-side response and invalidate
        mock_wc.config.authenticate.return_value = {"Authorization": "Bearer tok-B"}
        auth.invalidate()

        # Next get_token rebuilds WorkspaceClient AND re-authenticates
        assert auth.get_token() == "tok-B"
        # WorkspaceClient was constructed twice: initial + after invalidate
        assert mock_wc_class.call_count == 2

    @patch("deep_research.core.databricks_auth.get_settings")
    def test_invalidate_is_idempotent(self, mock_settings: MagicMock) -> None:
        """Calling invalidate() repeatedly does not raise."""
        mock_settings.return_value.databricks_token = "tok"
        mock_settings.return_value.databricks_host = "https://h.databricks.com"
        mock_settings.return_value.databricks_config_profile = None
        mock_settings.return_value.is_databricks_app = False

        auth = DatabricksAuth()
        auth.invalidate()
        auth.invalidate()  # no-op, must not raise

        assert auth._credential is None
        assert auth._client is None
        assert auth._base_url is None

    @patch("deep_research.core.databricks_auth.get_settings")
    def test_invalidate_distinct_from_clear_databricks_auth(
        self, mock_settings: MagicMock
    ) -> None:
        """invalidate() is instance-level; clear_databricks_auth() is module-level.

        After invalidate(), the singleton STILL holds the same DatabricksAuth
        instance. After clear_databricks_auth(), get_databricks_auth() builds
        a NEW one.
        """
        mock_settings.return_value.databricks_token = "tok"
        mock_settings.return_value.databricks_host = "https://h.databricks.com"
        mock_settings.return_value.databricks_config_profile = None
        mock_settings.return_value.is_databricks_app = False

        from deep_research.core.databricks_auth import get_databricks_auth

        auth1 = get_databricks_auth()
        auth1.invalidate()
        # Same instance returned by the singleton accessor
        assert get_databricks_auth() is auth1

        clear_databricks_auth()
        # Now a fresh instance is built
        auth2 = get_databricks_auth()
        assert auth2 is not auth1
