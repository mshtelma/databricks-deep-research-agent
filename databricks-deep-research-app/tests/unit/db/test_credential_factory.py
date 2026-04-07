"""Tests for credential factory and auto-detection."""

from unittest.mock import MagicMock, patch

from deep_research.db.credential_factory import (
    create_credential_provider,
    detect_lakebase_backend,
)


class TestDetectLakebaseBackend:
    """Tests for auto-detection of Lakebase backend type."""

    def test_autoscaling_via_settings(self) -> None:
        """ENDPOINT_NAME in settings → Autoscaling."""
        settings = MagicMock()
        settings.endpoint_name = "projects/abc/branches/def/endpoints/ghi"
        settings.lakebase_instance_name = None

        with patch.dict("os.environ", {}, clear=True):
            result = detect_lakebase_backend(settings)

        assert result == "autoscaling"

    def test_autoscaling_via_env(self) -> None:
        """ENDPOINT_NAME env var → Autoscaling."""
        settings = MagicMock()
        settings.endpoint_name = None
        settings.lakebase_instance_name = None

        with patch.dict("os.environ", {"ENDPOINT_NAME": "projects/x/branches/y/endpoints/z"}):
            result = detect_lakebase_backend(settings)

        assert result == "autoscaling"

    def test_provisioned_via_settings(self) -> None:
        """LAKEBASE_INSTANCE_NAME in settings → Provisioned."""
        settings = MagicMock()
        settings.endpoint_name = None
        settings.lakebase_instance_name = "deep-research-lakebase"

        with patch.dict("os.environ", {}, clear=True):
            result = detect_lakebase_backend(settings)

        assert result == "provisioned"

    def test_provisioned_via_pghost(self) -> None:
        """PGHOST env var → Provisioned."""
        settings = MagicMock()
        settings.endpoint_name = None
        settings.lakebase_instance_name = None

        with patch.dict("os.environ", {"PGHOST": "instance-abc.database.cloud.databricks.com"}):
            result = detect_lakebase_backend(settings)

        assert result == "provisioned"

    def test_no_backend(self) -> None:
        """No ENDPOINT_NAME or LAKEBASE_INSTANCE_NAME → None."""
        settings = MagicMock()
        settings.endpoint_name = None
        settings.lakebase_instance_name = None

        with patch.dict("os.environ", {}, clear=True):
            result = detect_lakebase_backend(settings)

        assert result is None

    def test_conflict_autoscaling_wins(self) -> None:
        """Both ENDPOINT_NAME and LAKEBASE_INSTANCE_NAME → Autoscaling with warning."""
        settings = MagicMock()
        settings.endpoint_name = "projects/abc/branches/def/endpoints/ghi"
        settings.lakebase_instance_name = "deep-research-lakebase"

        with patch.dict("os.environ", {}, clear=True):
            result = detect_lakebase_backend(settings)

        assert result == "autoscaling"


class TestCreateCredentialProvider:
    """Tests for factory function."""

    @patch("deep_research.db.credential_factory.detect_lakebase_backend")
    def test_creates_provisioned_provider(self, mock_detect: MagicMock) -> None:
        """Provisioned backend → LakebaseCredentialProvider."""
        mock_detect.return_value = "provisioned"
        settings = MagicMock()
        settings.lakebase_instance_name = "test-instance"

        with patch(
            "deep_research.db.lakebase_auth.LakebaseCredentialProvider"
        ) as mock_cls:
            provider = create_credential_provider(settings)

        mock_cls.assert_called_once_with(settings)
        assert provider == mock_cls.return_value

    @patch("deep_research.db.credential_factory.detect_lakebase_backend")
    def test_creates_autoscaling_provider(self, mock_detect: MagicMock) -> None:
        """Autoscaling backend → AutoscalingCredentialProvider."""
        mock_detect.return_value = "autoscaling"
        settings = MagicMock()
        settings.endpoint_name = "projects/abc/branches/def/endpoints/ghi"

        with patch(
            "deep_research.db.autoscaling_auth.AutoscalingCredentialProvider"
        ) as mock_cls:
            provider = create_credential_provider(settings)

        mock_cls.assert_called_once_with(settings)
        assert provider == mock_cls.return_value

    @patch("deep_research.db.credential_factory.detect_lakebase_backend")
    def test_returns_none_when_no_backend(self, mock_detect: MagicMock) -> None:
        """No backend → None."""
        mock_detect.return_value = None
        settings = MagicMock()

        provider = create_credential_provider(settings)

        assert provider is None
