"""Tests for credential factory and auto-detection."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from deep_research.db.credential_factory import (
    _meaningful,
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


class TestPlaceholderHandling:
    """Bundle template placeholders (``pending``, ``tbd``, empty) must never
    flow into backend selection."""

    def test_pending_endpoint_and_pghost_is_no_backend(
        self, caplog: "pytest.LogCaptureFixture"
    ) -> None:
        """Both env vars set to the placeholder string → no backend selected,
        with WARNING logs naming each source."""
        settings = MagicMock()
        settings.endpoint_name = None
        settings.lakebase_instance_name = None

        with caplog.at_level(logging.WARNING, logger="deep_research.db.credential_factory"):
            with patch.dict("os.environ", {"ENDPOINT_NAME": "pending", "PGHOST": "pending"}):
                result = detect_lakebase_backend(settings)

        assert result is None
        # One WARNING per filtered source so the operator can see both.
        names = [r.message for r in caplog.records]
        assert any("name=ENDPOINT_NAME" in m for m in names), names
        assert any("name=PGHOST" in m for m in names), names

    def test_pending_endpoint_falls_through_to_provisioned(self) -> None:
        """ENDPOINT_NAME=pending must NOT pin the autoscaling path; if a real
        PGHOST is also set, provisioned wins."""
        settings = MagicMock()
        settings.endpoint_name = None
        settings.lakebase_instance_name = None

        with patch.dict(
            "os.environ",
            {"ENDPOINT_NAME": "pending", "PGHOST": "ep-x.database.cloud.databricks.com"},
        ):
            result = detect_lakebase_backend(settings)

        assert result == "provisioned"

    def test_real_endpoint_with_pending_pghost_picks_autoscaling(self) -> None:
        """A real ENDPOINT_NAME plus a placeholder PGHOST → autoscaling,
        no conflict warning."""
        settings = MagicMock()
        settings.endpoint_name = None
        settings.lakebase_instance_name = None

        with patch.dict(
            "os.environ",
            {
                "ENDPOINT_NAME": "projects/p/branches/b/endpoints/primary",
                "PGHOST": "pending",
            },
        ):
            result = detect_lakebase_backend(settings)

        assert result == "autoscaling"

    def test_meaningful_filters_empty_and_whitespace(self) -> None:
        """Empty strings and whitespace-only values are placeholders too."""
        assert _meaningful(None) is None
        assert _meaningful("") is None
        assert _meaningful("   ") is None
        assert _meaningful("\t\n") is None

    def test_meaningful_filters_known_placeholders_case_insensitively(self) -> None:
        """``pending``, ``tbd``, ``todo`` in any case are placeholders."""
        for raw in ("pending", "PENDING", "Pending", "tbd", "TBD", "todo", "TODO"):
            assert _meaningful(raw) is None, raw

    def test_meaningful_preserves_real_values_and_strips(self) -> None:
        """Real values pass through stripped."""
        assert (
            _meaningful("  projects/p/branches/b/endpoints/primary  ")
            == "projects/p/branches/b/endpoints/primary"
        )
        # "pendingx" is a real value, not the placeholder.
        assert _meaningful("pendingx") == "pendingx"


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
