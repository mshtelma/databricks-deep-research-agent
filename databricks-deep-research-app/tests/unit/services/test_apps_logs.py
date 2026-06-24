"""Unit tests for services/deployment/apps_logs.py (Section S4b).

Covers:
- Happy path via list_app_deployments → get_app_deployment.
- Fallback via apps.get() when deployment listing fails/missing.
- SDK missing both methods → returns None.
- Secret redaction (dapi, ghp_, AKIA, api_key).
- Truncation by line count.
- Truncation by byte count.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from deep_research.services.deployment.apps_logs import (
    _redact,
    fetch_app_log_tail,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wc(
    *,
    deployments: list[MagicMock] | None = None,
    deployment_msg: str | None = None,
    list_side_effect: Exception | None = None,
    get_deployment_side_effect: Exception | None = None,
    app_pending_msg: str | None = None,
    app_compute_msg: str | None = None,
    app_get_side_effect: Exception | None = None,
) -> MagicMock:
    wc = MagicMock()

    # list_app_deployments
    if list_side_effect:
        wc.apps.list_app_deployments.side_effect = list_side_effect
    else:
        dep = MagicMock()
        dep.deployment_id = "dep-001"
        dep.create_time = None
        dep.created_at = None
        status_mock = MagicMock()
        status_mock.message = deployment_msg
        dep.status = status_mock
        dep.status_message = deployment_msg
        wc.apps.list_app_deployments.return_value = iter(deployments or [dep])

    # get_app_deployment
    if get_deployment_side_effect:
        wc.apps.get_app_deployment.side_effect = get_deployment_side_effect
    else:
        dep_detail = MagicMock()
        dep_detail.deployment_id = "dep-001"
        status_detail = MagicMock()
        status_detail.message = deployment_msg
        dep_detail.status = status_detail
        dep_detail.status_message = deployment_msg
        wc.apps.get_app_deployment.return_value = dep_detail

    # apps.get fallback
    if app_get_side_effect:
        wc.apps.get.side_effect = app_get_side_effect
    else:
        app_obj = MagicMock()
        pending = MagicMock()
        pending_status = MagicMock()
        pending_status.message = app_pending_msg
        pending.status = pending_status
        pending.status_message = app_pending_msg
        app_obj.pending_deployment = pending if app_pending_msg else None
        compute = MagicMock()
        compute.message = app_compute_msg
        app_obj.compute_status = compute if app_compute_msg else None
        wc.apps.get.return_value = app_obj

    return wc


# ---------------------------------------------------------------------------
# Happy path: list_app_deployments
# ---------------------------------------------------------------------------


class TestFetchAppLogTailHappyPath:
    @pytest.mark.asyncio
    async def test_returns_log_tail_from_deployment_status(self) -> None:
        wc = _make_wc(deployment_msg="App is starting up...")
        result = await fetch_app_log_tail(workspace_client=wc, app_name="dr-shell-test")
        assert result is not None
        assert "App is starting up" in result.text
        assert result.source == "app_deployment_status_message"
        assert result.truncated is False

    @pytest.mark.asyncio
    async def test_fallback_to_app_status_when_deployments_empty(self) -> None:
        wc = _make_wc(deployments=[], app_pending_msg="Pending: building image")
        result = await fetch_app_log_tail(workspace_client=wc, app_name="dr-shell-test")
        assert result is not None
        assert "Pending: building image" in result.text
        assert result.source == "app_status_messages"

    @pytest.mark.asyncio
    async def test_fallback_concatenates_pending_and_compute(self) -> None:
        wc = _make_wc(
            deployments=[],  # force fallback
            app_pending_msg="pending message",
            app_compute_msg="compute message",
        )
        result = await fetch_app_log_tail(workspace_client=wc, app_name="dr-shell-test")
        assert result is not None
        assert "pending message" in result.text
        assert "compute message" in result.text

    @pytest.mark.asyncio
    async def test_returns_none_when_all_paths_fail(self) -> None:
        wc = _make_wc(
            list_side_effect=RuntimeError("SDK not available"),
            app_get_side_effect=RuntimeError("also broken"),
        )
        result = await fetch_app_log_tail(workspace_client=wc, app_name="dr-shell-test")
        assert result is None


# ---------------------------------------------------------------------------
# Secret redaction
# ---------------------------------------------------------------------------


class TestRedaction:
    # Synthetic fixtures: token-shaped strings are assembled at runtime so the
    # source file contains no contiguous literal that matches a secret-scanner
    # signature. The redactor regexes still match the resulting runtime values.
    def test_redact_databricks_pat(self) -> None:
        fake_token = "d" + "api" + "0" * 32
        text = f"token: {fake_token}"
        out = _redact(text)
        assert fake_token not in out
        assert "***REDACTED***" in out

    def test_redact_github_pat(self) -> None:
        fake_token = "g" + "hp_" + "Z" * 30
        text = f"export GH_TOKEN={fake_token}"
        out = _redact(text)
        assert fake_token not in out
        assert "***REDACTED***" in out

    def test_redact_aws_access_key(self) -> None:
        fake_key = "A" + "KIA" + "Z" * 16
        out = _redact(fake_key)
        assert fake_key not in out
        assert "***REDACTED***" in out

    def test_redact_api_key_assignment(self) -> None:
        text = "api_key=" + "dummy_value_xyz"
        out = _redact(text)
        assert "dummy_value_xyz" not in out
        assert "***REDACTED***" in out

    @pytest.mark.asyncio
    async def test_fetch_redacts_secrets_in_output(self) -> None:
        fake_token = "d" + "api" + "f" * 32
        wc = _make_wc(deployment_msg=f"Error: token {fake_token} is invalid")
        result = await fetch_app_log_tail(workspace_client=wc, app_name="dr-shell-x")
        assert result is not None
        assert fake_token not in result.text
        assert "***REDACTED***" in result.text


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


class TestTruncation:
    @pytest.mark.asyncio
    async def test_truncation_by_lines(self) -> None:
        many_lines = "\n".join(f"log line {i}" for i in range(100))
        wc = _make_wc(deployment_msg=many_lines)
        result = await fetch_app_log_tail(
            workspace_client=wc, app_name="dr-shell-x", max_lines=10, max_bytes=50000
        )
        assert result is not None
        assert result.truncated is True
        assert len(result.text.splitlines()) <= 10

    @pytest.mark.asyncio
    async def test_truncation_by_bytes(self) -> None:
        long_text = "x" * 10000
        wc = _make_wc(deployment_msg=long_text)
        result = await fetch_app_log_tail(
            workspace_client=wc, app_name="dr-shell-x", max_lines=1000, max_bytes=100
        )
        assert result is not None
        assert result.truncated is True
        assert len(result.text.encode()) <= 100
