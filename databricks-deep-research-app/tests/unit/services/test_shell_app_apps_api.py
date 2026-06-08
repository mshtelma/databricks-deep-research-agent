"""Unit tests for services/deployment/shell_app_apps_api.py (Section S3 + S4).

Covers:
- S3 owner == deployer → apps.update called.
- S3 owner != deployer → app_name_collision result returned, workspace cleaned up.
- S4a tag preflight 404 → returns framework_tag_unreachable BEFORE workspace.upload.
- S4b reachability timeout → returns reachability_timeout with last_logs.
"""

from __future__ import annotations

import io
import zipfile
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from deep_research.models.agent_deployment import AgentDeployment, DeploymentMode
from deep_research.services.deployment.translator import Artifact

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_valid_zip() -> bytes:
    """Create a minimal in-memory zip that passes the size check."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("app.py", "# shell app")
    return buf.getvalue()


def _make_artifact(
    payload: bytes | None = None,
    *,
    metadata: dict[str, Any] | None = None,
) -> Artifact:
    return Artifact(
        mode=DeploymentMode.SHELL_APP,
        payload=payload if payload is not None else _make_valid_zip(),
        metadata=metadata,
    )


def _make_deployment(*, config: dict[str, Any] | None = None) -> MagicMock:
    d = MagicMock(spec=AgentDeployment)
    d.id = uuid4()
    d.config = config or {
        "mode": "shell_app",
        "app_name": "dr-shell-test",
        "framework_git_tag": "v1.0.0",
    }
    return d


def _make_workspace_client() -> MagicMock:
    wc = MagicMock()
    wc.current_user.me.return_value = MagicMock(user_name="alice@acme.com")
    wc.workspace.mkdirs.return_value = None
    wc.workspace.upload.return_value = None
    wc.workspace.delete.return_value = None
    # apps.create — succeeds by default
    app_obj = MagicMock()
    app_obj.url = "https://adb-123.net/apps/dr-shell-test"
    app_obj.compute_status = MagicMock(state="RUNNING")
    wc.apps.create.return_value.result.return_value = app_obj
    wc.apps.deploy.return_value.result.return_value = MagicMock()
    wc.apps.get.return_value = app_obj
    return wc


def _patch_source_client(wc: MagicMock) -> Any:
    return patch(
        "deep_research.core.databricks_auth.get_databricks_auth",
        return_value=MagicMock(get_client=MagicMock(return_value=wc)),
    )


def _make_settings(**overrides: object) -> MagicMock:
    settings = MagicMock()
    settings.deploy_here_framework_tag_preflight = True
    settings.framework_git_url = "https://github.com/owner/repo"
    settings.github_api_token = None
    settings.deploy_here_disclose_owner = True
    settings.deploy_here_reachability_timeout_seconds = 300.0
    settings.deploy_here_brave_secret_scope = "deep-research-secrets"
    settings.deploy_here_brave_secret_key = "BRAVE_API_KEY"
    for key, value in overrides.items():
        setattr(settings, key, value)
    return settings


def _patch_settings(**overrides: object) -> Any:
    return patch(
        "deep_research.services.deployment.shell_app_apps_api.get_settings",
        return_value=_make_settings(**overrides),
    )


def _reachability_result(
    *,
    reached: bool,
    timed_out: bool,
    failed: bool = False,
    last_state: str | None = None,
    last_message: str | None = None,
) -> Any:
    from deep_research.services.deployment.shell_app_apps_api import (
        ReachabilityProbeResult,
    )

    return ReachabilityProbeResult(
        reached=reached,
        timed_out=timed_out,
        failed=failed,
        last_state=last_state,
        last_message=last_message,
    )


# ---------------------------------------------------------------------------
# S3 — collision: owner == deployer → apps.deploy called
# ---------------------------------------------------------------------------


class TestAppNameSelection:
    @pytest.mark.asyncio
    async def test_uses_configured_app_name_for_apps_api(self) -> None:
        """Live deploy must use config.app_name, not the full deployment UUID."""
        from deep_research.services.deployment.shell_app_apps_api import (
            _deploy_via_apps_api,
        )

        wc = _make_workspace_client()
        config = {
            "mode": "shell_app",
            "app_name": "dr-shell-ui",
            "framework_git_tag": "v1.0.0",
        }
        deployment = _make_deployment(config=config)

        with (
            patch(
                "deep_research.services.deployment.shell_app_apps_api.probe_framework_tag",
                new=AsyncMock(return_value=MagicMock(reachable=True)),
            ),
            patch(
                "deep_research.services.deployment.shell_app_apps_api._probe_app_reachability_with_timeout",
                new=AsyncMock(return_value=_reachability_result(reached=True, timed_out=False)),
            ),
            _patch_settings(),
            _patch_source_client(wc),
        ):
            result = await _deploy_via_apps_api(
                _make_artifact(),
                config,
                deployment,
                wc,
            )

        assert result.success is True
        assert result.endpoint_name == "dr-shell-ui"
        assert result.external_resource_ids is not None
        assert result.external_resource_ids["app_name"] == "dr-shell-ui"
        create_arg = wc.apps.create.call_args.args[0]
        assert create_arg.name == "dr-shell-ui"
        assert create_arg.user_api_scopes == [
            "sql",
            "serving.serving-endpoints",
            "vectorsearch.vector-search-endpoints",
            "vectorsearch.vector-search-indexes",
            "dashboards.genie",
        ]
        assert wc.apps.deploy.call_args.kwargs["app_name"] == "dr-shell-ui"
        deployment_arg = wc.apps.deploy.call_args.kwargs["app_deployment"]
        assert [env.name for env in deployment_arg.env_vars] == [
            "MLFLOW_ENABLED",
            "MLFLOW_TRACKING_URI",
            "SHELL_APP_SSE_HEARTBEAT_SECONDS",
        ]
        wc.apps.get.assert_called_with(name="dr-shell-ui")

    def test_fallback_app_name_is_bounded_to_apps_limit(self) -> None:
        from deep_research.services.deployment.shell_app_apps_api import (
            _fallback_app_name,
        )

        name = _fallback_app_name("a528c6b0-59bb-4215-87b4-9de5116276c8")

        assert name.startswith("dr-shell-")
        assert len(name) <= 30

    @pytest.mark.asyncio
    async def test_web_search_apps_api_binds_brave_secret_resource_and_env(self) -> None:
        """Inline deploy must pass secret resources/env vars to the Apps SDK."""
        from deep_research.services.deployment.shell_app_apps_api import (
            _deploy_via_apps_api,
        )

        wc = _make_workspace_client()
        config = {
            "mode": "shell_app",
            "app_name": "dr-shell-web",
            "framework_git_tag": "v1.0.0",
        }
        artifact = _make_artifact(
            metadata={
                "requires_web_search": "true",
                "brave_secret_resource_name": "brave-api-key",
            }
        )

        with (
            patch(
                "deep_research.services.deployment.shell_app_apps_api.probe_framework_tag",
                new=AsyncMock(return_value=MagicMock(reachable=True)),
            ),
            patch(
                "deep_research.services.deployment.shell_app_apps_api._probe_app_reachability_with_timeout",
                new=AsyncMock(return_value=_reachability_result(reached=True, timed_out=False)),
            ),
            _patch_settings(),
            _patch_source_client(wc),
        ):
            result = await _deploy_via_apps_api(
                artifact,
                config,
                _make_deployment(config=config),
                wc,
            )

        assert result.success is True
        create_arg = wc.apps.create.call_args.args[0]
        assert create_arg.resources is not None
        secret_resource = create_arg.resources[0]
        assert secret_resource.name == "brave-api-key"
        assert secret_resource.secret.scope == "deep-research-secrets"
        assert secret_resource.secret.key == "BRAVE_API_KEY"
        assert str(secret_resource.secret.permission).endswith("READ")

        deployment_arg = wc.apps.deploy.call_args.kwargs["app_deployment"]
        env_by_name = {env.name: env for env in deployment_arg.env_vars}
        assert env_by_name["MLFLOW_ENABLED"].value == "false"
        assert env_by_name["MLFLOW_TRACKING_URI"].value == "databricks"
        assert env_by_name["SHELL_APP_SSE_HEARTBEAT_SECONDS"].value == "15"
        assert env_by_name["BRAVE_API_KEY"].value_from == "brave-api-key"


class TestPermissionDenied:
    @pytest.mark.asyncio
    async def test_apps_create_permission_denied_returns_missing_workspace_permission(
        self,
    ) -> None:
        """Real Apps API permission failures are surfaced from create/deploy."""
        from deep_research.services.deployment.shell_app_apps_api import (
            _deploy_via_apps_api,
        )

        PermissionDenied = type("PermissionDenied", (Exception,), {})
        wc = _make_workspace_client()
        wc.apps.create.return_value.result.side_effect = PermissionDenied("403 permission denied")

        with (
            patch(
                "deep_research.services.deployment.shell_app_apps_api.probe_framework_tag",
                new=AsyncMock(return_value=MagicMock(reachable=True)),
            ),
            _patch_settings(),
            _patch_source_client(wc),
        ):
            result = await _deploy_via_apps_api(
                _make_artifact(),
                {
                    "mode": "shell_app",
                    "app_name": "dr-shell-test",
                    "framework_git_tag": "v1.0.0",
                },
                _make_deployment(),
                wc,
            )

        assert result.success is False
        assert result.external_resource_ids is not None
        assert result.external_resource_ids.get("error_kind") == "missing_workspace_permission"
        wc.workspace.delete.assert_called()


class TestCollisionOwnerIsDeployer:
    @pytest.mark.asyncio
    async def test_owner_is_deployer_calls_apps_deploy(self) -> None:
        """When AlreadyExists + owner == deployer, apps.deploy must be called."""
        from deep_research.services.deployment.shell_app_apps_api import (
            _deploy_via_apps_api,
        )

        AlreadyExists = type("AlreadyExists", (Exception,), {})
        wc = _make_workspace_client()
        # Simulate apps.create raising AlreadyExists once
        wc.apps.create.return_value.result.side_effect = [AlreadyExists("already exists")]

        # apps_collision.resolve returns deployer_can_redeploy=True
        from deep_research.services.deployment.apps_collision import AppOwnershipCheck

        fake_check = AppOwnershipCheck(
            deployer_can_redeploy=True,
            existing_owner="alice@acme.com",
            failure_reason=None,
        )

        with (
            patch(
                "deep_research.services.deployment.shell_app_apps_api.resolve_apps_already_exists",
                new=AsyncMock(return_value=fake_check),
            ),
            patch(
                "deep_research.services.deployment.shell_app_apps_api.probe_framework_tag",
                new=AsyncMock(return_value=MagicMock(reachable=True)),
            ),
            patch(
                "deep_research.services.deployment.shell_app_apps_api._probe_app_reachability_with_timeout",
                new=AsyncMock(return_value=_reachability_result(reached=True, timed_out=False)),
            ),
            _patch_settings(),
            _patch_source_client(wc),
        ):
            result = await _deploy_via_apps_api(
                _make_artifact(),
                {"mode": "shell_app", "app_name": "dr-shell-test", "framework_git_tag": "v1.0.0"},
                _make_deployment(),
                wc,
            )

        assert result.success is True
        wc.apps.update.assert_called_once()
        assert wc.apps.update.call_args.kwargs["name"] == "dr-shell-test"
        update_app = wc.apps.update.call_args.kwargs["app"]
        assert update_app.user_api_scopes == [
            "sql",
            "serving.serving-endpoints",
            "vectorsearch.vector-search-endpoints",
            "vectorsearch.vector-search-indexes",
            "dashboards.genie",
        ]
        wc.apps.deploy.return_value.result.assert_called()


# ---------------------------------------------------------------------------
# S3 — collision: owner != deployer → app_name_collision
# ---------------------------------------------------------------------------


class TestCollisionOwnerIsOther:
    @pytest.mark.asyncio
    async def test_other_owner_returns_app_name_collision(self) -> None:
        """When AlreadyExists + deployer_can_redeploy=False, return app_name_collision."""
        from deep_research.services.deployment.shell_app_apps_api import (
            _deploy_via_apps_api,
        )

        AlreadyExists = type("AlreadyExists", (Exception,), {})
        wc = _make_workspace_client()
        wc.apps.create.return_value.result.side_effect = [AlreadyExists("already exists")]

        from deep_research.services.deployment.apps_collision import AppOwnershipCheck

        fake_check = AppOwnershipCheck(
            deployer_can_redeploy=False,
            existing_owner="bob@acme.com",
            failure_reason=None,
        )

        with (
            patch(
                "deep_research.services.deployment.shell_app_apps_api.resolve_apps_already_exists",
                new=AsyncMock(return_value=fake_check),
            ),
            patch(
                "deep_research.services.deployment.shell_app_apps_api.probe_framework_tag",
                new=AsyncMock(return_value=MagicMock(reachable=True)),
            ),
            _patch_settings(),
            _patch_source_client(wc),
        ):
            result = await _deploy_via_apps_api(
                _make_artifact(),
                {"mode": "shell_app", "app_name": "dr-shell-test", "framework_git_tag": "v1.0.0"},
                _make_deployment(),
                wc,
            )

        assert result.success is False
        assert result.external_resource_ids is not None
        assert result.external_resource_ids.get("error_kind") == "app_name_collision"
        # Suggested name should be present
        assert "suggested_name" in result.external_resource_ids
        # Workspace files must have been cleaned up
        wc.workspace.delete.assert_called()


# ---------------------------------------------------------------------------
# S4a — tag preflight 404 → framework_tag_unreachable BEFORE upload
# ---------------------------------------------------------------------------


class TestTagPreflight:
    @pytest.mark.asyncio
    async def test_404_tag_returns_framework_tag_unreachable_before_upload(self) -> None:
        """probe_framework_tag returning reachable=False must short-circuit
        before workspace.upload is called."""
        from deep_research.services.deployment.github_tag_probe import TagProbeResult
        from deep_research.services.deployment.shell_app_apps_api import (
            _deploy_via_apps_api,
        )

        wc = _make_workspace_client()
        unreachable = TagProbeResult(
            reachable=False,
            error_kind="framework_tag_unreachable",
            note="tag_not_found:v0.0.0",
        )

        with (
            patch(
                "deep_research.services.deployment.shell_app_apps_api.probe_framework_tag",
                new=AsyncMock(return_value=unreachable),
            ),
            _patch_settings(),
        ):
            result = await _deploy_via_apps_api(
                _make_artifact(),
                {"mode": "shell_app", "app_name": "dr-shell-test", "framework_git_tag": "v0.0.0"},
                _make_deployment(),
                wc,
            )

        assert result.success is False
        assert result.external_resource_ids is not None
        assert result.external_resource_ids.get("error_kind") == "framework_tag_unreachable"
        # workspace.upload must NOT have been called
        wc.workspace.upload.assert_not_called()

    @pytest.mark.asyncio
    async def test_preflight_disabled_skips_probe(self) -> None:
        """When deploy_here_framework_tag_preflight=False, probe is skipped."""
        from deep_research.services.deployment.shell_app_apps_api import (
            _deploy_via_apps_api,
        )

        wc = _make_workspace_client()
        mock_probe = AsyncMock()

        with (
            _patch_settings(deploy_here_framework_tag_preflight=False),
            patch(
                "deep_research.services.deployment.shell_app_apps_api.probe_framework_tag",
                new=mock_probe,
            ),
            patch(
                "deep_research.services.deployment.shell_app_apps_api._probe_app_reachability_with_timeout",
                new=AsyncMock(return_value=_reachability_result(reached=True, timed_out=False)),
            ),
            _patch_source_client(wc),
        ):
            await _deploy_via_apps_api(
                _make_artifact(),
                {"mode": "shell_app", "app_name": "dr-shell-test", "framework_git_tag": "v1.0.0"},
                _make_deployment(),
                wc,
            )

        mock_probe.assert_not_called()


# ---------------------------------------------------------------------------
# S4b — reachability timeout → reachability_timeout with last_logs
# ---------------------------------------------------------------------------


class TestReachabilityStates:
    @pytest.mark.asyncio
    async def test_active_state_is_reachable(self) -> None:
        """Databricks Apps reports healthy compute as ACTIVE in AIS logs."""
        from deep_research.services.deployment.shell_app_apps_api import (
            _probe_app_reachability_with_timeout,
        )

        wc = MagicMock()
        app = MagicMock()
        app.compute_status = MagicMock(
            state="ACTIVE",
            message="App compute is running.",
        )
        wc.apps.get.return_value = app

        with _patch_settings(deploy_here_reachability_timeout_seconds=1.0):
            result = await _probe_app_reachability_with_timeout(wc, "dr-shell-test")

        assert result.reached is True
        assert result.timed_out is False
        assert result.last_state == "ACTIVE"
        assert result.last_message == "App compute is running."


class TestReachabilityTimeout:
    @pytest.mark.asyncio
    async def test_timeout_returns_reachability_timeout_with_logs(self) -> None:
        """When the reachability probe times out, the result should have
        error_kind=reachability_timeout and last_logs from fetch_app_log_tail."""
        from deep_research.services.deployment.apps_logs import AppLogTail
        from deep_research.services.deployment.shell_app_apps_api import (
            _deploy_via_apps_api,
        )

        wc = _make_workspace_client()
        tail = AppLogTail(
            text="Error: pip install failed",
            truncated=False,
            source="app_deployment_status_message",
        )

        with (
            patch(
                "deep_research.services.deployment.shell_app_apps_api.probe_framework_tag",
                new=AsyncMock(return_value=MagicMock(reachable=True)),
            ),
            patch(
                "deep_research.services.deployment.shell_app_apps_api._probe_app_reachability_with_timeout",
                new=AsyncMock(
                    return_value=_reachability_result(
                        reached=False,
                        timed_out=True,
                    )
                ),
            ),
            patch(
                "deep_research.services.deployment.shell_app_apps_api.fetch_app_log_tail",
                new=AsyncMock(return_value=tail),
            ),
            _patch_settings(),
            _patch_source_client(wc),
        ):
            result = await _deploy_via_apps_api(
                _make_artifact(),
                {"mode": "shell_app", "app_name": "dr-shell-test", "framework_git_tag": "v1.0.0"},
                _make_deployment(),
                wc,
            )

        assert result.success is False
        assert result.external_resource_ids is not None
        assert result.external_resource_ids.get("error_kind") == "reachability_timeout"
        assert result.external_resource_ids.get("last_logs") == "Error: pip install failed"
        assert result.external_resource_ids.get("logs_truncated") is False

    @pytest.mark.asyncio
    async def test_timeout_with_no_logs_still_returns_error(self) -> None:
        """When log fetch returns None, reachability_timeout is still returned."""
        from deep_research.services.deployment.shell_app_apps_api import (
            _deploy_via_apps_api,
        )

        wc = _make_workspace_client()

        with (
            patch(
                "deep_research.services.deployment.shell_app_apps_api.probe_framework_tag",
                new=AsyncMock(return_value=MagicMock(reachable=True)),
            ),
            patch(
                "deep_research.services.deployment.shell_app_apps_api._probe_app_reachability_with_timeout",
                new=AsyncMock(
                    return_value=_reachability_result(
                        reached=False,
                        timed_out=True,
                    )
                ),
            ),
            patch(
                "deep_research.services.deployment.shell_app_apps_api.fetch_app_log_tail",
                new=AsyncMock(return_value=None),
            ),
            _patch_settings(),
            _patch_source_client(wc),
        ):
            result = await _deploy_via_apps_api(
                _make_artifact(),
                {"mode": "shell_app", "app_name": "dr-shell-test", "framework_git_tag": "v1.0.0"},
                _make_deployment(),
                wc,
            )

        assert result.success is False
        assert result.external_resource_ids is not None
        assert result.external_resource_ids.get("error_kind") == "reachability_timeout"
        assert result.external_resource_ids.get("last_logs") is None

    @pytest.mark.asyncio
    async def test_failed_state_returns_reachability_failed_with_logs(self) -> None:
        """Terminal app states should fail fast instead of waiting the full timeout."""
        from deep_research.services.deployment.apps_logs import AppLogTail
        from deep_research.services.deployment.shell_app_apps_api import (
            ReachabilityProbeResult,
            _deploy_via_apps_api,
        )

        wc = _make_workspace_client()
        tail = AppLogTail(
            text="ValueError: direct references are not allowed",
            truncated=False,
            source="app_status_messages",
        )

        with (
            patch(
                "deep_research.services.deployment.shell_app_apps_api.probe_framework_tag",
                new=AsyncMock(return_value=MagicMock(reachable=True)),
            ),
            patch(
                "deep_research.services.deployment.shell_app_apps_api._probe_app_reachability_with_timeout",
                new=AsyncMock(
                    return_value=ReachabilityProbeResult(
                        reached=False,
                        timed_out=False,
                        failed=True,
                        last_state="ERROR",
                        last_message="app process exited",
                    )
                ),
            ),
            patch(
                "deep_research.services.deployment.shell_app_apps_api.fetch_app_log_tail",
                new=AsyncMock(return_value=tail),
            ),
            _patch_settings(),
            _patch_source_client(wc),
        ):
            result = await _deploy_via_apps_api(
                _make_artifact(),
                {
                    "mode": "shell_app",
                    "app_name": "dr-shell-test",
                    "framework_git_tag": "v1.0.0",
                },
                _make_deployment(),
                wc,
            )

        assert result.success is False
        assert result.external_resource_ids is not None
        assert result.external_resource_ids.get("error_kind") == "reachability_failed"
        assert result.external_resource_ids.get("last_state") == "ERROR"
        assert "direct references" in str(result.external_resource_ids.get("last_logs"))
