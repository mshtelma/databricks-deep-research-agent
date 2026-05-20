"""Unit tests for ShellAppExporter.deploy_inline() and deactivate() (US-403).

Plan reference: Section P5 — backend unit tests for the Shell App "Deploy here"
feature.

Tests:
- deploy_inline delegates to _deploy_via_apps_api and returns its result.
- deactivate calls apps.delete + workspace.delete.
- deactivate treats NotFound from apps.delete as success (idempotent).
- deactivate raises DeploymentCleanupError on PermissionDenied.
- deactivate with no external_resource_ids is a no-op.
- Protocol conformance: ShellAppExporter satisfies both DeploymentTranslator and
  InlineDeploymentTranslator.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from deep_research.models.agent_deployment import AgentDeployment, DeploymentMode
from deep_research.services.deployment.shell_app import ShellAppExporter
from deep_research.services.deployment.translator import (
    Artifact,
    DeploymentCleanupError,
    DeploymentResult,
    DeploymentTranslator,
    InlineDeploymentTranslator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_artifact() -> Artifact:
    return Artifact(mode=DeploymentMode.SHELL_APP, payload=b"fake-zip-bytes")


def _make_deployment(
    external_resource_ids: dict[str, object] | None = None,
) -> AgentDeployment:
    d = MagicMock(spec=AgentDeployment)
    d.id = uuid4()
    d.external_resource_ids = external_resource_ids
    return d


# ---------------------------------------------------------------------------
# deploy_inline
# ---------------------------------------------------------------------------


class TestDeployInline:
    @pytest.mark.asyncio
    async def test_deploy_inline_delegates_to_helper(self) -> None:
        """deploy_inline must call _deploy_via_apps_api with the right args and
        return its result unchanged."""
        exporter = ShellAppExporter()
        artifact = _make_artifact()
        config: dict[str, object] = {"app_name": "dr-shell-x", "framework_git_tag": "v1.0.0"}
        deployment = _make_deployment()
        workspace_client = MagicMock()

        expected = DeploymentResult(
            success=True,
            endpoint_name="dr-shell-x",
            external_resource_ids={"app_name": "dr-shell-x"},
        )

        with patch(
            "deep_research.services.deployment.shell_app_apps_api._deploy_via_apps_api",
            new=AsyncMock(return_value=expected),
        ) as mock_deploy:
            result = await exporter.deploy_inline(
                artifact, config, deployment, workspace_client
            )

        mock_deploy.assert_awaited_once_with(
            artifact, config, deployment, workspace_client
        )
        assert result is expected
        assert result.success is True
        assert result.endpoint_name == "dr-shell-x"


# ---------------------------------------------------------------------------
# deactivate
# ---------------------------------------------------------------------------


class TestDeactivate:
    @pytest.mark.asyncio
    async def test_deactivate_apps_delete_called(self) -> None:
        """Both apps.delete and workspace.delete must be called when
        external_resource_ids contains both app_name and deployment_path."""
        exporter = ShellAppExporter()
        deployment = _make_deployment(
            external_resource_ids={
                "app_name": "dr-shell-app-abc",
                "deployment_path": "/Workspace/Users/u/dr-shell-apps/abc",
            }
        )

        mock_client = MagicMock()
        mock_client.apps.delete = MagicMock(return_value=None)
        mock_client.workspace.delete = MagicMock(return_value=None)

        with patch(
            "deep_research.core.databricks_auth.get_databricks_auth",
            return_value=MagicMock(get_client=MagicMock(return_value=mock_client)),
        ):
            await exporter.deactivate(deployment)

        mock_client.apps.delete.assert_called_once_with("dr-shell-app-abc")
        mock_client.workspace.delete.assert_called_once_with(
            path="/Workspace/Users/u/dr-shell-apps/abc",
            recursive=True,
        )

    @pytest.mark.asyncio
    async def test_deactivate_treats_apps_404_as_success(self) -> None:
        """apps.delete raising a NotFound-style exception must NOT raise
        DeploymentCleanupError — 404 is treated as idempotent success."""
        NotFound = type("NotFound", (Exception,), {})

        exporter = ShellAppExporter()
        deployment = _make_deployment(
            external_resource_ids={
                "app_name": "dr-shell-app-gone",
                "deployment_path": "/Workspace/Users/u/dr-shell-apps/gone",
            }
        )

        mock_client = MagicMock()
        mock_client.apps.delete = MagicMock(side_effect=NotFound("resource not found"))
        mock_client.workspace.delete = MagicMock(return_value=None)

        with patch(
            "deep_research.core.databricks_auth.get_databricks_auth",
            return_value=MagicMock(get_client=MagicMock(return_value=mock_client)),
        ):
            # Must NOT raise
            await exporter.deactivate(deployment)

        # Source cleanup is still called even when apps.delete was "not found"
        mock_client.workspace.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_deactivate_raises_cleanup_error_on_permission_denied(self) -> None:
        """apps.delete raising PermissionDenied must propagate as
        DeploymentCleanupError (not swallowed)."""
        PermissionDenied = type("PermissionDenied", (Exception,), {})

        exporter = ShellAppExporter()
        deployment = _make_deployment(
            external_resource_ids={
                "app_name": "dr-shell-app-deny",
            }
        )

        mock_client = MagicMock()
        mock_client.apps.delete = MagicMock(
            side_effect=PermissionDenied("permission denied")
        )

        with (
            patch(
                "deep_research.core.databricks_auth.get_databricks_auth",
                return_value=MagicMock(get_client=MagicMock(return_value=mock_client)),
            ),
            pytest.raises(DeploymentCleanupError),
        ):
            await exporter.deactivate(deployment)

    @pytest.mark.asyncio
    async def test_deactivate_no_external_resources_is_noop(self) -> None:
        """When external_resource_ids is empty/None the function must return
        without making any SDK calls."""
        exporter = ShellAppExporter()
        deployment_none = _make_deployment(external_resource_ids=None)
        deployment_empty = _make_deployment(external_resource_ids={})

        mock_client = MagicMock()

        with patch(
            "deep_research.core.databricks_auth.get_databricks_auth",
            return_value=MagicMock(get_client=MagicMock(return_value=mock_client)),
        ):
            await exporter.deactivate(deployment_none)
            await exporter.deactivate(deployment_empty)

        mock_client.apps.delete.assert_not_called()
        mock_client.workspace.delete.assert_not_called()


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_inline_protocol_conformance(self) -> None:
        """ShellAppExporter must satisfy both DeploymentTranslator and
        InlineDeploymentTranslator at runtime."""
        exporter = ShellAppExporter()
        assert isinstance(exporter, DeploymentTranslator), (
            "ShellAppExporter must satisfy DeploymentTranslator protocol"
        )
        assert isinstance(exporter, InlineDeploymentTranslator), (
            "ShellAppExporter must satisfy InlineDeploymentTranslator protocol "
            "after US-403 (deploy_inline added)"
        )
