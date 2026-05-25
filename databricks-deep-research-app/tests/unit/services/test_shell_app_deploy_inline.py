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
from deep_research.services.deployment.auth import WorkspaceClientResolver
from deep_research.services.deployment.translator import (
    Artifact,
    DeploymentCleanupError,
    DeploymentCleanupExhaustedError,
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


class TestDeactivateWorkspaceDeleteSpFallback:
    """Coverage for the split-identity workspace.delete cascade.

    The translator now resolves both a user_client (OBO when client_resolver
    is supplied, SP otherwise) AND an unconditional sp_client. The
    workspace.delete step tries user_client first; on PermissionDenied it
    falls back to sp_client; if both deny, raises
    DeploymentCleanupExhaustedError so the service can mark cleanup_failed
    once instead of burning 3 retries.
    """

    @pytest.mark.asyncio
    async def test_workspace_delete_falls_back_to_sp_on_obo_denied(self) -> None:
        """OBO user denied → SP succeeds → no exception raised, normal path."""
        PermissionDenied = type("PermissionDenied", (Exception,), {})

        exporter = ShellAppExporter()
        deployment = _make_deployment(
            external_resource_ids={
                "app_name": "dr-shell-app-x",
                "deployment_path": "/Workspace/Shared/deep-research-agent/shell-apps/x",
            }
        )

        obo_client = MagicMock()
        obo_client.apps.delete = MagicMock(return_value=None)
        obo_client.workspace.delete = MagicMock(
            side_effect=PermissionDenied("permission_denied")
        )

        sp_client = MagicMock()
        sp_client.workspace.delete = MagicMock(return_value=None)

        resolver = MagicMock(spec=WorkspaceClientResolver)
        resolver.resolve = MagicMock(return_value=obo_client)

        with patch(
            "deep_research.core.databricks_auth.get_databricks_auth",
            return_value=MagicMock(get_client=MagicMock(return_value=sp_client)),
        ):
            # Must NOT raise
            await exporter.deactivate(deployment, client_resolver=resolver)

        # Both identities were tried, in the expected order
        obo_client.workspace.delete.assert_called_once_with(
            path="/Workspace/Shared/deep-research-agent/shell-apps/x",
            recursive=True,
        )
        sp_client.workspace.delete.assert_called_once_with(
            path="/Workspace/Shared/deep-research-agent/shell-apps/x",
            recursive=True,
        )
        # apps.delete still ran under the OBO identity
        obo_client.apps.delete.assert_called_once_with("dr-shell-app-x")

    @pytest.mark.asyncio
    async def test_workspace_delete_raises_exhausted_on_dual_denial(self) -> None:
        """Both OBO and SP denied → DeploymentCleanupExhaustedError raised
        (NOT DeploymentCleanupError directly — the subclass signals
        determinism and tells the service layer to skip retry counters)."""
        PermissionDenied = type("PermissionDenied", (Exception,), {})

        exporter = ShellAppExporter()
        deployment = _make_deployment(
            external_resource_ids={
                "app_name": "dr-shell-dual-deny",
                "deployment_path": "/Workspace/Shared/deep-research-agent/shell-apps/y",
            }
        )

        obo_client = MagicMock()
        obo_client.apps.delete = MagicMock(return_value=None)
        obo_client.workspace.delete = MagicMock(
            side_effect=PermissionDenied("permission_denied")
        )

        sp_client = MagicMock()
        sp_client.workspace.delete = MagicMock(
            side_effect=PermissionDenied("permission_denied")
        )

        resolver = MagicMock(spec=WorkspaceClientResolver)
        resolver.resolve = MagicMock(return_value=obo_client)

        with (
            patch(
                "deep_research.core.databricks_auth.get_databricks_auth",
                return_value=MagicMock(
                    get_client=MagicMock(return_value=sp_client)
                ),
            ),
            pytest.raises(DeploymentCleanupExhaustedError) as exc_info,
        ):
            await exporter.deactivate(deployment, client_resolver=resolver)

        # The exhausted variant is a subclass of DeploymentCleanupError so
        # downstream catch-blocks for the base class still work as a safety
        # net, but the specific subclass is what the service layer matches
        # for the one-shot cleanup_failed path.
        assert isinstance(exc_info.value, DeploymentCleanupError)
        assert "workspace.delete" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_workspace_delete_404_does_not_trigger_sp_fallback(self) -> None:
        """A NotFound on the user_client must short-circuit to the
        already-gone branch; the SP fallback path must NOT run (avoid an
        unnecessary SDK round-trip when the resource is already gone)."""
        NotFound = type("NotFound", (Exception,), {})

        exporter = ShellAppExporter()
        deployment = _make_deployment(
            external_resource_ids={
                "app_name": "dr-shell-gone",
                "deployment_path": "/Workspace/Shared/deep-research-agent/shell-apps/z",
            }
        )

        obo_client = MagicMock()
        obo_client.apps.delete = MagicMock(return_value=None)
        obo_client.workspace.delete = MagicMock(
            side_effect=NotFound("path does not exist")
        )

        sp_client = MagicMock()
        sp_client.workspace.delete = MagicMock(return_value=None)

        resolver = MagicMock(spec=WorkspaceClientResolver)
        resolver.resolve = MagicMock(return_value=obo_client)

        with patch(
            "deep_research.core.databricks_auth.get_databricks_auth",
            return_value=MagicMock(get_client=MagicMock(return_value=sp_client)),
        ):
            # Must NOT raise — 404 is idempotent success
            await exporter.deactivate(deployment, client_resolver=resolver)

        # Only the OBO path was tried; SP was NOT consulted because there's
        # nothing to clean up.
        obo_client.workspace.delete.assert_called_once()
        sp_client.workspace.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_workspace_delete_non_auth_exception_uses_3strike(self) -> None:
        """A non-PermissionDenied / non-NotFound exception (e.g., 500, network)
        must fall through to the existing DeploymentCleanupError path so the
        3-strike retry pattern in agent_v2_service.delete kicks in. We must
        NOT misroute a transient error to the exhausted-immediately branch."""
        SomeUpstreamFlake = type("SomeUpstreamFlake", (Exception,), {})

        exporter = ShellAppExporter()
        deployment = _make_deployment(
            external_resource_ids={
                "app_name": "dr-shell-flake",
                "deployment_path": "/Workspace/Shared/deep-research-agent/shell-apps/f",
            }
        )

        obo_client = MagicMock()
        obo_client.apps.delete = MagicMock(return_value=None)
        obo_client.workspace.delete = MagicMock(
            side_effect=SomeUpstreamFlake("upstream returned 500")
        )

        sp_client = MagicMock()
        sp_client.workspace.delete = MagicMock(return_value=None)

        resolver = MagicMock(spec=WorkspaceClientResolver)
        resolver.resolve = MagicMock(return_value=obo_client)

        with (
            patch(
                "deep_research.core.databricks_auth.get_databricks_auth",
                return_value=MagicMock(
                    get_client=MagicMock(return_value=sp_client)
                ),
            ),
            pytest.raises(DeploymentCleanupError) as exc_info,
        ):
            await exporter.deactivate(deployment, client_resolver=resolver)

        # Critical: this MUST be the base class, NOT the exhausted subclass.
        # Transient errors get the 3-strike retry pattern, not the
        # immediate-cleanup_failed shortcut.
        assert not isinstance(exc_info.value, DeploymentCleanupExhaustedError)
        # And SP fallback was NOT triggered (only PermissionDenied triggers it).
        sp_client.workspace.delete.assert_not_called()


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
