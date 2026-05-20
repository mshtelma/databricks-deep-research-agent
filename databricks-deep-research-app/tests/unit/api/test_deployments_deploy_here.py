"""Unit tests for POST /api/v1/deployments/{id}/actions/deploy-here (US-403).

Plan reference: Section P5 — backend unit tests for the inline OBO deploy endpoint.

Environment is bootstrapped by tests/unit/api/conftest.py which sets:
  STORAGE_BACKEND=fake, DATABRICKS_HOST, DATABRICKS_TOKEN, etc.

Pattern mirrors tests/unit/api/test_chats.py:
  - Dependency overrides are applied before TestClient is instantiated.
  - TestClient is yielded from a pytest fixture (NOT used as a context manager)
    so the app lifespan does NOT run.
"""

from __future__ import annotations

from collections.abc import Generator
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from deep_research.core.auth import UserIdentity
from deep_research.db.session import get_db
from deep_research.main import app
from deep_research.middleware.auth import get_current_user_identity
from deep_research.models.agent_deployment import (
    AgentDeployment,
    DeploymentMode,
    DeploymentStatus,
)
from deep_research.schemas.deployment import CreateDeploymentRequest
from deep_research.services.deployment.capability_probe import CapabilityProbeCache
from deep_research.services.deployment.translator import (
    Artifact,
    DeploymentResult,
    ValidationResult,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OWNER_USER_ID = "user-owner-001"
_OTHER_USER_ID = "user-other-999"
_DEPLOYMENT_ID = uuid4()
_AGENT_ID = uuid4()
_REVISION_ID = uuid4()

_APP_NAME = "dr-shell-x"
_APP_URL = "https://adb-123.azuredatabricks.net/apps/dr-shell-x"
_DEPLOYMENT_PATH = "/Workspace/Users/owner@acme.com/dr-shell-apps/abc"
_TEST_HOST = "https://test.azuredatabricks.net"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_cache() -> CapabilityProbeCache:
    """Return a fresh empty CapabilityProbeCache with TTL=60."""
    return CapabilityProbeCache(ttl_seconds=60.0)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def _make_deployment(
    *,
    deployment_id: UUID = _DEPLOYMENT_ID,
    mode: str = DeploymentMode.SHELL_APP.value,
    status: str = DeploymentStatus.PENDING.value,
    deployed_by: str = _OWNER_USER_ID,
    external_resource_ids: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> MagicMock:
    d = MagicMock(spec=AgentDeployment)
    d.id = deployment_id
    d.agent_id = _AGENT_ID
    d.revision_id = _REVISION_ID
    d.mode = mode
    d.status = status
    d.deployed_by = deployed_by
    d.config = {
        "mode": mode,
        "app_name": "dr-shell-x",
        "framework_git_tag": "v1.0.0",
    }
    d.external_resource_ids = external_resource_ids
    d.endpoint_name = None
    d.model_name = None
    d.error_message = error_message
    d.cleanup_attempts = 0
    d.created_at = datetime.now(UTC)
    d.updated_at = d.created_at
    d.deactivated_at = None
    return d


def _make_revision(definition: dict[str, Any] | None = None) -> MagicMock:
    rev = MagicMock()
    rev.agent_id = _AGENT_ID
    rev.rev_id = _REVISION_ID
    rev.definition = definition or {
        "name": "Designed Agent",
        "description": "A saved workflow revision.",
        "root": {"type": "sequence", "children": []},
    }
    return rev


def _make_agent() -> MagicMock:
    ag = MagicMock()
    ag.id = _AGENT_ID
    return ag


def _make_success_result() -> DeploymentResult:
    return DeploymentResult(
        success=True,
        endpoint_name=_APP_NAME,
        external_resource_ids={
            "app_name": _APP_NAME,
            "app_url": _APP_URL,
            "deployment_path": _DEPLOYMENT_PATH,
        },
    )


def _make_failure_result() -> DeploymentResult:
    return DeploymentResult(
        success=False,
        error_message="boom",
        external_resource_ids={"error_kind": "unknown_error"},
    )


# ---------------------------------------------------------------------------
# Shared session factory
# ---------------------------------------------------------------------------


def _make_session(
    *,
    advisory_lock_ok: bool = True,
    revision_definition: dict[str, Any] | None = None,
) -> AsyncMock:
    session = AsyncMock()

    async def _session_get(model_cls: Any, pk: Any) -> Any:
        from deep_research.models.agent_v2 import AgentRevision

        if model_cls is AgentRevision:
            return _make_revision(revision_definition)
        return None

    session.get.side_effect = _session_get

    lock_row = MagicMock()
    lock_row.scalar.return_value = advisory_lock_ok
    session.execute = AsyncMock(return_value=lock_row)
    return session


# ---------------------------------------------------------------------------
# Client fixture factory (mirrors test_chats.py pattern)
# ---------------------------------------------------------------------------


def _make_test_client(
    *,
    user_id: str = _OWNER_USER_ID,
    session: AsyncMock | None = None,
) -> Generator[TestClient, None, None]:
    """Set up dependency overrides and yield a TestClient without triggering
    the app lifespan (matches the pattern in test_chats.py)."""
    if session is None:
        session = _make_session()

    async def _override_get_db() -> Any:
        yield session

    async def _override_get_current_user() -> UserIdentity:
        return UserIdentity(
            user_id=user_id,
            email=f"{user_id}@acme.com",
            display_name=user_id,
        )

    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_current_user_identity] = _override_get_current_user

    yield TestClient(app)

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Workspace client mock
# ---------------------------------------------------------------------------


def _make_workspace_client() -> MagicMock:
    wc = MagicMock()
    wc.config = MagicMock(host=_TEST_HOST)
    wc.apps.list.return_value = iter([])
    return wc


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestCreateDeploymentRunAsync:
    @pytest.mark.asyncio
    async def test_blocks_default_scaffold_revision_before_creating_row(self) -> None:
        from deep_research.api.v1 import deployments as deployments_module

        session = _make_session(
            revision_definition={
                "name": "Untitled Agent",
                "description": "",
                "root": {
                    "type": "sequence",
                    "children": [
                        {
                            "id": "coordinator",
                            "type": "agent",
                            "label": "Coordinator",
                            "config": {"subtype": "coordinator"},
                        },
                        {
                            "id": "plan-and-execute",
                            "type": "plan_and_execute",
                            "label": "Plan & Execute",
                            "config": {},
                        },
                        {
                            "id": "synthesizer",
                            "type": "agent",
                            "label": "Synthesizer",
                            "config": {"subtype": "synthesizer"},
                        },
                    ],
                },
            },
        )
        request = MagicMock()
        request.app.state.deployment_runner = MagicMock()

        agent_service = MagicMock()
        agent_service.get_for_user = AsyncMock(return_value=_make_agent())
        deployment_service = MagicMock()
        deployment_service.create = AsyncMock()
        translator = MagicMock()
        translator.validate = AsyncMock(return_value=ValidationResult(valid=True))

        body = CreateDeploymentRequest(
            agent_id=_AGENT_ID,
            revision_id=_REVISION_ID,
            config={"mode": "in_app"},
        )
        user = UserIdentity(
            user_id=_OWNER_USER_ID,
            email="owner@acme.com",
            display_name="Owner",
        )

        with (
            patch.object(deployments_module, "AgentV2Service", return_value=agent_service),
            patch.object(
                deployments_module,
                "DeploymentService",
                return_value=deployment_service,
            ),
            patch.object(deployments_module, "_translator_for", return_value=translator),
            pytest.raises(HTTPException) as exc_info,
        ):
            await deployments_module.create_deployment(
                body,
                request,
                user,
                session,
                run_async=False,
            )

        assert exc_info.value.status_code == 422
        detail = exc_info.value.detail
        assert detail["error_kind"] == "default_revision_not_deployable"
        assert detail["agent_id"] == str(_AGENT_ID)
        assert detail["revision_id"] == str(_REVISION_ID)
        assert detail["workflow_name"] == "Untitled Agent"
        assert detail["root_child_summary"] == [
            "coordinator:agent:Coordinator",
            "plan-and-execute:plan_and_execute:Plan & Execute",
            "synthesizer:agent:Synthesizer",
        ]
        assert "designed workflow revision" in detail["message"]
        translator.validate.assert_not_awaited()
        deployment_service.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_allows_default_shape_when_planner_guidance_present(self) -> None:
        from deep_research.api.v1 import deployments as deployments_module

        session = _make_session(
            revision_definition={
                "name": "Untitled Agent",
                "description": "",
                "root": {
                    "type": "sequence",
                    "children": [
                        {"id": "coordinator", "type": "agent", "label": "Coordinator"},
                        {
                            "id": "plan-and-execute",
                            "type": "plan_and_execute",
                            "label": "Plan & Execute",
                            "config": {"planner_guidance": "Plan around the user's saved goal."},
                        },
                        {"id": "synthesizer", "type": "agent", "label": "Synthesizer"},
                    ],
                },
            },
        )
        request = MagicMock()
        request.app.state.deployment_runner = MagicMock()
        deployment = _make_deployment(mode=DeploymentMode.IN_APP.value)

        agent_service = MagicMock()
        agent_service.get_for_user = AsyncMock(return_value=_make_agent())
        deployment_service = MagicMock()
        deployment_service.create = AsyncMock(return_value=deployment)
        translator = MagicMock()
        translator.validate = AsyncMock(return_value=ValidationResult(valid=True))

        body = CreateDeploymentRequest(
            agent_id=_AGENT_ID,
            revision_id=_REVISION_ID,
            config={"mode": "in_app"},
        )
        user = UserIdentity(
            user_id=_OWNER_USER_ID,
            email="owner@acme.com",
            display_name="Owner",
        )

        with (
            patch.object(deployments_module, "AgentV2Service", return_value=agent_service),
            patch.object(
                deployments_module,
                "DeploymentService",
                return_value=deployment_service,
            ),
            patch.object(deployments_module, "_translator_for", return_value=translator),
        ):
            response = await deployments_module.create_deployment(
                body,
                request,
                user,
                session,
                run_async=False,
            )

        assert response.id == deployment.id
        translator.validate.assert_awaited_once()
        deployment_service.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_async_false_creates_row_without_submitting_runner(self) -> None:
        from deep_research.api.v1 import deployments as deployments_module

        session = _make_session()
        request = MagicMock()
        request.app.state.deployment_runner = MagicMock()
        deployment = _make_deployment(mode=DeploymentMode.IN_APP.value)

        agent_service = MagicMock()
        agent_service.get_for_user = AsyncMock(return_value=_make_agent())
        deployment_service = MagicMock()
        deployment_service.create = AsyncMock(return_value=deployment)
        translator = MagicMock()
        translator.validate = AsyncMock(return_value=ValidationResult(valid=True))

        body = CreateDeploymentRequest(
            agent_id=_AGENT_ID,
            revision_id=_REVISION_ID,
            config={"mode": "in_app"},
        )
        user = UserIdentity(
            user_id=_OWNER_USER_ID,
            email="owner@acme.com",
            display_name="Owner",
        )

        with (
            patch.object(deployments_module, "AgentV2Service", return_value=agent_service),
            patch.object(
                deployments_module,
                "DeploymentService",
                return_value=deployment_service,
            ),
            patch.object(deployments_module, "_translator_for", return_value=translator),
        ):
            response = await deployments_module.create_deployment(
                body,
                request,
                user,
                session,
                run_async=False,
            )

        assert response.id == deployment.id
        request.app.state.deployment_runner.submit.assert_not_called()

    @pytest.mark.asyncio
    async def test_default_create_deployment_submits_runner(self) -> None:
        from deep_research.api.v1 import deployments as deployments_module

        session = _make_session()
        runner = MagicMock()
        request = MagicMock()
        request.app.state.deployment_runner = runner
        deployment = _make_deployment(mode=DeploymentMode.IN_APP.value)

        agent_service = MagicMock()
        agent_service.get_for_user = AsyncMock(return_value=_make_agent())
        deployment_service = MagicMock()
        deployment_service.create = AsyncMock(return_value=deployment)
        translator = MagicMock()
        translator.validate = AsyncMock(return_value=ValidationResult(valid=True))

        body = CreateDeploymentRequest(
            agent_id=_AGENT_ID,
            revision_id=_REVISION_ID,
            config={"mode": "in_app"},
        )
        user = UserIdentity(
            user_id=_OWNER_USER_ID,
            email="owner@acme.com",
            display_name="Owner",
        )

        with (
            patch.object(deployments_module, "AgentV2Service", return_value=agent_service),
            patch.object(
                deployments_module,
                "DeploymentService",
                return_value=deployment_service,
            ),
            patch.object(deployments_module, "_translator_for", return_value=translator),
        ):
            await deployments_module.create_deployment(
                body,
                request,
                user,
                session,
            )

        runner.submit.assert_called_once_with(deployment.id, _OWNER_USER_ID)


class TestDeactivateDeployment:
    @pytest.mark.asyncio
    async def test_failed_row_delete_marks_deactivated_for_explicit_cleanup(self) -> None:
        from deep_research.api.v1 import deployments as deployments_module

        session = AsyncMock()
        request = MagicMock()
        failed = _make_deployment(status=DeploymentStatus.FAILED.value)
        deactivated = _make_deployment(status=DeploymentStatus.DEACTIVATED.value)
        service = MagicMock()
        service.get = AsyncMock(return_value=failed)
        service.deactivate = AsyncMock(return_value=deactivated)
        translator_for = MagicMock()
        user = UserIdentity(
            user_id=_OWNER_USER_ID,
            email="owner@acme.com",
            display_name="Owner",
        )

        with (
            patch.object(deployments_module, "DeploymentService", return_value=service),
            patch.object(deployments_module, "_can_manage_deployment", AsyncMock(return_value=True)),
            patch.object(deployments_module, "_translator_for", translator_for),
        ):
            response = await deployments_module.deactivate_deployment(
                failed.id,
                request,
                user,
                session,
            )

        assert response.status == DeploymentStatus.DEACTIVATED
        service.deactivate.assert_awaited_once_with(failed.id)
        session.commit.assert_awaited_once()
        translator_for.assert_not_called()


class TestDeployHereHappyPath:
    def test_happy_path_returns_deploying_and_schedules_background(self) -> None:
        """deploy-here returns quickly and schedules the OBO deploy in background."""
        pending = _make_deployment()
        deploying = _make_deployment(status=DeploymentStatus.DEPLOYING.value)

        mock_service = MagicMock()
        mock_service.get = AsyncMock(return_value=pending)
        mock_service.update_status = AsyncMock(return_value=deploying)

        mock_translator = MagicMock()
        mock_translator.deploy_inline = AsyncMock()
        mock_translator.translate = AsyncMock(
            return_value=Artifact(mode=DeploymentMode.SHELL_APP, payload=b"zip")
        )

        mock_agent_service = MagicMock()
        mock_agent_service.get_for_user = AsyncMock(return_value=_make_agent())
        mock_agent_service.get_owned = AsyncMock(return_value=_make_agent())

        wc = _make_workspace_client()
        session = _make_session()

        for client in _make_test_client(session=session):
            with (
                patch(
                    "deep_research.api.v1.deployments.DeploymentService", return_value=mock_service
                ),
                patch(
                    "deep_research.api.v1.deployments.AgentV2Service",
                    return_value=mock_agent_service,
                ),
                patch(
                    "deep_research.api.v1.deployments._translator_for", return_value=mock_translator
                ),
                patch(
                    "deep_research.api.v1.deployments.get_user_workspace_client", return_value=wc
                ),
                patch(
                    "deep_research.api.v1.deployments.get_default_cache",
                    return_value=_fresh_cache(),
                ),
                patch(
                    "deep_research.api.v1.deployments._schedule_deploy_here_background"
                ) as schedule,
            ):
                resp = client.post(f"/api/v1/deployments/{_DEPLOYMENT_ID}/actions/deploy-here")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "deploying"
        session.refresh.assert_awaited_once_with(deploying)
        mock_translator.deploy_inline.assert_not_called()
        schedule.assert_called_once()
        assert schedule.call_args.kwargs["deployment_id"] == deploying.id
        assert schedule.call_args.kwargs["workspace_client"] is wc
        assert str(schedule.call_args.kwargs["worker_id"]).startswith("deploy-here:")
        wc.apps.list.assert_not_called()

    def test_already_deploying_returns_current_row_without_rescheduling(self) -> None:
        deploying = _make_deployment(status=DeploymentStatus.DEPLOYING.value)

        mock_service = MagicMock()
        mock_service.get = AsyncMock(return_value=deploying)

        mock_agent_service = MagicMock()
        mock_agent_service.get_owned = AsyncMock(return_value=_make_agent())

        for client in _make_test_client():
            with (
                patch(
                    "deep_research.api.v1.deployments.DeploymentService", return_value=mock_service
                ),
                patch(
                    "deep_research.api.v1.deployments.AgentV2Service",
                    return_value=mock_agent_service,
                ),
                patch(
                    "deep_research.api.v1.deployments._schedule_deploy_here_background"
                ) as schedule,
            ):
                resp = client.post(f"/api/v1/deployments/{_DEPLOYMENT_ID}/actions/deploy-here")

        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "deploying"
        schedule.assert_not_called()


# ---------------------------------------------------------------------------
# 404 cases
# ---------------------------------------------------------------------------


class TestDeployHere404Cases:
    def test_404_when_deployment_missing(self) -> None:
        """Bogus deployment_id → 404."""
        bogus_id = uuid4()
        mock_service = MagicMock()
        mock_service.get = AsyncMock(return_value=None)

        for client in _make_test_client():
            with patch(
                "deep_research.api.v1.deployments.DeploymentService", return_value=mock_service
            ):
                resp = client.post(f"/api/v1/deployments/{bogus_id}/actions/deploy-here")

        assert resp.status_code == 404

    def test_404_when_caller_is_not_authorized(self) -> None:
        """Different user who is neither deployer nor agent-owner → 404 (W9 cloak)."""
        deployment = _make_deployment(deployed_by=_OWNER_USER_ID)

        mock_service = MagicMock()
        mock_service.get = AsyncMock(return_value=deployment)

        mock_agent_service = MagicMock()
        mock_agent_service.get_owned = AsyncMock(return_value=None)

        for client in _make_test_client(user_id=_OTHER_USER_ID):
            with (
                patch(
                    "deep_research.api.v1.deployments.DeploymentService", return_value=mock_service
                ),
                patch(
                    "deep_research.api.v1.deployments.AgentV2Service",
                    return_value=mock_agent_service,
                ),
            ):
                resp = client.post(f"/api/v1/deployments/{_DEPLOYMENT_ID}/actions/deploy-here")

        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 409 cases
# ---------------------------------------------------------------------------


class TestDeployHere409Cases:
    def test_409_concurrent_deploy_when_advisory_lock_busy(self) -> None:
        """pg_try_advisory_xact_lock returns False → 409 deploy_already_in_progress."""
        deployment = _make_deployment()

        mock_service = MagicMock()
        mock_service.get = AsyncMock(return_value=deployment)

        mock_agent_service = MagicMock()
        mock_agent_service.get_owned = AsyncMock(return_value=_make_agent())

        mock_translator = MagicMock()
        mock_translator.deploy_inline = AsyncMock()

        wc = _make_workspace_client()

        # Session whose advisory lock returns False
        session = _make_session(advisory_lock_ok=False)

        for client in _make_test_client(session=session):
            with (
                patch(
                    "deep_research.api.v1.deployments.DeploymentService", return_value=mock_service
                ),
                patch(
                    "deep_research.api.v1.deployments.AgentV2Service",
                    return_value=mock_agent_service,
                ),
                patch(
                    "deep_research.api.v1.deployments._translator_for", return_value=mock_translator
                ),
                patch(
                    "deep_research.api.v1.deployments.get_user_workspace_client", return_value=wc
                ),
                patch(
                    "deep_research.api.v1.deployments.get_default_cache",
                    return_value=_fresh_cache(),
                ),
            ):
                resp = client.post(f"/api/v1/deployments/{_DEPLOYMENT_ID}/actions/deploy-here")

        assert resp.status_code == 409
        assert resp.json()["message"]["error_kind"] == "deploy_already_in_progress"

    def test_409_redeploy_requires_confirmation_when_active(self) -> None:
        """ACTIVE deployment + no confirm_redeploy → 409. With ?confirm_redeploy=1 → 200."""
        active = _make_deployment(status=DeploymentStatus.ACTIVE.value)

        # --- first call: no confirm → 409 ---
        mock_service_block = MagicMock()
        mock_service_block.get = AsyncMock(return_value=active)

        mock_agent_service = MagicMock()
        mock_agent_service.get_owned = AsyncMock(return_value=_make_agent())

        for client in _make_test_client():
            with (
                patch(
                    "deep_research.api.v1.deployments.DeploymentService",
                    return_value=mock_service_block,
                ),
                patch(
                    "deep_research.api.v1.deployments.AgentV2Service",
                    return_value=mock_agent_service,
                ),
            ):
                resp_no_confirm = client.post(
                    f"/api/v1/deployments/{_DEPLOYMENT_ID}/actions/deploy-here"
                )

        assert resp_no_confirm.status_code == 409
        assert resp_no_confirm.json()["message"]["error_kind"] == "redeploy_requires_confirmation"

        # --- second call: with confirm → 200 + background scheduling ---
        deploying = _make_deployment(status=DeploymentStatus.DEPLOYING.value)

        mock_service_ok = MagicMock()
        mock_service_ok.get = AsyncMock(return_value=active)
        mock_service_ok.update_status = AsyncMock(return_value=deploying)

        mock_translator = MagicMock()
        mock_translator.deploy_inline = AsyncMock()
        mock_translator.translate = AsyncMock(
            return_value=Artifact(mode=DeploymentMode.SHELL_APP, payload=b"zip")
        )

        mock_agent_service2 = MagicMock()
        mock_agent_service2.get_for_user = AsyncMock(return_value=_make_agent())
        mock_agent_service2.get_owned = AsyncMock(return_value=_make_agent())

        wc = _make_workspace_client()

        for client in _make_test_client():
            with (
                patch(
                    "deep_research.api.v1.deployments.DeploymentService",
                    return_value=mock_service_ok,
                ),
                patch(
                    "deep_research.api.v1.deployments.AgentV2Service",
                    return_value=mock_agent_service2,
                ),
                patch(
                    "deep_research.api.v1.deployments._translator_for", return_value=mock_translator
                ),
                patch(
                    "deep_research.api.v1.deployments.get_user_workspace_client", return_value=wc
                ),
                patch(
                    "deep_research.api.v1.deployments.get_default_cache",
                    return_value=_fresh_cache(),
                ),
                patch(
                    "deep_research.api.v1.deployments._schedule_deploy_here_background"
                ) as schedule,
            ):
                resp_confirmed = client.post(
                    f"/api/v1/deployments/{_DEPLOYMENT_ID}/actions/deploy-here?confirm_redeploy=1"
                )

        assert resp_confirmed.status_code == 200
        assert resp_confirmed.json()["status"] == "deploying"
        schedule.assert_called_once()


# ---------------------------------------------------------------------------
# 400 cases
# ---------------------------------------------------------------------------


class TestDeployHere400Cases:
    def test_400_mode_does_not_support_inline_deploy_for_in_app(self) -> None:
        """in_app deployment → 400 mode_does_not_support_inline_deploy."""
        in_app = _make_deployment(mode=DeploymentMode.IN_APP.value)

        mock_service = MagicMock()
        mock_service.get = AsyncMock(return_value=in_app)

        mock_agent_service = MagicMock()
        mock_agent_service.get_owned = AsyncMock(return_value=_make_agent())

        # A translator object with NO deploy_inline attribute
        no_inline_translator = MagicMock(spec=[])

        for client in _make_test_client():
            with (
                patch(
                    "deep_research.api.v1.deployments.DeploymentService", return_value=mock_service
                ),
                patch(
                    "deep_research.api.v1.deployments.AgentV2Service",
                    return_value=mock_agent_service,
                ),
                patch(
                    "deep_research.api.v1.deployments._translator_for",
                    return_value=no_inline_translator,
                ),
            ):
                resp = client.post(f"/api/v1/deployments/{_DEPLOYMENT_ID}/actions/deploy-here")

        assert resp.status_code == 400
        assert resp.json()["message"]["error_kind"] == "mode_does_not_support_inline_deploy"


# ---------------------------------------------------------------------------
# 403 cases
# ---------------------------------------------------------------------------


class TestDeployHereAdvisoryProbe:
    def test_apps_list_failure_does_not_block_real_deploy(self) -> None:
        """The old apps.list probe is advisory only; background deploy is authoritative."""
        pending = _make_deployment()
        deploying = _make_deployment(status=DeploymentStatus.DEPLOYING.value)

        mock_service = MagicMock()
        mock_service.get = AsyncMock(return_value=pending)
        mock_service.update_status = AsyncMock(return_value=deploying)

        mock_agent_service = MagicMock()
        mock_agent_service.get_for_user = AsyncMock(return_value=_make_agent())
        mock_agent_service.get_owned = AsyncMock(return_value=_make_agent())

        PermissionDenied = type("PermissionDenied", (Exception,), {})
        mock_translator = MagicMock()
        mock_translator.deploy_inline = AsyncMock()
        mock_translator.translate = AsyncMock(
            return_value=Artifact(mode=DeploymentMode.SHELL_APP, payload=b"zip")
        )

        wc = _make_workspace_client()
        wc.apps.list.side_effect = PermissionDenied("403 permission denied")

        for client in _make_test_client():
            with (
                patch(
                    "deep_research.api.v1.deployments.DeploymentService", return_value=mock_service
                ),
                patch(
                    "deep_research.api.v1.deployments.AgentV2Service",
                    return_value=mock_agent_service,
                ),
                patch(
                    "deep_research.api.v1.deployments._translator_for", return_value=mock_translator
                ),
                patch(
                    "deep_research.api.v1.deployments.get_user_workspace_client", return_value=wc
                ),
                patch(
                    "deep_research.api.v1.deployments.get_default_cache",
                    return_value=_fresh_cache(),
                ),
                patch(
                    "deep_research.api.v1.deployments._schedule_deploy_here_background"
                ) as schedule,
            ):
                resp = client.post(f"/api/v1/deployments/{_DEPLOYMENT_ID}/actions/deploy-here")

        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "deploying"
        schedule.assert_called_once()
        wc.apps.list.assert_not_called()


# ---------------------------------------------------------------------------
# Failure path
# ---------------------------------------------------------------------------


class TestDeployHereFailurePath:
    def test_failure_path_is_deferred_to_background_task(self) -> None:
        """deploy-here request returns deploying; background task records failures."""
        pending = _make_deployment()
        deploying = _make_deployment(status=DeploymentStatus.DEPLOYING.value)

        mock_service = MagicMock()
        mock_service.get = AsyncMock(return_value=pending)
        mock_service.update_status = AsyncMock(return_value=deploying)

        mock_agent_service = MagicMock()
        mock_agent_service.get_for_user = AsyncMock(return_value=_make_agent())
        mock_agent_service.get_owned = AsyncMock(return_value=_make_agent())

        mock_translator = MagicMock()
        mock_translator.deploy_inline = AsyncMock()
        mock_translator.translate = AsyncMock(
            return_value=Artifact(mode=DeploymentMode.SHELL_APP, payload=b"zip")
        )

        wc = _make_workspace_client()

        for client in _make_test_client():
            with (
                patch(
                    "deep_research.api.v1.deployments.DeploymentService", return_value=mock_service
                ),
                patch(
                    "deep_research.api.v1.deployments.AgentV2Service",
                    return_value=mock_agent_service,
                ),
                patch(
                    "deep_research.api.v1.deployments._translator_for", return_value=mock_translator
                ),
                patch(
                    "deep_research.api.v1.deployments.get_user_workspace_client", return_value=wc
                ),
                patch(
                    "deep_research.api.v1.deployments.get_default_cache",
                    return_value=_fresh_cache(),
                ),
                patch(
                    "deep_research.api.v1.deployments._schedule_deploy_here_background"
                ) as schedule,
            ):
                resp = client.post(f"/api/v1/deployments/{_DEPLOYMENT_ID}/actions/deploy-here")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "deploying"
        assert body.get("error_message") is None
        mock_translator.deploy_inline.assert_not_called()
        schedule.assert_called_once()


class _AsyncSessionContext:
    def __init__(self, session: AsyncMock) -> None:
        self._session = session

    async def __aenter__(self) -> AsyncMock:
        return self._session

    async def __aexit__(self, *_args: object) -> None:
        return None


class TestDeployHereBackgroundTask:
    @pytest.mark.asyncio
    async def test_background_success_marks_row_active(self) -> None:
        from deep_research.api.v1 import deployments as deployments_module

        session = AsyncMock()
        worker_id = "deploy-here:test-success"
        deploying = _make_deployment(status=DeploymentStatus.DEPLOYING.value)
        deploying.worker_id = worker_id
        result = _make_success_result()
        active = _make_deployment(
            status=DeploymentStatus.ACTIVE.value,
            external_resource_ids=result.external_resource_ids,
        )
        service = MagicMock()
        service.get = AsyncMock(return_value=deploying)
        service.update_status = AsyncMock(return_value=active)

        translator = MagicMock()
        translator.deploy_inline = AsyncMock(return_value=result)
        wc = _make_workspace_client()

        with (
            patch(
                "deep_research.db.session.get_session_maker",
                return_value=lambda: _AsyncSessionContext(session),
            ),
            patch.object(deployments_module, "DeploymentService", return_value=service),
            patch.object(deployments_module, "_translator_for", return_value=translator),
        ):
            await deployments_module._run_deploy_here_background(
                deployment_id=deploying.id,
                artifact=Artifact(mode=DeploymentMode.SHELL_APP, payload=b"zip"),
                workspace_client=wc,
                worker_id=worker_id,
            )

        translator.deploy_inline.assert_awaited_once()
        service.update_status.assert_awaited_once_with(
            deploying.id,
            DeploymentStatus.ACTIVE,
            endpoint_name=result.endpoint_name,
            model_name=result.model_name,
            external_resource_ids=result.external_resource_ids,
        )
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_background_failure_marks_row_failed(self) -> None:
        from deep_research.api.v1 import deployments as deployments_module

        session = AsyncMock()
        worker_id = "deploy-here:test-failure"
        deploying = _make_deployment(status=DeploymentStatus.DEPLOYING.value)
        deploying.worker_id = worker_id
        failed = _make_deployment(status=DeploymentStatus.FAILED.value)
        result = _make_failure_result()
        service = MagicMock()
        service.get = AsyncMock(return_value=deploying)
        service.update_status = AsyncMock(return_value=failed)

        translator = MagicMock()
        translator.deploy_inline = AsyncMock(return_value=result)
        wc = _make_workspace_client()

        with (
            patch(
                "deep_research.db.session.get_session_maker",
                return_value=lambda: _AsyncSessionContext(session),
            ),
            patch.object(deployments_module, "DeploymentService", return_value=service),
            patch.object(deployments_module, "_translator_for", return_value=translator),
        ):
            await deployments_module._run_deploy_here_background(
                deployment_id=deploying.id,
                artifact=Artifact(mode=DeploymentMode.SHELL_APP, payload=b"zip"),
                workspace_client=wc,
                worker_id=worker_id,
            )

        service.update_status.assert_awaited_once_with(
            deploying.id,
            DeploymentStatus.FAILED,
            error_message="boom",
            external_resource_ids={"error_kind": "unknown_error"},
        )
        session.commit.assert_awaited_once()


# ---------------------------------------------------------------------------
# GET /api/v1/deployments/can-deploy-here (Section S2)
# ---------------------------------------------------------------------------


class TestCanDeployHere:
    def test_can_deploy_true_when_probe_succeeds(self) -> None:
        """Apps probe succeeds → can_deploy=True, result cached."""
        wc = _make_workspace_client()
        cache = _fresh_cache()

        for client in _make_test_client():
            with (
                patch(
                    "deep_research.api.v1.deployments.get_user_workspace_client", return_value=wc
                ),
                patch("deep_research.api.v1.deployments.get_default_cache", return_value=cache),
            ):
                resp = client.get("/api/v1/deployments/can-deploy-here")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["can_deploy"] is True
        assert body["reason"] is None
        assert body["probe_status"] == "ok"
        # Regression guard: SDK 0.96 renamed `limit` → `page_size`. Asserting
        # the kwargs keeps the probe wired to the real signature so a future
        # rename is caught locally instead of in production.
        wc.apps.list.assert_called_once_with(page_size=1)
        assert wc.files.method_calls == []

    def test_can_deploy_false_when_permission_denied(self) -> None:
        """Apps probe raises permission error → can_deploy=False."""
        PermissionDenied = type("PermissionDenied", (Exception,), {})
        wc = MagicMock()
        wc.config = MagicMock(host=_TEST_HOST)
        wc.apps.list.side_effect = PermissionDenied("403 Forbidden")
        cache = _fresh_cache()

        for client in _make_test_client():
            with (
                patch(
                    "deep_research.api.v1.deployments.get_user_workspace_client", return_value=wc
                ),
                patch("deep_research.api.v1.deployments.get_default_cache", return_value=cache),
            ):
                resp = client.get("/api/v1/deployments/can-deploy-here")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["can_deploy"] is False
        assert body["reason"] == "missing_workspace_permission"
        assert body["probe_status"] == "denied"

    def test_cached_result_returned_without_probe(self) -> None:
        """Cached positive result is returned without calling apps.list again."""
        wc = _make_workspace_client()
        cache = _fresh_cache()
        # Pre-populate cache with a positive result
        cache.set(_OWNER_USER_ID, _TEST_HOST, ok=True)

        for client in _make_test_client():
            with (
                patch(
                    "deep_research.api.v1.deployments.get_user_workspace_client", return_value=wc
                ),
                patch("deep_research.api.v1.deployments.get_default_cache", return_value=cache),
            ):
                resp = client.get("/api/v1/deployments/can-deploy-here")

        assert resp.status_code == 200
        body = resp.json()
        assert body["can_deploy"] is True
        assert body["probe_status"] == "ok"
        # apps.list must NOT have been called (result came from cache)
        wc.apps.list.assert_not_called()

    def test_cached_failure_returned_without_probe(self) -> None:
        """Cached negative result is returned without re-probing."""
        wc = _make_workspace_client()
        cache = _fresh_cache()
        cache.set(_OWNER_USER_ID, _TEST_HOST, ok=False, reason="missing_workspace_permission")

        for client in _make_test_client():
            with (
                patch(
                    "deep_research.api.v1.deployments.get_user_workspace_client", return_value=wc
                ),
                patch("deep_research.api.v1.deployments.get_default_cache", return_value=cache),
            ):
                resp = client.get("/api/v1/deployments/can-deploy-here")

        assert resp.status_code == 200
        body = resp.json()
        assert body["can_deploy"] is False
        assert body["reason"] == "missing_workspace_permission"
        assert body["probe_status"] == "denied"
        wc.apps.list.assert_not_called()

    def test_actor_obo_when_forwarded_token_present(self) -> None:
        """X-Forwarded-Access-Token header → actor=obo."""
        wc = _make_workspace_client()
        cache = _fresh_cache()

        for client in _make_test_client():
            with (
                patch(
                    "deep_research.api.v1.deployments.get_user_workspace_client", return_value=wc
                ),
                patch("deep_research.api.v1.deployments.get_default_cache", return_value=cache),
            ):
                resp = client.get(
                    "/api/v1/deployments/can-deploy-here",
                    headers={"X-Forwarded-Access-Token": "dapi-token-abc"},
                )

        assert resp.status_code == 200
        assert resp.json()["actor"] == "obo"

    def test_actor_sp_fallback_when_no_forwarded_token(self) -> None:
        """No X-Forwarded-Access-Token → actor=sp_fallback."""
        wc = _make_workspace_client()
        cache = _fresh_cache()

        for client in _make_test_client():
            with (
                patch(
                    "deep_research.api.v1.deployments.get_user_workspace_client", return_value=wc
                ),
                patch("deep_research.api.v1.deployments.get_default_cache", return_value=cache),
            ):
                resp = client.get("/api/v1/deployments/can-deploy-here")

        assert resp.status_code == 200
        assert resp.json()["actor"] == "sp_fallback"

    def test_transient_error_returns_unknown_but_not_cached(self) -> None:
        """Generic probe error → probe_status=unknown, can_deploy=True, NOT cached."""
        wc = MagicMock()
        wc.config = MagicMock(host=_TEST_HOST)
        wc.apps.list.side_effect = ConnectionError("network unreachable")
        cache = _fresh_cache()

        for client in _make_test_client():
            with (
                patch(
                    "deep_research.api.v1.deployments.get_user_workspace_client", return_value=wc
                ),
                patch("deep_research.api.v1.deployments.get_default_cache", return_value=cache),
            ):
                resp = client.get("/api/v1/deployments/can-deploy-here")

        assert resp.status_code == 200
        body = resp.json()
        assert body["can_deploy"] is True
        # probe_error is internal — surfaced only as advisory probe_status.
        assert body["reason"] is None
        assert body["probe_status"] == "unknown"
        # probe_error is NOT cached — next request will retry
        assert cache.get(_OWNER_USER_ID, _TEST_HOST) is None


# ---------------------------------------------------------------------------
# POST /api/v1/deployments/can-deploy-here/refresh (Section S2)
# ---------------------------------------------------------------------------


class TestRefreshCanDeployHere:
    def test_refresh_invalidates_cache_and_re_probes(self) -> None:
        """Refresh endpoint clears old cache entry and re-runs the probe."""
        wc = _make_workspace_client()
        cache = _fresh_cache()
        # Pre-populate with a stale failure
        cache.set(_OWNER_USER_ID, _TEST_HOST, ok=False, reason="missing_workspace_permission")

        for client in _make_test_client():
            with (
                patch(
                    "deep_research.api.v1.deployments.get_user_workspace_client", return_value=wc
                ),
                patch("deep_research.api.v1.deployments.get_default_cache", return_value=cache),
            ):
                resp = client.post("/api/v1/deployments/can-deploy-here/refresh")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        # After refresh the probe succeeded (apps.list returns normally)
        assert body["can_deploy"] is True
        assert body["probe_status"] == "ok"
        # Cache now holds the fresh positive result
        result = cache.get(_OWNER_USER_ID, _TEST_HOST)
        assert result is not None
        assert result.ok is True

    def test_refresh_re_probes_even_when_cache_empty(self) -> None:
        """Refresh with no prior cache entry still runs the probe."""
        wc = _make_workspace_client()
        cache = _fresh_cache()

        for client in _make_test_client():
            with (
                patch(
                    "deep_research.api.v1.deployments.get_user_workspace_client", return_value=wc
                ),
                patch("deep_research.api.v1.deployments.get_default_cache", return_value=cache),
            ):
                resp = client.post("/api/v1/deployments/can-deploy-here/refresh")

        assert resp.status_code == 200
        body = resp.json()
        assert body["can_deploy"] is True
        assert body["probe_status"] == "ok"

    def test_refresh_returns_failure_when_permission_still_denied(self) -> None:
        """Refresh with permission still denied → can_deploy=False."""
        PermissionDenied = type("PermissionDenied", (Exception,), {})
        wc = MagicMock()
        wc.config = MagicMock(host=_TEST_HOST)
        wc.apps.list.side_effect = PermissionDenied("403 Forbidden")
        cache = _fresh_cache()
        cache.set(_OWNER_USER_ID, _TEST_HOST, ok=False, reason="missing_workspace_permission")

        for client in _make_test_client():
            with (
                patch(
                    "deep_research.api.v1.deployments.get_user_workspace_client", return_value=wc
                ),
                patch("deep_research.api.v1.deployments.get_default_cache", return_value=cache),
            ):
                resp = client.post("/api/v1/deployments/can-deploy-here/refresh")

        assert resp.status_code == 200
        body = resp.json()
        assert body["can_deploy"] is False
        assert body["reason"] == "missing_workspace_permission"
        assert body["probe_status"] == "denied"
