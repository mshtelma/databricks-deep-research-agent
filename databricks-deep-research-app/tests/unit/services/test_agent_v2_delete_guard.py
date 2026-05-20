"""Unit tests for AgentV2Service.delete() deletion guard (US-106).

Plan Section N.1 / N.2:
  - Default (force=False): raises ActiveDeploymentsError when count > 0.
  - Force=True: deactivates active rows, deletes terminal rows, deletes agent.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from deep_research.models.agent_deployment import (
    AgentDeployment,
    DeploymentMode,
    DeploymentStatus,
)
from deep_research.models.agent_v2 import AgentV2
from deep_research.models.visibility import AgentVisibility
from deep_research.services.agent_v2_service import (
    ActiveDeploymentsError,
    AgentV2Service,
)


def _make_active_deployment(agent_id: object) -> AgentDeployment:
    d = AgentDeployment(
        agent_id=agent_id,
        revision_id=uuid4(),
        mode=DeploymentMode.IN_APP.value,
        status=DeploymentStatus.ACTIVE.value,
        config={"mode": "in_app"},
        deployed_by="user-1",
    )
    d.id = uuid4()
    d.endpoint_name = None
    return d


def _make_agent(owner: str = "user-1") -> AgentV2:
    agent = AgentV2(
        id=uuid4(),
        owner_id=owner,
        name="research-agent",
        definition={},
        etag="abc",
        visibility=AgentVisibility.PRIVATE.value,
    )
    return agent


class TestDeleteWithoutActiveDeployments:
    @pytest.mark.asyncio
    async def test_delete_succeeds_when_no_deployments(
        self, mock_db_session: AsyncMock
    ) -> None:
        agent = _make_agent()
        service = AgentV2Service(mock_db_session)
        with (
            patch.object(service, "get_owned", AsyncMock(return_value=agent)),
            patch(
                "deep_research.services.agent_v2_service.DeploymentService"
            ) as MockSvc,
        ):
            instance = MockSvc.return_value
            instance.list_active_for_agent = AsyncMock(return_value=[])
            instance.delete_terminal_rows_for_agent = AsyncMock(return_value=0)

            result = await service.delete(agent.id, "user-1")

        assert result is True
        mock_db_session.delete.assert_awaited_once_with(agent)

    @pytest.mark.asyncio
    async def test_delete_returns_false_when_agent_missing(
        self, mock_db_session: AsyncMock
    ) -> None:
        service = AgentV2Service(mock_db_session)
        with patch.object(service, "get_owned", AsyncMock(return_value=None)):
            result = await service.delete(uuid4(), "user-1")
        assert result is False
        mock_db_session.delete.assert_not_called()


class TestDeleteWithActiveDeployments:
    @pytest.mark.asyncio
    async def test_default_raises_active_deployments_error(
        self, mock_db_session: AsyncMock
    ) -> None:
        agent = _make_agent()
        active = [
            _make_active_deployment(agent.id),
            _make_active_deployment(agent.id),
        ]
        service = AgentV2Service(mock_db_session)
        with (
            patch.object(service, "get_owned", AsyncMock(return_value=agent)),
            patch(
                "deep_research.services.agent_v2_service.DeploymentService"
            ) as MockSvc,
        ):
            MockSvc.return_value.list_active_for_agent = AsyncMock(
                return_value=active
            )

            with pytest.raises(ActiveDeploymentsError) as exc_info:
                await service.delete(agent.id, "user-1")

        assert exc_info.value.active_count == 2
        assert len(exc_info.value.deployments) == 2
        # Agent must NOT be deleted when guard fires.
        mock_db_session.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_force_runs_translator_cleanup_before_db_deactivate(
        self, mock_db_session: AsyncMock
    ) -> None:
        """W9 contract change: force=True MUST cascade through the per-mode
        translator (so external resources are torn down) BEFORE the DB-row
        deactivate flip. Pre-W9 the translator was skipped, leaking the
        external resources whenever a non-In-App deployment was force-
        deleted.
        """
        agent = _make_agent()
        active = [_make_active_deployment(agent.id)]
        service = AgentV2Service(mock_db_session)

        fake_translator = AsyncMock()
        with (
            patch.object(service, "get_owned", AsyncMock(return_value=agent)),
            patch(
                "deep_research.services.agent_v2_service.DeploymentService"
            ) as MockSvc,
            patch(
                "deep_research.services.agent_v2_service.translator_for",
                return_value=fake_translator,
            ) as mock_translator_for,
        ):
            instance = MockSvc.return_value
            instance.list_active_for_agent = AsyncMock(return_value=active)
            instance.deactivate = AsyncMock()
            instance.delete_terminal_rows_for_agent = AsyncMock(return_value=1)

            result = await service.delete(agent.id, "user-1", force=True)

        assert result is True
        # Translator deactivate must be called for the active deployment.
        mock_translator_for.assert_called_once_with(DeploymentMode.IN_APP)
        fake_translator.deactivate.assert_awaited_once_with(active[0])
        # Then the DB row is flipped to DEACTIVATED.
        instance.deactivate.assert_awaited_once_with(active[0].id)
        instance.delete_terminal_rows_for_agent.assert_awaited_once_with(
            agent.id
        )
        mock_db_session.delete.assert_awaited_once_with(agent)

    @pytest.mark.asyncio
    async def test_force_raises_cleanup_error_and_bumps_attempts(
        self, mock_db_session: AsyncMock
    ) -> None:
        """W9: if the translator raises DeploymentCleanupError during a
        force-delete cascade, the agent must NOT be deleted, the attempts
        counter increments, and the API receives the error to surface a
        409.
        """
        from deep_research.services.deployment import DeploymentCleanupError

        agent = _make_agent()
        # cleanup_attempts starts at 0; one failure → no cleanup_failed yet.
        deployment = _make_active_deployment(agent.id)
        deployment.cleanup_attempts = 0
        service = AgentV2Service(mock_db_session)

        fake_translator = AsyncMock()
        fake_translator.deactivate.side_effect = DeploymentCleanupError(
            "503 boom", resource="agents.delete_deployment"
        )
        with (
            patch.object(service, "get_owned", AsyncMock(return_value=agent)),
            patch(
                "deep_research.services.agent_v2_service.DeploymentService"
            ) as MockSvc,
            patch(
                "deep_research.services.agent_v2_service.translator_for",
                return_value=fake_translator,
            ),
        ):
            instance = MockSvc.return_value
            instance.list_active_for_agent = AsyncMock(return_value=[deployment])
            instance.increment_cleanup_attempts = AsyncMock()
            instance.mark_cleanup_failed = AsyncMock()

            with pytest.raises(DeploymentCleanupError):
                await service.delete(agent.id, "user-1", force=True)

        # First-attempt failure should bump attempts (NOT yet cleanup_failed).
        instance.increment_cleanup_attempts.assert_awaited_once_with(
            deployment.id
        )
        instance.mark_cleanup_failed.assert_not_called()
        # Agent must NOT be deleted when cleanup fails.
        mock_db_session.delete.assert_not_called()
