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


def _make_failed_deployment(agent_id: object) -> AgentDeployment:
    d = AgentDeployment(
        agent_id=agent_id,
        revision_id=uuid4(),
        mode=DeploymentMode.SHELL_APP.value,
        status=DeploymentStatus.FAILED.value,
        config={"mode": "shell_app"},
        deployed_by="user-1",
        error_message="server_shutdown",
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
            instance.list_failed_for_agent = AsyncMock(return_value=[])
            instance.deactivate = AsyncMock()
            instance.delete_terminal_rows_for_agent = AsyncMock(return_value=1)

            result = await service.delete(agent.id, "user-1", force=True)

        assert result is True
        # Translator deactivate must be called for the active deployment.
        mock_translator_for.assert_called_once_with(DeploymentMode.IN_APP)
        fake_translator.deactivate.assert_awaited_once_with(
            active[0], client_resolver=None
        )
        # Then the DB row is flipped to DEACTIVATED.
        instance.deactivate.assert_awaited_once_with(active[0].id)
        instance.delete_terminal_rows_for_agent.assert_awaited_once_with(
            agent.id, include_failed=True
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

    @pytest.mark.asyncio
    async def test_force_handles_cleanup_exhausted_in_one_shot(
        self, mock_db_session: AsyncMock
    ) -> None:
        """DeploymentCleanupExhaustedError signals a deterministic failure
        (e.g., workspace.delete denied by both OBO and SP). The service must:
          1. Mark the row cleanup_failed ONCE (no attempts bump).
          2. NOT re-raise — proceed to the FK-clearing pass.
          3. Physically remove the cleanup_failed row.
          4. Delete the parent agent.
        Burning 3 retries for a deterministic failure is exactly the UX bug
        this exception was introduced to avoid.
        """
        from deep_research.services.deployment import (
            DeploymentCleanupExhaustedError,
        )

        agent = _make_agent()
        deployment = _make_active_deployment(agent.id)
        deployment.cleanup_attempts = 0
        service = AgentV2Service(mock_db_session)

        fake_translator = AsyncMock()
        fake_translator.deactivate.side_effect = DeploymentCleanupExhaustedError(
            "workspace.delete denied by both OBO and SP. path=/Workspace/Shared/x",
            resource="workspace.delete",
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
            instance.list_failed_for_agent = AsyncMock(return_value=[])
            instance.increment_cleanup_attempts = AsyncMock()
            instance.mark_cleanup_failed = AsyncMock()
            instance.deactivate = AsyncMock()
            instance.delete_terminal_rows_for_agent = AsyncMock(return_value=1)

            # MUST NOT raise — the exhausted variant is handled internally.
            result = await service.delete(agent.id, "user-1", force=True)

        assert result is True
        # cleanup_failed marked exactly once, with the exception message.
        instance.mark_cleanup_failed.assert_awaited_once()
        called_id, called_kwargs = (
            instance.mark_cleanup_failed.await_args.args[0],
            instance.mark_cleanup_failed.await_args.kwargs,
        )
        assert called_id == deployment.id
        assert "workspace.delete" in called_kwargs["error_message"]
        # Attempts counter MUST NOT be bumped (that's the whole point —
        # deterministic failure should not consume a retry slot).
        instance.increment_cleanup_attempts.assert_not_called()
        # Row physically deleted by the FK-clearing pass.
        instance.delete_terminal_rows_for_agent.assert_awaited_once_with(
            agent.id, include_failed=True
        )
        # Parent agent deleted.
        mock_db_session.delete.assert_awaited_once_with(agent)


class TestForceDeleteCascadesFailedRows:
    """D1: force=True must flip FAILED rows to DEACTIVATED before the FK
    delete so the recovery sweep's ``error_message='server_shutdown'`` rows
    don't permanently block ``DELETE /agents-v2/{id}``.
    """

    @pytest.mark.asyncio
    async def test_force_delete_with_only_failed_deployment_succeeds(
        self, mock_db_session: AsyncMock
    ) -> None:
        """Agent with only a FAILED deployment (no active) is deletable via
        force=true in a single click — previously this raised IntegrityError
        because FAILED is in neither ACTIVE_STATUSES nor DELETABLE_STATUSES.
        """
        agent = _make_agent()
        failed = _make_failed_deployment(agent.id)
        service = AgentV2Service(mock_db_session)

        with (
            patch.object(service, "get_owned", AsyncMock(return_value=agent)),
            patch(
                "deep_research.services.agent_v2_service.DeploymentService"
            ) as MockSvc,
        ):
            instance = MockSvc.return_value
            instance.list_active_for_agent = AsyncMock(return_value=[])
            instance.list_failed_for_agent = AsyncMock(return_value=[failed])
            instance.deactivate = AsyncMock()
            instance.delete_terminal_rows_for_agent = AsyncMock(return_value=1)

            result = await service.delete(agent.id, "user-1", force=True)

        assert result is True
        # FAILED row was flipped to DEACTIVATED (no translator call).
        instance.deactivate.assert_awaited_once_with(failed.id)
        # Terminal-row cleanup must include FAILED rows.
        instance.delete_terminal_rows_for_agent.assert_awaited_once_with(
            agent.id, include_failed=True
        )
        mock_db_session.delete.assert_awaited_once_with(agent)

    @pytest.mark.asyncio
    async def test_force_delete_with_mixed_active_and_failed_succeeds(
        self, mock_db_session: AsyncMock
    ) -> None:
        """Force-delete handles both ACTIVE (via translator) and FAILED (no
        translator) rows in a single transaction.
        """
        agent = _make_agent()
        active = _make_active_deployment(agent.id)
        failed = _make_failed_deployment(agent.id)
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
            ),
        ):
            instance = MockSvc.return_value
            instance.list_active_for_agent = AsyncMock(return_value=[active])
            instance.list_failed_for_agent = AsyncMock(return_value=[failed])
            instance.deactivate = AsyncMock()
            instance.delete_terminal_rows_for_agent = AsyncMock(return_value=2)

            result = await service.delete(agent.id, "user-1", force=True)

        assert result is True
        # Translator called only for the ACTIVE row.
        fake_translator.deactivate.assert_awaited_once_with(
            active, client_resolver=None
        )
        # Both rows flipped to DEACTIVATED via the service.
        assert instance.deactivate.await_count == 2
        instance.deactivate.assert_any_await(active.id)
        instance.deactivate.assert_any_await(failed.id)
        mock_db_session.delete.assert_awaited_once_with(agent)

    @pytest.mark.asyncio
    async def test_default_delete_does_not_cascade_failed(
        self, mock_db_session: AsyncMock
    ) -> None:
        """force=False preserves forensics: FAILED rows are NOT touched and
        will still block the FK delete (translated to 409 by the API layer
        via the new IntegrityError handler).
        """
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
            instance.list_failed_for_agent = AsyncMock(return_value=[])
            instance.delete_terminal_rows_for_agent = AsyncMock(return_value=0)

            await service.delete(agent.id, "user-1", force=False)

        # FAILED cascade must NOT run on the default path.
        instance.list_failed_for_agent.assert_not_called()
        # Terminal-row cleanup uses include_failed=False on the default path.
        instance.delete_terminal_rows_for_agent.assert_awaited_once_with(
            agent.id, include_failed=False
        )


class TestForceDeleteThreadsResolver:
    """D3: the WorkspaceClientResolver supplied by the API handler must
    reach translator.deactivate so shell_app / mlflow can use the user's
    OBO-scoped client instead of the SP.
    """

    @pytest.mark.asyncio
    async def test_force_delete_passes_resolver_to_translator(
        self, mock_db_session: AsyncMock
    ) -> None:
        from deep_research.services.deployment.auth import (
            WorkspaceClientResolver,
        )

        agent = _make_agent()
        active = _make_active_deployment(agent.id)
        service = AgentV2Service(mock_db_session)

        resolver = WorkspaceClientResolver(obo_client=None)
        fake_translator = AsyncMock()
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
            instance.list_active_for_agent = AsyncMock(return_value=[active])
            instance.list_failed_for_agent = AsyncMock(return_value=[])
            instance.deactivate = AsyncMock()
            instance.delete_terminal_rows_for_agent = AsyncMock(return_value=1)

            await service.delete(
                agent.id, "user-1", force=True, client_resolver=resolver
            )

        fake_translator.deactivate.assert_awaited_once_with(
            active, client_resolver=resolver
        )
