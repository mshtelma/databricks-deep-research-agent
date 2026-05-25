"""Unit tests for DeploymentService (US-103).

Use mocked AsyncSession (no real DB). Each test instantiates
``DeploymentService(mock_db_session)`` and verifies side-effects on the mock.
"""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from deep_research.models.agent_deployment import (
    AgentDeployment,
    DeploymentMode,
    DeploymentStatus,
)
from deep_research.services.deployment_service import (
    DeploymentService,
    _decode_cursor,
    _encode_cursor,
)


def _make_deployment(
    *,
    deployment_id: object | None = None,
    agent_id: object | None = None,
    status: DeploymentStatus = DeploymentStatus.PENDING,
) -> AgentDeployment:
    d = AgentDeployment(
        agent_id=agent_id or uuid4(),
        revision_id=uuid4(),
        mode=DeploymentMode.IN_APP.value,
        status=status.value,
        config={"mode": "in_app"},
        deployed_by="user-1",
    )
    d.id = deployment_id or uuid4()
    d.cleanup_attempts = 0
    d.created_at = datetime.now(UTC)
    d.updated_at = datetime.now(UTC)
    d.deactivated_at = None
    d.endpoint_name = None
    d.model_name = None
    d.external_resource_ids = None
    d.error_message = None
    return d


class TestCursor:
    def test_encode_decode_round_trip(self) -> None:
        ts = datetime.now(UTC)
        did = uuid4()
        cursor = _encode_cursor(ts, did)
        revived_ts, revived_id = _decode_cursor(cursor)
        assert revived_ts == ts
        assert revived_id == did


class TestCreate:
    @pytest.mark.asyncio
    async def test_create_writes_pending_row(
        self, mock_db_session: AsyncMock
    ) -> None:
        service = DeploymentService(mock_db_session)
        deployment = await service.create(
            agent_id=uuid4(),
            revision_id=uuid4(),
            mode=DeploymentMode.IN_APP,
            config={"mode": "in_app"},
            deployed_by="user-1",
        )
        mock_db_session.add.assert_called_once()
        assert deployment.status == DeploymentStatus.PENDING.value
        assert deployment.mode == DeploymentMode.IN_APP.value
        assert deployment.config == {"mode": "in_app"}


class TestUpdateStatus:
    @pytest.mark.asyncio
    async def test_update_status_transitions_to_active(
        self, mock_db_session: AsyncMock
    ) -> None:
        deployment = _make_deployment(status=DeploymentStatus.PENDING)
        mock_db_session.get = AsyncMock(return_value=deployment)
        service = DeploymentService(mock_db_session)

        result = await service.update_status(
            deployment.id,
            DeploymentStatus.ACTIVE,
            endpoint_name="dr-shell-foo",
            external_resource_ids={"app_name": "dr-shell-foo"},
        )
        assert result.status == DeploymentStatus.ACTIVE.value
        assert result.endpoint_name == "dr-shell-foo"
        assert result.external_resource_ids == {"app_name": "dr-shell-foo"}
        mock_db_session.flush.assert_awaited()

    @pytest.mark.asyncio
    async def test_update_status_raises_on_missing_deployment(
        self, mock_db_session: AsyncMock
    ) -> None:
        mock_db_session.get = AsyncMock(return_value=None)
        service = DeploymentService(mock_db_session)
        with pytest.raises(ValueError, match="not found"):
            await service.update_status(uuid4(), DeploymentStatus.ACTIVE)


class TestDeactivate:
    @pytest.mark.asyncio
    async def test_deactivate_sets_status_and_timestamp(
        self, mock_db_session: AsyncMock
    ) -> None:
        deployment = _make_deployment(status=DeploymentStatus.ACTIVE)
        mock_db_session.get = AsyncMock(return_value=deployment)
        service = DeploymentService(mock_db_session)

        result = await service.deactivate(deployment.id)
        assert result.status == DeploymentStatus.DEACTIVATED.value
        assert result.deactivated_at is not None


class TestMarkCleanupFailed:
    @pytest.mark.asyncio
    async def test_marks_terminal_with_error(
        self, mock_db_session: AsyncMock
    ) -> None:
        deployment = _make_deployment(status=DeploymentStatus.ACTIVE)
        mock_db_session.get = AsyncMock(return_value=deployment)
        service = DeploymentService(mock_db_session)

        result = await service.mark_cleanup_failed(
            deployment.id, "endpoint delete returned 500"
        )
        assert result.status == DeploymentStatus.CLEANUP_FAILED.value
        assert result.error_message == "endpoint delete returned 500"
        assert result.deactivated_at is not None


class TestCountActive:
    @pytest.mark.asyncio
    async def test_count_active_for_agent(
        self, mock_db_session: AsyncMock
    ) -> None:
        result = MagicMock()
        result.scalar_one.return_value = 3
        mock_db_session.execute = AsyncMock(return_value=result)
        service = DeploymentService(mock_db_session)

        count = await service.count_active_for_agent(uuid4())
        assert count == 3
        mock_db_session.execute.assert_awaited_once()


class TestIncrementCleanupAttempts:
    @pytest.mark.asyncio
    async def test_increments_counter(
        self, mock_db_session: AsyncMock
    ) -> None:
        deployment = _make_deployment(status=DeploymentStatus.ACTIVE)
        deployment.cleanup_attempts = 1
        mock_db_session.get = AsyncMock(return_value=deployment)
        service = DeploymentService(mock_db_session)

        result = await service.increment_cleanup_attempts(deployment.id)
        assert result.cleanup_attempts == 2


class TestListFailedForAgent:
    """list_failed_for_agent powers the force-delete FAILED cascade (D1)."""

    @pytest.mark.asyncio
    async def test_returns_only_failed_rows(
        self, mock_db_session: AsyncMock
    ) -> None:
        agent_id = uuid4()
        failed_row = _make_deployment(
            agent_id=agent_id, status=DeploymentStatus.FAILED
        )
        scalars = MagicMock()
        scalars.all.return_value = [failed_row]
        result = MagicMock()
        result.scalars.return_value = scalars
        mock_db_session.execute = AsyncMock(return_value=result)
        service = DeploymentService(mock_db_session)

        rows = await service.list_failed_for_agent(agent_id)
        assert rows == [failed_row]
        # Statement targets FAILED status only.
        executed_stmt = str(mock_db_session.execute.call_args.args[0])
        assert "status" in executed_stmt


class TestDeleteTerminalRowsIncludeFailed:
    """delete_terminal_rows_for_agent must widen its deletable set when
    invoked from force-delete (include_failed=True) so the FK can clear.
    """

    @pytest.mark.asyncio
    async def test_default_excludes_failed_rows(
        self, mock_db_session: AsyncMock
    ) -> None:
        agent_id = uuid4()
        deactivated = _make_deployment(
            agent_id=agent_id, status=DeploymentStatus.DEACTIVATED
        )
        scalars = MagicMock()
        scalars.all.return_value = [deactivated]
        result = MagicMock()
        result.scalars.return_value = scalars
        mock_db_session.execute = AsyncMock(return_value=result)
        mock_db_session.delete = AsyncMock()
        service = DeploymentService(mock_db_session)

        count = await service.delete_terminal_rows_for_agent(agent_id)
        # Single SELECT (terminal) executed; no FAILED SELECT issued.
        assert mock_db_session.execute.await_count == 1
        assert count == 1

    @pytest.mark.asyncio
    async def test_include_failed_runs_extra_select(
        self, mock_db_session: AsyncMock
    ) -> None:
        agent_id = uuid4()
        deactivated = _make_deployment(
            agent_id=agent_id, status=DeploymentStatus.DEACTIVATED
        )
        failed = _make_deployment(
            agent_id=agent_id, status=DeploymentStatus.FAILED
        )

        # Two SELECTs in sequence: terminal then failed.
        terminal_scalars = MagicMock()
        terminal_scalars.all.return_value = [deactivated]
        terminal_result = MagicMock()
        terminal_result.scalars.return_value = terminal_scalars

        failed_scalars = MagicMock()
        failed_scalars.all.return_value = [failed]
        failed_result = MagicMock()
        failed_result.scalars.return_value = failed_scalars

        mock_db_session.execute = AsyncMock(
            side_effect=[terminal_result, failed_result]
        )
        mock_db_session.delete = AsyncMock()
        service = DeploymentService(mock_db_session)

        count = await service.delete_terminal_rows_for_agent(
            agent_id, include_failed=True
        )
        assert count == 2
        assert mock_db_session.execute.await_count == 2
        # Both rows passed to session.delete.
        assert mock_db_session.delete.await_count == 2


class TestPaginationBounds:
    @pytest.mark.asyncio
    async def test_list_for_user_caps_limit_at_max(
        self, mock_db_session: AsyncMock
    ) -> None:
        scalars = MagicMock()
        scalars.all.return_value = []
        result = MagicMock()
        result.scalars.return_value = scalars
        mock_db_session.execute = AsyncMock(return_value=result)
        service = DeploymentService(mock_db_session)

        # Request limit=10000; service must clamp to 200.
        items, next_cursor = await service.list_for_user("user-1", limit=10_000)
        assert items == []
        assert next_cursor is None
        # Inspect the bound limit on the executed statement.
        executed_stmt = mock_db_session.execute.call_args.args[0]
        assert executed_stmt._limit == 201  # bounded_limit (200) + 1 for cursor

    @pytest.mark.asyncio
    async def test_list_for_user_floors_limit_at_one(
        self, mock_db_session: AsyncMock
    ) -> None:
        scalars = MagicMock()
        scalars.all.return_value = []
        result = MagicMock()
        result.scalars.return_value = scalars
        mock_db_session.execute = AsyncMock(return_value=result)
        service = DeploymentService(mock_db_session)

        await service.list_for_user("user-1", limit=0)
        executed_stmt = mock_db_session.execute.call_args.args[0]
        assert executed_stmt._limit == 2  # bounded_limit (1) + 1
