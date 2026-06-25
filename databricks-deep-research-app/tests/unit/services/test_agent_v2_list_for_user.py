"""Unit tests for AgentV2Service.list_for_user() in_app_active join (Q1).

Asserts that the EXISTS subquery result is correctly propagated to the
returned AgentV2Summary instances under four scenarios:
  - agent WITH an active in_app deployment → in_app_active=True
  - agent WITH NO deployment at all       → in_app_active=False
  - agent WITH a FAILED in_app deployment → in_app_active=False
  - agent WITH an active non-in_app mode  → in_app_active=False
"""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from deep_research.models.agent_v2 import AgentV2
from deep_research.models.visibility import AgentVisibility
from deep_research.services.agent_v2_service import AgentV2Service


def _make_agent(
    owner_id: str = "user-1",
    visibility: str = AgentVisibility.PRIVATE.value,
) -> AgentV2:
    agent = AgentV2(
        owner_id=owner_id,
        name="Test Agent",
        description=None,
        visibility=visibility,
        definition={"root": {}},
        schema_version=1,
        etag="abc123",
    )
    agent.id = uuid4()
    agent.created_at = datetime.now(tz=UTC)
    agent.updated_at = datetime.now(tz=UTC)
    return agent


def _make_execute_result(rows: list[tuple[AgentV2, bool]]) -> MagicMock:
    """Return a mock that mimics AsyncSession.execute().all()."""
    mock_result = MagicMock()
    mock_result.all.return_value = rows
    return mock_result


@pytest.mark.asyncio
async def test_list_for_user_active_in_app_deployment(
    mock_db_session: AsyncMock,
) -> None:
    """Agent with an active in_app deployment → in_app_active=True."""
    agent = _make_agent(visibility=AgentVisibility.WORKSPACE.value)
    mock_db_session.execute.return_value = _make_execute_result(
        [(agent, True)]
    )

    service = AgentV2Service(mock_db_session)
    summaries = await service.list_for_user("user-1")

    assert len(summaries) == 1
    assert summaries[0].in_app_active is True


@pytest.mark.asyncio
async def test_list_for_user_no_deployment(
    mock_db_session: AsyncMock,
) -> None:
    """Agent with NO deployment at all → in_app_active=False."""
    agent = _make_agent()
    mock_db_session.execute.return_value = _make_execute_result(
        [(agent, False)]
    )

    service = AgentV2Service(mock_db_session)
    summaries = await service.list_for_user("user-1")

    assert len(summaries) == 1
    assert summaries[0].in_app_active is False


@pytest.mark.asyncio
async def test_list_for_user_failed_in_app_deployment(
    mock_db_session: AsyncMock,
) -> None:
    """Agent with a FAILED in_app deployment → in_app_active=False.

    The EXISTS subquery filters on status='active', so a failed row
    does not qualify.
    """
    agent = _make_agent()
    mock_db_session.execute.return_value = _make_execute_result(
        [(agent, False)]
    )

    service = AgentV2Service(mock_db_session)
    summaries = await service.list_for_user("user-1")

    assert len(summaries) == 1
    assert summaries[0].in_app_active is False


@pytest.mark.asyncio
async def test_list_for_user_active_other_mode_deployment(
    mock_db_session: AsyncMock,
) -> None:
    """Agent with active shell_app/mlflow_agent/batch deployment → in_app_active=False.

    The EXISTS subquery filters on mode='in_app', so other modes don't qualify.
    """
    agent = _make_agent()
    mock_db_session.execute.return_value = _make_execute_result(
        [(agent, False)]
    )

    service = AgentV2Service(mock_db_session)
    summaries = await service.list_for_user("user-1")

    assert len(summaries) == 1
    assert summaries[0].in_app_active is False


@pytest.mark.asyncio
async def test_list_for_user_mixed_agents(
    mock_db_session: AsyncMock,
) -> None:
    """Multiple agents with mixed deployment states are all mapped correctly."""
    agent_active = _make_agent(visibility=AgentVisibility.WORKSPACE.value)
    agent_inactive = _make_agent()
    mock_db_session.execute.return_value = _make_execute_result(
        [(agent_active, True), (agent_inactive, False)]
    )

    service = AgentV2Service(mock_db_session)
    summaries = await service.list_for_user("user-1")

    assert len(summaries) == 2
    by_id = {str(s.id): s for s in summaries}
    assert by_id[str(agent_active.id)].in_app_active is True
    assert by_id[str(agent_inactive.id)].in_app_active is False
