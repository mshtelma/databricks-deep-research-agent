"""Unit tests for deriving an agent's default 'verify sources' from its saved AST.

``_default_verify_sources`` reads the synthesizer node's ``grounding_mode`` so the
chat composer can seed the verify toggle from the agent's authored intent. Also
asserts the value is propagated to ``AgentV2Summary`` via ``list_for_user``.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from deep_research.models.agent_v2 import AgentV2
from deep_research.models.visibility import AgentVisibility
from deep_research.services.agent_v2_service import (
    AgentV2Service,
    _default_verify_sources,
)


def _agent_node(grounding_mode: str | None) -> dict[str, Any]:
    config: dict[str, Any] = {"subtype": "synthesizer"}
    if grounding_mode is not None:
        config["grounding_mode"] = grounding_mode
    return {"config": config, "children": []}


def _coordinator_node() -> dict[str, Any]:
    return {"config": {"subtype": "coordinator"}, "children": []}


def test_reclaim_synth_defaults_verify_on() -> None:
    definition = {"root": _agent_node("reclaim")}
    assert _default_verify_sources(definition) is True


def test_classical_lite_synth_defaults_verify_off() -> None:
    definition = {"root": _agent_node("classical_lite")}
    assert _default_verify_sources(definition) is False


def test_none_synth_defaults_verify_off() -> None:
    definition = {"root": _agent_node("none")}
    assert _default_verify_sources(definition) is False


def test_synth_without_grounding_defaults_verify_on() -> None:
    """A synthesizer that omits grounding_mode falls back to the reclaim safe floor."""
    definition = {"root": _agent_node(None)}
    assert _default_verify_sources(definition) is True


def test_no_synthesizer_defaults_verify_on() -> None:
    definition = {"root": _coordinator_node()}
    assert _default_verify_sources(definition) is True


def test_finds_synth_nested_in_children() -> None:
    definition = {
        "root": {
            "config": {},
            "children": [_coordinator_node(), _agent_node("classical_lite")],
        }
    }
    assert _default_verify_sources(definition) is False


def test_finds_synth_nested_in_loop_body() -> None:
    """Synthesizers inside a plan_and_execute / loop body (config.body) are found."""
    definition = {
        "root": {
            "config": {"body": _agent_node("reclaim")},
            "children": [],
        }
    }
    assert _default_verify_sources(definition) is True


def test_empty_definition_defaults_verify_on() -> None:
    assert _default_verify_sources({"root": {}}) is True
    assert _default_verify_sources({}) is True


def _make_agent(definition: dict[str, Any]) -> AgentV2:
    agent = AgentV2(
        owner_id="user-1",
        name="Test Agent",
        description=None,
        visibility=AgentVisibility.PRIVATE.value,
        definition=definition,
        schema_version=1,
        etag="abc123",
    )
    agent.id = uuid4()
    agent.created_at = datetime.now(tz=UTC)
    agent.updated_at = datetime.now(tz=UTC)
    return agent


@pytest.mark.asyncio
async def test_list_for_user_propagates_default_verify_sources(
    mock_db_session: AsyncMock,
) -> None:
    """End-to-end: the derived value reaches AgentV2Summary.default_verify_sources."""
    reclaim_agent = _make_agent({"root": _agent_node("reclaim")})
    lite_agent = _make_agent({"root": _agent_node("classical_lite")})
    result = MagicMock()
    result.all.return_value = [(reclaim_agent, False), (lite_agent, False)]
    mock_db_session.execute.return_value = result

    summaries = await AgentV2Service(mock_db_session).list_for_user("user-1")

    assert summaries[0].default_verify_sources is True
    assert summaries[1].default_verify_sources is False
