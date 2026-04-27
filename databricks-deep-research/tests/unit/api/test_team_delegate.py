"""``Team(strategy='delegate')`` compile tests."""

from __future__ import annotations

import pytest

from databricks_deep_research.api import Agent, SubAgent, Team
from databricks_deep_research.workflow.definition import NodeType


def test_delegate_compiles_root_with_subworkflow_children() -> None:
    leader = Agent(name="leader", instructions="Coordinate.")
    members = [
        Agent(name="researcher", instructions="Research."),
        Agent(name="synthesizer", instructions="Synthesize."),
    ]
    team = Team(members=members, leader=leader, strategy="delegate")
    wf = team.as_workflow()

    # Root is the sequence wrapping leader agent + subworkflow members.
    assert wf.root.type == NodeType.sequence
    child_types = [c.type for c in wf.root.children]
    assert NodeType.agent in child_types
    assert NodeType.subworkflow in child_types


def test_delegate_synthesizes_task_tool() -> None:
    leader = Agent(name="leader")
    members = [Agent(name="m1"), Agent(name="m2")]
    team = Team(members=members, leader=leader, strategy="delegate")
    wf = team.as_workflow()
    tool_names = [t.name for t in wf.tools]
    assert "task" in tool_names


def test_delegate_requires_leader() -> None:
    with pytest.raises(ValueError, match="requires `leader`"):
        Team(members=[Agent(name="x")], strategy="delegate")


def test_delegate_accepts_subagents_as_members() -> None:
    leader = Agent(name="leader")
    member = SubAgent(name="m1", instructions="Help.")
    team = Team(members=[member], leader=leader, strategy="delegate")
    wf = team.as_workflow()
    sub_nodes = [c for c in wf.root.children if c.type == NodeType.subworkflow]
    assert len(sub_nodes) == 1
    assert sub_nodes[0].id == "m1"


def test_delegate_member_order_preserved() -> None:
    leader = Agent(name="leader")
    members = [Agent(name=f"m{i}") for i in range(5)]
    team = Team(members=members, leader=leader, strategy="delegate")
    wf = team.as_workflow()
    sub_ids = [c.id for c in wf.root.children if c.type == NodeType.subworkflow]
    assert sub_ids == [f"m{i}" for i in range(5)]
