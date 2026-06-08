"""``Team(strategy='round_robin')`` compile tests."""

from __future__ import annotations

import pytest

from databricks_deep_research.api import Agent, SubAgent, Team
from databricks_deep_research.workflow.definition import NodeType


def test_round_robin_compiles_to_sequence() -> None:
    members = [Agent(name=f"m{i}") for i in range(3)]
    team = Team(members=members, strategy="round_robin")
    wf = team.as_workflow()
    assert wf.root.type == NodeType.sequence
    assert len(wf.root.children) == 3


def test_round_robin_preserves_order() -> None:
    members = [Agent(name=f"m{i}") for i in range(3)]
    team = Team(members=members, strategy="round_robin")
    wf = team.as_workflow()
    assert [c.id for c in wf.root.children] == ["m0", "m1", "m2"]


def test_round_robin_accepts_subagents() -> None:
    members = [SubAgent(name="sub1", instructions="Hi.")]
    team = Team(members=members, strategy="round_robin")
    wf = team.as_workflow()
    assert wf.root.children[0].id == "sub1"


def test_round_robin_does_not_require_leader() -> None:
    members = [Agent(name="m1"), Agent(name="m2")]
    # Should not raise.
    team = Team(members=members, strategy="round_robin")
    assert team.leader is None


def test_round_robin_rejects_invalid_member_type() -> None:
    with pytest.raises(TypeError, match="Agent or SubAgent"):
        team = Team(members=["not an agent"], strategy="round_robin")
        team.as_workflow()
