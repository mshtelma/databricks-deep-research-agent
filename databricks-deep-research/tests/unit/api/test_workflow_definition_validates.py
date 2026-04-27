"""Compiled :class:`WorkflowDefinition` survives ``extra="forbid"``."""

from __future__ import annotations

import pytest

from databricks_deep_research.agents.config import AgentNodeConfig
from databricks_deep_research.api import Agent, SubAgent, tool


def test_compiled_agent_node_validates_against_agentnodeconfig() -> None:
    @tool
    def f(x: str) -> str:
        """F"""
        return x

    agent = Agent(name="x", instructions="i", tools=[f], extras={"_framework_thread_id": "t"})
    wf = agent.as_workflow()
    cfg_dict = wf.root.config

    # AgentNodeConfig has extra="forbid"; must accept this dict.
    parsed = AgentNodeConfig(**cfg_dict)
    assert parsed.subtype == "custom"
    assert parsed.extras == {"_framework_thread_id": "t"}


def test_unknown_field_in_compiled_config_rejected() -> None:
    agent = Agent(name="x")
    wf = agent.as_workflow()
    bogus = dict(wf.root.config, totally_unknown_field=123)
    with pytest.raises(Exception):
        AgentNodeConfig(**bogus)


def test_subworkflow_inner_agent_node_validates() -> None:
    sub = SubAgent(name="sub", instructions="i")
    parent = Agent(name="boss", subagents=[sub])
    wf = parent.as_workflow()

    sub_node = next(c for c in wf.root.children if c.type.value == "subworkflow")
    inline = sub_node.config["inline"]
    inner_agent_cfg = inline["root"]["config"]

    parsed = AgentNodeConfig(**inner_agent_cfg)
    assert parsed.subtype == "custom"


def test_extras_field_survives_round_trip() -> None:
    parsed = AgentNodeConfig(subtype="custom", extras={"_framework_x": "y"})
    dumped = parsed.model_dump()
    reparsed = AgentNodeConfig(**dumped)
    assert reparsed.extras == {"_framework_x": "y"}
