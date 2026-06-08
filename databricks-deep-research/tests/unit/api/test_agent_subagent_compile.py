"""SubAgent compile tests: subworkflow nodes + compile-time task tool."""

from __future__ import annotations

import pytest

from databricks_deep_research.api import Agent, SubAgent, tool
from databricks_deep_research.workflow.definition import NodeType


def test_subagent_compiles_to_subworkflow_node() -> None:
    sub = SubAgent(name="helper", instructions="Help.")
    parent = Agent(name="boss", instructions="Coordinate.", subagents=[sub])
    wf = parent.as_workflow()

    # Root is a sequence wrapping agent + subworkflow children
    assert wf.root.type == NodeType.sequence
    child_types = [c.type for c in wf.root.children]
    assert NodeType.agent in child_types
    assert NodeType.subworkflow in child_types


def test_compile_time_task_tool_synthesized() -> None:
    sub_a = SubAgent(name="a")
    sub_b = SubAgent(name="b")
    parent = Agent(name="boss", subagents=[sub_a, sub_b])
    wf = parent.as_workflow()

    tool_names = [t.name for t in wf.tools]
    assert "task" in tool_names


def test_no_subagents_means_no_task_tool() -> None:
    @tool
    def f(x: str) -> str:
        """X"""
        return x

    parent = Agent(name="agent", tools=[f])
    wf = parent.as_workflow()
    tool_names = [t.name for t in wf.tools]
    assert "task" not in tool_names


def test_compile_does_not_mutate_agent_runtime_tools() -> None:
    @tool
    def f(x: str) -> str:
        """X"""
        return x

    sub = SubAgent(name="helper")
    parent = Agent(name="boss", tools=[f], subagents=[sub])

    before = list(parent.tools)
    parent.as_workflow()
    after = list(parent.tools)

    # Compile-time task injection is in the IR ToolDeclaration list,
    # NOT a mutation of the user's runtime ``tools`` collection.
    assert before == after


def test_subagent_pool_mode_preserved() -> None:
    sub = SubAgent(name="helper", pool_mode="isolate")
    parent = Agent(name="boss", subagents=[sub])
    wf = parent.as_workflow()
    sub_node = next(c for c in wf.root.children if c.type == NodeType.subworkflow)
    assert sub_node.config["pool_mode"] == "isolate"


def test_subagent_inner_inline_workflow_present() -> None:
    sub = SubAgent(name="helper", instructions="Inner")
    parent = Agent(name="boss", subagents=[sub])
    wf = parent.as_workflow()
    sub_node = next(c for c in wf.root.children if c.type == NodeType.subworkflow)
    assert "inline" in sub_node.config
    inline = sub_node.config["inline"]
    # Inline carries the inner workflow's serialized form
    assert inline["root"]["id"] == "helper"


def test_cycle_detection_raises() -> None:
    # Construct a parent agent and a subagent that *contains* the parent as a tool.
    parent = Agent(name="parent")
    sub = SubAgent(name="child", tools=[parent])
    parent.subagents = [sub]

    with pytest.raises(ValueError, match="Cycle detected"):
        parent.as_workflow()
