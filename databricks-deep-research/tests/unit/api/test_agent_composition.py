"""Composition tests: ``Sequence`` and ``Parallel`` IR structure."""

from __future__ import annotations

import pytest

from databricks_deep_research.api import Agent, Parallel, Sequence
from databricks_deep_research.workflow.definition import NodeType


def test_sequence_compiles_to_sequence_node() -> None:
    a = Agent(name="a")
    b = Agent(name="b")
    seq = Sequence(a, b)
    wf = seq.as_workflow()
    assert wf.root.type == NodeType.sequence
    assert len(wf.root.children) == 2


def test_parallel_compiles_to_parallel_node() -> None:
    a = Agent(name="a")
    b = Agent(name="b")
    par = Parallel(a, b)
    wf = par.as_workflow()
    assert wf.root.type == NodeType.parallel
    assert len(wf.root.children) == 2


def test_sequence_preserves_child_order() -> None:
    a = Agent(name="a")
    b = Agent(name="b")
    c = Agent(name="c")
    seq = Sequence(a, b, c)
    wf = seq.as_workflow()
    assert [c.id for c in wf.root.children] == ["a", "b", "c"]


def test_nested_composition_supported() -> None:
    a = Agent(name="a")
    b = Agent(name="b")
    c = Agent(name="c")
    nested = Sequence(a, Parallel(b, c))
    wf = nested.as_workflow()
    assert wf.root.type == NodeType.sequence
    assert wf.root.children[0].id == "a"
    assert wf.root.children[1].type == NodeType.parallel


def test_unsupported_child_type_raises() -> None:
    seq = Sequence("not an agent")
    with pytest.raises(TypeError, match="Unsupported composition child type"):
        seq.as_workflow()


def test_composition_dedupes_tool_declarations() -> None:
    from databricks_deep_research.api import tool

    @tool
    def shared(msg: str) -> str:
        """S"""
        return msg

    a = Agent(name="a", tools=[shared])
    b = Agent(name="b", tools=[shared])
    seq = Sequence(a, b)
    wf = seq.as_workflow()
    assert sum(1 for t in wf.tools if t.name == "shared") == 1
