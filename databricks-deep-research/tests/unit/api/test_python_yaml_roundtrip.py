"""YAML ↔ Python round-trip contract.

For each canonical fixture, assert ``load_workflow(save_workflow(agent.as_workflow()))``
yields an IR that is structurally equal to the original Python compilation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from databricks_deep_research.api import Agent, Parallel, Sequence, SubAgent, tool
from databricks_deep_research.workflow.loader import load_workflow, save_workflow


def _roundtrip_equal(agent_or_composite) -> bool:  # type: ignore[no-untyped-def]
    wf = agent_or_composite.as_workflow()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        save_workflow(wf, f.name)
        path = f.name
    wf2 = load_workflow(path)
    Path(path).unlink(missing_ok=True)
    return wf.model_dump(mode="json") == wf2.model_dump(mode="json")


def test_simple_agent_roundtrips() -> None:
    @tool
    def search(q: str) -> str:
        """Search."""
        return q

    a = Agent(name="solo", instructions="Solo.", tools=[search])
    assert _roundtrip_equal(a)


def test_synthesizer_with_pool_specs_roundtrips() -> None:
    from databricks_deep_research.api import PoolWriteSpec

    a = Agent(
        name="syn",
        subtype="synthesizer",
        pool_writes=[PoolWriteSpec(pool="sources", extract="sources")],
    )
    assert _roundtrip_equal(a)


def test_subagent_tree_roundtrips() -> None:
    @tool
    def helper_tool(q: str) -> str:
        """H"""
        return q

    sub = SubAgent(name="helper", instructions="Help.", tools=[helper_tool])
    parent = Agent(name="boss", instructions="Lead.", subagents=[sub])
    assert _roundtrip_equal(parent)


def test_sequence_roundtrips() -> None:
    a = Agent(name="a")
    b = Agent(name="b")
    seq = Sequence(a, b, name="my_seq")
    assert _roundtrip_equal(seq)


def test_parallel_roundtrips() -> None:
    a = Agent(name="a")
    b = Agent(name="b")
    par = Parallel(a, b, name="my_par")
    assert _roundtrip_equal(par)
