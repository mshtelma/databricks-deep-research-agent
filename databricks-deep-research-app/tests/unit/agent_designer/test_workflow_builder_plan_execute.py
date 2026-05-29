"""Phase 0 of the dataflow-enforcement plan.

The plan_and_execute body must be the *direct researcher* (the dead body
reflector whose control decision nothing read is removed), and the synthesizer
must read a real ``{evaluation}`` slot (not a no-op ``input_keys`` entry).

See docs/superpowers/plans/2026-05-29-scoped-bindings-dataflow-enforcement.md (Phase 0).
"""
from __future__ import annotations

import json
from collections.abc import Iterator

from deep_research.agent_designer.designer_types import (
    ToolDeclarationSpec,
    ToolPlan,
    WorkflowDesignBrief,
)
from deep_research.agent_designer.workflow_builder import build_web_research_workflow


def _walk_nodes(node: dict) -> Iterator[dict]:
    yield node
    config = node.get("config") if isinstance(node.get("config"), dict) else {}
    for nested_key in ("body", "evaluator", "planner"):
        nested = config.get(nested_key)
        if isinstance(nested, dict):
            yield from _walk_nodes(nested)
    for child in node.get("children") or []:
        if isinstance(child, dict):
            yield from _walk_nodes(child)


def _plan_execute_brief() -> WorkflowDesignBrief:
    return WorkflowDesignBrief(
        workflow_name="pae-phase0",
        topology="plan_and_execute",
        research_lanes=[
            {
                "description": "Web research lane.",
                "user_prompt_template": "Research {query}.",
            }
        ],
        tool_plan=ToolPlan(tools=[ToolDeclarationSpec(name="web", kind="web_search")]),
    )


def _build() -> dict:
    return build_web_research_workflow(
        intent="phase0 test", design_brief=_plan_execute_brief()
    )


def _find_node(ast: dict, node_id: str) -> dict:
    for node in _walk_nodes(ast["root"]):
        if node.get("id") == node_id:
            return node
    raise AssertionError(f"node {node_id!r} not found in workflow")


def test_plan_execute_body_is_direct_researcher() -> None:
    ast = _build()
    pae = _find_node(ast, "plan-and-execute")
    body = pae["config"]["body"]
    assert body["type"] == "agent", f"body is {body['type']!r}, expected 'agent'"
    assert body["config"]["subtype"] == "researcher"
    # The dead body reflector (output_key='reflection') must be gone entirely.
    assert all(
        n.get("config", {}).get("output_key") != "reflection"
        for n in _walk_nodes(ast["root"])
    )


def test_synthesizer_reads_evaluation_with_real_slot() -> None:
    ast = _build()
    synth = _find_node(ast, "synthesizer")
    assert "evaluation" in synth["config"]["input_keys"]
    assert "reflection" not in synth["config"]["input_keys"]
    # The slot must be referenced in the prompt so the read is not a no-op
    # (raw input_keys only render if the template references them).
    assert "{evaluation}" in json.dumps(synth["config"])
