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
from deep_research.agent_designer.semantic_validation import (
    detect_unspecialized_agents,
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


# ---------------------------------------------------------------------------
# PAE body specialization: the body must thread the scaffold/architect lane
# system_prompt instead of falling to the generic builtin. Reproduces the
# officeqa designer quality-gate failure (Check 2 in semantic_validation:
# "still on the generic researcher prompt") that fired only when the
# classifier routed the task to plan_and_execute.
# ---------------------------------------------------------------------------

_LANE_SPECIALIZATION = (
    "You are a resource-grounded research agent for this workflow. Use only "
    "the declared runtime tools (vector_search, table_read, compute) and "
    "preserve the evidence contract. Investigate the Treasury bulletin "
    "tables; cite exact rows; mark missing values as Data unavailable."
)

# The framework's generic builtin researcher opening (semantic_validation's
# _DEFAULT_METHOD_OPENING_MARKER). Its presence is what Check 2 blocks on.
_GENERIC_MARKER = "You are the Researcher agent for a deep research system."


def _plan_execute_brief_specialized() -> WorkflowDesignBrief:
    return WorkflowDesignBrief(
        workflow_name="pae-specialized",
        topology="plan_and_execute",
        research_lanes=[
            {
                "description": "Corpus lookup over Treasury chunks.",
                "system_prompt": _LANE_SPECIALIZATION,
                "user_prompt_template": "Investigate {query} over the corpus.",
            }
        ],
        tool_plan=ToolPlan(
            tools=[ToolDeclarationSpec(name="vs", kind="vector_search")]
        ),
    )


def test_pae_body_threads_lane_specialization() -> None:
    ast = build_web_research_workflow(
        intent="specialized pae", design_brief=_plan_execute_brief_specialized()
    )
    body = _find_node(ast, "plan-and-execute")["config"]["body"]
    sp = body["config"]["system_prompt"]
    # The discarded-specialization bug: body used to carry the generic builtin.
    assert _GENERIC_MARKER not in sp
    # The lane specialization actually reaches the body.
    assert "resource-grounded research agent" in sp
    assert len(sp) >= 200  # clears _MIN_SYSTEM_PROMPT_CHARS


def test_pae_body_empty_specialization_is_unchanged() -> None:
    # Regression guard: when no lane system_prompt exists (legacy brief), the
    # body keeps the legacy generic path — byte-identical to prior behavior.
    ast = _build()  # _plan_execute_brief() has no lane system_prompt
    body = _find_node(ast, "plan-and-execute")["config"]["body"]
    assert _GENERIC_MARKER in body["config"]["system_prompt"]


def test_pae_body_passes_unspecialized_gate() -> None:
    # Gate-level reproduction of the officeqa failure: detect_unspecialized_agents
    # must emit no blocking advice on the body system_prompt once specialized.
    ast = build_web_research_workflow(
        intent="specialized pae", design_brief=_plan_execute_brief_specialized()
    )
    blocking = [
        e
        for e in detect_unspecialized_agents(ast)
        if getattr(e, "severity", "blocking") == "blocking"
        and "config.body.config.system_prompt" in (e.path or "")
    ]
    assert blocking == [], f"unexpected blocking advice on body: {blocking}"
