"""Phase 2 — best_of_n topology.

Covers the full vertical: enum parity / drift guard, ``select_topology`` rule 0,
the $ref-free classifier schema, candidate-count bounds, the builder's structure
across N, gate-cleanliness (the §2.2 proof) plus fail-with-message variants, the
probe family-map + invariants, and the real ``build_blueprint`` deterministic
entry point.

best_of_n = coordinator -> parallel(evidence lane(s)) -> parallel(N candidate
synthesizers) -> complex-tier judge. Generic across tasks: nothing here asserts a
specific domain or benchmark answer.
"""

from __future__ import annotations

import json
from typing import Any, get_args

import pytest

from deep_research.agent_designer.blueprint import (
    PLACEHOLDER_PENDING_KEY,
    build_blueprint,
)
from deep_research.agent_designer.designer_types import (
    LaneSpec,
    TopologyKind,
    WorkflowDesignBrief,
)
from deep_research.agent_designer.probe import run_behavioral_probe
from deep_research.agent_designer.semantic_validation import (
    detect_generic_synthesizer_prompt,
    detect_grounded_research_contract,
    detect_unspecialized_agents,
)
from deep_research.agent_designer.structural_gate import detect_tool_access_contract
from deep_research.agent_designer.task_signature import (
    TOPOLOGIES,
    TaskSignature,
    TopologyName,
    select_topology,
)
from deep_research.agent_designer.workflow_builder import build_web_research_workflow

INTENT = (
    "Compare the cloud revenue growth and operating margin of Amazon AWS and "
    "Microsoft Azure over the last four fiscal quarters, and assess which has "
    "the stronger competitive position."
)

# Contract-compliant evidence lane (5 sub-questions, 3 output sections, search
# strategy, unknowns clause) so a built AST can clear the BLOCKING researcher
# user_prompt_template gate the way an architect-authored lane would.
LANE_UPT = (
    "You are investigating: **{query}**\n\n"
    "## Sub-questions\n"
    "1. What was AWS cloud revenue each of the last four fiscal quarters?\n"
    "2. What was Azure cloud revenue each of the last four fiscal quarters?\n"
    "3. What operating margin did each segment report per quarter?\n"
    "4. How did year-over-year growth rates compare across the two?\n"
    "5. What competitive-position signals appear in the filings?\n\n"
    "## Required output structure\n"
    "- Revenue table: per-quarter AWS vs Azure figures with citations.\n"
    "- Margin comparison: per-quarter operating margin with citations.\n"
    "- Competitive assessment grounded in the gathered evidence.\n\n"
    "## Search strategy\n"
    "- Query each company's quarterly earnings releases and 10-Q filings.\n"
    "- Prefer primary filings over secondary coverage; refine by fiscal period.\n\n"
    "## Definition of done\n"
    "Mark any unavailable figure 'Data unavailable' — never improvise."
)


def _brief() -> WorkflowDesignBrief:
    return WorkflowDesignBrief(
        workflow_name="aws_vs_azure_cloud",
        workflow_description=INTENT,
        user_goal=INTENT,
        required_outputs=[
            "cloud revenue comparison table",
            "operating margin comparison",
            "competitive assessment",
        ],
        research_lanes=[
            LaneSpec(
                description="AWS vs Azure cloud revenue and margin evidence",
                system_prompt=(
                    "Gather per-quarter cloud revenue and operating margin for "
                    "AWS and Azure from primary filings."
                ),
                user_prompt_template=LANE_UPT,
            )
        ],
    )


def _sig(count: int | None = 8, **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "asset_signature": "web_only",
        "retrieval_pattern": "open_research",
        "question_class": "comparative_analysis",
        "primary_evidence_kind": "web_articles",
        "expected_output_shape": "structured_report",
        "step_dependencies_present": False,
        "independent_workstreams_count": 1,
        "iteration_required": False,
        "output_aggregation_kind": "single_answer",
        "lane_descriptions": ["AWS vs Azure cloud revenue and margin evidence"],
        "coordination_pattern": "best_of_n",
    }
    if count is not None:
        payload["coordination_candidate_count"] = count
    payload.update(overrides)
    return payload


def _build(count: int | None = 8, **sig_overrides: Any) -> dict[str, Any]:
    return build_web_research_workflow(
        INTENT, design_brief=_brief(), task_signature=_sig(count, **sig_overrides)
    )


# --- enum parity / drift guard --------------------------------------------------


def test_topology_enum_parity() -> None:
    assert set(get_args(TopologyName)) == set(TOPOLOGIES)
    assert set(get_args(TopologyKind)) == set(TOPOLOGIES)
    assert "best_of_n" in set(TOPOLOGIES)


def test_coerce_topology_raises_on_unknown() -> None:
    assert WorkflowDesignBrief(topology="best_of_n").topology == "best_of_n"
    assert WorkflowDesignBrief().topology == "parallel_lanes"  # omitted -> default
    with pytest.raises(Exception):
        WorkflowDesignBrief(topology="totally_unknown_topology")


# --- selection ------------------------------------------------------------------


def test_select_topology_rule0() -> None:
    assert select_topology(TaskSignature.load_from_storage(_sig(8))) == "best_of_n"


def test_rule0_wins_over_structural_axes() -> None:
    sig = TaskSignature.load_from_storage(
        _sig(4, independent_workstreams_count=3, lane_descriptions=["a", "b", "c"])
    )
    assert select_topology(sig) == "best_of_n"


def test_no_coordination_pattern_unchanged() -> None:
    payload = _sig(8)
    payload.pop("coordination_pattern")
    assert select_topology(TaskSignature.load_from_storage(payload)) != "best_of_n"


# --- classifier schema / bounds -------------------------------------------------


def test_tool_schema_has_no_ref() -> None:
    schema = TaskSignature.tool_schema()
    blob = json.dumps(schema)
    assert "$ref" not in blob and "$defs" not in blob
    for field in (
        "coordination_pattern",
        "coordination_candidate_count",
        "coordination_judge_tier",
    ):
        assert field in schema["properties"]
        assert field not in set(schema.get("required", []))


@pytest.mark.parametrize("bad", [1, 11, 0, -1])
def test_candidate_count_bounds(bad: int) -> None:
    with pytest.raises(Exception):
        TaskSignature.load_from_storage(_sig(bad))


# --- builder structure ----------------------------------------------------------


@pytest.mark.parametrize("n", [2, 4, 8, 10])
def test_builder_structure(n: int) -> None:
    ast = _build(n)
    root = ast["root"]
    assert [c["type"] for c in root["children"]] == [
        "agent",
        "parallel",
        "parallel",
        "agent",
    ]
    coordinator, evidence_par, cand_par, judge = root["children"]
    assert coordinator["config"]["subtype"] == "coordinator"
    assert evidence_par["id"] == "evidence-lanes"
    assert cand_par["id"] == "candidate-generators"
    cands = cand_par["children"]
    assert len(cands) == n
    keys = [c["config"]["output_key"] for c in cands]
    assert keys == [f"candidate_{i}" for i in range(1, n + 1)]
    assert len(set(keys)) == n  # parallel requires unique output_keys
    for cand in cands:
        assert cand["config"]["subtype"] == "synthesizer"
        assert {"observations", "sources"} <= {
            p["pool"] for p in cand["config"]["pool_inject"]
        }
        assert any(
            p["pool"] == "candidates" for p in cand["config"].get("pool_writes", [])
        )
    # Diversity: distinct framing => distinct system prompts (8 stances cycled).
    assert len({c["config"]["system_prompt"] for c in cands}) == min(n, 8)
    assert judge["id"] == "judge"
    assert judge["config"]["subtype"] == "synthesizer"
    assert judge["config"]["model_tier"] == "complex"
    assert judge["config"]["output_key"] == "report"
    assert {"observations", "sources", "candidates"} <= {
        p["pool"] for p in judge["config"]["pool_inject"]
    }
    assert ast["output_keys"] == ["report"]
    assert {p["name"] for p in ast["pools"]} == {
        "sources",
        "observations",
        "candidates",
    }
    cand_pool = next(p for p in ast["pools"] if p["name"] == "candidates")
    assert cand_pool["max_items"] == n


def test_default_candidate_count() -> None:
    assert len(_build(None)["root"]["children"][2]["children"]) == 4


def test_judge_tier_override() -> None:
    ast = _build(4, coordination_judge_tier="analytical")
    assert ast["root"]["children"][3]["config"]["model_tier"] == "analytical"


# --- gates (the §2.2 proof) -----------------------------------------------------


def test_gates_pass_unmodified() -> None:
    ast = _build(8)
    assert detect_tool_access_contract(ast) == []
    assert detect_grounded_research_contract(ast) == []
    assert detect_generic_synthesizer_prompt(ast) == []
    blocking = [
        e
        for e in detect_unspecialized_agents(ast)
        if getattr(e, "severity", "blocking") == "blocking"
    ]
    assert blocking == []


def test_grounding_gate_fails_when_candidate_ungrounded() -> None:
    ast = _build(4)
    ast["root"]["children"][2]["children"][0]["config"]["pool_inject"] = []
    assert detect_grounded_research_contract(ast), "expected a grounding-contract error"


# --- probe ----------------------------------------------------------------------


def test_probe_no_false_mismatch_and_invariants() -> None:
    ast = _build(8)
    res = run_behavioral_probe(ast, task_signature=_sig(8))
    assert not any("topology_signature_mismatch" in g for g in res.gaps), res.gaps
    assert any(c.startswith("best_of_n_structure_ok") for c in res.conditional_passed)
    assert res.passed, res.gaps


def test_probe_flags_missing_candidates_pool() -> None:
    ast = _build(8)
    ast["pools"] = [p for p in ast["pools"] if p["name"] != "candidates"]
    res = run_behavioral_probe(ast, task_signature=_sig(8))
    assert any("best_of_n_missing_candidates_pool" in g for g in res.gaps)
    assert not res.passed


# --- blueprint (the real deterministic entry) -----------------------------------


def test_build_blueprint_routes_best_of_n() -> None:
    ast = build_blueprint(_sig(8), INTENT)
    cand_par = ast["root"]["children"][2]
    assert cand_par["id"] == "candidate-generators"
    assert len(cand_par["children"]) == 8
    # Only the evidence researcher(s) are placeholder-pending; synthesizers are not.
    pending = ast.get(PLACEHOLDER_PENDING_KEY) or []
    assert pending and all("researcher" in p for p in pending), pending
    assert not any(p.startswith("candidate-") or p == "judge" for p in pending)
    # Fingerprint is stable across rebuilds (immutability anchor).
    assert ast["structural_fingerprint"] == build_blueprint(_sig(8), INTENT)[
        "structural_fingerprint"
    ]
