"""Phase 5 — router topology.

router = classifier (router_classifier subtype, typed route discriminator) ->
conditional(branch_1..branch_M, default). Each branch is a self-contained evidence
front + grounded synthesizer writing the SAME ``report`` output_key; branch node
ids are namespaced by a route slug to satisfy the global unique-node-id rule.

These tests pin: enum/registry parity, rule-0 selection, builder structure +
namespaced ids, the typed discriminator (FW-2), gate-cleanliness (+ broken
variant), probe check 13 + topology-inference precedence (a branch's evidence
parallel must NOT make the prober return parallel_lanes), and determinism.
"""

from __future__ import annotations

import json
from typing import Any, get_args

from databricks_deep_research.workflow.loader import load_workflow_from_dict

from deep_research.agent_designer.blueprint import build_blueprint
from deep_research.agent_designer.designer_types import TopologyKind, WorkflowDesignBrief
from deep_research.agent_designer.probe import _topology_of_ast, run_behavioral_probe
from deep_research.agent_designer.semantic_validation import (
    detect_generic_synthesizer_prompt,
    detect_grounded_research_contract,
)
from deep_research.agent_designer.structural_gate import detect_tool_access_contract
from deep_research.agent_designer.task_signature import (
    TOPOLOGIES,
    TaskSignature,
    TopologyName,
    select_topology,
)
from deep_research.agent_designer.topology_registry import topology_registry

INTENT = "Answer cloud database questions across pricing, performance, and migration."
CASES = ["pricing questions", "performance deep-dives", "migration help"]


def _sig(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "asset_signature": "web_only",
        "retrieval_pattern": "open_research",
        "question_class": "open_research",
        "primary_evidence_kind": "web_articles",
        "expected_output_shape": "structured_report",
        "step_dependencies_present": False,
        "independent_workstreams_count": 1,
        "iteration_required": False,
        "output_aggregation_kind": "single_answer",
        "lane_descriptions": ["the primary concern"],
        "coordination_pattern": "router",
        "router_cases": list(CASES),
    }
    base.update(kw)
    return base


def _all_nodes(node: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(node, dict):
        return out
    out.append(node)
    for child in node.get("children") or []:
        out.extend(_all_nodes(child))
    return out


def _agents(ast: dict[str, Any]) -> list[dict[str, Any]]:
    return [n for n in _all_nodes(ast.get("root")) if n.get("type") == "agent"]


def _conditional(ast: dict[str, Any]) -> dict[str, Any]:
    conds = [n for n in _all_nodes(ast.get("root")) if n.get("type") == "conditional"]
    assert conds, "router AST has no conditional"
    return conds[0]


# --- enum / registry parity ----------------------------------------------------


def test_enum_parity() -> None:
    assert "router" in set(TOPOLOGIES)
    assert "router" in set(get_args(TopologyName))
    assert "router" in set(get_args(TopologyKind))
    assert set(topology_registry()) == set(TOPOLOGIES)


def test_coerce_topology_accepts_router() -> None:
    assert WorkflowDesignBrief(topology="router").topology == "router"


def test_select_topology_rule0() -> None:
    assert select_topology(TaskSignature.load_from_storage(_sig())) == "router"


# --- builder structure ---------------------------------------------------------


def test_builder_structure() -> None:
    ast = build_blueprint(_sig(), INTENT)
    root_children = ast["root"]["children"]
    assert [c["type"] for c in root_children] == ["agent", "conditional"]
    classifier = root_children[0]
    assert classifier["config"]["subtype"] == "router_classifier"
    assert classifier["config"]["output_key"] == "routing"
    enum = classifier["config"]["output_schema"]["properties"]["route"]["enum"]
    assert enum == CASES
    cond = _conditional(ast)
    # M branches, M-1 conditions, default = last branch (the fallback case)
    assert len(cond["children"]) == len(CASES)
    assert len(cond["config"]["conditions"]) == len(CASES) - 1
    assert cond["config"]["default_branch"] == len(CASES) - 1
    # conditions discriminate on routing.route against the leading cases
    cond_values = [c["value"] for c in cond["config"]["conditions"]]
    assert cond_values == CASES[:-1]
    assert all(c["key"] == "routing.route" for c in cond["config"]["conditions"])
    # every branch terminal synth writes the shared report output_key
    for branch in cond["children"]:
        synths = [
            n
            for n in _all_nodes(branch)
            if n.get("type") == "agent"
            and (n.get("config") or {}).get("subtype") == "synthesizer"
        ]
        assert synths and any(
            (s.get("config") or {}).get("output_key") == "report" for s in synths
        )


def test_branch_node_ids_unique_even_for_similar_cases() -> None:
    # near-identical case labels must still produce unique node ids (slug + index)
    ast = build_blueprint(_sig(router_cases=["pricing", "pricing", "other"]), INTENT)
    ids = [n["id"] for n in _all_nodes(ast.get("root")) if "id" in n]
    assert len(ids) == len(set(ids)), [i for i in ids if ids.count(i) > 1]


def test_defaults_when_too_few_cases() -> None:
    # < 2 cases falls back to a valid 2-branch default rather than hard-failing
    ast = build_blueprint(_sig(router_cases=["only one"]), INTENT)
    assert len(_conditional(ast)["children"]) == 2


def test_loads_and_validates() -> None:
    ast = build_blueprint(_sig(), INTENT)
    definition = load_workflow_from_dict(ast)
    assert definition.output_keys == ["report"]


# --- gates ---------------------------------------------------------------------


def test_gates_pass() -> None:
    ast = build_blueprint(_sig(), INTENT)
    assert detect_grounded_research_contract(ast) == []
    assert detect_generic_synthesizer_prompt(ast) == []
    assert detect_tool_access_contract(ast) == []


def test_generic_synth_gate_fails_when_branch_stripped() -> None:
    ast = build_blueprint(_sig(), INTENT)
    # blank one branch synthesizer's core prompt -> generic-synth gate flags it
    for node in _agents(ast):
        cfg = node.get("config") or {}
        if cfg.get("subtype") == "synthesizer":
            cfg["system_prompt"] = "Write a report."
            break
    assert detect_generic_synthesizer_prompt(ast)


# --- probe check 13 + inference precedence -------------------------------------


def test_topology_inference_is_router_not_parallel_lanes() -> None:
    # a branch's evidence parallel must NOT make the prober return parallel_lanes
    ast = build_blueprint(_sig(), INTENT)
    assert _topology_of_ast(ast) == "router"


def test_probe_structure_ok() -> None:
    sig = _sig()
    res = run_behavioral_probe(build_blueprint(sig, INTENT), sig)
    assert res.passed
    assert not [g for g in res.gaps if g.startswith("router_")]
    assert any(c.startswith("router_structure_ok") for c in res.conditional_passed)


def test_probe_flags_missing_classifier() -> None:
    sig = _sig()
    ast = build_blueprint(sig, INTENT)
    # demote the classifier subtype -> probe must flag the missing typed classifier
    for node in _agents(ast):
        if (node.get("config") or {}).get("subtype") == "router_classifier":
            node["config"]["subtype"] = "coordinator"
    res = run_behavioral_probe(ast, sig)
    assert any("router_no_classifier" in g for g in res.gaps)


# --- determinism ---------------------------------------------------------------


def test_build_blueprint_deterministic() -> None:
    first = build_blueprint(_sig(), INTENT)
    second = build_blueprint(_sig(), INTENT)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["structural_fingerprint"] == second["structural_fingerprint"]
