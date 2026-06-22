"""Phase 4 — iterative_refinement topology.

iterative_refinement = coordinator -> parallel(evidence lane(s)) -> loop(draft
producer -> coverage critic) -> finalizer. The draft producer is a single
synthesizer (participants=1, self-critique) or parallel(P proposers)->integrator
(participants>=2, multi-model ensemble). The loop exits when the reflector's
coverage_review.decision == "complete" (or max_iterations); each round's critique
is folded into the next via the harness-injected {revision_block_md} channel.

These tests pin: enum/registry parity, rule-0 selection, both participant shapes,
the loop until contract, gate-cleanliness (+ a broken variant), the prompt-var
whitelist (Codex round-1 finding #1 regression), and probe check 11.
"""

from __future__ import annotations

import json
from typing import Any, get_args

import pytest
from databricks_deep_research.workflow.loader import load_workflow_from_dict

from deep_research.agent_designer.blueprint import build_blueprint
from deep_research.agent_designer.designer_types import TopologyKind, WorkflowDesignBrief
from deep_research.agent_designer.probe import run_behavioral_probe
from deep_research.agent_designer.semantic_validation import (
    detect_generic_reflector_prompt,
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
from deep_research.agent_designer.workflow_builder import make_loop

INTENT = "Analyze the competitive dynamics of the cloud database market."

# The harness-supplied template var the draft producer reads for prior-round
# critique; it is whitelisted (workflow_builder._RUNTIME_TEMPLATE_KEYS) so it must
# NOT be coerced into any node's input_keys.
_REVISION_KEY = "revision_block_md"


def _sig(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "asset_signature": "web_only",
        "retrieval_pattern": "open_research",
        "question_class": "open_research",
        "primary_evidence_kind": "web_articles",
        "expected_output_shape": "structured_report",
        "step_dependencies_present": False,
        "independent_workstreams_count": 2,
        "iteration_required": False,
        "output_aggregation_kind": "cross_concern_synthesis",
        "lane_descriptions": ["market structure", "pricing dynamics"],
        "coordination_pattern": "iterative_refinement",
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
    body = (node.get("config") or {}).get("body")
    if isinstance(body, dict):
        out.extend(_all_nodes(body))
    return out


def _agents(ast: dict[str, Any]) -> list[dict[str, Any]]:
    return [n for n in _all_nodes(ast.get("root")) if n.get("type") == "agent"]


def _loop(ast: dict[str, Any]) -> dict[str, Any]:
    loops = [n for n in _all_nodes(ast.get("root")) if n.get("type") == "loop"]
    assert loops, "no loop node in iterative_refinement AST"
    return loops[0]


# --- enum / registry parity ----------------------------------------------------


def test_enum_parity() -> None:
    assert "iterative_refinement" in set(TOPOLOGIES)
    assert "iterative_refinement" in set(get_args(TopologyName))
    assert "iterative_refinement" in set(get_args(TopologyKind))
    assert set(topology_registry()) == set(TOPOLOGIES)


def test_coerce_topology_accepts_iterative_refinement() -> None:
    assert (
        WorkflowDesignBrief(topology="iterative_refinement").topology
        == "iterative_refinement"
    )


def test_select_topology_rule0() -> None:
    assert select_topology(TaskSignature.load_from_storage(_sig())) == "iterative_refinement"
    # rule 0 wins over lane/iteration signals
    sig = TaskSignature.load_from_storage(
        _sig(independent_workstreams_count=4, iteration_required=True)
    )
    assert select_topology(sig) == "iterative_refinement"


def test_select_topology_not_triggered_without_pattern() -> None:
    payload = _sig()
    payload.pop("coordination_pattern")
    assert (
        select_topology(TaskSignature.load_from_storage(payload))
        != "iterative_refinement"
    )


# --- make_loop helper ----------------------------------------------------------


def test_make_loop_shape() -> None:
    node = make_loop(
        node_id="L",
        label="L",
        children=[{"id": "a", "type": "agent", "config": {}, "children": []}],
        until={"type": "state", "key": "x.decision", "operator": "eq", "value": "complete"},
        max_iterations=5,
    )
    assert node["type"] == "loop"
    assert node["config"]["until"]["key"] == "x.decision"
    assert node["config"]["min_iterations"] == 1
    assert node["config"]["max_iterations"] == 5


# --- builder structure ---------------------------------------------------------


def test_builder_structure_self_critique() -> None:
    ast = build_blueprint(_sig(refine_participants=1, refine_max_iterations=2), INTENT)
    loop = _loop(ast)
    body_agents = [c for c in loop["children"] if c.get("type") == "agent"]
    subtypes = [(c.get("config") or {}).get("subtype") for c in body_agents]
    assert subtypes == ["synthesizer", "reflector"]
    # single-voice: no candidates pool, no proposer parallel
    assert "candidates" not in {p["name"] for p in ast["pools"]}
    assert not [c for c in loop["children"] if c.get("type") == "parallel"]
    assert loop["config"]["max_iterations"] == 2


@pytest.mark.parametrize("p", [2, 3, 4])
def test_builder_structure_ensemble(p: int) -> None:
    ast = build_blueprint(_sig(refine_participants=p, refine_max_iterations=4), INTENT)
    loop = _loop(ast)
    parallels = [c for c in loop["children"] if c.get("type") == "parallel"]
    assert parallels, "ensemble loop must contain a proposer parallel"
    proposers = parallels[0]["children"]
    assert len(proposers) == p
    assert all((n.get("config") or {}).get("subtype") == "synthesizer" for n in proposers)
    # unique output_keys among parallel children (workflow.validation)
    okeys = [(n.get("config") or {}).get("output_key") for n in proposers]
    assert len(okeys) == len(set(okeys))
    # proposers write the candidates pool; pool declared with max_items==p
    assert all(
        any(pw.get("pool") == "candidates" for pw in (n.get("config") or {}).get("pool_writes", []))
        for n in proposers
    )
    cand_pool = [pl for pl in ast["pools"] if pl["name"] == "candidates"]
    assert cand_pool and cand_pool[0]["max_items"] == p
    # integrator writes draft_report and injects candidates
    integ = [
        c for c in loop["children"]
        if c.get("type") == "agent"
        and (c.get("config") or {}).get("output_key") == "draft_report"
    ]
    assert integ and any(
        pj.get("pool") == "candidates" for pj in (integ[0]["config"].get("pool_inject") or [])
    )
    # capability diversity: proposers do not all share one tier
    tiers = {(n.get("config") or {}).get("model_tier") for n in proposers}
    assert len(tiers) >= 2


def test_draft_report_key_present_for_both_shapes() -> None:
    # The 'draft_report' output_key is load-bearing: build_revision_block_md keys
    # off it. Both P=1 and P>=2 must produce it inside the loop.
    for p in (1, 3):
        ast = build_blueprint(_sig(refine_participants=p), INTENT)
        loop = _loop(ast)
        producers = [
            n for n in _all_nodes(loop)
            if n.get("type") == "agent"
            and (n.get("config") or {}).get("output_key") == "draft_report"
        ]
        assert producers, f"P={p}: no draft_report producer in loop"


def test_until_wiring_and_critic_not_skip() -> None:
    ast = build_blueprint(_sig(refine_participants=2), INTENT)
    loop = _loop(ast)
    until = loop["config"]["until"]
    assert until["type"] == "state"
    assert until["key"] == "coverage_review.decision"
    assert until["operator"] == "eq"
    assert until["value"] == "complete"
    critics = [
        n for n in loop["children"]
        if (n.get("config") or {}).get("subtype") == "reflector"
    ]
    assert critics
    # the until operand producer must not be error-skip (else runtime hard-raise)
    assert (critics[0].get("error_handling") or {}).get("on_error") != "skip"


def test_iterations_default_and_field_bound() -> None:
    from deep_research.agent_designer.task_signature import SignatureError

    assert _loop(build_blueprint(_sig(), INTENT))["config"]["max_iterations"] == 3
    assert _loop(build_blueprint(_sig(refine_max_iterations=5), INTENT))["config"]["max_iterations"] == 5
    assert _loop(build_blueprint(_sig(refine_max_iterations=1), INTENT))["config"]["max_iterations"] == 1
    # the signature field caps the range (1..6) and fails closed out-of-range —
    # the builder's clamp is a secondary defense for non-signature callers.
    with pytest.raises(SignatureError):
        build_blueprint(_sig(refine_max_iterations=7), INTENT)
    with pytest.raises(SignatureError):
        build_blueprint(_sig(refine_participants=9), INTENT)


# --- loads + validates ---------------------------------------------------------


@pytest.mark.parametrize("p", [1, 2, 4])
def test_loads_and_validates(p: int) -> None:
    # build_blueprint already runs validate_generated_workflow (which loads +
    # validate_workflow, incl. the loop until condition-contract). A clean load
    # here is the explicit gate that the loop AST is framework-valid.
    ast = build_blueprint(_sig(refine_participants=p), INTENT)
    definition = load_workflow_from_dict(ast)
    assert definition.output_keys == ["report"]


# --- prompt-var whitelist (Codex round-1 finding #1 regression) ----------------


@pytest.mark.parametrize("p", [1, 3])
def test_no_unwhitelisted_prompt_var(p: int) -> None:
    # Codex round-1 finding #1: a non-whitelisted prompt var (the original
    # ``{reflection}`` design) gets coerced into input_keys and then rejected at
    # save-time. Every input_key must be produced upstream, a required input, OR a
    # harness-supplied whitelist key. This builder uses the whitelisted
    # ``{revision_block_md}`` critique channel instead of ``{reflection}``.
    from deep_research.agent_designer.workflow_builder import _RUNTIME_TEMPLATE_KEYS

    ast = build_blueprint(_sig(refine_participants=p), INTENT)
    produced = {(n.get("config") or {}).get("output_key") for n in _agents(ast)}
    produced |= set(ast.get("required_inputs") or [])
    allowed = produced | set(_RUNTIME_TEMPLATE_KEYS)
    saw_revision_channel = False
    for node in _agents(ast):
        cfg = node.get("config") or {}
        input_keys = set(cfg.get("input_keys") or [])
        # the original defect var is neither produced nor whitelisted
        assert "reflection" not in input_keys, node.get("id")
        dangling = input_keys - allowed
        assert not dangling, (node.get("id"), dangling)
        if _REVISION_KEY in input_keys:
            saw_revision_channel = True
    # positive: the draft producer wires the whitelisted critique channel
    assert saw_revision_channel, "revision_block_md critique channel not wired"


# --- gates ---------------------------------------------------------------------


@pytest.mark.parametrize("p", [1, 2, 4])
def test_all_gates_pass(p: int) -> None:
    ast = build_blueprint(_sig(refine_participants=p), INTENT)
    assert detect_grounded_research_contract(ast) == []
    assert detect_generic_synthesizer_prompt(ast) == []
    assert detect_generic_reflector_prompt(ast) == []
    assert detect_tool_access_contract(ast) == []


def test_generic_reflector_gate_fails_when_stripped() -> None:
    ast = build_blueprint(_sig(refine_participants=1), INTENT)
    # blank the critic's authored core prompt -> it no longer references domain
    # terms -> the generic-reflector gate must flag it.
    for node in _agents(ast):
        cfg = node.get("config") or {}
        if cfg.get("subtype") == "reflector":
            cfg["system_prompt"] = "You are a reviewer. Return JSON."
    assert detect_generic_reflector_prompt(ast), "stripped reflector should be flagged"


# --- probe check 11 ------------------------------------------------------------


@pytest.mark.parametrize("p", [1, 2, 4])
def test_probe_structure_ok(p: int) -> None:
    sig = _sig(refine_participants=p, refine_max_iterations=4)
    res = run_behavioral_probe(build_blueprint(sig, INTENT), sig)
    assert res.passed
    assert not [g for g in res.gaps if g.startswith("iterative_refinement_")]
    assert any(c.startswith("iterative_refinement_structure_ok") for c in res.conditional_passed)


def test_probe_flags_missing_loop() -> None:
    sig = _sig(refine_participants=2)
    ast = build_blueprint(sig, INTENT)
    # drop the loop node -> probe must flag the missing loop
    ast["root"]["children"] = [c for c in ast["root"]["children"] if c.get("type") != "loop"]
    res = run_behavioral_probe(ast, sig)
    assert any("iterative_refinement_missing_loop" in g for g in res.gaps)


# --- FW-1: per-proposer model family ------------------------------------------


def test_no_model_family_without_catalog() -> None:
    # default app.yaml has no model_families catalog → proposers carry NO
    # model_family (they diversify by tier + stance only).
    ast = build_blueprint(_sig(refine_participants=3), INTENT)
    proposers = [n for n in _agents(ast) if str(n.get("id", "")).startswith("proposer-")]
    assert proposers
    assert all("model_family" not in (n.get("config") or {}) for n in proposers)


def test_proposers_pinned_to_families_when_resolved(monkeypatch: Any) -> None:
    # When the resolver yields families (simulating a configured catalog), the
    # builder pins each proposer's model_family — and the AST still validates
    # (model_family is an accepted AgentNodeConfig field threaded through save).
    import deep_research.agent_designer.workflow_builder as wb

    monkeypatch.setattr(
        wb,
        "_resolve_proposer_families",
        lambda requested, count: [["claude", "llama"][i % 2] for i in range(count)],
    )
    ast = build_blueprint(_sig(refine_participants=3), INTENT)
    proposers = [n for n in _agents(ast) if str(n.get("id", "")).startswith("proposer-")]
    families = [(n.get("config") or {}).get("model_family") for n in proposers]
    assert families == ["claude", "llama", "claude"]


def test_resolve_proposer_families_drops_unknown(monkeypatch: Any) -> None:
    import deep_research.agent_designer.workflow_builder as wb

    class _Cfg:
        model_families = {"claude": ["x"], "llama": ["y"]}

    monkeypatch.setattr(
        "deep_research.core.app_config.get_app_config", lambda: _Cfg()
    )
    # requested families validated against the catalog; unknown ('gpt') dropped
    assert wb._resolve_proposer_families(["llama", "gpt"], 2) == ["llama", "llama"]
    # nothing requested → cycle the full catalog
    assert wb._resolve_proposer_families(None, 3) == ["claude", "llama", "claude"]


# --- determinism ---------------------------------------------------------------


@pytest.mark.parametrize("p", [1, 3])
def test_build_blueprint_deterministic(p: int) -> None:
    first = build_blueprint(_sig(refine_participants=p), INTENT)
    second = build_blueprint(_sig(refine_participants=p), INTENT)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["structural_fingerprint"] == second["structural_fingerprint"]
