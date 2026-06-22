"""build_blueprint persists the producing TaskSignature as AST metadata.

This is the foundation of the topology EDIT lane: a later edit retrieves the
real signature (``stored_signature``) instead of re-inferring it from arbitrary
AST shape. The stamp MUST NOT perturb the structural fingerprint (which the
PR-3 immutability check enforces), so we assert the recomputed fingerprint of
the stamped AST still equals the stored one.
"""

from __future__ import annotations

from typing import Any

from deep_research.agent_designer.blueprint import (
    build_blueprint,
    compute_structural_fingerprint,
)


def _parallel_sig() -> dict[str, Any]:
    return {
        "asset_signature": "web_only",
        "retrieval_pattern": "independent_lanes",
        "question_class": "open_research",
        "primary_evidence_kind": "web_articles",
        "expected_output_shape": "structured_report",
        "independent_workstreams_count": 3,
        "step_dependencies_present": False,
        "iteration_required": False,
        "output_aggregation_kind": "cross_concern_synthesis",
        "lane_descriptions": ["alpha", "beta", "gamma"],
    }


def _best_of_n_sig(count: int) -> dict[str, Any]:
    return {
        "asset_signature": "web_only",
        "retrieval_pattern": "open_research",
        "question_class": "comparative_analysis",
        "primary_evidence_kind": "web_articles",
        "expected_output_shape": "structured_report",
        "independent_workstreams_count": 1,
        "step_dependencies_present": False,
        "iteration_required": False,
        "output_aggregation_kind": "single_answer",
        "lane_descriptions": ["evidence"],
        "coordination_pattern": "best_of_n",
        "coordination_candidate_count": count,
    }


def test_build_blueprint_stamps_signature_parallel() -> None:
    ast = build_blueprint(_parallel_sig(), "compare three things", [])
    sig = ast.get("designer_signature")
    assert isinstance(sig, dict)
    assert sig["independent_workstreams_count"] == 3
    assert sig["asset_signature"] == "web_only"
    assert sig["lane_descriptions"] == ["alpha", "beta", "gamma"]


def test_build_blueprint_stamps_signature_best_of_n() -> None:
    ast = build_blueprint(_best_of_n_sig(6), "best of six", [])
    sig = ast.get("designer_signature")
    assert isinstance(sig, dict)
    assert sig["coordination_pattern"] == "best_of_n"
    assert sig["coordination_candidate_count"] == 6


def test_signature_stamp_does_not_perturb_fingerprint() -> None:
    # The stored fingerprint was computed BEFORE the signature was stamped.
    # Recomputing it on the fully-stamped AST must still match — proving the
    # metadata key is invisible to the structural projection and the PR-3
    # immutability check stays intact.
    for sig in (_parallel_sig(), _best_of_n_sig(4)):
        ast = build_blueprint(sig, "q", [])
        assert "designer_signature" in ast
        assert compute_structural_fingerprint(ast) == ast["structural_fingerprint"]
