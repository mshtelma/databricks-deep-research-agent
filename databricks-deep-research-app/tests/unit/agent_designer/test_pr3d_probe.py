"""PR3-D Layer 3a — synthetic behavioral_probe tests."""

from __future__ import annotations

import json
from typing import Any

import pytest
from databricks_deep_research.tools.protocol import ToolContext

from deep_research.agent_designer.framework_tools import (
    BehavioralProbeTool,
    builtin_designer_tools,
)
from deep_research.agent_designer.probe import (
    ProbeResult,
    run_behavioral_probe,
)


@pytest.fixture
def ctx() -> ToolContext:
    return ToolContext(query="")


def _signature_pipelined_period_basis() -> dict[str, Any]:
    return {
        "asset_signature": "corpus_only",
        "retrieval_pattern": "pipelined_retrieve_read_compute",
        "question_class": "numeric_aggregation",
        "question_ambiguity": ["period_basis"],
        "primary_evidence_kind": "structured_tables",
        "expected_output_shape": "single_number",
    }


def _signature_independent_lanes() -> dict[str, Any]:
    return {
        "asset_signature": "web_only",
        "retrieval_pattern": "independent_lanes",
        "question_class": "comparative_analysis",
        "question_ambiguity": [],
        "primary_evidence_kind": "web_articles",
        "expected_output_shape": "structured_report",
    }


def _conformant_pipelined_ast() -> dict[str, Any]:
    """A plan_and_execute AST that satisfies the pipelined+period_basis
    signature: monthly+annual tools bound, FY+CY in prompts, plan_and_execute
    topology."""
    return {
        "id": "wf",
        "name": "wf",
        "version": 1,
        "required_inputs": ["query"],
        "output_keys": ["final"],
        "tools": [
            {"name": "vs", "kind": "vector_search"},
            {"name": "dt", "kind": "table_read"},
            {"name": "comp", "kind": "compute"},
        ],
        "pools": [{"name": "sources", "dedup_key": "chunk_id"}],
        "root": {
            "id": "root",
            "type": "sequence",
            "label": "Root",
            "config": {},
            "children": [
                {
                    "id": "pae",
                    "type": "plan_and_execute",
                    "label": "Plan-and-execute",
                    "config": {
                        "planner": {"system_prompt": "Plan."},
                        "body": {
                            "id": "lane_1",
                            "type": "agent",
                            "label": "Lane 1",
                            "config": {
                                "subtype": "researcher",
                                "model_tier": "analytical",
                                "tools": ["vs", "dt", "comp"],
                                "user_prompt_template": (
                                    "Investigate {query}. Cover both fiscal year and calendar year."
                                ),
                            },
                        },
                    },
                },
                {
                    "id": "synthesizer",
                    "type": "agent",
                    "label": "Synth",
                    "config": {
                        "subtype": "synthesizer",
                        "pool_inject": ["sources", "observations"],
                    },
                },
            ],
        },
    }


def _wrong_topology_ast() -> dict[str, Any]:
    """A parallel_lanes AST under a pipelined signature — topology should
    fail the probe."""
    return {
        "id": "wf",
        "name": "wf",
        "version": 1,
        "required_inputs": ["query"],
        "output_keys": ["final"],
        "tools": [
            {"name": "vs", "kind": "vector_search"},
            {"name": "dt", "kind": "table_read"},
            {"name": "comp", "kind": "compute"},
        ],
        "pools": [{"name": "sources", "dedup_key": "chunk_id"}],
        "root": {
            "id": "root",
            "type": "sequence",
            "label": "Root",
            "config": {},
            "children": [
                {
                    "id": "lanes",
                    "type": "parallel",
                    "label": "Parallel",
                    "config": {},
                    "children": [
                        {
                            "id": "lane_1",
                            "type": "agent",
                            "label": "L1",
                            "config": {
                                "subtype": "researcher",
                                "tools": ["vs", "dt", "comp"],
                                "user_prompt_template": (
                                    "Investigate {query}. Cover both fiscal year and calendar year."
                                ),
                            },
                        }
                    ],
                },
                {
                    "id": "synthesizer",
                    "type": "agent",
                    "label": "Synth",
                    "config": {
                        "subtype": "synthesizer",
                        "pool_inject": ["sources"],
                    },
                },
            ],
        },
    }


def _missing_period_basis_ast() -> dict[str, Any]:
    """A pipelined AST whose lane prompt mentions ONLY FY — period_basis
    diversity gap."""
    ast = _conformant_pipelined_ast()
    body = ast["root"]["children"][0]["config"]["body"]
    body["config"]["user_prompt_template"] = (
        "Investigate {query}. Focus on fiscal year totals only."
    )
    return ast


def _bad_kind_ast() -> dict[str, Any]:
    """AST with a tool whose kind is not in ToolKind — invariant fail."""
    ast = _conformant_pipelined_ast()
    ast["tools"].append({"name": "bogus", "kind": "totally_made_up_kind"})
    return ast


def _lane_missing_tools_ast() -> dict[str, Any]:
    """AST whose lane has zero declared tools — invariant fail."""
    ast = _conformant_pipelined_ast()
    body = ast["root"]["children"][0]["config"]["body"]
    body["config"]["tools"] = []
    return ast


# ---------------------------------------------------------------------------
# Conformant-workflow path: probe passes
# ---------------------------------------------------------------------------


def test_probe_clears_conformant_pipelined_workflow() -> None:
    result = run_behavioral_probe(
        _conformant_pipelined_ast(),
        _signature_pipelined_period_basis(),
    )
    assert result.passed, f"unexpected gaps: {result.gaps}"
    assert "all_tool_kinds_registered" in result.invariants_passed
    assert "every_lane_has_bound_tools" in result.invariants_passed
    assert "every_lane_has_query_anchor" in result.invariants_passed
    assert "synthesizer_reads_pools" in result.invariants_passed
    assert any(x.startswith("topology_matches_signature") for x in result.conditional_passed)
    assert "period_basis_in_lane_prompts" in result.conditional_passed
    assert "numeric_aggregation_has_compute_and_table_read" in result.conditional_passed
    assert "structured_tables_has_table_read" in result.conditional_passed


# ---------------------------------------------------------------------------
# Gap detection: each pathological AST surfaces the right gap
# ---------------------------------------------------------------------------


def test_probe_catches_topology_signature_mismatch() -> None:
    result = run_behavioral_probe(
        _wrong_topology_ast(),
        _signature_pipelined_period_basis(),
    )
    assert not result.passed
    assert any("topology_signature_mismatch" in g for g in result.gaps)


def test_probe_catches_period_basis_gap() -> None:
    result = run_behavioral_probe(
        _missing_period_basis_ast(),
        _signature_pipelined_period_basis(),
    )
    assert not result.passed
    assert any("period_basis_query_diversity_gap" in g for g in result.gaps)
    assert any("missing=calendar_year" in g for g in result.gaps)


def test_probe_catches_unknown_tool_kind() -> None:
    result = run_behavioral_probe(
        _bad_kind_ast(),
        _signature_pipelined_period_basis(),
    )
    assert not result.passed
    assert any("unknown_tool_kinds" in g for g in result.gaps)


def test_probe_catches_lane_without_tools() -> None:
    result = run_behavioral_probe(
        _lane_missing_tools_ast(),
        _signature_pipelined_period_basis(),
    )
    assert not result.passed
    assert any("lanes_without_bound_tools" in g for g in result.gaps)


def test_probe_runs_without_signature() -> None:
    # Signature-independent invariants still run when no signature is
    # supplied; the conformant AST should still pass.
    result = run_behavioral_probe(_conformant_pipelined_ast(), None)
    assert result.passed
    assert "all_tool_kinds_registered" in result.invariants_passed
    # No conditional checks fired without a signature.
    assert result.conditional_passed == []


# ---------------------------------------------------------------------------
# Runtime-query check (opt-in)
# ---------------------------------------------------------------------------


def test_probe_runtime_queries_satisfy_period_basis() -> None:
    queries = [
        "army expenditures fiscal year 1945",
        "army monthly expenditures calendar year totals 1945",
    ]
    result = run_behavioral_probe(
        _conformant_pipelined_ast(),
        _signature_pipelined_period_basis(),
        runtime_queries=queries,
    )
    assert result.passed
    assert "runtime_query_axis_satisfied:period_basis" in result.conditional_passed
    assert result.runtime_queries == queries


def test_probe_runtime_queries_missing_calendar_year() -> None:
    queries = [
        "army expenditures fiscal year 1945",
        "army FY1945 totals",
    ]
    result = run_behavioral_probe(
        _conformant_pipelined_ast(),
        _signature_pipelined_period_basis(),
        runtime_queries=queries,
    )
    assert not result.passed
    assert any("runtime_query_axis_unsatisfied:period_basis" in g for g in result.gaps)


# ---------------------------------------------------------------------------
# Framework tool — registered and produces ProbeResult dict
# ---------------------------------------------------------------------------


async def test_behavioral_probe_tool_emits_probe_result(
    ctx: ToolContext,
) -> None:
    ast = _conformant_pipelined_ast()
    sig = _signature_pipelined_period_basis()
    tool = BehavioralProbeTool(
        state_getter=lambda: ast,
        signature_getter=lambda: sig,
    )
    result = await tool.execute({}, ctx)
    body = json.loads(result.content)
    assert body["passed"] is True
    assert "gaps" in body
    assert "invariants_passed" in body
    assert "conditional_passed" in body


def test_builtin_designer_tools_registers_behavioral_probe() -> None:
    names = {t.definition.name for t in builtin_designer_tools()}
    assert "behavioral_probe" in names


def test_probe_result_serialization_round_trip() -> None:
    pr = ProbeResult(
        passed=True,
        invariants_passed=["a"],
        conditional_passed=["b"],
        runtime_queries=["q1"],
    )
    d = pr.to_dict()
    assert d["passed"] is True
    assert d["gaps"] == []
    assert d["invariants_passed"] == ["a"]
    assert d["conditional_passed"] == ["b"]
    assert d["runtime_queries"] == ["q1"]


# ---------------------------------------------------------------------------
# Plan v2.2 — asset_signature ↔ tool_kinds invariant
#
# Catches the silent-fallback failure mode where the deterministic blueprint
# binds only public-web tools to lanes despite the classifier inferring
# ``corpus_only``/``structured_only`` from user_intent.
# ---------------------------------------------------------------------------


def _corpus_only_web_research_only_ast() -> dict[str, Any]:
    """corpus_only signature but every researcher binds only web_research.

    Models the exact failure mode observed in workflow 26b036b1-c2e: the
    classifier said ``corpus_only`` from free-text "use main.x.y vector
    search index", grounding stage didn't resolve any asset, and the
    deterministic blueprint silently defaulted to ``web_research``.
    """
    return {
        "id": "wf",
        "name": "wf",
        "version": 1,
        "required_inputs": ["query"],
        "output_keys": ["final"],
        "tools": [
            {"name": "web_research", "kind": "web_research"},
        ],
        "pools": [{"name": "sources", "dedup_key": "url"}],
        "root": {
            "id": "root",
            "type": "sequence",
            "label": "Root",
            "config": {},
            "children": [
                {
                    "id": "pae",
                    "type": "plan_and_execute",
                    "label": "Plan-and-execute",
                    "config": {
                        "planner": {"system_prompt": "Plan."},
                        "body": {
                            "id": "lane_1",
                            "type": "agent",
                            "label": "Lane 1",
                            "config": {
                                "subtype": "researcher",
                                "model_tier": "analytical",
                                "tools": ["web_research"],
                                "user_prompt_template": "Research {query}.",
                            },
                        },
                    },
                },
                {
                    "id": "synthesizer",
                    "type": "agent",
                    "label": "Synth",
                    "config": {
                        "subtype": "synthesizer",
                        "pool_inject": ["sources"],
                    },
                },
            ],
        },
    }


def test_probe_catches_corpus_only_signature_with_only_web_tools() -> None:
    """corpus_only + only web tools → asset_signature_tool_kind_mismatch gap."""
    sig = _signature_pipelined_period_basis()  # asset_signature == corpus_only
    result = run_behavioral_probe(
        _corpus_only_web_research_only_ast(),
        sig,
    )
    assert not result.passed
    assert any(g.startswith("asset_signature_tool_kind_mismatch") for g in result.gaps), (
        f"expected mismatch gap, got: {result.gaps}"
    )


def test_probe_passes_corpus_only_with_corpus_tools_bound() -> None:
    """corpus_only with vector_search bound to the lane → invariant satisfied."""
    sig = _signature_pipelined_period_basis()
    result = run_behavioral_probe(
        _conformant_pipelined_ast(),
        sig,
    )
    assert any(
        c.startswith("asset_signature_matches_tool_kinds:corpus_only")
        for c in result.conditional_passed
    )


def test_probe_skips_asset_signature_invariant_for_web_only() -> None:
    """web_only signature legitimately has only web tools — no gap."""
    web_only_sig = _signature_independent_lanes()  # web_only
    # Build a minimal web_only AST with one lane.
    ast: dict[str, Any] = {
        "id": "wf",
        "name": "wf",
        "version": 1,
        "required_inputs": ["query"],
        "output_keys": ["final"],
        "tools": [{"name": "web_research", "kind": "web_research"}],
        "pools": [{"name": "sources", "dedup_key": "url"}],
        "root": {
            "id": "root",
            "type": "sequence",
            "label": "Root",
            "config": {},
            "children": [
                {
                    "id": "lanes",
                    "type": "parallel",
                    "label": "P",
                    "config": {},
                    "children": [
                        {
                            "id": "lane_1",
                            "type": "agent",
                            "label": "L1",
                            "config": {
                                "subtype": "researcher",
                                "tools": ["web_research"],
                                "user_prompt_template": "Research {query}.",
                            },
                        },
                    ],
                },
                {
                    "id": "synthesizer",
                    "type": "agent",
                    "label": "Synth",
                    "config": {
                        "subtype": "synthesizer",
                        "pool_inject": ["sources"],
                    },
                },
            ],
        },
    }
    result = run_behavioral_probe(ast, web_only_sig)
    # No asset_signature gap for web_only.
    assert not any(g.startswith("asset_signature_tool_kind_mismatch") for g in result.gaps)
    # And no asset_signature conditional pass entry (only applies to
    # corpus_only/structured_only).
    assert not any(
        c.startswith("asset_signature_matches_tool_kinds") for c in result.conditional_passed
    )
