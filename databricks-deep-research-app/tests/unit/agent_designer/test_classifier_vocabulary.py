"""Plan v2.1 M5 — classifier vocabulary-robustness golden tests.

The classifier's system_prompt (in designer_workflow.yaml) instructs it
to derive structural axes from work shape, not from brief vocabulary.
This module asserts the DOWNSTREAM consequence: when the classifier
emits the correct structural axes (regardless of brief vocab), the
select_topology function routes to the right topology.

These are vocabulary-trap golden tests: each one simulates the
TaskSignature the classifier SHOULD emit for a vocabulary-misleading
brief, and asserts the resulting topology is the work-shape-correct
one (not the vocab-suggested one).

This complements `test_task_signature.py::test_v21_rule1_*` by tying
the three-rule precedence to the vocabulary-robustness contract in
the classifier prompt. Test fixtures here are signatures, not live
classifier output (no LLM call); the prompt itself is asserted in
`test_designer_workflow_yaml.py`.
"""
from __future__ import annotations

from deep_research.agent_designer.task_signature import (
    TaskSignature,
    select_topology,
)


def _sig_with_vocab_brief_open_research(**overrides: object) -> TaskSignature:
    """A signature the classifier might emit for a brief whose
    vocabulary suggests 'plan-and-execute' but whose work shape is
    six independent domains.

    The classifier's system_prompt mandates emitting
    independent_workstreams_count=6 from the explicit enumeration in
    the brief, not from the misleading vocabulary."""
    base: dict[str, object] = {
        "asset_signature": "web_only",
        # NB: retrieval_pattern is downstream/descriptive; the new
        # structural axes are authoritative for topology.
        "retrieval_pattern": "open_research",
        "question_class": "open_research",
        "primary_evidence_kind": "web_articles",
        "expected_output_shape": "structured_report",
        # The structural axes the classifier extracts:
        "independent_workstreams_count": 6,
        "step_dependencies_present": False,
        "iteration_required": False,
        "output_aggregation_kind": "cross_concern_synthesis",
        "lane_descriptions": [
            "fundamentals",
            "valuation",
            "risk",
            "market trends",
            "earnings",
            "competitors",
        ],
    }
    base.update(overrides)
    return TaskSignature(**base)


def _sig_with_vocab_brief_sequential_compute() -> TaskSignature:
    """A signature for a brief whose vocabulary suggests 'six parallel
    queries' but whose work shape is sequential retrieve→read→compute.

    The classifier must see through the 'parallel' vocab and emit
    step_dependencies_present=True with independent_workstreams_count=1."""
    return TaskSignature(
        asset_signature="corpus_only",
        retrieval_pattern="pipelined_retrieve_read_compute",
        question_class="numeric_aggregation",
        primary_evidence_kind="structured_tables",
        expected_output_shape="single_number",
        independent_workstreams_count=1,
        step_dependencies_present=True,
        iteration_required=True,
        output_aggregation_kind="single_answer",
        lane_descriptions=["retrieve→read→compute pipeline"],
    )


# ---------------------------------------------------------------------------
# Vocabulary trap #1: brief says "plan-and-execute" but work is parallel
# ---------------------------------------------------------------------------


def test_vocab_trap_plan_and_execute_with_six_independent_lanes() -> None:
    """Brief vocabulary: 'Use plan-and-execute patterns across six
    independent research domains'.

    When the classifier sees this, it MUST emit
    independent_workstreams_count=6 (from the enumeration) instead of
    blindly setting step_dependencies_present=True (from the
    'plan-and-execute' vocab). The downstream topology selector then
    correctly routes to parallel_lanes.

    This is the exact failure mode from the Investment scaffold-and-run:
    the architect blindly obeyed the 'plan-and-execute' vocabulary,
    yielding a 6-lane plan_and_execute that detect_topology_mismatch
    flagged at the >=4 threshold."""
    sig = _sig_with_vocab_brief_open_research()
    assert select_topology(sig) == "parallel_lanes"


def test_vocab_trap_iteration_required_does_not_demote_six_lanes() -> None:
    """Even if the brief signals iteration_required=True (e.g.,
    'continuously refine across the six domains'), the M4 Rule 1
    precedence keeps it at parallel_lanes — INDEPENDENCE WINS FIRST.

    Each lane handles its own iteration via its own ReAct loop;
    workflow-level iteration is not appropriate for independent
    concerns."""
    sig = _sig_with_vocab_brief_open_research(iteration_required=True)
    assert select_topology(sig) == "parallel_lanes"


def test_vocab_trap_dependencies_does_not_demote_six_lanes() -> None:
    """Even if the brief mentions per-lane dependencies (the classifier
    might overinterpret 'reconcile findings across domains' as
    step_dependencies_present=True), Rule 1 still wins."""
    sig = _sig_with_vocab_brief_open_research(step_dependencies_present=True)
    assert select_topology(sig) == "parallel_lanes"


# ---------------------------------------------------------------------------
# Vocabulary trap #2: brief says "parallel" but work is sequential
# ---------------------------------------------------------------------------


def test_vocab_trap_parallel_queries_in_sequence_to_plan_and_execute() -> None:
    """Brief vocabulary: 'Run six parallel queries in sequence and
    combine the results'.

    The 'parallel' vocab is misleading — the actual work is sequential
    (run query, then use result for next query). The classifier MUST
    emit step_dependencies_present=True with
    independent_workstreams_count=1, NOT count=6."""
    sig = _sig_with_vocab_brief_sequential_compute()
    assert select_topology(sig) == "plan_and_execute"


# ---------------------------------------------------------------------------
# Anti-hardcoding test: lane_descriptions are extractive (verbatim)
# ---------------------------------------------------------------------------


def test_lane_descriptions_passthrough_no_taxonomy_invented() -> None:
    """Verifies the classifier's extractive-only contract at the schema
    level: lane_descriptions accepts arbitrary short strings (the
    classifier copies them verbatim from the brief).

    If the schema had a hardcoded category enum here, the contract
    would be violated. This test asserts the field accepts free-form
    strings — domain-agnostic."""
    sig = _sig_with_vocab_brief_open_research(
        lane_descriptions=[
            "any phrase a",
            "any phrase b",
            "any phrase c",
            "any phrase d",
            "any phrase e",
            "any phrase f",
        ]
    )
    assert len(sig.lane_descriptions) == 6
    assert sig.lane_descriptions[0] == "any phrase a"


def test_select_topology_unchanged_by_lane_description_content() -> None:
    """Topology depends ONLY on structural axes
    (independent_workstreams_count, step_dependencies_present,
    iteration_required), never on the textual content of
    lane_descriptions. This guards against silent recipe-based
    hardcoding sneaking into the topology selector."""
    sig_a = _sig_with_vocab_brief_open_research(
        lane_descriptions=["foo a", "foo b", "foo c", "foo d", "foo e", "foo f"]
    )
    sig_b = _sig_with_vocab_brief_open_research(
        lane_descriptions=["bar 1", "bar 2", "bar 3", "bar 4", "bar 5", "bar 6"]
    )
    assert select_topology(sig_a) == select_topology(sig_b) == "parallel_lanes"


# ---------------------------------------------------------------------------
# Boundary cases
# ---------------------------------------------------------------------------


def test_single_concern_with_iteration_is_plan_and_execute() -> None:
    """count=1 + iteration=True → plan_and_execute (M4 Rule 2)."""
    sig = TaskSignature(
        asset_signature="corpus_only",
        retrieval_pattern="bounded_lookup",
        question_class="bounded_lookup",
        primary_evidence_kind="text_chunks",
        expected_output_shape="paragraph",
        independent_workstreams_count=1,
        step_dependencies_present=False,
        iteration_required=True,
        output_aggregation_kind="single_answer",
        lane_descriptions=["single iterative concern"],
    )
    assert select_topology(sig) == "plan_and_execute"


def test_two_independent_lanes_minimum_for_parallel() -> None:
    """count=2 is the minimum for parallel_lanes (Rule 1)."""
    sig = TaskSignature(
        asset_signature="web_only",
        retrieval_pattern="independent_lanes",
        question_class="open_research",
        primary_evidence_kind="web_articles",
        expected_output_shape="structured_report",
        independent_workstreams_count=2,
        step_dependencies_present=False,
        iteration_required=False,
        output_aggregation_kind="cross_concern_synthesis",
        lane_descriptions=["lane alpha", "lane beta"],
    )
    assert select_topology(sig) == "parallel_lanes"


def test_zero_lanes_treated_as_single_agent() -> None:
    """count=0 is degenerate; with no deps and no iteration, it must
    map to single_agent (Rule 3) — same as count=1."""
    sig = TaskSignature(
        asset_signature="web_only",
        retrieval_pattern="bounded_lookup",
        question_class="bounded_lookup",
        primary_evidence_kind="web_articles",
        expected_output_shape="paragraph",
        independent_workstreams_count=0,
        step_dependencies_present=False,
        iteration_required=False,
        output_aggregation_kind="single_answer",
        lane_descriptions=["degenerate single concern"],
    )
    assert select_topology(sig) == "single_agent"
