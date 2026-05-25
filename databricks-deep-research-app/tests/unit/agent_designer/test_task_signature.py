"""PR3-B Layer 1: TaskSignature pydantic + select_topology contract tests."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from deep_research.agent_designer.task_signature import (
    TOPOLOGIES,
    TaskSignature,
    select_topology,
)


def _sig(**overrides: object) -> TaskSignature:
    base: dict[str, object] = {
        "asset_signature": "corpus_only",
        "retrieval_pattern": "pipelined_retrieve_read_compute",
        "question_class": "numeric_aggregation",
        "question_ambiguity": ["period_basis"],
        "primary_evidence_kind": "structured_tables",
        "expected_output_shape": "single_number",
    }
    base.update(overrides)
    # Use load_from_storage so legacy-style fixtures without the v2.1
    # structural axes still build. Tests that need strict-emission
    # semantics call ``TaskSignature.from_classifier_emission`` directly.
    return TaskSignature.load_from_storage(base)


def test_select_topology_pipelined_to_plan_and_execute() -> None:
    assert select_topology(_sig(retrieval_pattern="pipelined_retrieve_read_compute")) == "plan_and_execute"


def test_select_topology_independent_lanes_to_parallel_lanes() -> None:
    assert select_topology(_sig(retrieval_pattern="independent_lanes")) == "parallel_lanes"


def test_select_topology_bounded_lookup_to_single_agent() -> None:
    assert select_topology(_sig(retrieval_pattern="bounded_lookup")) == "single_agent"


def test_select_topology_open_research_to_parallel_lanes() -> None:
    assert select_topology(_sig(retrieval_pattern="open_research")) == "parallel_lanes"


def test_select_topology_rejects_unknown_pattern() -> None:
    sig = _sig()
    object.__setattr__(sig, "retrieval_pattern", "nonsense_pattern")
    with pytest.raises(ValueError, match="unknown retrieval_pattern"):
        select_topology(sig)


def test_task_signature_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        TaskSignature(  # type: ignore[call-arg]
            asset_signature="corpus_only",
            retrieval_pattern="bounded_lookup",
            question_class="bounded_lookup",
            primary_evidence_kind="text_chunks",
            expected_output_shape="single_number",
            extra_field="nope",
        )


def test_task_signature_default_empty_ambiguity() -> None:
    # Use load_from_storage to fill the v2.1 structural-axis defaults;
    # this test asserts that legacy 5-field payloads still produce
    # sensible defaults for the descriptive fields.
    sig = TaskSignature.load_from_storage({
        "asset_signature": "web_only",
        "retrieval_pattern": "open_research",
        "question_class": "open_research",
        "primary_evidence_kind": "web_articles",
        "expected_output_shape": "paragraph",
    })
    assert sig.question_ambiguity == []
    assert sig.confidence == 1.0


def test_task_signature_confidence_bounded() -> None:
    with pytest.raises(ValidationError):
        TaskSignature(
            asset_signature="web_only",
            retrieval_pattern="open_research",
            question_class="open_research",
            primary_evidence_kind="web_articles",
            expected_output_shape="paragraph",
            confidence=1.5,
        )


def test_task_signature_rejects_unknown_axis() -> None:
    with pytest.raises(ValidationError):
        TaskSignature(
            asset_signature="corpus_only",
            retrieval_pattern="bounded_lookup",
            question_class="bounded_lookup",
            question_ambiguity=["not_a_real_axis"],  # type: ignore[list-item]
            primary_evidence_kind="text_chunks",
            expected_output_shape="single_number",
        )


def test_topologies_constant_complete() -> None:
    """Every topology select_topology can return must be in TOPOLOGIES."""
    returned = {
        select_topology(_sig(retrieval_pattern="pipelined_retrieve_read_compute")),
        select_topology(_sig(retrieval_pattern="independent_lanes")),
        select_topology(_sig(retrieval_pattern="bounded_lookup")),
        select_topology(_sig(retrieval_pattern="open_research")),
    }
    assert returned <= set(TOPOLOGIES)


# ---------------------------------------------------------------------------
# Plan v2.1 — structural-axis based select_topology precedence tests
#
# Three-rule precedence per plan M4 (codex CRITICAL-5 fix):
#   Rule 1: independent_workstreams_count >= 2 → parallel_lanes
#           (INDEPENDENCE WINS FIRST; iteration cannot override).
#   Rule 2: step_dependencies_present OR iteration_required → plan_and_execute.
#   Rule 3: otherwise → single_agent.
# ---------------------------------------------------------------------------


def _structural_sig(**overrides: object) -> TaskSignature:
    """Helper: a signature with at least one structural axis set, so the
    new precedence path is exercised (not the legacy retrieval_pattern
    fallback)."""
    base: dict[str, object] = {
        "asset_signature": "web_only",
        "retrieval_pattern": "open_research",  # deliberately wrong-vocab
        "question_class": "open_research",
        "primary_evidence_kind": "web_articles",
        "expected_output_shape": "structured_report",
        # Set at least one structural axis so _has_explicit_structural_axes
        # returns True and the v2.1 precedence runs.
        "independent_workstreams_count": 6,
        "lane_descriptions": ["lane a", "lane b", "lane c", "lane d", "lane e", "lane f"],
    }
    base.update(overrides)
    # Use load_from_storage so legacy-style fixtures parse; tests that
    # need strict-emission semantics call from_classifier_emission directly.
    return TaskSignature.load_from_storage(base)


def test_v21_rule1_independence_wins_over_iteration() -> None:
    """Rule 1: count>=2 maps to parallel_lanes even when iteration_required=True.

    This is the explicit codex CRITICAL-5 fix: under the original v2 ordering,
    a six-domain task with iteration_required=True would have routed back to
    plan_and_execute, recreating the Investment failure mode.
    """
    sig = _structural_sig(
        independent_workstreams_count=6,
        iteration_required=True,
        step_dependencies_present=True,
    )
    assert select_topology(sig) == "parallel_lanes"


def test_v21_rule1_independence_two_lanes() -> None:
    """Rule 1: minimum count for parallel_lanes is 2."""
    sig = _structural_sig(
        independent_workstreams_count=2,
        lane_descriptions=["lane a", "lane b"],
    )
    assert select_topology(sig) == "parallel_lanes"


def test_v21_rule2_step_dependencies_to_plan_and_execute() -> None:
    """Rule 2: single-stream with step deps → plan_and_execute."""
    sig = _structural_sig(
        independent_workstreams_count=1,
        step_dependencies_present=True,
        iteration_required=False,
        lane_descriptions=["pipeline"],
    )
    assert select_topology(sig) == "plan_and_execute"


def test_v21_rule2_iteration_to_plan_and_execute() -> None:
    """Rule 2: single-stream with iteration → plan_and_execute."""
    sig = _structural_sig(
        independent_workstreams_count=1,
        step_dependencies_present=False,
        iteration_required=True,
        lane_descriptions=["iterative work"],
    )
    assert select_topology(sig) == "plan_and_execute"


def test_v21_rule3_bare_to_single_agent() -> None:
    """Rule 3: single-stream, no deps, no iteration → single_agent."""
    sig = _structural_sig(
        independent_workstreams_count=1,
        step_dependencies_present=False,
        iteration_required=False,
        lane_descriptions=["bounded lookup"],
    )
    assert select_topology(sig) == "single_agent"


def test_v21_legacy_signature_falls_back_to_retrieval_pattern() -> None:
    """When all structural axes are at default values, the legacy PR3-B
    retrieval_pattern mapping is used. This preserves behavior for older
    signatures and existing test fixtures."""
    # _sig() leaves the new structural axes at defaults (count=1, deps=False,
    # iter=False, ...), so the legacy path runs.
    assert select_topology(_sig(retrieval_pattern="pipelined_retrieve_read_compute")) == "plan_and_execute"
    assert select_topology(_sig(retrieval_pattern="independent_lanes")) == "parallel_lanes"


def test_v21_backward_compat_missing_structural_fields_parse() -> None:
    """Backward-compat: serialized TaskSignature JSON that predates the
    structural-axis extension must still parse via ``load_from_storage``.

    The strict path (``from_classifier_emission``) rejects this payload —
    that is the failure-closed semantic for fresh classifier emissions
    per Plan v2.1 M11. The lenient path stays available for MLflow trace
    replay and legacy test fixtures.
    """
    legacy_payload = {
        "asset_signature": "corpus_only",
        "retrieval_pattern": "pipelined_retrieve_read_compute",
        "question_class": "numeric_aggregation",
        "primary_evidence_kind": "structured_tables",
        "expected_output_shape": "single_number",
        "confidence": 0.95,
        # NOTE: no step_dependencies_present, no independent_workstreams_count,
        # no iteration_required, no output_aggregation_kind, no lane_descriptions,
        # no axis_reasoning. load_from_storage fills the legacy defaults.
    }
    sig = TaskSignature.load_from_storage(legacy_payload)
    assert sig.step_dependencies_present is False
    assert sig.independent_workstreams_count == 1
    assert sig.iteration_required is False
    assert sig.output_aggregation_kind == "single_answer"
    assert sig.lane_descriptions == []
    assert sig.axis_reasoning is None
    # And select_topology uses the legacy retrieval_pattern fallback.
    assert select_topology(sig) == "plan_and_execute"
    # The strict path MUST reject the same payload — fresh emissions
    # cannot omit the structural axes.
    with pytest.raises(ValidationError):
        TaskSignature.from_classifier_emission(legacy_payload)


def test_v21_lane_descriptions_alone_triggers_structural_path() -> None:
    """Even with count=1 + no deps + no iteration, if lane_descriptions is
    non-empty, the structural path runs. A single-lane description should
    still produce single_agent via Rule 3."""
    sig = _structural_sig(
        independent_workstreams_count=1,
        step_dependencies_present=False,
        iteration_required=False,
        lane_descriptions=["bounded look-up"],
    )
    # Rule 3 applies → single_agent.
    assert select_topology(sig) == "single_agent"


def test_v21_axis_reasoning_optional_dict() -> None:
    """The new axis_reasoning field accepts a dict for low-confidence outputs."""
    sig = _structural_sig(
        confidence=0.55,
        axis_reasoning={
            "independent_workstreams_count": "brief enumerates 6 distinct concerns",
            "iteration_required": "no reflection signal in brief",
        },
    )
    assert sig.axis_reasoning is not None
    assert "independent_workstreams_count" in sig.axis_reasoning


def test_v21_independent_workstreams_count_bounded() -> None:
    """Field constraint: 0..8."""
    with pytest.raises(ValidationError):
        TaskSignature(
            asset_signature="web_only",
            retrieval_pattern="open_research",
            question_class="open_research",
            primary_evidence_kind="web_articles",
            expected_output_shape="paragraph",
            independent_workstreams_count=99,
        )


def test_v21_output_aggregation_kind_literal() -> None:
    """Field constraint: Literal['single_answer','cross_concern_synthesis','per_concern_report']."""
    with pytest.raises(ValidationError):
        TaskSignature(
            asset_signature="web_only",
            retrieval_pattern="open_research",
            question_class="open_research",
            primary_evidence_kind="web_articles",
            expected_output_shape="paragraph",
            output_aggregation_kind="not_an_aggregation_kind",  # type: ignore[arg-type]
        )
