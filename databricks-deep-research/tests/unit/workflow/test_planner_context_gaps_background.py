"""Unit tests for planner-context wiring of features 3.5 (knowledge_gaps)
and 3.6 (background grounding).

Covers:
  * ``build_planner_runtime_context`` renders ``{knowledge_gaps}`` when the
    replan feedback carries gaps, and ``{background}`` when the store exposes
    a background summary / query decomposition.
  * Both render to ``""`` when absent (== today's behavior).
  * ``ReplanFeedbackEntry`` carries ``knowledge_gaps`` (and defaults empty).
  * The runner's ``_extract_knowledge_gaps`` / ``_extract_rubric`` helpers
    pull the additive ReflectionOutput fields tolerantly from dict/model.
  * Stub replan: 2 gaps in -> the rendered planner context references them.

All pure / mocked — no LLM, no executor, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from databricks_deep_research.agents.output_models import ReflectionOutput
from databricks_deep_research.workflow.runtime.plan_execute_runner import (
    _extract_knowledge_gaps,
    _extract_rubric,
)
from databricks_deep_research.workflow.runtime.plan_execute_types import (
    PlanCycleContext,
    ReplanFeedbackEntry,
)
from databricks_deep_research.workflow.runtime.planner_context import (
    build_planner_runtime_context,
    format_background,
    format_knowledge_gaps,
)
from databricks_deep_research.workflow.state import WorkflowState


def _no_observations(_pools: dict[str, Any]) -> str:
    return "(no observations)"


@dataclass
class _StubPlannerConfig:
    """Minimal config exposing only the attrs build_planner_runtime_context reads."""

    min_iterations: int = 1
    max_iterations: int = 10
    planner_guidance: str = ""


def _build_context(
    *, cycle_ctx: PlanCycleContext, state: WorkflowState
) -> dict[str, Any]:
    return build_planner_runtime_context(
        config=_StubPlannerConfig(),
        state=state,
        pools={},
        cycle_ctx=cycle_ctx,
        format_all_observations=_no_observations,
    )


# ---------------------------------------------------------------------------
# Empty == today's behavior
# ---------------------------------------------------------------------------


def test_context_empty_gaps_and_background_by_default() -> None:
    state = WorkflowState(query="q")
    ctx = _build_context(cycle_ctx=PlanCycleContext(), state=state)
    assert ctx["knowledge_gaps"] == ""
    assert ctx["background"] == ""
    # Pre-existing keys still present (no regression).
    assert "reflector_feedback" in ctx
    assert "data_landscape" in ctx


# ---------------------------------------------------------------------------
# 3.5 — knowledge_gaps rendering
# ---------------------------------------------------------------------------


def test_format_knowledge_gaps_dedupes_and_orders() -> None:
    entries = [
        ReplanFeedbackEntry(
            reason="evaluator_replan",
            cycle=0,
            message="m",
            knowledge_gaps=["gap A", "gap B"],
        ),
        ReplanFeedbackEntry(
            reason="evaluator_replan",
            cycle=1,
            message="m",
            knowledge_gaps=["gap B", "gap C"],  # gap B duplicate -> dropped
        ),
    ]
    rendered = format_knowledge_gaps(entries)
    assert rendered == "- gap A\n- gap B\n- gap C"


def test_context_renders_gaps_when_present() -> None:
    cycle_ctx = PlanCycleContext(
        feedback_history=[
            ReplanFeedbackEntry(
                reason="evaluator_replan",
                cycle=0,
                message="needs more",
                knowledge_gaps=["missing 2025 revenue", "no competitor pricing"],
            )
        ]
    )
    ctx = _build_context(cycle_ctx=cycle_ctx, state=WorkflowState(query="q"))
    assert "missing 2025 revenue" in ctx["knowledge_gaps"]
    assert "no competitor pricing" in ctx["knowledge_gaps"]


def test_replan_feedback_entry_carries_gaps_and_defaults_empty() -> None:
    plain = ReplanFeedbackEntry(reason="empty_plan", cycle=0, message="m")
    assert plain.knowledge_gaps == []
    with_gaps = ReplanFeedbackEntry(
        reason="evaluator_replan", cycle=0, message="m", knowledge_gaps=["x"]
    )
    assert with_gaps.knowledge_gaps == ["x"]


# ---------------------------------------------------------------------------
# 3.6 — background grounding rendering
# ---------------------------------------------------------------------------


def test_format_background_empty_when_no_summary() -> None:
    assert format_background(WorkflowState(query="q")) == ""


def test_context_renders_background_summary_from_state() -> None:
    state = WorkflowState(query="q")
    state.append("bg", "background_summary", "Preliminary scan found 3 vendors.")
    state.append("bg", "query_decomposition", ["What is X?", "How does Y compare?"])
    ctx = _build_context(cycle_ctx=PlanCycleContext(), state=state)
    assert "Preliminary scan found 3 vendors." in ctx["background"]
    assert "What is X?" in ctx["background"]
    assert "How does Y compare?" in ctx["background"]
    assert "Query decomposition" in ctx["background"]


def test_background_summary_only_no_decomposition() -> None:
    state = WorkflowState(query="q")
    state.append("bg", "background_summary", "Just a summary.")
    out = format_background(state)
    assert out == "Just a summary."  # no decomposition section appended


# ---------------------------------------------------------------------------
# Runner extraction helpers (tolerant of dict + model)
# ---------------------------------------------------------------------------


def test_extract_knowledge_gaps_from_model_and_dict() -> None:
    model = ReflectionOutput(
        decision="continue", reasoning="r", knowledge_gaps=["a", "b"]
    )
    assert _extract_knowledge_gaps(model) == ["a", "b"]
    assert _extract_knowledge_gaps({"knowledge_gaps": ["c", "  ", None, "d"]}) == [
        "c",
        "d",
    ]
    # Junk / absent -> empty, never raises.
    assert _extract_knowledge_gaps({"knowledge_gaps": "nope"}) == []
    assert _extract_knowledge_gaps(None) == []
    assert _extract_knowledge_gaps("garbage") == []


def test_extract_rubric_from_model_and_dict() -> None:
    model = ReflectionOutput.model_validate(
        {
            "decision": "complete",
            "reasoning": "r",
            "rubric": {
                "completeness": 8,
                "depth": 7,
                "reliability": 9,
                "recency": 6,
                "overall": 7.5,
            },
        }
    )
    dumped = _extract_rubric(model)
    assert isinstance(dumped, dict)
    assert dumped["completeness"] == 8
    assert _extract_rubric({"rubric": {"completeness": 5}})["completeness"] == 5
    # Absent / junk -> None.
    assert _extract_rubric({"reasoning": "r"}) is None
    assert _extract_rubric(None) is None


# ---------------------------------------------------------------------------
# Stub replan end-to-end: 2 gaps in -> rendered planner context references them
# ---------------------------------------------------------------------------


def test_stub_replan_two_gaps_reach_planner_context() -> None:
    """Simulate the evaluator-replan path: a ReflectionOutput with 2 gaps is
    recorded on the feedback history (as the runner does), then surfaced in the
    planner runtime context the next plan cycle consumes."""
    reflection = ReflectionOutput(
        decision="adjust",
        reasoning="two areas uncovered",
        knowledge_gaps=["regulatory status unknown", "no pricing tiers found"],
    )
    cycle_ctx = PlanCycleContext()
    # Mirror plan_execute_runner's evaluator-replan append.
    cycle_ctx.feedback_history.append(
        ReplanFeedbackEntry(
            reason="evaluator_replan",
            cycle=0,
            message=reflection.reasoning,
            step_title="step-1",
            knowledge_gaps=_extract_knowledge_gaps(reflection),
        )
    )
    ctx = _build_context(cycle_ctx=cycle_ctx, state=WorkflowState(query="q"))
    assert "regulatory status unknown" in ctx["knowledge_gaps"]
    assert "no pricing tiers found" in ctx["knowledge_gaps"]
