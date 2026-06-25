"""Tests for cross-run aggregation of promotion traces (feature 6.1)."""

from __future__ import annotations

from databricks_deep_research.promotion import (
    PromotionStep,
    PromotionTrace,
    StepKind,
    aggregate_promotion_traces,
)


def _tool(order: int, name: str) -> PromotionStep:
    return PromotionStep(order=order, kind=StepKind.TOOL, tool_name=name)


def _agent(order: int, label: str) -> PromotionStep:
    return PromotionStep(order=order, kind=StepKind.AGENT, label=label)


def _trace(*steps: PromotionStep, is_degenerate: bool = False) -> PromotionTrace:
    return PromotionTrace(steps=list(steps), is_degenerate=is_degenerate)


def test_identical_traces_are_all_stable() -> None:
    t = _trace(_agent(0, "Researcher"), _tool(1, "web_search"))
    agg = aggregate_promotion_traces([t, t, t])
    assert agg.runs == 3
    assert [s.stable for s in agg.steps] == [True, True]
    assert [s.runs_seen for s in agg.steps] == [3, 3]
    assert agg.steps[0].signature == "agent:Researcher"
    assert agg.steps[1].signature == "tool:web_search"


def test_divergent_position_is_not_stable() -> None:
    common = _agent(0, "R")
    t1 = _trace(common, _tool(1, "b"))
    t2 = _trace(common, _tool(1, "b"))
    t3 = _trace(common, _tool(1, "c"))
    agg = aggregate_promotion_traces([t1, t2, t3])
    assert agg.steps[0].stable is True
    assert agg.steps[1].stable is False
    assert agg.steps[1].runs_seen == 2  # dominant "tool:b" seen in 2/3
    assert agg.steps[1].signature == "tool:b"


def test_different_lengths_union_to_max() -> None:
    t1 = _trace(_tool(0, "a"), _tool(1, "b"), _tool(2, "c"))
    t2 = _trace(_tool(0, "a"), _tool(1, "b"))
    agg = aggregate_promotion_traces([t1, t2])
    assert len(agg.steps) == 3
    assert agg.steps[0].stable and agg.steps[1].stable
    assert agg.steps[2].stable is False
    assert agg.steps[2].runs_seen == 1


def test_empty_list() -> None:
    agg = aggregate_promotion_traces([])
    assert agg.runs == 0
    assert agg.steps == []


def test_single_trace_is_structurally_stable() -> None:
    agg = aggregate_promotion_traces([_trace(_tool(0, "a"))])
    assert agg.runs == 1
    assert agg.steps[0].stable is True
    assert agg.steps[0].runs_seen == 1


def test_degenerate_runs_counted() -> None:
    agg = aggregate_promotion_traces(
        [_trace(_tool(0, "a"), is_degenerate=True), _trace(_tool(0, "a"))]
    )
    assert agg.total_runs_degenerate == 1
