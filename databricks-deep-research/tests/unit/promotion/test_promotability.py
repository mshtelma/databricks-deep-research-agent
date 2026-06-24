"""Tests for promotability scoring (feature 6.3, advisory slice)."""

from __future__ import annotations

from databricks_deep_research.promotion import (
    PromotionStep,
    PromotionTrace,
    StepKind,
    aggregate_promotion_traces,
    score_promotability,
)


def _tool(order: int, name: str) -> PromotionStep:
    return PromotionStep(order=order, kind=StepKind.TOOL, tool_name=name)


def _trace(*steps: PromotionStep, is_degenerate: bool = False) -> PromotionTrace:
    return PromotionTrace(steps=list(steps), is_degenerate=is_degenerate)


def test_stable_runs_pin_when_enough_evidence() -> None:
    t = _trace(_tool(0, "a"), _tool(1, "b"))
    report = score_promotability(aggregate_promotion_traces([t, t, t]))
    assert report.runs_aggregated == 3
    assert [r.verdict for r in report.regions] == ["pin"]
    assert report.regions[0].confidence == 1.0
    assert report.warnings == []


def test_thin_evidence_stays_bounded_with_warning() -> None:
    # One run: structurally stable, but below the pin threshold → conservative.
    report = score_promotability(aggregate_promotion_traces([_trace(_tool(0, "a"))]))
    assert [r.verdict for r in report.regions] == ["bounded_cell"]
    assert any("fewer than" in w for w in report.warnings)


def test_mixed_regions_grouped_contiguously() -> None:
    common0 = _tool(0, "a")
    common2 = _tool(2, "c")
    t1 = _trace(common0, _tool(1, "x"), common2)
    t2 = _trace(common0, _tool(1, "y"), common2)
    t3 = _trace(common0, _tool(1, "z"), common2)
    report = score_promotability(aggregate_promotion_traces([t1, t2, t3]))
    # pos0 stable → pin; pos1 varies → bounded; pos2 stable → pin
    assert [r.verdict for r in report.regions] == ["pin", "bounded_cell", "pin"]
    assert report.regions[0].start_order == 0 and report.regions[0].end_order == 0
    assert report.regions[1].start_order == 1
    assert report.regions[2].end_order == 2


def test_custom_min_runs_to_pin() -> None:
    t = _trace(_tool(0, "a"))
    # Lower the bar to 1 → a single stable run pins.
    report = score_promotability(
        aggregate_promotion_traces([t]), min_runs_to_pin=1
    )
    assert [r.verdict for r in report.regions] == ["pin"]


def test_degenerate_warning() -> None:
    t = _trace(_tool(0, "a"), is_degenerate=True)
    report = score_promotability(aggregate_promotion_traces([t, t, t]))
    assert any("simple" in w for w in report.warnings)


def test_no_runs_warns() -> None:
    report = score_promotability(aggregate_promotion_traces([]))
    assert report.runs_aggregated == 0
    assert report.regions == []
    assert report.warnings == ["no runs to aggregate"]
