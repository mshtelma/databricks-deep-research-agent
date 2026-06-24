"""Promotability scoring (spec feature 6.3, advisory slice).

Turns an :class:`AggregatedTrace` into a :class:`PromotabilityReport`: contiguous
**regions** each marked ``pin`` (recurred reliably → safe to fix as explicit
structure) or ``bounded_cell`` (varies, or too few runs to trust → keep as a
bounded autonomous Cell). Under "full architect authoring" this report is
*advisory*: it grounds the architect ("pin these, keep those flexible") and is a
post-hoc cross-check on the authored workflow.

Conservatism for thin evidence is enforced HERE, not in aggregation: with fewer
than ``min_runs_to_pin`` runs nothing is pinned, regardless of structural
stability.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from databricks_deep_research.promotion.aggregate import AggregatedStep, AggregatedTrace

# Default evidence bar for pinning a stable region as explicit structure.
# Config-driven: callers may override per the spec's ≥N-runs guidance.
MIN_RUNS_TO_PIN = 3

Verdict = Literal["pin", "bounded_cell"]


class RegionVerdict(BaseModel):
    """A contiguous run of aggregated steps with a pin/Cell recommendation."""

    model_config = ConfigDict(extra="ignore")

    start_order: int
    end_order: int
    signatures: list[str]
    verdict: Verdict
    runs_seen: int
    confidence: float
    note: str


class PromotabilityReport(BaseModel):
    """Advisory pin/Cell map over an aggregated trace, plus reviewer warnings."""

    model_config = ConfigDict(extra="ignore")

    runs_aggregated: int
    regions: list[RegionVerdict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _make_region(
    verdict: Verdict, steps: list[AggregatedStep], runs: int, *, low_evidence: bool
) -> RegionVerdict:
    runs_seen = min((s.runs_seen for s in steps), default=runs)
    confidence = round(runs_seen / runs, 3) if runs else 0.0
    if low_evidence:
        note = "insufficient runs to pin (low evidence) — kept as a bounded Cell"
    elif verdict == "pin":
        note = "recurred identically across runs — safe to pin as explicit structure"
    else:
        note = "varies across runs — kept as a bounded Cell"
    return RegionVerdict(
        start_order=steps[0].order,
        end_order=steps[-1].order,
        signatures=[s.signature for s in steps],
        verdict=verdict,
        runs_seen=runs_seen,
        confidence=confidence,
        note=note,
    )


def score_promotability(
    agg: AggregatedTrace, *, min_runs_to_pin: int = MIN_RUNS_TO_PIN
) -> PromotabilityReport:
    """Group the aggregated skeleton into pin/Cell regions with warnings."""
    warnings: list[str] = []
    if agg.runs == 0:
        return PromotabilityReport(
            runs_aggregated=0, regions=[], warnings=["no runs to aggregate"]
        )

    low_evidence = agg.runs < min_runs_to_pin
    if low_evidence:
        warnings.append(
            f"only {agg.runs} run(s) aggregated; fewer than {min_runs_to_pin} → "
            "regions kept as bounded Cells (low confidence). Run the agent more "
            "times for a tighter, more-pinned workflow."
        )
    if agg.total_runs_degenerate:
        warnings.append(
            f"{agg.total_runs_degenerate}/{agg.runs} run(s) were classified simple "
            "(little structure to promote)."
        )

    regions: list[RegionVerdict] = []
    current_verdict: Verdict | None = None
    current_steps: list[AggregatedStep] = []
    for step in agg.steps:
        pinnable = step.stable and not low_evidence
        verdict: Verdict = "pin" if pinnable else "bounded_cell"
        if current_verdict is None or current_verdict != verdict:
            if current_steps:
                regions.append(
                    _make_region(
                        current_verdict or "bounded_cell",
                        current_steps,
                        agg.runs,
                        low_evidence=low_evidence,
                    )
                )
            current_verdict = verdict
            current_steps = [step]
        else:
            current_steps.append(step)
    if current_steps:
        regions.append(
            _make_region(
                current_verdict or "bounded_cell",
                current_steps,
                agg.runs,
                low_evidence=low_evidence,
            )
        )

    return PromotabilityReport(
        runs_aggregated=agg.runs, regions=regions, warnings=warnings
    )


__all__ = [
    "MIN_RUNS_TO_PIN",
    "PromotabilityReport",
    "RegionVerdict",
    "Verdict",
    "score_promotability",
]
