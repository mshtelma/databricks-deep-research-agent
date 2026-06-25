"""Cross-run aggregation of promotion traces (spec feature 6.1 / input to 6.3).

Given N :class:`PromotionTrace`s of the same agent/task, separate the **stable
skeleton** (steps that recur in the same position with the same signature across
runs) from **variable regions** (steps that differ or are sometimes absent).

This module is purely *structural* — it reports what recurred, not whether to
pin it. The pin-vs-Cell *policy* (which depends on how many runs we have) lives
in :mod:`promotability`, so N=1 staying conservative is a policy choice there,
not a structural artifact here.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from databricks_deep_research.promotion.trace_model import (
    PromotionStep,
    PromotionTrace,
    StepKind,
)


def _signature(step: PromotionStep) -> str:
    """A position-alignment key for a step (identity used to compare across runs)."""
    if step.kind == StepKind.TOOL:
        return f"tool:{step.tool_name}"
    if step.kind == StepKind.AGENT:
        return f"agent:{step.subtype or step.label}"
    return step.kind.value


class AggregatedStep(BaseModel):
    """One position in the aggregated skeleton (union over runs, by position)."""

    model_config = ConfigDict(extra="ignore")

    order: int
    kind: StepKind
    signature: str
    runs_seen: int  # how many runs had THIS signature at THIS position
    stable: bool  # runs_seen == runs (present + identical across every run)
    tool_name: str = ""
    subtype: str = ""
    label: str = ""


class AggregatedTrace(BaseModel):
    """Positional union of N runs with per-position stability."""

    model_config = ConfigDict(extra="ignore")

    runs: int
    steps: list[AggregatedStep] = Field(default_factory=list)
    total_runs_degenerate: int = 0


def aggregate_promotion_traces(traces: list[PromotionTrace]) -> AggregatedTrace:
    """Aggregate N traces into a positional skeleton with per-position stability.

    Deterministic: at each position the dominant signature is chosen by count,
    ties broken by first-seen (input trace order). N=0 → empty; N=1 → every
    position ``runs_seen == 1`` and ``stable`` (the conservatism is applied later
    by :func:`promotability.score_promotability`).
    """
    runs = len(traces)
    if runs == 0:
        return AggregatedTrace(runs=0, steps=[], total_runs_degenerate=0)

    max_len = max((len(t.steps) for t in traces), default=0)
    agg_steps: list[AggregatedStep] = []
    for i in range(max_len):
        counts: dict[str, int] = {}
        reps: dict[str, PromotionStep] = {}
        for trace in traces:
            if i < len(trace.steps):
                step = trace.steps[i]
                sig = _signature(step)
                counts[sig] = counts.get(sig, 0) + 1
                reps.setdefault(sig, step)
        # Dominant signature at this position (deterministic tie-break: first-seen).
        best_sig = max(counts, key=lambda k: counts[k])
        rep = reps[best_sig]
        runs_seen = counts[best_sig]
        agg_steps.append(
            AggregatedStep(
                order=i,
                kind=rep.kind,
                signature=best_sig,
                runs_seen=runs_seen,
                stable=(runs_seen == runs),
                tool_name=rep.tool_name,
                subtype=rep.subtype,
                label=rep.label,
            )
        )

    degenerate = sum(1 for t in traces if t.is_degenerate)
    return AggregatedTrace(runs=runs, steps=agg_steps, total_runs_degenerate=degenerate)


__all__ = [
    "AggregatedStep",
    "AggregatedTrace",
    "aggregate_promotion_traces",
]
