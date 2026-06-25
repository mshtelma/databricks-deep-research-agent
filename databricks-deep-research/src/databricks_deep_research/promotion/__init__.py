"""Promotion layer (spec Wave 5 / Phase 6).

Turns observed run traces into validated, governed workflows. This package is
pure (no app/Designer imports) and is consumed directly via
``databricks_deep_research.promotion`` (it is intentionally NOT part of the
curated top-level public API).
"""

from databricks_deep_research.promotion.aggregate import (
    AggregatedStep,
    AggregatedTrace,
    aggregate_promotion_traces,
)
from databricks_deep_research.promotion.promotability import (
    MIN_RUNS_TO_PIN,
    PromotabilityReport,
    RegionVerdict,
    score_promotability,
)
from databricks_deep_research.promotion.trace_model import (
    PromotionStep,
    PromotionTrace,
    PromotionTraceBuilder,
    StepKind,
    extract_promotion_trace,
)

__all__ = [
    "MIN_RUNS_TO_PIN",
    "AggregatedStep",
    "AggregatedTrace",
    "PromotabilityReport",
    "PromotionStep",
    "PromotionTrace",
    "PromotionTraceBuilder",
    "RegionVerdict",
    "StepKind",
    "aggregate_promotion_traces",
    "extract_promotion_trace",
    "score_promotability",
]
