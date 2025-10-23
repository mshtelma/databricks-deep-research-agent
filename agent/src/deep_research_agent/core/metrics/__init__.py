"""Metric pipeline components for deterministic calculation orchestration."""

# NEW: Calculation Agent components
from .config import (
    UnifiedPlanningConfig,
    SourceType,
    ObservationType,
    LLMTier,
    GeographicLevel,
    AssumptionSeverity,
)
from .unified_models import (
    MetricSpec,
    DataSource,  # Backward compatibility alias
    UnifiedPlan,
    UserRequestAnalysis,
    ResponseTable,
    TableCell,
)
from .models import (
    DataPoint,
    ExtractedMetric,
    MetricSpecBundle,
    CalculationTask,
    CalculationPlan,
    CalculationEvent,
    CalculationValidation,
    CalculationProvenance,
)
from .metric_extractor import (
    MetricExtractor,
    extract_metric_with_llm,
)
from .formula_evaluator import (
    SimpleFormulaEvaluator,
    FormulaEvaluation,
    execute_calculation_with_safety,
)

# Legacy components removed (pipeline.py, planner.py, state.py, instrumentation.py, planner_prompts.py)
# Now using only UnifiedPlan-based calculation via calculation_agent

__all__ = [
    # NEW: Calculation Agent exports
    "UnifiedPlanningConfig",
    "SourceType",
    "ObservationType",
    "LLMTier",
    "GeographicLevel",
    "AssumptionSeverity",
    "MetricSpec",
    "DataSource",
    "UnifiedPlan",
    "UserRequestAnalysis",
    "ResponseTable",
    "TableCell",
    "DataPoint",
    "ExtractedMetric",
    "MetricExtractor",
    "extract_metric_with_llm",
    "SimpleFormulaEvaluator",
    "FormulaEvaluation",
    "execute_calculation_with_safety",
    # Active metric models (still used)
    "MetricSpecBundle",
    "CalculationTask",
    "CalculationPlan",
    "CalculationEvent",
    "CalculationValidation",
    "CalculationProvenance",
]
