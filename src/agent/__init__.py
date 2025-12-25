"""Agent package - Multi-agent research system."""

from src.agent.orchestrator import (
    OrchestrationConfig,
    OrchestrationResult,
    run_research,
    stream_research,
)
from src.agent.state import (
    Plan,
    PlanStep,
    QueryClassification,
    ReflectionDecision,
    ReflectionResult,
    ResearchState,
    SourceInfo,
    StepStatus,
    StepType,
)

__all__ = [
    # Orchestrator
    "run_research",
    "stream_research",
    "OrchestrationConfig",
    "OrchestrationResult",
    # State
    "ResearchState",
    "Plan",
    "PlanStep",
    "StepType",
    "StepStatus",
    "QueryClassification",
    "ReflectionDecision",
    "ReflectionResult",
    "SourceInfo",
]
