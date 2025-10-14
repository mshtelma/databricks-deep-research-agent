"""Data models that support the metric calculation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, field_validator

from ..report_generation.models import TableSpec


class CalculationTaskType(str, Enum):
    """Supported calculation task operations."""

    FORMULA = "formula"
    AGGREGATION = "aggregation"
    RATIO = "ratio"
    DELTA = "delta"
    NORMALIZATION = "normalization"
    LOOKUP = "lookup"


class CalculationProvenance(BaseModel):
    """Traceability metadata for a calculation output."""

    observation_ids: List[str] = Field(default_factory=list)
    source_tasks: List[str] = Field(default_factory=list)
    code_cell_id: Optional[str] = None
    notes: Optional[str] = None


class CalculationValidation(BaseModel):
    """Validation status captured after executing a task."""

    status: str = Field(default="pending", description="pending|success|failed")
    message: Optional[str] = None
    attempts: int = Field(default=0)


class CalculationTask(BaseModel):
    """Single unit of work produced by the planner."""

    task_id: str
    operation: CalculationTaskType
    description: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    output_key: str
    code: Optional[str] = None
    requires_research: bool = False
    fallback_research_query: Optional[str] = None
    expected_unit: Optional[str] = None
    tolerance: Optional[float] = None

    class Config:
        use_enum_values = True
        frozen = True


class CalculationPlan(BaseModel):
    """Ordered set of calculation tasks with dependency metadata."""

    tasks: List[CalculationTask] = Field(default_factory=list)
    dependencies: Dict[str, Set[str]] = Field(default_factory=dict)

    def ordered_tasks(self) -> List[CalculationTask]:
        """Return tasks in dependency-respecting order via simple topo sort."""
        if not self.dependencies:
            return list(self.tasks)

        dep_map: Dict[str, Set[str]] = {
            task.task_id: set(self.dependencies.get(task.task_id, set()))
            for task in self.tasks
        }
        task_lookup = {task.task_id: task for task in self.tasks}
        ready = [task_id for task_id, deps in dep_map.items() if not deps]
        ordered: List[CalculationTask] = []

        while ready:
            current = ready.pop(0)
            if current not in task_lookup:
                continue
            ordered.append(task_lookup[current])
            # Remove dependency
            for deps in dep_map.values():
                deps.discard(current)
            # Enqueue newly ready tasks
            for task_id, deps in dep_map.items():
                if task_id not in [t.task_id for t in ordered] and not deps:
                    if task_id not in ready:
                        ready.append(task_id)

        # Fallback to original ordering if topo ordering failed
        if len(ordered) != len(self.tasks):
            return list(self.tasks)
        return ordered


class CalculationEvent(BaseModel):
    """Execution event captured during pipeline run."""

    task_id: str
    status: str
    message: Optional[str] = None
    result: Optional[Any] = None
    attempts: int = 0
    needs_research: bool = False
    research_query: Optional[str] = None


class MissingDataRequest(BaseModel):
    """Request sent to Researcher when calculations need more data.
    
    This model captures what data is missing and why, enabling targeted
    research queries rather than generic "find more data" requests.
    """
    
    missing_metrics: List[str] = Field(
        ...,
        description="List of metric names that are missing"
    )
    suggested_queries: List[str] = Field(
        ...,
        description="Specific search queries to find missing data"
    )
    context: str = Field(
        ...,
        description="Explanation of why these metrics are needed"
    )
    priority: str = Field(
        default="medium",
        description="Priority level: critical, high, medium, low"
    )
    calculation_task_id: str = Field(
        ...,
        description="ID of the calculation task that needs this data"
    )
    
    @field_validator('priority')
    @classmethod
    def validate_priority(cls, v: str) -> str:
        """Ensure priority is one of allowed values."""
        allowed = {"critical", "high", "medium", "low"}
        if v not in allowed:
            raise ValueError(f"Priority must be one of {allowed}, got '{v}'")
        return v


@dataclass
class MetricSpecBundle:
    """Container produced by spec analyzer for downstream modules."""

    structural_understanding: str
    table_specifications: List[TableSpec] = field(default_factory=list)
    observation_count: int = 0
    truncated_observations: int = 0
    applied_plan_sections: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "structural_understanding": self.structural_understanding,
            "table_specifications": [spec.model_dump() for spec in self.table_specifications],
            "observation_count": self.observation_count,
            "truncated_observations": self.truncated_observations,
            "applied_plan_sections": self.applied_plan_sections or [],
        }


__all__ = [
    "CalculationTaskType",
    "CalculationProvenance",
    "CalculationValidation",
    "CalculationTask",
    "CalculationPlan",
    "CalculationEvent",
    "MissingDataRequest",
    "MetricSpecBundle",
]
