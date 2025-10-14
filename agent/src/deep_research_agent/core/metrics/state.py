"""State containers for the metric pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .models import MetricSpecBundle, CalculationPlan, CalculationEvent
from ..report_generation.models import CalculationContext, TableSpec


@dataclass
class MetricPipelineState:
    """Persisted state for metric pipeline execution."""

    spec_bundle: Optional[MetricSpecBundle] = None
    calculation_plan: Optional[CalculationPlan] = None
    calculation_context: Optional[CalculationContext] = None
    execution_summary: List[CalculationEvent] = field(default_factory=list)
    pending_research_queries: List[str] = field(default_factory=list)
    last_updated: Optional[datetime] = None
    
    # Iteration tracking for feedback loop
    iteration_count: int = 0
    max_iterations: int = 3

    def touch(self) -> None:
        """Update last_updated timestamp."""
        self.last_updated = datetime.utcnow()

    def add_event(self, event: CalculationEvent) -> None:
        """Add execution event to summary."""
        self.execution_summary.append(event)
        self.touch()
    
    def can_iterate(self) -> bool:
        """Check if pipeline can perform another iteration.
        
        Returns:
            True if not yet reached max iterations
        """
        return self.iteration_count < self.max_iterations
    
    def increment_iteration(self) -> None:
        """Increment iteration counter."""
        self.iteration_count += 1
        self.touch()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state for persistence in workflow state."""

        return {
            "spec_bundle": self.spec_bundle.to_dict() if self.spec_bundle else None,
            "calculation_plan": (
                self.calculation_plan.model_dump()
                if isinstance(self.calculation_plan, CalculationPlan)
                else None
            ),
            "calculation_context": (
                self.calculation_context.to_dict()
                if isinstance(self.calculation_context, CalculationContext)
                else None
            ),
            "execution_summary": [event.model_dump() for event in self.execution_summary],
            "pending_research_queries": list(self.pending_research_queries),
            "last_updated": (
                self.last_updated.isoformat() if self.last_updated else None
            ),
            "iteration_count": self.iteration_count,
            "max_iterations": self.max_iterations,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricPipelineState":
        """Rehydrate state from workflow payload."""

        spec_bundle = None
        if data.get("spec_bundle"):
            bundle = data["spec_bundle"]
            table_specs = [
                TableSpec(**spec) if not isinstance(spec, TableSpec) else spec
                for spec in bundle.get("table_specifications", [])
            ]
            spec_bundle = MetricSpecBundle(
                structural_understanding=bundle.get("structural_understanding", ""),
                table_specifications=table_specs,
                observation_count=bundle.get("observation_count", 0),
                truncated_observations=bundle.get("truncated_observations", 0),
                applied_plan_sections=bundle.get("applied_plan_sections") or [],
            )

        calculation_plan = None
        if data.get("calculation_plan"):
            calculation_plan = CalculationPlan(**data["calculation_plan"])

        calculation_context = None
        if data.get("calculation_context"):
            calculation_context = CalculationContext.from_dict(data["calculation_context"])

        execution_summary = [
            CalculationEvent(**event)
            for event in data.get("execution_summary", [])
        ]

        last_updated = None
        if data.get("last_updated"):
            try:
                last_updated = datetime.fromisoformat(data["last_updated"])
            except ValueError:
                last_updated = None

        state = cls(
            spec_bundle=spec_bundle,
            calculation_plan=calculation_plan,
            calculation_context=calculation_context,
            execution_summary=execution_summary,
            pending_research_queries=list(data.get("pending_research_queries", [])),
            last_updated=last_updated,
            iteration_count=data.get("iteration_count", 0),
            max_iterations=data.get("max_iterations", 3),
        )
        return state


__all__ = ["MetricPipelineState"]
