"""Plan schemas with source-aware step hints.

Extends the base PlanStep concept with source routing information
to enable intelligent data source selection per step.

Part of 007-enterprise-data-sources feature (T036).
"""

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


class StepSourceHint(BaseModel):
    """Hint for which data source to use for a research step.

    Guides the researcher on which sources to prioritize for this step.
    """

    source_name: str
    """Name of the recommended data source."""

    source_type: str
    """Type of source (vector_search, genie, web_search, etc.)."""

    priority: int = Field(default=2, ge=1, le=3)
    """Priority level: 1=must use, 2=should use, 3=may use."""

    query_hint: str | None = None
    """Suggested query to use with this source."""

    query_strategy: str | None = None
    """Rewrite strategy override for this step+source: multi_query | query2doc | schema_aware | step_back."""

    filters: dict[str, Any] | None = None
    """Suggested filters to apply (for VS sources)."""

    reasoning: str | None = None
    """Why this source is recommended for this step."""

    class Config:
        """Pydantic configuration."""

        use_enum_values = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_name": self.source_name,
            "source_type": self.source_type,
            "priority": self.priority,
            "query_hint": self.query_hint,
            "query_strategy": self.query_strategy,
            "filters": self.filters,
            "reasoning": self.reasoning,
        }


@dataclass
class PlanStepWithSources:
    """Research plan step with source routing hints.

    Extends the basic step with guidance on which data sources
    should be used during execution.
    """

    id: str
    """Unique step identifier."""

    title: str
    """Brief title for the step."""

    description: str
    """Detailed description of what to research."""

    step_type: str = "research"
    """Type of step: research or analysis."""

    needs_search: bool = True
    """Whether this step requires searching data sources."""

    source_hints: list[StepSourceHint] = field(default_factory=list)
    """Recommended data sources with priorities."""

    exclude_sources: list[str] = field(default_factory=list)
    """Sources that should NOT be used for this step."""

    require_all_sources: bool = False
    """If True, step is incomplete until all priority=1 sources consulted."""

    status: str = "pending"
    """Execution status: pending, in_progress, completed, skipped."""

    observation: str | None = None
    """Results/findings from executing this step."""

    def get_required_sources(self) -> list[StepSourceHint]:
        """Get sources with priority=1 (must use)."""
        return [h for h in self.source_hints if h.priority == 1]

    def get_recommended_sources(self) -> list[StepSourceHint]:
        """Get sources with priority=2 (should use)."""
        return [h for h in self.source_hints if h.priority == 2]

    def get_optional_sources(self) -> list[StepSourceHint]:
        """Get sources with priority=3 (may use)."""
        return [h for h in self.source_hints if h.priority == 3]

    def is_source_allowed(self, source_name: str) -> bool:
        """Check if a source is allowed for this step."""
        if source_name in self.exclude_sources:
            return False
        # If no hints, all sources are allowed
        if not self.source_hints:
            return True
        # If hints exist, only hinted sources are allowed
        return any(h.source_name == source_name for h in self.source_hints)

    def get_query_hint_for_source(self, source_name: str) -> str | None:
        """Get the query hint for a specific source."""
        for hint in self.source_hints:
            if hint.source_name == source_name:
                return hint.query_hint
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "step_type": self.step_type,
            "needs_search": self.needs_search,
            "source_hints": [h.to_dict() for h in self.source_hints],
            "exclude_sources": self.exclude_sources,
            "require_all_sources": self.require_all_sources,
            "status": self.status,
            "observation": self.observation,
        }


@dataclass
class SourceAwarePlan:
    """Research plan with source routing information.

    Contains steps with source hints based on the data landscape.
    """

    id: str
    """Unique plan identifier."""

    title: str
    """Plan title summarizing the research."""

    thought: str
    """Planner's reasoning about the approach."""

    steps: list[PlanStepWithSources] = field(default_factory=list)
    """Plan steps with source hints."""

    data_landscape_summary: str | None = None
    """Summary of available data sources used for planning."""

    has_enough_context: bool = False
    """Whether enough context is available to skip some steps."""

    iteration: int = 1
    """Planning iteration number."""

    def get_steps_by_source(self, source_name: str) -> list[PlanStepWithSources]:
        """Get all steps that hint at a specific source."""
        return [
            s for s in self.steps
            if any(h.source_name == source_name for h in s.source_hints)
        ]

    def get_enterprise_steps(self) -> list[PlanStepWithSources]:
        """Get steps that use enterprise sources."""
        enterprise_types = {"vector_search", "genie", "knowledge_assistant"}
        return [
            s for s in self.steps
            if any(h.source_type in enterprise_types for h in s.source_hints)
        ]

    def get_web_steps(self) -> list[PlanStepWithSources]:
        """Get steps that use web search."""
        return [
            s for s in self.steps
            if any(h.source_type == "web_search" for h in s.source_hints)
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "thought": self.thought,
            "steps": [s.to_dict() for s in self.steps],
            "data_landscape_summary": self.data_landscape_summary,
            "has_enough_context": self.has_enough_context,
            "iteration": self.iteration,
        }


def create_source_hint_from_discovery(
    discovery_result: Any,
    priority: int = 2,
    query_hint: str | None = None,
) -> StepSourceHint:
    """Create a StepSourceHint from a SourceDiscoveryResult.

    Helper function to convert discovery results to source hints.

    Args:
        discovery_result: SourceDiscoveryResult from data landscape.
        priority: Priority level (1-3).
        query_hint: Optional query hint override.

    Returns:
        StepSourceHint for the discovered source.
    """
    return StepSourceHint(
        source_name=discovery_result.source_name,
        source_type=discovery_result.source_type,
        priority=priority,
        query_hint=query_hint or (
            discovery_result.suggested_queries[0]
            if discovery_result.suggested_queries
            else None
        ),
        reasoning=f"Relevance score: {discovery_result.relevance_score:.2f}",
    )
