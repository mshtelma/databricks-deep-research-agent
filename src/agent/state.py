"""ResearchState model for multi-agent workflow."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


class StepType(str, Enum):
    """Type of research plan step."""

    RESEARCH = "research"  # Web search/crawl - executed by Researcher
    ANALYSIS = "analysis"  # Pure reasoning - executed by Synthesizer


class StepStatus(str, Enum):
    """Execution status of a plan step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"


class ReflectionDecision(str, Enum):
    """Decision made by Reflector agent."""

    CONTINUE = "continue"  # Proceed to next step
    ADJUST = "adjust"  # Return to Planner for replanning
    COMPLETE = "complete"  # Skip remaining steps, go to Synthesizer


@dataclass
class PlanStep:
    """A single step in a research plan."""

    id: str
    title: str
    description: str
    step_type: StepType
    needs_search: bool
    status: StepStatus = StepStatus.PENDING
    observation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "step_type": self.step_type.value,
            "needs_search": self.needs_search,
            "status": self.status.value,
            "observation": self.observation,
        }


@dataclass
class Plan:
    """A structured research plan created by Planner agent."""

    id: str
    title: str
    thought: str
    steps: list[PlanStep]
    has_enough_context: bool = False
    iteration: int = 1
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "thought": self.thought,
            "steps": [s.to_dict() for s in self.steps],
            "has_enough_context": self.has_enough_context,
            "iteration": self.iteration,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ReflectionResult:
    """Output from the Reflector agent."""

    decision: ReflectionDecision
    reasoning: str
    suggested_changes: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "decision": self.decision.value,
            "reasoning": self.reasoning,
            "suggested_changes": self.suggested_changes,
        }


@dataclass
class QueryClassification:
    """Result of analyzing a user query."""

    complexity: str  # simple, moderate, complex
    follow_up_type: str  # new_topic, clarification, complex_follow_up
    is_ambiguous: bool
    clarifying_questions: list[str] = field(default_factory=list)
    recommended_depth: str = "auto"
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "complexity": self.complexity,
            "follow_up_type": self.follow_up_type,
            "is_ambiguous": self.is_ambiguous,
            "clarifying_questions": self.clarifying_questions,
            "recommended_depth": self.recommended_depth,
            "reasoning": self.reasoning,
        }


@dataclass
class SourceInfo:
    """Information about a web source."""

    url: str
    title: str | None = None
    snippet: str | None = None
    content: str | None = None
    relevance_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "content": self.content,
            "relevance_score": self.relevance_score,
        }


class ResearchDepth(str, Enum):
    """Research depth levels controlling thoroughness."""

    AUTO = "auto"  # Automatically determined based on query complexity
    LIGHT = "light"  # 1-2 search iterations, quick answers
    MEDIUM = "medium"  # 3-5 search iterations, balanced research
    EXTENDED = "extended"  # 6-10 search iterations, thorough analysis


# Mapping from depth levels to max research steps
DEPTH_TO_STEPS: dict[str, tuple[int, int]] = {
    "light": (1, 3),  # min 1, max 3 steps
    "medium": (3, 6),  # min 3, max 6 steps
    "extended": (5, 10),  # min 5, max 10 steps
}

# Mapping from query complexity to default depth
COMPLEXITY_TO_DEPTH: dict[str, str] = {
    "simple": "light",
    "moderate": "medium",
    "complex": "extended",
}


@dataclass
class ResearchState:
    """Runtime state for multi-agent research workflow.

    Passed between agents during execution.
    """

    # Original query context
    query: str
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    session_id: UUID = field(default_factory=uuid4)

    # User preferences
    system_instructions: str | None = None  # Custom instructions from user preferences

    # Research depth configuration
    research_depth: str = "auto"  # auto, light, medium, extended
    effective_depth: str | None = None  # Resolved depth after auto selection

    # Clarification (Coordinator phase)
    enable_clarification: bool = True
    clarification_rounds: int = 0
    max_clarification_rounds: int = 3
    clarification_history: list[str] = field(default_factory=list)
    is_clarification_complete: bool = False

    # Query classification
    query_classification: QueryClassification | None = None
    is_simple_query: bool = False
    direct_response: str | None = None

    # Background investigation (pre-planning)
    background_investigation_results: str = ""

    # Planning
    current_plan: Plan | None = None
    plan_iterations: int = 0
    max_plan_iterations: int = 3

    # Step execution (Researcher phase)
    current_step_index: int = 0
    last_observation: str = ""
    all_observations: list[str] = field(default_factory=list)

    # Reflection
    last_reflection: ReflectionResult | None = None
    reflection_history: list[ReflectionResult] = field(default_factory=list)

    # Sources collected
    sources: list[SourceInfo] = field(default_factory=list)

    # Final output (Synthesizer phase)
    final_report: str = ""

    # Timing
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None

    # Cancellation support
    is_cancelled: bool = False

    def get_current_step(self) -> PlanStep | None:
        """Get the current step being executed."""
        if self.current_plan and self.current_step_index < len(self.current_plan.steps):
            return self.current_plan.steps[self.current_step_index]
        return None

    def has_more_steps(self) -> bool:
        """Check if there are more steps to execute."""
        if not self.current_plan:
            return False
        return self.current_step_index < len(self.current_plan.steps)

    def advance_step(self) -> None:
        """Advance to the next step."""
        self.current_step_index += 1

    def mark_step_complete(self, observation: str) -> None:
        """Mark current step as complete with observation."""
        step = self.get_current_step()
        if step:
            step.status = StepStatus.COMPLETED
            step.observation = observation
            self.last_observation = observation
            self.all_observations.append(observation)

    def add_source(self, source: SourceInfo) -> None:
        """Add a source to the collection."""
        # Avoid duplicates
        if not any(s.url == source.url for s in self.sources):
            self.sources.append(source)

    def complete(self, final_report: str) -> None:
        """Mark research as complete."""
        self.final_report = final_report
        self.completed_at = datetime.now(UTC)

    def cancel(self) -> None:
        """Mark research as cancelled."""
        self.is_cancelled = True
        self.completed_at = datetime.now(UTC)

    def resolve_depth(self) -> str:
        """Resolve effective research depth.

        If research_depth is 'auto', determines depth based on query complexity.
        Otherwise returns the explicitly set depth.

        Returns:
            Effective depth string (light, medium, or extended).
        """
        if self.effective_depth:
            return self.effective_depth

        if self.research_depth != "auto":
            self.effective_depth = self.research_depth
            return self.effective_depth

        # Auto-determine based on query complexity
        if self.query_classification:
            complexity = self.query_classification.complexity
            self.effective_depth = COMPLEXITY_TO_DEPTH.get(complexity, "medium")
        else:
            # Default to medium if no classification available
            self.effective_depth = "medium"

        return self.effective_depth

    def get_max_steps(self) -> int:
        """Get maximum number of research steps for current depth.

        Returns:
            Maximum number of steps to execute.
        """
        depth = self.resolve_depth()
        min_steps, max_steps = DEPTH_TO_STEPS.get(depth, (3, 6))
        return max_steps

    def get_min_steps(self) -> int:
        """Get minimum number of research steps for current depth.

        Returns:
            Minimum number of steps before early completion is allowed.
        """
        depth = self.resolve_depth()
        min_steps, max_steps = DEPTH_TO_STEPS.get(depth, (3, 6))
        return min_steps

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "session_id": str(self.session_id),
            "query_classification": self.query_classification.to_dict()
            if self.query_classification
            else None,
            "is_simple_query": self.is_simple_query,
            "current_plan": self.current_plan.to_dict() if self.current_plan else None,
            "plan_iterations": self.plan_iterations,
            "current_step_index": self.current_step_index,
            "all_observations": self.all_observations,
            "sources": [s.to_dict() for s in self.sources],
            "final_report": self.final_report,
            "is_cancelled": self.is_cancelled,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
