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
    # Extended fields for citation verification
    total_pages: int | None = None
    detected_sections: list[str] | None = None
    content_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "content": self.content,
            "relevance_score": self.relevance_score,
            "total_pages": self.total_pages,
            "detected_sections": self.detected_sections,
            "content_type": self.content_type,
        }


@dataclass
class EvidenceInfo:
    """Pre-selected evidence span for citation verification.

    Created during Stage 1 (Evidence Pre-Selection) of the citation pipeline.
    """

    source_url: str
    quote_text: str
    start_offset: int | None = None
    end_offset: int | None = None
    section_heading: str | None = None
    relevance_score: float | None = None
    has_numeric_content: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_url": self.source_url,
            "quote_text": self.quote_text,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "section_heading": self.section_heading,
            "relevance_score": self.relevance_score,
            "has_numeric_content": self.has_numeric_content,
        }


@dataclass
class ClaimInfo:
    """Atomic claim extracted from generated content.

    Created during Stage 2 (Interleaved Generation) of the citation pipeline.
    """

    claim_text: str
    claim_type: str  # "general" or "numeric"
    position_start: int
    position_end: int
    evidence: EvidenceInfo | None = None
    confidence_level: str | None = None  # "high", "medium", "low"
    verification_verdict: str | None = None  # "supported", "partial", "unsupported", "contradicted"
    verification_reasoning: str | None = None
    abstained: bool = False
    citation_key: str | None = None  # Primary key like "Arxiv", "Zhipu"
    citation_keys: list[str] | None = None  # All keys for multi-marker sentences

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "claim_text": self.claim_text,
            "claim_type": self.claim_type,
            "position_start": self.position_start,
            "position_end": self.position_end,
            "evidence": self.evidence.to_dict() if self.evidence else None,
            "confidence_level": self.confidence_level,
            "verification_verdict": self.verification_verdict,
            "verification_reasoning": self.verification_reasoning,
            "abstained": self.abstained,
            "citation_key": self.citation_key,
            "citation_keys": self.citation_keys,
        }


@dataclass
class VerificationSummaryInfo:
    """Summary of verification results for a message.

    Created after Stage 4 (Isolated Verification) completes.
    """

    total_claims: int = 0
    supported_count: int = 0
    partial_count: int = 0
    unsupported_count: int = 0
    contradicted_count: int = 0
    abstained_count: int = 0
    unsupported_rate: float = 0.0
    contradicted_rate: float = 0.0
    warning: bool = False
    citation_corrections: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_claims": self.total_claims,
            "supported_count": self.supported_count,
            "partial_count": self.partial_count,
            "unsupported_count": self.unsupported_count,
            "contradicted_count": self.contradicted_count,
            "abstained_count": self.abstained_count,
            "unsupported_rate": self.unsupported_rate,
            "contradicted_rate": self.contradicted_rate,
            "warning": self.warning,
            "citation_corrections": self.citation_corrections,
        }


class ResearchDepth(str, Enum):
    """Research depth levels controlling thoroughness."""

    AUTO = "auto"  # Automatically determined based on query complexity
    LIGHT = "light"  # 1-2 search iterations, quick answers
    MEDIUM = "medium"  # 3-5 search iterations, balanced research
    EXTENDED = "extended"  # 6-10 search iterations, thorough analysis


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

    # Citation verification (6-stage pipeline)
    evidence_pool: list[EvidenceInfo] = field(default_factory=list)  # Stage 1 output
    claims: list[ClaimInfo] = field(default_factory=list)  # Stage 2-4 output
    verification_summary: VerificationSummaryInfo | None = None  # Post Stage 4
    enable_citation_verification: bool = True  # Feature toggle

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

    def get_completed_steps(self) -> list[PlanStep]:
        """Get list of completed steps from current plan.

        Returns:
            List of PlanStep objects with status COMPLETED.
        """
        if not self.current_plan:
            return []
        return [s for s in self.current_plan.steps if s.status == StepStatus.COMPLETED]

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

        Uses centralized research_types configuration from app.yaml.

        Returns:
            Maximum number of steps to execute.
        """
        from src.agent.config import get_step_limits

        depth = self.resolve_depth()
        step_limits = get_step_limits(depth)
        return step_limits.max

    def get_min_steps(self) -> int:
        """Get minimum number of research steps for current depth.

        Uses centralized research_types configuration from app.yaml.

        Returns:
            Minimum number of steps before early completion is allowed.
        """
        from src.agent.config import get_step_limits

        depth = self.resolve_depth()
        step_limits = get_step_limits(depth)
        return step_limits.min

    def add_evidence(self, evidence: EvidenceInfo) -> None:
        """Add an evidence span to the pool."""
        self.evidence_pool.append(evidence)

    def add_claim(self, claim: ClaimInfo) -> None:
        """Add a claim to the claims list."""
        self.claims.append(claim)

    def update_verification_summary(self) -> None:
        """Update verification summary from current claims."""
        if not self.claims:
            self.verification_summary = None
            return

        supported = sum(1 for c in self.claims if c.verification_verdict == "supported")
        partial = sum(1 for c in self.claims if c.verification_verdict == "partial")
        unsupported = sum(
            1 for c in self.claims if c.verification_verdict == "unsupported"
        )
        contradicted = sum(
            1 for c in self.claims if c.verification_verdict == "contradicted"
        )
        abstained = sum(1 for c in self.claims if c.abstained)

        total = len(self.claims)
        verified = total - abstained

        self.verification_summary = VerificationSummaryInfo(
            total_claims=total,
            supported_count=supported,
            partial_count=partial,
            unsupported_count=unsupported,
            contradicted_count=contradicted,
            abstained_count=abstained,
            unsupported_rate=unsupported / verified if verified > 0 else 0.0,
            contradicted_rate=contradicted / verified if verified > 0 else 0.0,
            warning=(unsupported / verified > 0.20 if verified > 0 else False)
            or (contradicted / verified > 0.05 if verified > 0 else False),
        )

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
            "evidence_pool": [e.to_dict() for e in self.evidence_pool],
            "claims": [c.to_dict() for c in self.claims],
            "verification_summary": self.verification_summary.to_dict()
            if self.verification_summary
            else None,
            "enable_citation_verification": self.enable_citation_verification,
            "final_report": self.final_report,
            "is_cancelled": self.is_cancelled,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
