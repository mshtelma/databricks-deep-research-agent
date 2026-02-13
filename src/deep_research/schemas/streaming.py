"""Streaming event schemas for SSE."""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import Field

from deep_research.schemas.common import BaseSchema
from deep_research.schemas.research import PlanStepSummary


class BaseStreamEvent(BaseSchema):
    """Base class for all stream events."""

    event_type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AgentStartedEvent(BaseStreamEvent):
    """Agent started processing."""

    event_type: Literal["agent_started"] = "agent_started"
    agent: str  # coordinator, background_investigator, planner, researcher, reflector, synthesizer
    model_tier: str  # simple, analytical, complex


class AgentCompletedEvent(BaseStreamEvent):
    """Agent completed processing."""

    event_type: Literal["agent_completed"] = "agent_completed"
    agent: str
    duration_ms: int


class ClarificationNeededEvent(BaseStreamEvent):
    """Query requires clarification."""

    event_type: Literal["clarification_needed"] = "clarification_needed"
    questions: list[str]
    round: int  # 1-3


class PlanCreatedEvent(BaseStreamEvent):
    """Research plan created."""

    event_type: Literal["plan_created"] = "plan_created"
    plan_id: UUID
    title: str
    thought: str
    steps: list[PlanStepSummary]
    iteration: int


class StepStartedEvent(BaseStreamEvent):
    """Research step started."""

    event_type: Literal["step_started"] = "step_started"
    step_index: int
    step_id: str
    step_title: str
    step_type: str  # research, analysis


class StepCompletedEvent(BaseStreamEvent):
    """Research step completed."""

    event_type: Literal["step_completed"] = "step_completed"
    step_index: int
    step_id: str
    observation_summary: str
    sources_found: int
    file_sources_found: int = 0  # Sources from uploaded files


class ToolCallEvent(BaseStreamEvent):
    """Tool called during ReAct research loop."""

    event_type: Literal["tool_call"] = "tool_call"
    tool_name: str  # web_search, web_crawl, query_genie_*, search_*, ask_*
    tool_args: dict[str, Any]
    call_number: int
    source_type: str | None = None  # genie, vector_search, knowledge_assistant, web_search, web_crawl


class ToolResultEvent(BaseStreamEvent):
    """Tool execution completed."""

    event_type: Literal["tool_result"] = "tool_result"
    tool_name: str
    result_preview: str  # First 200 chars of result
    sources_crawled: int  # Total sources with content so far
    sources_added: int = 0  # Sources added by THIS specific call
    source_type: str | None = None  # genie, vector_search, knowledge_assistant, web_search, web_crawl


class ToolSkippedEvent(BaseStreamEvent):
    """Emitted when a tool is skipped due to source scope restrictions.

    Part of 008-data-source-selection feature.
    """

    event_type: Literal["tool_skipped"] = "tool_skipped"
    tool_name: str  # web_search, web_crawl, etc.
    source_type: str  # web_search, vector_search, genie, etc.
    reason: str  # Why the tool was skipped
    scope: str  # Active source scope that caused the skip
    step_index: int | None = None


class ReflectionDecisionEvent(BaseStreamEvent):
    """Reflector decision made."""

    event_type: Literal["reflection_decision"] = "reflection_decision"
    decision: str  # continue, adjust, complete
    reasoning: str
    suggested_changes: list[str] | None = None


class SynthesisStartedEvent(BaseStreamEvent):
    """Synthesis phase started."""

    event_type: Literal["synthesis_started"] = "synthesis_started"
    total_observations: int
    total_sources: int


class SynthesisProgressEvent(BaseStreamEvent):
    """Streaming content from synthesizer."""

    event_type: Literal["synthesis_progress"] = "synthesis_progress"
    content_chunk: str


class ResearchCompletedEvent(BaseStreamEvent):
    """Research completed."""

    event_type: Literal["research_completed"] = "research_completed"
    session_id: UUID
    total_steps_executed: int
    total_steps_skipped: int
    plan_iterations: int
    total_duration_ms: int
    final_report: str | None = None  # Markdown report for backwards compat
    structured_output: dict[str, Any] | None = None  # Structured JSON output


class StreamErrorEvent(BaseStreamEvent):
    """Error during streaming.

    Includes optional stack trace for debugging. Since Databricks Apps logs
    get purged, the stack trace is included in the event so the frontend
    can display it in a collapsible panel for troubleshooting.
    """

    event_type: Literal["error"] = "error"
    error_code: str
    error_message: str  # User-friendly error message
    recoverable: bool
    stack_trace: str | None = None  # Full Python traceback for debugging
    error_type: str | None = None  # Exception class name (e.g., "ValueError")


# Citation verification events
class ClaimGeneratedEvent(BaseStreamEvent):
    """Claim generated during interleaved synthesis."""

    event_type: Literal["claim_generated"] = "claim_generated"
    claim_text: str
    position_start: int
    position_end: int
    evidence_preview: str
    confidence_level: str  # "high", "medium", "low"


class ClaimVerifiedEvent(BaseStreamEvent):
    """Claim verification completed."""

    event_type: Literal["claim_verified"] = "claim_verified"
    claim_id: UUID
    claim_text: str
    position_start: int
    position_end: int
    verdict: str  # "supported", "partial", "unsupported", "contradicted"
    confidence_level: str
    evidence_preview: str
    reasoning: str | None = None
    # Citation keys for frontend citationData mapping (e.g., "Arxiv", "Zhipu")
    citation_key: str | None = None  # Primary citation key
    citation_keys: list[str] | None = None  # All keys for multi-source claims


class CitationCorrectedEvent(BaseStreamEvent):
    """Citation corrected during post-processing."""

    event_type: Literal["citation_corrected"] = "citation_corrected"
    claim_id: UUID
    correction_type: str  # "keep", "replace", "remove", "add_alternate"
    reasoning: str | None = None


class NumericClaimDetectedEvent(BaseStreamEvent):
    """Numeric claim detected with QA verification."""

    event_type: Literal["numeric_claim_detected"] = "numeric_claim_detected"
    claim_id: UUID
    raw_value: str
    normalized_value: str | None = None
    unit: str | None = None
    derivation_type: str  # "direct", "computed"
    qa_verified: bool = False


class VerificationSummaryEvent(BaseStreamEvent):
    """Verification summary for a message."""

    event_type: Literal["verification_summary"] = "verification_summary"
    message_id: UUID
    total_claims: int
    supported: int
    partial: int
    unsupported: int
    contradicted: int
    abstained_count: int
    citation_corrections: int
    warning: bool


class ResearchStartedEvent(BaseStreamEvent):
    """Emitted at the start of a research request.

    This event signals that the research has begun and provides
    pre-generated IDs for the message and research session.
    The frontend uses this to set up UI state before streaming begins.
    """

    event_type: Literal["research_started"] = "research_started"
    message_id: str  # UUID as string for JSON serialization
    research_session_id: str | None = None  # Only set for deep_research mode


class PersistenceCompletedEvent(BaseStreamEvent):
    """Emitted after successful database persistence.

    This event signals that the chat and all research data have been
    persisted to the database. For draft chats, this indicates the
    chat is now "real" and should be reflected in the UI.
    """

    event_type: Literal["persistence_completed"] = "persistence_completed"
    chat_id: str  # UUID as string for JSON serialization
    message_id: str
    research_session_id: str | None = None  # Only set for deep_research mode
    chat_title: str
    was_draft: bool  # True if chat was created, False if already existed
    counts: dict[str, int]  # Entity counts from persistence


# Custom phase execution events
class PhaseStartedEvent(BaseStreamEvent):
    """Emitted when a custom research phase starts execution."""

    event_type: Literal["phase_started"] = "phase_started"
    phase_name: str
    description: str = ""


class PhaseCompletedEvent(BaseStreamEvent):
    """Emitted when a custom research phase completes successfully."""

    event_type: Literal["phase_completed"] = "phase_completed"
    phase_name: str
    duration_ms: float = 0
    sources_count: int = 0


class PhaseSkippedEvent(BaseStreamEvent):
    """Emitted when a phase is skipped (should_run=False)."""

    event_type: Literal["phase_skipped"] = "phase_skipped"
    phase_name: str
    reason: str = "should_run returned False"


class PhaseErrorEvent(BaseStreamEvent):
    """Emitted when a custom research phase fails."""

    event_type: Literal["phase_error"] = "phase_error"
    phase_name: str
    error: str
    recoverable: bool = True


class CustomPhaseModeStartedEvent(BaseStreamEvent):
    """Emitted when research enters custom phase mode (planner disabled)."""

    event_type: Literal["custom_phase_mode_started"] = "custom_phase_mode_started"
    total_phases: int
    phase_names: list[str]


# =============================================================================
# Plan Review Events (007-enterprise-data-sources, US12, T040)
# =============================================================================


class PlanStepForReview(BaseSchema):
    """A plan step with source hints for user review.

    Extended version of PlanStepSummary that includes source routing
    information so users can see which data sources will be queried.
    """

    id: str
    title: str
    description: str
    step_type: str  # "research" or "analysis"
    needs_search: bool
    source_hints: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of source hints with name, type, priority, and query hints",
    )
    exclude_sources: list[str] = Field(
        default_factory=list,
        description="Sources explicitly excluded from this step",
    )


class PlanForReview(BaseSchema):
    """Complete research plan awaiting user review.

    Contains the full plan with source routing information so
    the user can understand and modify the research approach.
    """

    id: str
    title: str
    thought: str
    steps: list[PlanStepForReview]
    iteration: int
    data_landscape_summary: str | None = None


class PlanReviewEvent(BaseStreamEvent):
    """Emitted when plan review is enabled and plan is ready for user review.

    This event signals that the research is paused and waiting for user
    input. The frontend should display the plan for review and provide
    approve/edit/reject controls.

    Part of 007-enterprise-data-sources feature (US12, T040).
    """

    event_type: Literal["plan_review"] = "plan_review"

    plan: PlanForReview = Field(
        ...,
        description="The research plan with steps and source hints for review",
    )

    timeout_seconds: int = Field(
        default=300,
        description="Seconds until auto-proceed (if require_plan_approval=False)",
    )

    review_id: str = Field(
        ...,
        description="Unique ID for this review session (for response correlation)",
    )

    require_approval: bool = Field(
        default=False,
        description="If True, research will not proceed without explicit approval",
    )

    available_sources: list[str] = Field(
        default_factory=list,
        description="List of all available data sources that can be used",
    )


class PlanReviewResponseEvent(BaseStreamEvent):
    """User response to plan review.

    This event is sent by the frontend when the user responds to a
    PlanReviewEvent. It can approve, modify, or reject the plan.
    """

    event_type: Literal["plan_review_response"] = "plan_review_response"

    review_id: str = Field(
        ...,
        description="Review ID from the PlanReviewEvent being responded to",
    )

    action: str = Field(
        ...,
        description="User action: 'approve', 'approve_with_edits', 'reject'",
    )

    edited_plan: PlanForReview | None = Field(
        default=None,
        description="Modified plan if action is 'approve_with_edits'",
    )

    rejection_reason: str | None = Field(
        default=None,
        description="Reason if action is 'reject'",
    )


class PlanReviewTimeoutEvent(BaseStreamEvent):
    """Emitted when plan review times out and auto-proceeds.

    This event signals that the review timeout has elapsed and
    research is proceeding with the original plan.
    """

    event_type: Literal["plan_review_timeout"] = "plan_review_timeout"

    review_id: str = Field(
        ...,
        description="Review ID that timed out",
    )

    timeout_seconds: int = Field(
        ...,
        description="Timeout duration that elapsed",
    )


# Union type for all stream events
StreamEvent = (
    AgentStartedEvent
    | AgentCompletedEvent
    | ClarificationNeededEvent
    | PlanCreatedEvent
    | StepStartedEvent
    | StepCompletedEvent
    | ToolCallEvent
    | ToolResultEvent
    | ToolSkippedEvent  # 008-data-source-selection
    | ReflectionDecisionEvent
    | SynthesisStartedEvent
    | SynthesisProgressEvent
    | ResearchCompletedEvent
    | StreamErrorEvent
    # Citation verification events
    | ClaimGeneratedEvent
    | ClaimVerifiedEvent
    | CitationCorrectedEvent
    | NumericClaimDetectedEvent
    | VerificationSummaryEvent
    # Lifecycle events
    | ResearchStartedEvent
    | PersistenceCompletedEvent
    # Custom phase execution events
    | PhaseStartedEvent
    | PhaseCompletedEvent
    | PhaseSkippedEvent
    | PhaseErrorEvent
    | CustomPhaseModeStartedEvent
    # Plan review events (007-enterprise-data-sources)
    | PlanReviewEvent
    | PlanReviewResponseEvent
    | PlanReviewTimeoutEvent
)
