"""Streaming event schemas for SSE."""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import Field

from src.schemas.common import BaseSchema
from src.schemas.research import PlanStepSummary


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


class ToolCallEvent(BaseStreamEvent):
    """Tool called during ReAct research loop."""

    event_type: Literal["tool_call"] = "tool_call"
    tool_name: str  # web_search, web_crawl
    tool_args: dict[str, Any]
    call_number: int


class ToolResultEvent(BaseStreamEvent):
    """Tool execution completed."""

    event_type: Literal["tool_result"] = "tool_result"
    tool_name: str
    result_preview: str  # First 200 chars of result
    sources_crawled: int  # Total sources with content so far


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


class StreamErrorEvent(BaseStreamEvent):
    """Error during streaming."""

    event_type: Literal["error"] = "error"
    error_code: str
    error_message: str
    recoverable: bool


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
)
