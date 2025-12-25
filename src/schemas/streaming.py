"""Streaming event schemas for SSE."""

from datetime import datetime
from typing import Literal
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


# Union type for all stream events
StreamEvent = (
    AgentStartedEvent
    | AgentCompletedEvent
    | ClarificationNeededEvent
    | PlanCreatedEvent
    | StepStartedEvent
    | StepCompletedEvent
    | ReflectionDecisionEvent
    | SynthesisStartedEvent
    | SynthesisProgressEvent
    | ResearchCompletedEvent
    | StreamErrorEvent
)
