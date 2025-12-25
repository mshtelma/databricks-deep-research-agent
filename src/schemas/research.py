"""Research-related Pydantic schemas."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from src.models.research_session import ResearchDepth, ResearchStatus
from src.schemas.agent import QueryClassification, SourceResponse
from src.schemas.common import BaseSchema


class ReasoningStep(BaseSchema):
    """A single reasoning step."""

    step_number: int
    action: str  # search, fetch, reflect, synthesize
    input_summary: str
    output_summary: str
    timestamp: datetime


class PlanStepSummary(BaseSchema):
    """Summary of a plan step."""

    id: str
    title: str
    step_type: str  # research, analysis
    needs_search: bool


class ResearchPlanStep(PlanStepSummary):
    """Full plan step with status and observation."""

    description: str
    status: str  # pending, in_progress, completed, skipped
    observation: str | None = None


class ResearchPlan(BaseSchema):
    """Research plan structure."""

    id: UUID
    title: str
    thought: str
    steps: list[ResearchPlanStep]
    iteration: int
    created_at: datetime


class ResearchSession(BaseSchema):
    """Research session details."""

    id: UUID
    query_classification: QueryClassification | None = None
    research_depth: ResearchDepth
    reasoning_steps: list[ReasoningStep] = []
    status: ResearchStatus
    current_agent: str | None = None
    plan: ResearchPlan | None = None
    current_step_index: int | None = None
    plan_iterations: int
    started_at: datetime
    completed_at: datetime | None = None
    sources: list[SourceResponse] = []


class CancelResearchResponse(BaseSchema):
    """Response after cancelling research."""

    session_id: UUID
    status: Literal["cancelled"] = "cancelled"
    partial_results: str | None = None
