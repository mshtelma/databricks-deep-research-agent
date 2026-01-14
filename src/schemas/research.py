"""Research-related Pydantic schemas."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from src.models.research_session import ResearchDepth, ResearchStatus
from src.schemas.agent import QueryClassification, SourceResponse
from src.schemas.common import BaseSchema


class ReflectionStepSchema(BaseSchema):
    """Reflection result from Reflector agent.

    Stored in research_session.reasoning_steps JSONB column.
    """

    decision: str  # "continue", "adjust", "complete"
    reasoning: str
    suggested_changes: list[str] | None = None


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

    id: str  # Plan ID like "plan-iteration-1", not UUID
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
    reasoning_steps: list[ReflectionStepSchema] = []
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
