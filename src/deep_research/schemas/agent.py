"""Agent-related Pydantic schemas."""

from typing import Any
from uuid import UUID

from pydantic import Field

from deep_research.models.enums import SourceType
from deep_research.models.research_session import ResearchDepth
from deep_research.schemas.common import BaseSchema


class ContextMessage(BaseSchema):
    """Context message for agent query."""

    role: str = Field(..., pattern="^(user|agent)$")
    content: str


class AgentQueryRequest(BaseSchema):
    """Request schema for agent query endpoint."""

    query: str = Field(..., min_length=1)
    conversation_id: UUID | None = None
    messages: list[ContextMessage] | None = None
    research_depth: ResearchDepth = ResearchDepth.AUTO


class QueryClassification(BaseSchema):
    """Query classification result."""

    complexity: str  # simple, moderate, complex
    follow_up_type: str  # new_topic, clarification, complex_follow_up
    is_ambiguous: bool
    clarifying_questions: list[str] = []
    reasoning: str


class SourceResponse(BaseSchema):
    """Source in agent response."""

    id: UUID
    url: str
    title: str | None = None
    snippet: str | None = None
    relevance_score: float | None = None
    source_type: str = SourceType.WEB.value
    source_metadata: dict[str, Any] | None = None


class AgentQueryResponse(BaseSchema):
    """Response schema for agent query endpoint."""

    response: str
    sources: list[SourceResponse] = []
    query_classification: QueryClassification | None = None
    research_session_id: UUID
