"""Chat-related Pydantic schemas."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import Field

from deep_research.models.chat import ChatStatus, ChatType
from deep_research.schemas.agent import SourceResponse
from deep_research.schemas.citation import ClaimResponse, VerificationSummary
from deep_research.schemas.common import BaseSchema, TimestampMixin
from deep_research.schemas.research import (
    QueryClassification,
    ReflectionStepSchema,
    ResearchDepth,
    ResearchPlan,
    ResearchStatus,
)


class ChatBase(BaseSchema):
    """Base chat schema."""

    title: str | None = Field(None, max_length=200)


class ChatCreate(ChatBase):
    """Schema for creating a chat.

    Note: Incognito chats use a separate endpoint (POST /chats/incognito).
    """

    pass


class ChatUpdate(BaseSchema):
    """Schema for updating a chat."""

    title: str | None = Field(None, max_length=200)
    status: ChatStatus | None = None


class ChatResponse(ChatBase, TimestampMixin):
    """Schema for chat response."""

    id: UUID
    status: ChatStatus
    chat_type: ChatType = ChatType.REGULAR
    message_count: int = 0


class ChatListResponse(BaseSchema):
    """Paginated chat list response."""

    items: list[ChatResponse]
    total: int
    limit: int
    offset: int


class ResearchSessionInline(BaseSchema):
    """Research session with sources, for inline inclusion in full chat response."""

    id: UUID
    query_classification: QueryClassification | None = None
    research_depth: ResearchDepth
    reasoning_steps: list[ReflectionStepSchema] = Field(default_factory=list)
    status: ResearchStatus
    current_agent: str | None = None
    plan: ResearchPlan | None = None
    current_step_index: int | None = None
    plan_iterations: int
    started_at: datetime
    completed_at: datetime | None = None
    sources: list[SourceResponse] = Field(default_factory=list)


class MessageInline(BaseSchema):
    """Message with inline research session + pre-parsed claims."""

    id: UUID
    chat_id: UUID
    role: str  # 'user' | 'agent'
    content: str
    created_at: datetime
    is_edited: bool = False
    research_session: ResearchSessionInline | None = None
    claims: list[ClaimResponse] = Field(default_factory=list)
    verification_summary: VerificationSummary | None = None
    structured_output: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Agent-surface structured-output envelope ({version, binding, "
            "data, meta}) from the post-synthesis structuring pass; None "
            "for runs without surface output slots."
        ),
    )


class ChatFullResponse(BaseSchema, TimestampMixin):
    """Complete chat payload — chat + all messages + sessions + sources + claims."""

    id: UUID
    title: str | None = None
    status: ChatStatus
    chat_type: ChatType = ChatType.REGULAR
    messages: list[MessageInline] = Field(default_factory=list)
    message_count: int = 0
    surface_state: dict[str, Any] | None = None
