"""Message-related Pydantic schemas."""

from datetime import datetime
from uuid import UUID

from pydantic import Field

from src.models.message import MessageRole
from src.models.research_session import ResearchDepth
from src.schemas.common import BaseSchema


class MessageBase(BaseSchema):
    """Base message schema."""

    content: str = Field(..., min_length=1)


class SendMessageRequest(MessageBase):
    """Schema for sending a message."""

    research_depth: ResearchDepth = ResearchDepth.AUTO


class EditMessageRequest(BaseSchema):
    """Schema for editing a message."""

    content: str = Field(..., min_length=1)


class MessageResponse(BaseSchema):
    """Schema for message response."""

    id: UUID
    chat_id: UUID
    role: MessageRole
    content: str
    created_at: datetime
    is_edited: bool


class SendMessageResponse(BaseSchema):
    """Response after sending a message."""

    user_message: MessageResponse
    agent_message_id: UUID
    research_session_id: UUID


class EditMessageResponse(BaseSchema):
    """Response after editing a message."""

    message: MessageResponse
    removed_message_count: int


class RegenerateResponse(BaseSchema):
    """Response after regenerating a message."""

    new_message_id: UUID
    research_session_id: UUID


class MessageListResponse(BaseSchema):
    """Paginated message list response."""

    items: list[MessageResponse]
    total: int
    limit: int
    offset: int
