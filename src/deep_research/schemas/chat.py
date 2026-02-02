"""Chat-related Pydantic schemas."""

from uuid import UUID

from pydantic import Field

from deep_research.models.chat import ChatStatus, ChatType
from deep_research.schemas.common import BaseSchema, TimestampMixin


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
