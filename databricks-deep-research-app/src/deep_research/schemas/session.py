"""Incognito session Pydantic schemas."""

from datetime import datetime

from deep_research.schemas.common import BaseSchema


class IncognitoSessionStatus(BaseSchema):
    """Status of the user's incognito session.

    Used by the frontend to display session info and enforce
    the chat quota client-side before server validation.
    """

    has_session: bool
    chat_count: int
    max_chats: int = 5
    expires_at: datetime | None = None


class IncognitoChatListResponse(BaseSchema):
    """Response for listing incognito chats."""

    items: list["IncognitoChatResponse"]
    total: int
    session_expires_at: datetime | None = None


# Import ChatResponse to avoid circular imports
from deep_research.schemas.chat import ChatResponse  # noqa: E402


class IncognitoChatResponse(ChatResponse):
    """Response schema for incognito chat.

    Extends ChatResponse with incognito-specific fields.
    """

    pass
