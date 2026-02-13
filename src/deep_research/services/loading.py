"""Shared SQLAlchemy eager-loading options.

This module centralizes the repeated selectinload chains used across
services to prevent N+1 query problems with relationships.

JSONB Migration (Migration 011):
CLAIM_WITH_CITATIONS_OPTIONS and EVIDENCE_WITH_SOURCE_OPTIONS have been removed.
Citation data is now stored in verification_data JSONB column and doesn't need
eager loading.

Usage:
    from deep_research.services.loading import MESSAGE_WITH_CHAT_OPTIONS

    query = select(Message).options(*MESSAGE_WITH_CHAT_OPTIONS).where(...)
"""

from sqlalchemy.orm import selectinload

from deep_research.models.chat import Chat
from deep_research.models.message import Message
from deep_research.models.research_session import ResearchSession

# Message with chat relationship for authorization checks
# Used by verify_message_ownership() in authorization.py
MESSAGE_WITH_CHAT_OPTIONS = (selectinload(Message.chat),)

# Message with research session for list responses
# Used by MessageService for message list with session metadata
MESSAGE_WITH_SESSION_OPTIONS = (selectinload(Message.research_session),)

# Full chat with messages → research sessions → sources
# Used by ChatService.get_full() and GET /chats/{id}/full endpoint
CHAT_FULL_OPTIONS = (
    selectinload(Chat.messages)
    .selectinload(Message.research_session)
    .selectinload(ResearchSession.sources),
)
