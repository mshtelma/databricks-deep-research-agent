"""Message SQLAlchemy model."""

from enum import Enum
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import Boolean, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import BaseModel

if TYPE_CHECKING:
    from src.models.chat import Chat
    from src.models.message_feedback import MessageFeedback
    from src.models.research_session import ResearchSession


class MessageRole(str, Enum):
    """Message role enumeration."""

    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"  # For clarifying questions from agent


class Message(BaseModel):
    """Message model.

    Represents a single exchange in a chat conversation.
    """

    __tablename__ = "messages"

    # Foreign key to chat
    chat_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("chats.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Message content
    role: Mapped[MessageRole] = mapped_column(
        String(20),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    # Edit tracking
    is_edited: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    # Additional metadata
    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    # Relationships
    chat: Mapped["Chat"] = relationship("Chat", back_populates="messages")
    research_session: Mapped["ResearchSession | None"] = relationship(
        "ResearchSession",
        back_populates="message",
        uselist=False,
    )
    feedback: Mapped["MessageFeedback | None"] = relationship(
        "MessageFeedback",
        back_populates="message",
        uselist=False,
    )

    # Indexes
    __table_args__ = (
        Index("idx_messages_chat_created", "chat_id", "created_at"),
        # Full-text search index (created via raw SQL in migration)
    )

    @property
    def is_user_message(self) -> bool:
        """Check if this is a user message."""
        return self.role == MessageRole.USER

    @property
    def is_agent_message(self) -> bool:
        """Check if this is an agent message."""
        return self.role == MessageRole.AGENT
