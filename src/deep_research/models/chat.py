"""Chat SQLAlchemy model."""

from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Index, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from deep_research.db.base import BaseModel

if TYPE_CHECKING:
    from deep_research.models.message import Message
    from deep_research.models.research_session import ResearchSession
    from deep_research.models.source import Source


class ChatStatus(str, Enum):
    """Chat status enumeration."""

    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


class Chat(BaseModel):
    """Chat conversation model.

    Represents a conversation thread between a user and the agent.
    """

    __tablename__ = "chats"

    # User identification (Databricks workspace user ID)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    # Chat metadata
    title: Mapped[str | None] = mapped_column(String(200), nullable=True)
    status: Mapped[ChatStatus] = mapped_column(
        String(20),
        default=ChatStatus.ACTIVE,
        nullable=False,
    )

    # Soft delete timestamp
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Additional metadata (JSONB for flexibility)
    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    # Relationships
    messages: Mapped[list["Message"]] = relationship(
        "Message",
        back_populates="chat",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )
    sources: Mapped[list["Source"]] = relationship(
        "Source",
        back_populates="chat",
        cascade="all, delete-orphan",
        order_by="Source.fetched_at.desc()",
    )
    research_sessions: Mapped[list["ResearchSession"]] = relationship(
        "ResearchSession",
        back_populates="chat",
        cascade="all, delete-orphan",
        order_by="ResearchSession.created_at.desc()",
    )

    # Indexes
    __table_args__ = (
        Index("idx_chats_user_status", "user_id", "status"),
        Index("idx_chats_deleted_at", "deleted_at", postgresql_where=(deleted_at.isnot(None))),
    )

    @property
    def is_deleted(self) -> bool:
        """Check if chat is soft deleted."""
        return self.deleted_at is not None

    @property
    def is_archived(self) -> bool:
        """Check if chat is archived."""
        return self.status == ChatStatus.ARCHIVED

    def soft_delete(self) -> None:
        """Mark chat as soft deleted."""
        self.status = ChatStatus.DELETED
        self.deleted_at = datetime.now(UTC)

    def restore(self) -> None:
        """Restore a soft-deleted chat."""
        self.status = ChatStatus.ACTIVE
        self.deleted_at = None

    def archive(self) -> None:
        """Archive the chat."""
        self.status = ChatStatus.ARCHIVED

    def unarchive(self) -> None:
        """Unarchive the chat."""
        self.status = ChatStatus.ACTIVE
