"""Source SQLAlchemy model."""

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from deep_research.db.base import Base, UUIDMixin
from deep_research.models.enums import SourceType

if TYPE_CHECKING:
    from deep_research.models.chat import Chat
    from deep_research.models.evidence_span import EvidenceSpan
    from deep_research.models.research_session import ResearchSession


class Source(Base, UUIDMixin):
    """Source model.

    Represents a web resource referenced in research.
    """

    __tablename__ = "sources"

    # Foreign key to research session (matches migration column name)
    research_session_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("research_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Direct chat reference for chat-level source pool queries
    # Enables O(1) lookup of all sources in a chat without 4-table join
    # Nullable for backward compatibility with existing data
    chat_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("chats.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    # Source URL and metadata
    url: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        index=True,
    )
    title: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )
    snippet: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # Content (truncated to fit in context) - matches migration column name
    content: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # Relevance scoring
    relevance_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    # Fetch timestamp
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Source location metadata (for citation UX)
    total_pages: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    detected_sections: Mapped[str | None] = mapped_column(
        Text,  # JSON string of section headings
        nullable=True,
    )
    content_type: Mapped[str | None] = mapped_column(
        String(50),  # pdf, html, text, etc.
        nullable=True,
    )

    # Multi-source citation attribution (US7)
    source_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default=SourceType.WEB.value,
        server_default="web",
    )
    source_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        default=None,
    )

    # Tiered query modes: source tracking fields
    is_cited: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
    )
    step_index: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    step_title: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    crawl_status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="success",
        server_default="success",
    )
    error_reason: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # Relationships
    session: Mapped["ResearchSession"] = relationship(
        "ResearchSession",
        back_populates="sources",
    )
    chat: Mapped["Chat | None"] = relationship(
        "Chat",
        back_populates="sources",
    )
    evidence_spans: Mapped[list["EvidenceSpan"]] = relationship(
        "EvidenceSpan",
        back_populates="source",
        cascade="all, delete-orphan",
    )

    def set_content(self, content: str | None = None) -> None:
        """Set fetched content with truncation."""
        if content:
            # Truncate to max 50,000 characters
            self.content = content[:50000] if len(content) > 50000 else content
        self.fetched_at = datetime.now(UTC)
