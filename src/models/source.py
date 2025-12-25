"""Source SQLAlchemy model."""

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import DateTime, Float, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base, UUIDMixin

if TYPE_CHECKING:
    from src.models.research_session import ResearchSession


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

    # Relationship
    session: Mapped["ResearchSession"] = relationship(
        "ResearchSession",
        back_populates="sources",
    )

    def set_content(self, content: str | None = None) -> None:
        """Set fetched content with truncation."""
        if content:
            # Truncate to max 50,000 characters
            self.content = content[:50000] if len(content) > 50000 else content
        self.fetched_at = datetime.now(UTC)
