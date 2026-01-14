"""EvidenceSpan SQLAlchemy model.

Represents a minimal text passage from a source document that supports
one or more claims. Part of the claim-level citation verification pipeline.
"""

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, String, Text, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base, UUIDMixin

if TYPE_CHECKING:
    from src.models.citation import Citation
    from src.models.source import Source


class EvidenceSpan(Base, UUIDMixin):
    """Minimal text passage from source supporting claims.

    Evidence spans are extracted during Stage 1 (Evidence Pre-Selection)
    of the citation verification pipeline. Each span represents a
    citable quote from a source document.

    Attributes:
        source_id: Parent source document
        quote_text: The exact supporting quote
        start_offset: Start position in source content
        end_offset: End position in source content
        section_heading: Section/page context for location
        relevance_score: Ranking score for evidence relevance (0.0-1.0)
        has_numeric_content: Whether span contains numeric data
        created_at: Timestamp when span was extracted
    """

    __tablename__ = "evidence_spans"

    # Foreign key to source
    source_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("sources.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Quote content
    quote_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    # Position in source (optional)
    start_offset: Mapped[int | None] = mapped_column(
        nullable=True,
    )
    end_offset: Mapped[int | None] = mapped_column(
        nullable=True,
    )

    # Location context
    section_heading: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    # Relevance scoring (Stage 1)
    relevance_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    # Numeric content flag (for Stage 6 routing)
    has_numeric_content: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    source: Mapped["Source"] = relationship(
        "Source",
        back_populates="evidence_spans",
    )
    citations: Mapped[list["Citation"]] = relationship(
        "Citation",
        back_populates="evidence_span",
        cascade="all, delete-orphan",
    )

    # Indexes
    __table_args__ = (Index("idx_evidence_spans_relevance", "relevance_score"),)

    @property
    def has_position(self) -> bool:
        """Check if this span has position information."""
        return self.start_offset is not None and self.end_offset is not None

    @property
    def length(self) -> int | None:
        """Get the length of this span in characters."""
        if self.has_position:
            return self.end_offset - self.start_offset  # type: ignore[operator]
        return len(self.quote_text)

    def truncate_quote(self, max_length: int = 200) -> str:
        """Get a truncated version of the quote for display."""
        if len(self.quote_text) <= max_length:
            return self.quote_text
        return self.quote_text[: max_length - 3] + "..."
