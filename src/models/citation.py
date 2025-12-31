"""Citation SQLAlchemy model.

Represents a link between a claim and supporting evidence.
Many-to-many join table with additional attribution attributes.
"""

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base, UUIDMixin

if TYPE_CHECKING:
    from src.models.claim import Claim
    from src.models.evidence_span import EvidenceSpan


class Citation(Base, UUIDMixin):
    """Link between claim and supporting evidence.

    Citations are created during Stage 2 (Interleaved Generation) when
    claims are generated with evidence constraints. Each citation links
    a claim to an evidence span with a confidence score.

    Attributes:
        claim_id: The claim being cited
        evidence_span_id: The supporting evidence
        confidence_score: Confidence in the attribution (0.0-1.0)
        is_primary: Whether this is the primary citation (vs alternate)
        created_at: Timestamp when citation was created
    """

    __tablename__ = "citations"

    # Foreign key to claim
    claim_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("claims.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Foreign key to evidence span
    evidence_span_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("evidence_spans.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Confidence in the attribution
    confidence_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    # Primary vs alternate citation flag
    is_primary: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    claim: Mapped["Claim"] = relationship(
        "Claim",
        back_populates="citations",
    )
    evidence_span: Mapped["EvidenceSpan"] = relationship(
        "EvidenceSpan",
        back_populates="citations",
    )

    # Constraints
    __table_args__ = (
        UniqueConstraint("claim_id", "evidence_span_id", name="uq_claim_evidence"),
        CheckConstraint(
            "confidence_score IS NULL OR (confidence_score >= 0.0 AND confidence_score <= 1.0)",
            name="ck_citations_confidence_range",
        ),
    )

    @property
    def has_confidence(self) -> bool:
        """Check if this citation has a confidence score."""
        return self.confidence_score is not None

    @property
    def is_high_confidence(self) -> bool:
        """Check if this is a high-confidence citation (>0.85)."""
        return self.confidence_score is not None and self.confidence_score > 0.85
