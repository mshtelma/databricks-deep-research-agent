"""CitationCorrection SQLAlchemy model.

Tracks corrections made to citations during the CiteFix post-processing
stage (Stage 5) of the citation verification pipeline.
"""

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, String, Text, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from deep_research.db.base import Base, UUIDMixin

if TYPE_CHECKING:
    from deep_research.models.claim import Claim
    from deep_research.models.evidence_span import EvidenceSpan


class CitationCorrection(Base, UUIDMixin):
    """Tracks corrections made to citations during post-processing.

    Created during Stage 5 (Citation Correction) when a citation is
    found to be incorrect and needs to be fixed. Based on the CiteFix
    research pattern.

    Correction Types:
        - keep: Original citation is correct (~60% of cases)
        - replace: Found better citation from pool (~25% of cases)
        - remove: No valid citation exists (~10% of cases)
        - add_alternate: Multiple valid citations available (~5% of cases)

    Attributes:
        claim_id: The claim whose citation was corrected
        original_evidence_span_id: Original (incorrect) citation
        corrected_evidence_span_id: Corrected citation (NULL if removed)
        correction_type: Type of correction applied
        reasoning: Explanation for the correction
        created_at: Timestamp when correction was made
    """

    __tablename__ = "citation_corrections"

    # Foreign key to claim
    claim_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("claims.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Original evidence (may be NULL for 'add_alternate')
    original_evidence_span_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("evidence_spans.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Corrected evidence (NULL for 'remove')
    corrected_evidence_span_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("evidence_spans.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Correction type
    correction_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    # Explanation for the correction
    reasoning: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
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
        back_populates="corrections",
    )
    original_evidence: Mapped["EvidenceSpan | None"] = relationship(
        "EvidenceSpan",
        foreign_keys=[original_evidence_span_id],
    )
    corrected_evidence: Mapped["EvidenceSpan | None"] = relationship(
        "EvidenceSpan",
        foreign_keys=[corrected_evidence_span_id],
    )

    # Constraints and indexes
    __table_args__ = (
        CheckConstraint(
            "correction_type IN ('keep', 'replace', 'remove', 'add_alternate')",
            name="ck_citation_corrections_type",
        ),
        Index("idx_citation_corrections_type", "correction_type"),
    )

    @property
    def is_replacement(self) -> bool:
        """Check if this was a replacement correction."""
        return self.correction_type == "replace"

    @property
    def is_removal(self) -> bool:
        """Check if this was a removal correction."""
        return self.correction_type == "remove"

    @property
    def kept_original(self) -> bool:
        """Check if the original citation was kept."""
        return self.correction_type == "keep"
