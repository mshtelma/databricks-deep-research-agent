"""Claim SQLAlchemy model.

Represents an atomic factual assertion extracted from an agent response message.
Part of the claim-level citation verification pipeline.
"""

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import Boolean, CheckConstraint, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import BaseModel

if TYPE_CHECKING:
    from src.models.citation import Citation
    from src.models.citation_correction import CitationCorrection
    from src.models.message import Message
    from src.models.numeric_claim import NumericClaim


class Claim(BaseModel):
    """Atomic factual assertion extracted from agent response.

    Each claim represents a single verifiable statement that can be
    linked to evidence spans for attribution. Claims are extracted
    during the interleaved generation stage (Stage 2) of the
    citation verification pipeline.

    Attributes:
        message_id: Parent agent message containing this claim
        claim_text: The extracted claim text
        claim_type: Type of claim ('general' or 'numeric')
        confidence_level: HaluGate-style routing classification
        position_start: Start character offset in message content
        position_end: End character offset in message content
        verification_verdict: Four-tier verification result
        verification_reasoning: Explanation for the verdict
        abstained: Whether verification was abstained due to insufficient evidence
    """

    __tablename__ = "claims"

    # Foreign key to message
    message_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Claim content
    claim_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    claim_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    # Confidence classification (Stage 3)
    confidence_level: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
    )

    # Position in message
    position_start: Mapped[int] = mapped_column(
        nullable=False,
    )
    position_end: Mapped[int] = mapped_column(
        nullable=False,
    )

    # Verification results (Stage 4)
    verification_verdict: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
    )
    verification_reasoning: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # Abstention flag for insufficient evidence
    abstained: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    # Human-readable citation key for frontend mapping (primary key)
    # e.g., "Arxiv", "Zhipu", "Github-2"
    citation_key: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        doc="Primary citation key derived from source domain/title",
    )

    # All citation keys in this claim's sentence (for multi-citation sentences)
    # e.g., ["Arxiv", "Arxiv-2", "Github"] when sentence has multiple markers
    citation_keys: Mapped[list[str] | None] = mapped_column(
        ARRAY(String(50)),
        nullable=True,
        doc="All citation keys in this claim for multi-marker resolution",
    )

    # Relationships
    message: Mapped["Message"] = relationship(
        "Message",
        back_populates="claims",
    )
    citations: Mapped[list["Citation"]] = relationship(
        "Citation",
        back_populates="claim",
        cascade="all, delete-orphan",
    )
    corrections: Mapped[list["CitationCorrection"]] = relationship(
        "CitationCorrection",
        back_populates="claim",
        cascade="all, delete-orphan",
    )
    numeric_detail: Mapped["NumericClaim | None"] = relationship(
        "NumericClaim",
        back_populates="claim",
        uselist=False,
        cascade="all, delete-orphan",
    )

    # Table arguments: constraints and indexes
    __table_args__ = (
        CheckConstraint(
            "claim_type IN ('general', 'numeric')",
            name="ck_claims_type",
        ),
        CheckConstraint(
            "verification_verdict IS NULL OR "
            "verification_verdict IN ('supported', 'partial', 'unsupported', 'contradicted')",
            name="ck_claims_verdict",
        ),
        CheckConstraint(
            "confidence_level IS NULL OR "
            "confidence_level IN ('high', 'medium', 'low')",
            name="ck_claims_confidence",
        ),
        CheckConstraint(
            "position_start < position_end",
            name="ck_claims_position_order",
        ),
        Index("idx_claims_verdict", "verification_verdict"),
        Index("idx_claims_confidence", "confidence_level"),
    )

    @property
    def is_numeric(self) -> bool:
        """Check if this is a numeric claim."""
        return self.claim_type == "numeric"

    @property
    def is_verified(self) -> bool:
        """Check if this claim has been verified."""
        return self.verification_verdict is not None

    @property
    def is_supported(self) -> bool:
        """Check if this claim is fully supported."""
        return self.verification_verdict == "supported"

    @property
    def primary_citation(self) -> "Citation | None":
        """Get the primary citation for this claim."""
        for citation in self.citations:
            if citation.is_primary:
                return citation
        return self.citations[0] if self.citations else None
