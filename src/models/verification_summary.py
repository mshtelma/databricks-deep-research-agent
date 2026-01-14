"""VerificationSummary SQLAlchemy model.

Stores aggregated verification statistics for a message's claims.
Cached computation from claim verification results.
"""

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base, UUIDMixin

if TYPE_CHECKING:
    from src.models.message import Message


class VerificationSummary(Base, UUIDMixin):
    """Aggregated verification statistics for a message.

    Created after all claims in a message have been verified (Stage 4).
    Stores cached counts and rates for efficient retrieval.

    Attributes:
        message_id: The message these statistics are for
        total_claims: Total number of claims verified
        supported_count: Claims with SUPPORTED verdict
        partial_count: Claims with PARTIAL verdict
        unsupported_count: Claims with UNSUPPORTED verdict
        contradicted_count: Claims with CONTRADICTED verdict
        abstained_count: Claims where verification was abstained
        unsupported_rate: Ratio of unsupported claims (0.0-1.0)
        contradicted_rate: Ratio of contradicted claims (0.0-1.0)
        warning: True if rates exceed thresholds
        citation_corrections: Number of corrections made
        created_at: Timestamp when summary was computed
    """

    __tablename__ = "verification_summaries"

    # Foreign key to message (1:1 relationship)
    message_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    # Claim counts by verdict
    total_claims: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    supported_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    partial_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    unsupported_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    contradicted_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    abstained_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    # Computed rates
    unsupported_rate: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
    )
    contradicted_rate: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
    )

    # Warning flag (if rates exceed thresholds)
    warning: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    # Citation correction count
    citation_corrections: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationship
    message: Mapped["Message"] = relationship(
        "Message",
        back_populates="verification_summary",
    )

    @property
    def verified_count(self) -> int:
        """Get count of claims that were actually verified (not abstained)."""
        return self.total_claims - self.abstained_count

    @property
    def support_rate(self) -> float:
        """Calculate the support rate (supported + partial)."""
        if self.verified_count == 0:
            return 0.0
        return (self.supported_count + self.partial_count) / self.verified_count

    @property
    def full_support_rate(self) -> float:
        """Calculate the full support rate (supported only)."""
        if self.verified_count == 0:
            return 0.0
        return self.supported_count / self.verified_count

    def compute_warning(
        self,
        unsupported_threshold: float = 0.20,
        contradicted_threshold: float = 0.05,
    ) -> bool:
        """Check if warning thresholds are exceeded."""
        return (
            self.unsupported_rate > unsupported_threshold
            or self.contradicted_rate > contradicted_threshold
        )

    @classmethod
    def from_counts(
        cls,
        message_id: UUID,
        supported: int = 0,
        partial: int = 0,
        unsupported: int = 0,
        contradicted: int = 0,
        abstained: int = 0,
        corrections: int = 0,
        unsupported_threshold: float = 0.20,
        contradicted_threshold: float = 0.05,
    ) -> "VerificationSummary":
        """Create a summary from claim counts.

        Args:
            message_id: The message ID
            supported: Count of SUPPORTED verdicts
            partial: Count of PARTIAL verdicts
            unsupported: Count of UNSUPPORTED verdicts
            contradicted: Count of CONTRADICTED verdicts
            abstained: Count of abstained verifications
            corrections: Number of citation corrections
            unsupported_threshold: Warning threshold for unsupported rate
            contradicted_threshold: Warning threshold for contradicted rate

        Returns:
            Computed VerificationSummary instance
        """
        total = supported + partial + unsupported + contradicted + abstained
        verified = total - abstained

        unsupported_rate = unsupported / verified if verified > 0 else 0.0
        contradicted_rate = contradicted / verified if verified > 0 else 0.0

        warning = (
            unsupported_rate > unsupported_threshold
            or contradicted_rate > contradicted_threshold
        )

        return cls(
            message_id=message_id,
            total_claims=total,
            supported_count=supported,
            partial_count=partial,
            unsupported_count=unsupported,
            contradicted_count=contradicted,
            abstained_count=abstained,
            unsupported_rate=unsupported_rate,
            contradicted_rate=contradicted_rate,
            warning=warning,
            citation_corrections=corrections,
        )
