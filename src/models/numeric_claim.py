"""NumericClaim SQLAlchemy model.

Extended metadata for numeric claims requiring special provenance tracking.
1:1 extension of the Claim model for claims with claim_type='numeric'.
"""

from decimal import Decimal
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import CheckConstraint, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Numeric

from src.db.base import Base

if TYPE_CHECKING:
    from src.models.claim import Claim


class NumericClaim(Base):
    """Extended metadata for numeric claims.

    Numeric claims require special handling for provenance tracking,
    including normalization, derivation tracking, and QA-based
    verification (Stage 6 of the citation verification pipeline).

    Attributes:
        claim_id: Parent claim (1:1 extension)
        raw_value: The value as stated in source
        normalized_value: Standardized numeric value
        unit: Unit of measurement
        entity_reference: The entity the value describes
        derivation_type: How the value was obtained ('direct' or 'computed')
        computation_details: Calculation steps if derived
        assumptions: Applied assumptions (currency year, exchange rate, etc.)
        qa_verification: QAFactEval verification results
    """

    __tablename__ = "numeric_claims"

    # Primary key is also foreign key to claims (1:1 extension)
    claim_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("claims.id", ondelete="CASCADE"),
        primary_key=True,
    )

    # Raw value as stated
    raw_value: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    # Normalized value
    normalized_value: Mapped[Decimal | None] = mapped_column(
        Numeric,
        nullable=True,
    )

    # Unit of measurement
    unit: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    # Entity reference (what the number describes)
    entity_reference: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # Derivation type
    derivation_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    # Computation details (JSONB for structured data)
    computation_details: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    # Assumptions applied (currency year, exchange rate, rounding, etc.)
    assumptions: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    # QA verification results (Stage 6)
    qa_verification: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    # Relationship back to claim
    claim: Mapped["Claim"] = relationship(
        "Claim",
        back_populates="numeric_detail",
    )

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "derivation_type IN ('direct', 'computed')",
            name="ck_numeric_claims_derivation",
        ),
    )

    @property
    def is_derived(self) -> bool:
        """Check if this is a derived (computed) value."""
        return self.derivation_type == "computed"

    @property
    def is_direct(self) -> bool:
        """Check if this is a direct quote."""
        return self.derivation_type == "direct"

    @property
    def has_qa_verification(self) -> bool:
        """Check if QA verification was performed."""
        return self.qa_verification is not None

    @property
    def qa_verified_match(self) -> bool | None:
        """Check if QA verification matched (if performed)."""
        if self.qa_verification is None:
            return None
        return self.qa_verification.get("overall_match", False)

    def format_value(self) -> str:
        """Format the value with unit for display."""
        if self.normalized_value is not None and self.unit:
            return f"{self.normalized_value} {self.unit}"
        return self.raw_value
