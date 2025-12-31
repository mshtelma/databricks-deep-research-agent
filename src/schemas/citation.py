"""Pydantic schemas for citation verification API.

This module defines all request/response schemas for the claim-level
citation verification feature, including:
- Claim responses with verification status
- Evidence span responses with source metadata
- Citation correction responses
- Numeric claim details with QA verification
- Verification summaries and metrics
- Provenance export formats
"""

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import Field

from src.schemas.common import BaseSchema


# Enums for API layer (mirror database enums)
class ClaimTypeEnum(str, Enum):
    """Type of claim."""

    GENERAL = "general"
    NUMERIC = "numeric"


class VerificationVerdictEnum(str, Enum):
    """Four-tier verification verdict."""

    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"


class ConfidenceLevelEnum(str, Enum):
    """HaluGate-style confidence level."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CorrectionTypeEnum(str, Enum):
    """Citation correction type."""

    KEEP = "keep"
    REPLACE = "replace"
    REMOVE = "remove"
    ADD_ALTERNATE = "add_alternate"


class DerivationTypeEnum(str, Enum):
    """How a numeric value was obtained."""

    DIRECT = "direct"
    COMPUTED = "computed"


# Source metadata (denormalized for evidence cards)
class SourceMetadataResponse(BaseSchema):
    """Source metadata for evidence cards."""

    id: UUID
    title: str | None = None
    url: str | None = None
    author: str | None = None
    published_date: str | None = None
    content_type: str | None = None
    total_pages: int | None = None


# Evidence span response
class EvidenceSpanResponse(BaseSchema):
    """Evidence span from a source document."""

    id: UUID
    source_id: UUID
    quote_text: str
    start_offset: int | None = None
    end_offset: int | None = None
    section_heading: str | None = None
    relevance_score: float | None = None
    has_numeric_content: bool = False
    # Denormalized source metadata for convenience
    source: SourceMetadataResponse | None = None


# Citation response
class CitationResponse(BaseSchema):
    """Citation link between claim and evidence."""

    evidence_span: EvidenceSpanResponse
    confidence_score: float | None = None
    is_primary: bool = True


# Citation correction response
class CitationCorrectionResponse(BaseSchema):
    """Citation correction record."""

    id: UUID
    correction_type: CorrectionTypeEnum
    original_evidence: EvidenceSpanResponse | None = None
    corrected_evidence: EvidenceSpanResponse | None = None
    reasoning: str | None = None


# QA verification result for numeric claims
class QAVerificationResult(BaseSchema):
    """QA-based verification result for numeric claims."""

    question: str
    claim_answer: str
    evidence_answer: str
    match: bool
    normalized_comparison: dict[str, float] | None = None


# Computation step for derived values
class ComputationStep(BaseSchema):
    """Single computation step for derived numeric values."""

    operation: str
    inputs: list[dict] = Field(default_factory=list)
    result: float


# Numeric claim detail
class NumericClaimDetail(BaseSchema):
    """Extended details for numeric claims."""

    raw_value: str
    normalized_value: float | None = None
    unit: str | None = None
    entity_reference: str | None = None
    derivation_type: DerivationTypeEnum
    computation_details: dict | None = None
    assumptions: dict | None = None
    qa_verification: list[QAVerificationResult] | None = None


# Full claim response
class ClaimResponse(BaseSchema):
    """Full claim with verification and citation data."""

    id: UUID
    claim_text: str
    claim_type: ClaimTypeEnum
    confidence_level: ConfidenceLevelEnum | None = None
    position_start: int
    position_end: int
    verification_verdict: VerificationVerdictEnum | None = None
    verification_reasoning: str | None = None
    abstained: bool = False
    citations: list[CitationResponse] = Field(default_factory=list)
    corrections: list[CitationCorrectionResponse] = Field(default_factory=list)
    numeric_detail: NumericClaimDetail | None = None
    # Primary citation key for frontend mapping (e.g., "Arxiv", "Zhipu")
    citation_key: str | None = None
    # All citation keys in this claim for multi-marker sentences
    citation_keys: list[str] | None = None


# Verification summary
class VerificationSummary(BaseSchema):
    """Verification statistics for a message."""

    total_claims: int
    supported_count: int
    partial_count: int
    unsupported_count: int
    contradicted_count: int = 0
    abstained_count: int = 0
    unsupported_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Percentage of claims that are unsupported (0.0-1.0)",
    )
    contradicted_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Percentage of claims that are contradicted (0.0-1.0)",
    )
    warning: bool = Field(
        description="True if unsupported_rate > 0.20 or contradicted_rate > 0.05",
    )


# Correction metrics
class CorrectionMetrics(BaseSchema):
    """Citation correction statistics for a message."""

    total_corrections: int
    keep_count: int
    replace_count: int
    remove_count: int
    add_alternate_count: int
    correction_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Percentage of claims that needed correction (0.0-1.0)",
    )


# Response for GET /messages/{id}/claims
class MessageClaimsResponse(BaseSchema):
    """Response containing all claims for a message."""

    message_id: UUID
    claims: list[ClaimResponse]
    verification_summary: VerificationSummary
    correction_metrics: CorrectionMetrics | None = None


# Response for GET /claims/{id}/evidence
class ClaimEvidenceResponse(BaseSchema):
    """Response containing evidence for a specific claim."""

    claim_id: UUID
    claim_text: str
    verification_verdict: VerificationVerdictEnum | None = None
    citations: list[CitationResponse]


# Provenance export format
class ClaimProvenanceExport(BaseSchema):
    """Single claim in provenance export format."""

    claim_text: str
    claim_type: ClaimTypeEnum
    verdict: VerificationVerdictEnum | None = None
    confidence_level: ConfidenceLevelEnum | None = None
    citations: list[dict] = Field(default_factory=list)
    numeric_detail: NumericClaimDetail | None = None
    corrections: list[dict] = Field(default_factory=list)


class ProvenanceExport(BaseSchema):
    """Full provenance export for a message."""

    exported_at: datetime
    message_id: UUID
    claims: list[ClaimProvenanceExport]
    summary: VerificationSummary
