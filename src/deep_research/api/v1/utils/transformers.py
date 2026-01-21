"""Shared response transformers for API endpoints.

JSONB Migration (Migration 011):
This module provides JSONB-to-response transformers that read from the
verification_data JSONB column on research_sessions.

Legacy model-based transformers have been removed. All citation/claim data
is now read from JSONB and transformed to API response schemas.
"""

from typing import Any
from uuid import UUID, uuid5, NAMESPACE_DNS

from deep_research.schemas.citation import (
    CitationResponse,
    ClaimResponse,
    ClaimTypeEnum,
    ConfidenceLevelEnum,
    EvidenceSpanResponse,
    SourceMetadataResponse,
    VerificationSummary,
    VerificationVerdictEnum,
)


# Namespace for deterministic claim UUIDs
CLAIM_UUID_NAMESPACE = NAMESPACE_DNS


def build_empty_verification_summary() -> VerificationSummary:
    """Build an empty verification summary for pending/missing claims.

    Used when message doesn't exist yet (during streaming) or has no claims.

    Returns:
        VerificationSummary with all zeros.
    """
    return VerificationSummary(
        total_claims=0,
        supported_count=0,
        partial_count=0,
        unsupported_count=0,
        contradicted_count=0,
        abstained_count=0,
        unsupported_rate=0.0,
        contradicted_rate=0.0,
        warning=False,
    )


# =============================================================================
# JSONB-to-Response Transformers
# =============================================================================
#
# These functions transform JSONB data from the verification_data column
# into API response schemas. They generate deterministic UUIDs from position
# data to ensure stable claim IDs without needing to store UUIDs in JSONB.


def generate_claim_uuid(message_id: UUID, position_start: int, position_end: int) -> UUID:
    """Generate deterministic UUID for a claim based on position.

    This ensures the same claim always gets the same UUID across API calls,
    without needing to store UUIDs in JSONB.

    Args:
        message_id: The message ID this claim belongs to.
        position_start: Start character position in the message.
        position_end: End character position in the message.

    Returns:
        Deterministic UUID based on position.
    """
    name = f"{message_id}:{position_start}:{position_end}"
    return uuid5(CLAIM_UUID_NAMESPACE, name)


def jsonb_claim_to_response(
    claim_dict: dict[str, Any],
    message_id: UUID,
) -> ClaimResponse:
    """Transform JSONB claim dict to ClaimResponse schema.

    Args:
        claim_dict: Claim data from JSONB verification_data.claims[].
        message_id: The message ID for generating deterministic claim UUIDs.

    Returns:
        ClaimResponse schema ready for API response.
    """
    # Generate deterministic UUID from position
    claim_id = generate_claim_uuid(
        message_id,
        claim_dict["position_start"],
        claim_dict["position_end"],
    )

    # Build citation from embedded evidence
    citations: list[CitationResponse] = []
    evidence = claim_dict.get("evidence")
    if evidence:
        # Generate deterministic IDs for evidence and source
        evidence_id = uuid5(CLAIM_UUID_NAMESPACE, f"{claim_id}:evidence")
        source_id = uuid5(CLAIM_UUID_NAMESPACE, evidence["source_url"])

        source_metadata = SourceMetadataResponse(
            id=source_id,
            title=evidence.get("source_title"),
            url=evidence["source_url"],
            author=None,
            published_date=None,
            content_type=None,
            total_pages=None,
        )

        evidence_span = EvidenceSpanResponse(
            id=evidence_id,
            source_id=source_id,
            quote_text=evidence["quote_text"],
            start_offset=evidence.get("start_offset"),
            end_offset=evidence.get("end_offset"),
            section_heading=evidence.get("section_heading"),
            relevance_score=evidence.get("relevance_score"),
            has_numeric_content=evidence.get("has_numeric_content", False),
            source=source_metadata,
        )

        citations.append(CitationResponse(
            evidence_span=evidence_span,
            confidence_score=evidence.get("relevance_score"),
            is_primary=True,
        ))

    # Parse enums safely
    claim_type = ClaimTypeEnum(claim_dict["claim_type"])

    confidence_level = None
    if claim_dict.get("confidence_level"):
        confidence_level = ConfidenceLevelEnum(claim_dict["confidence_level"])

    verification_verdict = None
    if claim_dict.get("verification_verdict"):
        verification_verdict = VerificationVerdictEnum(claim_dict["verification_verdict"])

    return ClaimResponse(
        id=claim_id,
        claim_text=claim_dict["claim_text"],
        claim_type=claim_type,
        confidence_level=confidence_level,
        position_start=claim_dict["position_start"],
        position_end=claim_dict["position_end"],
        verification_verdict=verification_verdict,
        verification_reasoning=claim_dict.get("verification_reasoning"),
        abstained=claim_dict.get("abstained", False),
        citations=citations,
        corrections=[],  # Corrections not stored in JSONB
        numeric_detail=None,  # Numeric details not stored in JSONB
        citation_key=claim_dict.get("citation_key"),
        citation_keys=claim_dict.get("citation_keys"),
    )


def jsonb_summary_to_response(summary_dict: dict[str, Any]) -> VerificationSummary:
    """Transform JSONB summary dict to VerificationSummary schema.

    Args:
        summary_dict: Summary data from JSONB verification_data.summary.

    Returns:
        VerificationSummary schema ready for API response.
    """
    return VerificationSummary(
        total_claims=summary_dict.get("total_claims", 0),
        supported_count=summary_dict.get("supported_count", 0),
        partial_count=summary_dict.get("partial_count", 0),
        unsupported_count=summary_dict.get("unsupported_count", 0),
        contradicted_count=summary_dict.get("contradicted_count", 0),
        abstained_count=summary_dict.get("abstained_count", 0),
        unsupported_rate=summary_dict.get("unsupported_rate", 0.0),
        contradicted_rate=summary_dict.get("contradicted_rate", 0.0),
        warning=summary_dict.get("warning", False),
    )
