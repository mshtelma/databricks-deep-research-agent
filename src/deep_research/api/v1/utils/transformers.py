"""Shared response transformers for API endpoints.

This module consolidates response transformation logic previously duplicated across:
- citations.py (_claim_to_response, VerificationSummary construction)
- export endpoints (similar transformation logic)

All functions are designed to be pure (no side effects) and reusable.
"""

from typing import TYPE_CHECKING

from deep_research.schemas.citation import (
    CitationCorrectionResponse,
    CitationResponse,
    ClaimResponse,
    ClaimTypeEnum,
    ConfidenceLevelEnum,
    CorrectionTypeEnum,
    DerivationTypeEnum,
    EvidenceSpanResponse,
    NumericClaimDetail,
    SourceMetadataResponse,
    VerificationSummary,
    VerificationVerdictEnum,
)

if TYPE_CHECKING:
    from deep_research.models.citation import Citation
    from deep_research.models.claim import Claim
    from deep_research.models.evidence_span import EvidenceSpan
    from deep_research.models.numeric_claim import NumericClaim
    from deep_research.models.source import Source
    from deep_research.models.verification_summary import VerificationSummary as VerificationSummaryModel


def build_source_metadata(source: "Source | None") -> SourceMetadataResponse | None:
    """Build SourceMetadataResponse from a Source model.

    Args:
        source: Source model instance or None.

    Returns:
        SourceMetadataResponse or None if source is None.
    """
    if not source:
        return None
    return SourceMetadataResponse(
        id=source.id,
        title=source.title,
        url=source.url,
        author=None,  # Not stored in current model
        published_date=None,
        content_type=source.content_type,
        total_pages=source.total_pages,
    )


def build_evidence_span_response(
    span: "EvidenceSpan",
    *,
    include_source: bool = True,
) -> EvidenceSpanResponse:
    """Build EvidenceSpanResponse from an EvidenceSpan model.

    Args:
        span: EvidenceSpan model instance.
        include_source: Whether to include nested source metadata.

    Returns:
        EvidenceSpanResponse schema.
    """
    source_metadata = None
    if include_source and span.source:
        source_metadata = build_source_metadata(span.source)

    return EvidenceSpanResponse(
        id=span.id,
        source_id=span.source_id,
        quote_text=span.quote_text,
        start_offset=span.start_offset,
        end_offset=span.end_offset,
        section_heading=span.section_heading,
        relevance_score=span.relevance_score,
        has_numeric_content=span.has_numeric_content,
        source=source_metadata,
    )


def build_citation_response(citation: "Citation") -> CitationResponse:
    """Build CitationResponse from a Citation model.

    Args:
        citation: Citation model instance with evidence_span loaded.

    Returns:
        CitationResponse schema.
    """
    span = citation.evidence_span
    return CitationResponse(
        evidence_span=build_evidence_span_response(span),
        confidence_score=citation.confidence_score,
        is_primary=citation.is_primary,
    )


def build_numeric_detail(nd: "NumericClaim | None") -> NumericClaimDetail | None:
    """Build NumericClaimDetail from a NumericClaim model.

    Args:
        nd: NumericClaim model instance or None.

    Returns:
        NumericClaimDetail schema or None.
    """
    if not nd:
        return None
    return NumericClaimDetail(
        raw_value=nd.raw_value,
        normalized_value=float(nd.normalized_value) if nd.normalized_value else None,
        unit=nd.unit,
        entity_reference=nd.entity_reference,
        derivation_type=DerivationTypeEnum(nd.derivation_type),
        computation_details=nd.computation_details,
        assumptions=nd.assumptions,
        qa_verification=None,  # TODO: Parse QA verification if needed
    )


def claim_to_response(claim: "Claim") -> ClaimResponse:
    """Transform Claim model to ClaimResponse schema.

    Expects claim to have citations, corrections, and numeric_detail
    eager-loaded to avoid N+1 queries.

    Args:
        claim: Claim model with relationships loaded.

    Returns:
        ClaimResponse schema.
    """
    # Build citations list
    citations = [build_citation_response(c) for c in claim.citations]

    # Build corrections list
    corrections = [
        CitationCorrectionResponse(
            id=c.id,
            correction_type=CorrectionTypeEnum(c.correction_type),
            original_evidence=None,  # Simplified for now
            corrected_evidence=None,
            reasoning=c.reasoning,
        )
        for c in claim.corrections
    ]

    return ClaimResponse(
        id=claim.id,
        claim_text=claim.claim_text,
        claim_type=ClaimTypeEnum(claim.claim_type),
        confidence_level=ConfidenceLevelEnum(claim.confidence_level)
        if claim.confidence_level
        else None,
        position_start=claim.position_start,
        position_end=claim.position_end,
        verification_verdict=VerificationVerdictEnum(claim.verification_verdict)
        if claim.verification_verdict
        else None,
        verification_reasoning=claim.verification_reasoning,
        abstained=claim.abstained,
        citations=citations,
        corrections=corrections,
        numeric_detail=build_numeric_detail(claim.numeric_detail),
        citation_key=claim.citation_key,
        citation_keys=claim.citation_keys,
    )


def build_verification_summary(
    summary_model: "VerificationSummaryModel",
) -> VerificationSummary:
    """Transform VerificationSummaryModel to VerificationSummary schema.

    Args:
        summary_model: Database model with verification counts.

    Returns:
        VerificationSummary schema.
    """
    return VerificationSummary(
        total_claims=summary_model.total_claims,
        supported_count=summary_model.supported_count,
        partial_count=summary_model.partial_count,
        unsupported_count=summary_model.unsupported_count,
        contradicted_count=summary_model.contradicted_count,
        abstained_count=summary_model.abstained_count,
        unsupported_rate=summary_model.unsupported_rate,
        contradicted_rate=summary_model.contradicted_rate,
        warning=summary_model.warning,
    )


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
