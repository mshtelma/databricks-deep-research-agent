"""Citation verification endpoints.

Provides API endpoints for claim-level citation access:
- GET /messages/{id}/claims - List all claims for a message
- GET /claims/{id} - Get a specific claim with evidence
- GET /claims/{id}/evidence - Get evidence for a claim
- GET /messages/{id}/verification-summary - Get verification summary
- GET /messages/{id}/provenance - Export provenance data for a message
"""

from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.core.exceptions import NotFoundError
from src.db.session import get_db
from src.middleware.auth import CurrentUser
from src.models.claim import Claim
from src.models.message import Message
from src.schemas.citation import (
    CitationResponse,
    ClaimEvidenceResponse,
    ClaimProvenanceExport,
    ClaimResponse,
    ConfidenceLevelEnum,
    CorrectionMetrics,
    EvidenceSpanResponse,
    MessageClaimsResponse,
    ProvenanceExport,
    SourceMetadataResponse,
    VerificationSummary,
    VerificationVerdictEnum,
)
from src.services.chat_service import ChatService
from src.services.claim_service import ClaimService
from src.services.verification_summary_service import VerificationSummaryService

router = APIRouter()


async def _verify_message_ownership(
    message_id: UUID, user_id: str, db: AsyncSession
) -> Message:
    """Verify user owns the message's chat. Returns the message."""
    from sqlalchemy import select

    result = await db.execute(
        select(Message)
        .options(selectinload(Message.chat))
        .where(Message.id == message_id)
    )
    message = result.scalar_one_or_none()

    if not message:
        raise NotFoundError("Message", str(message_id))

    # Verify chat ownership
    chat_service = ChatService(db)
    chat = await chat_service.get(message.chat_id, user_id)
    if not chat:
        raise NotFoundError("Message", str(message_id))

    return message


def _claim_to_response(claim: Claim) -> ClaimResponse:
    """Convert Claim model to ClaimResponse schema."""
    from src.schemas.citation import (
        CitationCorrectionResponse,
        CorrectionTypeEnum,
        NumericClaimDetail,
        DerivationTypeEnum,
        ClaimTypeEnum,
    )

    # Build citations list
    citations = []
    for citation in claim.citations:
        span = citation.evidence_span
        source = span.source if span else None

        source_metadata = None
        if source:
            source_metadata = SourceMetadataResponse(
                id=source.id,
                title=source.title,
                url=source.url,
                author=None,  # Not stored in current model
                published_date=None,
                content_type=source.content_type,
                total_pages=source.total_pages,
            )

        evidence_span = EvidenceSpanResponse(
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

        citations.append(
            CitationResponse(
                evidence_span=evidence_span,
                confidence_score=citation.confidence_score,
                is_primary=citation.is_primary,
            )
        )

    # Build corrections list
    corrections = []
    for correction in claim.corrections:
        corrections.append(
            CitationCorrectionResponse(
                id=correction.id,
                correction_type=CorrectionTypeEnum(correction.correction_type),
                original_evidence=None,  # Simplified for now
                corrected_evidence=None,
                reasoning=correction.reasoning,
            )
        )

    # Build numeric detail if applicable
    numeric_detail = None
    if claim.numeric_detail:
        nd = claim.numeric_detail
        numeric_detail = NumericClaimDetail(
            raw_value=nd.raw_value,
            normalized_value=float(nd.normalized_value) if nd.normalized_value else None,
            unit=nd.unit,
            entity_reference=nd.entity_reference,
            derivation_type=DerivationTypeEnum(nd.derivation_type),
            computation_details=nd.computation_details,
            assumptions=nd.assumptions,
            qa_verification=None,  # TODO: Parse QA verification
        )

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
        numeric_detail=numeric_detail,
        citation_key=claim.citation_key,
        citation_keys=claim.citation_keys,
    )


@router.get("/messages/{message_id}/claims", response_model=MessageClaimsResponse)
async def list_message_claims(
    message_id: UUID,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
    include_corrections: bool = Query(False),
) -> MessageClaimsResponse:
    """List all claims for a message with verification summary.

    Returns all claims extracted from the agent message, including
    their citations, verification verdicts, and corrections.
    """
    # Verify ownership
    message = await _verify_message_ownership(message_id, user.user_id, db)

    # Get claims with all relationships
    claim_service = ClaimService(db)
    claims = await claim_service.list_by_message(message_id, include_citations=True)

    # Get or compute verification summary
    summary_service = VerificationSummaryService(db)
    summary_model = await summary_service.get_or_compute(message_id)

    # Build verification summary response
    summary = VerificationSummary(
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

    # Get correction metrics if requested
    correction_metrics = None
    if include_corrections:
        metrics = await summary_service.get_correction_metrics(message_id)
        total_claims = len(claims) or 1  # Avoid division by zero
        correction_metrics = CorrectionMetrics(
            total_corrections=metrics["total"],
            keep_count=metrics["keep"],
            replace_count=metrics["replace"],
            remove_count=metrics["remove"],
            add_alternate_count=metrics["add_alternate"],
            correction_rate=metrics["total"] / total_claims,
        )

    return MessageClaimsResponse(
        message_id=message_id,
        claims=[_claim_to_response(c) for c in claims],
        verification_summary=summary,
        correction_metrics=correction_metrics,
    )


@router.get("/claims/{claim_id}", response_model=ClaimResponse)
async def get_claim(
    claim_id: UUID,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> ClaimResponse:
    """Get a specific claim with all its evidence and metadata."""
    # Get claim with all relationships
    claim_service = ClaimService(db)
    claim = await claim_service.get_with_citations(claim_id)

    if not claim:
        raise NotFoundError("Claim", str(claim_id))

    # Verify ownership through the message's chat
    await _verify_message_ownership(claim.message_id, user.user_id, db)

    return _claim_to_response(claim)


@router.get("/claims/{claim_id}/evidence", response_model=ClaimEvidenceResponse)
async def get_claim_evidence(
    claim_id: UUID,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> ClaimEvidenceResponse:
    """Get evidence for a specific claim.

    Returns the claim text and all supporting evidence spans
    with source metadata for evidence card display.
    """
    # Get claim with citations
    claim_service = ClaimService(db)
    claim = await claim_service.get_with_citations(claim_id)

    if not claim:
        raise NotFoundError("Claim", str(claim_id))

    # Verify ownership
    await _verify_message_ownership(claim.message_id, user.user_id, db)

    # Build citations with full source metadata
    citations = []
    for citation in claim.citations:
        span = citation.evidence_span
        source = span.source if span else None

        source_metadata = None
        if source:
            source_metadata = SourceMetadataResponse(
                id=source.id,
                title=source.title,
                url=source.url,
                author=None,
                published_date=None,
                content_type=source.content_type,
                total_pages=source.total_pages,
            )

        evidence_span = EvidenceSpanResponse(
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

        citations.append(
            CitationResponse(
                evidence_span=evidence_span,
                confidence_score=citation.confidence_score,
                is_primary=citation.is_primary,
            )
        )

    return ClaimEvidenceResponse(
        claim_id=claim.id,
        claim_text=claim.claim_text,
        verification_verdict=VerificationVerdictEnum(claim.verification_verdict)
        if claim.verification_verdict
        else None,
        citations=citations,
    )


@router.get("/messages/{message_id}/verification-summary", response_model=VerificationSummary)
async def get_verification_summary(
    message_id: UUID,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> VerificationSummary:
    """Get verification summary for a message.

    Returns aggregated verification statistics including
    counts by verdict and warning status.
    """
    # Verify ownership
    await _verify_message_ownership(message_id, user.user_id, db)

    # Get or compute summary
    summary_service = VerificationSummaryService(db)
    summary_model = await summary_service.get_or_compute(message_id)

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


@router.get("/messages/{message_id}/provenance", response_model=ProvenanceExport)
async def export_provenance(
    message_id: UUID,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> ProvenanceExport:
    """Export provenance data for a message.

    Returns all claims with their citations, verification verdicts,
    corrections, and verification summary in an exportable format.
    Suitable for audit trails, compliance, and downstream processing.
    """
    from datetime import datetime, timezone
    from src.schemas.citation import ClaimTypeEnum, DerivationTypeEnum

    # Verify ownership
    await _verify_message_ownership(message_id, user.user_id, db)

    # Get claims with all relationships
    claim_service = ClaimService(db)
    claims = await claim_service.list_by_message(message_id, include_citations=True)

    # Get verification summary
    summary_service = VerificationSummaryService(db)
    summary_model = await summary_service.get_or_compute(message_id)

    # Build export claims
    export_claims: list[ClaimProvenanceExport] = []
    for claim in claims:
        # Build citations for export
        citations: list[dict[str, str | bool | None]] = []
        for citation in claim.citations:
            span = citation.evidence_span
            source = span.source if span else None
            citations.append({
                "source_url": source.url if source else None,
                "source_title": source.title if source else None,
                "quote": span.quote_text if span else "",
                "is_primary": citation.is_primary,
            })

        # Build corrections for export
        corrections: list[dict[str, str | None]] = []
        for correction in claim.corrections:
            corrections.append({
                "correction_type": correction.correction_type,
                "reasoning": correction.reasoning,
            })

        # Build numeric detail if applicable
        from src.schemas.citation import NumericClaimDetail

        numeric_detail = None
        if claim.numeric_detail:
            nd = claim.numeric_detail
            numeric_detail = NumericClaimDetail(
                raw_value=nd.raw_value,
                normalized_value=float(nd.normalized_value) if nd.normalized_value else None,
                unit=nd.unit,
                entity_reference=nd.entity_reference,
                derivation_type=DerivationTypeEnum(nd.derivation_type),
                computation_details=nd.computation_details,
                assumptions=nd.assumptions,
                qa_verification=None,
            )

        export_claims.append(
            ClaimProvenanceExport(
                claim_text=claim.claim_text,
                claim_type=ClaimTypeEnum(claim.claim_type),
                verdict=VerificationVerdictEnum(claim.verification_verdict)
                if claim.verification_verdict
                else None,
                confidence_level=ConfidenceLevelEnum(claim.confidence_level)
                if claim.confidence_level
                else None,
                citations=citations,
                numeric_detail=numeric_detail,
                corrections=corrections,
            )
        )

    # Build summary
    summary = VerificationSummary(
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

    return ProvenanceExport(
        exported_at=datetime.now(timezone.utc),
        message_id=message_id,
        claims=export_claims,
        summary=summary,
    )
