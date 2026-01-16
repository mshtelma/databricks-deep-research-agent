"""Citation verification endpoints.

Provides API endpoints for claim-level citation access:
- GET /messages/{id}/claims - List all claims for a message
- GET /claims/{id} - Get a specific claim with evidence
- GET /claims/{id}/evidence - Get evidence for a claim
- GET /messages/{id}/verification-summary - Get verification summary
- GET /messages/{id}/provenance - Export provenance data for a message (JSON or Markdown)
- GET /messages/{id}/report - Export research report as Markdown
"""

from uuid import UUID

from fastapi import APIRouter, Depends, Query
from fastapi.responses import PlainTextResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.v1.utils import (
    build_citation_response,
    build_empty_verification_summary,
    build_verification_summary,
    claim_to_response,
    verify_message_ownership,
)
from src.core.exceptions import NotFoundError
from src.db.session import get_db
from src.middleware.auth import CurrentUser
from src.schemas.citation import (
    ClaimEvidenceResponse,
    ClaimProvenanceExport,
    ClaimResponse,
    ConfidenceLevelEnum,
    CorrectionMetrics,
    MessageClaimsResponse,
    ProvenanceExport,
    VerificationSummary,
    VerificationVerdictEnum,
)
from src.services.claim_service import ClaimService
from src.services.verification_summary_service import VerificationSummaryService

router = APIRouter()


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

    NOTE: Returns empty claims (200 OK) instead of 404 when message not found.
    This supports frontend polling during the persistence race condition:
    - Message UUID is pre-generated before streaming
    - Claims are persisted after synthesis completes (~10-30s)
    - Frontend polls this endpoint until claims are available
    """
    # Try to verify ownership - return empty claims if message doesn't exist yet
    # This handles the race condition where frontend polls for claims before
    # the message has been persisted to the database
    try:
        await verify_message_ownership(message_id, user.user_id, db)
    except NotFoundError:
        # Message not persisted yet - return empty claims to allow frontend polling
        return MessageClaimsResponse(
            message_id=message_id,
            claims=[],
            verification_summary=build_empty_verification_summary(),
            correction_metrics=None,
        )

    # Get claims with all relationships
    claim_service = ClaimService(db)
    claims = await claim_service.list_by_message(message_id, include_citations=True)

    # Get or compute verification summary
    summary_service = VerificationSummaryService(db)
    summary_model = await summary_service.get_or_compute(message_id)

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
        claims=[claim_to_response(c) for c in claims],
        verification_summary=build_verification_summary(summary_model),
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
    await verify_message_ownership(claim.message_id, user.user_id, db)

    return claim_to_response(claim)


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
    await verify_message_ownership(claim.message_id, user.user_id, db)

    # Build citations using shared transformer
    citations = [build_citation_response(c) for c in claim.citations]

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
    await verify_message_ownership(message_id, user.user_id, db)

    # Get or compute summary
    summary_service = VerificationSummaryService(db)
    summary_model = await summary_service.get_or_compute(message_id)

    return build_verification_summary(summary_model)


@router.get("/messages/{message_id}/report")
async def export_report(
    message_id: UUID,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> PlainTextResponse:
    """Export research report as standalone markdown.

    Returns the agent synthesis with metadata and sources list
    as a downloadable markdown file.
    """
    import logging

    from src.services.export_service import ExportService

    logger = logging.getLogger(__name__)
    export_service = ExportService(db)

    try:
        content = await export_service.export_report_markdown(
            message_id=message_id,
            user_id=user.user_id,
        )
    except ValueError as e:
        raise NotFoundError("Message", str(message_id)) from e
    except Exception as e:
        logger.exception(f"Failed to export report for message {message_id}: {e}")
        raise NotFoundError("Message", str(message_id)) from e

    return PlainTextResponse(
        content=content,
        media_type="text/markdown",
        headers={
            "Content-Disposition": f'attachment; filename="report-{message_id}.md"'
        },
    )


@router.get("/messages/{message_id}/provenance", response_model=None)
async def export_provenance(
    message_id: UUID,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
    format: str = Query("json", pattern="^(json|markdown)$"),
) -> ProvenanceExport | PlainTextResponse:
    """Export provenance data for a message.

    Returns all claims with their citations, verification verdicts,
    corrections, and verification summary in an exportable format.
    Suitable for audit trails, compliance, and downstream processing.

    Args:
        format: Export format - "json" (default) or "markdown"
    """
    # Handle markdown format
    if format == "markdown":
        import logging

        from src.services.export_service import ExportService

        logger = logging.getLogger(__name__)
        export_service = ExportService(db)

        try:
            content = await export_service.export_provenance_markdown(
                message_id=message_id,
                user_id=user.user_id,
            )
        except ValueError as e:
            raise NotFoundError("Message", str(message_id)) from e
        except Exception as e:
            logger.exception(f"Failed to export provenance for message {message_id}: {e}")
            raise NotFoundError("Message", str(message_id)) from e

        return PlainTextResponse(
            content=content,
            media_type="text/markdown",
            headers={
                "Content-Disposition": f'attachment; filename="verification-{message_id}.md"'
            },
        )

    # JSON format (default)
    from datetime import datetime, timezone

    from src.api.v1.utils import build_numeric_detail
    from src.schemas.citation import ClaimTypeEnum

    # Verify ownership
    await verify_message_ownership(message_id, user.user_id, db)

    # Get claims with all relationships
    claim_service = ClaimService(db)
    claims = await claim_service.list_by_message(message_id, include_citations=True)

    # Get verification summary
    summary_service = VerificationSummaryService(db)
    summary_model = await summary_service.get_or_compute(message_id)

    # Build export claims
    export_claims: list[ClaimProvenanceExport] = []
    for claim in claims:
        # Build citations for export (simplified dict format)
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
                numeric_detail=build_numeric_detail(claim.numeric_detail),
                corrections=corrections,
            )
        )

    return ProvenanceExport(
        exported_at=datetime.now(timezone.utc),
        message_id=message_id,
        claims=export_claims,
        summary=build_verification_summary(summary_model),
    )
