"""Citation verification endpoints.

Provides API endpoints for claim-level citation access:
- GET /messages/{id}/claims - List all claims for a message
- GET /claims/{id} - Get a specific claim with evidence
- GET /claims/{id}/evidence - Get evidence for a claim
- GET /messages/{id}/verification-summary - Get verification summary
- GET /messages/{id}/provenance - Export provenance data for a message (JSON or Markdown)
- GET /messages/{id}/report - Export research report as Markdown

JSONB Migration (Migration 011):
Claims and verification data are now read from the verification_data JSONB column
on the research_sessions table instead of normalized tables.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, Query
from fastapi.responses import PlainTextResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from deep_research.api.v1.utils import (
    build_empty_verification_summary,
    generate_claim_uuid,
    jsonb_claim_to_response,
    jsonb_summary_to_response,
    verify_message_ownership,
)
from deep_research.core.exceptions import NotFoundError
from deep_research.db.session import get_db
from deep_research.middleware.auth import CurrentUser
from deep_research.models.research_session import ResearchSession
from deep_research.schemas.citation import (
    CitationResponse,
    ClaimEvidenceResponse,
    ClaimProvenanceExport,
    ClaimResponse,
    ClaimTypeEnum,
    ConfidenceLevelEnum,
    CorrectionMetrics,
    EvidenceSpanResponse,
    MessageClaimsResponse,
    ProvenanceExport,
    SourceMetadataResponse,
    VerificationSummary,
    VerificationVerdictEnum,
)

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

    JSONB Migration: Now reads from verification_data JSONB column.
    """
    # Try to verify ownership - return empty claims if message doesn't exist yet
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

    # Get research session with verification_data JSONB
    result = await db.execute(
        select(ResearchSession).where(ResearchSession.message_id == message_id)
    )
    session = result.scalar_one_or_none()

    if not session or not session.verification_data:
        return MessageClaimsResponse(
            message_id=message_id,
            claims=[],
            verification_summary=build_empty_verification_summary(),
            correction_metrics=None,
        )

    # Transform JSONB to response schemas
    verification_data = session.verification_data
    claims = [
        jsonb_claim_to_response(c, message_id)
        for c in verification_data.get("claims", [])
    ]
    summary = jsonb_summary_to_response(verification_data.get("summary", {}))

    # Correction metrics not tracked in JSONB (deprecated feature)
    correction_metrics = None
    if include_corrections:
        # Return zero metrics for backwards compatibility
        total_claims = len(claims) or 1
        correction_metrics = CorrectionMetrics(
            total_corrections=0,
            keep_count=0,
            replace_count=0,
            remove_count=0,
            add_alternate_count=0,
            correction_rate=0.0,
        )

    return MessageClaimsResponse(
        message_id=message_id,
        claims=claims,
        verification_summary=summary,
        correction_metrics=correction_metrics,
    )


@router.get("/claims/{claim_id}", response_model=ClaimResponse)
async def get_claim(
    claim_id: UUID,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
    message_id: UUID | None = Query(None, description="Message ID for JSONB lookup"),
) -> ClaimResponse:
    """Get a specific claim with all its evidence and metadata.

    JSONB Migration: Claims are now stored in JSONB and looked up by position.
    The claim_id is a deterministic UUID generated from (message_id, position_start, position_end).
    Requires message_id query parameter for efficient lookup.
    """
    if not message_id:
        # Without message_id, we can't efficiently look up the claim in JSONB
        # Return 404 as this endpoint requires the new query parameter
        raise NotFoundError("Claim", str(claim_id))

    # Verify ownership
    await verify_message_ownership(message_id, user.user_id, db)

    # Get research session with verification_data
    result = await db.execute(
        select(ResearchSession).where(ResearchSession.message_id == message_id)
    )
    session = result.scalar_one_or_none()

    if not session or not session.verification_data:
        raise NotFoundError("Claim", str(claim_id))

    # Search for the claim by matching generated UUID
    verification_data = session.verification_data
    for claim_dict in verification_data.get("claims", []):
        generated_id = generate_claim_uuid(
            message_id,
            claim_dict["position_start"],
            claim_dict["position_end"],
        )
        if generated_id == claim_id:
            return jsonb_claim_to_response(claim_dict, message_id)

    raise NotFoundError("Claim", str(claim_id))


@router.get("/claims/{claim_id}/evidence", response_model=ClaimEvidenceResponse)
async def get_claim_evidence(
    claim_id: UUID,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
    message_id: UUID | None = Query(None, description="Message ID for JSONB lookup"),
) -> ClaimEvidenceResponse:
    """Get evidence for a specific claim.

    Returns the claim text and all supporting evidence spans
    with source metadata for evidence card display.

    JSONB Migration: Evidence is now embedded in the claim JSONB.
    Requires message_id query parameter for efficient lookup.
    """
    if not message_id:
        raise NotFoundError("Claim", str(claim_id))

    # Verify ownership
    await verify_message_ownership(message_id, user.user_id, db)

    # Get research session with verification_data
    result = await db.execute(
        select(ResearchSession).where(ResearchSession.message_id == message_id)
    )
    session = result.scalar_one_or_none()

    if not session or not session.verification_data:
        raise NotFoundError("Claim", str(claim_id))

    # Search for the claim by matching generated UUID
    verification_data = session.verification_data
    for claim_dict in verification_data.get("claims", []):
        generated_id = generate_claim_uuid(
            message_id,
            claim_dict["position_start"],
            claim_dict["position_end"],
        )
        if generated_id == claim_id:
            # Build citations from embedded evidence
            citations: list[CitationResponse] = []
            evidence = claim_dict.get("evidence")
            if evidence:
                from uuid import uuid5, NAMESPACE_DNS
                evidence_id = uuid5(NAMESPACE_DNS, f"{claim_id}:evidence")
                source_id = uuid5(NAMESPACE_DNS, evidence["source_url"])

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

            # Parse verdict
            verdict = None
            if claim_dict.get("verification_verdict"):
                verdict = VerificationVerdictEnum(claim_dict["verification_verdict"])

            return ClaimEvidenceResponse(
                claim_id=claim_id,
                claim_text=claim_dict["claim_text"],
                verification_verdict=verdict,
                citations=citations,
            )

    raise NotFoundError("Claim", str(claim_id))


@router.get("/messages/{message_id}/verification-summary", response_model=VerificationSummary)
async def get_verification_summary(
    message_id: UUID,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> VerificationSummary:
    """Get verification summary for a message.

    Returns aggregated verification statistics including
    counts by verdict and warning status.

    JSONB Migration: Summary is now read from verification_data JSONB.
    """
    # Verify ownership
    await verify_message_ownership(message_id, user.user_id, db)

    # Get research session with verification_data
    result = await db.execute(
        select(ResearchSession).where(ResearchSession.message_id == message_id)
    )
    session = result.scalar_one_or_none()

    if not session or not session.verification_data:
        return build_empty_verification_summary()

    summary_dict = session.verification_data.get("summary", {})
    return jsonb_summary_to_response(summary_dict)


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

    from deep_research.services.export_service import ExportService

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

    JSONB Migration: Now reads from verification_data JSONB.
    """
    # Handle markdown format
    if format == "markdown":
        import logging

        from deep_research.services.export_service import ExportService

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

    # Verify ownership
    await verify_message_ownership(message_id, user.user_id, db)

    # Get research session with verification_data
    result = await db.execute(
        select(ResearchSession).where(ResearchSession.message_id == message_id)
    )
    session = result.scalar_one_or_none()

    if not session or not session.verification_data:
        return ProvenanceExport(
            exported_at=datetime.now(timezone.utc),
            message_id=message_id,
            claims=[],
            summary=build_empty_verification_summary(),
        )

    verification_data = session.verification_data

    # Build export claims from JSONB
    export_claims: list[ClaimProvenanceExport] = []
    for claim_dict in verification_data.get("claims", []):
        # Build citations for export (simplified dict format)
        citations: list[dict[str, str | bool | None]] = []
        evidence = claim_dict.get("evidence")
        if evidence:
            citations.append({
                "source_url": evidence.get("source_url"),
                "source_title": evidence.get("source_title"),
                "quote": evidence.get("quote_text", ""),
                "is_primary": True,
            })

        # Parse enums
        claim_type = ClaimTypeEnum(claim_dict["claim_type"])
        verdict = None
        if claim_dict.get("verification_verdict"):
            verdict = VerificationVerdictEnum(claim_dict["verification_verdict"])
        confidence = None
        if claim_dict.get("confidence_level"):
            confidence = ConfidenceLevelEnum(claim_dict["confidence_level"])

        export_claims.append(
            ClaimProvenanceExport(
                claim_text=claim_dict["claim_text"],
                claim_type=claim_type,
                verdict=verdict,
                confidence_level=confidence,
                citations=citations,
                numeric_detail=None,  # Numeric details not stored in JSONB
                corrections=[],  # Corrections not stored in JSONB
            )
        )

    # Get summary from JSONB
    summary = jsonb_summary_to_response(verification_data.get("summary", {}))

    return ProvenanceExport(
        exported_at=datetime.now(timezone.utc),
        message_id=message_id,
        claims=export_claims,
        summary=summary,
    )
