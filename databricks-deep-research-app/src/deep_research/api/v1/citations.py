"""Citation verification endpoints.

Provides API endpoints for claim-level citation access:
- GET /messages/{id}/claims - List all claims for a message
- GET /claims/{id} - Get a specific claim with evidence
- GET /claims/{id}/evidence - Get evidence for a claim
- GET /messages/{id}/verification-summary - Get verification summary
- GET /messages/{id}/provenance - Export provenance data (JSON or Markdown)
- GET /messages/{id}/report - Export research report as Markdown

Storage model:
Claims and verification data live in the event-sourced storage stack
(``chat_state.state.research_sessions[].verification_data``) and are read via
the cached ``IResearchSessionService``. The JSON endpoints require ``chat_id``
(the frontend always knows the active chat) so the lookup is a single
chat-document read and ownership can be enforced; the legacy normalized
``research_sessions`` table is no longer queried. The ``/report`` and
``/provenance?format=markdown`` endpoints already compose cached services via
``IExportService`` and need no ``chat_id``.
"""

import logging
from datetime import UTC
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from fastapi.responses import PlainTextResponse

from deep_research.api.v1.utils import (
    build_empty_verification_summary,
    generate_claim_uuid,
    jsonb_claim_to_response,
    jsonb_summary_to_response,
)
from deep_research.core.deps import (
    get_chat_service,
    get_export_service,
    get_research_session_service,
)
from deep_research.core.exceptions import NotFoundError
from deep_research.middleware.auth import CurrentUser
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
from deep_research.services._protocols import (
    IChatService,
    IExportService,
    IResearchSessionService,
)

router = APIRouter()

logger = logging.getLogger(__name__)


async def _load_verification_data(
    message_id: UUID,
    chat_id: UUID,
    user_id: str,
    chat_service: IChatService,
    rs_service: IResearchSessionService,
) -> dict | None:
    """Return the ``verification_data`` dict for ``message_id`` within ``chat_id``.

    Reads from the event-sourced storage stack. Returns ``None`` when:
    - the user does not own ``chat_id`` (ownership check — closes IDOR: a user
      passing their own ``chat_id`` with a foreign ``message_id`` gets nothing),
    - no research session in that chat matches ``message_id``, or
    - no verification data has been persisted yet.

    Callers map ``None`` to an empty response so the frontend can poll during the
    post-synthesis persistence window without leaking message existence.
    """
    chat = await chat_service.get_for_user(chat_id, user_id)
    if chat is None:
        logger.info(
            "CITATIONS_LOAD message_id=%s chat_id=%s result=chat_not_owned",
            message_id, chat_id,
        )
        return None
    rs = await rs_service.get_by_message(message_id, chat_id=chat_id)
    if rs is None:
        logger.info(
            "CITATIONS_LOAD message_id=%s chat_id=%s result=session_not_found_for_message",
            message_id, chat_id,
        )
        return None
    verification_data = getattr(rs, "verification_data", None)
    n_claims = len((verification_data or {}).get("claims", [])) if verification_data else 0
    logger.info(
        "CITATIONS_LOAD message_id=%s chat_id=%s result=ok n_claims=%s",
        message_id, chat_id, n_claims,
    )
    return verification_data or None


@router.get("/messages/{message_id}/claims", response_model=MessageClaimsResponse)
async def list_message_claims(
    message_id: UUID,
    user: CurrentUser,
    chat_id: UUID = Query(..., description="Chat that owns the message"),
    include_corrections: bool = Query(False),
    chat_service: IChatService = Depends(get_chat_service),
    rs_service: IResearchSessionService = Depends(get_research_session_service),
) -> MessageClaimsResponse:
    """List all claims for a message with verification summary.

    Returns empty claims (200 OK) when the claims are not yet persisted, the
    message is unknown, or the chat is not owned by the user. This supports
    frontend polling during the persistence race condition (the message UUID is
    pre-generated before streaming; claims land after synthesis completes) and
    avoids leaking message existence across users.
    """
    verification_data = await _load_verification_data(
        message_id, chat_id, user.user_id, chat_service, rs_service
    )
    if not verification_data:
        return MessageClaimsResponse(
            message_id=message_id,
            claims=[],
            verification_summary=build_empty_verification_summary(),
            correction_metrics=None,
        )

    claims = [
        jsonb_claim_to_response(c, message_id)
        for c in verification_data.get("claims", [])
    ]
    summary = jsonb_summary_to_response(verification_data.get("summary", {}))

    # Correction metrics are not tracked in JSONB (deprecated feature).
    correction_metrics = None
    if include_corrections:
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
    message_id: UUID = Query(..., description="Message ID for the claim lookup"),
    chat_id: UUID = Query(..., description="Chat that owns the message"),
    chat_service: IChatService = Depends(get_chat_service),
    rs_service: IResearchSessionService = Depends(get_research_session_service),
) -> ClaimResponse:
    """Get a specific claim with all its evidence and metadata.

    The claim_id is a deterministic UUID generated from (message_id,
    position_start, position_end); we scan the message's claims for the match.
    """
    verification_data = await _load_verification_data(
        message_id, chat_id, user.user_id, chat_service, rs_service
    )
    if not verification_data:
        raise NotFoundError("Claim", str(claim_id))

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
    message_id: UUID = Query(..., description="Message ID for the claim lookup"),
    chat_id: UUID = Query(..., description="Chat that owns the message"),
    chat_service: IChatService = Depends(get_chat_service),
    rs_service: IResearchSessionService = Depends(get_research_session_service),
) -> ClaimEvidenceResponse:
    """Get evidence for a specific claim.

    Returns the claim text and all supporting evidence spans with source
    metadata for evidence card display.
    """
    verification_data = await _load_verification_data(
        message_id, chat_id, user.user_id, chat_service, rs_service
    )
    if not verification_data:
        raise NotFoundError("Claim", str(claim_id))

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
                from uuid import NAMESPACE_DNS, uuid5
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
    chat_id: UUID = Query(..., description="Chat that owns the message"),
    chat_service: IChatService = Depends(get_chat_service),
    rs_service: IResearchSessionService = Depends(get_research_session_service),
) -> VerificationSummary:
    """Get verification summary for a message.

    Returns aggregated verification statistics including counts by verdict and
    warning status. Reads from the storage stack ``verification_data`` JSONB.
    """
    verification_data = await _load_verification_data(
        message_id, chat_id, user.user_id, chat_service, rs_service
    )
    if not verification_data:
        return build_empty_verification_summary()
    return jsonb_summary_to_response(verification_data.get("summary", {}))


@router.get("/messages/{message_id}/report")
async def export_report(
    message_id: UUID,
    user: CurrentUser,
    export_service: IExportService = Depends(get_export_service),
) -> PlainTextResponse:
    """Export research report as standalone markdown.

    Returns the agent synthesis with metadata and sources list as a downloadable
    markdown file. ``IExportService`` composes cached services and resolves the
    chat from the message internally, so no ``chat_id`` is required here.
    """
    import logging

    logger = logging.getLogger(__name__)

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
    chat_id: UUID = Query(..., description="Chat that owns the message"),
    format: str = Query("json", pattern="^(json|markdown)$"),
    export_service: IExportService = Depends(get_export_service),
    chat_service: IChatService = Depends(get_chat_service),
    rs_service: IResearchSessionService = Depends(get_research_session_service),
) -> ProvenanceExport | PlainTextResponse:
    """Export provenance data for a message.

    Returns all claims with their citations, verification verdicts, and
    verification summary in an exportable format for audit trails / compliance.

    Args:
        format: Export format - "json" (default) or "markdown".
    """
    # Markdown format (composes cached services via IExportService).
    if format == "markdown":
        import logging

        logger = logging.getLogger(__name__)

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

    # JSON format (default) — read verification_data from the storage stack.
    from datetime import datetime

    verification_data = await _load_verification_data(
        message_id, chat_id, user.user_id, chat_service, rs_service
    )
    if not verification_data:
        return ProvenanceExport(
            exported_at=datetime.now(UTC),
            message_id=message_id,
            claims=[],
            summary=build_empty_verification_summary(),
        )

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

    summary = jsonb_summary_to_response(verification_data.get("summary", {}))

    return ProvenanceExport(
        exported_at=datetime.now(UTC),
        message_id=message_id,
        claims=export_claims,
        summary=summary,
    )
