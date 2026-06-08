"""Research session endpoints.

Provides:
- Research cancellation
- Active research detection for reconnection
- Research event polling for hydration
- Research state retrieval

Note: The deprecated SSE streaming endpoints (GET/POST /{chat_id}/stream) have been
removed. Use the Jobs API (POST /api/v1/research/jobs) for new research requests
and GET /api/v1/research/jobs/{session_id}/stream for SSE streaming.
"""

import contextlib
import logging
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel

from deep_research.api.v1.utils.transformers import status_str
from deep_research.core.deps import (
    get_chat_service,
    get_message_service,
    get_research_event_service,
    get_research_session_service,
    get_storage,
)
from deep_research.core.exceptions import NotFoundError
from deep_research.middleware.auth import CurrentUser
from deep_research.models.research_session import ResearchStatus
from deep_research.schemas.research import CancelResearchResponse
from deep_research.services._protocols import (
    IChatService,
    IMessageService,
    IResearchEventService,
    IResearchSessionService,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/{session_id}/cancel", response_model=CancelResearchResponse)
async def cancel_research(
    session_id: UUID,
    request: Request,
    user: CurrentUser,
    rs_service: IResearchSessionService = Depends(get_research_session_service),
) -> CancelResearchResponse:
    """Cancel in-progress research.

    Stops the research operation within 2 seconds. Partial results are preserved.
    """
    # SECURITY: Resolve chat_id via the storage stack job list, then verify ownership.
    stack = get_storage(request)
    pairs = await stack.backend.list_user_jobs(user.user_id)

    chat_id: UUID | None = None
    for pair_chat_id, pair_session in pairs:
        if getattr(pair_session, "id", None) == session_id:
            chat_id = pair_chat_id
            break

    if chat_id is None:
        raise NotFoundError("ResearchSession", str(session_id))

    # Proceed with cancellation after ownership verified via list_user_jobs
    session = await rs_service.cancel(session_id, chat_id=chat_id)

    if not session:
        raise NotFoundError("ResearchSession", str(session_id))

    # Get partial results if available
    partial_results = None
    if session.observations:
        obs_list = (
            session.observations.get("items", [])
            if isinstance(session.observations, dict)
            else session.observations
        )
        partial_results = "\n\n".join(
            obs.get("observation", "") for obs in obs_list if obs.get("observation")
        ) or None

    return CancelResearchResponse(
        session_id=session_id,
        status="cancelled",
        partial_results=partial_results,
    )


# =============================================================================
# Reconnection Endpoints for Crash Resilience
# =============================================================================
# These endpoints enable frontend to reconnect to in-progress research
# after browser reload/crash. Uses polling-based approach (no WebSockets).
# =============================================================================


class ActiveResearchResponse(BaseModel):
    """Response for checking active research status."""

    has_active_research: bool
    session_id: UUID | None = None
    status: str | None = None  # "in_progress", "completed", "failed"
    last_sequence_number: int | None = None
    query: str | None = None
    query_mode: str | None = None  # "simple", "web_search", "deep_research"
    started_at: str | None = None  # ISO timestamp for timer display


@router.get("/{chat_id}/research/active")
async def get_active_research(
    chat_id: UUID,
    user: CurrentUser,
    chat_service: IChatService = Depends(get_chat_service),
    rs_service: IResearchSessionService = Depends(get_research_session_service),
    event_service: IResearchEventService = Depends(get_research_event_service),
) -> ActiveResearchResponse:
    """Check if there's an in-progress research session for this chat.

    Frontend calls this on page load to detect if reconnection is needed.
    Returns session info if research is in progress, otherwise has_active_research=False.
    """
    # Verify user has access to this chat
    chat = await chat_service.get_by_id(chat_id)
    if chat is None or chat.user_id != user.user_id:
        return ActiveResearchResponse(has_active_research=False)

    # Get most recent active research session for this chat
    rs = await rs_service.get_active_session_by_chat(chat_id, user.user_id)

    if rs is None:
        return ActiveResearchResponse(has_active_research=False)

    # Get last sequence number from events
    events = await event_service.get_events_since_sequence(  # type: ignore[attr-defined]
        research_session_id=rs.id,
        since_sequence=0,
        limit=100000,
    )
    events_data = event_service.events_to_list(events)  # type: ignore[attr-defined]
    last_seq = 0
    for ev in events_data:
        seq = ev.get("sequenceNumber") or ev.get("sequence_number")
        if seq is not None:
            with contextlib.suppress(ValueError, TypeError):
                last_seq = max(last_seq, int(seq))

    status_value = status_str(rs.status)
    is_in_progress = status_value == ResearchStatus.IN_PROGRESS.value

    query_mode = rs.execution_state.get("query_mode") if rs.execution_state else None

    return ActiveResearchResponse(
        has_active_research=is_in_progress,
        session_id=rs.id,
        status=status_value,
        last_sequence_number=last_seq,
        query=rs.query,
        query_mode=query_mode,
        started_at=rs.started_at.isoformat() if rs.started_at else None,
    )


class ResearchEventsResponse(BaseModel):
    """Response for fetching research events."""

    events: list[dict[str, Any]]
    session_status: str
    has_more: bool  # True if session still IN_PROGRESS


@router.get("/{chat_id}/research/{session_id}/events")
async def get_research_events(
    chat_id: UUID,
    session_id: UUID,
    user: CurrentUser,
    since_sequence: int = Query(0, alias="sinceSequence"),
    limit: int = Query(100),
    chat_service: IChatService = Depends(get_chat_service),
    rs_service: IResearchSessionService = Depends(get_research_session_service),
    event_service: IResearchEventService = Depends(get_research_event_service),
) -> ResearchEventsResponse:
    """Get events for reconnection.

    Fetches events with sequence_number > since_sequence.
    Frontend polls this every 2 seconds during reconnection until has_more=False.
    """
    # Verify session belongs to this chat and user owns the chat (security)
    rs = await rs_service.get(session_id, chat_id=chat_id)
    if not rs:
        raise NotFoundError("ResearchSession", str(session_id))

    # Verify user has access to this chat
    chat = await chat_service.get_by_id(chat_id)
    if chat and chat.user_id != user.user_id:
        raise NotFoundError("Chat", str(chat_id))

    # Fetch events since sequence number
    events = await event_service.get_events_since_sequence(  # type: ignore[attr-defined]
        research_session_id=session_id,
        since_sequence=since_sequence,
        limit=limit,
    )

    # Convert to frontend format
    events_data = event_service.events_to_list(events)  # type: ignore[attr-defined]

    status_val = status_str(rs.status)

    return ResearchEventsResponse(
        events=events_data,
        session_status=status_val,
        has_more=status_val == ResearchStatus.IN_PROGRESS.value,
    )


class ResearchStateResponse(BaseModel):
    """Response for getting final research state."""

    session_id: UUID
    status: str
    query: str | None = None
    plan: dict[str, Any] | None = None
    observations: list[dict[str, Any]] | None = None
    current_step_index: int | None = None
    plan_iterations: int | None = None
    final_report: str | None = None
    completed_at: str | None = None


@router.get("/{chat_id}/research/{session_id}/state")
async def get_research_state(
    chat_id: UUID,
    session_id: UUID,
    user: CurrentUser,
    chat_service: IChatService = Depends(get_chat_service),
    rs_service: IResearchSessionService = Depends(get_research_session_service),
    message_service: IMessageService = Depends(get_message_service),
) -> ResearchStateResponse:
    """Get final research state for completed session.

    Returns plan, observations, final_report for UI hydration after reconnection.
    """
    # Verify session exists and belongs to this chat
    rs = await rs_service.get(session_id, chat_id=chat_id)
    if not rs:
        raise NotFoundError("ResearchSession", str(session_id))

    # Verify user has access to this chat
    chat = await chat_service.get_by_id(chat_id)
    if chat and chat.user_id != user.user_id:
        raise NotFoundError("Chat", str(chat_id))

    # Get agent message content (final report)
    final_report: str | None = None
    if rs.message_id:
        message = await message_service.get_with_chat(rs.message_id, chat_id)
        if message:
            final_report = message.content

    status_value = status_str(rs.status)

    return ResearchStateResponse(
        session_id=rs.id,
        status=status_value,
        query=rs.query,
        plan=rs.plan,
        observations=rs.observations,
        current_step_index=rs.current_step,
        plan_iterations=None,
        final_report=final_report,
        completed_at=rs.completed_at.isoformat() if rs.completed_at else None,
    )
