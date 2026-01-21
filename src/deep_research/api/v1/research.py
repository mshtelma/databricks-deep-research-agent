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

import logging
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from deep_research.api.v1.utils import verify_chat_access
from deep_research.core.exceptions import AuthorizationError, NotFoundError
from deep_research.db.session import get_db
from deep_research.middleware.auth import CurrentUser
from deep_research.schemas.research import CancelResearchResponse
from deep_research.services.chat_service import ChatService
from deep_research.services.research_session_service import ResearchSessionService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/{session_id}/cancel", response_model=CancelResearchResponse)
async def cancel_research(
    session_id: UUID,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> CancelResearchResponse:
    """Cancel in-progress research.

    Stops the research operation within 2 seconds. Partial results are preserved.
    """
    service = ResearchSessionService(db)
    session = await service.cancel(session_id)

    if not session:
        raise NotFoundError("ResearchSession", str(session_id))

    # Get partial results if available
    partial_results = None
    if session.observations:
        # Join all observations collected so far
        partial_results = "\n\n".join(
            obs.get("observation", "") for obs in session.observations if obs.get("observation")
        )

    await db.commit()

    return CancelResearchResponse(
        session_id=session_id,
        status="cancelled",
        partial_results=partial_results if partial_results else None,
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
    db: AsyncSession = Depends(get_db),
) -> ActiveResearchResponse:
    """Check if there's an in-progress research session for this chat.

    Frontend calls this on page load to detect if reconnection is needed.
    Returns session info if research is in progress, otherwise has_active_research=False.
    """
    from sqlalchemy import func, select

    from deep_research.models.message import Message
    from deep_research.models.research_event import ResearchEvent
    from deep_research.models.research_session import ResearchSession, ResearchStatus

    # Verify user has access to this chat
    try:
        await verify_chat_access(chat_id, user.user_id, db)
    except AuthorizationError:
        return ActiveResearchResponse(has_active_research=False)

    # Get most recent research session for this chat
    stmt = (
        select(ResearchSession)
        .join(Message, ResearchSession.message_id == Message.id)
        .where(Message.chat_id == chat_id)
        .order_by(ResearchSession.created_at.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    session = result.scalar_one_or_none()

    if not session:
        return ActiveResearchResponse(has_active_research=False)

    # Get last sequence number from events
    last_seq = await db.scalar(
        select(func.max(ResearchEvent.sequence_number)).where(
            ResearchEvent.research_session_id == session.id
        )
    )

    # Handle status as either enum or string (DB may return string directly)
    status_value = session.status.value if hasattr(session.status, "value") else session.status
    is_in_progress = status_value == ResearchStatus.IN_PROGRESS.value

    return ActiveResearchResponse(
        has_active_research=is_in_progress,
        session_id=session.id,
        status=status_value,
        last_sequence_number=last_seq or 0,
        query=session.query,
        query_mode=session.query_mode,
        started_at=session.started_at.isoformat() if session.started_at else None,
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
    db: AsyncSession = Depends(get_db),
) -> ResearchEventsResponse:
    """Get events for reconnection.

    Fetches events with sequence_number > since_sequence.
    Frontend polls this every 2 seconds during reconnection until has_more=False.
    """
    from deep_research.models.message import Message
    from deep_research.models.research_session import ResearchSession, ResearchStatus
    from deep_research.services.research_event_service import ResearchEventService

    # Verify session belongs to this chat (security)
    session = await db.get(ResearchSession, session_id)
    if not session:
        raise NotFoundError("ResearchSession", str(session_id))

    # Verify chat ownership via message
    message = await db.get(Message, session.message_id)
    if not message or message.chat_id != chat_id:
        raise AuthorizationError(f"Access denied to session {session_id}")

    # Verify user has access to this chat
    chat_service = ChatService(db)
    chat = await chat_service.get_by_id(chat_id)
    if chat and chat.user_id != user.user_id:
        raise AuthorizationError(f"Access denied to chat {chat_id}")

    # Fetch events since sequence number
    event_service = ResearchEventService(db)
    events = await event_service.get_events_since_sequence(
        research_session_id=session_id,
        since_sequence=since_sequence,
        limit=limit,
    )

    # Convert to frontend format
    events_data = event_service.events_to_list(events)

    # Handle status as either enum or string
    status_val = session.status.value if hasattr(session.status, "value") else session.status

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
    db: AsyncSession = Depends(get_db),
) -> ResearchStateResponse:
    """Get final research state for completed session.

    Returns plan, observations, final_report for UI hydration after reconnection.
    """
    from deep_research.models.message import Message
    from deep_research.models.research_session import ResearchSession

    # Verify session exists
    session = await db.get(ResearchSession, session_id)
    if not session:
        raise NotFoundError("ResearchSession", str(session_id))

    # Get agent message content
    message = await db.get(Message, session.message_id)
    if not message or message.chat_id != chat_id:
        raise AuthorizationError(f"Access denied to session {session_id}")

    # Verify user has access to this chat
    chat_service = ChatService(db)
    chat = await chat_service.get_by_id(chat_id)
    if chat and chat.user_id != user.user_id:
        raise AuthorizationError(f"Access denied to chat {chat_id}")

    # Handle status as either enum or string
    status_str = session.status.value if hasattr(session.status, "value") else session.status

    return ResearchStateResponse(
        session_id=session.id,
        status=status_str,
        query=session.query,
        plan=session.plan,
        observations=session.observations,
        current_step_index=session.current_step_index,
        plan_iterations=session.plan_iterations,
        final_report=message.content,
        completed_at=session.completed_at.isoformat() if session.completed_at else None,
    )
