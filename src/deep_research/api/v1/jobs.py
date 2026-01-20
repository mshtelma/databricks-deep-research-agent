"""Research job management endpoints.

Provides background job submission, listing, cancellation, and event streaming.

Key Features:
- Jobs run in background (decoupled from HTTP request lifecycle)
- Per-user concurrency limits (default: 2 concurrent jobs)
- SSE streaming for real-time event delivery
- Reconnection support via sinceSequence parameter
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from deep_research.schemas.common import BaseSchema

from deep_research.core.logging_utils import get_logger
from deep_research.db.session import get_db
from deep_research.middleware.auth import CurrentUser
from deep_research.models.research_session import ResearchSession, ResearchStatus
from deep_research.services.job_manager import get_job_manager
from deep_research.services.research_event_service import ResearchEventService

router = APIRouter(prefix="/research/jobs", tags=["Jobs"])
logger = get_logger(__name__)


# =============================================================================
# Request/Response Models
# =============================================================================


class SubmitJobRequest(BaseModel):
    """Request body for submitting a new research job."""

    chat_id: UUID = Field(..., description="Chat to associate the research with")
    query: str = Field(..., min_length=1, max_length=10000, description="Research query")
    query_mode: str = Field(
        default="deep_research",
        pattern="^(simple|web_search|deep_research)$",
        description="Query mode",
    )
    research_depth: str = Field(
        default="auto",
        pattern="^(auto|light|medium|extended)$",
        description="Research depth",
    )
    verify_sources: bool = Field(
        default=True,
        description="Enable citation verification pipeline",
    )


class JobResponse(BaseSchema):
    """Response model for a single job.

    Uses camelCase aliases via BaseSchema for frontend compatibility.
    """

    session_id: UUID = Field(..., description="Research session ID")
    status: str = Field(..., description="Job status")
    query: str = Field(..., description="Research query")
    query_mode: str = Field(..., description="Query mode")
    chat_id: UUID = Field(..., description="Associated chat ID")
    started_at: str | None = Field(None, description="ISO timestamp when job started")
    completed_at: str | None = Field(None, description="ISO timestamp when job completed")
    current_step: int | None = Field(None, description="Current step index")
    total_steps: int | None = Field(None, description="Total number of planned steps")
    error_message: str | None = Field(None, description="Error message if failed")


class JobListResponse(BaseSchema):
    """Response model for listing jobs.

    Uses camelCase aliases via BaseSchema for frontend compatibility.
    """

    jobs: list[JobResponse] = Field(..., description="List of jobs")
    active_count: int = Field(..., description="Number of currently running jobs")
    limit: int = Field(..., description="Max concurrent jobs allowed")
    limit_reached: bool = Field(..., description="Whether user is at concurrency limit")


class CancelJobResponse(BaseSchema):
    """Response model for job cancellation.

    Uses camelCase aliases via BaseSchema for frontend compatibility.
    """

    session_id: UUID = Field(..., description="Cancelled session ID")
    status: str = Field(..., description="New status (cancelled)")


class JobEventResponse(BaseSchema):
    """Response for single event in polling endpoint.

    Uses camelCase aliases via BaseSchema for frontend compatibility.
    """

    id: str
    event_type: str
    timestamp: str
    sequence_number: int | None
    payload: dict[str, Any]


class JobEventsResponse(BaseSchema):
    """Response for fetching job events (polling).

    Uses camelCase aliases via BaseSchema for frontend compatibility.
    """

    events: list[JobEventResponse]
    session_status: str
    has_more: bool  # True if session still IN_PROGRESS


# =============================================================================
# Endpoints
# =============================================================================


@router.post("", response_model=JobResponse)
async def submit_job(
    request: Request,
    body: SubmitJobRequest,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> JobResponse:
    """Submit a new background research job.

    Creates a ResearchSession immediately and starts research in background.
    Returns the session ID for tracking. Use /stream to get events.

    Raises:
        HTTPException 429: If user has reached max concurrent jobs limit.
    """
    job_manager = get_job_manager()

    # Get services from app state
    llm = request.app.state.llm_client
    brave_client = request.app.state.brave_client
    crawler = request.app.state.web_crawler

    # Get conversation history from database
    conversation_history: list[dict[str, str]] = []
    try:
        from deep_research.services.message_service import MessageService

        message_service = MessageService(db)
        conversation_history = await message_service.get_conversation_history(
            body.chat_id, limit=10
        )
        logger.info(
            "JOB_CONVERSATION_HISTORY_LOADED",
            chat_id=str(body.chat_id),
            message_count=len(conversation_history),
        )
    except Exception as e:
        logger.warning(
            "JOB_CONVERSATION_HISTORY_FAILED",
            error=str(e),
        )

    # Get user's system instructions from preferences
    system_instructions: str | None = None
    try:
        from deep_research.services.preferences_service import PreferencesService

        preferences_service = PreferencesService(db)
        system_instructions = await preferences_service.get_system_instructions(
            user.user_id
        )
    except Exception as e:
        logger.warning(
            "JOB_PREFERENCES_LOAD_FAILED",
            error=str(e),
        )

    # Submit job
    session = await job_manager.submit_job(
        user_id=user.user_id,
        chat_id=body.chat_id,
        query=body.query,
        query_mode=body.query_mode,
        research_depth=body.research_depth,
        verify_sources=body.verify_sources,
        llm=llm,
        brave_client=brave_client,
        crawler=crawler,
        conversation_history=conversation_history,
        system_instructions=system_instructions,
        db=db,
    )

    logger.info(
        "JOB_SUBMITTED_API",
        session_id=str(session.id),
        user_id=user.user_id,
    )

    return _session_to_response(session)


@router.get("", response_model=JobListResponse)
async def list_jobs(
    user: CurrentUser,
    status: str | None = Query(
        None,
        pattern="^(in_progress|completed|failed|cancelled)$",
        description="Filter by status",
    ),
    limit: int = Query(50, ge=1, le=100, description="Max jobs to return"),
    db: AsyncSession = Depends(get_db),
) -> JobListResponse:
    """List user's research jobs.

    Returns jobs ordered by creation time (newest first).
    Use status filter to get only running, completed, or failed jobs.
    """
    job_manager = get_job_manager()
    jobs = await job_manager.get_user_jobs(user.user_id, status, db, limit=limit)

    # Count active jobs
    active_count = sum(
        1
        for j in jobs
        if (j.status.value if hasattr(j.status, "value") else j.status)
        == ResearchStatus.IN_PROGRESS.value
    )

    from deep_research.services.job_manager import get_max_concurrent_jobs

    max_concurrent = get_max_concurrent_jobs()
    return JobListResponse(
        jobs=[_session_to_response(j) for j in jobs],
        active_count=active_count,
        limit=max_concurrent,
        limit_reached=active_count >= max_concurrent,
    )


@router.get("/{session_id}", response_model=JobResponse)
async def get_job(
    session_id: UUID,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> JobResponse:
    """Get details for a specific job.

    Returns current status, progress, and metadata.
    """
    session = await db.get(ResearchSession, session_id)

    if not session or session.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="Job not found")

    return _session_to_response(session)


@router.delete("/{session_id}", response_model=CancelJobResponse)
async def cancel_job(
    session_id: UUID,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> CancelJobResponse:
    """Cancel a running job.

    Stops the research operation. Partial results are preserved in the database.
    """
    job_manager = get_job_manager()
    success = await job_manager.cancel_job(session_id, user.user_id, db)

    if not success:
        raise HTTPException(status_code=404, detail="Job not found or access denied")

    logger.info(
        "JOB_CANCELLED_API",
        session_id=str(session_id),
        user_id=user.user_id,
    )

    return CancelJobResponse(
        session_id=session_id,
        status="cancelled",
    )


@router.get("/{session_id}/stream")
async def stream_job_events(
    session_id: UUID,
    user: CurrentUser,
    since_sequence: int = Query(
        0,
        alias="sinceSequence",
        ge=0,
        description="Resume from this sequence number",
    ),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Stream events for a job via Server-Sent Events.

    Reconnection-safe: pass sinceSequence to resume from last received event.
    Events are replayed in order, then new events stream as they occur.

    Event Format:
    ```
    data: {"eventType": "...", "sequenceNumber": N, "payload": {...}}
    ```

    Special event when job completes:
    ```
    data: {"eventType": "job_completed", "status": "completed|failed|cancelled"}
    ```
    """
    # Verify ownership
    session = await db.get(ResearchSession, session_id)
    if not session or session.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="Job not found")

    logger.info(
        "JOB_STREAM_STARTED",
        session_id=str(session_id),
        since_sequence=since_sequence,
    )

    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events from the research session."""
        event_service = ResearchEventService(db)
        last_seq = since_sequence
        poll_interval = 0.5  # seconds

        while True:
            # Get new events since last sequence
            events = await event_service.get_events_since_sequence(
                research_session_id=session_id,
                since_sequence=last_seq,
                limit=50,
            )

            # Emit each event
            for event in events:
                event_dict = event_service.event_to_dict(event)
                yield f"data: {json.dumps(event_dict)}\n\n"
                if event.sequence_number:
                    last_seq = event.sequence_number

            # Refresh session status
            await db.refresh(session)

            # Check if job completed
            status_val = (
                session.status.value
                if hasattr(session.status, "value")
                else session.status
            )

            if status_val != ResearchStatus.IN_PROGRESS.value:
                # Emit final status event and close
                final_event = {
                    "eventType": "job_completed",
                    "status": status_val,
                }
                yield f"data: {json.dumps(final_event)}\n\n"
                logger.info(
                    "JOB_STREAM_COMPLETED",
                    session_id=str(session_id),
                    status=status_val,
                )
                break

            # Wait before polling again
            await asyncio.sleep(poll_interval)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/{session_id}/events", response_model=JobEventsResponse)
async def get_job_events(
    session_id: UUID,
    user: CurrentUser,
    since_sequence: int = Query(
        0,
        alias="sinceSequence",
        ge=0,
        description="Return events after this sequence number",
    ),
    limit: int = Query(100, ge=1, le=500, description="Max events to return"),
    db: AsyncSession = Depends(get_db),
) -> JobEventsResponse:
    """Get events for a job (polling endpoint).

    Alternative to SSE streaming for clients that don't support SSE.
    Fetches events with sequence_number > sinceSequence.

    Returns has_more=True if job is still in progress.
    """
    # Verify ownership
    session = await db.get(ResearchSession, session_id)
    if not session or session.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="Job not found")

    # Get events
    event_service = ResearchEventService(db)
    events = await event_service.get_events_since_sequence(
        research_session_id=session_id,
        since_sequence=since_sequence,
        limit=limit,
    )

    # Convert to response format
    event_responses = [
        JobEventResponse(
            id=str(e.id),
            event_type=e.event_type,
            timestamp=e.timestamp.isoformat(),
            sequence_number=e.sequence_number,
            payload=e.payload,
        )
        for e in events
    ]

    status_val = (
        session.status.value if hasattr(session.status, "value") else session.status
    )

    return JobEventsResponse(
        events=event_responses,
        session_status=status_val,
        has_more=status_val == ResearchStatus.IN_PROGRESS.value,
    )


@router.get("/chat/{chat_id}/active", response_model=JobResponse | None)
async def get_chat_active_job(
    chat_id: UUID,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> JobResponse | None:
    """Get the active job for a specific chat.

    Returns the in-progress job if one exists, otherwise None.
    Used by frontend to detect if research is already running for this chat.
    """
    job_manager = get_job_manager()
    session = await job_manager.get_chat_active_job(chat_id, user.user_id, db)

    if not session:
        return None

    return _session_to_response(session)


# =============================================================================
# Helper Functions
# =============================================================================


def _session_to_response(session: ResearchSession) -> JobResponse:
    """Convert ResearchSession to API response.

    Args:
        session: ResearchSession model instance.

    Returns:
        JobResponse with formatted fields.
    """
    plan = session.plan or {}
    steps = plan.get("steps", [])

    # Handle status as either enum or string
    status_val = (
        session.status.value if hasattr(session.status, "value") else session.status
    )

    return JobResponse(
        session_id=session.id,
        status=status_val,
        query=session.query,
        query_mode=session.query_mode,
        chat_id=session.chat_id,
        started_at=session.started_at.isoformat() if session.started_at else None,
        completed_at=session.completed_at.isoformat() if session.completed_at else None,
        current_step=session.current_step_index,
        total_steps=len(steps) if steps else None,
        error_message=session.error_message,
    )
