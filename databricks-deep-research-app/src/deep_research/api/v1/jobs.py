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
from contextlib import AbstractAsyncContextManager, nullcontext
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from deep_research.api.v1.utils import verify_chat_access
from deep_research.api.v1.utils.transformers import status_str
from deep_research.core.config import Settings, get_settings
from deep_research.core.deps import (
    get_research_event_service,
    get_storage_optional,
)
from deep_research.core.logging_utils import get_logger
from deep_research.db.session import get_db
from deep_research.middleware.auth import CurrentUser
from deep_research.models.research_session import ResearchSession, ResearchStatus
from deep_research.schemas.common import BaseSchema
from deep_research.schemas.source_scope import SourceScope
from deep_research.services._impl_factory import (
    make_message_service,
    make_preferences_service,
    make_research_event_service,
)
from deep_research.services._protocols import IResearchEventService
from deep_research.services.job_manager import get_job_manager

router = APIRouter(prefix="/research/jobs", tags=["Jobs"])
logger = get_logger(__name__)


# =============================================================================
# Tolerant sub-read helpers
# =============================================================================
#
# `submit_job` reads conversation history and user preferences before queueing
# the job. Both reads are *tolerant*: if they fail the job must still submit.
# Previously these blocks caught every exception without rolling back the
# shared `AsyncSession`, so an INSERT failure inside `PreferencesService.
# get_preferences` (e.g., FK violation on `user_preferences.user_id -> users`
# for a fresh cached-mode user) poisoned the transaction and the downstream
# `_count_user_active_jobs` SELECT failed with
# `InFailedSQLTransactionError: current transaction is aborted`.
#
# Fix:
#  - Route through `make_message_service` / `make_preferences_service` so
#    cached deployments never hit the SQL tables from this path.
#  - In the legacy SQL branch, wrap each read in SAVEPOINT so a failure
#    rolls back only the savepoint and the outer request transaction stays
#    usable.


def _tolerant_tx_guard(
    db: AsyncSession, settings: Settings
) -> AbstractAsyncContextManager[Any]:
    """Return a savepoint context on the legacy SQL branch, else a no-op."""
    if settings.storage_service_impl == "sqlalchemy_legacy":
        return db.begin_nested()
    return nullcontext()


async def _load_conversation_history(
    db: AsyncSession,
    settings: Settings,
    storage_stack: Any,
    chat_id: UUID,
    is_draft: bool,
) -> list[dict[str, str]]:
    """Tolerantly load the last 10 messages for prompt context.

    Returns `[]` on any failure or when the chat is a brand-new draft
    (the cached backend has not hydrated a document for it yet, and a
    request would log a misleading `chat does not exist` warning).
    """
    if is_draft:
        return []

    try:
        async with _tolerant_tx_guard(db, settings):
            message_service = make_message_service(
                settings, storage_stack, session=db
            )
            history = await message_service.get_conversation_history(
                chat_id, limit=10
            )
            logger.info(
                "JOB_CONVERSATION_HISTORY_LOADED",
                chat_id=str(chat_id),
                message_count=len(history),
            )
            return history
    except Exception as e:
        orig = getattr(e, "orig", None)
        logger.warning(
            "JOB_CONVERSATION_HISTORY_FAILED",
            chat_id=str(chat_id),
            error=str(e),
            error_type=type(e).__name__,
            sqlstate=getattr(orig, "sqlstate", None),
        )
        return []


async def _load_system_instructions(
    db: AsyncSession,
    settings: Settings,
    storage_stack: Any,
    user_id: str,
) -> str | None:
    """Tolerantly load the user's system-instruction preference."""
    try:
        async with _tolerant_tx_guard(db, settings):
            preferences_service = make_preferences_service(
                settings, storage_stack, session=db
            )
            return await preferences_service.get_system_instructions(user_id)
    except Exception as e:
        orig = getattr(e, "orig", None)
        logger.warning(
            "JOB_PREFERENCES_LOAD_FAILED",
            user_id=user_id,
            error=str(e),
            error_type=type(e).__name__,
            sqlstate=getattr(orig, "sqlstate", None),
        )
        return None


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
    output_type: str | None = Field(
        default=None,
        description="Output type for structured output (e.g., 'meeting_prep'). If not specified, uses default synthesis_report.",
    )
    # Source selection fields (Feature 008)
    source_scope: SourceScope | None = Field(
        default=None,
        description="Source scope: enterprise_only, web_only, or all. Defaults to all if not specified.",
    )
    enabled_sources: list[str] | None = Field(
        default=None,
        description="Whitelist of source IDs to use. If None, all sources in scope are used.",
    )
    disabled_sources: list[str] = Field(
        default=[],
        description="Blacklist of source IDs to exclude from research.",
    )
    file_ids: list[str] | None = Field(
        default=None,
        description="Uploaded file IDs to include in research context.",
    )
    agent_id: str | None = Field(
        default=None,
        description="Custom agent ID to use for this research job.",
    )
    turn_intent: str = Field(
        default="auto",
        pattern="^(auto|chat|research)$",
        description=(
            "Per-turn routing for custom-agent chats: 'auto' (classify intent), "
            "'chat' (answer from already-gathered data, no re-run), or 'research' "
            "(force a fresh agent run). Ignored unless agent_id is set and the chat "
            "has prior research."
        ),
    )
    enable_plan_review: bool = Field(
        default=False,
        description="If true, pause after plan creation for user review.",
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
    # Get PluginManager for custom phase mode (may be None if not available)
    plugin_manager = getattr(request.app.state, "plugin_manager", None)

    # Verify user owns the chat (or it's a new draft)
    is_draft, _ = await verify_chat_access(
        body.chat_id, user.user_id, db, request=request
    )

    # Load tolerant sub-reads via the service factory (cached mode skips SQL).
    settings = get_settings()
    storage_stack = get_storage_optional(request)

    conversation_history = await _load_conversation_history(
        db, settings, storage_stack, body.chat_id, is_draft=is_draft
    )
    system_instructions = await _load_system_instructions(
        db, settings, storage_stack, user.user_id
    )

    # Get OBO token for enterprise data source authentication (007-enterprise Phase 2)
    user_token = getattr(request.state, "obo_token", None)

    # PR3a: forward the in-process HITL broker through to OrchestrationConfig.
    # ``app.state.approval_broker`` is set in main.py lifespan when HITL is
    # configured; absent on entry points without FastAPI app state (e.g.
    # agent_server CLI) where HITL gating is disabled by design.
    approval_broker = getattr(request.app.state, "approval_broker", None)

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
        output_type=body.output_type,
        source_scope=body.source_scope.value if body.source_scope else None,
        enabled_sources=body.enabled_sources,
        disabled_sources=body.disabled_sources,
        plugin_manager=plugin_manager,
        db=db,
        user_token=user_token,
        file_ids=body.file_ids,
        agent_id=body.agent_id,
        turn_intent=body.turn_intent,
        enable_plan_review=body.enable_plan_review,
        approval_broker=approval_broker,
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
        1 for j in jobs if status_str(j.status) == ResearchStatus.IN_PROGRESS.value
    )

    from deep_research.services.job_manager import get_max_concurrent_jobs

    max_concurrent = get_max_concurrent_jobs()
    return JobListResponse(
        jobs=[_session_to_response(j) for j in jobs],
        active_count=active_count,
        limit=max_concurrent,
        limit_reached=active_count >= max_concurrent,
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

    IMPORTANT: this route MUST be declared before any `/jobs/{chat_id}/...`
    route that takes two UUID path params. FastAPI resolves routes in
    declaration order, and the literal prefix ``/chat/`` must take priority
    over UUID-typed ``{chat_id}`` binding that would otherwise 422-fail.
    """
    job_manager = get_job_manager()
    session = await job_manager.get_chat_active_job(chat_id, user.user_id, db)

    if not session:
        return None

    return _session_to_response(session)


@router.get("/{chat_id}/{session_id}", response_model=JobResponse)
async def get_job(
    chat_id: UUID,
    session_id: UUID,
    request: Request,
    user: CurrentUser,
    settings: Settings = Depends(get_settings),
    db: AsyncSession = Depends(get_db),
) -> JobResponse:
    """Get details for a specific job.

    Returns current status, progress, and metadata. ``chat_id`` is part of
    the URL so the server can hydrate the ChatDocument (cached mode) or
    verify ownership (legacy mode) with a single round-trip.
    """
    from deep_research.agent.session_lookup import load_session_control_view

    stack = get_storage_optional(request)
    view = await load_session_control_view(
        chat_id,
        session_id,
        user.user_id,
        settings=settings,
        storage_stack=stack,
        db=db,
    )
    if view is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return _view_to_response(view)


@router.delete("/{chat_id}/{session_id}", response_model=CancelJobResponse)
async def cancel_job(
    chat_id: UUID,
    session_id: UUID,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> CancelJobResponse:
    """Cancel a running job.

    Stops the research operation. Partial results are preserved in the database.
    """
    job_manager = get_job_manager()
    success = await job_manager.cancel_job(chat_id, session_id, user.user_id, db)

    if not success:
        raise HTTPException(status_code=404, detail="Job not found or access denied")

    logger.info(
        "JOB_CANCELLED_API",
        session_id=str(session_id),
        chat_id=str(chat_id),
        user_id=user.user_id,
    )

    return CancelJobResponse(
        session_id=session_id,
        status="cancelled",
    )


@router.get("/{chat_id}/{session_id}/stream")
async def stream_job_events(
    request: Request,
    chat_id: UUID,
    session_id: UUID,
    user: CurrentUser,
    since_sequence: int = Query(
        0,
        alias="sinceSequence",
        ge=0,
        description="Resume from this sequence number",
    ),
    settings: Settings = Depends(get_settings),
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

    Note:
        Validation uses a short-lived session that is released before the
        SSE stream starts — holding the request-scoped ``Depends(get_db)``
        open for minutes caused ``InterfaceError: connection is closed`` at
        cleanup commit when PgBouncer/Lakebase reaped the idle connection.
        The generator uses its own independent session for the streaming
        lifetime.
    """
    from deep_research.agent.session_lookup import load_session_control_view
    from deep_research.db.session import get_session_maker

    # Retrieve the storage stack once so validation + polling share the same
    # instance. In cached mode the helper reads the ChatDocument through it;
    # in legacy mode the helper falls back to db.get and the stack is unused.
    stack = get_storage_optional(request)

    session_maker = get_session_maker()
    async with session_maker() as validation_db:
        view = await load_session_control_view(
            chat_id,
            session_id,
            user.user_id,
            settings=settings,
            storage_stack=stack,
            db=validation_db,
        )
        if view is None:
            raise HTTPException(status_code=404, detail="Job not found")

    # Capture validated values for closure (validation_db is released above)
    validated_session_id = session_id
    validated_chat_id = chat_id

    logger.info(
        "JOB_STREAM_STARTED",
        session_id=str(session_id),
        chat_id=str(chat_id),
        since_sequence=since_sequence,
    )

    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events using independent session.

        Creates ONE independent session for the entire generator lifetime.
        This session is NOT tied to the HTTP request lifecycle, preventing
        CancelledError from propagating to the connection pool on client disconnect.
        """
        from deep_research.db.session import get_session_maker

        session_maker = get_session_maker()
        last_seq = since_sequence
        poll_interval = 0.5  # seconds

        # Single session for entire generator - properly cleaned up by context manager
        # F-RE: route event reads through the factory so cached impl is used when
        # STORAGE_SERVICE_IMPL=cached. Session status is resolved through
        # load_session_control_view which handles both cached (index →
        # ChatDocument) and legacy (db.get) modes.
        async with session_maker() as independent_db:
            try:
                while True:
                    event_service = make_research_event_service(
                        settings, stack, session=independent_db
                    )
                    events = await event_service.get_events_since_sequence(
                        research_session_id=validated_session_id,
                        since_sequence=last_seq,
                        limit=50,
                    )

                    # Emit each event
                    for event in events:
                        event_dict = event_service.event_to_dict(event)
                        yield f"data: {json.dumps(event_dict)}\n\n"
                        if event.sequence_number:
                            last_seq = event.sequence_number

                    current_view = await load_session_control_view(
                        validated_chat_id,
                        validated_session_id,
                        user.user_id,
                        settings=settings,
                        storage_stack=stack,
                        db=independent_db,
                    )
                    if current_view is None:
                        # Session deleted - client will need to reload
                        break

                    status_val = current_view.status.value

                    if status_val != ResearchStatus.IN_PROGRESS.value:
                        # Emit final status event and close
                        final_event = {
                            "eventType": "job_completed",
                            "status": status_val,
                        }
                        yield f"data: {json.dumps(final_event)}\n\n"
                        logger.info(
                            "JOB_STREAM_COMPLETED",
                            session_id=str(validated_session_id),
                            status=status_val,
                        )
                        break

                    # Wait before polling again
                    await asyncio.sleep(poll_interval)

            except asyncio.CancelledError:
                # Client disconnected - graceful cleanup, no error logging
                logger.info(
                    "JOB_STREAM_CLIENT_DISCONNECTED",
                    session_id=str(validated_session_id),
                )
                # Don't re-raise - let context manager clean up session properly
            except Exception as e:
                logger.error(
                    "JOB_STREAM_ERROR",
                    session_id=str(validated_session_id),
                    error=str(e),
                    exc_info=True,
                )
                # Yield error event before exiting
                error_event = {"eventType": "error", "message": "Stream error occurred"}
                yield f"data: {json.dumps(error_event)}\n\n"
                # Don't re-raise - let context manager clean up session properly

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/{chat_id}/{session_id}/events", response_model=JobEventsResponse)
async def get_job_events(
    chat_id: UUID,
    session_id: UUID,
    request: Request,
    user: CurrentUser,
    since_sequence: int = Query(
        0,
        alias="sinceSequence",
        ge=0,
        description="Return events after this sequence number",
    ),
    limit: int = Query(100, ge=1, le=500, description="Max events to return"),
    settings: Settings = Depends(get_settings),
    db: AsyncSession = Depends(get_db),
    event_service: IResearchEventService = Depends(get_research_event_service),
) -> JobEventsResponse:
    """Get events for a job (polling endpoint).

    Alternative to SSE streaming for clients that don't support SSE.
    Fetches events with sequence_number > sinceSequence.

    Returns has_more=True if job is still in progress.
    """
    from deep_research.agent.session_lookup import load_session_control_view

    # Verify ownership via the unified session lookup (no legacy ORM fetch).
    stack = get_storage_optional(request)
    view = await load_session_control_view(
        chat_id,
        session_id,
        user.user_id,
        settings=settings,
        storage_stack=stack,
        db=db,
    )
    if view is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # Get events (F-RE: routed through factory)
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

    status_val = view.status.value

    return JobEventsResponse(
        events=event_responses,
        session_status=status_val,
        has_more=status_val == ResearchStatus.IN_PROGRESS.value,
    )


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

    status_val = status_str(session.status)

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


def _view_to_response(view: Any) -> JobResponse:
    """Convert a SessionControlView to API response.

    Used by the `GET /jobs/{chat_id}/{session_id}` endpoint after the
    unified session lookup — no second ChatDocument walk required.
    """
    return JobResponse(
        session_id=view.id,
        status=view.status.value,
        query=view.query,
        query_mode=view.query_mode,
        chat_id=view.chat_id,
        started_at=view.started_at.isoformat() if view.started_at else None,
        completed_at=view.completed_at.isoformat() if view.completed_at else None,
        current_step=view.current_step,
        total_steps=view.total_steps,
        error_message=view.error_message,
    )
