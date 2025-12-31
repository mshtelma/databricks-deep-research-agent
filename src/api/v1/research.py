"""Research streaming endpoints.

Key Design: Deferred Database Materialization
- UUIDs are generated in memory when streaming starts
- NO database writes occur until synthesis completes successfully
- All data is persisted atomically in the orchestrator
- Benefits: No orphaned records, no cleanup needed on failure
"""

import contextlib
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any
from uuid import UUID, uuid4

import mlflow
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.agent.orchestrator import OrchestrationConfig, stream_research
from src.agent.tools.web_crawler import WebCrawler
from src.core.exceptions import AuthorizationError, NotFoundError
from src.db.session import get_db
from src.middleware.auth import CurrentUser
from src.schemas.research import CancelResearchResponse
from src.schemas.streaming import StreamEvent
from src.services.chat_service import ChatService
from src.services.llm.client import LLMClient
from src.services.preferences_service import PreferencesService
from src.services.research_session_service import ResearchSessionService
from src.services.search.brave import BraveSearchClient

# Note: MessageService and ResearchSessionService are still imported for:
# - MessageService: get_conversation_history()
# - ResearchSessionService: cancel_research()

router = APIRouter()
logger = logging.getLogger(__name__)


async def _verify_chat_access(
    chat_id: UUID, user_id: str, db: AsyncSession
) -> bool:
    """Verify user can access this chat for streaming.

    Authorization logic for draft chat support:
    - If chat doesn't exist: allow (draft chat flow) -> return True
    - If chat exists and owned by user: allow -> return False
    - If chat exists but owned by another user: reject with 403

    Args:
        chat_id: Chat UUID to check.
        user_id: Current user's ID.
        db: Database session.

    Returns:
        True if chat is a draft (doesn't exist), False if persisted.

    Raises:
        AuthorizationError: If chat exists but belongs to another user.
    """
    chat_service = ChatService(db)
    chat = await chat_service.get_by_id(chat_id)

    if chat is None:
        # Chat doesn't exist - this is a draft, allow streaming
        logger.info(f"Chat {chat_id} is a draft (not in DB), allowing stream")
        return True

    if chat.user_id != user_id:
        # Chat exists but belongs to another user - reject
        logger.warning(
            f"User {user_id} attempted to access chat {chat_id} owned by {chat.user_id}"
        )
        raise AuthorizationError(f"Access denied to chat {chat_id}")

    # Chat exists and user owns it
    return False


async def _verify_chat_ownership(
    chat_id: UUID, user_id: str, db: AsyncSession
) -> None:
    """Verify user owns the chat. Raises NotFoundError if not.

    DEPRECATED: Use _verify_chat_access() for stream endpoints.
    Kept for backwards compatibility with other endpoints.
    """
    chat_service = ChatService(db)
    chat = await chat_service.get(chat_id, user_id)
    if not chat:
        raise NotFoundError("Chat", str(chat_id))


class ConversationMessage(BaseModel):
    """A message in conversation history."""

    role: str
    content: str


@router.get("/{chat_id}/stream")
async def stream_research_endpoint(
    request: Request,
    chat_id: UUID,
    user: CurrentUser,
    query: str = Query(default=""),
    research_depth: str = Query(default="auto", pattern="^(auto|light|medium|extended)$"),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Stream research progress (SSE).

    Server-Sent Events stream for real-time multi-agent research progress.

    Event Flow (5-Agent Architecture):
    - agent_started (coordinator)
    - agent_completed (coordinator)
    - [clarification_needed]? (if query is ambiguous)
    - agent_started (background_investigator)
    - agent_completed (background_investigator)
    - agent_started (planner)
    - plan_created (research plan with steps)
    - agent_completed (planner)
    - [RESEARCH LOOP]:
        - step_started
        - agent_started (researcher)
        - agent_completed (researcher)
        - step_completed
        - agent_started (reflector)
        - reflection_decision (CONTINUE, ADJUST, or COMPLETE)
        - agent_completed (reflector)
    - synthesis_started
    - agent_started (synthesizer)
    - synthesis_progress* (streaming content chunks)
    - agent_completed (synthesizer)
    - research_completed
    - persistence_completed (after successful DB write)
    """
    # Verify user can access chat (returns True if draft, False if persisted)
    is_draft = await _verify_chat_access(chat_id, user.user_id, db)

    # Get services from app state
    llm: LLMClient = request.app.state.llm_client
    brave_client: BraveSearchClient = request.app.state.brave_client
    crawler: WebCrawler = request.app.state.web_crawler

    # Get conversation history from database if available
    conversation_history: list[dict[str, str]] = []
    try:
        from src.services.message_service import MessageService

        message_service = MessageService(db)
        conversation_history = await message_service.get_conversation_history(
            chat_id, limit=10
        )
        logger.info(
            f"Loaded {len(conversation_history)} messages from chat {chat_id}"
        )
    except Exception as e:
        # If we can't load history (e.g., table doesn't exist), continue without it
        logger.warning(f"Could not load conversation history: {e}")

    # Get user's system instructions from preferences
    system_instructions: str | None = None
    try:
        preferences_service = PreferencesService(db)
        system_instructions = await preferences_service.get_system_instructions(
            user.user_id
        )
    except Exception as e:
        logger.warning(f"Could not load user preferences: {e}")

    # DEFERRED PERSISTENCE: Generate UUIDs in memory but DON'T persist yet
    # Database writes happen in orchestrator only AFTER synthesis succeeds
    agent_message_id = uuid4()
    research_session_id = uuid4()

    logger.info(
        f"Starting research with pre-generated IDs: message={agent_message_id} session={research_session_id}"
    )

    async def generate_sse_events() -> AsyncGenerator[str, None]:
        """Generate SSE events from the research orchestrator.

        Persistence is handled by the orchestrator after synthesis completes.
        This function only handles event streaming.
        """

        def format_event(event: StreamEvent | str) -> str:
            """Format an event for SSE."""
            if isinstance(event, str):
                # Raw string content (shouldn't happen often)
                return f"data: {json.dumps({'event_type': 'content', 'content': event})}\n\n"
            else:
                # Pydantic model - convert to dict
                event_dict = _event_to_dict(event)
                return f"data: {json.dumps(event_dict)}\n\n"

        try:
            # Emit research_started with pre-generated IDs for frontend optimistic updates
            research_started_event = {
                "event_type": "research_started",
                "message_id": str(agent_message_id),
                "research_session_id": str(research_session_id),
            }
            yield f"data: {json.dumps(research_started_event)}\n\n"

            # Create orchestration config with pre-generated UUIDs
            config = OrchestrationConfig(
                research_depth=research_depth,
                system_instructions=system_instructions,
                message_id=agent_message_id,
                research_session_id=research_session_id,
                is_draft=is_draft,
            )

            # Stream events - orchestrator handles persistence on success
            async for event in stream_research(
                query=query,
                llm=llm,
                brave_client=brave_client,
                crawler=crawler,
                conversation_history=conversation_history,
                user_id=user.user_id,
                chat_id=str(chat_id),
                config=config,
                db=db,
            ):
                yield format_event(event)

            # Orchestrator persists data on success - nothing to do here

        except Exception as e:
            logger.exception(f"Error during research stream: {e}")
            # No cleanup needed - nothing was written to DB
            error_event = {
                "event_type": "error",
                "error_code": "STREAM_ERROR",
                "error_message": str(e),
                "recoverable": False,
            }
            yield f"data: {json.dumps(error_event)}\n\n"
        finally:
            # Flush async traces before response ends
            with contextlib.suppress(Exception):
                mlflow.flush_trace_async_logging()

    return StreamingResponse(
        generate_sse_events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _event_to_dict(event: StreamEvent) -> dict[str, Any]:
    """Convert a StreamEvent to a dictionary for JSON serialization."""
    from datetime import datetime

    # Get the event type from the class name or event_type field
    event_dict = event.model_dump(by_alias=True)

    # Ensure event_type is set
    if "event_type" not in event_dict:
        # Derive from class name: AgentStartedEvent -> agent_started
        class_name = type(event).__name__
        event_type = class_name.replace("Event", "")
        # Convert CamelCase to snake_case
        import re

        event_type = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", event_type)
        event_type = re.sub("([a-z0-9])([A-Z])", r"\1_\2", event_type).lower()
        event_dict["event_type"] = event_type

    # Handle special type serialization (UUID, datetime)
    def serialize_value(value: Any) -> Any:
        if isinstance(value, UUID):
            return str(value)
        elif isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, dict):
            return {k: serialize_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [serialize_value(v) for v in value]
        return value

    event_dict = {k: serialize_value(v) for k, v in event_dict.items()}

    return event_dict


@router.post("/{chat_id}/stream")
async def stream_research_with_history(
    request: Request,
    chat_id: UUID,
    user: CurrentUser,
    query: str = Query(default=""),
    research_depth: str = Query(default="auto", pattern="^(auto|light|medium|extended)$"),
    history: list[ConversationMessage] | None = None,
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Stream research with explicit conversation history (POST variant).

    This endpoint allows the frontend to pass conversation history
    directly instead of loading from database.
    """
    # Verify user can access chat (returns True if draft, False if persisted)
    is_draft = await _verify_chat_access(chat_id, user.user_id, db)

    # Get services from app state
    llm: LLMClient = request.app.state.llm_client
    brave_client: BraveSearchClient = request.app.state.brave_client
    crawler: WebCrawler = request.app.state.web_crawler

    # Convert history to dict format
    conversation_history: list[dict[str, str]] = []
    if history:
        conversation_history = [
            {"role": msg.role, "content": msg.content} for msg in history
        ]

    # Get user's system instructions from preferences
    system_instructions: str | None = None
    try:
        preferences_service = PreferencesService(db)
        system_instructions = await preferences_service.get_system_instructions(
            user.user_id
        )
    except Exception as e:
        logger.warning(f"Could not load user preferences: {e}")

    # DEFERRED PERSISTENCE: Generate UUIDs in memory but DON'T persist yet
    # Database writes happen in orchestrator only AFTER synthesis succeeds
    agent_message_id = uuid4()
    research_session_id = uuid4()

    logger.info(
        f"Starting research (with history) with pre-generated IDs: "
        f"message={agent_message_id} session={research_session_id}"
    )

    async def generate_sse_events() -> AsyncGenerator[str, None]:
        """Generate SSE events from the research orchestrator.

        Persistence is handled by the orchestrator after synthesis completes.
        """

        def format_event(event: StreamEvent | str) -> str:
            if isinstance(event, str):
                return f"data: {json.dumps({'event_type': 'content', 'content': event})}\n\n"
            else:
                event_dict = _event_to_dict(event)
                return f"data: {json.dumps(event_dict)}\n\n"

        try:
            # Emit research_started with pre-generated IDs for frontend
            research_started_event = {
                "event_type": "research_started",
                "message_id": str(agent_message_id),
                "research_session_id": str(research_session_id),
            }
            yield f"data: {json.dumps(research_started_event)}\n\n"

            # Create orchestration config with pre-generated UUIDs
            config = OrchestrationConfig(
                research_depth=research_depth,
                system_instructions=system_instructions,
                message_id=agent_message_id,
                research_session_id=research_session_id,
                is_draft=is_draft,
            )

            # Stream events - orchestrator handles persistence on success
            async for event in stream_research(
                query=query,
                llm=llm,
                brave_client=brave_client,
                crawler=crawler,
                conversation_history=conversation_history,
                user_id=user.user_id,
                chat_id=str(chat_id),
                config=config,
                db=db,
            ):
                yield format_event(event)

            # Orchestrator persists data on success - nothing to do here

        except Exception as e:
            logger.exception(f"Error during research stream: {e}")
            # No cleanup needed - nothing was written to DB
            error_event = {
                "event_type": "error",
                "error_code": "STREAM_ERROR",
                "error_message": str(e),
                "recoverable": False,
            }
            yield f"data: {json.dumps(error_event)}\n\n"
        finally:
            # Flush async traces before response ends
            with contextlib.suppress(Exception):
                mlflow.flush_trace_async_logging()

    return StreamingResponse(
        generate_sse_events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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
