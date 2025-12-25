"""Research streaming endpoints."""

import contextlib
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any
from uuid import UUID

import mlflow
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.agent.orchestrator import stream_research
from src.agent.tools.web_crawler import WebCrawler
from src.core.exceptions import NotFoundError
from src.db.session import get_db
from src.middleware.auth import CurrentUser
from src.schemas.research import CancelResearchResponse
from src.schemas.streaming import StreamEvent
from src.services.chat_service import ChatService
from src.services.llm.client import LLMClient
from src.services.search.brave import BraveSearchClient

router = APIRouter()
logger = logging.getLogger(__name__)


async def _verify_chat_ownership(
    chat_id: UUID, user_id: str, db: AsyncSession
) -> None:
    """Verify user owns the chat. Raises NotFoundError if not."""
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
    """
    # Verify user owns the chat
    await _verify_chat_ownership(chat_id, user.user_id, db)

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

    async def generate_sse_events() -> AsyncGenerator[str, None]:
        """Generate SSE events from the research orchestrator."""

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
            async for event in stream_research(
                query=query,
                llm=llm,
                brave_client=brave_client,
                crawler=crawler,
                conversation_history=conversation_history,
                user_id=user.user_id,
                chat_id=str(chat_id),
            ):
                yield format_event(event)

        except Exception as e:
            logger.exception(f"Error during research stream: {e}")
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
    event_dict = event.model_dump()

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
    history: list[ConversationMessage] | None = None,
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Stream research with explicit conversation history (POST variant).

    This endpoint allows the frontend to pass conversation history
    directly instead of loading from database.
    """
    # Verify user owns the chat
    await _verify_chat_ownership(chat_id, user.user_id, db)

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

    async def generate_sse_events() -> AsyncGenerator[str, None]:
        """Generate SSE events from the research orchestrator."""

        def format_event(event: StreamEvent | str) -> str:
            if isinstance(event, str):
                return f"data: {json.dumps({'event_type': 'content', 'content': event})}\n\n"
            else:
                event_dict = _event_to_dict(event)
                return f"data: {json.dumps(event_dict)}\n\n"

        try:
            async for event in stream_research(
                query=query,
                llm=llm,
                brave_client=brave_client,
                crawler=crawler,
                conversation_history=conversation_history,
                user_id=user.user_id,
                chat_id=str(chat_id),
            ):
                yield format_event(event)

        except Exception as e:
            logger.exception(f"Error during research stream: {e}")
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
    # TODO: Implement with ResearchSessionService
    raise NotFoundError("ResearchSession", str(session_id))
