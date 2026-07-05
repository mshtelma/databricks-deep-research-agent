"""Message endpoints."""

from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, Query, Request

from deep_research.agent.structured_evidence import load_run_artifacts
from deep_research.core.deps import (
    get_chat_service,
    get_feedback_service,
    get_message_service,
    get_storage_optional,
)
from deep_research.core.exceptions import (
    ConflictError,
    NotFoundError,
    ValidationError,
)
from deep_research.middleware.auth import CurrentUser
from deep_research.models.message import MessageRole
from deep_research.schemas.feedback import FeedbackRequest, FeedbackResponse
from deep_research.schemas.message import (
    EditMessageRequest,
    EditMessageResponse,
    MessageListResponse,
    MessageResponse,
    RegenerateResponse,
    RestructureRequest,
    RestructureResponse,
    SendMessageRequest,
    SendMessageResponse,
)
from deep_research.schemas.research import ResearchSession as ResearchSessionSchema
from deep_research.services._protocols import (
    IChatService,
    IFeedbackService,
    IMessageService,
)

router = APIRouter()


def _research_session_to_schema(
    session: Any | None,
) -> ResearchSessionSchema | None:
    """Convert ResearchSession (ORM or cached namespace) to schema."""
    if session is None:
        return None

    return ResearchSessionSchema(
        id=session.id,
        query_classification=session.query_classification,
        research_depth=session.research_depth,
        reasoning_steps=session.reasoning_steps or [],
        status=session.status,
        current_agent=getattr(session, "current_agent", None),
        plan=session.plan,
        current_step_index=getattr(session, "current_step_index", None)
        or getattr(session, "current_step", None),
        plan_iterations=getattr(session, "plan_iterations", None),
        started_at=session.started_at,
        completed_at=session.completed_at,
        sources=[],  # Don't load sources to avoid N+1 queries
    )


def _message_to_response(msg: Any) -> MessageResponse:
    """Convert Message model to MessageResponse schema."""
    return MessageResponse(
        id=msg.id,
        chat_id=msg.chat_id,
        role=msg.role,
        content=msg.content or "",  # Content can be None for in-progress agent messages
        created_at=msg.created_at,
        is_edited=msg.is_edited,
        research_session=_research_session_to_schema(
            getattr(msg, "research_session", None)
        ),
    )


@router.get("/chats/{chat_id}/messages", response_model=MessageListResponse)
async def list_messages(
    chat_id: UUID,
    user: CurrentUser,
    message_service: IMessageService = Depends(get_message_service),
    chat_service: IChatService = Depends(get_chat_service),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> MessageListResponse:
    """List messages in a chat."""
    # Verify user owns the chat
    chat = await chat_service.get_for_user(chat_id, user.user_id)
    if not chat:
        raise NotFoundError("Chat", str(chat_id))

    messages, total = await message_service.list_messages(
        chat_id=chat_id,
        limit=limit,
        offset=offset,
    )

    return MessageListResponse(
        items=[_message_to_response(msg) for msg in messages],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post("/chats/{chat_id}/messages", response_model=SendMessageResponse, status_code=201)
async def send_message(
    chat_id: UUID,
    request: SendMessageRequest,
    user: CurrentUser,
    message_service: IMessageService = Depends(get_message_service),
    chat_service: IChatService = Depends(get_chat_service),
) -> SendMessageResponse:
    """Send a message and get agent response.

    Sends a user message and triggers agent research. Returns immediately
    with message IDs. Use SSE endpoint to stream the agent response.
    """
    # Verify user owns the chat
    chat = await chat_service.get_for_user(chat_id, user.user_id)
    if not chat:
        raise NotFoundError("Chat", str(chat_id))

    # Create user message
    user_message = await message_service.create(
        chat_id=chat_id,
        role=MessageRole.USER,
        content=request.content,
    )

    # Auto-title chat from first message (only updates if title is not set)
    await chat_service.update_title_from_message(chat_id, request.content)

    # Create placeholder agent message (will be filled by streaming)
    session_id = uuid4()

    return SendMessageResponse(
        user_message=_message_to_response(user_message),
        agent_message_id=uuid4(),  # Placeholder, actual message created by stream
        research_session_id=session_id,
    )


@router.get("/chats/{chat_id}/messages/{message_id}")
async def get_message(
    chat_id: UUID,
    message_id: UUID,
    user: CurrentUser,
    message_service: IMessageService = Depends(get_message_service),
    chat_service: IChatService = Depends(get_chat_service),
    include_research_session: bool = Query(False),
) -> MessageResponse:
    """Get message details."""
    # Verify user owns the chat
    chat = await chat_service.get_for_user(chat_id, user.user_id)
    if not chat:
        raise NotFoundError("Chat", str(chat_id))

    message = await message_service.get_with_chat(message_id, chat_id)
    if not message:
        raise NotFoundError("Message", str(message_id))
    return _message_to_response(message)


@router.patch("/chats/{chat_id}/messages/{message_id}", response_model=EditMessageResponse)
async def edit_message(
    chat_id: UUID,
    message_id: UUID,
    request: EditMessageRequest,
    user: CurrentUser,
    message_service: IMessageService = Depends(get_message_service),
    chat_service: IChatService = Depends(get_chat_service),
) -> EditMessageResponse:
    """Edit a user message.

    Edits a user message content. Invalidates (removes) all subsequent
    messages in the conversation thread.
    """
    # Verify user owns the chat
    chat = await chat_service.get_for_user(chat_id, user.user_id)
    if not chat:
        raise NotFoundError("Chat", str(chat_id))

    # Get the original message
    original = await message_service.get_with_chat(message_id, chat_id)
    if not original:
        raise NotFoundError("Message", str(message_id))

    # Only user messages can be edited
    if original.role != MessageRole.USER:
        raise NotFoundError("Message", str(message_id))

    # Delete subsequent messages
    deleted_count = await message_service.delete_subsequent(chat_id, original.created_at)

    # Update the message content
    updated = await message_service.update_content(message_id, request.content, chat_id=chat_id)
    if not updated:
        raise NotFoundError("Message", str(message_id))

    return EditMessageResponse(
        message=_message_to_response(updated),
        removed_message_count=deleted_count,
    )


@router.post(
    "/chats/{chat_id}/messages/{message_id}/regenerate",
    response_model=RegenerateResponse,
    status_code=201,
)
async def regenerate_message(
    chat_id: UUID,
    message_id: UUID,
    user: CurrentUser,
    message_service: IMessageService = Depends(get_message_service),
    chat_service: IChatService = Depends(get_chat_service),
) -> RegenerateResponse:
    """Regenerate agent response.

    Regenerates the agent response for the preceding user message.
    Creates a new agent message with fresh research results.
    """
    # Verify user owns the chat
    chat = await chat_service.get_for_user(chat_id, user.user_id)
    if not chat:
        raise NotFoundError("Chat", str(chat_id))

    # Get the message (should be an agent message)
    original = await message_service.get_with_chat(message_id, chat_id)
    if not original:
        raise NotFoundError("Message", str(message_id))

    # Delete the old agent message and any after it
    await message_service.delete_subsequent(chat_id, original.created_at)

    # Create a new session for regeneration
    new_session_id = uuid4()
    new_message_id = uuid4()

    return RegenerateResponse(
        new_message_id=new_message_id,
        research_session_id=new_session_id,
    )


@router.post(
    "/chats/{chat_id}/messages/{message_id}/feedback",
    response_model=FeedbackResponse,
    status_code=201,
)
async def submit_feedback(
    chat_id: UUID,
    message_id: UUID,
    request: FeedbackRequest,
    user: CurrentUser,
    message_service: IMessageService = Depends(get_message_service),
    feedback_service: IFeedbackService = Depends(get_feedback_service),
    chat_service: IChatService = Depends(get_chat_service),
) -> FeedbackResponse:
    """Submit feedback on agent message."""
    # Verify user owns the chat
    chat = await chat_service.get_for_user(chat_id, user.user_id)
    if not chat:
        raise NotFoundError("Chat", str(chat_id))

    # Verify message exists
    message = await message_service.get_with_chat(message_id, chat_id)
    if not message:
        raise NotFoundError("Message", str(message_id))

    # Create actual feedback record
    try:
        feedback = await feedback_service.create_feedback(
            message_id=message_id,
            user_id=user.user_id,
            rating=request.rating,
            feedback_text=request.feedback_text,
            feedback_category=request.feedback_category,
        )
    except ValueError as e:
        raise NotFoundError("Feedback", str(e)) from e

    return FeedbackResponse(
        id=feedback.id,
        message_id=feedback.message_id,
        rating=feedback.rating.value,
        feedback_text=feedback.feedback_text,
        feedback_category=feedback.feedback_category,
        created_at=feedback.created_at,
    )


@router.get("/chats/{chat_id}/messages/{message_id}/copy")
async def get_message_content(
    chat_id: UUID,
    message_id: UUID,
    user: CurrentUser,
    message_service: IMessageService = Depends(get_message_service),
    chat_service: IChatService = Depends(get_chat_service),
) -> dict[str, Any]:
    """Get message content for clipboard.

    Returns plain text content suitable for copying to clipboard.
    """
    # Verify user owns the chat
    chat = await chat_service.get_for_user(chat_id, user.user_id)
    if not chat:
        raise NotFoundError("Chat", str(chat_id))

    message = await message_service.get_with_chat(message_id, chat_id)
    if not message:
        raise NotFoundError("Message", str(message_id))

    return {"content": message.content}


# ---------------------------------------------------------------------------
# Structured-output restructure (per-slot retry)
# ---------------------------------------------------------------------------

_RESTRUCTURE_IN_PROGRESS_WINDOW_S = 300  # 409 while a recent run is pending


def _now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def _recent(generated_at: Any, window_s: int) -> bool:
    from datetime import UTC, datetime

    if not isinstance(generated_at, str):
        return False
    try:
        stamp = datetime.fromisoformat(generated_at)
    except ValueError:
        return False
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    return (datetime.now(UTC) - stamp).total_seconds() < window_s


async def _restructure_in_background(
    *,
    chat_id: UUID,
    message_id: UUID,
    user_id: str,
    requested: set[str],
    storage_stack: Any | None,
    llm: Any | None,
    marked_envelope: dict[str, Any],
    artifacts: Any,
) -> None:
    """Re-run the requested slot wires; NEVER leaves slots stuck pending."""
    import copy
    import logging

    from deep_research.agent.orchestration_config import OrchestrationConfig
    from deep_research.agent.persistence import (
        update_structured_output_independent,
    )
    from deep_research.agent.structured_surface import (
        load_agent_surface,
        structure_and_update,
    )
    from deep_research.surface.output_schema import (
        collect_output_slots,
        resolve_binding_for_run,
    )

    logger = logging.getLogger(__name__)
    binding = str(marked_envelope.get("binding") or "")
    agent_id = marked_envelope.get("agent_id")

    try:
        loaded = (
            await load_agent_surface(
                OrchestrationConfig(agent_id=str(agent_id)), user_id, None
            )
            if agent_id
            else None
        )
        resolved = None
        fresh_etag: str | None = None
        if loaded is not None:
            surface, fresh_etag = loaded
            resolved = resolve_binding_for_run(
                collect_output_slots(surface), binding
            )
        if resolved is None:
            raise RuntimeError(
                "agent surface or binding is no longer available"
            )

        runnable = requested & set(resolved.slots)
        prior = marked_envelope
        if requested - runnable:
            prior = copy.deepcopy(marked_envelope)
            for slot in requested - runnable:
                prior["meta"]["slots"][slot] = {
                    "status": "failed",
                    "error": "slot is no longer declared by the surface",
                }
        if not runnable:
            prior["generated_at"] = _now_iso()
            await update_structured_output_independent(
                chat_id=chat_id,
                research_session_id=artifacts.research_session_id,
                envelope=prior,
                storage_stack=storage_stack,
            )
            return

        await structure_and_update(
            binding=binding,
            agent_id=str(agent_id) if agent_id else None,
            surface_etag=fresh_etag,
            slots=resolved.slots,
            report=artifacts.report,
            claims=artifacts.claims,
            sources=artifacts.sources,
            chat_id=chat_id,
            research_session_id=artifacts.research_session_id,
            storage_stack=storage_stack,
            llm=llm,
            only_slots=runnable,
            prior_envelope=prior,
        )
    except Exception as exc:  # noqa: BLE001 — never-stuck-pending guarantee
        logger.exception(
            "RESTRUCTURE_BACKGROUND_FAILED message=%s", str(message_id)[:8]
        )
        failed = copy.deepcopy(marked_envelope)
        slots_meta = failed.get("meta", {}).get("slots") or {}
        for slot in requested:
            if slots_meta.get(slot, {}).get("status") == "pending":
                slots_meta[slot] = {
                    "status": "failed",
                    "error": str(exc)[:200],
                }
        failed["generated_at"] = _now_iso()
        try:
            await update_structured_output_independent(
                chat_id=chat_id,
                research_session_id=artifacts.research_session_id,
                envelope=failed,
                storage_stack=storage_stack,
            )
        except Exception:  # noqa: BLE001 — best effort; FE has stale-pending
            logger.exception(
                "RESTRUCTURE_FAILED_STUB_WRITE_FAILED message=%s",
                str(message_id)[:8],
            )


@router.post(
    "/chats/{chat_id}/messages/{message_id}/restructure",
    response_model=RestructureResponse,
    status_code=202,
)
async def restructure_message(
    chat_id: UUID,
    message_id: UUID,
    request: RestructureRequest,
    user: CurrentUser,
    background_tasks: BackgroundTasks,
    fastapi_request: Request,
    chat_service: IChatService = Depends(get_chat_service),
) -> RestructureResponse:
    """Re-run structured-output slots for a completed agent message.

    Marks the requested slots ``pending`` (persisted immediately so the UI
    shows skeletons), schedules the per-slot wires in the background, and
    returns 202. The frontend polls the chat until no slot is pending.
    """
    import copy

    chat = await chat_service.get_for_user(chat_id, user.user_id)
    if not chat:
        raise NotFoundError("Chat", str(chat_id))

    storage_stack = get_storage_optional(fastapi_request)
    artifacts = await load_run_artifacts(chat_id, message_id, storage_stack)
    if artifacts is None or not isinstance(artifacts.envelope, dict):
        raise NotFoundError("StructuredOutput", str(message_id))

    envelope = artifacts.envelope
    slots_meta = envelope.get("meta", {}).get("slots") or {}
    if not slots_meta:
        raise NotFoundError("StructuredOutput", str(message_id))

    requested = set(request.slots) if request.slots else set(slots_meta)
    unknown = requested - set(slots_meta)
    if unknown:
        raise ValidationError(
            f"unknown slot(s): {', '.join(sorted(unknown)[:5])}"
        )
    if any(
        slots_meta.get(slot, {}).get("status") == "pending"
        for slot in requested
    ) and _recent(
        envelope.get("generated_at"), _RESTRUCTURE_IN_PROGRESS_WINDOW_S
    ):
        raise ConflictError(
            "a restructure for this message is already in progress"
        )

    marked = copy.deepcopy(envelope)
    for slot in requested:
        marked["meta"]["slots"][slot] = {"status": "pending"}
    marked["generated_at"] = _now_iso()
    from deep_research.agent.persistence import (
        update_structured_output_independent,
    )

    await update_structured_output_independent(
        chat_id=chat_id,
        research_session_id=artifacts.research_session_id,
        envelope=marked,
        storage_stack=storage_stack,
    )

    background_tasks.add_task(
        _restructure_in_background,
        chat_id=chat_id,
        message_id=message_id,
        user_id=user.user_id,
        requested=requested,
        storage_stack=storage_stack,
        llm=getattr(fastapi_request.app.state, "llm_client", None),
        marked_envelope=marked,
        artifacts=artifacts,
    )
    return RestructureResponse(status="accepted", slots=sorted(requested))
