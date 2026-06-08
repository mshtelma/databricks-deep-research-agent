"""Message endpoints."""

from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Query

from deep_research.core.deps import (
    get_chat_service,
    get_feedback_service,
    get_message_service,
)
from deep_research.core.exceptions import NotFoundError
from deep_research.middleware.auth import CurrentUser
from deep_research.models.message import MessageRole
from deep_research.schemas.feedback import FeedbackRequest, FeedbackResponse
from deep_research.schemas.message import (
    EditMessageRequest,
    EditMessageResponse,
    MessageListResponse,
    MessageResponse,
    RegenerateResponse,
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
