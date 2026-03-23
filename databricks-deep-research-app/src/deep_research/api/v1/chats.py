"""Chat endpoints."""

import json as json_module
from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Cookie, Depends, Query, Response
from fastapi.responses import PlainTextResponse
from sqlalchemy.ext.asyncio import AsyncSession

from deep_research.api.v1.utils.transformers import (
    jsonb_claim_to_response,
    jsonb_summary_to_response,
)
from deep_research.core.config import get_settings
from deep_research.core.exceptions import NotFoundError, ValidationError
from deep_research.db.session import get_db
from deep_research.middleware.auth import CurrentUser
from deep_research.models.chat import Chat, ChatStatus, ChatType
from deep_research.models.incognito_session import MAX_INCOGNITO_CHATS, SESSION_TTL_HOURS
from deep_research.schemas.agent import SourceResponse
from deep_research.schemas.chat import (
    ChatCreate,
    ChatFullResponse,
    ChatListResponse,
    ChatResponse,
    ChatUpdate,
    MessageInline,
    ResearchSessionInline,
)
from deep_research.schemas.session import IncognitoChatListResponse, IncognitoSessionStatus
from deep_research.services.chat_service import ChatService
from deep_research.services.export_service import ExportService
from deep_research.services.session_service import SessionService

router = APIRouter()

# Cookie name for incognito session
INCOGNITO_SESSION_COOKIE = "incognito_session"


def _chat_to_response(chat: "Chat") -> ChatResponse:
    """Convert Chat model to ChatResponse schema."""
    return ChatResponse(
        id=chat.id,
        title=chat.title,
        status=chat.status,
        chat_type=chat.chat_type or ChatType.REGULAR,
        message_count=0,  # TODO: Add message count relationship
        created_at=chat.created_at,
        updated_at=chat.updated_at,
    )


# =============================================================================
# Static Routes (must come before dynamic /{chat_id} routes)
# =============================================================================


@router.get("", response_model=ChatListResponse)
async def list_chats(
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
    status: ChatStatus | None = Query(None, description="Filter by status. If not provided, returns all non-deleted chats."),
    search: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> ChatListResponse:
    """List user's chats.

    Returns paginated list of user's chats, sorted by most recent activity.
    Supports filtering by status and full-text search.
    """
    service = ChatService(db)
    chats, total = await service.list(
        user_id=user.user_id,
        status=status,
        limit=limit,
        offset=offset,
        search=search,
    )

    return ChatListResponse(
        items=[_chat_to_response(chat) for chat in chats],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post("", response_model=ChatResponse, status_code=201)
async def create_chat(
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
    request: ChatCreate | None = None,
) -> ChatResponse:
    """Create a new chat."""
    service = ChatService(db)
    chat = await service.create(
        user_id=user.user_id,
        title=request.title if request else None,
    )
    await db.commit()
    return _chat_to_response(chat)


# =============================================================================
# Static Incognito Routes
# =============================================================================


@router.post("/incognito", response_model=ChatResponse, status_code=201)
async def create_incognito_chat(
    response: Response,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
    request: ChatCreate | None = None,
    incognito_session: str | None = Cookie(None, alias=INCOGNITO_SESSION_COOKIE),
) -> ChatResponse:
    """Create a new incognito (ephemeral) chat.

    Incognito chats are:
    - Stored server-side (survives page refresh)
    - Associated with a browser session via httpOnly cookie
    - Automatically deleted when session expires (1-hour idle timeout)
    - Limited to 5 concurrent chats per session

    Sets a session cookie if one doesn't exist.
    """
    session_service = SessionService(db)
    chat_service = ChatService(db)

    # Get or create session
    session, token, is_new = await session_service.get_or_create_session(
        user_id=user.user_id,
        session_token=incognito_session,
    )

    # Check quota
    can_create = await session_service.can_create_chat(session.id)
    if not can_create:
        raise ValidationError(
            f"Maximum {MAX_INCOGNITO_CHATS} incognito chats reached. "
            "Please close some incognito chats or convert them to regular chats."
        )

    # Create incognito chat
    chat = Chat(
        user_id=user.user_id,
        title=request.title if request else None,
        chat_type=ChatType.INCOGNITO,
        incognito_session_id=session.id,
    )
    chat = await chat_service.add(chat)
    await db.commit()

    # Set session cookie if new
    if is_new:
        settings = get_settings()
        response.set_cookie(
            key=INCOGNITO_SESSION_COOKIE,
            value=token,
            httponly=True,
            # Use 'lax' in dev (allows cross-port requests), 'strict' in production
            samesite="strict" if settings.is_production else "lax",
            max_age=SESSION_TTL_HOURS * 3600,  # 1 hour
            # secure=True requires HTTPS; browsers silently reject secure cookies over HTTP
            secure=settings.is_production,
            # Restrict cookie to API paths for additional security
            path="/api/v1",
        )

    return _chat_to_response(chat)


@router.get("/incognito", response_model=IncognitoChatListResponse)
async def list_incognito_chats(
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
    incognito_session: str | None = Cookie(None, alias=INCOGNITO_SESSION_COOKIE),
) -> IncognitoChatListResponse:
    """List incognito chats for the current browser session.

    Returns only chats associated with the session token in the cookie.
    """
    if not incognito_session:
        return IncognitoChatListResponse(
            items=[],
            total=0,
            session_expires_at=None,
        )

    session_service = SessionService(db)
    # Use secure method that verifies user ownership
    session = await session_service.get_by_token_for_user(incognito_session, user.user_id)

    if not session:
        return IncognitoChatListResponse(
            items=[],
            total=0,
            session_expires_at=None,
        )

    # Touch session to extend TTL
    session.touch()
    await session_service.update(session)

    # Get incognito chats for this session
    chat_service = ChatService(db)
    chats = await chat_service.list_incognito_for_session(session.id)

    return IncognitoChatListResponse(
        items=[_chat_to_response(chat) for chat in chats],
        total=len(chats),
        session_expires_at=session.expires_at,
    )


@router.get("/session/incognito", response_model=IncognitoSessionStatus)
async def get_incognito_session_status(
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
    incognito_session: str | None = Cookie(None, alias=INCOGNITO_SESSION_COOKIE),
) -> IncognitoSessionStatus:
    """Get incognito session status and quota information.

    Returns whether a session exists, current chat count, and expiry time.
    """
    session_service = SessionService(db)
    # Pass user_id for ownership verification
    status = await session_service.get_session_status(incognito_session, user.user_id)

    return IncognitoSessionStatus(
        has_session=status["has_session"],
        chat_count=status["chat_count"],
        max_chats=status["max_chats"],
        expires_at=datetime.fromisoformat(status["expires_at"]) if status["expires_at"] else None,
    )


# =============================================================================
# Dynamic Routes (/{chat_id} patterns - must come after static routes)
# =============================================================================


@router.get("/{chat_id}/full", response_model=ChatFullResponse)
async def get_chat_full(
    chat_id: UUID,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> ChatFullResponse:
    """Load complete chat with all messages, research sessions, sources, and claims.

    Single API call replaces the waterfall of:
    - GET /chats/{id}/messages
    - GET /messages/{id}/claims
    """
    service = ChatService(db)
    chat = await service.get_full(chat_id, user.user_id)
    if not chat:
        raise NotFoundError("Chat", str(chat_id))

    messages: list[MessageInline] = []
    for msg in sorted(chat.messages, key=lambda m: m.created_at):
        session_schema = None
        claims = []
        verification_summary = None

        if msg.research_session:
            rs = msg.research_session
            sources = [
                SourceResponse(
                    id=s.id,
                    url=s.url,
                    title=s.title,
                    snippet=s.snippet,
                    relevance_score=s.relevance_score,
                    source_type=s.source_type or "web",
                    source_metadata=s.source_metadata,
                    is_cited=s.is_cited,
                )
                for s in (rs.sources or [])
            ]
            session_schema = ResearchSessionInline(
                id=rs.id,
                query_classification=rs.query_classification,
                research_depth=rs.research_depth,
                reasoning_steps=rs.reasoning_steps or [],
                status=rs.status,
                current_agent=rs.current_agent,
                plan=rs.plan,
                current_step_index=rs.current_step_index,
                plan_iterations=rs.plan_iterations,
                started_at=rs.started_at,
                completed_at=rs.completed_at,
                sources=sources,
            )

            # Pre-parse claims using existing JSONB transformers
            if rs.verification_data:
                claims = [
                    jsonb_claim_to_response(c, msg.id)
                    for c in rs.verification_data.get("claims", [])
                ]
                raw_summary = rs.verification_data.get("summary")
                if raw_summary:
                    verification_summary = jsonb_summary_to_response(raw_summary)

        messages.append(MessageInline(
            id=msg.id,
            chat_id=msg.chat_id,
            role=msg.role,
            content=msg.content or "",
            created_at=msg.created_at,
            is_edited=msg.is_edited,
            research_session=session_schema,
            claims=claims,
            verification_summary=verification_summary,
        ))

    return ChatFullResponse(
        id=chat.id,
        title=chat.title,
        status=chat.status,
        chat_type=chat.chat_type or ChatType.REGULAR,
        created_at=chat.created_at,
        updated_at=chat.updated_at,
        messages=messages,
        message_count=len(messages),
    )


@router.get("/{chat_id}", response_model=ChatResponse)
async def get_chat(
    chat_id: UUID,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
    include_messages: bool = Query(True),
) -> ChatResponse:
    """Get chat details with optional messages."""
    service = ChatService(db)
    chat = await service.get_for_user(chat_id, user.user_id)
    if not chat:
        raise NotFoundError("Chat", str(chat_id))
    return _chat_to_response(chat)


@router.patch("/{chat_id}", response_model=ChatResponse)
async def update_chat(
    chat_id: UUID,
    request: ChatUpdate,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    """Update chat (rename, archive)."""
    service = ChatService(db)
    chat = await service.update_chat(
        chat_id=chat_id,
        user_id=user.user_id,
        title=request.title,
        status=request.status,
    )
    if not chat:
        raise NotFoundError("Chat", str(chat_id))
    await db.commit()
    return _chat_to_response(chat)


@router.delete("/{chat_id}", status_code=204)
async def delete_chat(
    chat_id: UUID,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete chat (soft delete).

    Soft deletes the chat. Recoverable for 30 days.
    """
    service = ChatService(db)
    deleted = await service.soft_delete(chat_id, user.user_id)
    if not deleted:
        raise NotFoundError("Chat", str(chat_id))
    await db.commit()


@router.post("/{chat_id}/restore", response_model=ChatResponse)
async def restore_chat(
    chat_id: UUID,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    """Restore deleted chat.

    Restores a soft-deleted chat within the 30-day recovery window.
    """
    service = ChatService(db)
    chat = await service.restore(chat_id, user.user_id)
    if not chat:
        raise NotFoundError("Chat", str(chat_id))
    await db.commit()
    return _chat_to_response(chat)


@router.get("/{chat_id}/export")
async def export_chat(
    chat_id: UUID,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
    format: str = Query(..., pattern="^(markdown|json)$"),
) -> PlainTextResponse:
    """Export chat as Markdown or JSON.

    For PDF export, use client-side rendering with the JSON or Markdown output.

    Args:
        chat_id: Chat ID to export.
        format: Export format (markdown or json).

    Returns:
        PlainTextResponse with exported content.
    """
    export_service = ExportService(db)

    try:
        if format == "markdown":
            content = await export_service.export_markdown(
                chat_id=chat_id,
                user_id=user.user_id,
            )
            return PlainTextResponse(
                content=content,
                media_type="text/markdown",
                headers={
                    "Content-Disposition": f'attachment; filename="chat-{chat_id}.md"'
                },
            )
        else:  # json
            data = await export_service.export_json(
                chat_id=chat_id,
                user_id=user.user_id,
            )
            content = json_module.dumps(data, indent=2)
            return PlainTextResponse(
                content=content,
                media_type="application/json",
                headers={
                    "Content-Disposition": f'attachment; filename="chat-{chat_id}.json"'
                },
            )
    except ValueError as e:
        raise NotFoundError("Chat", str(chat_id)) from e


@router.post("/{chat_id}/convert", response_model=ChatResponse)
async def convert_incognito_to_regular(
    chat_id: UUID,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
    incognito_session: str | None = Cookie(None, alias=INCOGNITO_SESSION_COOKIE),
) -> ChatResponse:
    """Convert an incognito chat to a regular (permanent) chat.

    Preserves the chat ID and all content. The chat will no longer
    be deleted when the session expires.
    """
    chat_service = ChatService(db)
    session_service = SessionService(db)

    # Get the chat
    chat = await chat_service.get_for_user(chat_id, user.user_id)
    if not chat:
        raise NotFoundError("Chat", str(chat_id))

    # Verify it's an incognito chat
    if chat.chat_type != ChatType.INCOGNITO:
        raise ValidationError("Chat is not an incognito chat")

    # Verify session ownership - use secure method that validates user_id
    if incognito_session:
        session = await session_service.get_by_token_for_user(incognito_session, user.user_id)
        if session and chat.incognito_session_id != session.id:
            raise NotFoundError("Chat", str(chat_id))

    # Convert to regular
    chat.convert_to_regular()
    await chat_service.update(chat)
    await db.commit()

    return _chat_to_response(chat)
