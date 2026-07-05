"""Chat endpoints."""

import json as json_module
import logging
from datetime import datetime
from typing import Any, Protocol
from uuid import UUID

from fastapi import APIRouter, Cookie, Depends, Query, Request, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, field_validator
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from deep_research.api.v1.utils.transformers import (
    jsonb_claim_to_response,
    jsonb_summary_to_response,
)
from deep_research.core.config import get_settings
from deep_research.core.deps import (
    get_chat_service,
    get_export_service,
    get_session_service,
    get_storage_optional,
)
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
from deep_research.services._protocols import IChatService, IExportService, ISessionService

router = APIRouter()
logger = logging.getLogger(__name__)

# Cookie name for incognito session
INCOGNITO_SESSION_COOKIE = "incognito_session"

# Maximum serialized size (bytes) for a surface_state_patch body.
_SURFACE_STATE_MAX_BYTES = 128 * 1024  # 128 KB


class SurfaceStatePatchRequest(BaseModel):
    """Request body for PUT /chats/{chat_id}/surface-state."""

    model_config = {"extra": "forbid"}

    surface_state: dict[str, Any] = Field(
        ...,
        description="Per-agent surface state patch. Keys are agent-id strings; "
        "values are dicts with optional data_model, action_runs, surface_etag fields.",
    )

    @field_validator("surface_state")
    @classmethod
    def validate_surface_state(cls, v: dict[str, Any]) -> dict[str, Any]:
        for key, val in v.items():
            if not isinstance(key, str):
                raise ValueError("surface_state keys must be strings (agent IDs)")
            if not isinstance(val, dict):
                raise ValueError(f"surface_state[{key!r}] must be a dict")
        return v


class _ChatLike(Protocol):
    id: UUID
    user_id: str
    title: str | None
    status: Any
    chat_type: Any
    incognito_session_id: UUID | None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None


def _research_session_inline(
    rs: Any, sources: list[SourceResponse]
) -> ResearchSessionInline | None:
    """Build a ``ResearchSessionInline`` from an ORM or cached-view session.

    The cached session view (``services/cached/chat._state_session_to_view``)
    emits empty dicts / ``None`` for fields that ``ResearchSessionInline`` types
    as sub-models (``query_classification``, ``plan``) or a required int
    (``plan_iterations``). Coerce empty → ``None`` and non-int → ``0``; if a
    *non-empty but malformed* plan/classification still fails validation, retry
    without them so the session (and its sources + the frontend citation gate
    that keys on ``!!m.researchSession``) survives. Only return ``None`` if even
    the minimal session is invalid — so a malformed session can NEVER 500 the
    whole ``/chats/{id}/full`` render, and inline claims still resolve.
    """
    raw_iters = getattr(rs, "plan_iterations", None)
    common: dict[str, Any] = {
        "id": rs.id,
        "research_depth": rs.research_depth,
        "reasoning_steps": getattr(rs, "reasoning_steps", None) or [],
        "status": rs.status,
        "current_agent": getattr(rs, "current_agent", None),
        "current_step_index": getattr(rs, "current_step_index", None),
        "plan_iterations": raw_iters if isinstance(raw_iters, int) else 0,
        "started_at": rs.started_at,
        "completed_at": getattr(rs, "completed_at", None),
        "sources": sources,
    }
    raw_qc = getattr(rs, "query_classification", None) or None
    raw_plan = getattr(rs, "plan", None) or None
    for qc, plan in ((raw_qc, raw_plan), (None, None)):
        try:
            return ResearchSessionInline(query_classification=qc, plan=plan, **common)
        except PydanticValidationError:
            continue
    logger.warning(
        "CHAT_FULL_SESSION_INLINE_SKIPPED session=%s", str(getattr(rs, "id", "?"))[:8]
    )
    return None


def _chat_to_response(chat: _ChatLike) -> ChatResponse:
    """Convert Chat model or ChatView to ChatResponse schema."""
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
    request: Request,
    user: CurrentUser,
    chat_service: IChatService = Depends(get_chat_service),
    status: ChatStatus | None = Query(None, description="Filter by status. If not provided, returns all non-deleted chats."),
    search: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> ChatListResponse:
    """List user's chats.

    Returns paginated list of user's chats, sorted by most recent activity.
    Supports filtering by status and full-text search.
    """
    chats, total = await chat_service.list(
        user_id=user.user_id,
        status=status,
        limit=limit,
        offset=offset,
        search=search,
    )

    # Storage-side prefetch: hydrate the user's top-3 recent chats in the
    # background so the next `GET /chats/{id}` or `POST /research` hits a
    # warm cache. Fire-and-forget; never blocks the response.
    stack = get_storage_optional(request)
    if stack is not None:
        import asyncio

        asyncio.create_task(stack.hydrator.prefetch(user.user_id, top_n=3))

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
    chat_service: IChatService = Depends(get_chat_service),
    request: ChatCreate | None = None,
) -> ChatResponse:
    """Create a new chat."""
    chat = await chat_service.create(
        user_id=user.user_id,
        title=request.title if request else None,
    )
    await chat_service.commit()
    return _chat_to_response(chat)


# =============================================================================
# Static Incognito Routes
# =============================================================================


@router.post("/incognito", response_model=ChatResponse, status_code=201)
async def create_incognito_chat(
    response: Response,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
    chat_service: IChatService = Depends(get_chat_service),
    session_service: ISessionService = Depends(get_session_service),
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
    await chat_service.commit()

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
    chat_service: IChatService = Depends(get_chat_service),
    session_service: ISessionService = Depends(get_session_service),
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
    session_service: ISessionService = Depends(get_session_service),
    incognito_session: str | None = Cookie(None, alias=INCOGNITO_SESSION_COOKIE),
) -> IncognitoSessionStatus:
    """Get incognito session status and quota information.

    Returns whether a session exists, current chat count, and expiry time.
    """
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
    chat_service: IChatService = Depends(get_chat_service),
) -> ChatFullResponse:
    """Load complete chat with all messages, research sessions, sources, and claims.

    Single API call replaces the waterfall of:
    - GET /chats/{id}/messages
    - GET /messages/{id}/claims
    """
    chat = await chat_service.get_full(chat_id, user.user_id)
    if not chat:
        raise NotFoundError("Chat", str(chat_id))

    messages: list[MessageInline] = []
    # Support both legacy ORM Chat (chat.messages) and ChatFullViewCached (chat.messages)
    chat_messages = getattr(chat, "messages", [])
    for msg in sorted(chat_messages, key=lambda m: m.created_at):
        session_schema = None
        claims = []
        verification_summary = None
        structured_output = None

        rs_obj = getattr(msg, "research_session", None)
        if rs_obj:
            rs = rs_obj
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
            session_schema = _research_session_inline(rs, sources)

            # Pre-parse claims using existing JSONB transformers
            verification_data = getattr(rs, "verification_data", None)
            if verification_data:
                claims = [
                    jsonb_claim_to_response(c, msg.id)
                    for c in verification_data.get("claims", [])
                ]
                raw_summary = verification_data.get("summary")
                if raw_summary:
                    verification_summary = jsonb_summary_to_response(raw_summary)
                # Agent-surface structured-output envelope — both storage
                # impls store it inside verification_data (no migration).
                raw_structured = verification_data.get("structured_output")
                if isinstance(raw_structured, dict):
                    structured_output = raw_structured

        # chat_id may be None on cached path — derive from the chat itself
        msg_chat_id = getattr(msg, "chat_id", None) or chat_id

        messages.append(MessageInline(
            id=msg.id,
            chat_id=msg_chat_id,
            role=msg.role,
            content=msg.content or "",
            created_at=msg.created_at,
            is_edited=msg.is_edited,
            research_session=session_schema,
            claims=claims,
            verification_summary=verification_summary,
            structured_output=structured_output,
        ))

    # Extract surface_state from whichever storage impl returned the chat.
    # Cached path: ChatFullViewCached.surface_state (populated from ChatState.chat.metadata).
    # Legacy path: Chat ORM object has metadata_ JSONB; surface_state lives under that key.
    raw_surface_state = getattr(chat, "surface_state", None)
    if raw_surface_state is None:
        raw_meta = getattr(chat, "metadata_", None) or {}
        candidate = raw_meta.get("surface_state") if isinstance(raw_meta, dict) else None
        if isinstance(candidate, dict):
            raw_surface_state = candidate

    # Support both legacy Chat.id and ChatFullViewCached.id
    return ChatFullResponse(
        id=chat.id,
        title=chat.title,
        status=chat.status,
        chat_type=chat.chat_type or ChatType.REGULAR,
        created_at=chat.created_at,
        updated_at=chat.updated_at,
        messages=messages,
        message_count=len(messages),
        surface_state=raw_surface_state,
    )


@router.get("/{chat_id}", response_model=ChatResponse)
async def get_chat(
    chat_id: UUID,
    request: Request,
    user: CurrentUser,
    chat_service: IChatService = Depends(get_chat_service),
    include_messages: bool = Query(True),
) -> ChatResponse:
    """Get chat details with optional messages."""
    chat = await chat_service.get_for_user(chat_id, user.user_id)
    if not chat:
        raise NotFoundError("Chat", str(chat_id))

    # Storage-side hydration start: kick off the background hydration so
    # the next research turn on this chat reads from a warm cache.
    stack = get_storage_optional(request)
    if stack is not None:
        stack.hydrator.start(chat_id, user_id=user.user_id)

    return _chat_to_response(chat)


@router.patch("/{chat_id}", response_model=ChatResponse)
async def update_chat(
    chat_id: UUID,
    request: ChatUpdate,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
    chat_service: IChatService = Depends(get_chat_service),
) -> ChatResponse:
    """Update chat (rename, archive)."""
    chat = await chat_service.update_chat(
        chat_id=chat_id,
        user_id=user.user_id,
        title=request.title,
        status=request.status,
    )
    if not chat:
        raise NotFoundError("Chat", str(chat_id))
    await chat_service.commit()
    return _chat_to_response(chat)


@router.delete("/{chat_id}", status_code=204)
async def delete_chat(
    chat_id: UUID,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
    chat_service: IChatService = Depends(get_chat_service),
) -> None:
    """Delete chat (soft delete).

    Soft deletes the chat. Recoverable for 30 days.
    """
    deleted = await chat_service.soft_delete(chat_id, user.user_id)
    if not deleted:
        raise NotFoundError("Chat", str(chat_id))
    await chat_service.commit()


@router.post("/{chat_id}/restore", response_model=ChatResponse)
async def restore_chat(
    chat_id: UUID,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
    chat_service: IChatService = Depends(get_chat_service),
) -> ChatResponse:
    """Restore deleted chat.

    Restores a soft-deleted chat within the 30-day recovery window.
    """
    chat = await chat_service.restore(chat_id, user.user_id)
    if not chat:
        raise NotFoundError("Chat", str(chat_id))
    await chat_service.commit()
    return _chat_to_response(chat)


@router.put("/{chat_id}/surface-state")
async def update_surface_state(
    chat_id: UUID,
    body: SurfaceStatePatchRequest,
    request: Request,
    user: CurrentUser,
    chat_service: IChatService = Depends(get_chat_service),
) -> dict[str, Any]:
    """Persist per-agent UI surface state for a chat.

    The ``surface_state`` dict is shallow-merged per-agent into the chat's
    persisted metadata. ``action_runs`` within each agent entry use
    newest-updated_at-wins semantics with idempotent-replay (same session_id
    + same status → no-op). Body is capped at 128 KB serialized.

    Incognito chats are regular chats server-side; surface state is persisted
    for them as well.

    Returns ``{"ok": true}`` on success.
    """
    # Size guard: reject oversized payloads before touching storage.
    raw_body = await request.body()
    if len(raw_body) > _SURFACE_STATE_MAX_BYTES:
        from deep_research.core.exceptions import ValidationError as AppValidationError
        raise AppValidationError(
            f"surface_state payload too large ({len(raw_body)} bytes); "
            f"limit is {_SURFACE_STATE_MAX_BYTES} bytes"
        )

    result = await chat_service.update_chat(
        chat_id=chat_id,
        user_id=user.user_id,
        surface_state_patch=body.surface_state,
    )
    if result is None:
        raise NotFoundError("Chat", str(chat_id))
    await chat_service.commit()
    return {"ok": True}


@router.get("/{chat_id}/export")
async def export_chat(
    chat_id: UUID,
    user: CurrentUser,
    format: str = Query(..., pattern="^(markdown|json)$"),
    export_service: IExportService = Depends(get_export_service),
) -> PlainTextResponse:
    """Export chat as Markdown or JSON.

    For PDF export, use client-side rendering with the JSON or Markdown output.

    Args:
        chat_id: Chat ID to export.
        format: Export format (markdown or json).

    Returns:
        PlainTextResponse with exported content.
    """
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
    chat_service: IChatService = Depends(get_chat_service),
    session_service: ISessionService = Depends(get_session_service),
    incognito_session: str | None = Cookie(None, alias=INCOGNITO_SESSION_COOKIE),
) -> ChatResponse:
    """Convert an incognito chat to a regular (permanent) chat.

    Preserves the chat ID and all content. The chat will no longer
    be deleted when the session expires.
    """
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

    # Convert to regular — update via service
    # Build a mutable proxy for convert_to_regular
    class _ChatProxy:
        def __init__(self, c: _ChatLike) -> None:
            self._c = c
            # Copy mutable attrs
            self.id = c.id
            self.user_id = c.user_id
            self.title = c.title
            self.status = c.status
            self.chat_type = ChatType.REGULAR  # converted
            self.incognito_session_id = None   # cleared
            self.created_at = c.created_at
            self.updated_at = c.updated_at
            self.deleted_at = c.deleted_at

    proxy = _ChatProxy(chat)
    result = await chat_service.update(proxy)
    await chat_service.commit()

    return _chat_to_response(result)
