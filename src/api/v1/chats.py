"""Chat endpoints."""

from uuid import UUID

from fastapi import APIRouter, Depends, Query
from fastapi.responses import PlainTextResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.exceptions import NotFoundError
from src.db.session import get_db
from src.middleware.auth import CurrentUser
from src.models.chat import Chat, ChatStatus
from src.schemas.chat import (
    ChatCreate,
    ChatListResponse,
    ChatResponse,
    ChatUpdate,
)
from src.services.chat_service import ChatService
from src.services.export_service import ExportService

router = APIRouter()


@router.get("", response_model=ChatListResponse)
async def list_chats(
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
    status: ChatStatus = Query(ChatStatus.ACTIVE),
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


@router.get("/{chat_id}", response_model=ChatResponse)
async def get_chat(
    chat_id: UUID,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
    include_messages: bool = Query(True),
) -> ChatResponse:
    """Get chat details with optional messages."""
    service = ChatService(db)
    chat = await service.get(chat_id, user.user_id)
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
    chat = await service.update(
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
            import json as json_module

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


def _chat_to_response(chat: "Chat") -> ChatResponse:
    """Convert Chat model to ChatResponse schema."""
    return ChatResponse(
        id=chat.id,
        title=chat.title,
        status=chat.status,
        message_count=0,  # TODO: Add message count relationship
        created_at=chat.created_at,
        updated_at=chat.updated_at,
    )
