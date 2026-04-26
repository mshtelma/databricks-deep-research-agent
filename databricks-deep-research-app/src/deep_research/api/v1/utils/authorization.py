"""Centralized authorization utilities for API endpoints.

This module consolidates authorization logic previously duplicated across:
- messages.py (_verify_chat_ownership)
- citations.py (_verify_message_ownership)
- research.py (_verify_chat_access, _verify_chat_ownership)
- export_service.py (_verify_message_ownership)
"""

import logging
from typing import TYPE_CHECKING, Any
from uuid import UUID

from fastapi import Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from deep_research.core.config import get_settings
from deep_research.core.exceptions import NotFoundError
from deep_research.models.chat import Chat
from deep_research.models.message import Message
from deep_research.services._impl_factory import make_chat_service

if TYPE_CHECKING:
    from deep_research.storage.factory import StorageStack

logger = logging.getLogger(__name__)


def _resolve_stack(
    request: Request | None,
    storage_stack: Any | None,
) -> "StorageStack | None":
    """Return the process-singleton `StorageStack` for a call site.

    Prefers an explicit `storage_stack` kwarg; otherwise reads from the
    `Request.app.state` when available. Returns None for legacy-only call
    sites — the factory will fall back to `session=db`.
    """
    if storage_stack is not None:
        return storage_stack
    if request is not None:
        return getattr(request.app.state, "storage_stack", None)
    return None


async def verify_chat_ownership(
    chat_id: UUID,
    user_id: str,
    db: AsyncSession,
    *,
    request: Request | None = None,
    storage_stack: Any | None = None,
) -> object:
    """Verify user owns the chat.

    Args:
        chat_id: Chat UUID to check.
        user_id: Current user's ID.
        db: Database session.
        request: Optional FastAPI Request, used to resolve the process-level
            ``StorageStack`` when running under ``storage_service_impl=cached``.
        storage_stack: Optional explicit stack (e.g., background task context).

    Returns:
        The Chat (or ChatView) if owned by user.

    Raises:
        NotFoundError: If chat not found or not owned by user.
    """
    settings = get_settings()
    stack = _resolve_stack(request, storage_stack)
    chat_service = make_chat_service(settings, stack=stack, session=db)
    chat = await chat_service.get_for_user(chat_id, user_id)
    if not chat:
        raise NotFoundError("Chat", str(chat_id))
    return chat


async def verify_chat_access(
    chat_id: UUID,
    user_id: str,
    db: AsyncSession,
    *,
    request: Request | None = None,
    storage_stack: Any | None = None,
) -> tuple[bool, object | None]:
    """Verify user can access chat with draft support.

    Authorization logic for draft chat flow:
    - If chat doesn't exist: allow (draft) -> returns (True, None)
    - If chat exists and owned by user: allow -> returns (False, chat)
    - If chat exists but owned by another: reject with 403

    Args:
        chat_id: Chat UUID to check.
        user_id: Current user's ID.
        db: Database session.
        request: Optional FastAPI Request, used to resolve the process-level
            ``StorageStack`` when running under ``storage_service_impl=cached``.
        storage_stack: Optional explicit stack (e.g., background task context).

    Returns:
        Tuple of (is_draft, chat).
        - is_draft=True means chat doesn't exist yet (draft flow).
        - is_draft=False means chat exists and is owned by user.

    Raises:
        AuthorizationError: If chat exists but belongs to another user.
    """
    settings = get_settings()
    stack = _resolve_stack(request, storage_stack)
    chat_service = make_chat_service(settings, stack=stack, session=db)
    chat = await chat_service.get_by_id(chat_id)

    if chat is None:
        logger.info("Chat %s is a draft (not in DB), allowing access", chat_id)
        return True, None

    if chat.user_id != user_id:
        logger.warning(
            "User %s attempted to access chat %s owned by %s",
            user_id, chat_id, chat.user_id,
        )
        raise NotFoundError("Chat", str(chat_id))

    return False, chat


async def verify_message_ownership(
    message_id: UUID,
    user_id: str,
    db: AsyncSession,
    *,
    allow_dev_anonymous: bool = False,
) -> Message:
    """Verify user owns the message's chat.

    Args:
        message_id: Message UUID to check.
        user_id: Current user's ID.
        db: Database session.
        allow_dev_anonymous: If True, allows anonymous access in dev mode.
            Default is False (opt-in) to enforce least privilege.

    Returns:
        The Message if authorized.

    Raises:
        NotFoundError: If message not found or not authorized.
    """
    # Step 1: Get message with chat relationship eagerly loaded
    result = await db.execute(
        select(Message)
        .options(selectinload(Message.chat))
        .where(Message.id == message_id)
    )
    message = result.scalar_one_or_none()

    if not message:
        raise NotFoundError("Message", str(message_id))

    # Step 2: Get chat (via relationship or direct query if not loaded)
    chat: Chat | None = message.chat
    if not chat:
        chat_result = await db.execute(
            select(Chat).where(Chat.id == message.chat_id)
        )
        chat = chat_result.scalar_one_or_none()

    if not chat:
        raise NotFoundError("Message", str(message_id))

    # Step 3: Check ownership with optional dev mode bypass
    settings = get_settings()
    if chat.user_id != user_id:
        if settings.is_production:
            # Production: strict ownership check
            raise NotFoundError("Message", str(message_id))
        if not allow_dev_anonymous or user_id != "anonymous":
            # Development with real user or anonymous bypass disabled
            raise NotFoundError("Message", str(message_id))
        # Development with anonymous: allow access for testing
        logger.debug(
            f"DEV_MODE: Allowing anonymous access to message {message_id} "
            f"owned by {chat.user_id}"
        )

    return message
