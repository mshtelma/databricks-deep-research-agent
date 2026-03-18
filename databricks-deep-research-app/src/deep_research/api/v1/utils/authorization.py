"""Centralized authorization utilities for API endpoints.

This module consolidates authorization logic previously duplicated across:
- messages.py (_verify_chat_ownership)
- citations.py (_verify_message_ownership)
- research.py (_verify_chat_access, _verify_chat_ownership)
- export_service.py (_verify_message_ownership)
"""

import logging
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from deep_research.core.config import get_settings
from deep_research.core.exceptions import AuthorizationError, NotFoundError
from deep_research.models.chat import Chat
from deep_research.models.message import Message
from deep_research.services.chat_service import ChatService

logger = logging.getLogger(__name__)


async def verify_chat_ownership(
    chat_id: UUID,
    user_id: str,
    db: AsyncSession,
) -> Chat:
    """Verify user owns the chat.

    Args:
        chat_id: Chat UUID to check.
        user_id: Current user's ID.
        db: Database session.

    Returns:
        The Chat if owned by user.

    Raises:
        NotFoundError: If chat not found or not owned by user.
    """
    chat_service = ChatService(db)
    chat = await chat_service.get_for_user(chat_id, user_id)
    if not chat:
        raise NotFoundError("Chat", str(chat_id))
    return chat


async def verify_chat_access(
    chat_id: UUID,
    user_id: str,
    db: AsyncSession,
) -> tuple[bool, Chat | None]:
    """Verify user can access chat with draft support.

    Authorization logic for draft chat flow:
    - If chat doesn't exist: allow (draft) -> returns (True, None)
    - If chat exists and owned by user: allow -> returns (False, chat)
    - If chat exists but owned by another: reject with 403

    Args:
        chat_id: Chat UUID to check.
        user_id: Current user's ID.
        db: Database session.

    Returns:
        Tuple of (is_draft, chat).
        - is_draft=True means chat doesn't exist yet (draft flow).
        - is_draft=False means chat exists and is owned by user.

    Raises:
        AuthorizationError: If chat exists but belongs to another user.
    """
    chat_service = ChatService(db)
    chat = await chat_service.get_by_id(chat_id)

    if chat is None:
        logger.info(f"Chat {chat_id} is a draft (not in DB), allowing access")
        return True, None

    if chat.user_id != user_id:
        logger.warning(
            f"User {user_id} attempted to access chat {chat_id} owned by {chat.user_id}"
        )
        raise AuthorizationError(f"Access denied to chat {chat_id}")

    return False, chat


async def verify_message_ownership(
    message_id: UUID,
    user_id: str,
    db: AsyncSession,
    *,
    allow_dev_anonymous: bool = True,
) -> Message:
    """Verify user owns the message's chat.

    Args:
        message_id: Message UUID to check.
        user_id: Current user's ID.
        db: Database session.
        allow_dev_anonymous: If True, allows anonymous access in dev mode.
            Default is True for backward compatibility.

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
