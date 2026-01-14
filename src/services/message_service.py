"""Message service - CRUD operations for messages."""

import logging
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import and_, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.models.message import Message, MessageRole

logger = logging.getLogger(__name__)


class MessageService:
    """Service for managing messages."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize message service.

        Args:
            session: Database session.
        """
        self._session = session

    async def create(
        self,
        chat_id: UUID,
        role: MessageRole,
        content: str,
    ) -> Message:
        """Create a new message.

        Args:
            chat_id: Chat ID.
            role: Message role (user or assistant).
            content: Message content.

        Returns:
            Created message.

        Note:
            Research session association is done via set_research_session()
            since the Message model uses a relationship, not a foreign key.
        """
        message = Message(
            chat_id=chat_id,
            role=role,
            content=content,
        )
        self._session.add(message)
        await self._session.flush()
        await self._session.refresh(message)
        logger.info(f"Created {role} message {message.id} in chat {chat_id}")
        return message

    async def get(self, message_id: UUID) -> Message | None:
        """Get a message by ID.

        Args:
            message_id: Message ID.

        Returns:
            Message if found, None otherwise.
        """
        result = await self._session.execute(
            select(Message).where(Message.id == message_id)
        )
        return result.scalar_one_or_none()

    async def get_with_chat(self, message_id: UUID, chat_id: UUID) -> Message | None:
        """Get a message by ID and chat ID.

        Args:
            message_id: Message ID.
            chat_id: Chat ID.

        Returns:
            Message if found, None otherwise.
        """
        result = await self._session.execute(
            select(Message).where(
                and_(
                    Message.id == message_id,
                    Message.chat_id == chat_id,
                )
            )
        )
        return result.scalar_one_or_none()

    async def list_messages(
        self,
        chat_id: UUID,
        limit: int = 100,
        offset: int = 0,
        before: datetime | None = None,
    ) -> tuple[list[Message], int]:
        """List messages in a chat.

        Args:
            chat_id: Chat ID.
            limit: Maximum number of messages.
            offset: Number of messages to skip.
            before: Only return messages before this time.

        Returns:
            Tuple of (messages, total_count).
        """
        conditions = [Message.chat_id == chat_id]

        if before:
            conditions.append(Message.created_at < before)

        # Get total count
        count_query = select(func.count(Message.id)).where(and_(*conditions))
        count_result = await self._session.execute(count_query)
        total = count_result.scalar() or 0

        # Get messages ordered by creation time, with research_session eager loaded
        query = (
            select(Message)
            .options(selectinload(Message.research_session))
            .where(and_(*conditions))
            .order_by(Message.created_at.asc())
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(query)
        messages = list(result.scalars().all())

        return messages, total

    async def update_content(
        self,
        message_id: UUID,
        content: str,
    ) -> Message | None:
        """Update message content (mark as edited).

        Args:
            message_id: Message ID.
            content: New content.

        Returns:
            Updated message or None if not found.
        """
        message = await self.get(message_id)
        if not message:
            return None

        message.content = content
        message.is_edited = True
        message.updated_at = datetime.now(UTC)
        await self._session.flush()
        await self._session.refresh(message)
        logger.info(f"Updated message {message_id}")
        return message

    async def delete_subsequent(self, chat_id: UUID, after: datetime) -> int:
        """Delete all messages in a chat after a certain time.

        Used when editing a message to cascade delete subsequent messages.

        Args:
            chat_id: Chat ID.
            after: Delete messages created after this time.

        Returns:
            Number of deleted messages.
        """
        result = await self._session.execute(
            delete(Message).where(
                and_(
                    Message.chat_id == chat_id,
                    Message.created_at > after,
                )
            )
        )
        count = result.rowcount
        # Flush to ensure deletion is visible before subsequent operations
        await self._session.flush()
        logger.info(f"Deleted {count} messages in chat {chat_id} after {after}")
        return count

    async def set_research_session(
        self,
        message_id: UUID,
        research_session_id: UUID,
    ) -> Message | None:
        """Associate a message with a research session.

        Args:
            message_id: Message ID.
            research_session_id: Research session ID.

        Returns:
            Updated message or None if not found.
        """
        message = await self.get(message_id)
        if not message:
            return None

        message.research_session_id = research_session_id
        await self._session.flush()
        await self._session.refresh(message)
        return message

    async def get_conversation_history(
        self,
        chat_id: UUID,
        limit: int = 20,
    ) -> list[dict[str, str]]:
        """Get conversation history for context.

        Args:
            chat_id: Chat ID.
            limit: Maximum number of messages.

        Returns:
            List of message dicts with role and content.
        """
        query = (
            select(Message)
            .where(Message.chat_id == chat_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        result = await self._session.execute(query)
        messages = list(result.scalars().all())

        # Return in chronological order
        # Handle both enum (MessageRole) and string role values from DB
        # Filter out messages with no content (e.g., placeholder agent messages)
        return [
            {
                "role": msg.role.value if hasattr(msg.role, "value") else str(msg.role),
                "content": msg.content or "",
            }
            for msg in reversed(messages)
            if msg.content  # Skip messages with NULL content
        ]
