"""Chat service - CRUD operations for chats."""

import logging
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import and_, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.chat import Chat, ChatStatus

logger = logging.getLogger(__name__)

# Number of days before soft-deleted chats are permanently purged
PURGE_AFTER_DAYS = 30


class ChatService:
    """Service for managing chats."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize chat service.

        Args:
            session: Database session.
        """
        self._session = session

    async def create(self, user_id: str, title: str | None = None) -> Chat:
        """Create a new chat.

        Args:
            user_id: User ID.
            title: Optional chat title.

        Returns:
            Created chat.
        """
        chat = Chat(user_id=user_id, title=title)
        self._session.add(chat)
        await self._session.flush()
        await self._session.refresh(chat)
        logger.info(f"Created chat {chat.id} for user {user_id}")
        return chat

    async def get(self, chat_id: UUID, user_id: str) -> Chat | None:
        """Get a chat by ID.

        Args:
            chat_id: Chat ID.
            user_id: User ID (for ownership check).

        Returns:
            Chat if found and owned by user, None otherwise.
        """
        result = await self._session.execute(
            select(Chat).where(
                and_(
                    Chat.id == chat_id,
                    Chat.user_id == user_id,
                    Chat.deleted_at.is_(None),
                )
            )
        )
        return result.scalar_one_or_none()

    async def get_by_id(self, chat_id: UUID) -> Chat | None:
        """Get a chat by ID without user ownership filter.

        Used for authorization checks to distinguish:
        - Chat doesn't exist (allow draft chat flow)
        - Chat exists but belongs to another user (reject with 403)

        Args:
            chat_id: Chat ID.

        Returns:
            Chat if found (regardless of owner), None if not found.
        """
        result = await self._session.execute(
            select(Chat).where(
                and_(
                    Chat.id == chat_id,
                    Chat.deleted_at.is_(None),
                )
            )
        )
        return result.scalar_one_or_none()

    async def list(
        self,
        user_id: str,
        status: ChatStatus | None = None,
        limit: int = 50,
        offset: int = 0,
        search: str | None = None,
    ) -> tuple[list[Chat], int]:
        """List chats for a user.

        Args:
            user_id: User ID.
            status: Optional status filter.
            limit: Maximum number of chats.
            offset: Number of chats to skip.
            search: Optional search term.

        Returns:
            Tuple of (chats, total_count).
        """
        # Base condition
        conditions = [
            Chat.user_id == user_id,
            Chat.deleted_at.is_(None),
        ]

        if status:
            conditions.append(Chat.status == status)

        if search:
            conditions.append(Chat.title.ilike(f"%{search}%"))

        # Get total count
        count_query = select(func.count(Chat.id)).where(and_(*conditions))
        count_result = await self._session.execute(count_query)
        total = count_result.scalar() or 0

        # Get chats
        query = (
            select(Chat)
            .where(and_(*conditions))
            .order_by(Chat.updated_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(query)
        chats = list(result.scalars().all())

        return chats, total

    async def update(
        self,
        chat_id: UUID,
        user_id: str,
        title: str | None = None,
        status: ChatStatus | None = None,
    ) -> Chat | None:
        """Update a chat.

        Args:
            chat_id: Chat ID.
            user_id: User ID.
            title: New title.
            status: New status.

        Returns:
            Updated chat or None if not found.
        """
        chat = await self.get(chat_id, user_id)
        if not chat:
            return None

        if title is not None:
            chat.title = title
        if status is not None:
            chat.status = status

        chat.updated_at = datetime.now(UTC)
        await self._session.flush()
        await self._session.refresh(chat)
        logger.info(f"Updated chat {chat_id}")
        return chat

    async def soft_delete(self, chat_id: UUID, user_id: str) -> bool:
        """Soft delete a chat.

        Args:
            chat_id: Chat ID.
            user_id: User ID.

        Returns:
            True if deleted, False if not found.
        """
        chat = await self.get(chat_id, user_id)
        if not chat:
            return False

        chat.status = ChatStatus.DELETED
        chat.deleted_at = datetime.now(UTC)
        chat.updated_at = datetime.now(UTC)
        await self._session.flush()
        logger.info(f"Soft deleted chat {chat_id}")
        return True

    async def restore(self, chat_id: UUID, user_id: str) -> Chat | None:
        """Restore a soft-deleted chat.

        Args:
            chat_id: Chat ID.
            user_id: User ID.

        Returns:
            Restored chat or None if not found.
        """
        result = await self._session.execute(
            select(Chat).where(
                and_(
                    Chat.id == chat_id,
                    Chat.user_id == user_id,
                    Chat.deleted_at.is_not(None),
                )
            )
        )
        chat = result.scalar_one_or_none()
        if not chat:
            return None

        chat.status = ChatStatus.ACTIVE
        chat.deleted_at = None
        chat.updated_at = datetime.now(UTC)
        await self._session.flush()
        await self._session.refresh(chat)
        logger.info(f"Restored chat {chat_id}")
        return chat

    async def update_title_from_message(
        self,
        chat_id: UUID,
        message_content: str,
    ) -> None:
        """Update chat title from first message if not set.

        Args:
            chat_id: Chat ID.
            message_content: Message content to use for title.
        """
        result = await self._session.execute(select(Chat).where(Chat.id == chat_id))
        chat = result.scalar_one_or_none()
        if chat and not chat.title:
            # Truncate to reasonable title length
            title = message_content[:100].strip()
            if len(message_content) > 100:
                title = title.rsplit(" ", 1)[0] + "..."
            chat.title = title
            chat.updated_at = datetime.now(UTC)
            await self._session.flush()

    async def purge_deleted_chats(self, days_old: int = PURGE_AFTER_DAYS) -> int:
        """Permanently delete chats that were soft-deleted more than N days ago.

        This is a background job method that should be called periodically.
        It permanently removes chats and their associated messages.

        Args:
            days_old: Number of days after soft-delete before permanent purge.

        Returns:
            Number of chats permanently deleted.
        """
        cutoff_date = datetime.now(UTC) - timedelta(days=days_old)

        # Find chats to purge
        result = await self._session.execute(
            select(Chat.id).where(
                and_(
                    Chat.deleted_at.is_not(None),
                    Chat.deleted_at < cutoff_date,
                )
            )
        )
        chat_ids = [row[0] for row in result.all()]

        if not chat_ids:
            logger.info("No chats to purge")
            return 0

        # Delete chats (cascade will handle messages)
        await self._session.execute(
            delete(Chat).where(Chat.id.in_(chat_ids))
        )
        await self._session.flush()

        logger.info(f"Permanently purged {len(chat_ids)} deleted chats")
        return len(chat_ids)
