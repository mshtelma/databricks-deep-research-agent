"""Export service for converting chats to various formats."""

import logging
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.services.chat_service import ChatService
from src.services.message_service import MessageService

logger = logging.getLogger(__name__)


class ExportService:
    """Service for exporting chats to various formats."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize export service.

        Args:
            session: Database session.
        """
        self._session = session
        self._chat_service = ChatService(session)
        self._message_service = MessageService(session)

    async def export_markdown(
        self,
        chat_id: UUID,
        user_id: str,
        include_metadata: bool = True,
        include_sources: bool = True,
    ) -> str:
        """Export chat to Markdown format.

        Args:
            chat_id: Chat ID to export.
            user_id: User ID for ownership verification.
            include_metadata: Include chat metadata header.
            include_sources: Include source citations.

        Returns:
            Markdown formatted string.

        Raises:
            ValueError: If chat not found or not owned by user.
        """
        # Get chat
        chat = await self._chat_service.get(chat_id, user_id)
        if not chat:
            raise ValueError(f"Chat {chat_id} not found")

        # Get messages
        messages, _ = await self._message_service.list_messages(
            chat_id=chat_id,
            limit=1000,  # Get all messages
            offset=0,
        )

        # Build markdown
        lines: list[str] = []

        # Header
        if include_metadata:
            title = chat.title or "Untitled Chat"
            lines.extend([
                f"# {title}",
                "",
                f"**Exported**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"**Created**: {chat.created_at.strftime('%Y-%m-%d %H:%M:%S') if chat.created_at else 'Unknown'}",
                "",
                "---",
                "",
            ])

        # Messages
        for msg in messages:
            role_label = "**You**" if msg.role.value == "user" else "**Agent**"
            lines.extend([
                f"### {role_label}",
                "",
                msg.content or "",
                "",
            ])

            # Add sources for agent messages if available
            if include_sources and msg.role.value == "agent":
                sources = await self._get_message_sources(msg.id)
                if sources:
                    lines.extend([
                        "#### Sources",
                        "",
                    ])
                    for source in sources:
                        lines.append(f"- [{source['title']}]({source['url']})")
                    lines.append("")

        return "\n".join(lines)

    async def _get_message_sources(self, message_id: UUID) -> list[dict[str, Any]]:
        """Get sources for a message from research session.

        Args:
            message_id: Message ID.

        Returns:
            List of source dictionaries with title and url.
        """
        from sqlalchemy import select

        from src.models.research_session import ResearchSession
        from src.models.source import Source

        # Get research session for this message
        result = await self._session.execute(
            select(ResearchSession).where(ResearchSession.message_id == message_id)
        )
        session = result.scalar_one_or_none()

        if not session:
            return []

        # Get sources
        result = await self._session.execute(
            select(Source).where(Source.research_session_id == session.id)
        )
        sources = result.scalars().all()

        return [
            {"title": s.title or s.url, "url": s.url}
            for s in sources
        ]

    async def export_json(
        self,
        chat_id: UUID,
        user_id: str,
    ) -> dict[str, Any]:
        """Export chat to JSON format.

        Args:
            chat_id: Chat ID to export.
            user_id: User ID for ownership verification.

        Returns:
            Dictionary with chat data.

        Raises:
            ValueError: If chat not found or not owned by user.
        """
        # Get chat
        chat = await self._chat_service.get(chat_id, user_id)
        if not chat:
            raise ValueError(f"Chat {chat_id} not found")

        # Get messages
        messages, total = await self._message_service.list_messages(
            chat_id=chat_id,
            limit=1000,
            offset=0,
        )

        return {
            "id": str(chat.id),
            "title": chat.title,
            "status": chat.status.value if chat.status else None,
            "created_at": chat.created_at.isoformat() if chat.created_at else None,
            "updated_at": chat.updated_at.isoformat() if chat.updated_at else None,
            "message_count": total,
            "messages": [
                {
                    "id": str(msg.id),
                    "role": msg.role.value,
                    "content": msg.content,
                    "created_at": msg.created_at.isoformat() if msg.created_at else None,
                    "is_edited": msg.is_edited,
                }
                for msg in messages
            ],
        }
