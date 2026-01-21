"""Export service for converting chats to various formats.

JSONB Migration (Migration 011):
Claims and verification data are now read from the verification_data JSONB column
on the research_sessions table instead of normalized tables.

N+1 Query Optimization:
Uses batch loading for sources to avoid O(2n) queries during export.
The _get_sources_for_messages() method loads all sources for multiple messages
in a single JOIN query.
"""

import logging
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from deep_research.core.exceptions import NotFoundError
from deep_research.models.message import Message
from deep_research.models.research_session import ResearchSession
from deep_research.models.source import Source
from deep_research.services.chat_service import ChatService
from deep_research.services.message_service import MessageService

logger = logging.getLogger(__name__)


class ExportService:
    """Service for exporting chats to various formats.

    Supports dependency injection for testability.

    JSONB Migration: Claims and verification data are now read from
    verification_data JSONB column instead of claim_service/summary_service.
    """

    def __init__(
        self,
        session: AsyncSession,
        chat_service: ChatService | None = None,
        message_service: MessageService | None = None,
    ) -> None:
        """Initialize export service.

        Args:
            session: Database session.
            chat_service: Optional injected ChatService (created if None).
            message_service: Optional injected MessageService (created if None).
        """
        self._session = session
        self._chat_service = chat_service or ChatService(session)
        self._message_service = message_service or MessageService(session)

    async def export_markdown(
        self,
        chat_id: UUID,
        user_id: str,
        include_metadata: bool = True,
        include_sources: bool = True,
    ) -> str:
        """Export chat to Markdown format.

        Uses batch loading for sources to avoid N+1 queries.

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
        chat = await self._chat_service.get_for_user(chat_id, user_id)
        if not chat:
            raise ValueError(f"Chat {chat_id} not found")

        # Get messages
        messages, _ = await self._message_service.list_messages(
            chat_id=chat_id,
            limit=1000,  # Get all messages
            offset=0,
        )

        # Batch load sources for ALL agent messages at once (N+1 fix)
        sources_by_message: dict[UUID, list[dict[str, Any]]] = {}
        if include_sources:
            agent_message_ids = [
                msg.id for msg in messages if msg.role.value == "agent"
            ]
            sources_by_message = await self._get_sources_for_messages(agent_message_ids)

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

            # Add sources for agent messages if available (O(1) lookup)
            if include_sources and msg.role.value == "agent":
                sources = sources_by_message.get(msg.id, [])
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

        Note:
            For batch operations, prefer _get_sources_for_messages() to avoid N+1 queries.
        """
        # Get research session for this message
        session_result = await self._session.execute(
            select(ResearchSession).where(ResearchSession.message_id == message_id)
        )
        session = session_result.scalar_one_or_none()

        if not session:
            return []

        # Get sources
        sources_result = await self._session.execute(
            select(Source).where(Source.research_session_id == session.id)
        )
        sources = sources_result.scalars().all()

        return [
            {"title": s.title or s.url, "url": s.url}
            for s in sources
        ]

    async def _get_sources_for_messages(
        self,
        message_ids: list[UUID],
    ) -> dict[UUID, list[dict[str, Any]]]:
        """Batch load sources for multiple messages in ONE query.

        Uses JOIN to avoid N+1 queries. Returns mapping from message_id
        to list of source dicts.

        This optimizes export operations from O(2n) queries to O(1) where n
        is the number of agent messages.

        Args:
            message_ids: List of message IDs to load sources for.

        Returns:
            Dict mapping message_id to list of {"title": str, "url": str}.
        """
        if not message_ids:
            return {}

        # Single query with JOIN - O(1) instead of O(2n)
        result = await self._session.execute(
            select(Source, ResearchSession.message_id)
            .join(ResearchSession, Source.research_session_id == ResearchSession.id)
            .where(ResearchSession.message_id.in_(message_ids))
        )

        # Group by message_id in memory
        sources_by_message: dict[UUID, list[dict[str, Any]]] = defaultdict(list)
        for source, message_id in result.all():
            sources_by_message[message_id].append({
                "title": source.title or source.url,
                "url": source.url,
            })

        return dict(sources_by_message)

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
        chat = await self._chat_service.get_for_user(chat_id, user_id)
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

    async def _get_message_with_auth(
        self,
        message_id: UUID,
        user_id: str,
    ) -> Message:
        """Get message after verifying authorization.

        Uses the shared authorization utility from API layer.

        Args:
            message_id: Message ID.
            user_id: User ID for authorization.

        Returns:
            The message if authorized.

        Raises:
            ValueError: If message not found or not owned by user.
        """
        # Local import to avoid circular import through api.v1.__init__.py
        from deep_research.api.v1.utils.authorization import verify_message_ownership

        try:
            return await verify_message_ownership(message_id, user_id, self._session)
        except NotFoundError as e:
            raise ValueError(str(e)) from e

    async def export_report_markdown(
        self,
        message_id: UUID,
        user_id: str,
    ) -> str:
        """Export agent response as standalone markdown report.

        Args:
            message_id: Agent message ID containing the synthesis.
            user_id: User ID for authorization.

        Returns:
            Markdown formatted research report.

        Raises:
            ValueError: If message not found or not owned by user.
        """
        # Verify ownership and get message
        message = await self._get_message_with_auth(message_id, user_id)

        if not message.content:
            raise ValueError("Message has no content to export")

        # Get research session for metadata
        session_result = await self._session.execute(
            select(ResearchSession).where(ResearchSession.message_id == message_id)
        )
        session = session_result.scalar_one_or_none()

        lines: list[str] = []

        # Title from query or fallback
        if session and session.query:
            query_title = session.query[:100]
            if len(session.query) > 100:
                query_title += "..."
        else:
            query_title = "Research Report"

        lines.extend([
            f"# {query_title}",
            "",
        ])

        # Metadata
        lines.append(
            "*Generated by Deep Research Agent*  "
        )
        lines.append(
            f"*Date: {datetime.now(UTC).strftime('%Y-%m-%d')}*"
        )

        if session and session.research_depth:
            # Handle both enum and string values (DB may return string)
            depth_value = (
                session.research_depth.value
                if hasattr(session.research_depth, "value")
                else str(session.research_depth)
            )
            depth_label = depth_value.title()
            lines[-1] = f"*Date: {datetime.now(UTC).strftime('%Y-%m-%d')} | Depth: {depth_label}*"

        lines.extend(["", "---", ""])

        # Main content
        lines.append(message.content)

        lines.extend(["", "---", ""])

        # Sources section
        if session:
            sources_result = await self._session.execute(
                select(Source)
                .where(Source.research_session_id == session.id)
                .order_by(Source.fetched_at)
            )
            sources = sources_result.scalars().all()

            if sources:
                lines.extend(["## Sources", ""])
                for i, source in enumerate(sources, 1):
                    title = source.title or source.url
                    lines.append(f"{i}. **{title}** - {source.url}")
                lines.extend([""])

        return "\n".join(lines)

    async def export_provenance_markdown(
        self,
        message_id: UUID,
        user_id: str,
    ) -> str:
        """Export verification report as markdown.

        Args:
            message_id: Agent message ID.
            user_id: User ID for authorization.

        Returns:
            Markdown formatted verification report.

        Raises:
            ValueError: If message not found or not owned by user.

        JSONB Migration: Now reads from verification_data JSONB column.
        """
        # Verify ownership
        await self._get_message_with_auth(message_id, user_id)

        # Get research session with verification_data
        result = await self._session.execute(
            select(ResearchSession).where(ResearchSession.message_id == message_id)
        )
        session = result.scalar_one_or_none()

        lines: list[str] = []

        # Header
        lines.extend([
            "# Verification Report",
            "",
            f"*Generated on {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC*",
            "",
        ])

        if not session or not session.verification_data:
            lines.extend([
                "*No claims found for this message.*",
                "",
            ])
            return "\n".join(lines)

        verification_data = session.verification_data
        summary_dict = verification_data.get("summary", {})
        claims_data = verification_data.get("claims", [])

        # Summary table
        total = summary_dict.get("total_claims", 0) or 1  # Avoid division by zero
        supported = summary_dict.get("supported_count", 0)
        partial = summary_dict.get("partial_count", 0)
        unsupported = summary_dict.get("unsupported_count", 0)
        contradicted = summary_dict.get("contradicted_count", 0)

        lines.extend([
            "| Metric | Count |",
            "|--------|-------|",
            f"| Total Claims | {total} |",
            f"| Supported | {supported} ({supported * 100 // total}%) |",
            f"| Partial | {partial} ({partial * 100 // total}%) |",
            f"| Unsupported | {unsupported} ({unsupported * 100 // total}%) |",
            f"| Contradicted | {contradicted} ({contradicted * 100 // total}%) |",
            "",
        ])

        if summary_dict.get("warning"):
            lines.extend([
                "> **Warning**: High rate of unsupported or contradicted claims detected.",
                "",
            ])

        lines.extend(["---", ""])

        # Claims detail
        if claims_data:
            lines.extend(["## Claims", ""])

            for i, claim_dict in enumerate(claims_data, 1):
                verdict = (
                    claim_dict.get("verification_verdict", "").upper()
                    if claim_dict.get("verification_verdict")
                    else "PENDING"
                )
                lines.extend([
                    f"### {i}. {verdict}",
                    "",
                    f"> \"{claim_dict.get('claim_text', '')}\"",
                    "",
                ])

                # Evidence from embedded data
                evidence = claim_dict.get("evidence")
                if evidence:
                    lines.append("**Evidence:**")
                    title = evidence.get("source_title") or evidence.get("source_url", "")
                    url = evidence.get("source_url", "")
                    lines.append(f"- [{title}]({url}) (Primary)")
                    quote_text = evidence.get("quote_text", "")
                    if quote_text:
                        quote = quote_text[:200]
                        if len(quote_text) > 200:
                            quote += "..."
                        lines.append(f"  > \"{quote}\"")
                    lines.append("")

                lines.extend(["---", ""])
        else:
            lines.extend([
                "*No claims found for this message.*",
                "",
            ])

        return "\n".join(lines)
