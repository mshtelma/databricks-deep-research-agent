"""Export service for converting chats to various formats."""

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.v1.utils import verify_message_ownership
from src.core.exceptions import NotFoundError
from src.models.message import Message
from src.models.research_session import ResearchSession
from src.models.source import Source
from src.services.chat_service import ChatService
from src.services.claim_service import ClaimService
from src.services.message_service import MessageService
from src.services.verification_summary_service import VerificationSummaryService

logger = logging.getLogger(__name__)


class ExportService:
    """Service for exporting chats to various formats.

    Supports dependency injection for testability.
    """

    def __init__(
        self,
        session: AsyncSession,
        chat_service: ChatService | None = None,
        message_service: MessageService | None = None,
        claim_service: ClaimService | None = None,
        summary_service: VerificationSummaryService | None = None,
    ) -> None:
        """Initialize export service.

        Args:
            session: Database session.
            chat_service: Optional injected ChatService (created if None).
            message_service: Optional injected MessageService (created if None).
            claim_service: Optional injected ClaimService (created if None).
            summary_service: Optional injected VerificationSummaryService (created if None).
        """
        self._session = session
        self._chat_service = chat_service or ChatService(session)
        self._message_service = message_service or MessageService(session)
        self._claim_service = claim_service or ClaimService(session)
        self._summary_service = summary_service or VerificationSummaryService(session)

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
        chat = await self._chat_service.get_for_user(chat_id, user_id)
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
        """
        # Verify ownership
        await self._get_message_with_auth(message_id, user_id)

        # Get claims with citations (use injected service)
        claims = await self._claim_service.list_by_message(
            message_id, include_citations=True
        )

        # Get verification summary (use injected service)
        summary = await self._summary_service.get_or_compute(message_id)

        lines: list[str] = []

        # Header
        lines.extend([
            "# Verification Report",
            "",
            f"*Generated on {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC*",
            "",
        ])

        # Summary table
        total = summary.total_claims or 1  # Avoid division by zero

        lines.extend([
            "| Metric | Count |",
            "|--------|-------|",
            f"| Total Claims | {summary.total_claims} |",
            f"| Supported | {summary.supported_count} ({summary.supported_count * 100 // total}%) |",
            f"| Partial | {summary.partial_count} ({summary.partial_count * 100 // total}%) |",
            f"| Unsupported | {summary.unsupported_count} ({summary.unsupported_count * 100 // total}%) |",
            f"| Contradicted | {summary.contradicted_count} ({summary.contradicted_count * 100 // total}%) |",
            "",
        ])

        if summary.warning:
            lines.extend([
                "> **Warning**: High rate of unsupported or contradicted claims detected.",
                "",
            ])

        lines.extend(["---", ""])

        # Claims detail
        if claims:
            lines.extend(["## Claims", ""])

            for i, claim in enumerate(claims, 1):
                verdict = (
                    claim.verification_verdict.upper()
                    if claim.verification_verdict
                    else "PENDING"
                )
                lines.extend([
                    f"### {i}. {verdict}",
                    "",
                    f"> \"{claim.claim_text}\"",
                    "",
                ])

                # Evidence from citations
                if claim.citations:
                    lines.append("**Evidence:**")
                    for citation in claim.citations:
                        span = citation.evidence_span
                        source = span.source if span else None
                        if source and span:
                            title = source.title or source.url
                            primary_mark = " (Primary)" if citation.is_primary else ""
                            lines.append(f"- [{title}]({source.url}){primary_mark}")
                            if span.quote_text:
                                quote = span.quote_text[:200]
                                if len(span.quote_text) > 200:
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
