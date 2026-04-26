"""Cache-backed ``IExportService`` — assembles exports from cached services.

All reads go through ``IChatService``, ``IMessageService``,
``IResearchSessionService``, and ``ISourceService`` — no direct SQL.

This service is READ-ONLY; it never mutates any state.

Architecture decision (F-OTHER.5): export_report_markdown and
export_provenance_markdown access ``research_session.verification_data``
which is stored in ``ChatState.research_sessions[].verification_data``
via ``CachedResearchSessionService``. The cached path therefore covers
all four export methods without any ORM fallback.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from deep_research.services._protocols import (
    IChatService,
    IExportService,
    IMessageService,
    IResearchSessionService,
    ISourceService,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class CachedExportService(IExportService):
    """``IExportService`` implemented by composing cached service reads.

    Thread-safety: stateless; all methods are independently safe for
    concurrent calls.
    """

    def __init__(
        self,
        chat_service: IChatService,
        message_service: IMessageService,
        research_session_service: IResearchSessionService,
        source_service: ISourceService,
    ) -> None:
        self._chat_service = chat_service
        self._message_service = message_service
        self._research_session_service = research_session_service
        self._source_service = source_service

    # -------------------------------------------------------------------------
    # Public export methods
    # -------------------------------------------------------------------------

    async def export_markdown(
        self,
        chat_id: UUID,
        user_id: str,
        include_metadata: bool = True,
        include_sources: bool = True,
    ) -> str:
        """Export full chat conversation as Markdown.

        Raises:
            ValueError: If chat is not found or not owned by ``user_id``.
        """
        chat = await self._chat_service.get_for_user(chat_id, user_id)
        if not chat:
            raise ValueError(f"Chat {chat_id} not found")

        messages, _ = await self._message_service.list_messages(
            chat_id=chat_id, limit=1000, offset=0
        )

        # Build per-message source mapping (by research_session_id on each msg)
        sources_by_message: dict[UUID, list[dict[str, Any]]] = {}
        if include_sources:
            for msg in messages:
                role_val = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
                if role_val != "agent":
                    continue
                rs_id = getattr(msg, "research_session_id", None)
                if rs_id is None:
                    continue
                try:
                    srcs = await self._source_service.list_by_session(
                        rs_id, chat_id=chat_id
                    )
                    sources_by_message[msg.id] = [
                        {"title": s.title or s.url, "url": s.url} for s in srcs
                    ]
                except Exception:
                    logger.debug(
                        "EXPORT_SOURCES_LOAD_FAILED msg=%s", msg.id, exc_info=True
                    )

        lines: list[str] = []

        if include_metadata:
            title = chat.title or "Untitled Chat"
            lines.extend(
                [
                    f"# {title}",
                    "",
                    f"**Exported**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    f"**Created**: {chat.created_at.strftime('%Y-%m-%d %H:%M:%S') if chat.created_at else 'Unknown'}",
                    "",
                    "---",
                    "",
                ]
            )

        for msg in messages:
            role_val = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            role_label = "**You**" if role_val == "user" else "**Agent**"
            lines.extend([f"### {role_label}", "", msg.content or "", ""])

            if include_sources and role_val == "agent":
                sources = sources_by_message.get(msg.id, [])
                if sources:
                    lines.extend(["#### Sources", ""])
                    for source in sources:
                        lines.append(f"- [{source['title']}]({source['url']})")
                    lines.append("")

        return "\n".join(lines)

    async def export_json(
        self,
        chat_id: UUID,
        user_id: str,
    ) -> dict[str, Any]:
        """Export full chat conversation as a JSON-serialisable dict.

        Raises:
            ValueError: If chat is not found or not owned by ``user_id``.
        """
        chat = await self._chat_service.get_for_user(chat_id, user_id)
        if not chat:
            raise ValueError(f"Chat {chat_id} not found")

        messages, total = await self._message_service.list_messages(
            chat_id=chat_id, limit=1000, offset=0
        )

        return {
            "id": str(chat.id),
            "title": chat.title,
            "status": chat.status.value if chat.status and hasattr(chat.status, "value") else chat.status,
            "created_at": chat.created_at.isoformat() if chat.created_at else None,
            "updated_at": chat.updated_at.isoformat() if chat.updated_at else None,
            "message_count": total,
            "messages": [
                {
                    "id": str(msg.id),
                    "role": msg.role.value if hasattr(msg.role, "value") else str(msg.role),
                    "content": msg.content,
                    "created_at": msg.created_at.isoformat() if msg.created_at else None,
                    "is_edited": getattr(msg, "is_edited", False),
                }
                for msg in messages
            ],
        }

    async def export_report_markdown(
        self,
        message_id: UUID,
        user_id: str,
    ) -> str:
        """Export a single agent message as a standalone Markdown report.

        Raises:
            ValueError: If message is not found or not owned by ``user_id``.
        """
        # Ownership check via message lookup — requires chat_id.
        # We search across the user's accessible chats for this message.
        message, chat_id = await self._find_message_for_user(message_id, user_id)
        if not message:
            raise ValueError(f"Message {message_id} not found or not accessible")
        if not message.content:
            raise ValueError("Message has no content to export")

        # Load research session for metadata
        rs = await self._get_research_session_for_message(message_id, chat_id)

        lines: list[str] = []

        if rs and getattr(rs, "query", None):
            query_title = rs.query[:100]
            if len(rs.query) > 100:
                query_title += "..."
        else:
            query_title = "Research Report"

        lines.extend([f"# {query_title}", ""])
        lines.append("*Generated by Deep Research Agent*  ")
        date_str = datetime.now(UTC).strftime("%Y-%m-%d")

        if rs and getattr(rs, "research_depth", None):
            depth = rs.research_depth
            depth_val = depth.value if hasattr(depth, "value") else str(depth)
            lines.append(f"*Date: {date_str} | Depth: {depth_val.title()}*")
        else:
            lines.append(f"*Date: {date_str}*")

        lines.extend(["", "---", "", message.content, "", "---", ""])

        if rs and chat_id:
            rs_id = rs.id
            try:
                sources = await self._source_service.list_by_session(
                    rs_id, chat_id=chat_id
                )
            except Exception:
                sources = []

            if sources:
                lines.extend(["## Sources", ""])
                for i, source in enumerate(sources, 1):
                    title = source.title or source.url
                    lines.append(f"{i}. **{title}** - {source.url}")
                lines.append("")

        return "\n".join(lines)

    async def export_provenance_markdown(
        self,
        message_id: UUID,
        user_id: str,
    ) -> str:
        """Export the verification / provenance report for a message as Markdown.

        Raises:
            ValueError: If message is not found or not owned by ``user_id``.
        """
        message, chat_id = await self._find_message_for_user(message_id, user_id)
        if not message:
            raise ValueError(f"Message {message_id} not found or not accessible")

        rs = await self._get_research_session_for_message(message_id, chat_id)

        lines: list[str] = [
            "# Verification Report",
            "",
            f"*Generated on {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC*",
            "",
        ]

        verification_data = None
        if rs:
            verification_data = getattr(rs, "verification_data", None)

        if not verification_data:
            lines.extend(["*No claims found for this message.*", ""])
            return "\n".join(lines)

        summary_dict = verification_data.get("summary", {})
        claims_data = verification_data.get("claims", [])

        total = summary_dict.get("total_claims", 0) or 1
        supported = summary_dict.get("supported_count", 0)
        partial = summary_dict.get("partial_count", 0)
        unsupported = summary_dict.get("unsupported_count", 0)
        contradicted = summary_dict.get("contradicted_count", 0)

        lines.extend(
            [
                "| Metric | Count |",
                "|--------|-------|",
                f"| Total Claims | {total} |",
                f"| Supported | {supported} ({supported * 100 // total}%) |",
                f"| Partial | {partial} ({partial * 100 // total}%) |",
                f"| Unsupported | {unsupported} ({unsupported * 100 // total}%) |",
                f"| Contradicted | {contradicted} ({contradicted * 100 // total}%) |",
                "",
            ]
        )

        if summary_dict.get("warning"):
            lines.extend(
                ["> **Warning**: High rate of unsupported or contradicted claims detected.", ""]
            )

        lines.extend(["---", ""])

        if claims_data:
            lines.extend(["## Claims", ""])
            for i, claim_dict in enumerate(claims_data, 1):
                verdict = (
                    claim_dict.get("verification_verdict", "").upper()
                    if claim_dict.get("verification_verdict")
                    else "PENDING"
                )
                lines.extend(
                    [
                        f"### {i}. {verdict}",
                        "",
                        f"> \"{claim_dict.get('claim_text', '')}\"",
                        "",
                    ]
                )
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
            lines.extend(["*No claims found for this message.*", ""])

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    async def _find_message_for_user(
        self,
        message_id: UUID,
        user_id: str,
    ) -> tuple[Any, UUID | None]:
        """Locate a message by searching the user's chats.

        Returns ``(message, chat_id)`` or ``(None, None)`` if not found.

        Note: This is an O(chats) search. It is acceptable for the cold
        export path (rare, user-triggered). A future optimisation could
        add a message-to-chat index.
        """
        # Try to get all chats for the user (up to 200 for the search window)
        try:
            chats, _ = await self._chat_service.list(user_id, limit=200, offset=0)
        except Exception:
            logger.warning("EXPORT_CHAT_LIST_FAILED user=%s", user_id, exc_info=True)
            return None, None

        for chat in chats:
            chat_id: UUID = chat.id
            try:
                messages, _ = await self._message_service.list_messages(
                    chat_id=chat_id, limit=1000, offset=0
                )
                for msg in messages:
                    if msg.id == message_id:
                        return msg, chat_id
            except Exception:
                continue

        return None, None

    async def _get_research_session_for_message(
        self,
        message_id: UUID,
        chat_id: UUID | None,
    ) -> Any | None:
        """Return the research session linked to ``message_id`` or None."""
        if chat_id is None:
            return None
        try:
            return await self._research_session_service.get_by_message(
                message_id, chat_id=chat_id
            )
        except Exception:
            logger.debug(
                "EXPORT_RS_LOAD_FAILED message=%s chat=%s", message_id, chat_id, exc_info=True
            )
            return None
