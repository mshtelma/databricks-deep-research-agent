"""Retry-path run-artifact loading + re-export of the framework evidence core.

The pure evidence assembly (``build_evidence`` / ``render_evidence_block`` /
``build_legend`` + ``EvidenceItem``) moved to
``databricks_deep_research.surface.evidence`` so the standalone shell-app shares
it. The DB-backed ``load_run_artifacts`` (restructure endpoint) stays here
because it reads the app's storage (cached ChatState doc / legacy ORM rows).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from databricks_deep_research.surface.evidence import (
    EVIDENCE_BLOCK_BUDGET,
    FULL_CONTENT_TOP_K,
    MAX_EVIDENCE_ITEMS,
    EvidenceItem,
    build_evidence,
    build_legend,
    render_evidence_block,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunArtifacts:
    """A persisted run's artifacts needed to (re)fill structured slots."""

    report: str
    claims: list[Any]
    sources: list[Any]
    research_session_id: UUID
    envelope: dict[str, Any] | None


async def load_run_artifacts(
    chat_id: UUID,
    message_id: UUID,
    storage_stack: Any | None = None,
) -> RunArtifacts | None:
    """Load report/claims/sources/envelope for one agent message.

    Retry-path loader for the restructure endpoint: reads the persisted run
    on BOTH storage modes (cached ChatState doc / legacy ORM rows via an
    independent session). Claims come back as the plain dicts stored in
    ``verification_data["claims"]`` — every consumer here is dict-tolerant.
    Fail-soft: returns None when the message or its research session is
    missing.
    """
    from deep_research.core.config import get_settings

    settings = get_settings()
    if settings.storage_service_impl == "cached" and storage_stack is not None:
        try:
            doc = await storage_stack.cache.get(chat_id)
        except Exception:  # noqa: BLE001 — fail-soft by contract
            logger.exception("RUN_ARTIFACTS_CACHED_LOAD_FAILED chat=%s",
                             str(chat_id)[:8])
            return None
        if doc is None:
            return None
        message = next(
            (m for m in doc.state.messages if m.id == message_id), None
        )
        session = next(
            (
                rs
                for rs in doc.state.research_sessions
                if rs.message_id == message_id
            ),
            None,
        )
        if message is None or session is None:
            return None
        verification = session.verification_data or {}
        sources = [
            s
            for s in doc.state.sources
            if (s.metadata or {}).get("research_session_id")
            == str(session.id)
        ]
        return RunArtifacts(
            report=message.content or "",
            claims=list(verification.get("claims") or []),
            sources=sources,
            research_session_id=session.id,
            envelope=verification.get("structured_output"),
        )

    from sqlalchemy import select

    from deep_research.db.session import get_session_maker
    from deep_research.models.message import Message
    from deep_research.models.research_session import ResearchSession
    from deep_research.models.source import Source

    session_maker = get_session_maker()
    try:
        async with session_maker() as db:
            msg_row = await db.execute(
                select(Message.content).where(
                    Message.id == message_id, Message.chat_id == chat_id
                )
            )
            msg = msg_row.first()
            if msg is None:
                return None
            rs_row = await db.execute(
                select(
                    ResearchSession.id, ResearchSession.verification_data
                )
                .where(ResearchSession.message_id == message_id)
                .order_by(ResearchSession.started_at.desc())
            )
            rs = rs_row.first()
            if rs is None:
                return None
            source_rows = await db.execute(
                select(Source).where(Source.research_session_id == rs[0])
            )
            sources = list(source_rows.scalars().all())
    except Exception:  # noqa: BLE001 — fail-soft by contract
        logger.exception("RUN_ARTIFACTS_LOAD_FAILED message=%s",
                         str(message_id)[:8])
        return None

    verification = dict(rs[1] or {})
    return RunArtifacts(
        report=msg[0] or "",
        claims=list(verification.get("claims") or []),
        sources=sources,
        research_session_id=rs[0],
        envelope=verification.get("structured_output"),
    )


__all__ = [
    "EVIDENCE_BLOCK_BUDGET",
    "FULL_CONTENT_TOP_K",
    "MAX_EVIDENCE_ITEMS",
    "EvidenceItem",
    "RunArtifacts",
    "build_evidence",
    "build_legend",
    "load_run_artifacts",
    "render_evidence_block",
]
