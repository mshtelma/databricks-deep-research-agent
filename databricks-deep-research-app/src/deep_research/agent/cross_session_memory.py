"""Cross-session memory READ path (app side) — recall ``(user_id, agent_id)`` facts.

The WRITE half is shipped: every completed research turn consolidates its
verified claims into durable ``chat_memory_findings`` (see
``turn_consolidation.consolidate_turn_knowledge`` /
``CachedChatMemoryService.consolidate_from_pool``). This module is the READ
half — at run start it retrieves a user's durable facts and injects them as a
spotlighted **role=user** block so the agent recalls prior corrections /
preferences across sessions (spec §4.1, DeerFlow-informed).

Two pieces:

* ``CrossSessionMemoryStore`` — the storage seam. ``read_facts(user_id,
  agent_id, ...)`` yields framework ``CrossSessionFact`` projections. Injectable
  so the keying contract is honored regardless of which backend populates it and
  so the READ pipeline is unit-testable with a stub.
* ``inject_cross_session_memory`` — the fail-soft, hard-timeout wrapper the
  orchestrator calls. A memory-backend error OR a slow read MUST NEVER block or
  fail the research request: the read is bounded with ``asyncio.wait_for`` and
  guarded by a broad ``try/except`` that logs and degrades to **no memory**.

The default concrete store (``ChatMemoryFindingsStore``) reads the existing
per-chat ``chat_memory_findings`` across the user's chats (the WRITE-path table),
giving cross-session recall without a new table or any WRITE-path change.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Protocol

from databricks_deep_research.memory import (
    DEFAULT_MAX_FACTS,
    DEFAULT_MIN_CONFIDENCE,
    ConfidenceLabel,
    CrossSessionFact,
    build_cross_session_memory_message,
)

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

DEFAULT_READ_TIMEOUT_SECONDS: float = 3.0
"""Hard ceiling on the cross-session read. A read slower than this degrades to
no-memory rather than delaying the research request."""


class CrossSessionMemoryStore(Protocol):
    """Storage seam for cross-session memory.

    Implementations retrieve durable facts for ``(user_id, agent_id)`` ordered
    most-relevant/recent first. They MUST be read-only and SHOULD apply their
    own confidence filter + cap where the backend can (the framework
    ``select_facts`` policy re-applies both, so a loose store still yields a
    bounded, thresholded block).
    """

    async def read_facts(
        self,
        *,
        user_id: str,
        agent_id: str | None,
        exclude_chat_id: UUID | None = None,
        min_confidence: ConfidenceLabel = DEFAULT_MIN_CONFIDENCE,
        limit: int = DEFAULT_MAX_FACTS,
    ) -> list[CrossSessionFact]:
        """Return durable facts for ``(user_id, agent_id)``; never raises for
        an empty result (returns ``[]``)."""
        ...


class ChatMemoryFindingsStore:
    """Default store: read durable facts from ``chat_memory_findings``.

    Cross-session recall is achieved by reading the user's findings across all
    their chats (optionally excluding the current chat so only PRIOR sessions
    contribute). The current schema has no ``agent_id`` column on chats — the
    binding arrives per-request — so ``agent_id`` is accepted for the keying
    contract and logged; scoping is by ``user_id`` over the durable findings the
    WRITE path produced. A future migration that stamps ``agent_id`` onto
    findings can tighten this without changing the seam.

    Legacy (SQLAlchemy session) path only. The cached path is event-sourced and
    its cross-chat read is a follow-up; ``inject_cross_session_memory`` fails
    soft to no-memory when no session is available, so the default behavior is
    unchanged on cached deployments.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def read_facts(
        self,
        *,
        user_id: str,
        agent_id: str | None,
        exclude_chat_id: UUID | None = None,
        min_confidence: ConfidenceLabel = DEFAULT_MIN_CONFIDENCE,
        limit: int = DEFAULT_MAX_FACTS,
    ) -> list[CrossSessionFact]:
        from sqlalchemy import select

        from deep_research.models.chat import Chat
        from deep_research.models.chat_memory_finding import ChatMemoryFinding

        # Read a small confidence-allowed superset ordered newest-first; the
        # framework ``select_facts`` policy applies the final cap + ordering.
        allowed = _confidence_at_or_above(min_confidence)
        stmt = (
            select(ChatMemoryFinding)
            .join(Chat, Chat.id == ChatMemoryFinding.chat_id)
            .where(Chat.user_id == user_id)
            .where(Chat.deleted_at.is_(None))
            .where(ChatMemoryFinding.confidence.in_(allowed))
            .order_by(ChatMemoryFinding.created_at.desc())
            .limit(max(limit, 0) * 4 or 1)
        )
        if exclude_chat_id is not None:
            stmt = stmt.where(ChatMemoryFinding.chat_id != exclude_chat_id)

        rows = (await self._session.execute(stmt)).scalars().all()
        logger.info(
            "CROSS_SESSION_STORE_READ user_id=%s agent_id=%s rows=%d",
            user_id,
            agent_id,
            len(rows),
        )
        return [
            CrossSessionFact(
                content=row.content,
                confidence=_coerce_confidence(row.confidence),
                updated_at=row.created_at,
                origin=row.origin,
            )
            for row in rows
            if (row.content or "").strip()
        ]


async def inject_cross_session_memory(
    *,
    store: CrossSessionMemoryStore | None,
    user_id: str | None,
    agent_id: str | None,
    exclude_chat_id: UUID | None = None,
    min_confidence: ConfidenceLabel = DEFAULT_MIN_CONFIDENCE,
    max_facts: int = DEFAULT_MAX_FACTS,
    timeout_seconds: float = DEFAULT_READ_TIMEOUT_SECONDS,
) -> dict[str, str] | None:
    """Read remembered facts and return a spotlighted ``role=user`` message.

    Fail-soft is the #1 invariant: ANY error or a read slower than
    ``timeout_seconds`` degrades to ``None`` (no memory injected) — it NEVER
    raises and NEVER blocks the research request. Returns ``None`` when:

    * no store / no ``user_id`` (nothing to read),
    * the store yields nothing above the confidence threshold,
    * the read times out, or
    * the read raises (logged, swallowed).

    The returned message (when non-``None``) is
    ``{"role": "user", "content": <spotlight-wrapped facts>}`` — untrusted
    remembered content marked as DATA via the OWASP spotlighting sentinels.
    """
    if store is None or not user_id:
        return None

    try:
        facts = await asyncio.wait_for(
            store.read_facts(
                user_id=user_id,
                agent_id=agent_id,
                exclude_chat_id=exclude_chat_id,
                min_confidence=min_confidence,
                limit=max_facts,
            ),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        logger.warning(
            "CROSS_SESSION_MEMORY_READ_TIMEOUT user_id=%s agent_id=%s timeout=%.1fs "
            "— degrading to no-memory",
            user_id,
            agent_id,
            timeout_seconds,
        )
        return None
    except Exception:
        logger.exception(
            "CROSS_SESSION_MEMORY_READ_FAILED user_id=%s agent_id=%s "
            "— degrading to no-memory",
            user_id,
            agent_id,
        )
        return None

    message = build_cross_session_memory_message(
        facts,
        min_confidence=min_confidence,
        max_facts=max_facts,
    )
    if message is None:
        logger.info(
            "CROSS_SESSION_MEMORY_EMPTY user_id=%s agent_id=%s facts_read=%d",
            user_id,
            agent_id,
            len(facts),
        )
        return None
    logger.info(
        "CROSS_SESSION_MEMORY_INJECTED user_id=%s agent_id=%s facts_read=%d chars=%d",
        user_id,
        agent_id,
        len(facts),
        len(message["content"]),
    )
    return message


_CONFIDENCE_ORDER: tuple[ConfidenceLabel, ...] = ("high", "medium", "low")


def _confidence_at_or_above(min_confidence: ConfidenceLabel) -> list[str]:
    """Return the confidence labels at or above ``min_confidence`` (for the DB
    ``IN`` filter). Unknown floor ⇒ all labels (loose, the policy re-filters)."""
    if min_confidence not in _CONFIDENCE_ORDER:
        return list(_CONFIDENCE_ORDER)
    cutoff = _CONFIDENCE_ORDER.index(min_confidence)
    return list(_CONFIDENCE_ORDER[: cutoff + 1])


def _coerce_confidence(value: str | None) -> ConfidenceLabel:
    """Map a stored confidence string to a valid label; unknown ⇒ ``low``."""
    label = (value or "").lower()
    if label in _CONFIDENCE_ORDER:
        return label  # type: ignore[return-value]
    return "low"


__all__ = [
    "ChatMemoryFindingsStore",
    "CrossSessionMemoryStore",
    "DEFAULT_READ_TIMEOUT_SECONDS",
    "inject_cross_session_memory",
]
