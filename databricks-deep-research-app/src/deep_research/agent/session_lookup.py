"""Storage-aware ResearchSession lookup.

In cached storage mode the authoritative session state lives inside the
``ChatDocument`` JSONB blob keyed by ``chat_id``. Legacy mode stores each
session as a row in ``research_sessions``. Callers pass both ``chat_id``
and ``session_id`` (surfaced via path params on the control-plane URLs),
so the cached-mode lookup is an O(1) walk inside an already-hydrated
document — no auxiliary index table needed.

This module provides a single helper that works in both modes. It returns
a read-only ``SessionControlView`` — not a SQLAlchemy row — so callers
cannot accidentally mutate it and expect ``db.commit()`` to propagate.
Writes must go through the cached-aware persistence helpers in
``persistence.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from deep_research.core.config import Settings
from deep_research.models.research_session import ResearchSession, ResearchStatus


@dataclass(frozen=True)
class SessionControlView:
    """Read-only projection of the fields the control plane needs.

    The control plane = SSE validation, SSE polling status check, cancel
    verification, worker post-stream lifecycle decisions, plus the
    ``GET /jobs/{chat_id}/{session_id}`` status endpoint. It does not need
    the chat-scoped payload (full plan/observations/verification_data) —
    those live in the ChatDocument and are read by the streaming/export
    endpoints through different code paths.

    ``query``, ``query_mode``, ``current_step``, and ``total_steps`` are
    projected here so the status endpoint can build a full ``JobResponse``
    without a second ChatDocument walk.
    """

    id: UUID
    user_id: str
    chat_id: UUID
    status: ResearchStatus
    message_id: UUID | None
    started_at: datetime
    completed_at: datetime | None
    error_message: str | None
    query: str
    query_mode: str
    current_step: int | None
    total_steps: int | None


def _coerce_status(raw: Any) -> ResearchStatus:
    """Accept both ResearchStatus and the bare string stored in ChatDocument."""
    if isinstance(raw, ResearchStatus):
        return raw
    try:
        return ResearchStatus(raw)
    except ValueError:
        # Unknown status — surface as IN_PROGRESS so callers treat it as live.
        # A stale/unknown value must never silently become a terminal status.
        return ResearchStatus.IN_PROGRESS


async def load_session_control_view(
    chat_id: UUID,
    session_id: UUID,
    user_id: str,
    *,
    settings: Settings,
    storage_stack: Any | None,
    db: AsyncSession,
) -> SessionControlView | None:
    """Resolve a session to the control-plane projection.

    Callers supply ``chat_id`` from the URL path (trusted only after an
    ownership check against ``user_id``). Returns ``None`` when the chat or
    session does not exist, or when the caller does not own the chat, so
    the control-plane endpoint can return 404.
    """
    if settings.storage_service_impl == "cached" and storage_stack is not None:
        try:
            doc = await storage_stack.cache.get(chat_id, user_id=user_id)
        except Exception:
            # ChatDocument deleted or backend error — treat as not found so
            # the caller returns 404 instead of crashing the SSE generator.
            return None

        # Ownership check: chat_id in the URL is untrusted until we verify
        # the authenticated user owns it.
        if str(doc.meta.user_id) != user_id:
            return None

        rs = doc.state.get_research_session(session_id)
        if rs is None:
            return None
        exec_state = rs.execution_state or {}
        plan = rs.plan or {}
        plan_steps = plan.get("steps") if isinstance(plan, dict) else None
        total_steps = len(plan_steps) if isinstance(plan_steps, list) else None
        return SessionControlView(
            id=rs.id,
            user_id=user_id,
            chat_id=chat_id,
            status=_coerce_status(rs.status),
            message_id=rs.message_id,
            started_at=rs.started_at,
            completed_at=rs.completed_at,
            error_message=exec_state.get("error_message"),
            query=str(exec_state.get("query") or ""),
            query_mode=str(exec_state.get("query_mode") or "deep_research"),
            current_step=rs.current_step,
            total_steps=total_steps,
        )

    # Legacy path: row in research_sessions. chat_id/user_id from URL are
    # still enforced via a match against the row to keep the ownership
    # semantics consistent between modes.
    sess = await db.get(ResearchSession, session_id)
    if sess is None:
        return None
    if sess.chat_id != chat_id or sess.user_id != user_id:
        return None
    plan = sess.plan or {}
    plan_steps = plan.get("steps") if isinstance(plan, dict) else None
    total_steps = len(plan_steps) if isinstance(plan_steps, list) else None
    return SessionControlView(
        id=sess.id,
        user_id=sess.user_id,
        chat_id=sess.chat_id,
        status=_coerce_status(sess.status),
        message_id=sess.message_id,
        started_at=sess.started_at,
        completed_at=sess.completed_at,
        error_message=sess.error_message,
        query=sess.query or "",
        query_mode=sess.query_mode or "deep_research",
        current_step=sess.current_step_index,
        total_steps=total_steps,
    )
