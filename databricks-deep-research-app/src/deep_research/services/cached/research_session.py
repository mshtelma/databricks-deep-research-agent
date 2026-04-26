"""Cache-backed `IResearchSessionService`.

Storage model: `ChatState.research_sessions[]`. Legacy `ResearchSession`
row columns that aren't on `ResearchSessionState` (`research_depth`,
`error_message`, `reasoning_steps`, `user_id`) are stored inside the
`execution_state` dict on the session entry. Chat-level `user_id` is
authoritative via `ChatMeta.user_id`.

Return shape: `SimpleNamespace` mirroring the legacy `ResearchSession`
ORM surface.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from uuid import UUID

from deep_research.services._cached_base import _CachedServiceBase
from deep_research.services._protocols import IResearchSessionService
from deep_research.storage.documents import ResearchSessionState

if TYPE_CHECKING:
    from deep_research.storage.factory import StorageStack


logger = logging.getLogger(__name__)

_STATUS_IN_PROGRESS = "in_progress"
_STATUS_COMPLETED = "completed"
_STATUS_FAILED = "failed"
_STATUS_CANCELLED = "cancelled"


def _session_to_namespace(
    s: ResearchSessionState,
    *,
    chat_id: UUID,
    user_id: str | None = None,
) -> SimpleNamespace:
    exec_state = dict(s.execution_state or {})
    return SimpleNamespace(
        id=s.id,
        chat_id=chat_id,
        user_id=exec_state.get("user_id") or user_id,
        message_id=s.message_id,
        query=exec_state.get("query", ""),
        research_depth=exec_state.get("research_depth", "auto"),
        query_classification=s.query_classification,
        plan=s.plan,
        observations=s.observations,
        reasoning_steps=exec_state.get("reasoning_steps", []),
        execution_state=exec_state,
        verification_data=s.verification_data,
        current_step=s.current_step,
        status=s.status,
        error_message=exec_state.get("error_message"),
        started_at=s.started_at,
        created_at=s.started_at,
        completed_at=s.completed_at,
    )


class CachedResearchSessionService(_CachedServiceBase, IResearchSessionService):
    """`IResearchSessionService` via `ChatState.research_sessions[]`."""

    def __init__(self, stack: "StorageStack") -> None:
        super().__init__(stack)

    async def create(
        self,
        message_id: UUID,
        query: str,
        research_depth: str = "auto",
        query_classification: dict[str, Any] | None = None,
        *,
        chat_id: UUID,
    ) -> SimpleNamespace:
        new = ResearchSessionState(
            message_id=message_id,
            status=_STATUS_IN_PROGRESS,
            query_classification=query_classification or {},
            execution_state={
                "query": query,
                "research_depth": research_depth,
                "reasoning_steps": [],
            },
        )

        def _apply(doc: Any) -> None:
            doc.state.upsert_research_session(new)

        await self._mutate_chat(chat_id, _apply, dirty="state")
        logger.info("RESEARCH_SESSION_CREATED chat_id=%s id=%s", chat_id, new.id)
        return _session_to_namespace(new, chat_id=chat_id)

    async def get(
        self, session_id: UUID, *, chat_id: UUID
    ) -> SimpleNamespace | None:
        doc = await self._read_chat(chat_id)
        for s in doc.state.research_sessions:
            if s.id == session_id:
                return _session_to_namespace(s, chat_id=chat_id, user_id=doc.meta.user_id)
        return None

    async def get_by_message(
        self, message_id: UUID, *, chat_id: UUID
    ) -> SimpleNamespace | None:
        doc = await self._read_chat(chat_id)
        for s in doc.state.research_sessions:
            if s.message_id == message_id:
                return _session_to_namespace(s, chat_id=chat_id, user_id=doc.meta.user_id)
        return None

    async def get_active_session_by_chat(
        self,
        chat_id: UUID,
        user_id: str,
    ) -> SimpleNamespace | None:
        doc = await self._read_chat(chat_id)
        if doc.meta.user_id != user_id:
            return None
        active = [
            s for s in doc.state.research_sessions if s.status == _STATUS_IN_PROGRESS
        ]
        if not active:
            return None
        active.sort(key=lambda s: s.started_at, reverse=True)
        return _session_to_namespace(active[0], chat_id=chat_id, user_id=user_id)

    async def update_plan(
        self,
        session_id: UUID,
        plan: dict[str, Any],
        *,
        chat_id: UUID,
    ) -> SimpleNamespace | None:
        return await self._mutate_session(
            session_id,
            chat_id,
            lambda s: setattr(s, "plan", plan),
        )

    async def add_observation(
        self,
        session_id: UUID,
        observation: str,
        step_index: int,
        *,
        chat_id: UUID,
    ) -> SimpleNamespace | None:
        def _apply(s: ResearchSessionState) -> None:
            obs = list(s.observations.get("items", []))
            obs.append({
                "step_index": step_index,
                "observation": observation,
                "timestamp": datetime.now(UTC).isoformat(),
            })
            s.observations = {**s.observations, "items": obs}

        return await self._mutate_session(session_id, chat_id, _apply)

    async def add_reasoning_step(
        self,
        session_id: UUID,
        step_type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        *,
        chat_id: UUID,
    ) -> SimpleNamespace | None:
        def _apply(s: ResearchSessionState) -> None:
            exec_state = dict(s.execution_state or {})
            steps = list(exec_state.get("reasoning_steps", []))
            steps.append({
                "type": step_type,
                "content": content,
                "metadata": metadata or {},
                "timestamp": datetime.now(UTC).isoformat(),
            })
            exec_state["reasoning_steps"] = steps
            s.execution_state = exec_state

        return await self._mutate_session(session_id, chat_id, _apply)

    async def complete(
        self,
        session_id: UUID,
        error_message: str | None = None,
        *,
        chat_id: UUID,
    ) -> SimpleNamespace | None:
        now = datetime.now(UTC)

        def _apply(s: ResearchSessionState) -> None:
            if error_message:
                s.status = _STATUS_FAILED
                exec_state = dict(s.execution_state or {})
                exec_state["error_message"] = error_message
                s.execution_state = exec_state
            else:
                s.status = _STATUS_COMPLETED
            s.completed_at = now

        result = await self._mutate_session(session_id, chat_id, _apply)
        if result is not None:
            logger.info("RESEARCH_SESSION_COMPLETED chat_id=%s id=%s", chat_id, session_id)
        return result

    async def cancel(
        self,
        session_id: UUID,
        *,
        chat_id: UUID,
    ) -> SimpleNamespace | None:
        now = datetime.now(UTC)

        def _apply(s: ResearchSessionState) -> None:
            if s.status != _STATUS_IN_PROGRESS:
                return
            s.status = _STATUS_CANCELLED
            s.completed_at = now

        result = await self._mutate_session(session_id, chat_id, _apply)
        if result is not None:
            logger.info("RESEARCH_SESSION_CANCELLED chat_id=%s id=%s", chat_id, session_id)
        return result

    async def update_classification(
        self,
        session_id: UUID,
        classification: dict[str, Any],
        *,
        chat_id: UUID,
    ) -> SimpleNamespace | None:
        return await self._mutate_session(
            session_id,
            chat_id,
            lambda s: setattr(s, "query_classification", classification),
        )

    # --- Internal ------------------------------------------------------

    async def _mutate_session(
        self,
        session_id: UUID,
        chat_id: UUID,
        fn: Any,
    ) -> SimpleNamespace | None:
        result: dict[str, ResearchSessionState | None] = {"s": None}

        def _apply(doc: Any) -> None:
            for s in doc.state.research_sessions:
                if s.id == session_id:
                    fn(s)
                    result["s"] = s
                    return

        await self._mutate_chat(chat_id, _apply, dirty="state")
        s = result["s"]
        if s is None:
            return None
        doc = await self._read_chat(chat_id)
        return _session_to_namespace(s, chat_id=chat_id, user_id=doc.meta.user_id)
