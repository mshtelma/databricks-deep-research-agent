"""Stream a grounded "chat about prior research" answer — no new research run.

Reuses the existing simple-query handler (``agent.nodes.coordinator.handle_simple_query``)
and the chat-scoped source pool (``ChatSourcePoolService``) so a follow-up is
answered from the report + sources already gathered in the chat, with a
``search_sources`` tool for deep retrieval.

This module only *streams content events*; persistence and the terminal
``PersistenceCompletedEvent`` are owned by the caller (``framework_orchestrator``),
which holds the persistence helpers — keeping this module free of a back-import.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING
from uuid import UUID

from deep_research.agent.nodes.coordinator import handle_simple_query
from deep_research.agent.state import ResearchState, SourceInfo
from deep_research.core.logging_utils import get_logger
from deep_research.schemas.streaming import (
    AgentCompletedEvent,
    AgentStartedEvent,
    StreamEvent,
    SynthesisProgressEvent,
    SynthesisStartedEvent,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from deep_research.services.chat_source_pool_service import ChatSourcePoolService
    from deep_research.services.llm.client import LLMClient
    from deep_research.services.llm.embedder import Embedder
    from deep_research.storage.factory import StorageStack

logger = get_logger(__name__)


def _as_uuid(value: str | UUID) -> UUID:
    return value if isinstance(value, UUID) else UUID(str(value))


async def _load_chat_sources(
    db: AsyncSession,
    chat_id: UUID,
    embedder: Embedder | None,
    storage_stack: StorageStack | None,
) -> tuple[ChatSourcePoolService | None, list[SourceInfo]]:
    """Build the chat source pool (indexed) and the SourceInfo summary list."""
    from deep_research.services.chat_source_pool_service import ChatSourcePoolService

    pool = ChatSourcePoolService(db, embedder=embedder, storage_stack=storage_stack)
    db_sources = await pool.get_all_sources(chat_id)
    await pool.build_search_index(chat_id)
    sources = [
        SourceInfo(
            url=src.url,
            title=src.title,
            snippet=src.snippet,
            content=getattr(src, "content", None),
        )
        for src in db_sources
    ]
    return pool, sources


async def stream_chat_about_results(
    *,
    query: str,
    conversation_history: list[dict[str, str]] | None,
    chat_id: str | UUID,
    db: AsyncSession | None,
    llm: LLMClient,
    prior_findings_summary: str = "",
    embedder: Embedder | None = None,
    storage_stack: StorageStack | None = None,
) -> AsyncGenerator[StreamEvent, None]:
    """Stream a grounded answer over the chat's prior research.

    Emits the same synthesis SSE events the simple-query short-circuit uses, so
    the frontend renders it as a normal assistant answer. Opens an independent
    DB session when ``db`` is None (cached-stream path) so source search still
    works; degrades to conversation-only context if no DB is reachable.
    """
    chat_source_pool: ChatSourcePoolService | None = None
    sources: list[SourceInfo] = []
    cid = _as_uuid(chat_id)

    owns_session = False
    session = db
    try:
        if session is None:
            from deep_research.db.session import get_session_maker

            session = get_session_maker()()
            await session.__aenter__()
            owns_session = True

        if session is not None:
            try:
                chat_source_pool, sources = await _load_chat_sources(
                    session, cid, embedder, storage_stack
                )
            except Exception as exc:  # noqa: BLE001 — source search is best-effort
                logger.warning(
                    "FOLLOWUP_POOL_INIT_FAILED", chat_id=str(cid), error=str(exc)[:200]
                )
                chat_source_pool, sources = None, []

        observations = [prior_findings_summary] if prior_findings_summary else []
        state = ResearchState(
            query=query,
            conversation_history=list(conversation_history or []),
            sources=sources,
            all_observations=observations,
        )

        logger.info(
            "FOLLOWUP_CHAT_ANSWER_START",
            chat_id=str(cid),
            sources=len(sources),
            has_findings=bool(prior_findings_summary),
            has_pool=chat_source_pool is not None,
        )

        yield SynthesisStartedEvent(total_observations=len(observations), total_sources=len(sources))
        yield AgentStartedEvent(agent="synthesizer", model_tier="simple")
        # The concrete ChatSourcePoolService satisfies the coordinator's curated
        # IChatSourcePoolService param at runtime (build_search_index + search).
        async for chunk in handle_simple_query(state, llm, chat_source_pool):  # type: ignore[arg-type]
            if chunk:
                yield SynthesisProgressEvent(content_chunk=chunk)
        yield AgentCompletedEvent(agent="synthesizer", duration_ms=0)
    finally:
        if owns_session and session is not None:
            await session.__aexit__(None, None, None)
