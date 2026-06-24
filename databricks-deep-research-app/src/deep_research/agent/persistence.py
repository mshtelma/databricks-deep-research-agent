"""Persistence layer for research data.

This module handles persisting research artifacts (sources, verification_data JSONB)
to the database after synthesis completes.

Key Design: Two-Phase Persistence for Crash Resilience

Phase 1 - At START (persist_research_session_start_independent):
- Creates chat (upsert), user message, agent message placeholder (NULL content)
- Creates research_session with IN_PROGRESS status
- This satisfies FK constraint for research_events during streaming

Phase 2 - At END (persist_research_session_complete_update_independent):
- Updates agent message with final_report content
- Updates research_session to COMPLETED status with verification_data JSONB
- Persists sources in atomic transaction

JSONB Migration (Migration 011):
- Claims, evidence, citations are now stored in verification_data JSONB column
- This reduces write queries from 45-200+ to 3-5 queries
- Sources are still persisted separately for deduplication

Draft Chat Support:
- For draft chats, the chat row is created atomically with messages
- Uses INSERT ON CONFLICT to handle race conditions safely
"""

import logging
import re
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from deep_research.agent.chat_title import derive_chat_title_from_query
from deep_research.agent.state import ClaimInfo, ResearchState
from deep_research.models.chat import Chat, ChatStatus
from deep_research.models.message import Message, MessageRole
from deep_research.models.research_session import ResearchSession, ResearchStatus
from deep_research.models.source import Source

logger = logging.getLogger(__name__)

# Citation key pattern - matches [Key] or [Key-N] format
# This pattern is shared across the codebase for consistency:
# - claim_generator.py (generation)
# - pipeline.py (extraction)
# - react_synthesizer.py (validation)
# - frontend citationPlugin.ts (rendering)
CITATION_KEY_PATTERN = re.compile(r"\[([A-Za-z][A-Za-z0-9-]*(?:-\d+)?)\]")


def _ensure_citation_key(claim: ClaimInfo) -> dict[str, Any]:
    """Build claim dict, extracting citation_key from claim_text if missing.

    This function is a defensive fallback for the grey references bug. If a claim
    doesn't have a citation_key set (e.g., due to a bug in an upstream stage), we
    extract it from the claim_text using the citation key pattern.

    Args:
        claim: ClaimInfo object to convert.

    Returns:
        Dict representation of the claim with citation_key ensured.
    """
    claim_dict = claim.to_dict()

    # If citation_key is already present, return as-is
    if claim_dict.get("citation_key"):
        return claim_dict

    # Fallback: extract citation_key from claim_text
    matches = CITATION_KEY_PATTERN.findall(claim.claim_text)
    if matches:
        claim_dict["citation_key"] = matches[0]
        if len(matches) > 1:
            claim_dict["citation_keys"] = matches
        logger.warning(
            "CITATION_KEY_EXTRACTED_FALLBACK claim_text=%s extracted_key=%s",
            claim.claim_text[:50],
            matches[0],
        )

    return claim_dict


async def persist_research_data(
    state: ResearchState,
    message_id: UUID,  # noqa: ARG001
    research_session_id: UUID,
    db: AsyncSession,
    chat_id: UUID | None = None,
) -> dict[str, int]:
    """Persist sources + JSONB verification_data to database after synthesis.

    This function persists:
    1. Sources (with ON CONFLICT upsert for deduplication)
    2. verification_data JSONB (claims + summary in single UPDATE)

    Args:
        state: Research state containing sources, evidence, and claims.
        _message_id: ID of the agent message (unused - kept for backwards compat).
        research_session_id: ID of the research session for sources.
        db: Database session.
        chat_id: Optional chat ID for chat-level source pool queries.

    Returns:
        Dict with counts of persisted entities.
    """
    counts: dict[str, int] = {"sources": 0, "claims": len(state.claims)}

    # Step 1: Persist sources and build URL -> Source mapping
    url_to_source = await _persist_sources(state, research_session_id, db, chat_id)
    counts["sources"] = len(url_to_source)

    # Step 1b: Compute and persist is_cited from claims evidence
    cited_urls: set[str] = set()
    for claim in state.claims:
        if claim.evidence and claim.evidence.source_url:
            cited_urls.add(claim.evidence.source_url)

    if cited_urls and url_to_source:
        cited_source_ids = [
            s.id for url, s in url_to_source.items() if url in cited_urls
        ]
        if cited_source_ids:
            await db.execute(
                update(Source)
                .where(Source.id.in_(cited_source_ids))
                .values(is_cited=True)
            )
            logger.info(
                "PERSIST_IS_CITED_UPDATED cited_sources=%d total_cited_urls=%d",
                len(cited_source_ids),
                len(cited_urls),
            )

    # Step 2: Build verification_data JSONB
    verification_data = _build_verification_data(state, url_to_source)

    # Step 3: Update research_session with verification_data
    if verification_data:
        await db.execute(
            update(ResearchSession)
            .where(ResearchSession.id == research_session_id)
            .values(verification_data=verification_data)
        )

    logger.info(
        "PERSIST_RESEARCH_DATA_COMPLETE sources=%d claims=%d verification_data=%s",
        counts["sources"],
        counts["claims"],
        "present" if verification_data else "empty",
    )

    return counts


async def _persist_sources(
    state: ResearchState,
    research_session_id: UUID,
    db: AsyncSession,
    chat_id: UUID | None,
) -> dict[str, Source]:
    """Persist sources and build URL -> Source mapping.

    Uses atomic upsert (ON CONFLICT) to handle race conditions and duplicates.

    Args:
        state: Research state containing sources.
        research_session_id: ID of the research session.
        db: Database session.
        chat_id: Optional chat ID for source pool queries.

    Returns:
        Dict mapping source URL to Source model.
    """
    url_to_source: dict[str, Source] = {}
    sources_failed = 0

    for source_info in state.sources:
        try:
            async with db.begin_nested():
                # Atomic upsert using ON CONFLICT - no race condition possible
                stmt = pg_insert(Source).values(
                    research_session_id=research_session_id,
                    chat_id=chat_id,
                    url=source_info.url,
                    title=source_info.title,
                    snippet=source_info.snippet,
                    content=source_info.content,
                    relevance_score=source_info.relevance_score,
                    source_type=source_info.source_type or "web",
                ).on_conflict_do_update(
                    index_elements=["chat_id", "url"],
                    set_={
                        "content": source_info.content,
                        "research_session_id": research_session_id,
                        "source_type": source_info.source_type or "web",
                        "fetched_at": func.now(),
                    },
                ).returning(Source.id, Source.title, Source.url)

                result = await db.execute(stmt)
                row = result.one()

                # Create a minimal Source object for mapping
                source = Source(
                    id=row.id,
                    title=row.title or source_info.title,
                    url=row.url,
                )
                url_to_source[source_info.url] = source
        except Exception as e:
            sources_failed += 1
            logger.warning(
                "PERSIST_SOURCE_FAILED source_url=%s error=%s",
                source_info.url[:100],
                str(e)[:200],
            )

    logger.info(
        "PERSIST_SOURCES_COMPLETE sources_persisted=%d sources_failed=%d",
        len(url_to_source),
        sources_failed,
    )

    return url_to_source


def _build_verification_data(
    state: ResearchState,
    url_to_source: dict[str, Source],
) -> dict[str, Any] | None:
    """Build JSONB from ResearchState using existing to_dict() methods.

    Uses _ensure_citation_key() to handle the grey references bug by extracting
    citation_key from claim_text if missing.

    Args:
        state: Research state containing claims and verification summary.
        url_to_source: Mapping from source URL to Source model.

    Returns:
        JSONB-compatible dict or None if no claims.
    """
    if not state.claims and not state.verification_summary:
        return None

    claims_data = []
    for claim in state.claims:
        # Use _ensure_citation_key for fallback extraction (grey references fix)
        claim_dict = _ensure_citation_key(claim)

        # Embed source_title for display (avoid source lookup on read)
        if claim.evidence and claim.evidence.source_url in url_to_source:
            source = url_to_source[claim.evidence.source_url]
            if claim_dict.get("evidence"):
                claim_dict["evidence"]["source_title"] = source.title

        claims_data.append(claim_dict)

    summary_data = (
        state.verification_summary.to_dict()
        if state.verification_summary
        else _compute_summary_from_claims(state.claims)
    )

    return {"claims": claims_data, "summary": summary_data}


def _compute_summary_from_claims(claims: list[ClaimInfo]) -> dict[str, Any]:
    """Compute summary from claims (fallback if not provided).

    Args:
        claims: List of ClaimInfo objects.

    Returns:
        Summary dict compatible with VerificationSummaryInfo.to_dict().
    """
    if not claims:
        return {
            "total_claims": 0,
            "supported_count": 0,
            "partial_count": 0,
            "unsupported_count": 0,
            "contradicted_count": 0,
            "abstained_count": 0,
            "unsupported_rate": 0.0,
            "contradicted_rate": 0.0,
            "warning": False,
            "citation_corrections": 0,
        }

    verdicts = [c.verification_verdict for c in claims if c.verification_verdict]
    abstained = sum(1 for c in claims if c.abstained)
    verified = len(verdicts)

    supported = verdicts.count("supported")
    partial = verdicts.count("partial")
    unsupported = verdicts.count("unsupported")
    contradicted = verdicts.count("contradicted")

    unsupported_rate = unsupported / verified if verified > 0 else 0.0
    contradicted_rate = contradicted / verified if verified > 0 else 0.0

    return {
        "total_claims": len(claims),
        "supported_count": supported,
        "partial_count": partial,
        "unsupported_count": unsupported,
        "contradicted_count": contradicted,
        "abstained_count": abstained,
        "unsupported_rate": unsupported_rate,
        "contradicted_rate": contradicted_rate,
        "warning": unsupported_rate > 0.20 or contradicted_rate > 0.05,
        "citation_corrections": 0,
    }


async def persist_complete_research(
    db: AsyncSession,
    chat_id: UUID,
    user_id: str,
    user_query: str,
    message_id: UUID,
    research_session_id: UUID,
    research_depth: str,
    state: ResearchState,
) -> dict[str, int]:
    """Persist ALL research data in a single atomic transaction.

    This is the main entry point for deferred persistence. It creates all
    database records only AFTER synthesis completes successfully.

    For draft chats (chat doesn't exist yet), this function also creates
    the chat row atomically with the messages.

    Order matters for FK constraints:
    0. Chat (create if not exists - for draft chat support)
    1. User message (requires chat_id)
    2. Agent message (requires chat_id, uses pre-generated UUID)
    3. Research session (requires message_id FK)
    4. Sources (requires research_session_id FK)
    5. verification_data JSONB (claims + summary in single UPDATE)
    6. Update chat.updated_at

    Args:
        db: Database session.
        chat_id: ID of the chat to add messages to.
        user_id: User ID who owns the chat (required for draft chat creation).
        user_query: The user's original query.
        message_id: Pre-generated UUID for the agent message.
        research_session_id: Pre-generated UUID for the research session.
        research_depth: Research depth setting (auto/light/medium/extended).
        state: Research state containing final_report, sources, claims, etc.

    Returns:
        Dict with counts of persisted entities.
    """
    counts: dict[str, int] = {
        "chat_created": 0,
        "user_message": 0,
        "agent_message": 0,
        "research_session": 0,
        "sources": 0,
        "claims": 0,
    }

    # Step 0: Create or ensure chat exists (race-safe upsert for draft support)
    # This handles the case where the chat doesn't exist yet (draft chat)
    # or already exists (follow-up message in existing chat)
    chat_title = derive_chat_title_from_query(user_query)

    stmt = pg_insert(Chat).values(
        id=chat_id,
        user_id=user_id,
        title=chat_title,
        status=ChatStatus.ACTIVE,
    ).on_conflict_do_update(
        index_elements=["id"],
        # On conflict, preserve a non-empty existing title (rename protection)
        # and fill in the derived title ONLY when the current title is empty.
        # Also always bump updated_at.
        set_={
            "updated_at": datetime.now(UTC),
            "title": func.coalesce(func.nullif(Chat.title, ""), chat_title),
        },
        # Safety: Only update if the chat belongs to the same user
        # This prevents a malicious user from "stealing" another user's chat ID
        where=(Chat.user_id == user_id),
    )

    await db.execute(stmt)
    # Upsert always succeeds - either insert or update
    counts["chat_created"] = 1
    logger.debug(f"Ensured chat {chat_id} exists for user {user_id}")

    # Step 1: Create user message
    user_message = Message(
        chat_id=chat_id,
        role=MessageRole.USER,
        content=user_query,
    )
    db.add(user_message)
    counts["user_message"] = 1
    logger.debug(f"Created user message for chat {chat_id}")

    # Step 2: Create agent message with pre-generated UUID and synthesized content
    agent_message = Message(
        id=message_id,  # Use pre-generated UUID
        chat_id=chat_id,
        role=MessageRole.AGENT,
        content=state.final_report,
    )
    db.add(agent_message)
    counts["agent_message"] = 1
    logger.debug(f"Created agent message {message_id} with {len(state.final_report)} chars")

    # Step 3: Create research session with pre-generated UUID
    # Map ResearchState attributes to ResearchSession columns:
    # - state.all_observations -> observations (JSONB)
    # - state.current_plan.to_dict() -> plan (JSONB)
    # - state.reflection_history -> reasoning_steps (JSONB array)
    observations_data = [{"observation": obs} for obs in state.all_observations] if state.all_observations else None
    research_session = ResearchSession(
        id=research_session_id,  # Use pre-generated UUID
        message_id=message_id,
        query=user_query,
        research_depth=research_depth,
        query_mode=state.query_mode,  # Tiered query modes: simple, web_search, deep_research
        status=ResearchStatus.COMPLETED,
        observations=observations_data,
        plan=state.current_plan.to_dict() if state.current_plan else None,
        reasoning_steps=[r.to_dict() for r in state.reflection_history] if state.reflection_history else [],
        current_step_index=state.current_step_index,
        plan_iterations=state.plan_iterations,
        completed_at=datetime.now(UTC),
        promotion_trace=getattr(state, "promotion_trace", None),
    )
    db.add(research_session)
    counts["research_session"] = 1
    logger.debug(f"Created research session {research_session_id}")

    # Flush to ensure FKs are satisfied before persisting sources
    await db.flush()

    # Step 4-5: Persist sources + verification_data JSONB
    # Pass chat_id for chat-level source pool queries in follow-ups
    research_counts = await persist_research_data(
        state=state,
        message_id=message_id,
        research_session_id=research_session_id,
        db=db,
        chat_id=chat_id,
    )

    # Merge counts
    counts.update(research_counts)

    # Step 6: Update chat.updated_at to reflect new activity
    # This ensures the chat appears at the top of the sidebar list
    await db.execute(
        update(Chat)
        .where(Chat.id == chat_id)
        .values(updated_at=datetime.now(UTC))
    )

    logger.info(
        f"PERSIST_COMPLETE_RESEARCH chat={chat_id} user={user_id} "
        f"message={message_id} session={research_session_id} "
        f"chat_created={counts['chat_created']} sources={counts['sources']} "
        f"claims={counts['claims']}"
    )

    return counts


async def persist_simple_message(
    db: AsyncSession,
    chat_id: UUID,
    user_id: str,
    user_query: str,
    message_id: UUID,
    content: str,
) -> dict[str, int]:
    """Persist chat + messages for simple mode (no research session).

    Simple mode doesn't perform web search or create research sessions.
    This function only persists the chat and messages.

    Args:
        db: Database session.
        chat_id: ID of the chat.
        user_id: User ID who owns the chat.
        user_query: The user's original query.
        message_id: Pre-generated UUID for the agent message.
        content: The agent's response content.

    Returns:
        Dict with counts of persisted entities.
    """
    counts = {
        "chat_created": 0,
        "user_message": 0,
        "agent_message": 0,
    }

    # Step 0: Create or ensure chat exists (race-safe upsert)
    chat_title = user_query[:47] + "..." if len(user_query) > 50 else user_query

    stmt = pg_insert(Chat).values(
        id=chat_id,
        user_id=user_id,
        title=chat_title,
        status=ChatStatus.ACTIVE,
    ).on_conflict_do_update(
        index_elements=["id"],
        set_={"updated_at": datetime.now(UTC)},
        where=(Chat.user_id == user_id),
    )

    await db.execute(stmt)
    counts["chat_created"] = 1

    # Step 1: Create user message
    user_message = Message(
        chat_id=chat_id,
        role=MessageRole.USER,
        content=user_query,
    )
    db.add(user_message)
    counts["user_message"] = 1

    # Step 2: Create agent message with pre-generated UUID
    agent_message = Message(
        id=message_id,
        chat_id=chat_id,
        role=MessageRole.AGENT,
        content=content,
    )
    db.add(agent_message)
    counts["agent_message"] = 1

    # Flush to persist
    await db.flush()

    # Update chat.updated_at
    await db.execute(
        update(Chat)
        .where(Chat.id == chat_id)
        .values(updated_at=datetime.now(UTC))
    )

    logger.info(
        f"PERSIST_SIMPLE_MESSAGE chat={chat_id} user={user_id} "
        f"message={message_id} content_len={len(content)}"
    )

    return counts


# ---------------------------------------------------------------------------
# Cached-path helpers (no session_maker; use StorageStack document model)
# ---------------------------------------------------------------------------


async def _persist_session_start_cached(
    *,
    stack: Any,
    chat_id: UUID,
    user_id: str,
    user_query: str,
    user_message_id: UUID,
    agent_message_id: UUID,
    research_session_id: UUID,
    research_depth: str,
    query_mode: str,
) -> None:
    """Cached-path implementation of persist_research_session_start_independent.

    Writes chat (upsert), user message, agent message placeholder, and
    research session into the StorageStack document model.  No SQLAlchemy
    session is opened.
    """
    from deep_research.storage.documents import (
        Message as DocMessage,
    )
    from deep_research.storage.documents import (
        ResearchSessionState,
    )

    now = datetime.now(UTC)
    chat_title = user_query[:47] + "..." if len(user_query) > 50 else user_query

    user_msg = DocMessage(
        id=user_message_id,
        role="user",
        content=user_query,
        ts=now,
    )
    agent_msg = DocMessage(
        id=agent_message_id,
        role="agent",
        content="",
        ts=now,
        metadata={"placeholder": True},
    )
    session_state = ResearchSessionState(
        id=research_session_id,
        message_id=agent_message_id,
        status="in_progress",
        started_at=now,
        execution_state={
            "query": user_query,
            "research_depth": research_depth,
            "query_mode": query_mode,
            "user_id": user_id,
            "reasoning_steps": [],
        },
    )

    def _apply(doc: Any) -> None:
        # Set title on first write; don't overwrite an existing title.
        if not doc.meta.title:
            doc.meta.title = chat_title
            doc.state.chat.title = chat_title
        doc.state.add_message(user_msg)
        doc.state.add_message(agent_msg)
        doc.state.upsert_research_session(session_state)
        doc.meta.updated_at = now

    # Hydrate into cache first (creates doc if it doesn't exist yet).
    await stack.cache.get(chat_id, user_id=user_id, title_hint=chat_title)
    await stack.cache.mutate(chat_id, _apply, dirty="both")
    # Flush immediately so downstream readers see the session row.
    await stack.queue.flush_chat_now(chat_id)

    logger.info(
        "RESEARCH_SESSION_STARTED_CACHED session=%s chat=%s user=%s",
        str(research_session_id)[:8], str(chat_id)[:8], user_id,
    )


async def _persist_session_complete_cached(
    *,
    stack: Any,
    chat_id: UUID,
    research_session_id: UUID,
    agent_message_id: UUID,
    state: ResearchState,
) -> dict[str, int]:
    """Cached-path implementation of persist_research_session_complete_update_independent."""
    from deep_research.storage.documents import Source as DocSource

    now = datetime.now(UTC)
    counts: dict[str, int] = {"sources": 0, "claims": len(state.claims)}

    # Build source document objects.
    sources_to_add: list[DocSource] = []
    url_seen: set[str] = set()
    for src_info in state.sources:
        if src_info.url in url_seen:
            continue
        url_seen.add(src_info.url)
        sources_to_add.append(
            DocSource(
                url=src_info.url,
                title=(src_info.title or "")[:500],
                source_type=src_info.source_type or "web",
                metadata={
                    "research_session_id": str(research_session_id),
                    "snippet": src_info.snippet,
                    "content": (src_info.content or "")[:50_000] if src_info.content else None,
                    "relevance_score": src_info.relevance_score,
                    "fetched_at": now.isoformat(),
                },
            )
        )
    counts["sources"] = len(sources_to_add)

    # Build verification_data.
    verification_data = _build_verification_data(
        state,
        {src.url: type("_S", (), {"id": src.id, "title": src.title, "url": src.url})()
         for src in sources_to_add},
    )

    final_report = state.final_report or ""

    def _apply(doc: Any) -> None:
        # 1. Update agent message content.
        for m in doc.state.messages:
            if m.id == agent_message_id:
                m.content = final_report
                meta = dict(m.metadata or {})
                meta.pop("placeholder", None)
                meta["updated_at"] = now.isoformat()
                m.metadata = meta
                break

        # 2. Update research session status.
        observations_data = (
            [{"observation": obs} for obs in state.all_observations]
            if state.all_observations
            else []
        )
        for rs in doc.state.research_sessions:
            if rs.id == research_session_id:
                rs.status = "completed"
                rs.completed_at = now
                rs.observations = {"items": observations_data}
                # ResearchSessionState.plan/verification_data are typed as
                # non-nullable dicts — write {} (not None) so re-reads via
                # ChatState.model_validate don't reject our own writes.
                rs.plan = state.current_plan.to_dict() if state.current_plan else {}
                rs.verification_data = verification_data or {}
                exec_state = dict(rs.execution_state or {})
                exec_state["reasoning_steps"] = (
                    [r.to_dict() for r in state.reflection_history]
                    if state.reflection_history
                    else []
                )
                rs.execution_state = exec_state
                rs.promotion_trace = getattr(state, "promotion_trace", None)
                break

        # Derive chat title only when empty — preserves manual renames.
        if not (doc.meta.title or "").strip():
            derived_title = derive_chat_title_from_query(state.query)
            if derived_title:
                doc.meta.title = derived_title
                doc.state.chat.title = derived_title
                logger.info(
                    "CHAT_TITLE_DERIVED_CACHED chat=%s title=%r",
                    str(chat_id)[:8], derived_title[:60],
                )

        # 3. Add sources (dedup by URL via add_source).
        for src in sources_to_add:
            doc.state.add_source(src)

        doc.meta.updated_at = now

    # Doc must already be hydrated (started at session-start); get() is a no-op
    # on a warm cache entry but ensures hydration if somehow missed.
    await stack.cache.get(chat_id)
    await stack.cache.mutate(chat_id, _apply, dirty="both")
    await stack.queue.flush_chat_now(chat_id)

    # Cross-turn memory (flag-gated, fail-open): consolidate this turn's VERIFIED
    # claims into durable per-chat findings so later turns can read/cite them.
    # A consolidation error must never fail the already-persisted research turn.
    from deep_research.core.config import get_settings as _get_settings

    if _get_settings().chat_memory_unified and verification_data:
        try:
            from deep_research.agent.turn_consolidation import (
                consolidate_turn_knowledge,
            )
            from deep_research.services.cached.chat_memory import (
                CachedChatMemoryService,
            )

            _mem = CachedChatMemoryService(stack)
            await _mem.hydrate(chat_id)
            _written = await consolidate_turn_knowledge(
                _mem, chat_id, research_session_id, verification_data
            )
            logger.info(
                "CHAT_MEMORY_CONSOLIDATED_CACHED chat=%s session=%s findings=%d",
                str(chat_id)[:8], str(research_session_id)[:8], _written,
            )
        except Exception:
            logger.exception(
                "CHAT_MEMORY_CONSOLIDATION_FAILED chat=%s session=%s",
                str(chat_id)[:8], str(research_session_id)[:8],
            )

    logger.info(
        "RESEARCH_SESSION_COMPLETED_CACHED session=%s sources=%d claims=%d",
        str(research_session_id)[:8], counts["sources"], counts["claims"],
    )
    return counts


async def _persist_session_failed_cached(
    *,
    stack: Any,
    chat_id: UUID,
    research_session_id: UUID,
    agent_message_id: UUID,
    error_message: str,
) -> None:
    """Cached-path implementation of persist_research_session_failed_independent."""
    now = datetime.now(UTC)
    error_content = f"Research failed: {error_message}"

    def _apply(doc: Any) -> None:
        for m in doc.state.messages:
            if m.id == agent_message_id:
                m.content = error_content
                meta = dict(m.metadata or {})
                meta.pop("placeholder", None)
                meta["updated_at"] = now.isoformat()
                m.metadata = meta
                break
        for rs in doc.state.research_sessions:
            if rs.id == research_session_id:
                rs.status = "failed"
                rs.completed_at = now
                exec_state = dict(rs.execution_state or {})
                exec_state["error_message"] = error_message
                rs.execution_state = exec_state
                break
        doc.meta.updated_at = now

    await stack.cache.get(chat_id)
    await stack.cache.mutate(chat_id, _apply, dirty="both")
    await stack.queue.flush_chat_now(chat_id)
    logger.warning(
        "RESEARCH_SESSION_FAILED_CACHED session=%s error=%s",
        str(research_session_id)[:8], error_message[:100],
    )


async def _persist_session_cancelled_cached(
    *,
    stack: Any,
    chat_id: UUID,
    research_session_id: UUID,
) -> None:
    """Cached-path implementation of persist_research_session_cancelled_independent."""
    now = datetime.now(UTC)

    def _apply(doc: Any) -> None:
        for rs in doc.state.research_sessions:
            if rs.id == research_session_id:
                rs.status = "cancelled"
                rs.completed_at = now
                break
        doc.meta.updated_at = now

    await stack.cache.get(chat_id)
    await stack.cache.mutate(chat_id, _apply, dirty="both")
    await stack.queue.flush_chat_now(chat_id)
    logger.info(
        "RESEARCH_SESSION_CANCELLED_CACHED session=%s",
        str(research_session_id)[:8],
    )


async def _persist_session_completed_cached(
    *,
    stack: Any,
    chat_id: UUID,
    research_session_id: UUID,
) -> None:
    """Cached-path status-only COMPLETED transition.

    Counterpart to ``_persist_session_complete_cached`` (which also writes the
    final report + sources + claims). Simple-mode runs persist their answer
    message separately, so here we only flip the session status so the
    JobManager's end-of-stream check sees a terminal state instead of force-
    failing a successfully-answered session.
    """
    now = datetime.now(UTC)

    def _apply(doc: Any) -> None:
        for rs in doc.state.research_sessions:
            if rs.id == research_session_id and rs.status == "in_progress":
                rs.status = "completed"
                rs.completed_at = now
                break
        doc.meta.updated_at = now

    await stack.cache.get(chat_id)
    await stack.cache.mutate(chat_id, _apply, dirty="both")
    await stack.queue.flush_chat_now(chat_id)
    logger.info(
        "RESEARCH_SESSION_COMPLETED_SIMPLE_CACHED session=%s",
        str(research_session_id)[:8],
    )


async def _persist_simple_message_cached(
    *,
    stack: Any,
    chat_id: UUID,
    user_id: str,
    user_query: str,
    message_id: UUID,
    content: str,
) -> dict[str, int]:
    """Cached-path implementation of persist_simple_message_independent."""
    from deep_research.storage.documents import Message as DocMessage

    now = datetime.now(UTC)
    chat_title = user_query[:47] + "..." if len(user_query) > 50 else user_query

    user_msg = DocMessage(role="user", content=user_query, ts=now)
    agent_msg = DocMessage(id=message_id, role="agent", content=content, ts=now)

    def _apply(doc: Any) -> None:
        if not doc.meta.title:
            doc.meta.title = chat_title
            doc.state.chat.title = chat_title
        doc.state.add_message(user_msg)
        doc.state.add_message(agent_msg)
        doc.meta.updated_at = now

    await stack.cache.get(chat_id, user_id=user_id, title_hint=chat_title)
    await stack.cache.mutate(chat_id, _apply, dirty="both")
    await stack.queue.flush_chat_now(chat_id)
    logger.info(
        "SIMPLE_MESSAGE_PERSISTED_CACHED chat=%s message=%s",
        str(chat_id)[:8], str(message_id)[:8],
    )
    return {"chat_created": 1, "user_message": 1, "agent_message": 1}


async def _persist_simple_message_update_cached(
    *,
    stack: Any,
    chat_id: UUID,
    message_id: UUID,
    content: str,
) -> dict[str, int]:
    """Cached-path implementation of persist_simple_message_update_independent."""
    now = datetime.now(UTC)

    def _apply(doc: Any) -> None:
        for m in doc.state.messages:
            if m.id == message_id:
                m.content = content
                meta = dict(m.metadata or {})
                meta.pop("placeholder", None)
                meta["updated_at"] = now.isoformat()
                m.metadata = meta
                break
        doc.meta.updated_at = now

    await stack.cache.get(chat_id)
    await stack.cache.mutate(chat_id, _apply, dirty="state")
    await stack.queue.flush_chat_now(chat_id)
    logger.info(
        "SIMPLE_MESSAGE_UPDATED_CACHED chat=%s message=%s",
        str(chat_id)[:8], str(message_id)[:8],
    )
    return {"messages_updated": 1}


async def _persist_complete_research_cached(
    *,
    stack: Any,
    chat_id: UUID,
    user_id: str,
    user_query: str,
    message_id: UUID,
    research_session_id: UUID,
    research_depth: str,
    state: ResearchState,
) -> dict[str, int]:
    """Cached-path implementation of persist_complete_research_independent.

    Single-phase: creates chat, messages, research session, and sources
    in one document mutation.
    """
    from deep_research.storage.documents import (
        Message as DocMessage,
    )
    from deep_research.storage.documents import (
        ResearchSessionState,
    )
    from deep_research.storage.documents import (
        Source as DocSource,
    )

    now = datetime.now(UTC)
    chat_title = user_query[:47] + "..." if len(user_query) > 50 else user_query
    counts: dict[str, int] = {
        "chat_created": 1,
        "user_message": 1,
        "agent_message": 1,
        "research_session": 1,
        "sources": 0,
        "claims": len(state.claims),
    }

    # Build source objects.
    sources_to_add: list[DocSource] = []
    url_seen: set[str] = set()
    for src_info in state.sources:
        if src_info.url in url_seen:
            continue
        url_seen.add(src_info.url)
        sources_to_add.append(
            DocSource(
                url=src_info.url,
                title=(src_info.title or "")[:500],
                source_type=src_info.source_type or "web",
                metadata={
                    "research_session_id": str(research_session_id),
                    "snippet": src_info.snippet,
                    "content": (src_info.content or "")[:50_000] if src_info.content else None,
                    "relevance_score": src_info.relevance_score,
                    "fetched_at": now.isoformat(),
                },
            )
        )
    counts["sources"] = len(sources_to_add)

    verification_data = _build_verification_data(
        state,
        {src.url: type("_S", (), {"id": src.id, "title": src.title, "url": src.url})()
         for src in sources_to_add},
    )

    observations_data = (
        [{"observation": obs} for obs in state.all_observations]
        if state.all_observations
        else []
    )

    user_msg = DocMessage(role="user", content=user_query, ts=now)
    agent_msg = DocMessage(
        id=message_id,
        role="agent",
        content=state.final_report or "",
        ts=now,
    )
    session_state = ResearchSessionState(
        id=research_session_id,
        message_id=message_id,
        status="completed",
        started_at=now,
        completed_at=now,
        observations={"items": observations_data},
        plan=state.current_plan.to_dict() if state.current_plan else None,
        verification_data=verification_data,
        execution_state={
            "query": user_query,
            "research_depth": research_depth,
            "user_id": user_id,
            "reasoning_steps": (
                [r.to_dict() for r in state.reflection_history]
                if state.reflection_history
                else []
            ),
        },
        promotion_trace=getattr(state, "promotion_trace", None),
    )

    def _apply(doc: Any) -> None:
        if not doc.meta.title:
            doc.meta.title = chat_title
            doc.state.chat.title = chat_title
        doc.state.add_message(user_msg)
        doc.state.add_message(agent_msg)
        doc.state.upsert_research_session(session_state)
        for src in sources_to_add:
            doc.state.add_source(src)
        doc.meta.updated_at = now

    await stack.cache.get(chat_id, user_id=user_id, title_hint=chat_title)
    await stack.cache.mutate(chat_id, _apply, dirty="both")
    await stack.queue.flush_chat_now(chat_id)
    logger.info(
        "COMPLETE_RESEARCH_PERSISTED_CACHED chat=%s session=%s sources=%d claims=%d",
        str(chat_id)[:8], str(research_session_id)[:8],
        counts["sources"], counts["claims"],
    )
    return counts


async def persist_research_session_start_independent(
    chat_id: UUID,
    user_id: str,
    user_query: str,
    user_message_id: UUID,
    agent_message_id: UUID,
    research_session_id: UUID,
    research_depth: str,
    query_mode: str,
    *,
    # Job management columns (optional for backwards compatibility)
    worker_id: str | None = None,
    last_heartbeat: datetime | None = None,
    verify_sources: bool = True,
    storage_stack: Any | None = None,
) -> None:
    """Create minimal research session at START with IN_PROGRESS status.

    This function creates all the database records needed at the START of
    research, before any events are generated:
    1. Chat (upsert to handle drafts)
    2. User message
    3. Agent message placeholder (NULL content - will be updated at end)
    4. Research session with IN_PROGRESS status

    Uses independent DB session to survive request cancellation.
    When storage_stack is provided and storage_service_impl == "cached",
    routes through the StorageStack cache layer (no session_maker needed).

    Args:
        chat_id: ID of the chat.
        user_id: User ID who owns the chat.
        user_query: The user's original query.
        user_message_id: Pre-generated UUID for the user message.
        agent_message_id: Pre-generated UUID for the agent message.
        research_session_id: Pre-generated UUID for the research session.
        research_depth: Research depth setting (auto/light/medium/extended).
        query_mode: Query mode (simple/web_search/deep_research).
        worker_id: Worker instance ID for job tracking (optional).
        last_heartbeat: Initial heartbeat timestamp (optional).
        verify_sources: Whether citation verification is enabled (default True).
        storage_stack: Optional StorageStack; when provided and impl==cached,
            skips session_maker entirely.
    """
    from deep_research.core.config import get_settings

    settings = get_settings()
    if settings.storage_service_impl == "cached" and storage_stack is not None:
        await _persist_session_start_cached(
            stack=storage_stack,
            chat_id=chat_id,
            user_id=user_id,
            user_query=user_query,
            user_message_id=user_message_id,
            agent_message_id=agent_message_id,
            research_session_id=research_session_id,
            research_depth=research_depth,
            query_mode=query_mode,
        )
        return

    from deep_research.db.session import get_session_maker

    session_maker = get_session_maker()
    async with session_maker() as db:
        try:
            # 1. Upsert chat (handles draft → active transition)
            chat_title = user_query[:47] + "..." if len(user_query) > 50 else user_query
            stmt = pg_insert(Chat).values(
                id=chat_id,
                user_id=user_id,
                title=chat_title,
                status=ChatStatus.ACTIVE,
            ).on_conflict_do_update(
                index_elements=["id"],
                set_={"updated_at": datetime.now(UTC)},
                where=(Chat.user_id == user_id),
            )
            await db.execute(stmt)

            # 2. Create user message
            user_msg = Message(
                id=user_message_id,
                chat_id=chat_id,
                role=MessageRole.USER,
                content=user_query,
            )
            db.add(user_msg)

            # 3. Create agent message placeholder (NULL content - will be updated at end)
            agent_msg = Message(
                id=agent_message_id,
                chat_id=chat_id,
                role=MessageRole.AGENT,
                content=None,  # Will be updated with final_report at end
            )
            db.add(agent_msg)

            # 4. Create research session with IN_PROGRESS status
            research_session = ResearchSession(
                id=research_session_id,
                message_id=agent_message_id,
                query=user_query,
                research_depth=research_depth,
                query_mode=query_mode,
                status=ResearchStatus.IN_PROGRESS,  # KEY: IN_PROGRESS at start!
                # Job management columns (migration 009)
                user_id=user_id,
                chat_id=chat_id,
                worker_id=worker_id,
                last_heartbeat=last_heartbeat,
                verify_sources=verify_sources,
            )
            db.add(research_session)

            await db.commit()
            logger.info(
                f"RESEARCH_SESSION_STARTED session={research_session_id} "
                f"chat={chat_id} user={user_id}"
            )
        except Exception:
            await db.rollback()
            raise


async def persist_research_session_complete_update_independent(
    chat_id: UUID,
    research_session_id: UUID,
    agent_message_id: UUID,
    state: ResearchState,
    *,
    storage_stack: Any | None = None,
) -> dict[str, int]:
    """Update existing research session to COMPLETED and persist all data.

    Called after synthesis succeeds. Updates:
    - Agent message content (final_report)
    - Research session status (COMPLETED) and final state
    - Sources and verification_data JSONB (claims + summary)

    Uses independent DB session to survive request cancellation.
    When storage_stack is provided and impl==cached, skips session_maker.

    Args:
        chat_id: ID of the chat (for source pool queries).
        research_session_id: UUID of the existing research session.
        agent_message_id: UUID of the agent message to update.
        state: Research state containing final_report, sources, claims, etc.
        storage_stack: Optional StorageStack; when provided and impl==cached,
            skips session_maker entirely.

    Returns:
        Dict with counts of persisted entities.
    """
    from deep_research.core.config import get_settings

    settings = get_settings()
    if settings.storage_service_impl == "cached" and storage_stack is not None:
        return await _persist_session_complete_cached(
            stack=storage_stack,
            chat_id=chat_id,
            research_session_id=research_session_id,
            agent_message_id=agent_message_id,
            state=state,
        )

    from deep_research.db.session import get_session_maker

    session_maker = get_session_maker()
    async with session_maker() as db:
        try:
            # 1. Update agent message with final content
            await db.execute(
                update(Message)
                .where(Message.id == agent_message_id)
                .values(content=state.final_report, updated_at=datetime.now(UTC))
            )

            # 2. Update research session with final state (except verification_data)
            observations_data = (
                [{"observation": obs} for obs in state.all_observations]
                if state.all_observations
                else None
            )
            await db.execute(
                update(ResearchSession)
                .where(ResearchSession.id == research_session_id)
                .values(
                    status=ResearchStatus.COMPLETED,
                    observations=observations_data,
                    plan=state.current_plan.to_dict() if state.current_plan else None,
                    reasoning_steps=(
                        [r.to_dict() for r in state.reflection_history]
                        if state.reflection_history
                        else []
                    ),
                    current_step_index=state.current_step_index,
                    plan_iterations=state.plan_iterations,
                    completed_at=datetime.now(UTC),
                    promotion_trace=getattr(state, "promotion_trace", None),
                )
            )

            # Flush to ensure session update is visible
            await db.flush()

            # 3. Persist sources + verification_data JSONB
            counts = await persist_research_data(
                state=state,
                message_id=agent_message_id,
                research_session_id=research_session_id,
                db=db,
                chat_id=chat_id,
            )

            # 4. Update chat.updated_at and derive title when empty.
            # SELECT ... FOR UPDATE closes the race window with concurrent
            # PATCH rename inside the same transaction on Postgres.
            # (SQLite silently ignores FOR UPDATE; covered by contract suite.)
            chat_row = await db.execute(
                select(Chat.title).where(Chat.id == chat_id).with_for_update()
            )
            current_title = (chat_row.scalar_one_or_none() or "").strip()
            update_values: dict[str, Any] = {"updated_at": datetime.now(UTC)}
            if not current_title:
                derived_title = derive_chat_title_from_query(state.query)
                if derived_title:
                    update_values["title"] = derived_title
                    logger.info(
                        "CHAT_TITLE_DERIVED_ORM chat=%s title=%r",
                        str(chat_id)[:8], derived_title[:60],
                    )
            await db.execute(
                update(Chat).where(Chat.id == chat_id).values(**update_values)
            )

            await db.commit()
            logger.info(
                f"RESEARCH_SESSION_COMPLETED session={research_session_id} "
                f"sources={counts.get('sources', 0)} claims={counts.get('claims', 0)}"
            )
            return counts
        except Exception:
            await db.rollback()
            raise


async def persist_research_session_failed_independent(
    research_session_id: UUID,
    agent_message_id: UUID,
    error_message: str,
    *,
    storage_stack: Any | None = None,
    chat_id: UUID | None = None,
) -> None:
    """Update research session to FAILED status.

    Called when research fails after session was created at START.
    When storage_stack is provided and impl==cached, skips session_maker.

    Args:
        research_session_id: UUID of the research session to update.
        agent_message_id: UUID of the agent message to update.
        error_message: Error message to store in agent message content.
        storage_stack: Optional StorageStack; when provided and impl==cached,
            skips session_maker entirely.
        chat_id: Required when storage_stack is provided (chat-scoped storage).
    """
    from deep_research.core.config import get_settings

    settings = get_settings()
    if settings.storage_service_impl == "cached" and storage_stack is not None and chat_id is not None:
        await _persist_session_failed_cached(
            stack=storage_stack,
            chat_id=chat_id,
            research_session_id=research_session_id,
            agent_message_id=agent_message_id,
            error_message=error_message,
        )
        return

    from deep_research.db.session import get_session_maker

    session_maker = get_session_maker()
    async with session_maker() as db:
        try:
            # Update research session status to FAILED
            await db.execute(
                update(ResearchSession)
                .where(ResearchSession.id == research_session_id)
                .values(
                    status=ResearchStatus.FAILED,
                    completed_at=datetime.now(UTC),
                )
            )

            # Update agent message with error indication
            await db.execute(
                update(Message)
                .where(Message.id == agent_message_id)
                .values(
                    content=f"Research failed: {error_message}",
                    updated_at=datetime.now(UTC),
                )
            )

            await db.commit()
            logger.warning(
                f"RESEARCH_SESSION_FAILED session={research_session_id} "
                f"error={error_message[:100]}"
            )
        except Exception:
            await db.rollback()
            raise


async def persist_research_session_cancelled_independent(
    research_session_id: UUID,
    *,
    storage_stack: Any | None = None,
    chat_id: UUID | None = None,
) -> None:
    """Update research session to CANCELLED status.

    Called when a running job is cancelled (user action or task cancellation).
    When storage_stack is provided and impl==cached, skips session_maker.

    Args:
        research_session_id: UUID of the research session to cancel.
        storage_stack: Optional StorageStack; when provided and impl==cached,
            skips session_maker entirely.
        chat_id: Required when storage_stack is provided (chat-scoped storage).
    """
    from deep_research.core.config import get_settings

    settings = get_settings()
    if (
        settings.storage_service_impl == "cached"
        and storage_stack is not None
        and chat_id is not None
    ):
        await _persist_session_cancelled_cached(
            stack=storage_stack,
            chat_id=chat_id,
            research_session_id=research_session_id,
        )
        return

    from deep_research.db.session import get_session_maker

    session_maker = get_session_maker()
    async with session_maker() as db:
        try:
            await db.execute(
                update(ResearchSession)
                .where(ResearchSession.id == research_session_id)
                .where(ResearchSession.status == ResearchStatus.IN_PROGRESS)
                .values(
                    status=ResearchStatus.CANCELLED,
                    completed_at=datetime.now(UTC),
                )
            )
            await db.commit()
            logger.info(
                f"RESEARCH_SESSION_CANCELLED session={research_session_id}"
            )
        except Exception:
            await db.rollback()
            raise


async def persist_research_session_completed_independent(
    research_session_id: UUID,
    *,
    storage_stack: Any | None = None,
    chat_id: UUID | None = None,
) -> None:
    """Status-only COMPLETED transition for a research session.

    Used by simple-query mode: the answer message is persisted by the
    ``persist_simple_message_*`` helpers, but the pre-created session is left
    ``in_progress`` — which the JobManager's end-of-stream check treats as a
    stuck/failed run and force-fails (``persistence_transition_missing``). This
    flips the session to COMPLETED without touching the (already-saved) message.
    Only transitions a session still ``IN_PROGRESS`` so it never clobbers a
    session already marked failed/cancelled. When storage_stack is provided and
    impl==cached, skips session_maker.

    Args:
        research_session_id: UUID of the research session to complete.
        storage_stack: Optional StorageStack; when provided and impl==cached,
            skips session_maker entirely.
        chat_id: Required when storage_stack is provided (chat-scoped storage).
    """
    from deep_research.core.config import get_settings

    settings = get_settings()
    if (
        settings.storage_service_impl == "cached"
        and storage_stack is not None
        and chat_id is not None
    ):
        await _persist_session_completed_cached(
            stack=storage_stack,
            chat_id=chat_id,
            research_session_id=research_session_id,
        )
        return

    from deep_research.db.session import get_session_maker

    session_maker = get_session_maker()
    async with session_maker() as db:
        try:
            await db.execute(
                update(ResearchSession)
                .where(ResearchSession.id == research_session_id)
                .where(ResearchSession.status == ResearchStatus.IN_PROGRESS)
                .values(
                    status=ResearchStatus.COMPLETED,
                    completed_at=datetime.now(UTC),
                )
            )
            await db.commit()
            logger.info(
                f"RESEARCH_SESSION_COMPLETED_SIMPLE session={research_session_id}"
            )
        except Exception:
            await db.rollback()
            raise


async def persist_simple_message_independent(
    chat_id: UUID,
    user_id: str,
    user_query: str,
    message_id: UUID,
    content: str,
    *,
    storage_stack: Any | None = None,
) -> dict[str, int]:
    """Persist simple message with an independent database session.

    Use this for shielded operations where the request-scoped session
    may be cleaned up before persistence completes.
    When storage_stack is provided and impl==cached, skips session_maker.

    Args:
        chat_id: ID of the chat.
        user_id: User ID who owns the chat.
        user_query: The user's original query.
        message_id: Pre-generated UUID for the agent message.
        content: The agent's response content.
        storage_stack: Optional StorageStack; when provided and impl==cached,
            skips session_maker entirely.

    Returns:
        Dict with counts of persisted entities.
    """
    from deep_research.core.config import get_settings

    settings = get_settings()
    if settings.storage_service_impl == "cached" and storage_stack is not None:
        return await _persist_simple_message_cached(
            stack=storage_stack,
            chat_id=chat_id,
            user_id=user_id,
            user_query=user_query,
            message_id=message_id,
            content=content,
        )

    from deep_research.db.session import get_session_maker

    session_maker = get_session_maker()
    async with session_maker() as db:
        try:
            counts = await persist_simple_message(
                db=db,
                chat_id=chat_id,
                user_id=user_id,
                user_query=user_query,
                message_id=message_id,
                content=content,
            )
            await db.commit()
            return counts
        except Exception:
            await db.rollback()
            raise


async def persist_simple_message_update_independent(
    message_id: UUID,
    content: str,
    *,
    storage_stack: Any | None = None,
    chat_id: UUID | None = None,
) -> dict[str, int]:
    """Update pre-created simple mode message with content.

    Used when JobManager pre-created the session and agent message placeholder.
    In this case, the chat, user message, and agent message already exist,
    we just need to update the agent message content.
    When storage_stack is provided and impl==cached, skips session_maker.

    Args:
        message_id: UUID of the agent message to update.
        content: The agent's response content.
        storage_stack: Optional StorageStack; when provided and impl==cached,
            skips session_maker entirely.
        chat_id: Required when storage_stack is provided (chat-scoped storage).

    Returns:
        Dict with counts of updated entities.
    """
    from deep_research.core.config import get_settings

    settings = get_settings()
    if settings.storage_service_impl == "cached" and storage_stack is not None and chat_id is not None:
        return await _persist_simple_message_update_cached(
            stack=storage_stack,
            chat_id=chat_id,
            message_id=message_id,
            content=content,
        )

    from deep_research.db.session import get_session_maker

    session_maker = get_session_maker()
    async with session_maker() as db:
        try:
            # Update existing agent message with content
            await db.execute(
                update(Message)
                .where(Message.id == message_id)
                .values(content=content, updated_at=datetime.now(UTC))
            )
            await db.commit()

            logger.info(
                f"PERSIST_SIMPLE_MESSAGE_UPDATE message={message_id} "
                f"content_len={len(content)}"
            )

            return {"messages_updated": 1}
        except Exception:
            await db.rollback()
            raise


async def persist_complete_research_independent(
    chat_id: UUID,
    user_id: str,
    user_query: str,
    message_id: UUID,
    research_session_id: UUID,
    research_depth: str,
    state: ResearchState,
    *,
    storage_stack: Any | None = None,
) -> dict[str, int]:
    """Persist research data with an independent database session.

    Use this for shielded operations where the request-scoped session
    may be cleaned up before persistence completes (e.g., when client
    disconnects during streaming).

    This function creates its own database session that is independent
    of the FastAPI request lifecycle, ensuring persistence completes
    even if the HTTP request is cancelled.
    When storage_stack is provided and impl==cached, skips session_maker.

    Args:
        chat_id: ID of the chat to add messages to.
        user_id: User ID who owns the chat.
        user_query: The user's original query.
        message_id: Pre-generated UUID for the agent message.
        research_session_id: Pre-generated UUID for the research session.
        research_depth: Research depth setting (auto/light/medium/extended).
        state: Research state containing final_report, sources, claims, etc.
        storage_stack: Optional StorageStack; when provided and impl==cached,
            skips session_maker entirely.

    Returns:
        Dict with counts of persisted entities.
    """
    from deep_research.core.config import get_settings

    settings = get_settings()
    if settings.storage_service_impl == "cached" and storage_stack is not None:
        return await _persist_complete_research_cached(
            stack=storage_stack,
            chat_id=chat_id,
            user_id=user_id,
            user_query=user_query,
            message_id=message_id,
            research_session_id=research_session_id,
            research_depth=research_depth,
            state=state,
        )

    from deep_research.db.session import get_session_maker

    session_maker = get_session_maker()
    async with session_maker() as db:
        try:
            counts = await persist_complete_research(
                db=db,
                chat_id=chat_id,
                user_id=user_id,
                user_query=user_query,
                message_id=message_id,
                research_session_id=research_session_id,
                research_depth=research_depth,
                state=state,
            )
            await db.commit()
            return counts
        except Exception:
            await db.rollback()
            raise
