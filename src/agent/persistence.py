"""Persistence layer for research data.

This module handles persisting research artifacts (sources, claims, evidence, citations)
to the database after synthesis completes.

Key Design: Deferred Database Materialization
- UUIDs are generated in memory before streaming starts
- NO database writes occur until synthesis completes successfully
- All data is persisted in a single atomic transaction
- Benefits: No orphaned records, no cleanup needed on failure

Draft Chat Support:
- For draft chats, the chat row is created atomically with messages
- Uses INSERT ON CONFLICT to handle race conditions safely
"""

import logging
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.agent.state import ResearchState
from src.models.chat import Chat, ChatStatus
from src.models.source import Source
from src.models.message import Message, MessageRole
from src.models.research_session import ResearchSession, ResearchStatus
from src.services.citation_service import CitationService
from src.services.claim_service import ClaimService
from src.services.evidence_span_service import EvidenceSpanService
# SourceService no longer used - using direct ON CONFLICT upsert for sources
from src.services.verification_summary_service import VerificationSummaryService

logger = logging.getLogger(__name__)


async def persist_research_data(
    state: ResearchState,
    message_id: UUID,
    research_session_id: UUID,
    db: AsyncSession,
    chat_id: UUID | None = None,
) -> dict[str, int]:
    """Persist all research data to database after synthesis completes.

    This function persists data in the correct order to satisfy foreign key constraints:
    1. Sources (no FK deps, optional chat_id for source pool)
    2. Evidence spans (requires source_id)
    3. Claims (requires message_id)
    4. Citations (requires claim_id + evidence_span_id)
    5. Verification summary (computed from claims in DB)

    Args:
        state: Research state containing sources, evidence, and claims.
        message_id: ID of the agent message to associate claims with.
        research_session_id: ID of the research session for sources.
        db: Database session.
        chat_id: Optional chat ID for chat-level source pool queries.

    Returns:
        Dict with counts of persisted entities.
    """
    evidence_service = EvidenceSpanService(db)
    claim_service = ClaimService(db)
    citation_service = CitationService(db)
    summary_service = VerificationSummaryService(db)

    counts = {
        "sources": 0,
        "evidence_spans": 0,
        "claims": 0,
        "citations": 0,
    }

    # Step 1: Persist sources and build URL -> source_id mapping
    # Uses atomic upsert (ON CONFLICT) to handle race conditions and duplicates
    url_to_source_id: dict[str, UUID] = {}

    for source_info in state.sources:
        try:
            # Atomic upsert using ON CONFLICT - no race condition possible
            # Use (chat_id, url) columns to deduplicate sources at chat level
            # NOTE: Must use index_elements (not constraint) because uq_sources_chat_url
            # was created as a UNIQUE INDEX, not a table constraint (migration 006)
            stmt = pg_insert(Source).values(
                research_session_id=research_session_id,
                chat_id=chat_id,  # For chat-level source pool queries
                url=source_info.url,
                title=source_info.title,
                snippet=source_info.snippet,
                content=source_info.content,
                relevance_score=source_info.relevance_score,
            ).on_conflict_do_update(
                index_elements=["chat_id", "url"],
                set_={
                    "content": source_info.content,
                    "research_session_id": research_session_id,
                    "fetched_at": func.now(),
                },
            ).returning(Source.id)

            result = await db.execute(stmt)
            source_id = result.scalar_one()
            url_to_source_id[source_info.url] = source_id
            counts["sources"] += 1
        except Exception as e:
            logger.warning(f"Failed to persist source {source_info.url}: {e}")

    logger.info(
        f"Persisted {counts['sources']} sources, "
        f"mapped {len(url_to_source_id)} URLs to source IDs"
    )

    # Step 2: Persist evidence spans and build lookup key -> span_id mapping
    evidence_key_to_id: dict[str, UUID] = {}

    for evidence in state.evidence_pool:
        evidence_source_id = url_to_source_id.get(evidence.source_url)
        if not evidence_source_id:
            logger.warning(
                f"Source not found for evidence, skipping: {evidence.source_url}"
            )
            continue

        try:
            span = await evidence_service.create(
                source_id=evidence_source_id,
                quote_text=evidence.quote_text,
                start_offset=evidence.start_offset,
                end_offset=evidence.end_offset,
                section_heading=evidence.section_heading,
                relevance_score=evidence.relevance_score,
                has_numeric_content=evidence.has_numeric_content,
            )
            # Key for lookup: source_url + truncated quote_text hash
            # Using first 100 chars of quote to avoid very long keys
            key = _make_evidence_key(evidence.source_url, evidence.quote_text)
            evidence_key_to_id[key] = span.id
            counts["evidence_spans"] += 1
        except Exception as e:
            logger.warning(f"Failed to persist evidence span: {e}")

    logger.info(f"Persisted {counts['evidence_spans']} evidence spans")

    # Step 3: Persist claims and their citations
    for claim_info in state.claims:
        try:
            claim = await claim_service.create(
                message_id=message_id,
                claim_text=claim_info.claim_text,
                claim_type=claim_info.claim_type,
                position_start=claim_info.position_start,
                position_end=claim_info.position_end,
                confidence_level=claim_info.confidence_level,
                verification_verdict=claim_info.verification_verdict,
                verification_reasoning=claim_info.verification_reasoning,
                abstained=claim_info.abstained,
                citation_key=claim_info.citation_key,
                citation_keys=claim_info.citation_keys,
            )
            counts["claims"] += 1

            logger.debug(
                "CLAIM_PERSISTED claim_id=%s citation_key=%s verdict=%s text=%s",
                str(claim.id),
                claim.citation_key,
                claim.verification_verdict,
                claim.claim_text[:60],
            )

            # Create citation if claim has evidence
            if claim_info.evidence:
                key = _make_evidence_key(
                    claim_info.evidence.source_url, claim_info.evidence.quote_text
                )
                evidence_span_id = evidence_key_to_id.get(key)

                if evidence_span_id:
                    await citation_service.create(
                        claim_id=claim.id,
                        evidence_span_id=evidence_span_id,
                        confidence_score=claim_info.evidence.relevance_score,
                        is_primary=True,
                    )
                    counts["citations"] += 1
                else:
                    logger.warning(
                        f"Evidence span not found for claim citation: {key[:100]}..."
                    )

        except Exception as e:
            logger.warning(f"Failed to persist claim: {e}")

    logger.info(
        f"Persisted {counts['claims']} claims with {counts['citations']} citations"
    )

    # Step 4: Compute and persist verification summary
    if state.verification_summary or counts["claims"] > 0:
        try:
            await summary_service.compute_summary(message_id)
            logger.info(f"Computed verification summary for message {message_id}")
        except Exception as e:
            logger.warning(f"Failed to compute verification summary: {e}")

    return counts


def _make_evidence_key(source_url: str, quote_text: str) -> str:
    """Create a lookup key for evidence span.

    Uses source URL + hash of quote text to create a unique key
    that can be used to look up the persisted evidence span ID.

    Args:
        source_url: URL of the source.
        quote_text: Quote text from the evidence.

    Returns:
        Unique key string.
    """
    # Use hash to handle arbitrarily long quotes
    quote_hash = hash(quote_text)
    return f"{source_url}:{quote_hash}"


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
    5. Evidence spans (requires source_id FK)
    6. Claims (requires message_id FK)
    7. Citations (requires claim_id + evidence_span_id FKs)
    8. Verification summary (computed from claims)
    9. Update chat.updated_at

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
    counts = {
        "chat_created": 0,
        "user_message": 0,
        "agent_message": 0,
        "research_session": 0,
        "sources": 0,
        "evidence_spans": 0,
        "claims": 0,
        "citations": 0,
    }

    # Step 0: Create or ensure chat exists (race-safe upsert for draft support)
    # This handles the case where the chat doesn't exist yet (draft chat)
    # or already exists (follow-up message in existing chat)
    chat_title = user_query[:47] + "..." if len(user_query) > 50 else user_query

    stmt = pg_insert(Chat).values(
        id=chat_id,
        user_id=user_id,
        title=chat_title,
        status=ChatStatus.ACTIVE,
    ).on_conflict_do_update(
        index_elements=["id"],
        set_={"updated_at": datetime.now(UTC)},
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
    )
    db.add(research_session)
    counts["research_session"] = 1
    logger.debug(f"Created research session {research_session_id}")

    # Flush to ensure FKs are satisfied before persisting sources/claims
    await db.flush()

    # Step 4-8: Persist sources, evidence, claims, citations using existing logic
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

    # Step 9: Update chat.updated_at to reflect new activity
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


async def persist_simple_message_independent(
    chat_id: UUID,
    user_id: str,
    user_query: str,
    message_id: UUID,
    content: str,
) -> dict[str, int]:
    """Persist simple message with an independent database session.

    Use this for shielded operations where the request-scoped session
    may be cleaned up before persistence completes.

    Args:
        chat_id: ID of the chat.
        user_id: User ID who owns the chat.
        user_query: The user's original query.
        message_id: Pre-generated UUID for the agent message.
        content: The agent's response content.

    Returns:
        Dict with counts of persisted entities.
    """
    from src.db.session import get_session_maker

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


async def persist_complete_research_independent(
    chat_id: UUID,
    user_id: str,
    user_query: str,
    message_id: UUID,
    research_session_id: UUID,
    research_depth: str,
    state: ResearchState,
) -> dict[str, int]:
    """Persist research data with an independent database session.

    Use this for shielded operations where the request-scoped session
    may be cleaned up before persistence completes (e.g., when client
    disconnects during streaming).

    This function creates its own database session that is independent
    of the FastAPI request lifecycle, ensuring persistence completes
    even if the HTTP request is cancelled.

    Args:
        chat_id: ID of the chat to add messages to.
        user_id: User ID who owns the chat.
        user_query: The user's original query.
        message_id: Pre-generated UUID for the agent message.
        research_session_id: Pre-generated UUID for the research session.
        research_depth: Research depth setting (auto/light/medium/extended).
        state: Research state containing final_report, sources, claims, etc.

    Returns:
        Dict with counts of persisted entities.
    """
    from src.db.session import get_session_maker

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
