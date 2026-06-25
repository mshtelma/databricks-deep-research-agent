"""Post-run consolidation: persist a turn's verified knowledge into chat memory.

Called by the orchestrator after a research/web turn completes. Applies the
trust policy (``extract_consolidatable_claims``) to the persisted
``verification_data`` and writes only the verified claims into the durable
chat-memory store, so the NEXT turn can read and cite them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from deep_research.agent.consolidation_policy import extract_consolidatable_claims

if TYPE_CHECKING:
    from deep_research.services._protocols import IChatMemoryService


async def consolidate_turn_knowledge(
    memory: IChatMemoryService,
    chat_id: UUID,
    research_session_id: UUID | None,
    verification_data: dict[str, Any] | None,
    *,
    source_step: int = 1,
    coverage_topics: list[dict[str, Any]] | None = None,
) -> int:
    """Persist this turn's verified claims as durable findings (+ coverage).

    Returns the number of findings written (0 when there is nothing verified
    to persist). Assumes ``memory.hydrate(chat_id)`` was already called this
    turn (the orchestrator hydrates chat memory before running the workflow).

    ``coverage_topics`` (Phase 2e) records what this turn covered, for the
    Phase-3a routing gate. A turn with coverage but no new verified claims
    still writes coverage (so "covered" is recorded even when nothing new was
    confirmed). Topic *extraction* (what granularity to record) is wired by the
    caller; this helper is the trust-gated pass-through.
    """
    claims = extract_consolidatable_claims(verification_data)
    if not claims and not coverage_topics:
        return 0
    return await memory.consolidate_from_pool(
        chat_id,
        claims=claims,
        observations=[],
        research_session_id=research_session_id,
        source_step=source_step,
        coverage_topics=coverage_topics,
    )
