"""Structured-output slot filling for agent surfaces — app integration.

The generation core (per-slot wires, evidence, envelope assembly) moved to
``databricks_deep_research.surface.generation`` so the standalone shell-app
shares it. This module keeps the APP-side pieces:

* ``load_agent_surface`` — refetch the agents_v2 surface (the engine
  ``WorkflowDefinition`` drops it) for the structuring pass;
* ``structure_and_update`` — the run-time/retry path = framework generation
  followed by the app's targeted ``verification_data`` persistence;
* ``run_slot_wires`` — a thin wrapper preserving the ``llm=None`` convenience
  default (constructs the app ``LLMClient``) the orchestrator relies on.

The remaining names (``build_envelope_v2``, ``build_pending_envelope``,
``apply_slot_guards``, ``SlotOutcome``, ``SlotWireResults``,
``CITATION_MARKER_RE``) are re-exported from the framework for the existing
import paths (orchestrator, restructure endpoint, tests).
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from databricks_deep_research.surface.evidence import EvidenceItem
from databricks_deep_research.surface.generation import (
    CITATION_MARKER_RE,
    WIRE_CONCURRENCY,
    WIRE_TIMEOUT_S,
    SlotOutcome,
    SlotWireResults,
    apply_slot_guards,
    build_envelope_v2,
    build_pending_envelope,
    build_structured_envelope,
)
from databricks_deep_research.surface.generation import (
    run_slot_wires as _fw_run_slot_wires,
)
from databricks_deep_research.surface.output_schema import SlotSpec

from deep_research.agent.orchestration_config import OrchestrationConfig
from deep_research.services.llm.client import LLMClient

logger = logging.getLogger(__name__)

# App-side threshold: skip the structuring pass for reports too short to have
# produced fillable content (read by the orchestrator).
MIN_REPORT_CHARS = 200


async def load_agent_surface(
    config: OrchestrationConfig,
    user_id: str | None,
    db: Any | None,
) -> tuple[dict[str, Any], str | None] | None:
    """Fetch the agent's raw surface (and etag) for the structuring pass.

    The engine ``WorkflowDefinition`` never carries the surface (the loader
    drops unknown keys by design), so the pass refetches the agents_v2 row —
    mirroring ``_resolve_agent_v2_workflow`` including the independent-session
    fallback. Fail-soft: any problem returns None and the pass is skipped.
    """
    if not config.agent_id or not user_id:
        return None
    try:
        agent_uuid = UUID(config.agent_id)
    except ValueError:
        return None

    try:
        from deep_research.services.agent_v2_service import (
            AgentV2Service,
            _compute_etag,
        )

        if db is not None:
            agent = await AgentV2Service(db).get_for_user(agent_uuid, user_id)
        else:
            from deep_research.db.session import get_session_maker

            session_maker = get_session_maker()
            async with session_maker() as session:
                agent = await AgentV2Service(session).get_for_user(
                    agent_uuid, user_id
                )
    except Exception as exc:  # noqa: BLE001 — fail-soft by contract
        logger.warning(
            "STRUCTURED_SURFACE_LOOKUP_FAILED agent_id=%s error=%s",
            config.agent_id,
            str(exc)[:200],
        )
        return None

    if agent is None:
        return None
    definition = agent.definition if isinstance(agent.definition, dict) else None
    surface = definition.get("surface") if definition else None
    if not isinstance(surface, dict):
        return None
    try:
        etag: str | None = _compute_etag(agent.definition, agent.updated_at)
    except Exception:  # noqa: BLE001 — etag is informational metadata
        etag = None
    return (surface, etag)


async def run_slot_wires(
    *,
    slots: dict[str, SlotSpec],
    evidence: list[EvidenceItem],
    claims: list[Any],
    report: str,
    llm: LLMClient | None = None,
    only_slots: set[str] | None = None,
    concurrency: int = WIRE_CONCURRENCY,
    wire_timeout_s: float = WIRE_TIMEOUT_S,
) -> SlotWireResults:
    """Run one wire per slot (app wrapper: ``llm=None`` → a fresh ``LLMClient``)."""
    client = llm if llm is not None else LLMClient()
    return await _fw_run_slot_wires(
        slots=slots,
        evidence=evidence,
        claims=claims,
        report=report,
        llm=client,
        only_slots=only_slots,
        concurrency=concurrency,
        wire_timeout_s=wire_timeout_s,
    )


async def structure_and_update(
    *,
    binding: str,
    agent_id: str | None,
    surface_etag: str | None,
    slots: dict[str, SlotSpec],
    report: str,
    claims: list[Any],
    sources: list[Any],
    chat_id: UUID | None,
    research_session_id: UUID,
    storage_stack: Any | None = None,
    llm: LLMClient | None = None,
    only_slots: set[str] | None = None,
    prior_envelope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The ONE structuring code path shared by run time and retry.

    Framework generation (evidence → wires → v2 envelope, merged over
    *prior_envelope* on a partial rerun) followed by the app's targeted
    ``verification_data`` persistence. Always returns the envelope it attempted
    to persist; persistence problems are logged, never raised.
    """
    client = llm if llm is not None else LLMClient()
    envelope = await build_structured_envelope(
        binding=binding,
        agent_id=agent_id,
        surface_etag=surface_etag,
        slots=slots,
        report=report,
        claims=claims,
        sources=sources,
        llm=client,
        only_slots=only_slots,
        prior_envelope=prior_envelope,
    )

    from deep_research.agent.persistence import (
        update_structured_output_independent,
    )

    written = await update_structured_output_independent(
        chat_id=chat_id,
        research_session_id=research_session_id,
        envelope=envelope,
        storage_stack=storage_stack,
    )
    if not written:
        logger.warning(
            "STRUCTURED_UPDATE_SKIPPED session=%s binding=%s",
            str(research_session_id)[:8],
            binding,
        )
    return envelope


__all__ = [
    "CITATION_MARKER_RE",
    "MIN_REPORT_CHARS",
    "SlotOutcome",
    "SlotWireResults",
    "apply_slot_guards",
    "build_envelope_v2",
    "build_pending_envelope",
    "load_agent_surface",
    "run_slot_wires",
    "structure_and_update",
]
