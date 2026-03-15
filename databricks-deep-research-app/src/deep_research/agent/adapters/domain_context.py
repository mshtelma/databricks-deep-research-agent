"""Domain context tracker — event forwarding and persistence delta.

This is a thin event-forwarding adapter (NOT a state reconstructor).
Each framework ``StreamEvent`` is pattern-matched to produce app-level
SSE events, and a ``PersistenceDelta`` is accumulated for database
writes.

Architecture:
    process_event(StreamEvent) → list[AppSSEEvent]
    should_persist() → bool
    get_persistence_delta() → PersistenceDelta
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from databricks_deep_research.events.types import (
    AgentOutputEvent,
    AgentStreamChunkEvent,
    BackgroundCompletedEvent,
    BranchSelectedEvent,
    CitationCorrectedEvent,
    ClaimGeneratedEvent,
    ClaimVerifiedEvent,
    CoordinatorClassifiedEvent,
    EvaluationDecisionEvent,
    ItemCompletedEvent,
    ItemStartedEvent,
    ItemsExtractedEvent,
    LoopExitEvent,
    LoopIterationEvent,
    NodeCompletedEvent,
    NodeErrorEvent,
    NodeSkippedEvent,
    NodeStartedEvent,
    NumericClaimDetectedEvent,
    PlanAndExecuteExitEvent,
    PlanCreatedEvent,
    ReflectionDecisionEvent,
    ReplanTriggeredEvent,
    StreamEvent,
    SynthesisStartedEvent,
    ToolCallEvent,
    ToolResultEvent,
    VerificationSummaryEvent,
    WorkflowCompletedEvent,
    WorkflowStartedEvent,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Persistence delta
# ---------------------------------------------------------------------------


@dataclass
class PersistenceDelta:
    """Accumulated changes to persist to the database.

    Each field tracks additions/changes since the last persist call.
    After ``get_persistence_delta()`` is called, the delta is reset.
    """

    # Session metadata (set once)
    complexity: str | None = None
    recommended_depth: str | None = None
    is_simple: bool = False
    direct_response: str | None = None

    # Plan (replaced on each PlanCreatedEvent)
    plan: dict[str, Any] | None = None
    plan_steps: list[dict[str, Any]] | None = None

    # Accumulated (append-only between checkpoints)
    new_sources: list[dict[str, Any]] = field(default_factory=list)
    new_observations: list[str] = field(default_factory=list)
    step_updates: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Background
    data_landscape: dict[str, Any] | None = None
    query_decomposition: list[str] | None = None

    # Final
    final_report: str | None = None

    # Verification
    verification_summary: dict[str, Any] | None = None

    # Tracking
    _dirty: bool = False

    # Per-step source tracking (reset at ItemStarted, read at ItemCompleted)
    _step_sources_found: int = 0


# ---------------------------------------------------------------------------
# App SSE event (matches existing frontend protocol)
# ---------------------------------------------------------------------------


@dataclass
class AppSSEEvent:
    """An SSE event to send to the frontend."""

    event_type: str
    data: dict[str, Any]


# ---------------------------------------------------------------------------
# Domain context tracker
# ---------------------------------------------------------------------------


class DomainContextTracker:
    """Event-forwarding adapter between framework and app.

    Each handler is 5-15 lines.  No state reconstruction — events carry
    all the metadata the app needs.
    """

    def __init__(self) -> None:
        self._delta = PersistenceDelta()
        self._events_processed = 0
        self._persist_interval = 5  # persist every N events

    def process_event(self, event: StreamEvent) -> list[AppSSEEvent]:
        """Convert a framework event to app SSE events.

        Returns a list because some framework events map to multiple
        app events (e.g., workflow_completed → status + report).
        """
        self._events_processed += 1
        handler = _HANDLERS.get(type(event))
        if handler:
            return handler(event, self._delta)  # type: ignore[no-any-return]
        logger.debug(
            "FWK_EVENT_UNHANDLED type=%s node_id=%s",
            type(event).__name__,
            event.node_id,
        )
        return []

    def should_persist(self) -> bool:
        """Check if accumulated delta should be written to DB."""
        return self._delta._dirty and self._events_processed % self._persist_interval == 0

    def get_persistence_delta(self) -> PersistenceDelta:
        """Return and reset the accumulated delta."""
        delta = self._delta
        # Preserve in-flight step source count across delta resets
        carry_sources = delta._step_sources_found
        # Carry final_report forward — it's set-once and must survive
        # periodic resets (the periodic _persist_delta() is a no-op,
        # so the post-loop read must still see it).
        carry_report = delta.final_report
        # Carry verification_summary forward — also set-once at the end
        # of the pipeline and must survive periodic resets.
        carry_verification = delta.verification_summary
        self._delta = PersistenceDelta()
        self._delta._step_sources_found = carry_sources
        if carry_report:
            self._delta.final_report = carry_report
        if carry_verification:
            self._delta.verification_summary = carry_verification
        return delta

    @property
    def events_processed(self) -> int:
        """Total events processed since creation."""
        return self._events_processed


# ---------------------------------------------------------------------------
# Event handlers (each 5-15 lines)
# ---------------------------------------------------------------------------


def _handle_coordinator(event: StreamEvent, delta: PersistenceDelta) -> list[AppSSEEvent]:
    e = event  # type: CoordinatorClassifiedEvent
    delta.complexity = e.complexity
    delta.recommended_depth = e.recommended_depth
    delta.is_simple = e.is_simple
    delta.direct_response = e.direct_response
    delta._dirty = True
    return [AppSSEEvent(
        event_type="coordinator_classified",
        data={
            "complexity": e.complexity,
            "recommended_depth": e.recommended_depth,
            "is_simple": e.is_simple,
            "direct_response": e.direct_response,
            "follow_up_type": e.follow_up_type,
        },
    )]


def _handle_background(event: StreamEvent, delta: PersistenceDelta) -> list[AppSSEEvent]:
    e = event  # type: BackgroundCompletedEvent
    delta.data_landscape = e.data_landscape
    delta.query_decomposition = e.query_decomposition
    delta._dirty = True
    return [AppSSEEvent(
        event_type="background_completed",
        data={
            "sources_discovered": e.sources_discovered,
            "data_landscape_summary": e.data_landscape_summary,
        },
    )]


def _handle_plan_created(event: StreamEvent, delta: PersistenceDelta) -> list[AppSSEEvent]:
    e = event  # type: PlanCreatedEvent
    delta.plan = {"title": e.title, "thought": e.thought, "steps": e.steps}
    delta.plan_steps = e.steps
    delta._dirty = True
    return [AppSSEEvent(
        event_type="plan_created",
        data={
            "plan_id": e.plan_id,
            "title": e.title,
            "thought": e.thought,
            "steps": e.steps,
            "iteration": e.iteration,
            "has_enough_context": e.has_enough_context,
        },
    )]


def _handle_item_started(event: StreamEvent, delta: PersistenceDelta) -> list[AppSSEEvent]:
    e = event  # type: ItemStartedEvent
    delta._step_sources_found = 0  # Reset for new step
    return [AppSSEEvent(
        event_type="step_started",
        data={
            "item_index": e.item_index,
            "item_summary": e.item_summary,
            "total_items": e.total_items,
        },
    )]


def _handle_item_completed(event: StreamEvent, delta: PersistenceDelta) -> list[AppSSEEvent]:
    e = event  # type: ItemCompletedEvent
    sources_found = delta._step_sources_found
    delta._step_sources_found = 0  # Reset for next step
    delta.step_updates[str(e.item_index)] = {
        "status": "completed",
        "sources_found": sources_found,
    }
    delta._dirty = True
    return [AppSSEEvent(
        event_type="step_completed",
        data={
            "item_index": e.item_index,
            "items_processed": e.items_processed,
            "sources_found": sources_found,
        },
    )]


def _handle_reflection(event: StreamEvent, delta: PersistenceDelta) -> list[AppSSEEvent]:
    e = event  # type: ReflectionDecisionEvent
    return [AppSSEEvent(
        event_type="reflection_decision",
        data={"decision": e.decision, "reasoning": e.reasoning},
    )]


def _handle_stream_chunk(event: StreamEvent, delta: PersistenceDelta) -> list[AppSSEEvent]:
    e = event  # type: AgentStreamChunkEvent
    return [AppSSEEvent(
        event_type="stream_chunk",
        data={"chunk": e.chunk, "node_id": e.node_id, "subtype": e.subtype},
    )]


def _handle_synthesis_started(event: StreamEvent, delta: PersistenceDelta) -> list[AppSSEEvent]:
    return [AppSSEEvent(
        event_type="synthesis_started",
        data={
            "total_observations": getattr(event, "total_observations", 0),
            "total_sources": getattr(event, "total_sources", 0),
        },
    )]


def _handle_agent_output(event: StreamEvent, delta: PersistenceDelta) -> list[AppSSEEvent]:
    e = event  # type: AgentOutputEvent
    if e.output_key == "report":
        delta.final_report = e.output_preview
        delta._dirty = True
    return [AppSSEEvent(
        event_type="agent_output",
        data={
            "node_id": e.node_id,
            "output_key": e.output_key,
            "output_preview": e.output_preview,
        },
    )]


def _handle_workflow_completed(event: StreamEvent, delta: PersistenceDelta) -> list[AppSSEEvent]:
    # WorkflowCompletedEvent carries the FULL report from state.get("report").
    # AgentOutputEvent.output_preview is truncated to 200 chars (harness.py:227),
    # so this is the authoritative source for the complete final report.
    e: WorkflowCompletedEvent = event  # type: ignore[assignment]
    if e.final_report:
        delta.final_report = e.final_report
    delta._dirty = True
    return [AppSSEEvent(
        event_type="workflow_completed",
        data={"workflow_id": event.node_id},
    )]


def _handle_node_error(event: StreamEvent, delta: PersistenceDelta) -> list[AppSSEEvent]:
    e = event  # type: NodeErrorEvent
    return [AppSSEEvent(
        event_type="node_error",
        data={
            "node_id": e.node_id,
            "error": e.error_message,
            "will_retry": e.will_retry,
        },
    )]


# --- Verification event handlers ---


def _handle_claim_generated(event: StreamEvent, delta: PersistenceDelta) -> list[AppSSEEvent]:
    e = event  # type: ClaimGeneratedEvent
    return [AppSSEEvent(
        event_type="research_progress",
        data={
            "progress_type": "claim_generated",
            "claim_index": e.claim_index,
            "claim_text": e.claim_text,
            "citation_keys": e.citation_keys,
        },
    )]


def _handle_claim_verified(event: StreamEvent, delta: PersistenceDelta) -> list[AppSSEEvent]:
    e = event  # type: ClaimVerifiedEvent
    delta._dirty = True
    return [AppSSEEvent(
        event_type="research_progress",
        data={
            "progress_type": "claim_verified",
            "claim_index": e.claim_index,
            "verdict": e.verdict,
            "confidence": e.confidence,
            "evidence_snippet": e.evidence_snippet,
            "claim_text": e.claim_text,
        },
    )]


def _handle_citation_corrected(event: StreamEvent, delta: PersistenceDelta) -> list[AppSSEEvent]:
    e = event  # type: CitationCorrectedEvent
    return [AppSSEEvent(
        event_type="research_progress",
        data={
            "progress_type": "citation_corrected",
            "claim_index": e.claim_index,
            "action": e.action,
            "original_key": e.original_key,
            "corrected_key": e.corrected_key,
        },
    )]


def _handle_numeric_claim_detected(event: StreamEvent, delta: PersistenceDelta) -> list[AppSSEEvent]:
    e = event  # type: NumericClaimDetectedEvent
    return [AppSSEEvent(
        event_type="research_progress",
        data={
            "progress_type": "numeric_claim_detected",
            "claim_index": e.claim_index,
            "numeric_value": e.numeric_value,
            "verification_status": e.verification_status,
        },
    )]


def _handle_verification_summary(event: StreamEvent, delta: PersistenceDelta) -> list[AppSSEEvent]:
    e = event  # type: VerificationSummaryEvent
    delta.verification_summary = {
        "total_claims": e.total_claims,
        "verified_claims": e.verified_claims,
        "corrected_citations": e.corrected_citations,
        "removed_claims": e.removed_claims,
        "softened_claims": e.softened_claims,
        "overall_confidence": e.overall_confidence,
    }
    delta._dirty = True
    return [AppSSEEvent(
        event_type="research_progress",
        data={
            "progress_type": "verification_summary",
            "total_claims": e.total_claims,
            "verified_claims": e.verified_claims,
            "corrected_citations": e.corrected_citations,
            "removed_claims": e.removed_claims,
            "softened_claims": e.softened_claims,
            "overall_confidence": e.overall_confidence,
        },
    )]


# --- Progress event handlers (provide frontend visibility) ---


def _handle_replan_triggered(event: StreamEvent, delta: PersistenceDelta) -> list[AppSSEEvent]:
    e: ReplanTriggeredEvent = event  # type: ignore[assignment]
    return [AppSSEEvent(
        event_type="research_progress",
        data={
            "progress_type": "replan_triggered",
            "cycle": e.cycle,
            "reason": e.reason,
            "items_remaining": e.items_remaining,
        },
    )]


def _handle_tool_call(event: StreamEvent, delta: PersistenceDelta) -> list[AppSSEEvent]:
    e: ToolCallEvent = event  # type: ignore[assignment]
    return [AppSSEEvent(
        event_type="research_progress",
        data={
            "progress_type": "tool_call",
            "tool_name": e.tool_name,
            "node_id": e.node_id,
        },
    )]


def _handle_tool_result(event: StreamEvent, delta: PersistenceDelta) -> list[AppSSEEvent]:
    e: ToolResultEvent = event  # type: ignore[assignment]
    # Store cumulative source count (replace, not add — it's already cumulative
    # from ReactLoop: source_count=len(sources) where sources grows per tool call)
    delta._step_sources_found = e.source_count
    return [AppSSEEvent(
        event_type="research_progress",
        data={
            "progress_type": "tool_result",
            "tool_name": e.tool_name,
            "result_summary": e.result_summary,
            "source_count": e.source_count,
        },
    )]


def _handle_evaluation_decision(event: StreamEvent, delta: PersistenceDelta) -> list[AppSSEEvent]:
    e: EvaluationDecisionEvent = event  # type: ignore[assignment]
    return [AppSSEEvent(
        event_type="research_progress",
        data={
            "progress_type": "evaluation_decision",
            "decision": e.decision,
            "reasoning": e.reasoning,
            "items_processed": e.items_processed,
        },
    )]


# --- No-op handler for infrastructure events (acknowledged, not forwarded) ---


def _handle_noop(event: StreamEvent, delta: PersistenceDelta) -> list[AppSSEEvent]:
    return []


# Handler dispatch table
_HANDLERS: dict[type, Any] = {
    CoordinatorClassifiedEvent: _handle_coordinator,
    BackgroundCompletedEvent: _handle_background,
    PlanCreatedEvent: _handle_plan_created,
    ItemStartedEvent: _handle_item_started,
    ItemCompletedEvent: _handle_item_completed,
    ReflectionDecisionEvent: _handle_reflection,
    AgentStreamChunkEvent: _handle_stream_chunk,
    SynthesisStartedEvent: _handle_synthesis_started,
    AgentOutputEvent: _handle_agent_output,
    WorkflowCompletedEvent: _handle_workflow_completed,
    NodeErrorEvent: _handle_node_error,
    ClaimGeneratedEvent: _handle_claim_generated,
    ClaimVerifiedEvent: _handle_claim_verified,
    CitationCorrectedEvent: _handle_citation_corrected,
    NumericClaimDetectedEvent: _handle_numeric_claim_detected,
    VerificationSummaryEvent: _handle_verification_summary,
    # Progress events (provide frontend visibility)
    ReplanTriggeredEvent: _handle_replan_triggered,
    ToolCallEvent: _handle_tool_call,
    ToolResultEvent: _handle_tool_result,
    EvaluationDecisionEvent: _handle_evaluation_decision,
    # Infrastructure events (acknowledged, not forwarded to frontend)
    WorkflowStartedEvent: _handle_noop,
    NodeStartedEvent: _handle_noop,
    NodeCompletedEvent: _handle_noop,
    NodeSkippedEvent: _handle_noop,
    LoopIterationEvent: _handle_noop,
    LoopExitEvent: _handle_noop,
    BranchSelectedEvent: _handle_noop,
    ItemsExtractedEvent: _handle_noop,
    PlanAndExecuteExitEvent: _handle_noop,
}


__all__ = [
    "AppSSEEvent",
    "DomainContextTracker",
    "PersistenceDelta",
]
