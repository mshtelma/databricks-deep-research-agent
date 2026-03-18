"""Unit tests for the domain context tracker (domain_context.py).

Tests event forwarding (framework StreamEvent -> AppSSEEvent),
persistence delta accumulation, and should_persist / reset logic.
"""

from __future__ import annotations

from typing import Any

import pytest

from databricks_deep_research.events.types import (
    AgentOutputEvent,
    AgentStreamChunkEvent,
    BackgroundCompletedEvent,
    CoordinatorClassifiedEvent,
    ItemCompletedEvent,
    ItemStartedEvent,
    NodeErrorEvent,
    PlanCreatedEvent,
    ReflectionDecisionEvent,
    SynthesisStartedEvent,
    ToolCallEvent,
    ToolResultEvent,
    WorkflowCompletedEvent,
    WorkflowStartedEvent,
)
from deep_research.agent.adapters.domain_context import (
    AppSSEEvent,
    DomainContextTracker,
    PersistenceDelta,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TS = "2026-03-09T12:00:00Z"


def _base(node_id: str = "test-node") -> dict[str, str]:
    """Base fields for event creation."""
    return {"node_id": node_id, "timestamp": _TS}


# ---------------------------------------------------------------------------
# Tests — CoordinatorClassifiedEvent
# ---------------------------------------------------------------------------


class TestHandleCoordinator:
    """Tests for coordinator_classified event forwarding."""

    def test_produces_coordinator_classified_sse(self) -> None:
        tracker = DomainContextTracker()
        event = CoordinatorClassifiedEvent(
            **_base(),
            complexity="complex",
            recommended_depth="extended",
            is_simple=False,
            direct_response=None,
            follow_up_type="deep_research",
        )

        sse_events = tracker.process_event(event)

        assert len(sse_events) == 1
        assert sse_events[0].event_type == "coordinator_classified"
        assert sse_events[0].data["complexity"] == "complex"
        assert sse_events[0].data["recommended_depth"] == "extended"
        assert sse_events[0].data["is_simple"] is False
        assert sse_events[0].data["follow_up_type"] == "deep_research"

    def test_sets_persistence_delta_fields(self) -> None:
        tracker = DomainContextTracker()
        event = CoordinatorClassifiedEvent(
            **_base(),
            complexity="simple",
            recommended_depth="light",
            is_simple=True,
            direct_response="42 is the answer.",
        )

        tracker.process_event(event)
        delta = tracker.get_persistence_delta()

        assert delta.complexity == "simple"
        assert delta.recommended_depth == "light"
        assert delta.is_simple is True
        assert delta.direct_response == "42 is the answer."

    def test_marks_delta_dirty(self) -> None:
        tracker = DomainContextTracker()
        event = CoordinatorClassifiedEvent(
            **_base(),
            complexity="moderate",
            recommended_depth="medium",
        )

        tracker.process_event(event)

        # Delta should be dirty after coordinator event
        assert tracker._delta._dirty is True


# ---------------------------------------------------------------------------
# Tests — BackgroundCompletedEvent
# ---------------------------------------------------------------------------


class TestHandleBackground:
    """Tests for background_completed event forwarding."""

    def test_produces_background_completed_sse(self) -> None:
        tracker = DomainContextTracker()
        event = BackgroundCompletedEvent(
            **_base(),
            sources_discovered=5,
            data_landscape_summary="Found papers on quantum computing",
            data_landscape={"topics": ["quantum"]},
            query_decomposition=["sub-q1", "sub-q2"],
        )

        sse_events = tracker.process_event(event)

        assert len(sse_events) == 1
        assert sse_events[0].event_type == "background_completed"
        assert sse_events[0].data["sources_discovered"] == 5
        assert sse_events[0].data["data_landscape_summary"] == "Found papers on quantum computing"

    def test_accumulates_delta(self) -> None:
        tracker = DomainContextTracker()
        event = BackgroundCompletedEvent(
            **_base(),
            data_landscape={"topics": ["ML"]},
            query_decomposition=["q1"],
        )

        tracker.process_event(event)
        delta = tracker.get_persistence_delta()

        assert delta.data_landscape == {"topics": ["ML"]}
        assert delta.query_decomposition == ["q1"]


# ---------------------------------------------------------------------------
# Tests — PlanCreatedEvent
# ---------------------------------------------------------------------------


class TestHandlePlanCreated:
    """Tests for plan_created event forwarding."""

    def test_produces_plan_created_sse(self) -> None:
        tracker = DomainContextTracker()
        steps = [{"title": "Step 1", "query": "search X"}]
        event = PlanCreatedEvent(
            **_base(),
            plan_id="plan-123",
            title="Research Plan",
            thought="Need to investigate X",
            steps=steps,
            iteration=1,
            has_enough_context=False,
        )

        sse_events = tracker.process_event(event)

        assert len(sse_events) == 1
        assert sse_events[0].event_type == "plan_created"
        assert sse_events[0].data["plan_id"] == "plan-123"
        assert sse_events[0].data["title"] == "Research Plan"
        assert sse_events[0].data["steps"] == steps
        assert sse_events[0].data["iteration"] == 1

    def test_replaces_plan_in_delta(self) -> None:
        tracker = DomainContextTracker()
        steps_v1 = [{"title": "Old Step"}]
        steps_v2 = [{"title": "New Step"}]

        tracker.process_event(PlanCreatedEvent(
            **_base(), plan_id="p1", title="V1", thought="t1", steps=steps_v1,
        ))
        tracker.process_event(PlanCreatedEvent(
            **_base(), plan_id="p2", title="V2", thought="t2", steps=steps_v2,
        ))

        delta = tracker.get_persistence_delta()

        assert delta.plan["title"] == "V2"
        assert delta.plan_steps == steps_v2


# ---------------------------------------------------------------------------
# Tests — ItemStartedEvent / ItemCompletedEvent
# ---------------------------------------------------------------------------


class TestHandleItems:
    """Tests for step_started and step_completed event forwarding."""

    def test_item_started_sse(self) -> None:
        tracker = DomainContextTracker()
        event = ItemStartedEvent(
            **_base(),
            item_index=0,
            item_summary="Search for X",
            total_items=3,
        )

        sse_events = tracker.process_event(event)

        assert len(sse_events) == 1
        assert sse_events[0].event_type == "step_started"
        assert sse_events[0].data["item_index"] == 0
        assert sse_events[0].data["total_items"] == 3

    def test_item_completed_sse(self) -> None:
        tracker = DomainContextTracker()
        event = ItemCompletedEvent(
            **_base(),
            item_index=2,
            items_processed=3,
        )

        sse_events = tracker.process_event(event)

        assert len(sse_events) == 1
        assert sse_events[0].event_type == "step_completed"
        assert sse_events[0].data["item_index"] == 2

    def test_item_completed_accumulates_step_updates(self) -> None:
        tracker = DomainContextTracker()

        tracker.process_event(ItemCompletedEvent(
            **_base(), item_index=0, items_processed=1,
        ))
        tracker.process_event(ItemCompletedEvent(
            **_base(), item_index=1, items_processed=2,
        ))

        delta = tracker.get_persistence_delta()

        assert "0" in delta.step_updates
        assert "1" in delta.step_updates
        assert delta.step_updates["0"]["status"] == "completed"
        assert delta.step_updates["1"]["status"] == "completed"


# ---------------------------------------------------------------------------
# Tests — ReflectionDecisionEvent
# ---------------------------------------------------------------------------


class TestHandleReflection:
    """Tests for reflection_decision event forwarding."""

    def test_produces_reflection_decision_sse(self) -> None:
        tracker = DomainContextTracker()
        event = ReflectionDecisionEvent(
            **_base(),
            decision="continue",
            reasoning="Need more data on topic Y",
        )

        sse_events = tracker.process_event(event)

        assert len(sse_events) == 1
        assert sse_events[0].event_type == "reflection_decision"
        assert sse_events[0].data["decision"] == "continue"
        assert sse_events[0].data["reasoning"] == "Need more data on topic Y"


# ---------------------------------------------------------------------------
# Tests — AgentStreamChunkEvent
# ---------------------------------------------------------------------------


class TestHandleStreamChunk:
    """Tests for stream_chunk event forwarding."""

    def test_produces_stream_chunk_sse(self) -> None:
        tracker = DomainContextTracker()
        event = AgentStreamChunkEvent(
            **_base(node_id="synthesizer"),
            chunk="Here is the report...",
            subtype="synthesis",
        )

        sse_events = tracker.process_event(event)

        assert len(sse_events) == 1
        assert sse_events[0].event_type == "stream_chunk"
        assert sse_events[0].data["chunk"] == "Here is the report..."
        assert sse_events[0].data["node_id"] == "synthesizer"
        assert sse_events[0].data["subtype"] == "synthesis"


# ---------------------------------------------------------------------------
# Tests — SynthesisStartedEvent
# ---------------------------------------------------------------------------


class TestHandleSynthesisStarted:
    """Tests for synthesis_started event forwarding."""

    def test_produces_synthesis_started_sse(self) -> None:
        tracker = DomainContextTracker()
        event = SynthesisStartedEvent(
            **_base(),
            total_observations=15,
            total_sources=8,
        )

        sse_events = tracker.process_event(event)

        assert len(sse_events) == 1
        assert sse_events[0].event_type == "synthesis_started"
        assert sse_events[0].data["total_observations"] == 15
        assert sse_events[0].data["total_sources"] == 8


# ---------------------------------------------------------------------------
# Tests — AgentOutputEvent
# ---------------------------------------------------------------------------


class TestHandleAgentOutput:
    """Tests for agent_output event forwarding."""

    def test_produces_agent_output_sse(self) -> None:
        tracker = DomainContextTracker()
        event = AgentOutputEvent(
            **_base(),
            output_key="report",
            output_preview="# Final Report\n...",
        )

        sse_events = tracker.process_event(event)

        assert len(sse_events) == 1
        assert sse_events[0].event_type == "agent_output"
        assert sse_events[0].data["output_key"] == "report"
        assert sse_events[0].data["output_preview"] == "# Final Report\n..."

    def test_report_key_sets_final_report_in_delta(self) -> None:
        tracker = DomainContextTracker()
        event = AgentOutputEvent(
            **_base(),
            output_key="report",
            output_preview="The final report content.",
        )

        tracker.process_event(event)
        delta = tracker.get_persistence_delta()

        assert delta.final_report == "The final report content."

    def test_non_report_key_does_not_set_final_report(self) -> None:
        tracker = DomainContextTracker()
        event = AgentOutputEvent(
            **_base(),
            output_key="findings",
            output_preview="Some findings.",
        )

        tracker.process_event(event)
        delta = tracker.get_persistence_delta()

        assert delta.final_report is None


# ---------------------------------------------------------------------------
# Tests — WorkflowCompletedEvent
# ---------------------------------------------------------------------------


class TestHandleWorkflowCompleted:
    """Tests for workflow_completed event forwarding."""

    def test_produces_workflow_completed_sse(self) -> None:
        tracker = DomainContextTracker()
        event = WorkflowCompletedEvent(
            **_base(node_id="main"),
            workflow_id="wf-123",
            duration_ms=42000.0,
            total_tokens=50000,
        )

        sse_events = tracker.process_event(event)

        assert len(sse_events) == 1
        assert sse_events[0].event_type == "workflow_completed"
        assert sse_events[0].data["workflow_id"] == "main"

    def test_marks_delta_dirty(self) -> None:
        tracker = DomainContextTracker()
        event = WorkflowCompletedEvent(
            **_base(),
            workflow_id="wf-123",
            duration_ms=1000.0,
            total_tokens=100,
        )

        tracker.process_event(event)

        assert tracker._delta._dirty is True


# ---------------------------------------------------------------------------
# Tests — NodeErrorEvent
# ---------------------------------------------------------------------------


class TestHandleNodeError:
    """Tests for node_error event forwarding."""

    def test_produces_node_error_sse(self) -> None:
        tracker = DomainContextTracker()
        event = NodeErrorEvent(
            **_base(node_id="researcher"),
            error_message="Rate limit exceeded",
            will_retry=True,
            retry_attempt=1,
        )

        sse_events = tracker.process_event(event)

        assert len(sse_events) == 1
        assert sse_events[0].event_type == "node_error"
        assert sse_events[0].data["node_id"] == "researcher"
        assert sse_events[0].data["error"] == "Rate limit exceeded"
        assert sse_events[0].data["will_retry"] is True


# ---------------------------------------------------------------------------
# Tests — Unknown events
# ---------------------------------------------------------------------------


class TestUnknownEvents:
    """Tests for events without a registered handler."""

    def test_unknown_event_returns_empty_list(self) -> None:
        tracker = DomainContextTracker()
        # WorkflowStartedEvent is NOT in _HANDLERS
        event = WorkflowStartedEvent(
            **_base(),
            workflow_id="wf-1",
            workflow_name="Test",
        )

        sse_events = tracker.process_event(event)

        assert sse_events == []

    def test_unknown_event_still_increments_counter(self) -> None:
        tracker = DomainContextTracker()
        event = WorkflowStartedEvent(
            **_base(),
            workflow_id="wf-1",
            workflow_name="Test",
        )

        tracker.process_event(event)

        assert tracker.events_processed == 1

    def test_tool_call_event_produces_progress(self) -> None:
        """ToolCallEvent is now handled, producing a research_progress event."""
        tracker = DomainContextTracker()
        event = ToolCallEvent(
            **_base(),
            tool_name="web_search",
            arguments={"query": "test"},
        )

        sse_events = tracker.process_event(event)

        assert len(sse_events) == 1
        assert sse_events[0].event_type == "research_progress"
        assert sse_events[0].data["progress_type"] == "tool_call"
        assert sse_events[0].data["tool_name"] == "web_search"


# ---------------------------------------------------------------------------
# Tests — Persistence logic
# ---------------------------------------------------------------------------


class TestPersistenceLogic:
    """Tests for should_persist / get_persistence_delta."""

    def test_should_persist_false_when_not_dirty(self) -> None:
        tracker = DomainContextTracker()

        # Process an unknown event (no handler, delta not dirty)
        event = WorkflowStartedEvent(
            **_base(), workflow_id="wf-1", workflow_name="Test",
        )
        # Process 5 events to hit the interval
        for _ in range(5):
            tracker.process_event(event)

        assert tracker.should_persist() is False

    def test_should_persist_true_at_interval_when_dirty(self) -> None:
        tracker = DomainContextTracker()

        # Process a handled event (makes delta dirty)
        coord_event = CoordinatorClassifiedEvent(
            **_base(),
            complexity="simple",
            recommended_depth="light",
        )
        unknown_event = WorkflowStartedEvent(
            **_base(), workflow_id="wf-1", workflow_name="Test",
        )

        tracker.process_event(coord_event)  # event 1, dirty=True

        # Process more events to reach interval (5)
        for _ in range(4):
            tracker.process_event(unknown_event)

        # Now at event 5, dirty=True => should_persist=True
        assert tracker.events_processed == 5
        assert tracker.should_persist() is True

    def test_should_persist_false_between_intervals(self) -> None:
        tracker = DomainContextTracker()

        coord_event = CoordinatorClassifiedEvent(
            **_base(),
            complexity="simple",
            recommended_depth="light",
        )

        tracker.process_event(coord_event)  # event 1
        assert tracker.should_persist() is False

        tracker.process_event(coord_event)  # event 2
        assert tracker.should_persist() is False

    def test_get_persistence_delta_resets(self) -> None:
        tracker = DomainContextTracker()

        event = CoordinatorClassifiedEvent(
            **_base(),
            complexity="complex",
            recommended_depth="extended",
        )
        tracker.process_event(event)

        delta1 = tracker.get_persistence_delta()
        assert delta1.complexity == "complex"
        assert delta1._dirty is True

        # After reset, new delta should be clean
        delta2 = tracker.get_persistence_delta()
        assert delta2.complexity is None
        assert delta2._dirty is False

    def test_events_processed_counter(self) -> None:
        tracker = DomainContextTracker()

        assert tracker.events_processed == 0

        event = CoordinatorClassifiedEvent(
            **_base(),
            complexity="simple",
            recommended_depth="light",
        )
        tracker.process_event(event)
        tracker.process_event(event)

        assert tracker.events_processed == 2


# ---------------------------------------------------------------------------
# Tests — PersistenceDelta dataclass
# ---------------------------------------------------------------------------


class TestPersistenceDelta:
    """Tests for PersistenceDelta defaults."""

    def test_default_values(self) -> None:
        delta = PersistenceDelta()

        assert delta.complexity is None
        assert delta.recommended_depth is None
        assert delta.is_simple is False
        assert delta.direct_response is None
        assert delta.plan is None
        assert delta.plan_steps is None
        assert delta.new_sources == []
        assert delta.new_observations == []
        assert delta.step_updates == {}
        assert delta.data_landscape is None
        assert delta.query_decomposition is None
        assert delta.final_report is None
        assert delta._dirty is False
        assert delta._step_sources_found == 0


# ---------------------------------------------------------------------------
# Tests — Source count tracking
# ---------------------------------------------------------------------------


class TestSourceCountTracking:
    """Tests for per-step source count propagation.

    Validates that ToolResultEvent.source_count flows through to
    StepCompletedEvent via PersistenceDelta._step_sources_found.
    """

    def test_tool_result_stores_source_count(self) -> None:
        tracker = DomainContextTracker()
        event = ToolResultEvent(
            **_base(),
            tool_name="web_search",
            result_summary="Found results",
            source_count=5,
        )

        tracker.process_event(event)

        assert tracker._delta._step_sources_found == 5

    def test_cumulative_count_replaces_not_sums(self) -> None:
        """source_count is cumulative from ReactLoop — store latest, don't add."""
        tracker = DomainContextTracker()

        tracker.process_event(ToolResultEvent(
            **_base(), tool_name="web_search",
            result_summary="First", source_count=3,
        ))
        tracker.process_event(ToolResultEvent(
            **_base(), tool_name="web_crawl",
            result_summary="Second", source_count=7,
        ))

        assert tracker._delta._step_sources_found == 7  # Not 10

    def test_item_completed_includes_sources_found(self) -> None:
        tracker = DomainContextTracker()

        tracker.process_event(ToolResultEvent(
            **_base(), tool_name="web_search",
            result_summary="Found results", source_count=8,
        ))
        sse_events = tracker.process_event(ItemCompletedEvent(
            **_base(), item_index=0, items_processed=1,
        ))

        assert len(sse_events) == 1
        assert sse_events[0].data["sources_found"] == 8

    def test_item_completed_resets_counter(self) -> None:
        tracker = DomainContextTracker()

        tracker.process_event(ToolResultEvent(
            **_base(), tool_name="web_search",
            result_summary="Found results", source_count=5,
        ))
        tracker.process_event(ItemCompletedEvent(
            **_base(), item_index=0, items_processed=1,
        ))

        assert tracker._delta._step_sources_found == 0

    def test_item_started_resets_counter(self) -> None:
        tracker = DomainContextTracker()
        tracker._delta._step_sources_found = 5

        tracker.process_event(ItemStartedEvent(
            **_base(), item_index=1, item_summary="Next step", total_items=3,
        ))

        assert tracker._delta._step_sources_found == 0

    def test_step_with_no_tool_calls_has_zero_sources(self) -> None:
        """When no ToolResultEvent occurs between start and complete, sources_found=0."""
        tracker = DomainContextTracker()

        tracker.process_event(ItemStartedEvent(
            **_base(), item_index=0, item_summary="Step 1", total_items=1,
        ))
        sse_events = tracker.process_event(ItemCompletedEvent(
            **_base(), item_index=0, items_processed=1,
        ))

        assert sse_events[0].data["sources_found"] == 0

    def test_delta_reset_preserves_step_sources(self) -> None:
        """get_persistence_delta() must carry _step_sources_found to new delta."""
        tracker = DomainContextTracker()

        tracker.process_event(ToolResultEvent(
            **_base(), tool_name="web_search",
            result_summary="Found results", source_count=5,
        ))

        # Simulate mid-step persistence reset
        tracker.get_persistence_delta()

        # Counter should survive the reset
        assert tracker._delta._step_sources_found == 5

        # ItemCompleted should still see the correct count
        sse_events = tracker.process_event(ItemCompletedEvent(
            **_base(), item_index=0, items_processed=1,
        ))
        assert sse_events[0].data["sources_found"] == 5


# ---------------------------------------------------------------------------
# Tests — Final report survives delta reset
# ---------------------------------------------------------------------------


class TestFinalReportDeltaReset:
    """Tests that final_report is carried forward across get_persistence_delta() resets.

    Regression test for the bug where periodic persistence (every 5 events)
    called get_persistence_delta(), which reset the delta and discarded
    final_report before the post-loop read.
    """

    def test_final_report_survives_delta_reset(self) -> None:
        """Process WorkflowCompletedEvent, reset delta twice — both should have report."""
        tracker = DomainContextTracker()
        report_text = "# Full Research Report\n\nThis is the complete report."

        event = WorkflowCompletedEvent(
            **_base(node_id="main"),
            workflow_id="wf-123",
            duration_ms=42000.0,
            total_tokens=50000,
            final_report=report_text,
        )

        tracker.process_event(event)

        # First delta read should have the report
        delta1 = tracker.get_persistence_delta()
        assert delta1.final_report == report_text

        # Second delta read (simulates post-loop read after periodic reset)
        # should ALSO have the report carried forward
        delta2 = tracker.get_persistence_delta()
        assert delta2.final_report == report_text

    def test_empty_final_report_not_carried(self) -> None:
        """Verify None/empty report is not spuriously carried forward.

        WorkflowCompletedEvent.final_report defaults to "" but the handler
        only sets delta.final_report when truthy, so empty string is skipped.
        """
        tracker = DomainContextTracker()

        # WorkflowCompletedEvent with empty report (default)
        event = WorkflowCompletedEvent(
            **_base(node_id="main"),
            workflow_id="wf-123",
            duration_ms=1000.0,
            total_tokens=100,
        )

        tracker.process_event(event)

        # Empty string is falsy — handler skips it, delta stays None
        delta1 = tracker.get_persistence_delta()
        assert delta1.final_report is None

        # After reset, new delta should also be None (nothing to carry)
        delta2 = tracker.get_persistence_delta()
        assert delta2.final_report is None

    def test_report_from_agent_output_also_survives(self) -> None:
        """AgentOutputEvent with key='report' should also survive delta reset."""
        tracker = DomainContextTracker()

        event = AgentOutputEvent(
            **_base(),
            output_key="report",
            output_preview="Truncated preview...",
        )

        tracker.process_event(event)

        delta1 = tracker.get_persistence_delta()
        assert delta1.final_report == "Truncated preview..."

        # Carried forward
        delta2 = tracker.get_persistence_delta()
        assert delta2.final_report == "Truncated preview..."
