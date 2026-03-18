"""Tests for event types and discriminated union deserialization."""

from __future__ import annotations

from pydantic import TypeAdapter

from databricks_deep_research.events.types import (
    AgentOutputEvent,
    AgentStreamChunkEvent,
    CoordinatorClassifiedEvent,
    FrameworkEvent,
    ItemStartedEvent,
    NodeCompletedEvent,
    NodeStartedEvent,
    PlanCreatedEvent,
    WorkflowCompletedEvent,
    WorkflowFailedEvent,
    WorkflowStartedEvent,
)

# TypeAdapter for discriminated union deserialization
_event_adapter = TypeAdapter(FrameworkEvent)


class TestEventConstruction:
    def test_node_started(self) -> None:
        evt = NodeStartedEvent(
            node_id="n1", timestamp="2025-01-01T00:00:00Z",
            node_type="agent", label="Researcher",
        )
        assert evt.event_type == "node_started"
        assert evt.node_id == "n1"

    def test_node_completed(self) -> None:
        evt = NodeCompletedEvent(
            node_id="n1", timestamp="2025-01-01T00:00:00Z",
            duration_ms=1234.5,
        )
        assert evt.event_type == "node_completed"
        assert evt.duration_ms == 1234.5

    def test_agent_output(self) -> None:
        evt = AgentOutputEvent(
            node_id="a1", timestamp="t",
            output_key="report", output_preview="The report...",
        )
        assert evt.event_type == "agent_output"

    def test_agent_stream_chunk(self) -> None:
        evt = AgentStreamChunkEvent(
            node_id="s1", timestamp="t",
            chunk="Hello", subtype="synthesizer",
        )
        assert evt.subtype == "synthesizer"

    def test_plan_created(self) -> None:
        evt = PlanCreatedEvent(
            node_id="p1", timestamp="t",
            plan_id="plan_1", title="Research Plan",
            thought="Let's investigate", steps=[{"query": "test"}],
            has_enough_context=False,
        )
        assert evt.event_type == "plan_created"
        assert len(evt.steps) == 1

    def test_coordinator_classified(self) -> None:
        evt = CoordinatorClassifiedEvent(
            node_id="c1", timestamp="t",
            complexity="deep", recommended_depth="extended",
            direct_response=None, follow_up_type="research",
        )
        assert evt.event_type == "coordinator_classified"
        assert evt.direct_response is None

    def test_item_started(self) -> None:
        evt = ItemStartedEvent(
            node_id="pe1", timestamp="t",
            item_index=0, item_summary="Search for X", total_items=5,
        )
        assert evt.event_type == "item_started"
        assert evt.total_items == 5

    def test_workflow_completed(self) -> None:
        evt = WorkflowCompletedEvent(
            node_id="root", timestamp="t",
            workflow_id="w1", duration_ms=5000.0, total_tokens=10000,
            final_report="Done", total_sources=15, total_steps_executed=3,
        )
        assert evt.total_sources == 15

    def test_workflow_failed(self) -> None:
        evt = WorkflowFailedEvent(
            node_id="root",
            timestamp="t",
            workflow_id="w1",
            duration_ms=2500.0,
            error_type="RuntimeError",
            error_message="boom",
            total_sources=2,
            total_steps_executed=1,
        )
        assert evt.error_type == "RuntimeError"
        assert evt.total_sources == 2


class TestDiscriminatedUnion:
    def test_deserialize_node_started(self) -> None:
        data = {
            "event_type": "node_started",
            "node_id": "n1",
            "timestamp": "t",
            "node_type": "agent",
            "label": "Test",
        }
        evt = _event_adapter.validate_python(data)
        assert isinstance(evt, NodeStartedEvent)
        assert evt.label == "Test"

    def test_deserialize_workflow_started(self) -> None:
        data = {
            "event_type": "workflow_started",
            "node_id": "root",
            "timestamp": "t",
            "workflow_id": "w1",
            "workflow_name": "Test Workflow",
        }
        evt = _event_adapter.validate_python(data)
        assert isinstance(evt, WorkflowStartedEvent)

    def test_deserialize_plan_created(self) -> None:
        data = {
            "event_type": "plan_created",
            "node_id": "p1",
            "timestamp": "t",
            "plan_id": "plan_1",
            "title": "Plan",
            "thought": "Thinking...",
            "steps": [{"query": "test"}],
        }
        evt = _event_adapter.validate_python(data)
        assert isinstance(evt, PlanCreatedEvent)

    def test_deserialize_workflow_failed(self) -> None:
        data = {
            "event_type": "workflow_failed",
            "node_id": "root",
            "timestamp": "t",
            "workflow_id": "w1",
            "duration_ms": 2500.0,
            "error_type": "RuntimeError",
            "error_message": "boom",
            "total_sources": 2,
            "total_steps_executed": 1,
        }
        evt = _event_adapter.validate_python(data)
        assert isinstance(evt, WorkflowFailedEvent)

    def test_roundtrip_via_json(self) -> None:
        original = AgentOutputEvent(
            node_id="a1", timestamp="2025-01-01T00:00:00Z",
            output_key="report", output_preview="Preview",
        )
        json_str = original.model_dump_json()
        restored = _event_adapter.validate_json(json_str)
        assert isinstance(restored, AgentOutputEvent)
        assert restored.output_key == "report"
