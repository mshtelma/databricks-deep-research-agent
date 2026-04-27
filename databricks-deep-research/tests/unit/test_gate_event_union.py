"""Gate events round-trip through the FrameworkEvent discriminated union."""

from __future__ import annotations

from databricks_deep_research.events.hitl import (
    GateDeniedEvent,
    GateResumedEvent,
    GateTimeoutEvent,
    GateWaitingEvent,
)


def test_gate_waiting_event_discriminator() -> None:
    e = GateWaitingEvent(
        request_id="r1", tool_name="commit_to_delta", arguments={"x": 1}, reason="prod"
    )
    payload = e.model_dump()
    assert payload["event_type"] == "gate_waiting"
    assert payload["tool_name"] == "commit_to_delta"


def test_gate_resumed_event() -> None:
    e = GateResumedEvent(request_id="r1", approver="alice")
    assert e.event_type == "gate_resumed"
    assert e.approver == "alice"


def test_gate_denied_event() -> None:
    e = GateDeniedEvent(request_id="r1", reason="too risky", approver="bob")
    assert e.event_type == "gate_denied"
    assert e.reason == "too risky"


def test_gate_timeout_event() -> None:
    e = GateTimeoutEvent(request_id="r1", elapsed_seconds=300.0)
    assert e.event_type == "gate_timeout"
    assert e.elapsed_seconds == 300.0


def test_event_type_unique_per_class() -> None:
    types = {
        GateWaitingEvent.model_fields["event_type"].default,
        GateResumedEvent.model_fields["event_type"].default,
        GateDeniedEvent.model_fields["event_type"].default,
        GateTimeoutEvent.model_fields["event_type"].default,
    }
    assert len(types) == 4


def test_all_gate_events_serialize_to_json() -> None:
    for event in [
        GateWaitingEvent(request_id="r", tool_name="x"),
        GateResumedEvent(request_id="r"),
        GateDeniedEvent(request_id="r"),
        GateTimeoutEvent(request_id="r"),
    ]:
        json_str = event.model_dump_json()
        assert event.event_type in json_str
