"""Tests for promotion-grade trace capture (spec feature 6.1).

Covers the load-bearing guarantees: argument *shapes* (never values) are
recorded, container/noise events are ignored, status/output annotations land,
the step cap is graceful, JSONB round-trips are forward-compatible, and
``observe`` never raises.
"""

from __future__ import annotations

import json

from databricks_deep_research.events.types import (
    AgentOutputEvent,
    AgentStreamChunkEvent,
    BranchSelectedEvent,
    CoordinatorClassifiedEvent,
    LoopIterationEvent,
    NodeCompletedEvent,
    NodeStartedEvent,
    StreamEvent,
    SynthesisStartedEvent,
    TokenUsageEvent,
    ToolCallEvent,
)
from databricks_deep_research.promotion import (
    PromotionTrace,
    PromotionTraceBuilder,
    StepKind,
    extract_promotion_trace,
)
from databricks_deep_research.promotion.trace_model import _arg_shape

TS = "2026-01-01T00:00:00Z"


def test_arg_shape_records_types_not_values() -> None:
    secret = "TOP-SECRET-" + "x" * 5000
    shape = _arg_shape(
        {"q": secret, "cfg": {"a": 1}, "empty": None, "n": 7, "flag": True}
    )
    assert shape == {
        "q": "str",
        "cfg": "dict",
        "empty": "NoneType",
        "n": "int",
        "flag": "bool",
    }
    assert "TOP-SECRET" not in json.dumps(shape)


def test_arg_shape_non_dict_returns_empty() -> None:
    assert _arg_shape(None) == {}
    assert _arg_shape("nope") == {}
    assert _arg_shape(["a", "b"]) == {}


def test_tool_call_becomes_step_without_value_leak() -> None:
    secret = "leak-me-1234567890"
    ev = ToolCallEvent(
        node_id="n1",
        timestamp=TS,
        tool_name="web_search",
        arguments={"query": secret, "k": 5},
    )
    trace = extract_promotion_trace([ev], run_id="r1", query_shape="q")
    assert len(trace.steps) == 1
    step = trace.steps[0]
    assert step.kind == StepKind.TOOL
    assert step.tool_name == "web_search"
    assert step.arg_shape == {"query": "str", "k": "int"}
    # No raw argument value anywhere in the persisted form.
    assert secret not in trace.model_dump_json()


def test_node_started_agent_is_step_container_skipped() -> None:
    agent = NodeStartedEvent(
        node_id="a1", timestamp=TS, node_type="agent", label="Researcher"
    )
    container = NodeStartedEvent(
        node_id="seq", timestamp=TS, node_type="sequence", label="Pipeline"
    )
    trace = extract_promotion_trace([container, agent])
    assert [s.kind for s in trace.steps] == [StepKind.AGENT]
    assert trace.steps[0].node_id == "a1"
    assert trace.steps[0].label == "Researcher"


def test_branch_and_loop_steps() -> None:
    br = BranchSelectedEvent(
        node_id="cond", timestamp=TS, branch_index=2, condition_summary="x>0"
    )
    lp = LoopIterationEvent(node_id="loop", timestamp=TS, iteration=3, max_iterations=5)
    trace = extract_promotion_trace([br, lp])
    assert trace.steps[0].kind == StepKind.DECISION
    assert trace.steps[0].branch_taken == 2
    assert trace.steps[1].kind == StepKind.LOOP
    assert trace.steps[1].loop_iteration == 3


def test_node_completed_annotates_status() -> None:
    agent = NodeStartedEvent(node_id="a1", timestamp=TS, node_type="agent", label="R")
    done = NodeCompletedEvent(
        node_id="a1", timestamp=TS, duration_ms=12.0, status="failed"
    )
    trace = extract_promotion_trace([agent, done])
    assert trace.steps[0].status == "failed"


def test_agent_output_annotates_produced_key() -> None:
    agent = NodeStartedEvent(node_id="a1", timestamp=TS, node_type="agent", label="R")
    out = AgentOutputEvent(
        node_id="a1", timestamp=TS, output_key="findings", output_preview="..."
    )
    trace = extract_promotion_trace([agent, out])
    assert trace.steps[0].produced_key == "findings"


def test_token_usage_takes_max() -> None:
    e1 = TokenUsageEvent(
        node_id="n",
        timestamp=TS,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        budget_remaining=-1,
    )
    e2 = TokenUsageEvent(
        node_id="n",
        timestamp=TS,
        prompt_tokens=20,
        completion_tokens=10,
        total_tokens=30,
        budget_remaining=-1,
    )
    trace = extract_promotion_trace([e1, e2])
    assert trace.total_tokens == 30
    assert trace.steps == []


def test_coordinator_simple_marks_degenerate() -> None:
    ev = CoordinatorClassifiedEvent(
        node_id="c",
        timestamp=TS,
        complexity="low",
        recommended_depth="light",
        is_simple=True,
    )
    trace = extract_promotion_trace([ev])
    assert trace.is_degenerate is True


def test_noise_and_unknown_events_ignored() -> None:
    chunk = AgentStreamChunkEvent(node_id="s", timestamp=TS, chunk="hello")
    unknown = StreamEvent(event_type="some_future_event", node_id="x", timestamp=TS)
    trace = extract_promotion_trace([chunk, unknown])
    assert trace.steps == []
    assert trace.captured_event_count == 0


def test_synthesis_step() -> None:
    ev = SynthesisStartedEvent(
        node_id="syn", timestamp=TS, total_observations=3, total_sources=4
    )
    trace = extract_promotion_trace([ev])
    assert trace.steps[0].kind == StepKind.SYNTHESIS


def test_max_steps_cap_records_dropped() -> None:
    events = [
        ToolCallEvent(node_id=f"n{i}", timestamp=TS, tool_name="t", arguments={})
        for i in range(10)
    ]
    trace = extract_promotion_trace(events, max_steps=3)
    assert len(trace.steps) == 3
    assert trace.captured_event_count == 3
    assert trace.dropped_event_count == 7


def test_orders_are_contiguous() -> None:
    events: list[StreamEvent] = [
        NodeStartedEvent(node_id="a1", timestamp=TS, node_type="agent", label="A"),
        ToolCallEvent(node_id="a1", timestamp=TS, tool_name="t", arguments={"q": "x"}),
        BranchSelectedEvent(
            node_id="c", timestamp=TS, branch_index=0, condition_summary="s"
        ),
    ]
    trace = extract_promotion_trace(events)
    assert [s.order for s in trace.steps] == [0, 1, 2]


def test_jsonb_roundtrip_is_forward_compatible() -> None:
    agent = NodeStartedEvent(node_id="a1", timestamp=TS, node_type="agent", label="R")
    trace = extract_promotion_trace([agent], run_id="r1", query_shape="q")
    dumped = trace.model_dump()
    dumped["unknown_future_field"] = {"x": 1}  # a newer writer added a field
    dumped["steps"][0]["unknown_step_field"] = 2
    restored = PromotionTrace.model_validate(dumped)
    assert restored.run_id == "r1"
    assert restored.steps[0].node_id == "a1"


def test_builder_observe_never_raises_on_unknown_event() -> None:
    builder = PromotionTraceBuilder()
    builder.observe(StreamEvent(event_type="weird", node_id="x", timestamp=TS))
    trace = builder.build(run_id="r")
    assert trace.steps == []


def test_query_shape_is_bounded() -> None:
    trace = extract_promotion_trace([], query_shape="z" * 1000)
    assert len(trace.query_shape) == 200
