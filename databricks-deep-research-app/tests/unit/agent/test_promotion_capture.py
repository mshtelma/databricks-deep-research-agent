"""Tests for app-side promotion trace capture (spec feature 6.1)."""

from __future__ import annotations

from databricks_deep_research.events.types import (
    NodeStartedEvent,
    StreamEvent,
    ToolCallEvent,
)

from deep_research.agent.promotion_capture import PromotionTraceCollector

TS = "2026-01-01T00:00:00Z"


def test_collector_builds_trace_dict() -> None:
    collector = PromotionTraceCollector(run_id="rs1")
    collector.observe(
        NodeStartedEvent(node_id="a1", timestamp=TS, node_type="agent", label="R")
    )
    collector.observe(
        ToolCallEvent(
            node_id="a1", timestamp=TS, tool_name="web_search", arguments={"q": "x"}
        )
    )
    trace = collector.build(query_shape="hello")
    assert trace is not None
    assert trace["run_id"] == "rs1"
    assert trace["query_shape"] == "hello"
    assert len(trace["steps"]) == 2
    assert trace["steps"][1]["tool_name"] == "web_search"
    assert trace["steps"][1]["arg_shape"] == {"q": "str"}


def test_collector_returns_none_when_no_structural_steps() -> None:
    # A simple/degenerate run observes no structural steps → nothing to persist.
    collector = PromotionTraceCollector(run_id="rs1")
    assert collector.build(query_shape="x") is None


def test_collector_observe_is_fail_soft() -> None:
    collector = PromotionTraceCollector(run_id="rs1")
    # An unknown/odd event must never raise out of the hot loop.
    collector.observe(StreamEvent(event_type="weird", node_id="x", timestamp=TS))
    assert collector.build() is None


def test_build_state_proxy_carries_promotion_trace() -> None:
    # The state proxy threads promotion_trace through to the persistence layer.
    from deep_research.agent.framework_orchestrator import _build_state_proxy
    from deep_research.agent.orchestration_config import OrchestrationConfig

    config = OrchestrationConfig()
    trace = {"schema_version": 1, "run_id": "rs1", "steps": []}
    proxy = _build_state_proxy(config, "report", None, promotion_trace=trace)
    assert proxy.promotion_trace == trace

    # And defaults to None when not provided (legacy callers unaffected).
    proxy_default = _build_state_proxy(config, "report", None)
    assert proxy_default.promotion_trace is None
