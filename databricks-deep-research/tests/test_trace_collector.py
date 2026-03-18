from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from tests.trace_collector import (
    SpanNode,
    TraceInfo,
    TraceReport,
    _trace_health,
    generate_normalized_summary,
    generate_summary_markdown,
)


def _span(
    name: str,
    *,
    span_type: str = "CHAIN",
    attributes: dict[str, str] | None = None,
) -> SpanNode:
    return SpanNode(
        name=name,
        span_type=span_type,
        duration_ms=100.0,
        status="STATUS_CODE_OK",
        attributes=attributes or {},
    )


def test_trace_health_marks_tool_runs_without_sources_degraded() -> None:
    root = _span(
        "workflow.Classical Enterprise Vector Search Pipeline",
        attributes={
            "workflow.terminal_status": "completed",
            "workflow.duration_ms": "1000",
            "workflow.total_sources": "0",
            "workflow.total_sources_raw": "4",
            "workflow.total_sources_accepted": "0",
        },
    )
    tool = _span(
        "tool.vector_search",
        span_type="SpanType.TOOL",
        attributes={"tool.accepted_source_count": "0", "tool.raw_source_count": "4"},
    )

    health, reasons = _trace_health("OK", root, [root, tool])

    assert health == "DEGRADED"
    assert "tool_calls_without_accepted_sources" in reasons


def test_trace_health_zero_sources_without_tool_calls_is_ok() -> None:
    root = _span(
        "workflow.Single Agent Test",
        attributes={
            "workflow.terminal_status": "completed",
            "workflow.duration_ms": "1000",
            "workflow.total_sources": "0",
            "workflow.total_sources_raw": "0",
            "workflow.total_sources_accepted": "0",
        },
    )

    health, reasons = _trace_health("OK", root, [root])

    assert health == "OK"
    assert reasons == []


def test_trace_health_marks_missing_tools_degraded() -> None:
    root = _span(
        "workflow.Mixed Sources Research Pipeline",
        attributes={
            "workflow.terminal_status": "completed",
            "workflow.duration_ms": "1000",
            "workflow.total_sources": "0",
            "workflow.total_sources_raw": "0",
            "workflow.total_sources_accepted": "0",
            "workflow.missing_declared_tools": "2",
        },
    )
    tool = _span("tool.web_search", span_type="SpanType.TOOL")

    health, reasons = _trace_health("OK", root, [root, tool])

    assert health == "DEGRADED"
    assert "missing_declared_tools=2" in reasons


def test_trace_health_marks_zero_executed_plan_steps_degraded() -> None:
    root = _span(
        "workflow.Mixed Sources Research Pipeline",
        attributes={
            "workflow.terminal_status": "completed",
            "workflow.duration_ms": "1000",
            "workflow.total_sources": "1",
            "workflow.total_sources_raw": "2",
            "workflow.total_sources_accepted": "2",
            "workflow.total_steps_executed": "0",
            "workflow.plan_exit_reasons": '["empty_plan"]',
        },
    )
    plan = _span("plan_cycle_0", attributes={"plan.items_count": "0"})

    health, reasons = _trace_health("OK", root, [root, plan])

    assert health == "DEGRADED"
    assert "zero_executed_plan_steps" in reasons


def test_trace_health_allows_zero_executed_steps_when_planner_had_enough_context() -> None:
    root = _span(
        "workflow.Mixed Sources Research Pipeline",
        attributes={
            "workflow.terminal_status": "completed",
            "workflow.duration_ms": "1000",
            "workflow.total_sources": "1",
            "workflow.total_sources_raw": "1",
            "workflow.total_sources_accepted": "1",
            "workflow.total_steps_executed": "0",
            "workflow.plan_exit_reasons": '["planner_sufficient_context"]',
        },
    )
    plan = _span("plan_cycle_0", attributes={"plan.items_count": "0"})

    health, reasons = _trace_health("OK", root, [root, plan])

    assert health == "OK"
    assert reasons == []


def test_trace_health_marks_missing_terminal_metadata_error() -> None:
    root = _span(
        "workflow.Advanced Research",
        attributes={"workflow.total_sources": "0"},
    )
    tool = _span(
        "tool.web_search",
        span_type="SpanType.TOOL",
        attributes={"tool.accepted_source_count": "1", "tool.raw_source_count": "3"},
    )

    health, reasons = _trace_health("OK", root, [root, tool])

    assert health == "ERROR"
    assert any(reason.startswith("missing_terminal_metadata(") for reason in reasons)


def test_trace_health_marks_failed_terminal_status_error() -> None:
    root = _span(
        "workflow.Advanced Research",
        attributes={
            "workflow.terminal_status": "failed",
            "workflow.duration_ms": "1000",
            "workflow.total_sources": "0",
            "workflow.total_sources_raw": "3",
            "workflow.total_sources_accepted": "0",
            "workflow.error_type": "PermissionDeniedError",
        },
    )

    health, reasons = _trace_health("OK", root, [root])

    assert health == "ERROR"
    assert reasons == ["workflow_failed(PermissionDeniedError)"]


def test_generate_summary_markdown_uses_health_status_and_reasons() -> None:
    now = datetime.now(tz=UTC)
    root = _span(
        "workflow.Enterprise Research Pipeline",
        attributes={
            "workflow.terminal_status": "completed",
            "workflow.duration_ms": "1000",
            "workflow.total_sources": "0",
            "workflow.total_sources_raw": "2",
            "workflow.total_sources_accepted": "0",
        },
    )
    trace = TraceInfo(
        request_id="req-1",
        status="OK",
        health_status="DEGRADED",
        health_reasons=["tool_calls_without_accepted_sources"],
        duration_ms=1250.0,
        timestamp=now,
        root=root,
        all_spans=[root],
    )
    report = TraceReport(
        traces=[trace],
        session_start=now,
        session_end=now,
        output_dir=Path("test-traces/mock"),
    )

    markdown = generate_summary_markdown(report)

    assert "1 DEGRADED" in markdown
    assert "[DEGRADED]" in markdown
    assert "tool_calls_without_accepted_sources" in markdown


def test_non_workflow_trace_ok_for_missing_metadata() -> None:
    """Non-workflow traces (root name not 'workflow.*') get OK even without
    workflow.terminal_status and other workflow attributes."""
    root = _span("tool.web_search", attributes={})

    health, reasons = _trace_health("OK", root, [root])

    assert health == "OK"
    assert reasons == []


def test_non_workflow_trace_error_for_span_status() -> None:
    """Non-workflow traces with ERROR span status get ERROR health."""
    root = SpanNode(
        name="tool.vector_search",
        span_type="SpanType.TOOL",
        duration_ms=0.0,
        status="STATUS_CODE_ERROR",
        attributes={},
    )

    health, reasons = _trace_health("OK", root, [root])

    assert health == "ERROR"
    assert "span_status_error" in reasons


def test_workflow_trace_still_checks_metadata() -> None:
    """Workflow traces (root name 'workflow.*') still require terminal metadata."""
    root = _span(
        "workflow.Advanced Research",
        attributes={"workflow.total_sources": "0"},
    )

    health, reasons = _trace_health("OK", root, [root])

    assert health == "ERROR"
    assert any(reason.startswith("missing_terminal_metadata(") for reason in reasons)


def test_non_workflow_trace_with_terminal_status_treated_as_workflow() -> None:
    """A trace with workflow.terminal_status (but non-workflow name) is treated
    as a workflow trace for health classification."""
    root = _span(
        "custom_pipeline",
        attributes={
            "workflow.terminal_status": "completed",
            "workflow.duration_ms": "500",
            "workflow.total_sources": "1",
            "workflow.total_sources_raw": "1",
            "workflow.total_sources_accepted": "1",
        },
    )

    health, reasons = _trace_health("OK", root, [root])

    assert health == "OK"
    assert reasons == []


def test_generate_normalized_summary_structure() -> None:
    """generate_normalized_summary returns correct structure."""
    now = datetime.now(tz=UTC)
    root = _span(
        "workflow.Test Pipeline",
        attributes={
            "workflow.terminal_status": "completed",
            "workflow.duration_ms": "1000",
            "workflow.total_sources": "2",
            "workflow.total_sources_raw": "3",
            "workflow.total_sources_accepted": "2",
            "workflow.plan_exit_reasons": '["items_exhausted"]',
        },
    )
    tool = _span(
        "tool.web_search",
        span_type="SpanType.TOOL",
        attributes={
            "tool.source_kind": "web_search",
            "tool.accepted_source_count": "2",
        },
    )
    trace = TraceInfo(
        request_id="req-1",
        status="OK",
        health_status="OK",
        health_reasons=[],
        duration_ms=1000.0,
        timestamp=now,
        root=root,
        all_spans=[root, tool],
    )
    report = TraceReport(
        traces=[trace],
        session_start=now,
        session_end=now,
        output_dir=Path("test-traces/mock"),
    )

    summary = generate_normalized_summary(report)

    assert summary["total_traces"] == 1
    assert summary["health_summary"]["OK"] == 1
    assert len(summary["traces"]) == 1
    assert summary["traces"][0]["name"] == "workflow.Test Pipeline"
    assert summary["traces"][0]["health"] == "OK"
    assert summary["traces"][0]["tool_calls"] == 1
    assert summary["traces"][0]["source_families"]["web_search"] == 2
