"""Test trace harness — collect MLflow traces after test runs.

Provides:
- TraceCollector: configures MLflow, records timestamps, downloads traces
- SpanNode / TraceInfo: span tree reconstruction from flat MLflow span lists
- Report generation: summary.md, performance.md, traces.json
- Terminal summary: compact span tree printed to stdout

Usage (via conftest.py fixture):
    pytest tests/complex -v -s --collect-traces
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Credential detection (reuse pattern from complex/conftest.py)
# ---------------------------------------------------------------------------


def _has_databricks_creds() -> bool:
    return bool(os.getenv("DATABRICKS_TOKEN") or os.getenv("DATABRICKS_CONFIG_PROFILE"))


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class SpanNode:
    """A span with its children, for tree display."""

    name: str
    span_type: str
    duration_ms: float
    status: str
    attributes: dict[str, Any]
    children: list[SpanNode] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        """Extract total tokens from agent/react attributes."""
        for prefix in ("agent.", "react."):
            val = self.attributes.get(f"{prefix}total_tokens")
            if val is not None:
                try:
                    return int(str(val).strip('"'))
                except (ValueError, TypeError):
                    continue
        return 0

    @property
    def tool_calls(self) -> int:
        val = self.attributes.get("react.total_calls")
        if val is not None:
            try:
                return int(str(val).strip('"'))
            except (ValueError, TypeError):
                pass
        return 0


@dataclass
class TraceInfo:
    """Parsed trace with span tree."""

    request_id: str
    status: str
    health_status: str
    health_reasons: list[str]
    duration_ms: float
    timestamp: datetime
    root: SpanNode | None
    all_spans: list[SpanNode]


@dataclass
class TraceReport:
    """Complete report from a test session."""

    traces: list[TraceInfo]
    session_start: datetime
    session_end: datetime
    output_dir: Path


# ---------------------------------------------------------------------------
# Span tree building
# ---------------------------------------------------------------------------


def _build_span_tree(spans: list[Any]) -> tuple[SpanNode | None, list[SpanNode]]:
    """Build a parent→child tree from a flat list of MLflow Span objects or dicts.

    Handles both dict-form spans (from DataFrame/JSON) and object-form spans
    (from MLflow Trace objects), using dual-access with multiple field name variants.

    Returns (root_node, all_nodes).
    """
    nodes: dict[str, SpanNode] = {}
    parent_map: dict[str, str] = {}

    def _g(span: Any, key: str, *alt_keys: str, default: Any = None) -> Any:
        """Dual-access helper: tries multiple key names on dicts or objects."""
        for k in (key, *alt_keys):
            val = span.get(k) if isinstance(span, dict) else getattr(span, k, None)
            if val is not None:
                return val
        return default

    for span in spans:
        span_id = _g(span, "span_id", default=str(id(span)))
        parent_id = _g(span, "parent_span_id", "parent_id")

        # Duration from nanosecond timestamps
        start_ns = _g(span, "start_time_unix_nano", "start_time_ns", default=0)
        end_ns = _g(span, "end_time_unix_nano", "end_time_ns", default=0)
        duration_ms = (end_ns - start_ns) / 1_000_000 if end_ns > start_ns else 0.0

        # Status: handle dict {"code": "STATUS_CODE_OK"} and object forms
        status_raw = _g(span, "status")
        if isinstance(status_raw, dict):
            status = status_raw.get("code", "UNKNOWN")
        elif status_raw is not None:
            status = str(getattr(status_raw, "status_code", status_raw))
        else:
            status = "UNKNOWN"

        # Attributes
        raw_attrs = _g(span, "attributes", default={})
        attrs = dict(raw_attrs) if isinstance(raw_attrs, dict) else {}

        # Span type: direct field or from attributes
        span_type = _g(span, "span_type") or attrs.get("mlflow.spanType", "CHAIN")

        node = SpanNode(
            name=_g(span, "name", default="unknown"),
            span_type=str(span_type).strip('"'),
            duration_ms=duration_ms,
            status=status,
            attributes=attrs,
        )
        nodes[span_id] = node
        if parent_id:
            parent_map[span_id] = parent_id

    # Link children to parents
    all_nodes = list(nodes.values())
    root: SpanNode | None = None

    for span_id, node in nodes.items():
        pid = parent_map.get(span_id)
        if pid and pid in nodes:
            nodes[pid].children.append(node)
        else:
            # Root node (no parent or parent not in this trace).
            # If multiple roots, pick the longest duration.
            if root is None or node.duration_ms > root.duration_ms:
                root = node

    return root, all_nodes


# ---------------------------------------------------------------------------
# JSON serialization helper
# ---------------------------------------------------------------------------


def _serialize_value(val: Any) -> Any:
    """Make a value JSON-serializable."""
    if isinstance(val, str | int | float | bool | type(None)):
        return val
    if isinstance(val, list | tuple):
        return [_serialize_value(v) for v in val]
    if isinstance(val, dict):
        return {str(k): _serialize_value(v) for k, v in val.items()}
    if isinstance(val, datetime):
        return val.isoformat()
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    return str(val)


def _strip_quotes(value: str) -> str:
    return value.strip().strip('"').strip("'")


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(_strip_quotes(str(value))))
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    lowered = _strip_quotes(str(value)).lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    return None


def _trace_health(trace_status: str, root: SpanNode | None, all_spans: list[SpanNode]) -> tuple[str, list[str]]:
    """Classify trace health from workflow + tool attributes."""
    if "OK" not in trace_status.upper():
        return "ERROR", [f"trace_state={trace_status}"]

    root_attrs = root.attributes if root else {}

    # Classify by root span kind — non-workflow traces don't have workflow metadata
    root_name = (root.name or "") if root else ""
    is_workflow_trace = (
        root_name.startswith("workflow")
        or root_attrs.get("workflow.terminal_status") is not None
    )
    if not is_workflow_trace:
        # Non-workflow traces: classify by span status only
        if root and "ERROR" in str(getattr(root, "status", "")).upper():
            return "ERROR", ["span_status_error"]
        return "OK", []

    reasons: list[str] = []
    terminal_status = _strip_quotes(str(root_attrs.get("workflow.terminal_status", ""))).lower()
    duration_ms = _as_int(root_attrs.get("workflow.duration_ms"))
    terminal_sources = _as_int(root_attrs.get("workflow.total_sources"))
    terminal_sources_raw = _as_int(root_attrs.get("workflow.total_sources_raw"))
    terminal_sources_accepted = _as_int(root_attrs.get("workflow.total_sources_accepted"))
    blocked_steps = _as_int(root_attrs.get("workflow.blocked_steps")) or 0
    missing_declared_tools = _as_int(root_attrs.get("workflow.missing_declared_tools")) or 0
    total_steps_executed = _as_int(root_attrs.get("workflow.total_steps_executed")) or 0
    raw_plan_exit_reasons = root_attrs.get("workflow.plan_exit_reasons", "[]")
    plan_exit_reasons: list[str] = []
    if isinstance(raw_plan_exit_reasons, str):
        try:
            parsed_reasons = json.loads(_strip_quotes(raw_plan_exit_reasons))
            if isinstance(parsed_reasons, list):
                plan_exit_reasons = [str(reason) for reason in parsed_reasons]
        except json.JSONDecodeError:
            plan_exit_reasons = []

    terminal_attrs_present = (
        duration_ms is not None
        and terminal_sources is not None
        and terminal_sources_raw is not None
        and terminal_sources_accepted is not None
    )
    if terminal_status == "failed":
        error_type = _strip_quotes(str(root_attrs.get("workflow.error_type", ""))) or "unknown"
        return "ERROR", [f"workflow_failed({error_type})"]
    if terminal_status not in {"completed", "cancelled"} or not terminal_attrs_present:
        missing_fields: list[str] = []
        if duration_ms is None:
            missing_fields.append("workflow.duration_ms")
        if terminal_sources is None:
            missing_fields.append("workflow.total_sources")
        if terminal_sources_raw is None:
            missing_fields.append("workflow.total_sources_raw")
        if terminal_sources_accepted is None:
            missing_fields.append("workflow.total_sources_accepted")
        if terminal_status not in {"completed", "cancelled"}:
            missing_fields.append("workflow.terminal_status")
        return "ERROR", [f"missing_terminal_metadata({','.join(missing_fields)})"]

    total_sources_accepted = terminal_sources_accepted
    if total_sources_accepted is None:
        total_sources_accepted = _as_int(root_attrs.get("workflow.total_sources"))
    total_sources_raw = terminal_sources_raw

    tool_spans = [
        span for span in all_spans
        if span.span_type == "SpanType.TOOL" or span.name.startswith("tool.")
    ]
    plan_spans_present = any(span.name.startswith("plan_cycle_") for span in all_spans)
    tool_calls = len(tool_spans)
    accepted_from_tools = sum(
        _as_int(span.attributes.get("tool.accepted_source_count")) or 0
        for span in tool_spans
    )
    raw_from_tools = sum(
        _as_int(span.attributes.get("tool.raw_source_count")) or 0
        for span in tool_spans
    )
    failed_tool_calls = sum(
        1
        for span in tool_spans
        if _as_bool(span.attributes.get("tool.success")) is False
    )

    if missing_declared_tools > 0:
        reasons.append(f"missing_declared_tools={missing_declared_tools}")
    if blocked_steps > 0:
        reasons.append(f"blocked_steps={blocked_steps}")
    if plan_spans_present and total_steps_executed == 0:
        if plan_exit_reasons and all(reason == "planner_sufficient_context" for reason in plan_exit_reasons):
            pass
        else:
            reasons.append("zero_executed_plan_steps")

    accepted_total = total_sources_accepted
    if accepted_total is None and accepted_from_tools > 0:
        accepted_total = accepted_from_tools

    # Only treat zero sources as degraded when the workflow actually attempted tool work.
    if tool_calls > 0 and (accepted_total or 0) <= 0:
        reasons.append("tool_calls_without_accepted_sources")

    if tool_calls > 0 and failed_tool_calls == tool_calls and failed_tool_calls > 0:
        reasons.append("all_tool_calls_failed")

    if total_sources_raw is not None and total_sources_raw > 0 and (accepted_total or 0) <= 0:
        reasons.append(
            f"all_sources_rejected(raw={total_sources_raw},accepted={accepted_total or 0})"
        )
    elif raw_from_tools > 0 and accepted_from_tools <= 0:
        reasons.append(
            f"all_sources_rejected(raw={raw_from_tools},accepted={accepted_from_tools})"
        )

    if reasons:
        deduped: list[str] = []
        for reason in reasons:
            if reason not in deduped:
                deduped.append(reason)
        return "DEGRADED", deduped

    return "OK", []


# ---------------------------------------------------------------------------
# TraceCollector
# ---------------------------------------------------------------------------


class TraceCollector:
    """Configures MLflow, records timestamps, downloads and reports on traces."""

    def __init__(self, output_dir: str = "test-traces") -> None:
        self._output_dir = output_dir
        self._start_time: datetime | None = None
        self._experiment_id: str | None = None

    def setup_mlflow(self) -> bool:
        """Configure MLflow tracking. Returns True if successful."""
        if not _has_databricks_creds():
            return False

        try:
            from databricks_deep_research.tracing import setup_mlflow_tracing

            experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME")
            if not experiment_name:
                logger.warning(
                    "MLFLOW_EXPERIMENT_NAME not set — add it to .env.test"
                )
                return False

            if not setup_mlflow_tracing(experiment_name=experiment_name):
                return False

            # experiment_id needed by _search_traces() for filter query
            import mlflow

            exp = mlflow.get_experiment_by_name(experiment_name)
            self._experiment_id = exp.experiment_id if exp else None
            return True
        except Exception as e:
            logger.warning("MLflow setup failed: %s", e)
            return False

    def start(self) -> None:
        """Record session start time."""
        self._start_time = datetime.now(tz=UTC)

    def collect(self) -> TraceReport | None:
        """Flush MLflow, search traces, build report."""
        if not self._experiment_id or not self._start_time:
            return None

        # 1. Flush async traces
        self._flush()

        # 2. Search traces created after start_time
        traces_data = self._search_traces()
        if traces_data is None:
            return None

        # 3. Parse into TraceInfo objects
        traces = self._parse_traces(traces_data)

        # 4. Build report
        session_end = datetime.now(tz=UTC)
        output_dir = self._make_output_dir(session_end)
        report = TraceReport(
            traces=traces,
            session_start=self._start_time,
            session_end=session_end,
            output_dir=output_dir,
        )

        # 5. Save outputs
        self._save_traces_json(traces_data, output_dir)
        self._save_summary(report, output_dir)
        self._save_performance(report, output_dir)
        self._save_normalized_summary(report, output_dir)

        return report

    # -- Internal methods ---------------------------------------------------

    def _flush(self) -> None:
        """Flush async MLflow trace logging."""
        try:
            from databricks_deep_research.tracing import shutdown_mlflow_tracing

            shutdown_mlflow_tracing()
        except Exception as e:
            logger.debug("MLflow flush failed: %s", e)
        # Allow time for traces to propagate to backend before searching
        time.sleep(3)

    def _search_traces(self) -> Any | None:
        """Search for traces created during this test session."""
        assert self._start_time is not None
        try:
            import mlflow

            # search_traces returns a pandas DataFrame
            start_ms = int(self._start_time.timestamp() * 1000)
            traces_df = mlflow.search_traces(
                locations=[self._experiment_id],
                filter_string=f"timestamp_ms >= {start_ms}",
            )
            if traces_df is None or (hasattr(traces_df, "empty") and traces_df.empty):
                logger.info("No traces found for this test session")
                return None
            return traces_df
        except Exception as e:
            logger.warning("Trace search failed: %s", e)
            return None

    def _parse_traces(self, traces_df: Any) -> list[TraceInfo]:
        """Parse DataFrame rows into TraceInfo objects with span trees."""
        traces: list[TraceInfo] = []

        for _, row in traces_df.iterrows():
            request_id = str(row.get("request_id", "unknown"))
            status = str(row.get("state", "UNKNOWN"))

            # Duration
            duration_ms = 0.0
            raw_duration = row.get("execution_duration")
            if raw_duration is not None:
                with contextlib.suppress(ValueError, TypeError):
                    duration_ms = float(raw_duration)

            # Timestamp
            timestamp = self._start_time or datetime.now(tz=UTC)
            raw_ts = row.get("timestamp_ms")
            if raw_ts is not None:
                with contextlib.suppress(ValueError, TypeError, OSError):
                    timestamp = datetime.fromtimestamp(int(raw_ts) / 1000, tz=UTC)

            # Build span tree
            root, all_spans = self._extract_spans(row, request_id)
            health_status, health_reasons = _trace_health(status, root, all_spans)

            # Use root span duration when execution_duration is 0
            if duration_ms == 0.0 and root is not None and root.duration_ms > 0:
                duration_ms = root.duration_ms

            traces.append(
                TraceInfo(
                    request_id=request_id,
                    status=status,
                    health_status=health_status,
                    health_reasons=health_reasons,
                    duration_ms=duration_ms,
                    timestamp=timestamp,
                    root=root,
                    all_spans=all_spans,
                )
            )

        # Sort by timestamp
        traces.sort(key=lambda t: t.timestamp)
        return traces

    def _extract_spans(
        self, row: Any, request_id: str
    ) -> tuple[SpanNode | None, list[SpanNode]]:
        """Extract spans from a trace row, trying multiple approaches."""
        # Approach 1: Pre-parsed "spans" column (list of dicts, most common)
        raw_spans = row.get("spans")
        if isinstance(raw_spans, list) and raw_spans:
            return _build_span_tree(raw_spans)

        # Approach 2: Parse "trace" JSON string
        trace_val = row.get("trace")
        if isinstance(trace_val, str):
            try:
                parsed = json.loads(trace_val)
                spans = parsed.get("data", {}).get("spans", [])
                if spans:
                    return _build_span_tree(spans)
            except (json.JSONDecodeError, AttributeError, KeyError):
                pass

        # Approach 3: MLflow Trace object (some MLflow versions return objects)
        if trace_val is not None and not isinstance(trace_val, str):
            data = getattr(trace_val, "data", None)
            if data is not None:
                spans = getattr(data, "spans", None)
                if spans:
                    return _build_span_tree(spans)

        # Approach 4: MlflowClient fallback
        try:
            from mlflow import MlflowClient

            client = MlflowClient()
            trace = client.get_trace(request_id)
            if trace is not None:
                data = getattr(trace, "data", None)
                if data is not None:
                    spans = getattr(data, "spans", None)
                    if spans:
                        return _build_span_tree(spans)
        except Exception as e:
            logger.debug("get_trace(%s) failed: %s", request_id, e)

        return None, []

    def _make_output_dir(self, session_end: datetime) -> Path:
        """Create timestamped output directory."""
        ts = session_end.strftime("%Y-%m-%d_%H-%M-%S")
        out = Path(self._output_dir) / ts
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _save_traces_json(self, traces_df: Any, output_dir: Path) -> None:
        """Save raw trace data as JSON."""
        try:
            records: list[dict[str, Any]] = []
            for _, row in traces_df.iterrows():
                record: dict[str, Any] = {}
                for col in traces_df.columns:
                    val = row.get(col)
                    record[col] = _serialize_value(val)
                records.append(record)

            path = output_dir / "traces.json"
            path.write_text(json.dumps(records, indent=2, default=str))
        except Exception as e:
            logger.warning("Failed to save traces.json: %s", e)

    def _save_summary(self, report: TraceReport, output_dir: Path) -> None:
        """Save summary.md report."""
        try:
            md = generate_summary_markdown(report)
            (output_dir / "summary.md").write_text(md)
        except Exception as e:
            logger.warning("Failed to save summary.md: %s", e)

    def _save_performance(self, report: TraceReport, output_dir: Path) -> None:
        """Save performance.md report."""
        try:
            md = generate_performance_markdown(report)
            (output_dir / "performance.md").write_text(md)
        except Exception as e:
            logger.warning("Failed to save performance.md: %s", e)

    def _save_normalized_summary(self, report: TraceReport, output_dir: Path) -> None:
        """Save trace-summary.json with per-trace metrics and source coverage."""
        try:
            summary = generate_normalized_summary(report)
            (output_dir / "trace-summary.json").write_text(
                json.dumps(summary, indent=2, default=str)
            )
        except Exception as e:
            logger.warning("Failed to save trace-summary.json: %s", e)

    def print_terminal_summary(self, report: TraceReport) -> None:
        """Print compact tree summary to stdout."""
        total_tokens = sum(
            span.tokens for t in report.traces for span in t.all_spans
        )
        ok_count = sum(1 for t in report.traces if t.health_status == "OK")
        degraded_count = sum(1 for t in report.traces if t.health_status == "DEGRADED")
        err_count = sum(1 for t in report.traces if t.health_status == "ERROR")
        duration_s = (report.session_end - report.session_start).total_seconds()

        print()
        print("=" * 50)
        print(" TRACE COLLECTION")
        print("=" * 50)
        print(
            f" Session: {report.session_start:%H:%M:%S} -> "
            f"{report.session_end:%H:%M:%S} UTC ({duration_s:.0f}s)"
        )
        print(
            f" Traces: {len(report.traces)} "
            f"({ok_count} OK, {degraded_count} DEGRADED, {err_count} ERROR) | "
            f"{total_tokens:,} tokens"
        )
        print()

        for i, trace in enumerate(report.traces, 1):
            root_name = trace.root.name if trace.root else "unknown"
            status_tag = trace.health_status
            print(
                f" Trace {i}: {root_name} [{status_tag}] "
                f"{trace.duration_ms / 1000:.1f}s"
            )
            if trace.health_reasons:
                print(f"   reasons: {', '.join(trace.health_reasons)}")
            if trace.root:
                _print_span_tree(trace.root, indent=2, is_last=True, prefix="")
            print()

        print(f" Reports: {report.output_dir}/")
        print("=" * 50)
        print()


# ---------------------------------------------------------------------------
# Report generators
# ---------------------------------------------------------------------------


def generate_summary_markdown(report: TraceReport) -> str:
    """Generate summary.md with overview table and span trees."""
    total_tokens = sum(
        span.tokens for t in report.traces for span in t.all_spans
    )
    ok_count = sum(1 for t in report.traces if t.health_status == "OK")
    degraded_count = sum(1 for t in report.traces if t.health_status == "DEGRADED")
    err_count = sum(1 for t in report.traces if t.health_status == "ERROR")
    duration_s = (report.session_end - report.session_start).total_seconds()

    lines = [
        "# Test Trace Report",
        "",
        f"- **Session**: {report.session_start:%Y-%m-%d %H:%M:%S} -> "
        f"{report.session_end:%H:%M:%S} UTC ({duration_s:.0f}s)",
        f"- **Traces**: {len(report.traces)} ({ok_count} OK, {degraded_count} DEGRADED, {err_count} ERROR)",
        f"- **Total tokens**: {total_tokens:,}",
        "",
    ]

    for i, trace in enumerate(report.traces, 1):
        root_name = trace.root.name if trace.root else "unknown"
        status_tag = trace.health_status
        lines.append(
            f"## Trace {i}: {root_name} [{status_tag}] "
            f"{trace.duration_ms / 1000:.1f}s"
        )
        lines.append("")
        if trace.health_reasons:
            lines.append(f"Health reasons: {', '.join(trace.health_reasons)}")
            lines.append("")
        lines.append("| Span | Duration | Tokens | Status |")
        lines.append("|------|----------|--------|--------|")

        if trace.root:
            _collect_span_table_rows(trace.root, lines, depth=0)

        lines.append("")

    return "\n".join(lines)


def generate_normalized_summary(report: TraceReport) -> dict[str, Any]:
    """Compact JSON summary with per-trace metrics and source family coverage."""
    traces: list[dict[str, Any]] = []
    for trace_info in report.traces:
        root_attrs = trace_info.root.attributes if trace_info.root else {}
        tool_spans = [
            s for s in trace_info.all_spans
            if s.span_type == "SpanType.TOOL" or s.name.startswith("tool.")
        ]
        source_families: dict[str, int] = {}
        for span in tool_spans:
            kind = str(span.attributes.get("tool.source_kind", "unknown"))
            accepted = _as_int(span.attributes.get("tool.accepted_source_count")) or 0
            source_families[kind] = source_families.get(kind, 0) + accepted

        raw_exit = str(root_attrs.get("workflow.plan_exit_reasons", "[]"))
        traces.append({
            "name": trace_info.root.name if trace_info.root else "unknown",
            "health": trace_info.health_status,
            "health_reasons": trace_info.health_reasons,
            "duration_ms": trace_info.root.duration_ms if trace_info.root else 0,
            "exit_reason": _strip_quotes(raw_exit),
            "tool_calls": len(tool_spans),
            "source_families": source_families,
        })

    health_counts: dict[str, int] = {"OK": 0, "DEGRADED": 0, "ERROR": 0}
    for t in traces:
        health_counts[t["health"]] = health_counts.get(t["health"], 0) + 1

    return {
        "total_traces": len(traces),
        "health_summary": health_counts,
        "traces": traces,
    }


def generate_performance_markdown(report: TraceReport) -> str:
    """Generate performance.md with token budget and latency analysis."""
    lines = [
        "# Performance Analysis",
        "",
    ]

    # Token budget table
    lines.append("## Token Budget")
    lines.append("")
    lines.append("| Agent | Prompt | Completion | Total |")
    lines.append("|-------|--------|------------|-------|")

    for trace in report.traces:
        for span in trace.all_spans:
            prompt = span.attributes.get("agent.prompt_tokens") or span.attributes.get(
                "react.prompt_tokens"
            )
            completion = span.attributes.get(
                "agent.completion_tokens"
            ) or span.attributes.get("react.completion_tokens")
            total = span.tokens
            if total > 0:
                p = int(prompt) if prompt else 0
                c = int(completion) if completion else 0
                lines.append(f"| {span.name} | {p:,} | {c:,} | {total:,} |")

    lines.append("")

    # Latency waterfall
    lines.append("## Latency Waterfall")
    lines.append("")
    lines.append("```")

    max_duration = 1.0
    waterfall_spans: list[tuple[str, float]] = []
    for trace in report.traces:
        if trace.root:
            _collect_waterfall_spans(trace.root, waterfall_spans)
    if waterfall_spans:
        max_duration = max(d for _, d in waterfall_spans)

    for name, dur_ms in waterfall_spans:
        dur_s = dur_ms / 1000
        bar_len = int((dur_ms / max_duration) * 40) if max_duration > 0 else 0
        bar = "#" * max(bar_len, 1)
        lines.append(f"{name:30s} |{bar:40s}| ({dur_s:.1f}s)")

    lines.append("```")
    lines.append("")

    # Tool statistics
    lines.append("## Tool Statistics")
    lines.append("")
    lines.append("| Tool | Calls | Avg Duration (ms) |")
    lines.append("|------|-------|--------------------|")

    tool_stats: dict[str, list[float]] = {}
    for trace in report.traces:
        for span in trace.all_spans:
            if span.span_type == "SpanType.TOOL" or span.name.startswith("tool."):
                tool_name = span.attributes.get("tool.name", span.name)
                tool_stats.setdefault(str(tool_name), []).append(span.duration_ms)

    for tool_name, durations in sorted(tool_stats.items()):
        avg_ms = sum(durations) / len(durations) if durations else 0
        lines.append(f"| {tool_name} | {len(durations)} | {avg_ms:.0f} |")

    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers for tree display
# ---------------------------------------------------------------------------


def _collect_span_table_rows(
    node: SpanNode, lines: list[str], depth: int
) -> None:
    """Recursively collect span data into markdown table rows."""
    indent = "  " * depth
    dur_s = node.duration_ms / 1000
    tokens = node.tokens
    tok_str = f"{tokens:,}" if tokens > 0 else "-"
    lines.append(f"| {indent}{node.name} | {dur_s:.1f}s | {tok_str} | {node.status} |")
    for child in node.children:
        _collect_span_table_rows(child, lines, depth + 1)


def _collect_waterfall_spans(
    node: SpanNode, result: list[tuple[str, float]]
) -> None:
    """Collect spans for waterfall chart (depth-first)."""
    result.append((node.name, node.duration_ms))
    for child in node.children:
        _collect_waterfall_spans(child, result)


def _print_span_tree(
    node: SpanNode,
    indent: int,
    is_last: bool,
    prefix: str,
) -> None:
    """Print a span tree to stdout with box-drawing characters."""
    connector = "  └─ " if is_last else "  ├─ "

    dur_s = node.duration_ms / 1000
    tokens = node.tokens
    tok_str = f"  {tokens:,} tok" if tokens > 0 else ""

    # Don't print connector for root
    if prefix == "" and indent == 2:
        line = f"{'  ' * indent}{node.name:30s} {dur_s:>6.1f}s{tok_str}"
    else:
        line = f"{prefix}{connector}{node.name:30s} {dur_s:>6.1f}s{tok_str}"

    print(line)

    # Child prefix
    if prefix == "" and indent == 2:
        child_prefix = "  " * indent
    else:
        child_prefix = prefix + ("     " if is_last else "  │  ")

    for i, child in enumerate(node.children):
        child_is_last = i == len(node.children) - 1
        _print_span_tree(child, indent, child_is_last, child_prefix)
