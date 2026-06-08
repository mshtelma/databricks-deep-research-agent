#!/usr/bin/env python3
"""Analyze and export MLflow traces from Databricks workspace.

Three modes:

* ``list`` (default) — search recent traces in the canonical experiment
  with optional ``--surface`` / ``--app-name`` / ``--agent-v2-id``
  filters, plus a time window. Exports a summary CSV + JSON to
  ``scripts/output/``.

* ``download --request-id <tr-XXX>`` — pull ONE specific trace and
  expand it into a structured directory under
  ``scripts/output/traces/<request_id>/`` (trace.json, spans.json,
  lane_outputs/, synthesizer_claims.json, final_report.md). The unit
  of debugging.

Usage:
    uv run python scripts/analyze_traces.py [list]
    uv run python scripts/analyze_traces.py --surface shell-app
    uv run python scripts/analyze_traces.py --agent-v2-id <uuid>
    uv run python scripts/analyze_traces.py download --request-id tr-XXX
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from dotenv import load_dotenv

# Load .env from project root
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

# Output directory
OUTPUT_DIR = Path(__file__).parent / "output"

# Canonical experiment for all DRE surfaces (designer-chat, main-chat,
# shell-app). Override with MLFLOW_EXPERIMENT_NAME in the env if needed.
_CANONICAL_EXPERIMENT = "/Shared/deep-research-agent-experiments"


def setup_mlflow() -> str:
    """Configure MLflow to connect to Databricks workspace."""
    mlflow.set_tracking_uri("databricks")

    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", _CANONICAL_EXPERIMENT)

    print(f"Connecting to experiment: {experiment_name}")

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"ERROR: Experiment not found: {experiment_name}")
        print("\nListing available experiments...")
        for exp in mlflow.search_experiments(max_results=20):
            print(f"  - {exp.name}")
        sys.exit(1)

    print(f"Experiment ID: {experiment.experiment_id}")
    print(f"Artifact Location: {experiment.artifact_location}")
    return experiment.experiment_id


def _build_filter_string(args: argparse.Namespace) -> str | None:
    """Compose an MLflow trace-search filter from CLI flags.

    Returns ``None`` when no filters are set so the search returns
    everything in the experiment. Otherwise joins filters with ``and`` —
    e.g., ``tag.\"dr.surface\" = 'shell-app' and tag.\"dr.app_name\" = 'dr-shell-XXX'``.
    """
    clauses: list[str] = []
    if args.surface:
        clauses.append(f"tag.\"dr.surface\" = '{args.surface}'")
    if args.app_name:
        clauses.append(f"tag.\"dr.app_name\" = '{args.app_name}'")
    if args.agent_v2_id:
        clauses.append(f"tag.\"dr.agent_v2_id\" = '{args.agent_v2_id}'")
    return " and ".join(clauses) if clauses else None


def search_all_traces(
    experiment_id: str,
    *,
    filter_string: str | None = None,
    max_results: int = 100,
) -> pd.DataFrame:
    """Search traces in the experiment, optionally filtered by dr.* tags."""
    print(f"\nSearching for traces (max {max_results})...")
    if filter_string:
        print(f"Filter: {filter_string}")

    kwargs: dict[str, Any] = {
        "locations": [experiment_id],
        "max_results": max_results,
    }
    if filter_string:
        kwargs["filter_string"] = filter_string

    traces_df = mlflow.search_traces(**kwargs)

    if traces_df.empty:
        print("No traces found!")
        return traces_df

    print(f"Found {len(traces_df)} traces")
    return traces_df


def export_traces_to_csv(traces_df: pd.DataFrame, filename: str = "traces.csv") -> None:
    """Export traces DataFrame to CSV."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / filename
    traces_df.to_csv(output_path, index=False)
    print(f"\nExported traces to: {output_path}")


def _serialize_value(val: object) -> object:
    """Serialize a value for JSON export."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(val, "isoformat"):
        return val.isoformat()
    if hasattr(val, "total_seconds"):
        return val.total_seconds()
    if hasattr(val, "tolist"):
        return val.tolist()
    if hasattr(val, "to_dict"):
        return val.to_dict()
    if isinstance(val, list):
        return [_serialize_value(v) for v in val]
    if isinstance(val, dict):
        return {k: _serialize_value(v) for k, v in val.items()}
    if isinstance(val, (int, float, bool, str)):
        return val
    return str(val)


def export_traces_to_json(traces_df: pd.DataFrame, filename: str = "traces.json") -> None:
    """Export traces DataFrame to JSON with all fields."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / filename

    records = []
    for _, row in traces_df.iterrows():
        record = {}
        for col in traces_df.columns:
            record[col] = _serialize_value(row[col])
        records.append(record)

    with open(output_path, "w") as f:
        json.dump(records, f, indent=2, default=str)

    print(f"Exported traces to: {output_path}")


def print_trace_details(traces_df: pd.DataFrame, limit: int = 5) -> None:
    """Print detailed information about traces."""
    print("\n" + "=" * 80)
    print(f"TRACE DETAILS (showing first {min(limit, len(traces_df))})")
    print("=" * 80)

    for idx, row in traces_df.head(limit).iterrows():
        print(f"\n{'─' * 80}")
        print(f"TRACE #{idx + 1}")
        print(f"{'─' * 80}")

        for col in sorted(traces_df.columns):
            val = row[col]
            try:
                is_valid = val is not None and (
                    not hasattr(val, "__len__") or len(val) > 0
                )
            except (TypeError, ValueError):
                is_valid = pd.notna(val) if not hasattr(val, "__iter__") else True

            if is_valid:
                val_str = str(val)
                if len(val_str) > 200:
                    val_str = val_str[:200] + "..."
                print(f"  {col}: {val_str}")


def analyze_trace_statistics(traces_df: pd.DataFrame) -> None:
    """Analyze and print trace statistics."""
    print("\n" + "=" * 80)
    print("TRACE STATISTICS")
    print("=" * 80)

    print(f"\nTotal traces: {len(traces_df)}")

    if "state" in traces_df.columns:
        print("\nState Distribution:")
        for state, count in traces_df["state"].value_counts().items():
            print(f"  {state}: {count}")

    if "execution_duration" in traces_df.columns:
        durations = traces_df["execution_duration"].dropna()
        if len(durations) > 0:
            duration_secs = durations / 1000.0
            print("\nExecution Duration:")
            print(f"  Min:    {duration_secs.min():.1f}s")
            print(f"  Max:    {duration_secs.max():.1f}s")
            print(f"  Mean:   {duration_secs.mean():.1f}s")
            print(f"  Median: {duration_secs.median():.1f}s")

    if "request_time" in traces_df.columns:
        timestamps = traces_df["request_time"].dropna()
        if len(timestamps) > 0:
            print("\nTime Range:")
            min_ts = datetime.fromtimestamp(timestamps.min() / 1000, tz=timezone.utc)
            max_ts = datetime.fromtimestamp(timestamps.max() / 1000, tz=timezone.utc)
            print(f"  Earliest: {min_ts.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(f"  Latest:   {max_ts.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    # Surface distribution from dr.surface tags (if present).
    if "tags" in traces_df.columns:
        surfaces: dict[str, int] = {}
        for tags in traces_df["tags"].dropna():
            if isinstance(tags, dict):
                surface = tags.get("dr.surface")
                if surface:
                    surfaces[surface] = surfaces.get(surface, 0) + 1
        if surfaces:
            print("\nSurface Distribution (dr.surface tag):")
            for surface, count in sorted(surfaces.items(), key=lambda kv: -kv[1]):
                print(f"  {surface}: {count}")


def _walk_spans(
    spans: list[dict[str, Any]] | None,
    fn: Any,
) -> None:
    """Walk a flat list of spans (Databricks traces serialize spans flat)."""
    if not isinstance(spans, list):
        return
    for span in spans:
        if isinstance(span, dict):
            fn(span)


def _decode_span_events(span: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract events from a span; events carry the framework's StreamEvent payloads."""
    events = span.get("events") or []
    return [e for e in events if isinstance(e, dict)]


def download_one_trace(request_id: str) -> None:
    """Fetch a single trace and expand it into a debug-friendly directory."""
    print(f"\nDownloading trace: {request_id}")
    trace = mlflow.get_trace(request_id)
    if trace is None:
        print(f"ERROR: trace not found: {request_id}")
        sys.exit(2)

    trace_dir = OUTPUT_DIR / "traces" / request_id
    trace_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing to: {trace_dir}")

    # 1) trace.json — full top-level metadata, serialized via the MLflow API.
    try:
        trace_dict = trace.to_dict()  # type: ignore[attr-defined]
    except Exception:
        trace_dict = {"raw": str(trace)}
    (trace_dir / "trace.json").write_text(
        json.dumps(_serialize_value(trace_dict), indent=2, default=str),
        encoding="utf-8",
    )

    # 2) spans.json — every span flat, with name + duration + events count.
    spans: list[Any] = list(getattr(trace, "data", None).spans) if getattr(trace, "data", None) else []
    spans_payload = [_serialize_value(getattr(s, "to_dict", lambda: {})()) for s in spans]
    (trace_dir / "spans.json").write_text(
        json.dumps(spans_payload, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"  Spans: {len(spans_payload)}")

    # 3) lane_outputs/ — one file per ``agent_output`` event whose key
    # matches findings_lane_* / findings / report. Captures the actual
    # state the synthesizer received.
    lane_dir = trace_dir / "lane_outputs"
    lane_dir.mkdir(exist_ok=True)
    lane_count = 0
    for span_dict in spans_payload:
        if not isinstance(span_dict, dict):
            continue
        for event in _decode_span_events(span_dict):
            payload = event.get("attributes") or {}
            event_name = event.get("name") or payload.get("event_type")
            if event_name == "agent_output":
                key = payload.get("output_key") or payload.get("key") or "unknown"
                if isinstance(key, str) and (
                    key.startswith("findings") or key in {"report", "coordination"}
                ):
                    safe = key.replace("/", "_")
                    (lane_dir / f"{safe}.json").write_text(
                        json.dumps(payload, indent=2, default=str),
                        encoding="utf-8",
                    )
                    lane_count += 1
    print(f"  Lane outputs captured: {lane_count}")

    # 4) synthesizer_claims.json — every claim_generated + claim_verified +
    # citation_corrected + verification_summary event, in arrival order.
    claims: list[dict[str, Any]] = []
    for span_dict in spans_payload:
        if not isinstance(span_dict, dict):
            continue
        for event in _decode_span_events(span_dict):
            event_name = event.get("name") or ""
            if event_name in {
                "claim_generated",
                "claim_verified",
                "citation_corrected",
                "verification_summary",
                "numeric_claim_detected",
                "synthesis_started",
            }:
                claims.append(event)
    (trace_dir / "synthesizer_claims.json").write_text(
        json.dumps(claims, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"  Synthesizer events: {len(claims)}")

    # 5) final_report.md — extract the workflow_completed event's output if
    # the trace contains one; else search agent_output with key=report.
    final_report = ""
    for span_dict in spans_payload:
        if not isinstance(span_dict, dict):
            continue
        for event in _decode_span_events(span_dict):
            payload = event.get("attributes") or {}
            event_name = event.get("name") or payload.get("event_type") or ""
            if event_name == "agent_output" and payload.get("output_key") == "report":
                final_report = str(payload.get("value") or payload.get("output") or "")
                break
        if final_report:
            break
    (trace_dir / "final_report.md").write_text(
        final_report or "(no report event found)", encoding="utf-8"
    )

    print(f"\nDONE. See: {trace_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze and export MLflow traces from Databricks workspace."
    )
    sub = parser.add_subparsers(dest="mode")

    # default 'list' mode (also reachable without subcommand)
    list_parser = sub.add_parser("list", help="Search recent traces with optional filters")
    for p in (parser, list_parser):
        p.add_argument(
            "--surface",
            choices=["designer-chat", "main-chat", "shell-app"],
            help="Filter by dr.surface tag",
        )
        p.add_argument("--app-name", help="Filter by dr.app_name tag (shell-app traces)")
        p.add_argument("--agent-v2-id", help="Filter by dr.agent_v2_id tag (cross-surface)")
        p.add_argument(
            "--max-results", type=int, default=100,
            help="Maximum number of traces to return (default 100)",
        )

    dl_parser = sub.add_parser("download", help="Download one trace by request_id")
    dl_parser.add_argument("--request-id", required=True, help="Trace request_id (tr-XXX)")

    args = parser.parse_args()
    mode = args.mode or "list"

    print("=" * 80)
    print("MLflow Trace Analyzer & Exporter")
    print("=" * 80)

    experiment_id = setup_mlflow()

    if mode == "download":
        download_one_trace(args.request_id)
        return

    # list mode
    filter_string = _build_filter_string(args)
    traces_df = search_all_traces(
        experiment_id,
        filter_string=filter_string,
        max_results=args.max_results,
    )

    if traces_df.empty:
        print("\nNo traces to analyze. Exiting.")
        return

    analyze_trace_statistics(traces_df)
    print_trace_details(traces_df, limit=3)

    export_traces_to_csv(traces_df)
    export_traces_to_json(traces_df)

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)
    print(f"\nOutput files in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
