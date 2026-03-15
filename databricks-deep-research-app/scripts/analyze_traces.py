#!/usr/bin/env python3
"""Analyze and export MLflow traces from Databricks workspace.

Usage:
    uv run python scripts/analyze_traces.py

Exports all trace data to CSV/JSON files in the scripts/output/ directory.
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pandas as pd
from dotenv import load_dotenv

# Load .env from project root
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

# Output directory
OUTPUT_DIR = Path(__file__).parent / "output"


def setup_mlflow() -> str:
    """Configure MLflow to connect to Databricks workspace."""
    # Set tracking URI to Databricks
    mlflow.set_tracking_uri("databricks")

    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        "/Workspace/Users/michael.shtelma@databricks.com/experiments/deep-research-agent",
    )

    print(f"Connecting to experiment: {experiment_name}")

    # Get experiment ID
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


def search_all_traces(experiment_id: str, max_results: int = 100) -> pd.DataFrame:
    """Search and return all traces from the experiment."""
    print(f"\nSearching for traces (max {max_results})...")

    # Search all traces - get as DataFrame to see all columns
    # Use 'locations' parameter (experiment_ids is deprecated)
    traces_df = mlflow.search_traces(
        locations=[experiment_id],
        max_results=max_results,
    )

    if traces_df.empty:
        print("No traces found!")
        return traces_df

    print(f"Found {len(traces_df)} traces")
    print(f"\nAll columns ({len(traces_df.columns)}):")
    for col in sorted(traces_df.columns):
        print(f"  - {col}")

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

    # Check for pandas NA
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass  # Not a scalar, continue

    # Handle datetime/timestamp
    if hasattr(val, "isoformat"):
        return val.isoformat()

    # Handle timedelta
    if hasattr(val, "total_seconds"):
        return val.total_seconds()

    # Handle numpy arrays
    if hasattr(val, "tolist"):
        return val.tolist()

    # Handle MLflow Trace objects
    if hasattr(val, "to_dict"):
        return val.to_dict()

    # Handle lists and dicts recursively
    if isinstance(val, list):
        return [_serialize_value(v) for v in val]
    if isinstance(val, dict):
        return {k: _serialize_value(v) for k, v in val.items()}

    # Handle basic types
    if isinstance(val, (int, float, bool, str)):
        return val

    # Fallback to string
    return str(val)


def export_traces_to_json(traces_df: pd.DataFrame, filename: str = "traces.json") -> None:
    """Export traces DataFrame to JSON with all fields."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / filename

    # Convert DataFrame to list of dicts, handling non-serializable types
    records = []
    for _, row in traces_df.iterrows():
        record = {}
        for col in traces_df.columns:
            val = row[col]
            record[col] = _serialize_value(val)
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
            # Handle arrays and scalars differently for notna check
            try:
                is_valid = val is not None and (
                    not hasattr(val, "__len__") or len(val) > 0
                )
            except (TypeError, ValueError):
                is_valid = pd.notna(val) if not hasattr(val, "__iter__") else True

            if is_valid:
                # Truncate long values
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

    # State distribution (new API uses 'state' not 'status')
    if "state" in traces_df.columns:
        print("\nState Distribution:")
        state_counts = traces_df["state"].value_counts()
        for state, count in state_counts.items():
            print(f"  {state}: {count}")

    # Execution duration stats (in milliseconds)
    if "execution_duration" in traces_df.columns:
        durations = traces_df["execution_duration"].dropna()
        if len(durations) > 0:
            # Convert ms to seconds for display
            duration_secs = durations / 1000.0
            print("\nExecution Duration:")
            print(f"  Min:    {duration_secs.min():.1f}s ({duration_secs.min()/60:.1f} min)")
            print(f"  Max:    {duration_secs.max():.1f}s ({duration_secs.max()/60:.1f} min)")
            print(f"  Mean:   {duration_secs.mean():.1f}s ({duration_secs.mean()/60:.1f} min)")
            print(f"  Median: {duration_secs.median():.1f}s ({duration_secs.median()/60:.1f} min)")

    # Time range (request_time is Unix timestamp in milliseconds)
    if "request_time" in traces_df.columns:
        timestamps = traces_df["request_time"].dropna()
        if len(timestamps) > 0:
            print("\nTime Range:")
            min_ts = datetime.fromtimestamp(timestamps.min() / 1000, tz=timezone.utc)
            max_ts = datetime.fromtimestamp(timestamps.max() / 1000, tz=timezone.utc)
            print(f"  Earliest: {min_ts.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(f"  Latest:   {max_ts.strftime('%Y-%m-%d %H:%M:%S UTC')}")


def main() -> None:
    """Main entry point."""
    print("=" * 80)
    print("MLflow Trace Analyzer & Exporter")
    print("=" * 80)

    # Setup
    experiment_id = setup_mlflow()

    # Search traces
    traces_df = search_all_traces(experiment_id, max_results=100)

    if traces_df.empty:
        print("\nNo traces to analyze. Exiting.")
        return

    # Analyze
    analyze_trace_statistics(traces_df)

    # Print details
    print_trace_details(traces_df, limit=3)

    # Export
    export_traces_to_csv(traces_df)
    export_traces_to_json(traces_df)

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)
    print(f"\nOutput files in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
