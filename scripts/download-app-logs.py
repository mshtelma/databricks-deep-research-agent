#!/usr/bin/env python3
"""Download logs from Databricks App via /logz/batch endpoint.

Usage:
    uv run python scripts/download-app-logs.py <app_name> <profile> [options]

Examples:
    # Fetch logs once
    uv run python scripts/download-app-logs.py deep-research-agent-dre-dev e2-demo-west

    # Follow logs continuously (poll every 5s)
    uv run python scripts/download-app-logs.py deep-research-agent-dre-dev e2-demo-west -f

    # Search for errors
    uv run python scripts/download-app-logs.py deep-research-agent-dre-dev e2-demo-west --search ERROR
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime

import httpx


def get_token(profile: str) -> str:
    """Get OAuth token from Databricks CLI."""
    result = subprocess.run(
        ["databricks", "auth", "token", "--profile", profile],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: Failed to get token: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    try:
        data = json.loads(result.stdout)
        return data.get("access_token") or data.get("token_value") or ""
    except json.JSONDecodeError:
        return result.stdout.strip()


def get_app_url(app_name: str, profile: str) -> str:
    """Get app URL from Databricks CLI."""
    result = subprocess.run(
        ["databricks", "apps", "get", app_name, "--profile", profile, "--output", "json"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: Failed to get app info: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    try:
        data = json.loads(result.stdout)
        return data.get("url") or ""
    except json.JSONDecodeError:
        print(f"ERROR: Invalid JSON from databricks apps get", file=sys.stderr)
        sys.exit(1)


def fetch_logs(app_url: str, token: str, search: str = "") -> list:
    """Fetch logs from /logz/batch endpoint."""
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{app_url.rstrip('/')}/logz/batch"

    try:
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            logs = resp.json()
    except httpx.HTTPStatusError as e:
        print(f"ERROR: HTTP {e.response.status_code}: {e.response.text[:200]}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return []

    if not isinstance(logs, list):
        return []

    # Filter by search term if provided
    if search:
        logs = [log for log in logs if search.lower() in log.get("message", "").lower()]

    return logs


def display_logs(logs: list, last_ts: int = 0) -> int:
    """Display logs to stdout, return max timestamp seen."""
    if not logs:
        return last_ts

    # Sort by timestamp
    logs = sorted(logs, key=lambda x: x.get("timestamp", 0))
    max_ts = last_ts

    for log in logs:
        ts = log.get("timestamp", 0)

        # Skip already-displayed logs
        if ts <= last_ts:
            continue

        src = log.get("source", "UNKNOWN")
        msg = log.get("message", "").rstrip()

        # Format timestamp
        if ts:
            time_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            max_ts = max(max_ts, ts)
        else:
            time_str = "        "

        # Format source (pad to 6 chars)
        src_str = src[:6].ljust(6)

        print(f"[{time_str}] [{src_str}] {msg}")

    return max_ts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch logs from Databricks App via /logz/batch endpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch logs once
  python download-app-logs.py deep-research-agent-dre-dev e2-demo-west

  # Follow logs continuously
  python download-app-logs.py deep-research-agent-dre-dev e2-demo-west -f

  # Search for errors
  python download-app-logs.py deep-research-agent-dre-dev e2-demo-west --search ERROR

  # Follow + search with custom interval
  python download-app-logs.py deep-research-agent-dre-dev e2-demo-west -f --search ERROR --interval 2
        """,
    )
    parser.add_argument("app_name", help="Databricks App name")
    parser.add_argument("profile", help="Databricks CLI profile")
    parser.add_argument("--search", "-s", default="", help="Filter logs by search term")
    parser.add_argument("--follow", "-f", action="store_true", help="Follow logs (poll continuously)")
    parser.add_argument("--interval", "-i", type=int, default=5, help="Poll interval in seconds (default: 5)")

    args = parser.parse_args()

    # Get authentication
    token = get_token(args.profile)
    if not token:
        print("ERROR: Could not get OAuth token", file=sys.stderr)
        sys.exit(1)

    # Get app URL
    app_url = get_app_url(args.app_name, args.profile)
    if not app_url:
        print("ERROR: Could not get app URL", file=sys.stderr)
        sys.exit(1)

    print(f"Fetching logs from {app_url}/logz/batch", file=sys.stderr)
    if args.search:
        print(f"Filtering by: '{args.search}'", file=sys.stderr)
    if args.follow:
        print(f"Following logs (Ctrl+C to stop)...", file=sys.stderr)
    print("-" * 50, file=sys.stderr)

    last_ts = 0
    log_count = 0

    try:
        while True:
            logs = fetch_logs(app_url, token, args.search)
            new_ts = display_logs(logs, last_ts)

            # Count new logs
            if logs:
                new_logs = [l for l in logs if l.get("timestamp", 0) > last_ts]
                log_count += len(new_logs)

            last_ts = new_ts

            # If not following, exit after first fetch
            if not args.follow:
                break

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print(f"\nStopped. Displayed {log_count} log entries.", file=sys.stderr)
        sys.exit(0)

    if not args.follow:
        print(f"Displayed {log_count} log entries.", file=sys.stderr)


if __name__ == "__main__":
    main()
