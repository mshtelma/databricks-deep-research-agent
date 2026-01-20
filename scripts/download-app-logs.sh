#!/bin/bash
# Download logs from a Databricks App via /logz/batch endpoint
#
# Usage: ./scripts/download-app-logs.sh <app_name> <profile> [options]
# Example: ./scripts/download-app-logs.sh deep-research-agent-dre-dev e2-demo-west
#          ./scripts/download-app-logs.sh deep-research-agent-dre-dev e2-demo-west -f
#          ./scripts/download-app-logs.sh deep-research-agent-dre-dev e2-demo-west --search ERROR
#
# Options are passed through to the Python script:
#   -f, --follow     Follow logs continuously (poll every 5s)
#   -s, --search     Filter logs by search term
#   -i, --interval   Poll interval in seconds (default: 5)

set -e

APP_NAME="${1:?Usage: $0 <app_name> <profile> [options]}"
PROFILE="${2:?Usage: $0 <app_name> <profile> [options]}"
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use uv to run with project dependencies
uv run python "$SCRIPT_DIR/download-app-logs.py" "$APP_NAME" "$PROFILE" "$@"
