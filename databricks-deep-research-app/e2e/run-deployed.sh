#!/usr/bin/env bash
# Run the Playwright e2e suite against a DEPLOYED Databricks App.
#
# Resolves the app URL + a workspace OAuth token from the Databricks CLI profile
# and hands them to Playwright via E2E_BASE_URL / E2E_BEARER_TOKEN (see
# playwright.config.ts deployed mode). No local server is started.
#
# Usage:
#   ./run-deployed.sh [playwright args...]
#   PROFILE=ais APP_NAME=deep-research-agent-ais ./run-deployed.sh deterministic-functions.spec.ts
set -euo pipefail

PROFILE="${PROFILE:-ais}"
APP_NAME="${APP_NAME:-deep-research-agent-ais}"

echo "Resolving deployed app '$APP_NAME' (profile $PROFILE)..."
APP_URL="${E2E_BASE_URL:-$(databricks apps get "$APP_NAME" --profile "$PROFILE" --output json | python3 -c 'import sys,json;print(json.load(sys.stdin)["url"])')}"
TOKEN="$(databricks auth token --profile "$PROFILE" | python3 -c 'import sys,json;print(json.load(sys.stdin)["access_token"])')"

if [[ -z "$APP_URL" || -z "$TOKEN" ]]; then
  echo "ERROR: could not resolve app URL or token" >&2
  exit 1
fi
echo "  URL: $APP_URL"
echo "  token: ${#TOKEN} chars"

cd "$(dirname "$0")"
E2E_BASE_URL="$APP_URL" E2E_BEARER_TOKEN="$TOKEN" npx playwright test "$@"
