#!/bin/bash
# Databricks Apps entrypoint — force-reinstalls local wheels before starting uvicorn.
#
# Why: Databricks Apps containers persist across deploys. pip skips wheel
# installation when the version matches a prior install, even if contents changed.
# --force-reinstall --no-deps reinstalls ONLY the wheel (no dep resolution, ~1s each).
#
# Port injection: Databricks Apps appends --port XXXX to the command array.
# $@ forwards those args to uvicorn. exec replaces bash for proper signal handling.

set -euo pipefail

# Force-reinstall framework wheel
echo "[entrypoint] Installing framework wheel..."
FRAMEWORK_WHEEL=$(ls -t wheels/databricks_deep_research-*.whl 2>/dev/null | head -1 || true)
if [ -n "$FRAMEWORK_WHEEL" ]; then
    python -m pip install --force-reinstall --no-deps --no-cache-dir "$FRAMEWORK_WHEEL" 2>&1 | tail -1
    echo "[entrypoint] Installed: $FRAMEWORK_WHEEL"
else
    echo "[entrypoint] ERROR: No framework wheel found in wheels/" >&2
    exit 1
fi

# Force-reinstall app wheel (contains latest code + static assets)
echo "[entrypoint] Installing app wheel..."
APP_WHEEL=$(ls -t wheels/databricks_deep_research_app-*.whl 2>/dev/null | head -1 || true)
if [ -n "$APP_WHEEL" ]; then
    python -m pip install --force-reinstall --no-deps --no-cache-dir "$APP_WHEEL" 2>&1 | tail -1
    echo "[entrypoint] Installed: $APP_WHEEL"
else
    echo "[entrypoint] WARNING: No app wheel found, using pip-installed version"
fi

echo "[entrypoint] Starting uvicorn with args: $@"
exec uvicorn deep_research.main:app "$@"
