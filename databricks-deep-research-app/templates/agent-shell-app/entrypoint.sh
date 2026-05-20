#!/usr/bin/env bash
# Shell-app entrypoint. Databricks Apps build precedence:
# - if requirements.txt is present, Apps uses pip directly
# - otherwise if pyproject.toml + uv.lock are present, Apps uses `uv sync`
#
# We commit pyproject.toml + uv.lock and DELIBERATELY ship NO requirements.txt
# so `uv sync` runs at build time. See plan Section E.4 (build/install
# precedence rule).
set -euo pipefail

uv sync --frozen
exec uv run uvicorn app:app --host 0.0.0.0 --port 8000
