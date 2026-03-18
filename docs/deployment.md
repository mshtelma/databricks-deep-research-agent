# Deployment & Lakebase Operations Guide

Complete reference for deploying, operating, and troubleshooting the Deep Research Agent on Databricks Apps with Lakebase Autoscaling.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Architecture Overview](#2-architecture-overview)
3. [Target System](#3-target-system)
4. [Full Deployment Walkthrough](#4-full-deployment-walkthrough)
5. [How Lakebase Connectivity Works](#5-how-lakebase-connectivity-works)
6. [Local Development](#6-local-development)
7. [Database Management](#7-database-management)
8. [App Startup Flow](#8-app-startup-flow-on-databricks)
9. [Troubleshooting](#9-troubleshooting)
10. [Monitoring](#10-monitoring)
11. [Critical Design Decisions](#11-critical-design-decisions)

---

## 1. Prerequisites

| Tool | Install | Purpose |
|------|---------|---------|
| Databricks CLI | [docs.databricks.com](https://docs.databricks.com/dev-tools/cli/install.html) | Bundle deploy, app management, logs |
| `uv` | `curl -LsSf https://astral.sh/uv/install.sh \| sh` | Python package management |
| Node.js | [nodejs.org](https://nodejs.org/) | Frontend build |
| `jq` | `brew install jq` / `apt install jq` | JSON parsing in deploy scripts |

**Databricks CLI profiles** must be configured in `~/.databrickscfg` for each target workspace. The profile name is referenced in `databricks.yml` under each target.

**Brave API key** must be stored in a Databricks secret scope. The scope and key names are configurable via `--var brave_secret_scope=<scope>` during deployment (default: `deep-research-secrets` / `BRAVE_API_KEY`).

---

## 2. Architecture Overview

```
Developer laptop
    │
    ├── make deploy TARGET=dev
    │       │
    │       ├── databricks bundle deploy   ──→  Databricks workspace
    │       │                                      ├── Lakebase Autoscaling project
    │       │                                      │     ├── Branch: production
    │       │                                      │     └── Endpoint: primary (READ_WRITE)
    │       │                                      └── Databricks App
    │       │                                            ├── entrypoint.sh → uvicorn
    │       │                                            ├── Framework wheel (reinstalled each start)
    │       │                                            └── Config: ENDPOINT_NAME, PGHOST, LAKEBASE_DATABASE
    │       │
    │       ├── alembic upgrade head       ──→  Lakebase (deep_research DB)
    │       └── grant-app-permissions.sh   ──→  GRANT ALL to app service principal
    │
    └── make dev DEV_TARGET=dev
            │
            └── uvicorn (local)  ──→  Lakebase Autoscaling (same DB)
```

### Key Files

| File | Role |
|------|------|
| `databricks.yml` | Infrastructure-as-code: variables, targets, app resource, Lakebase project, model endpoint permissions, secret bindings |
| `app.yaml` | Base app spec synced to workspace. Sets provisioned-mode defaults (`LAKEBASE_INSTANCE_NAME`). The `config:` section in `databricks.yml` overrides env vars at deploy time with autoscaling values |
| `entrypoint.sh` | Container entrypoint. Force-reinstalls the framework wheel, then `exec uvicorn` |
| `config/app.yaml` | Runtime config: model endpoints, agent roles, research depth profiles, search settings |
| `Makefile` | Orchestrates the full 9-step deployment pipeline, database operations, dev servers, and testing |

---

## 3. Target System

### Defined Targets

| Target | Profile | Suffix | Project ID | Default | Notes |
|--------|---------|--------|------------|---------|-------|
| `dev` | `e2-demo-west` | `dev` | `deep-research-dev` | No | Primary development |
| `ais` | `ais` | `ais` | `deep-research-ais` | Yes | AIS production |
| `e2e` | `ais` | `e2e` | `deep-research-e2e` | No | E2E test isolation |
| `local-dev` | `ais` | `local` | `deep-research-local` | No | Local dev Lakebase |

### How Variables Resolve

```
TARGET=dev
    → databricks.yml targets.dev.workspace.profile = "e2-demo-west"
    → databricks.yml targets.dev.variables.resource_suffix = "dev"
    → PROJECT_ID = "deep-research-dev"
    → App name = "${var.app_name}-${var.resource_suffix}" = "deep-research-agent-dev"
```

The `--var` mechanism passes values at deploy time. Variables in `databricks.yml` have defaults (e.g., `endpoint_name: "pending"`, `pghost: "pending"`) that are overridden by the deploy script with real discovered values.

**Constraint**: `app_name` + `-` + `resource_suffix` must be <= 30 characters (Databricks app name limit).

---

## 4. Full Deployment Walkthrough

```bash
make deploy TARGET=dev                    # Standard deployment
make deploy TARGET=dev BRAVE_SCOPE=msh    # With custom secret scope
```

### Automatic Prerequisites

Before the 9 steps begin, `make deploy` runs:
- **`build-framework`**: Builds the framework package wheel into `wheels/`
- **`build`**: Runs `npm ci && npm run build` in `frontend/`, outputs to `static/`
- **`requirements`**: Generates `requirements.txt` with framework wheel reference + `--find-links wheels/`

### Step 1: Bootstrap Bundle Deploy

```bash
databricks bundle deploy -t dev --var lakebase_database=postgres
```

**Why `postgres`?** The target `deep_research` database doesn't exist yet. `postgres` is a system database that always exists on any PostgreSQL instance. This bootstrap deploy creates:
- The Lakebase Autoscaling project (`deep-research-dev`)
- The Databricks App resource with all bindings (secrets, model endpoints, OBO scopes)
- The app's source code sync

The variables `endpoint_name` and `pghost` default to `"pending"` at this stage.

### Step 2: Discover Autoscaling Endpoint

```python
discover_autoscaling_endpoint("deep-research-dev")
# → "projects/deep-research-dev/branches/production/endpoints/primary"
```

Uses the Databricks SDK to list project branches and endpoints. Finds the first `READ_WRITE` endpoint in the project.

### Step 3: Wait for Endpoint + Resolve PGHOST

Polls the endpoint status at 10-second intervals with a 300-second timeout until the endpoint is ready. Then resolves the hostname:

```python
ep = client.postgres.get_endpoint(name=endpoint_name)
PGHOST = ep.status.hosts.host
# → e.g., "deep-research-dev-endpoint.database.cloud.databricks.com"
```

### Step 4: Create `deep_research` Database

Connects to the `databricks_postgres` system database (autoscaling's bootstrap DB) and creates `deep_research`:

```python
# Env: ENDPOINT_NAME=<real>, PGHOST=<real>, LAKEBASE_INSTANCE_NAME=""
ensure_database_exists()
# → CREATE DATABASE "deep_research" (idempotent: checks pg_database first)
```

**Key**: `LAKEBASE_INSTANCE_NAME=""` signals the autoscaling backend. The bootstrap module uses `databricks_postgres` (not `postgres`) as the system database for autoscaling.

### Step 5: Re-deploy Bundle with Real Values

```bash
databricks bundle deploy -t dev \
    --var lakebase_database=deep_research \
    --var endpoint_name=projects/deep-research-dev/branches/production/endpoints/primary \
    --var pghost=deep-research-dev-endpoint.database.cloud.databricks.com
```

Updates the app's config section with real `ENDPOINT_NAME` and `PGHOST` environment variables. This is what makes the app connect to autoscaling instead of falling back to provisioned mode.

### Step 6: Run Database Migrations

```bash
DATABRICKS_CONFIG_PROFILE=e2-demo-west \
ENDPOINT_NAME=<real> PGHOST=<real> \
LAKEBASE_INSTANCE_NAME="" LAKEBASE_DATABASE=deep_research \
    uv run alembic upgrade head
```

**Who runs this**: The **developer** (not the app service principal).

**Why**: The app SP has `CAN_CONNECT_AND_CREATE` permission on the Lakebase project but cannot create tables in the `public` schema. The developer creates tables, and Step 8 grants the SP permission to access them.

### Step 7: Write `.env.<target>` File

Creates `../.env.dev` (one directory above the app) with connection info:

```bash
# Lakebase connection for target: dev
# Generated by: make deploy TARGET=dev
DATABRICKS_CONFIG_PROFILE=e2-demo-west
ENDPOINT_NAME=projects/deep-research-dev/branches/production/endpoints/primary
PGHOST=deep-research-dev-endpoint.database.cloud.databricks.com
LAKEBASE_DATABASE=deep_research
```

This enables running `make dev DEV_TARGET=dev` to connect the local dev server to the deployed database.

### Step 8: Grant Permissions + Start App

**Permission grant**:
```bash
ENDPOINT_NAME=<real> ./scripts/grant-app-permissions.sh "" e2-demo-west deep_research deep-research-agent-dev
```
Finds the app's service principal and runs `GRANT ALL ON ALL TABLES/SEQUENCES IN SCHEMA public`.

**App start**:
```bash
databricks bundle run -t dev \
    --var endpoint_name=<real> --var pghost=<real> \
    deep_research_agent
```

**CRITICAL**: Must use `bundle run` (not `databricks apps start`). `bundle run` re-deploys the config section (with real `ENDPOINT_NAME`/`PGHOST`) and then starts the app. `apps start` just starts with whatever config was last synced to `app.yaml` — which has no `ENDPOINT_NAME` — causing the app to fall back to provisioned mode and crash.

### Step 9: Deployment Summary

```bash
databricks bundle summary -t dev
```

**Verify**:
1. Visit the app URL shown in the summary
2. Check `/health` returns `{"status": "healthy"}`
3. Create a new chat to verify database connectivity

---

## 5. How Lakebase Connectivity Works

### Two Backends

| Backend | Mode | Uses | When |
|---------|------|------|------|
| **Autoscaling** | Serverless, elastic | `ENDPOINT_NAME` + `PGHOST` | Production (deployed apps) |
| **Provisioned** | Managed instances | `LAKEBASE_INSTANCE_NAME` | Legacy / local dev fallback |

### Backend Detection (`credential_factory.py`)

Detection priority:
1. `ENDPOINT_NAME` env var or `settings.endpoint_name` → **Autoscaling**
2. `LAKEBASE_INSTANCE_NAME` or `PGHOST` → **Provisioned**
3. Neither → No Lakebase (falls back to `DATABASE_URL`)

If **both** `ENDPOINT_NAME` and `LAKEBASE_INSTANCE_NAME` are set, autoscaling wins with a `LAKEBASE_BACKEND_CONFLICT` warning. This can happen because `app.yaml` sets `LAKEBASE_INSTANCE_NAME` while `databricks.yml` config section adds `ENDPOINT_NAME`.

### Credential Lifecycle

```
┌──────────────────────────────────────────────────────────────────┐
│ Token Lifecycle                                                   │
│                                                                   │
│  0 min        45 min           55 min          60 min             │
│  ├─────────────┼────────────────┼───────────────┤                │
│  │  Valid       │  Pool recycle   │  Refresh buf  │  Expired      │
│  │             │  (connections   │  (token        │               │
│  │             │   refreshed)    │   refreshed)   │               │
│  └─────────────┴────────────────┴───────────────┘                │
│                                                                   │
│  pool_recycle=2700s (45 min) — stale pooled connections recycled  │
│  TOKEN_REFRESH_BUFFER=15 min — proactive refresh at 45 min mark  │
│  TOKEN_LIFETIME=1 hour — absolute token expiry                    │
└──────────────────────────────────────────────────────────────────┘
```

**OAuth token generation** (different API per backend):
- Autoscaling: `client.postgres.generate_database_credential(endpoint=...)`
- Provisioned: `client.database.generate_database_credential(instance_names=[...])`

**Proactive refresh** in `get_engine()` (`session.py`):
1. Checks if cached credential's `is_expired` property returns `True` (current time >= expires_at - 15 min buffer)
2. If expired: force-refreshes credential, disposes old engine, creates new engine on next request
3. Engine disposal is fire-and-forget via `asyncio.create_task()` in the running event loop

**Fallback refresh** in `get_db()`:
- If a database operation fails with an auth error (`"invalid password"` / `"invalid authorization"`), triggers `refresh_engine_credentials()` for the next request

### Connection String

```
postgresql+asyncpg://{username}:{token}@{host}:5432/{database}
```

- **SSL**: Always required for Lakebase (`connect_args={"ssl": True}`)
- **Username**: `PGUSER` env var → JWT `sub` claim → `client.current_user.me()` (fallback)
- **Pool config**: `pool_size=10`, `max_overflow=20`, `pool_pre_ping=True`

### The `app.yaml` vs `databricks.yml` Precedence

| Setting | `app.yaml` (base) | `databricks.yml` config section |
|---------|--------------------|---------------------------------|
| `LAKEBASE_INSTANCE_NAME` | `deep-research-lakebase` | (not set) |
| `LAKEBASE_DATABASE` | `deep_research` | `${var.lakebase_database}` |
| `ENDPOINT_NAME` | **(not present)** | `${var.endpoint_name}` |
| `PGHOST` | **(not present)** | `${var.pghost}` |

**Result when deployed**: Both `LAKEBASE_INSTANCE_NAME` and `ENDPOINT_NAME` are present in the app's environment. `credential_factory.py` detects `ENDPOINT_NAME` → selects autoscaling → logs `LAKEBASE_BACKEND_CONFLICT` warning → works correctly.

**CRITICAL**: `ENDPOINT_NAME` must NOT appear in `app.yaml`. The config section in `databricks.yml` adds it. If `app.yaml` also had `ENDPOINT_NAME`, DAB merge precedence is unclear (`app.yaml` may take precedence over the config section), which could result in a stale or incorrect value.

---

## 6. Local Development

### Standard: Remote Lakebase (Recommended)

**One-time setup**:
```bash
make db-provision TARGET=local-dev    # Provisions Autoscaling, writes ../.env.local-dev
```

Or use quickstart for first-time setup:
```bash
make quickstart    # Checks prerequisites, installs deps, creates .env.local
```

**Daily development**:
```bash
make dev                      # Sources ../.env + ../.env.local-dev → connects to remote Lakebase
# Backend on :8000, Frontend on :5173 with hot reload
```

### Running Against a Deployed Database

```bash
make dev DEV_TARGET=dev       # Sources ../.env + ../.env.dev → connects to dev Lakebase
make dev DEV_TARGET=ais       # Sources ../.env + ../.env.ais → connects to ais Lakebase
```

### Fallback: Local PostgreSQL

For offline development or when Lakebase is unavailable:

```bash
make db-local                 # Starts Docker PostgreSQL + runs migrations
make dev                      # Uses DATABASE_URL from .env
```

Set in `.env`:
```bash
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/postgres
```

### Env File Layout

All env files live in the **parent directory** (`../` relative to the app):

```
../.env              ← Shared secrets (BRAVE_API_KEY, etc.) — NOT touched by provisioning
../.env.local-dev    ← Written by: make db-provision TARGET=local-dev
../.env.e2e          ← Written by: make db-provision TARGET=e2e
../.env.dev          ← Written by: make deploy TARGET=dev (Step 7)
../.env.ais          ← Written by: make deploy TARGET=ais (Step 7)
```

The `load-env` Makefile helper sources `../.env` first, then `../.env.{target}` (target vars override shared ones):
```makefile
$(call load-env,$(DEV_TARGET))
# → set -a && . ../.env && . ../.env.local-dev && set +a
```

---

## 7. Database Management

### Run Migrations

```bash
# Local env (default database)
make db-migrate

# Local env with database suffix
make db-migrate DB_SUFFIX=dev           # → deep_research_dev

# Remote target (auto-detects Autoscaling vs Provisioned)
make db-migrate-remote TARGET=dev
make db-migrate-remote TARGET=ais
```

### Check Migration Status

```bash
make db-status
```

### Reset Schema (Destructive)

Drops ALL tables and enum types, then re-runs migrations:

```bash
make db-reset                           # Local env
make db-reset TARGET=dev                # Remote, auto-detects backend
```

### Clean Data (Preserves Schema)

Deletes all chats/messages but keeps the table structure:

```bash
make clean_db                           # Default database
make clean_db DB_SUFFIX=dev             # deep_research_dev
make clean-e2e                          # E2E database
```

### Grant Permissions

After running migrations, grant the app SP access to the newly created tables:

```bash
# Autoscaling (set ENDPOINT_NAME)
ENDPOINT_NAME=<endpoint> ./scripts/grant-app-permissions.sh "" <profile> deep_research <app-name>

# Provisioned (pass instance name)
./scripts/grant-app-permissions.sh <instance-name> <profile> deep_research <app-name>
```

### Cleanup Orphaned Resources

When `db-provision` or `deploy` fails due to stale resources from interrupted deploys:

```bash
make db-cleanup TARGET=local-dev        # Removes orphaned Lakebase resources + stale Terraform state
```

---

## 8. App Startup Flow (on Databricks)

### Container Startup (`entrypoint.sh`)

1. **Force-reinstall framework wheel**: `pip install --force-reinstall --no-deps --no-cache-dir wheels/databricks_deep_research-*.whl`
   - Why: Databricks Apps containers persist across deploys. pip skips install when version matches, even if wheel contents changed.
   - Impact: ~1s for a ~175KB wheel
2. **Start uvicorn**: `exec uvicorn deep_research.main:app "$@"`
   - `exec` replaces bash for proper signal handling
   - Databricks Apps appends `--port XXXX` to the command array; `$@` forwards it

### FastAPI Lifespan Startup (`main.py`)

1. **Config loading** (fail-fast): Validates `config/app.yaml`. Exits with code 1 if invalid.
2. **Tracing setup**: Configures MLflow for observability.
3. **Credential lazy init**: Token is generated on first DB request, not at startup. This avoids blocking startup on Lakebase connectivity.
4. **Shared services**: `LLMClient`, `BraveSearchClient`, `WebCrawler` (all stored on `app.state`).
5. **Plugin discovery**: Discovers and loads plugins via entry points.
6. **Job manager**: Background job orchestration. Started with `await job_manager.start(session_maker)`.
7. **Session cleanup**: Background task runs every 5 minutes, deleting expired incognito sessions.

### Request Serving

- `/health` → `{"status": "healthy", "service": "deep-research-agent"}`
- `/api/v1/*` → API routes (chats, messages, research, etc.)
- `/*` → SPA static files (catch-all, must be registered last)

### Graceful Shutdown (15-second window)

1. Cancel session cleanup task
2. Stop job manager (cancels running research jobs)
3. Flush MLflow traces
4. Close shared services (LLM client, web crawler, Brave client)
5. Close database connections (`close_db()`)

---

## 9. Troubleshooting

### App crashes with "Database instance not found"

**Cause**: App started without `ENDPOINT_NAME` env var. Falls back to provisioned backend → looks for `LAKEBASE_INSTANCE_NAME=deep-research-lakebase` instance → doesn't exist.

**Fix**: Ensure Step 8 uses `databricks bundle run --var endpoint_name=... --var pghost=...` (NOT `databricks apps start`). See [Critical Design Decision #1](#11-critical-design-decisions).

### "workspace limit" or "branch already exists" during deploy

**Cause**: Interrupted previous deploy left orphaned Autoscaling resources in the workspace.

**Fix**:
```bash
make db-cleanup TARGET=<target>
# Then retry:
make deploy TARGET=<target>
```

### Migrations fail with permission error

**Cause**: Running as the app service principal (has `CAN_CONNECT_AND_CREATE` but not table-level permissions in `public` schema).

**Fix**: Run migrations as the developer, then grant permissions to the app SP:
```bash
make db-migrate-remote TARGET=dev
ENDPOINT_NAME=<ep> ./scripts/grant-app-permissions.sh "" <profile> deep_research <app-name>
```

### Token expiry / auth failures during requests

**Cause**: Stale OAuth token (1-hour lifetime).

**Fix**: Auto-heals on next request via proactive refresh in `get_engine()`. For persistent issues, restart the app. Check logs for `LAKEBASE_CREDENTIAL_EXPIRY_CHECK` entries.

### App starts but database queries fail

**Cause**: Permissions not granted after migration. The developer created tables, but the app SP hasn't been granted access.

**Fix**:
```bash
ENDPOINT_NAME=<ep> ./scripts/grant-app-permissions.sh "" <profile> deep_research <app-name>
```

### App starts but connects to wrong database

**Cause**: `ENDPOINT_NAME` has value `"pending"` (the default from `databricks.yml`). The deploy script didn't complete Step 5 (re-deploy with real values).

**Fix**: Re-run the deployment from Step 5 onward, or manually run:
```bash
databricks bundle deploy -t <target> \
    --var endpoint_name=<real> --var pghost=<real> --var lakebase_database=deep_research
databricks bundle run -t <target> \
    --var endpoint_name=<real> --var pghost=<real> deep_research_agent
```

### Framework wheel not updated after deploy

**Cause**: The wheel version didn't change between deploys, so pip skipped the install even though contents changed.

**Fix**: `entrypoint.sh` uses `--force-reinstall`, which handles this automatically. If you're seeing stale code, verify `entrypoint.sh` is being used as the app's command (check `databricks.yml` config section).

---

## 10. Monitoring

### Logs

```bash
make logs TARGET=dev                              # Fetch once
make logs TARGET=dev FOLLOW=-f                    # Follow real-time
make logs TARGET=dev SEARCH="--search ERROR"      # Filter by keyword
make logs TARGET=dev FOLLOW=-f SEARCH="--search ERROR"  # Combine
```

Logs are fetched via the `/logz/batch` REST API. They are **not persisted** when app compute shuts down.

### Key Log Markers

| Marker | Meaning |
|--------|---------|
| `LAKEBASE_BACKEND_SELECTED backend=autoscaling` | Correct backend detected |
| `LAKEBASE_BACKEND_SELECTED backend=provisioned` | Using provisioned (expected for local dev) |
| `LAKEBASE_BACKEND_CONFLICT` | Both `ENDPOINT_NAME` and `LAKEBASE_INSTANCE_NAME` set (autoscaling wins) |
| `AUTOSCALING_CREDENTIAL_GENERATING` | Generating new OAuth token for autoscaling |
| `LAKEBASE_CREDENTIAL_GENERATED` | Token generated (includes preview, expiry) |
| `LAKEBASE_CREDENTIAL_EXPIRY_CHECK` | Checking if token needs refresh |
| `LAKEBASE_ENGINE_REFRESH_TRIGGERED` | Token expired, refreshing engine |
| `LAKEBASE_ENGINE_DISPOSED` | Old engine disposed after token refresh |

### Local Dev Logs

```bash
tail -f /tmp/deep-research-dev.log    # Dev server (make dev)
tail -f /tmp/deep-research-prod.log   # Prod server (make prod)
```

---

## 11. Critical Design Decisions

These are the design decisions that are easy to forget and cause outages if violated.

### 1. `bundle run` not `apps start` for Step 8

`databricks bundle run` = re-deploy config (writes `ENDPOINT_NAME`/`PGHOST` to the app's environment) + start the app.

`databricks apps start` = just start with whatever config was last synced from `app.yaml`. Since `app.yaml` does **not** have `ENDPOINT_NAME`, the app falls back to provisioned mode → looks for `deep-research-lakebase` instance → crashes.

### 2. Two-phase bootstrap

Must deploy with `lakebase_database=postgres` first because `deep_research` doesn't exist yet. After the Autoscaling endpoint is ready, create the database, then re-deploy with `lakebase_database=deep_research`.

### 3. `"pending"` not `""` for default variable values

DAB strips empty-string values from env var entries, producing bare `name:`-only entries without `value:` that the Apps runtime rejects. Using `"pending"` as a placeholder avoids this.

### 4. Developer runs migrations, not the app

The app SP has `CAN_CONNECT_AND_CREATE` on the Lakebase project but cannot create tables in the `public` schema. The developer runs migrations to create tables, then grants the SP permission via `grant-app-permissions.sh`.

### 5. `ENDPOINT_NAME` must not appear in `app.yaml`

The `config:` section in `databricks.yml` adds `ENDPOINT_NAME` and `PGHOST`. If `app.yaml` also had `ENDPOINT_NAME`, DAB merge precedence could cause the `app.yaml` value to win over the `databricks.yml` config section value, breaking autoscaling connectivity.

### 6. Framework wheel force-reinstall in entrypoint

Databricks Apps containers persist across deploys. pip skips wheel installation when the version string matches a prior install, even if the wheel contents have changed. `entrypoint.sh` uses `--force-reinstall --no-deps` to ensure the latest wheel is always installed (~1 second overhead).
