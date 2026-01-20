# Deployment Guide

## Overview

The Deep Research Agent is deployed as a Databricks App with Lakebase (PostgreSQL) for persistence. This guide covers local development, production deployment, and operations.

## Quick Start

### Local Development

```bash
# Install dependencies
make install

# Start development servers (backend + frontend)
make dev

# Access UI at http://localhost:5173
# API at http://localhost:8000
```

### Production Build

```bash
# Build frontend to static/
make build

# Run production server
make prod

# Access at http://localhost:8000
```

### Deploy to Databricks

```bash
# Deploy to dev workspace
make deploy TARGET=dev BRAVE_SCOPE=secret-scope-for-brave-key

# Deploy to production
make deploy TARGET=databricks-cli-profile
```

## Environment Setup

### Required Environment Variables

```bash
# .env file

# Databricks Authentication (choose one)
DATABRICKS_CONFIG_PROFILE=your-profile-name  # Recommended
# OR
DATABRICKS_HOST=https://your-workspace.databricks.com
DATABRICKS_TOKEN=your-personal-access-token

# Lakebase Configuration
LAKEBASE_INSTANCE_NAME=your-instance-name
LAKEBASE_DATABASE=deep_research

# Brave Search API
BRAVE_API_KEY=your-brave-api-key

# Optional
APP_CONFIG_PATH=config/app.yaml
LOG_LEVEL=INFO
```

### Local PostgreSQL (Alternative)

For local development without Lakebase:

```bash
# Use local PostgreSQL
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/deep_research
```

## Databricks Apps Deployment

### Single Command Deployment

```bash
make deploy TARGET=dev BRAVE_SCOPE=msh
```

This executes an 8-step deployment pipeline:

1. **Build frontend**: `npm run build` → `static/`
2. **Generate requirements**: `pyproject.toml` → `requirements.txt`
3. **Bootstrap deploy**: Deploy with `postgres` database
4. **Wait for Lakebase**: Poll until instance is connectable
5. **Create database**: Create `deep_research` database
6. **Full deploy**: Re-deploy with `deep_research` configured
7. **Run migrations**: Apply schema with developer credentials
8. **Grant permissions**: Grant table access to app service principal

### Two-Phase Deployment Architecture

**Problem**: Chicken-and-egg scenario
- App needs `LAKEBASE_DATABASE` environment variable
- Database doesn't exist until Lakebase instance is created
- Lakebase instance is created by bundle deploy

**Solution**: Two-phase deployment

```
Phase 1: Bootstrap
┌────────────────────────────────────────────────────────────────┐
│ databricks bundle deploy -t dev                                 │
│ LAKEBASE_DATABASE=postgres  (always exists in PostgreSQL)       │
│                                                                 │
│ Creates: App, Lakebase instance                                 │
│ Waits: Until Lakebase is connectable                            │
└────────────────────────────────────────────────────────────────┘
           ↓
Phase 2: Database Setup
┌────────────────────────────────────────────────────────────────┐
│ ./scripts/create-database.sh                                    │
│ Creates: deep_research database                                 │
└────────────────────────────────────────────────────────────────┘
           ↓
Phase 3: Complete Deployment
┌────────────────────────────────────────────────────────────────┐
│ databricks bundle deploy -t dev                                 │
│ LAKEBASE_DATABASE=deep_research                                 │
│                                                                 │
│ Updates: App configuration with correct database                │
└────────────────────────────────────────────────────────────────┘
           ↓
Phase 4: Schema Migration
┌────────────────────────────────────────────────────────────────┐
│ alembic upgrade head                                            │
│ Creates: All tables (owned by developer)                        │
└────────────────────────────────────────────────────────────────┘
           ↓
Phase 5: Permission Grant
┌────────────────────────────────────────────────────────────────┐
│ ./scripts/grant-app-permissions.sh                              │
│ Grants: SELECT/INSERT/UPDATE/DELETE to app service principal    │
└────────────────────────────────────────────────────────────────┘
```

### Permission Model

```
┌─────────────────────────────────────────────────────────────────┐
│  Permission Problem                                              │
├─────────────────────────────────────────────────────────────────┤
│  1. Developer runs migrations → Tables owned by developer       │
│  2. App has CAN_CONNECT_AND_CREATE on database                  │
│  3. CAN_CONNECT_AND_CREATE ≠ SELECT/INSERT/UPDATE/DELETE        │
│  4. App service principal cannot access tables it doesn't own   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Solution: Post-Migration GRANT Statements                      │
├─────────────────────────────────────────────────────────────────┤
│  GRANT ALL ON ALL TABLES IN SCHEMA public TO <app_sp>;          │
│  GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO <app_sp>;       │
│  ALTER DEFAULT PRIVILEGES ... GRANT ALL ON TABLES TO <app_sp>;  │
│  ALTER DEFAULT PRIVILEGES ... GRANT ALL ON SEQUENCES TO <app_sp>│
└─────────────────────────────────────────────────────────────────┘
```


## Operations

### View Logs

```bash
# Fetch logs once
make logs TARGET=dev

# Follow logs (poll every 5s)
make logs TARGET=dev FOLLOW=-f

# Filter logs by term
make logs TARGET=dev SEARCH="--search ERROR"
```

### Restart App

```bash
# Restart after config changes
databricks bundle run -t dev deep_research_agent
```

### Check Status

```bash
# Deployment status
databricks bundle summary -t dev
```

### Manual Migrations

```bash
# Run migrations manually
make db-migrate-remote TARGET=dev
```

## Bundle Configuration

### databricks.yml Structure

The Databricks Asset Bundle configuration defines all infrastructure:

```yaml
# databricks.yml

bundle:
  name: deep-research-agent

variables:
  app_name:
    default: "deep-research-agent"
  lakebase_instance_name:
    default: "deep-research-lakebase"
  lakebase_database:
    default: "deep_research"
  lakebase_capacity:
    default: "CU_1"  # CU_1, CU_2, CU_4, etc.
  brave_secret_scope:
    default: "deep-research-secrets"
  resource_suffix:
    default: "dra"

resources:
  # Lakebase instance (PostgreSQL)
  database_instances:
    deep_research_lakebase:
      name: ${var.lakebase_instance_name}
      capacity: ${var.lakebase_capacity}

  # Databricks App
  apps:
    deep_research_agent:
      name: ${var.app_name}-${var.resource_suffix}
      source_code_path: .
      resources:
        # Secret access
        - name: brave-api-key
          secret:
            scope: ${var.brave_secret_scope}
            key: BRAVE_API_KEY
            permission: READ

        # Database access
        - name: database
          database:
            database_name: ${var.lakebase_database}
            instance_name: ${resources.database_instances.deep_research_lakebase.name}
            permission: CAN_CONNECT_AND_CREATE

        # Model endpoints (CAN_QUERY permission for each)
        - name: endpoint-sonnet
          serving_endpoint:
            name: databricks-claude-sonnet-4-5
            permission: CAN_QUERY
        # ... more endpoints

targets:
  dev:
    mode: development
    workspace:
      profile: <cli profile>
    variables:
      resource_suffix: "dev"

  prod:
    default: true
    workspace:
      profile: <cli profile>
    variables:
      resource_suffix: "prod"
```

### app.yaml Structure

The app.yaml configures the runtime environment:

```yaml
# app.yaml

# Command to start the app (port injected by Databricks)
command: ["uvicorn", "src.main:app"]

env:
  # MLflow Integration
  - name: MLFLOW_TRACKING_URI
    value: "databricks"
  - name: MLFLOW_EXPERIMENT_NAME
    value: "/Shared/deep-research-agent-experiments"

  # App Configuration
  - name: SERVE_STATIC
    value: "true"
  - name: APP_ENV
    value: "production"

  # Lakebase Configuration
  - name: LAKEBASE_INSTANCE_NAME
    value: "deep-research-lakebase"
  - name: LAKEBASE_DATABASE
    value: "deep_research"

  # Secrets (injected from bundle resources)
  - name: BRAVE_API_KEY
    valueFrom: "brave-api-key"  # References resource name from databricks.yml
```

### Deploy with Custom Variables

```bash
# Custom Lakebase capacity
databricks bundle deploy -t dev --var lakebase_capacity=CU_2

# Custom secret scope
make deploy TARGET=dev BRAVE_SCOPE=my-scope

# Custom database name
databricks bundle deploy -t dev --var lakebase_database=my_database
```

## Lakebase Authentication

### OAuth Token Flow

```
Developer Machine                    Databricks Profile              Lakebase Instance
       │                                    │                              │
       │ 1. Run migrations                  │                              │
       │    DATABRICKS_CONFIG_PROFILE=...   │                              │
       ├───────────────────────────────────>│                              │
       │                                    │ 2. Get OAuth token           │
       │                                    │    via generate_database_    │
       │                                    │    credential()              │
       │                                    ├─────────────────────────────>│
       │                                    │                              │
       │ 3. Connect with OAuth token        │                              │
       │    user="token", pass=<oauth>      │                              │
       ├──────────────────────────────────────────────────────────────────>│
       │                                    │                              │
       │ 4. Execute SQL                     │                              │
       ├──────────────────────────────────────────────────────────────────>│
```

### Token Characteristics

- **Lifetime**: 1 hour
- **Refresh buffer**: 5 minutes before expiry
- **Username**: Always `"token"` for OAuth connections
- **Host**: `{LAKEBASE_INSTANCE_NAME}.database.cloud.databricks.com`

### Token Refresh Fix

The system proactively refreshes tokens before expiry:

```python
# src/db/session.py

def get_session_maker():
    """
    CRITICAL: Always call get_engine() to trigger token refresh check.
    """
    engine = get_engine()  # Always call to check token expiry
    return async_sessionmaker(engine, ...)

def get_engine():
    """
    Proactive token refresh when approaching expiry.
    """
    if _should_refresh_token():
        _dispose_existing_engine()
        _create_engine_with_fresh_credentials()
    return _engine
```

## Testing

### Test Tiers

| Tier | Command | Config | Credentials |
|------|---------|--------|-------------|
| Unit | `make test` | N/A | None |
| Integration | `make test-integration` | `app.test.yaml` | Required |
| Complex | `make test-complex` | `app.yaml` | Required |
| E2E | `make e2e` | `app.yaml` | Required |

### Running Tests

```bash
# Unit tests (fast, no credentials)
make test

# Integration tests (real LLM/Brave)
make test-integration

# Complex long-running tests
make test-complex

# E2E Playwright tests
make e2e

# E2E with Playwright UI
make e2e-ui

# All tests
make test-all
```

### Test Configuration

`config/app.test.yaml` uses minimal settings for speed:

```yaml
agents:
  planner:
    max_plan_iterations: 1

research_types:
  light:
    steps:
      min: 1
      max: 2

citation_verification:
  enable_verification_retrieval: false
```

## Monitoring

### MLflow Tracing

All research executions are traced in MLflow:

```python
# Traces grouped by user and chat
mlflow.update_current_trace(metadata={
    "mlflow.trace.user": user_id,
    "mlflow.trace.session": chat_id,
})
```

### Trace Hierarchy

```
research_orchestration (CHAIN)
├── coordinator (AGENT)
│   └── llm_simple (LLM)
├── planner (AGENT)
│   └── llm_analytical (LLM)
├── researcher_step_1 (AGENT)
│   └── llm_analytical (LLM)
├── reflector (AGENT)
│   └── llm_simple (LLM)
└── synthesizer (AGENT)
    └── llm_complex (LLM)
```

### Health Endpoint

```bash
curl http://localhost:8000/v1/health
```

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "database": "connected",
  "llm": "available"
}
```

## Troubleshooting

### Common Issues

#### InvalidPasswordError after ~1 hour

**Cause**: OAuth token expired without refresh

**Solution**: Ensure `get_session_maker()` calls `get_engine()` on every request

**Fixed in**: 2026-01-11 (see `src/db/session.py`)

#### Tables not accessible by app

**Cause**: Tables owned by developer, not app service principal

**Solution**: Run permission grants after migrations

```bash
./scripts/grant-app-permissions.sh <instance> <workspace> <database> <app>
```

#### Database not found during deploy

**Cause**: Two-phase deployment not complete

**Solution**: Ensure `scripts/create-database.sh` runs between bootstrap and final deploy

#### Rate limit errors

**Cause**: LLM endpoint rate limits exceeded

**Solution**: System automatically retries with exponential backoff

### Key Scripts

| Script | Purpose |
|--------|---------|
| `scripts/create-database.sh` | Create deep_research database |
| `scripts/wait-for-lakebase.sh` | Wait for Lakebase to be connectable |
| `scripts/grant-app-permissions.sh` | Grant table permissions to app SP |
| `scripts/download-app-logs.py` | Fetch app logs via REST API |

### Key Files

| File | Purpose |
|------|---------|
| `databricks.yml` | Bundle configuration |
| `app.yaml` | App deployment config |
| `Makefile` | Build and deployment commands |
| `src/db/session.py` | Database session management |
| `src/db/lakebase_auth.py` | OAuth token provider |
| `src/db/grant_permissions.py` | Permission grant module |

## See Also

- [Architecture](./architecture.md) - System overview
- [Configuration](./configuration.md) - YAML settings
- [API Reference](./api.md) - REST endpoints
