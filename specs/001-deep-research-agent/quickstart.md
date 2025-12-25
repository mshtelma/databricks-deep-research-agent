# Quickstart: Deep Research Agent

**Date**: 2025-12-22 (Updated)
**Feature**: 001-deep-research-agent

This guide covers setting up and running the Deep Research Agent locally for development.

## Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.11+ | Backend runtime |
| Node.js | 20+ | Frontend build (development only) |
| uv | Latest | Python package manager |
| Databricks CLI | Latest | Authentication & workspace access |

> **Note**: Node.js is only required for frontend development. Production deployment uses pre-built static files served by FastAPI.

### Required Accounts & Keys

1. **Databricks Workspace** - With access to:
   - Foundation Model APIs (or external model endpoints)
   - Lakebase (PostgreSQL) preview
   - MLflow tracking

2. **Brave Search API Key** - Get from [brave.com/search/api](https://brave.com/search/api)
   - Free tier: 2,000 requests/month

---

## Initial Setup

### 1. Clone and Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd databricks-deep-research-agent

# Install backend dependencies
cd backend
uv sync

# Install frontend dependencies
cd ../frontend
npm install
```

### 2. Configure Databricks Authentication

```bash
# Configure Databricks CLI profile
databricks configure --profile deep-research

# Verify connection
databricks auth describe --profile deep-research
```

### 3. Set Environment Variables

Create `.env` file in the project root (not backend/):

**Option A: Profile-Based Authentication (Recommended)**

```bash
# .env - Profile-based authentication
# Required for both LLM access and Lakebase OAuth

# Databricks Profile (required)
DATABRICKS_CONFIG_PROFILE=deep-research

# Lakebase Configuration (OAuth-authenticated PostgreSQL)
LAKEBASE_INSTANCE_NAME=your-instance-name  # e.g., "msh-deep-research" (NOT full hostname)
LAKEBASE_DATABASE=deep_research
# Host is derived automatically: {instance_name}.database.cloud.databricks.com

# API Keys
BRAVE_API_KEY=your-brave-search-api-key

# MLflow
MLFLOW_TRACKING_URI=databricks
MLFLOW_EXPERIMENT_NAME=/Workspace/Users/your-email@databricks.com/experiments/deep-research-agent

# CORS (comma-separated)
CORS_ORIGINS=http://localhost:5173,http://localhost:3000

# Development Settings
APP_ENV=development
DEBUG=true
LOG_LEVEL=DEBUG
```

**Option B: Direct Token Authentication (Fallback)**

```bash
# .env - Direct token authentication (for local development without Lakebase)

# Databricks Configuration
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=your-personal-access-token

# Local PostgreSQL (when Lakebase not configured)
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/deep_research

# API Keys
BRAVE_API_KEY=your-brave-search-api-key

# Development Settings
APP_ENV=development
DEBUG=true
LOG_LEVEL=DEBUG
```

> **Important**: Profile-based auth (`DATABRICKS_CONFIG_PROFILE`) is required for Lakebase OAuth. Direct token auth works for LLM access but not for Lakebase.

**frontend/.env** (optional - only needed if API URL changes)
```bash
# Not required in most cases - frontend uses relative API paths (/api/v1)
# which work both in development (via Vite proxy) and production (same origin)
# VITE_API_BASE_URL=http://localhost:8000/api/v1
```

### 4. Configure Model Endpoints

Create `backend/config/models.yaml`:

```yaml
# Model Roles - Define defaults for each tier
# Roles reference endpoints by ID; endpoint values override role defaults
model_roles:
  simple:  # Tier 1: Lightweight tasks (query classification, simple responses)
    endpoints: [gpt-4o-mini, claude-haiku]
    temperature: 0.5
    max_tokens: 8000              # Output token limit
    reasoning_effort: low
    rotation_strategy: priority
    fallback_on_429: true

  analytical:  # Tier 2: Medium complexity (research synthesis)
    endpoints: [gpt-4o, gpt-4o-mini]
    temperature: 0.7
    max_tokens: 12000
    reasoning_effort: medium
    rotation_strategy: priority
    fallback_on_429: true

  complex:  # Tier 3: Heavy tasks (deep reasoning, reflection)
    endpoints: [claude-sonnet, gpt-4o]
    temperature: 0.7
    max_tokens: 25000
    reasoning_effort: high
    reasoning_budget: 8000        # Token budget for extended thinking
    rotation_strategy: priority
    fallback_on_429: true

# Endpoints - Define endpoint-specific values (required) and optional overrides
endpoints:
  claude-sonnet:
    endpoint_identifier: databricks-claude-3-5-sonnet
    max_context_window: 200000    # Input context limit (REQUIRED)
    tokens_per_minute: 100000     # Rate limit (REQUIRED)

  gpt-4o:
    endpoint_identifier: databricks-gpt-4o
    max_context_window: 128000
    tokens_per_minute: 150000

  gpt-4o-mini:
    endpoint_identifier: databricks-gpt-4o-mini
    max_context_window: 128000
    tokens_per_minute: 200000

  claude-haiku:
    endpoint_identifier: databricks-claude-3-haiku
    max_context_window: 200000
    tokens_per_minute: 300000
    temperature: 0.3              # Override role default
```

### 5. Initialize Database

```bash
cd backend

# Run Alembic migrations
uv run alembic upgrade head
```

---

## Running Locally

### Development Mode (Two Terminals)

For development with hot module replacement:

**Terminal 1 - Backend:**
```bash
make dev-backend
# Or: cd backend && uv run uvicorn src.main:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
make dev-frontend
# Or: cd frontend && npm run dev
```

Access the application at **http://localhost:5173** (frontend proxies API to backend).

Backend endpoints:
- API: http://localhost:8000/api/v1
- OpenAPI Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### Production Mode (Unified)

Build frontend and run everything from a single server:

```bash
# Build frontend to backend/static/ and start server
make prod
```

Access the application at **http://localhost:8000** (frontend and API on same origin).

This mirrors production deployment where FastAPI serves both the API and static files.

---

## Development Workflow

### Type Checking

```bash
# Backend (strict mode required by constitution)
cd backend
uv run mypy src --strict

# Frontend
cd frontend
npm run typecheck
```

### Linting

```bash
# Backend
cd backend
uv run ruff check src
uv run ruff format src

# Frontend
cd frontend
npm run lint
```

### Testing

The project has a comprehensive testing strategy with three levels:
1. **Python Unit Tests** - Fast, isolated tests with mocks
2. **Frontend Unit Tests** - React component and hook tests
3. **E2E Tests** - Browser-based full-stack tests

#### Python Testing

```bash
# Run all Python unit tests (fast, with mocks)
uv run pytest tests/unit -v

# Run with coverage report
uv run pytest tests/unit --cov=src --cov-report=html

# Run integration tests (requires Databricks credentials)
uv run pytest tests/integration -v

# Run specific test file
uv run pytest tests/unit/services/test_chat_service.py -v

# Run tests matching pattern
uv run pytest -k "test_coordinator" -v
```

**Coverage Thresholds:**
- Services: 80% minimum
- API endpoints: 70% minimum
- Agent nodes: 60% minimum

#### Frontend Testing

```bash
# Run frontend unit tests
cd frontend
npm run test

# Run with UI (interactive)
npm run test:ui

# Run with coverage
npm run test:coverage

# Run in watch mode
npm run test -- --watch
```

### E2E Testing with Playwright

The project includes comprehensive end-to-end tests using Playwright. These tests validate the complete user journey from UI through backend.

```bash
# Build frontend and run E2E tests (recommended)
make e2e

# Run E2E with interactive UI mode (for debugging)
make e2e-ui

# Install Playwright browsers (first time setup)
cd e2e && npm install && npx playwright install

# Run from e2e directory directly
cd e2e
npx playwright test

# Run specific test file
npx playwright test tests/smoke.spec.ts

# Run tests with specific browser
npx playwright test --project=chromium

# Debug mode (step through tests)
npx playwright test --debug
```

**E2E Test Environment Variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `E2E_BASE_URL` | `http://localhost:8000` | Base URL for tests |
| `CI` | `false` | Set to `true` in CI for retries and single worker |
| `E2E_SKIP_LIVE_SERVICES` | `false` | Skip tests requiring live LLM/search services |

**Test Categories:**

| Category | File | Purpose |
|----------|------|---------|
| Smoke | `smoke.spec.ts` | Fast validation (<30s) |
| Research Flow | `research-flow.spec.ts` | Core research with citations |
| Follow-up | `follow-up.spec.ts` | Context-aware responses |
| Stop/Cancel | `stop-cancel.spec.ts` | Cancel operations (<2s) |
| Edit Message | `edit-message.spec.ts` | Edit and regenerate flow |
| Regenerate | `regenerate.spec.ts` | Response regeneration |
| Error Handling | `error-handling.spec.ts` | Edge cases and recovery |

**CI/CD Integration:**

E2E tests run automatically in GitHub Actions on push to main/master and on PRs.
Test reports are uploaded as artifacts and retained for 30 days.

View test results at: `.github/workflows/e2e.yml`

---

## Quick Verification

### 1. Test Backend Health

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "healthy", "version": "1.0.0"}
```

### 2. Test Agent Endpoint

```bash
curl -X POST http://localhost:8000/api/v1/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?"}'
```

Expected: A simple response without deep research (query classified as "simple").

### 3. Test Research Query

```bash
curl -X POST http://localhost:8000/api/v1/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the latest developments in quantum computing error correction?", "research_depth": "light"}'
```

Expected: Research response with citations and sources.

### 4. Test Frontend

1. Open http://localhost:5173 (dev mode) or http://localhost:8000 (prod mode)
2. Verify chat interface loads
3. Send a test message
4. Verify reasoning steps stream in real-time

---

## Troubleshooting

### Common Issues

#### "Databricks authentication failed" or "No Databricks token available"
```bash
# Re-authenticate with profile
databricks auth login --profile deep-research

# Verify token is valid
databricks auth token --profile deep-research

# Check profile is correctly configured
databricks auth describe --profile deep-research
```

#### "DATABRICKS_CONFIG_PROFILE is required for Lakebase auth"
Lakebase OAuth requires profile-based authentication. Make sure:
```bash
# 1. Set DATABRICKS_CONFIG_PROFILE in .env
DATABRICKS_CONFIG_PROFILE=deep-research

# 2. Set LAKEBASE_INSTANCE_NAME (not the full hostname!)
LAKEBASE_INSTANCE_NAME=your-instance-name
```

#### "Database instance is not found"
```bash
# List available Lakebase instances
databricks lakebase instances list

# Get the instance name (NOT the full hostname)
# Example: if host is "msh-deep-research.database.cloud.databricks.com"
# Then LAKEBASE_INSTANCE_NAME=msh-deep-research
```

#### "OpenAIError: The api_key client option must be set"
This means no valid token was found. For profile-based auth:
```bash
# Verify your profile works
databricks auth login --profile deep-research
databricks auth token --profile deep-research

# Then ensure DATABRICKS_CONFIG_PROFILE is set in .env
```

#### "Brave Search API rate limited"
- Check your usage at [api-dashboard.search.brave.com](https://api-dashboard.search.brave.com)
- Consider upgrading from free tier if needed
- System will automatically queue and retry with backoff

#### "Model endpoint not found"
```bash
# List available endpoints
databricks serving-endpoints list

# Update models.yaml with correct endpoint names
```

#### "Type errors in mypy"
```bash
# Ensure all dependencies are installed
uv sync

# Check for stubs
uv add --dev types-requests types-pyyaml

# Run with more verbose output
uv run mypy src --strict --show-error-codes
```

---

## Next Steps

1. **Read the Spec**: [spec.md](./spec.md) - Full feature specification
2. **Review Data Model**: [data-model.md](./data-model.md) - Entity definitions
3. **Explore API**: [contracts/openapi.yaml](./contracts/openapi.yaml) - API contract
4. **Check Research**: [research.md](./research.md) - Technology decisions

---

## Project Structure Reference

```
databricks-deep-research-agent/
├── src/                     # Python backend source
│   ├── agent/               # Plain async Python agent
│   │   ├── orchestrator.py
│   │   ├── state.py
│   │   └── nodes/           # 5 agent nodes
│   ├── api/                 # FastAPI routes
│   │   └── v1/
│   ├── models/              # Pydantic models
│   ├── services/            # Business logic
│   ├── core/                # Config, auth, tracing
│   ├── db/                  # Database utilities
│   └── static_files.py      # SPA serving logic
├── tests/                   # Python tests
│   ├── conftest.py          # Root fixtures
│   ├── unit/                # Unit tests (with mocks)
│   │   ├── conftest.py
│   │   ├── services/
│   │   ├── api/
│   │   ├── agent/
│   │   └── tools/
│   └── integration/         # Integration tests (real services)
├── frontend/                # React frontend source
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── hooks/
│   │   ├── services/
│   │   └── types/
│   ├── tests/               # Frontend unit tests
│   └── package.json
├── e2e/                     # Playwright E2E tests
│   ├── tests/               # Test specs
│   ├── pages/               # Page objects
│   ├── fixtures/
│   └── utils/
├── static/                  # Built frontend (gitignored)
├── pyproject.toml
├── Makefile                 # Build automation
└── specs/
    └── 001-deep-research-agent/
        ├── spec.md          # Feature specification
        ├── plan.md          # Implementation plan
        ├── research.md      # Technology research
        ├── data-model.md    # Entity definitions
        ├── quickstart.md    # This file
        └── contracts/       # API contracts
```
