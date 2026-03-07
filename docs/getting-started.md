# Getting Started

A step-by-step guide to running the Deep Research Agent locally.

## Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | 3.11+ | Backend runtime |
| Node.js | 18+ | Frontend build |
| [uv](https://docs.astral.sh/uv/) | latest | Python package manager |
| Brave API key | - | Web search (get one at [brave.com/search/api](https://brave.com/search/api/)) |
| Databricks workspace | - | LLM endpoints + Lakebase database |

## Step 1: Clone & Install

```bash
git clone <repo-url>
cd databricks-deep-research-agent

# Install all dependencies (Python backend + Node frontend + Playwright)
make install
```

This runs `uv sync` for the Python backend and `npm install` for the frontend.

## Step 2: Configure Environment

Create a `.env` file in the project root. Choose one of two database options:

### Option A: Remote Lakebase (Recommended)

This is the standard setup — your local dev environment connects to a remote Databricks Lakebase instance.

```bash
# Databricks authentication (pick one)
DATABRICKS_CONFIG_PROFILE=your-profile-name
# OR
DATABRICKS_HOST=https://your-workspace.databricks.com
DATABRICKS_TOKEN=your-token

# Lakebase connection
LAKEBASE_INSTANCE_NAME=your-instance-name
LAKEBASE_DATABASE=deep_research

# Search
BRAVE_API_KEY=your-brave-api-key
```

### Option B: Local PostgreSQL (Fallback)

For offline development or when Lakebase is unavailable:

```bash
# Databricks auth (still needed for LLM endpoints)
DATABRICKS_CONFIG_PROFILE=your-profile-name

# Local database
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/postgres

# Search
BRAVE_API_KEY=your-brave-api-key
```

## Step 3: Set Up Database

### With Remote Lakebase (Option A)

```bash
make db-migrate
```

### With Local PostgreSQL (Option B)

```bash
# Start PostgreSQL in Docker
make db-local

# Run migrations
make db-migrate
```

Verify with `make db-status` to confirm all migrations are applied.

## Step 4: Start Dev Servers

```bash
make dev
```

This starts two servers concurrently:
- **Backend** (FastAPI): http://localhost:8000
- **Frontend** (Vite + React): http://localhost:5173

Both servers support hot reload — code changes are reflected immediately.

## Step 5: Run Your First Research Query

1. Open http://localhost:5173 in your browser
2. Type a research question (e.g., "What are the latest developments in quantum computing?")
3. Select a query mode:
   - **Simple** — direct LLM response (~3s)
   - **Web Search** — quick search with citations (~15s)
   - **Deep Research** — full multi-agent pipeline (~2min)
4. Watch the research process unfold in the activity panel

## Running Tests

```bash
# Unit tests (fast, mocked, no credentials required)
make test

# Integration tests (requires Brave API key + Databricks credentials)
make test-integration

# Frontend tests
make test-frontend

# End-to-end tests (Playwright)
make e2e
```

## Deploying to Databricks

### 1. Configure Databricks CLI

Ensure your Databricks CLI is configured with the target workspace profile:

```bash
databricks configure --profile dev
```

### 2. Add Secrets

Set the Brave API key as a Databricks secret (the deployment reads it from the secret scope).

### 3. Deploy

```bash
make deploy TARGET=dev
```

This builds the frontend, runs database migrations on the remote Lakebase instance, grants permissions to the app service principal, and starts the Databricks App.

For detailed deployment options, see [Deployment](./deployment.md).

## Next Steps

- [Architecture](./architecture.md) — understand the system design and agent pipeline
- [Custom Agents](./custom-agents.md) — create reusable research profiles with model and source overrides
- [Configuration](./configuration.md) — customize models, search, and research depth via YAML
- [Deployment](./deployment.md) — full Databricks Apps deployment guide
- [API Reference](./api.md) — REST endpoints and SSE event types
