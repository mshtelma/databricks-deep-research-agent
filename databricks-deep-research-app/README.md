# Deep Research App

A production FastAPI + React application for multi-agent research with claim-level citation verification, built on the [databricks-deep-research](../databricks-deep-research/) framework. Features a streaming chat UI, Databricks Lakebase persistence, enterprise data source integration, and MLflow tracing.

## Built on databricks-deep-research

This app uses the **databricks-deep-research** framework for agent orchestration, workflow execution, and citation verification. The framework provides the workflow engine, builtin agents, tool protocol, and streaming events -- the app adds persistence, authentication, a REST API, and a React frontend.

- Framework README: [`../databricks-deep-research/README.md`](../databricks-deep-research/README.md)
- Framework docs (41 files): [`../databricks-deep-research/docs/`](../databricks-deep-research/docs/index.md)

## Features

- **5-Agent Pipeline** -- Coordinator, Planner, Researcher, Reflector, Synthesizer with step-by-step reflection
- **3 Query Modes** -- Simple (<3s), Web Search (<15s), Deep Research (<2min) for progressive disclosure
- **Streaming Chat UI** -- React 18 + TanStack Query with real-time SSE progress updates
- **7-Stage Citation Verification** -- Every factual claim traced to evidence with verification verdicts
- **Custom Agents** -- Reusable research profiles with model, source, and prompt overrides
- **Enterprise Data Sources** -- Vector Search, Genie, Knowledge Assistants via OBO token flow
- **Lakebase Persistence** -- PostgreSQL on Databricks with OAuth token refresh and Alembic migrations
- **MLflow Tracing** -- Full observability with `trace_span` integration

## Quick Start

```bash
# Install all dependencies (backend + frontend + E2E)
make install

# Start development servers
make dev
# Backend at http://localhost:8000
# Frontend at http://localhost:5173
```

## Project Structure

```
src/deep_research/              # Python backend
├── agent/                      # Agent orchestration
│   ├── orchestrator.py         # Main async pipeline (legacy path)
│   ├── framework_orchestrator.py  # Framework path (use_framework=True)
│   ├── adapters/               # Framework <-> App adapters
│   ├── nodes/                  # Agent implementations
│   ├── pipeline/               # Pipeline configuration
│   ├── prompts/                # Agent prompts
│   └── tools/                  # App-specific tools
├── api/v1/                     # FastAPI routes
├── models/                     # SQLAlchemy models
├── schemas/                    # Pydantic schemas
├── services/                   # Business logic
│   ├── llm/                    # LLM client with tier routing
│   ├── citation/               # 7-stage verification pipeline
│   └── search/                 # Brave Search API
├── core/                       # Config, auth, tracing
├── db/                         # Database, migrations
│   └── migrations/versions/    # Alembic migration files
└── main.py                     # FastAPI entry point

frontend/src/                   # React frontend
├── components/                 # UI components
├── hooks/                      # React hooks (streaming, etc.)
├── pages/                      # Page components
└── api/                        # API client

e2e/                            # Playwright E2E tests
tests/                          # Python tests
├── unit/                       # Fast, mocked, no credentials
├── integration/                # Real LLM/Brave, requires creds
└── complex/                    # Long-running production tests
config/                         # YAML configuration
scripts/                        # Utility scripts
```

## Configuration

All settings are centralized in `config/app.yaml` -- model endpoints, research depth profiles, agent behavior.

See the [Configuration Guide](../docs/configuration.md) for details.

## Database

Uses **Databricks Lakebase** (PostgreSQL) with Alembic migrations.

```bash
make db-migrate    # Run migrations on configured Lakebase
make db-status     # Check current migration status
make db-reset      # Reset schema (drops ALL data)
```

Local PostgreSQL fallback: `make db-local` (for offline development only).

## Testing

```bash
make test              # Unit tests (fast, mocked)
make test-integration  # Integration tests (real LLM/Brave)
make test-complex      # Long-running tests
make e2e               # Playwright E2E tests
make test-all          # All Python + Frontend tests
```

## Deployment

```bash
# Deploy to Databricks Apps
make deploy TARGET=dev

# View logs
make logs TARGET=dev FOLLOW=-f
```

See the [Deployment Guide](../docs/deployment.md) for the full pipeline.

## Documentation

- **App documentation**: [`../docs/README.md`](../docs/README.md) -- architecture, API, deployment, configuration
- **Framework documentation**: [`../databricks-deep-research/docs/index.md`](../databricks-deep-research/docs/index.md) -- workflow engine, agents, tools, pools, events, citation pipeline

## License

Proprietary
