# Databricks Deep Research Agent

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![TypeScript](https://img.shields.io/badge/typescript-5.x-blue.svg)](https://www.typescriptlang.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.x-blue.svg)](https://react.dev/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-website-blue.svg)](https://mshtelma.github.io/databricks-deep-research-agent/)

A **uv workspace monorepo** containing two packages: a standalone multi-agent orchestration framework (`databricks-deep-research`) and a production application (`databricks-deep-research-app`) built on it. Features a 5-agent research architecture with a 7-stage citation verification pipeline grounded in peer-reviewed research, deployed on Databricks infrastructure.

> 📖 **[Documentation](https://mshtelma.github.io/databricks-deep-research-agent/)**  ·  🚀 **[Deploy to Databricks](https://mshtelma.github.io/databricks-deep-research-agent/getting-started/deploy/)**  ·  🧩 **[Agent Designer](https://mshtelma.github.io/databricks-deep-research-agent/concepts/agent-designer/)**

## Key Features

- **Agent Designer** - Build a multi-agent research workflow from a one-line prompt in a chat + canvas UI, then deploy it to any of five runtime targets ([designer](docs/agent-designer.md) · [deployment](docs/agent-deployment.md))
- **YAML Workflow Engine** - Declarative research pipelines with 8 node types and plan-and-execute orchestration ([framework docs](databricks-deep-research/docs/index.md))
- **5-Agent Architecture** - Coordinator, Planner, Researcher, Reflector, and Synthesizer with step-by-step reflection
- **7-Stage Citation Pipeline** - Evidence pre-selection, interleaved generation, confidence classification, isolated verification, citation correction, numeric QA, and ARE-style revision
- **Multi-Provider Web Search** - Databricks built-in web search (default, no external key), Brave, or Jina behind one `SearchClient` protocol ([search providers](databricks-deep-research/docs/guides/search-providers.md))
- **Delta / SQL Table Tools** - Structured research directly over Delta tables with the `table_*` tool family ([SQL table tools](databricks-deep-research/docs/guides/sql-table-tools.md))
- **Tiered Query Modes** - Simple (<3s), Web Search (<15s), and Deep Research (<2min) for progressive disclosure
- **Dataflow Validation** - Workflows are checked for dangling reads and dead stores at load time ([dataflow validation](databricks-deep-research/docs/guides/dataflow-validation.md))
- **Scientific Grounding** - Every factual claim traced to evidence with verification verdicts based on ARE, FActScore, SAFE, CoVe, and ReClaim patterns
- **Conversation Memory & HITL** - Chat memory (entities, files, findings, coverage) and human-in-the-loop approval gates
- **Real-time Streaming** - Server-Sent Events (SSE) for live research progress updates
- **Enterprise Ready** - OAuth token refresh, automatic failover, atomic persistence on Databricks Lakebase

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DEEP RESEARCH AGENT                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────────────────────────────────────────────┐ │
│  │   Frontend  │    │                 5-Agent Orchestrator                │ │
│  │  React 18   │◀──▶│  ┌───────────┐ ┌─────────┐ ┌────────────┐          │ │
│  │  TanStack   │SSE │  │Coordinator│→│ Planner │→│ Researcher │          │ │
│  │  Tailwind   │    │  └───────────┘ └─────────┘ └────────────┘          │ │
│  └─────────────┘    │        ↓                         ↓                  │ │
│                     │  ┌───────────┐           ┌────────────┐             │ │
│  ┌─────────────┐    │  │ Reflector │◀──────────│ Synthesizer│             │ │
│  │  FastAPI    │    │  └───────────┘           └────────────┘             │ │
│  │  REST API   │◀──▶│        │                       │                    │ │
│  │  /v1/...    │    │        └───────────────────────┘                    │ │
│  └─────────────┘    │              7-Stage Citation Pipeline               │ │
│                     └─────────────────────────────────────────────────────┘ │
│  ┌─────────────┐    ┌─────────────────────────────────────────────────────┐ │
│  │ Lakebase    │◀──▶│     Databricks Foundation Model Endpoints           │ │
│  │ PostgreSQL  │    │  ┌─────────┐ ┌────────────┐ ┌───────────┐          │ │
│  │ Persistence │    │  │ Simple  │ │ Analytical │ │  Complex  │          │ │
│  └─────────────┘    │  │ (Gemini)│ │  (Claude)  │ │(Claude ER)│          │ │
│                     │  └─────────┘ └────────────┘ └───────────┘          │ │
│  ┌─────────────┐    └─────────────────────────────────────────────────────┘ │
│  │ Web Search  │                                                            │
│  │  providers  │    ┌─────────────────────────────────────────────────────┐ │
│  └─────────────┘    │              MLflow Tracing (3.8+)                  │ │
│                     └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Agent Designer

The **Agent Designer** turns a one-line prompt ("a research agent over our docs index plus public web, with reflection, under 30s") into a runnable multi-agent workflow. An LLM *architect* assembles a typed workflow AST through tool calls; a chat panel and a visual canvas are two synchronized views over that same AST. YAML and Mermaid are deterministic *export* formats — the AST (stored as JSONB on Lakebase, with immutable revisions) is the source of truth.

A designed agent deploys to five runtime targets from the same AST: **In-App** (run inside this app), **MLflow Agent** (Model Serving endpoint), **Shell App** (standalone Databricks App with the framework bundled as a wheel, per-request OBO), **Spark Batch** (run over a Delta table column), and **direct programmatic serving**.

- [Agent Designer](docs/agent-designer.md) — authoring, the design brief, designer tools, topologies, and the chat/canvas UI.
- [Deploying a Designed Agent](docs/agent-deployment.md) — the five deploy targets, the deployment API, and pre-flight permission probes.

## Quick Start

### Prerequisites

| Requirement | Version | Purpose |
|------------|---------|---------|
| Python | 3.11+ | Backend runtime |
| Node.js | 18+ | Frontend build & development |
| uv | latest | Python package manager |
| Databricks CLI | latest | Deployment (if deploying to Databricks) |

> Web search works out of the box via Databricks' built-in search — **no external search key required**. Brave/Jina are opt-in (see [Web Search Providers](https://mshtelma.github.io/databricks-deep-research-agent/guides/web-search-providers/)).

### Local Development

```bash
# Install all dependencies (backend, frontend, E2E)
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

# Run production server (serves UI from static/)
make prod

# Access at http://localhost:8000
```

## Databricks Deployment

> 📖 **Full step-by-step guide:** <https://mshtelma.github.io/databricks-deep-research-agent/getting-started/deploy/>

### One-Command Deployment

```bash
# Deploy to your workspace (build + Lakebase + migrations + start)
make deploy TARGET=dev

# BRAVE_SCOPE is optional — only if a web tool pins provider: brave
# make deploy TARGET=dev BRAVE_SCOPE=your-secret-scope
```

This single command executes the complete 9-step deployment pipeline:

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Full Deployment Pipeline                             │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Step 1: Build frontend (npm run build → static/)                      │
│     ↓                                                                  │
│  Step 2: Generate requirements.txt from pyproject.toml                  │
│     ↓                                                                  │
│  Step 3: Bootstrap deploy with postgres (creates Lakebase instance)     │
│     ↓                                                                  │
│  Step 4: Wait for Lakebase to be ready (~30-60s for new instances)     │
│     ↓                                                                  │
│  Step 5: Create deep_research database                                  │
│     ↓                                                                  │
│  Step 6: Re-deploy bundle with deep_research database                   │
│     ↓                                                                  │
│  Step 7: Run migrations with developer credentials                      │
│     ↓                                                                  │
│  Step 8: Grant table permissions to app service principal               │
│     ↓                                                                  │
│  Step 9: Start app and show deployment summary                          │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

### Prerequisites for Databricks Deployment

1. **Databricks CLI configured** with workspace profiles:
   ```bash
   # Configure CLI with your workspace
   databricks configure --profile e2-demo-west

   # Verify configuration
   databricks auth describe --profile e2-demo-west
   ```

2. **(Optional) Brave API key** — only if you set `provider: brave`. The default `databricks` provider uses built-in web search and needs no key:
   ```bash
   databricks secrets create-scope your-secret-scope
   databricks secrets put-secret your-secret-scope BRAVE_API_KEY
   ```

3. **Model endpoints available** in your workspace:
   - `databricks-claude-sonnet-4-5` (analytical tier)
   - `databricks-claude-haiku-4-5` (simple tier)
   - `databricks-claude-opus-4-5` (complex tier)

### Target Workspace Mapping

| TARGET | CLI Profile | Description |
|--------|-------------|-------------|
| `dev` | `e2-demo-west` | Development workspace |
| `ais` | `ais` | Production workspace |

### Operations Commands

```bash
# View application logs
make logs TARGET=dev                    # Fetch logs once
make logs TARGET=dev FOLLOW=-f          # Follow logs in real-time
make logs TARGET=dev SEARCH="--search ERROR"  # Filter by term

# Restart app after config changes
databricks bundle run -t dev deep_research_agent

# Check deployment status
databricks bundle summary -t dev

# Run migrations manually (usually not needed)
make db-migrate-remote TARGET=dev
```

### Why Two-Phase Deployment?

Deploying to Databricks Apps with Lakebase requires solving a chicken-and-egg problem:

1. The app needs `LAKEBASE_DATABASE=deep_research` environment variable
2. The database doesn't exist until the Lakebase instance is created
3. The Lakebase instance is created by the bundle deploy

**Solution**: Deploy twice - first with `postgres` (always exists), then with `deep_research` after creating the database.

### Permission Model

Tables are owned by the developer who runs migrations, not the app's service principal. The app needs explicit GRANT statements to access tables:

```sql
GRANT ALL ON ALL TABLES IN SCHEMA public TO <app_service_principal>;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO <app_service_principal>;
```

This is handled automatically by `scripts/grant-app-permissions.sh` during deployment.

## Environment Configuration

### Local Development (.env file)

```bash
# Databricks Authentication (choose one)
DATABRICKS_CONFIG_PROFILE=e2-demo-west  # Recommended: profile-based
# OR
DATABRICKS_HOST=https://your-workspace.databricks.com
DATABRICKS_TOKEN=your-personal-access-token

# Lakebase (when using Databricks Lakebase)
LAKEBASE_INSTANCE_NAME=deep-research-lakebase
LAKEBASE_DATABASE=deep_research

# OR Local PostgreSQL (alternative for local dev)
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/deep_research

# Web search: default provider is `databricks` (built-in, no key needed).
# Only set this when using provider: brave:
# BRAVE_API_KEY=your-brave-api-key

# Optional
APP_CONFIG_PATH=config/app.yaml
LOG_LEVEL=INFO
```

### Databricks Apps (app.yaml)

Environment variables are configured in `app.yaml` for deployed apps:
- `LAKEBASE_INSTANCE_NAME` - Instance name for OAuth token generation
- `LAKEBASE_DATABASE` - Target database name
- `BRAVE_API_KEY` - Optional; injected from secret scope only when using `provider: brave`
- `MLFLOW_TRACKING_URI=databricks` - Automatic tracing

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `InvalidPasswordError` after ~1 hour | OAuth token expired | Fixed in session.py - tokens auto-refresh |
| Tables not accessible by app | Tables owned by developer | Run `grant-app-permissions.sh` |
| Database not found during deploy | Two-phase deploy incomplete | Let `make deploy` complete all steps |
| Rate limit errors (429) | LLM endpoint throttled | Automatic retry with exponential backoff |
| Migrations fail | Wrong profile/credentials | Check `DATABRICKS_CONFIG_PROFILE` |

### Debugging Commands

```bash
# Check database connectivity
uv run python -c "from deep_research.db.session import get_engine; print(get_engine())"

# Verify migrations
uv run alembic current

# Test LLM endpoint
uv run python -c "from deep_research.services.llm.client import LLMClient; ..."

# Check app logs (deployed)
make logs TARGET=dev FOLLOW=-f SEARCH="--search ERROR"
```

### Key Scripts

| Script | Purpose |
|--------|---------|
| `scripts/quickstart.sh` | Set up local development environment |
| `scripts/grant-app-permissions.sh` | Grant table access to app service principal |
| `scripts/download-app-logs.py` | Fetch app logs via REST API |
| `scripts/clean-db.sh` | Clean database data |
| `scripts/db-cleanup.py` | Remove orphaned provisioning resources |
| `scripts/kill-server.sh` | Kill running dev server |
| `scripts/purge_deleted_chats.py` | Purge soft-deleted chats |
| `scripts/analyze_traces.py` | Analyze MLflow traces |

## Documentation

📖 **Full documentation site:** <https://mshtelma.github.io/databricks-deep-research-agent/>

### Framework Documentation

The **databricks-deep-research** framework ships extensive documentation covering the workflow engine, agent harness, tool protocol, pool system, streaming events, search providers, SQL/table tools, dataflow validation, and the citation pipeline.

Full index: [`databricks-deep-research/docs/index.md`](databricks-deep-research/docs/index.md)

| Track | Time | What You'll Learn |
|-------|------|-------------------|
| **Quick Start** | ~15 min | Installation, first workflow, architecture overview |
| **Workflow Builder** | ~1-2 hours | YAML authoring, builtin agents/tools, pools, events |
| **Deep Dive** | ~half day | Custom tools/agents, citation pipeline, full reference |

### Application Documentation

| Document | Description |
|----------|-------------|
| [Architecture](./docs/architecture.md) | System design, technology stack, key decisions |
| [5-Agent System](./docs/agents.md) | Agent responsibilities, orchestration flow, state management |
| [Citation Pipeline](./docs/citation-pipeline.md) | 7-stage verification pipeline with scientific foundations |
| [LLM Interaction](./docs/llm-interaction.md) | Model tier routing, structured output, ReAct pattern |
| [Scientific Foundations](./docs/scientific-foundations.md) | Research papers and how they're applied |
| [Configuration](./docs/configuration.md) | YAML config system, environment variables |
| [Data Models](./docs/data-models.md) | Entity definitions and relationships |
| [API Reference](./docs/api.md) | REST endpoints and SSE event types |
| [Deployment](./docs/deployment.md) | Databricks Apps deployment guide |
| [Agent Designer](./docs/agent-designer.md) | Chat-based workflow authoring: design brief, designer tools, topologies, UI |
| [Agent Deployment](./docs/agent-deployment.md) | Deploying a designed agent to its five runtime targets |

## Technology Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Backend | Python | 3.11+ |
| Frontend | TypeScript/React | 18.x |
| Orchestration | databricks-deep-research | 0.2.0 |
| Framework | FastAPI | 0.109+ |
| Database | Databricks Lakebase | PostgreSQL |
| LLM Client | AsyncOpenAI | 1.10+ |
| Observability | MLflow | 3.8+ |
| Web Scraping | Trafilatura | 2.0+ |
| Web Search | Databricks built-in (default) · Brave · Jina | - |

## Configuration

All settings are centralized in `databricks-deep-research-app/config/app.yaml`:

```yaml
# Model tier routing
models:
  simple:
    endpoints: [databricks-gemini-flash]
  analytical:
    endpoints: [databricks-claude-sonnet]
  complex:
    endpoints: [databricks-claude-sonnet-er]

# Research depth profiles
research_types:
  light:   { steps: {min: 1, max: 3}, researcher: {mode: classic} }
  medium:  { steps: {min: 3, max: 6}, researcher: {mode: react} }
  extended: { steps: {min: 5, max: 10}, researcher: {mode: react} }
```

See [Configuration Guide](./docs/configuration.md) for full details.

## Testing

```bash
# Unit tests (fast, no credentials)
make test

# Integration tests (real LLM/Brave)
make test-integration

# Complex long-running tests
make test-complex

# E2E Playwright tests
make e2e

# All tests
make test-all
```

## Scientific Foundations

The citation verification pipeline implements patterns from peer-reviewed research:

| Pattern | Paper | Application |
|---------|-------|-------------|
| **ReClaim** | [arXiv:2407.01796](https://arxiv.org/abs/2407.01796) | Interleaved generation with evidence constraints |
| **FActScore** | [arXiv:2305.14251](https://arxiv.org/abs/2305.14251) | Atomic fact decomposition |
| **SAFE** | [arXiv:2403.18802](https://arxiv.org/abs/2403.18802) | Multi-step reasoning with search |
| **ARE** | [arXiv:2410.16708](https://arxiv.org/abs/2410.16708) | Atomic facts for retrieval |
| **CoVe** | [arXiv:2309.11495](https://arxiv.org/abs/2309.11495) | Isolated verification |
| **CiteFix** | [arXiv:2504.15629](https://arxiv.org/abs/2504.15629) | Hybrid citation correction |
| **QAFactEval** | [arXiv:2112.08542](https://arxiv.org/abs/2112.08542) | QA-based numeric verification |

See [Scientific Foundations](./docs/scientific-foundations.md) for detailed explanations.

## Repository Structure

```
pyproject.toml                          # uv workspace root
databricks-deep-research/               # Framework package (v0.2.0)
├── src/databricks_deep_research/       # Workflow engine, agents, tools, pools, citation
├── docs/                               # Framework docs (concepts, guides, reference, security)
├── examples/                           # YAML workflow examples
└── tests/
databricks-deep-research-app/           # Application package
├── src/deep_research/                  # FastAPI backend
├── frontend/                           # React frontend
├── e2e/                                # Playwright E2E tests
├── config/                             # YAML configuration
├── scripts/                            # Utility scripts
└── tests/
docs/                                   # Application documentation
examples/sales-research-agent/          # Example extension project
specs/                                  # Feature specifications
```

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for the full guide, and [SECURITY.md](./SECURITY.md) to report vulnerabilities.

1. Follow the guidelines in [CLAUDE.md](./CLAUDE.md)
2. Ensure all tests pass: `make test-all`
3. Type check: `make typecheck`
4. Lint: `make lint`

## License

Licensed under the [Apache License 2.0](LICENSE). Copyright &copy; 2026 the Databricks Deep Research Agent contributors.
