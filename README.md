# Databricks Deep Research Agent

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![TypeScript](https://img.shields.io/badge/typescript-5.x-blue.svg)](https://www.typescriptlang.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.x-blue.svg)](https://react.dev/)
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)]()

A production-grade multi-agent research system with claim-level citation verification, built on Databricks infrastructure. Combines a 5-agent orchestration architecture with a 7-stage verification pipeline grounded in peer-reviewed research.

## Key Features

- **5-Agent Architecture** - Coordinator, Planner, Researcher, Reflector, and Synthesizer with step-by-step reflection
- **7-Stage Citation Pipeline** - Evidence pre-selection, interleaved generation, confidence classification, isolated verification, citation correction, numeric QA, and ARE-style revision
- **Tiered Query Modes** - Simple (<3s), Web Search (<15s), and Deep Research (<2min) for progressive disclosure
- **Scientific Grounding** - Every factual claim traced to evidence with verification verdicts based on ARE, FActScore, SAFE, CoVe, and ReClaim patterns
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
│  │ Brave Search│                                                            │
│  │     API     │    ┌─────────────────────────────────────────────────────┐ │
│  └─────────────┘    │              MLflow Tracing (3.8+)                  │ │
│                     └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

| Requirement | Version | Purpose |
|------------|---------|---------|
| Python | 3.11+ | Backend runtime |
| Node.js | 18+ | Frontend build & development |
| uv | latest | Python package manager |
| Databricks CLI | latest | Deployment (if deploying to Databricks) |
| Brave Search API key | - | Web search functionality |

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

### One-Command Deployment

```bash
# Deploy to dev workspace (includes all setup)
make deploy TARGET=dev BRAVE_SCOPE=your-secret-scope

# Deploy to production
make deploy TARGET=ais
```

This single command executes the complete 8-step deployment pipeline:

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

2. **Brave API key in secret scope**:
   ```bash
   # Create secret scope (if not exists)
   databricks secrets create-scope your-secret-scope

   # Add Brave API key
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

# Brave Search API
BRAVE_API_KEY=your-brave-api-key

# Optional
APP_CONFIG_PATH=config/app.yaml
LOG_LEVEL=INFO
```

### Databricks Apps (app.yaml)

Environment variables are configured in `app.yaml` for deployed apps:
- `LAKEBASE_INSTANCE_NAME` - Instance name for OAuth token generation
- `LAKEBASE_DATABASE` - Target database name
- `BRAVE_API_KEY` - Injected from secret scope via `valueFrom`
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
uv run python -c "from src.db.session import get_engine; print(get_engine())"

# Verify migrations
uv run alembic current

# Test LLM endpoint
uv run python -c "from src.services.llm.client import LLMClient; ..."

# Check app logs (deployed)
make logs TARGET=dev FOLLOW=-f SEARCH="--search ERROR"
```

### Key Scripts

| Script | Purpose |
|--------|---------|
| `scripts/create-database.sh` | Create deep_research database in Lakebase |
| `scripts/wait-for-lakebase.sh` | Wait for Lakebase instance to be ready |
| `scripts/grant-app-permissions.sh` | Grant table access to app service principal |
| `scripts/download-app-logs.py` | Fetch app logs via REST API |
| `scripts/quickstart.sh` | Set up local development environment |

## Documentation

Comprehensive documentation is available in the [`docs/`](./docs/) folder:

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

## Technology Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Backend | Python | 3.11+ |
| Frontend | TypeScript/React | 18.x |
| Framework | FastAPI | 0.109+ |
| Database | Databricks Lakebase | PostgreSQL |
| LLM Client | AsyncOpenAI | 1.10+ |
| Observability | MLflow | 3.8+ |
| Web Scraping | Trafilatura | 2.0+ |
| Search | Brave Search API | - |

## Configuration

All settings are centralized in `config/app.yaml`:

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

## Project Structure

```
src/                            # Python backend
├── agent/                      # 5-agent orchestration
│   ├── orchestrator.py         # Main async pipeline
│   ├── nodes/                  # Agent implementations
│   └── prompts/                # Agent prompts
├── api/                        # FastAPI routes
├── services/
│   ├── llm/                    # LLM client with tier routing
│   ├── citation/               # 7-stage verification pipeline
│   └── search/                 # Brave Search API
└── db/                         # Lakebase persistence

frontend/                       # React frontend
├── src/
│   ├── components/             # UI components
│   ├── hooks/                  # React hooks (streaming, etc.)
│   └── pages/                  # Page components
└── tests/                      # Vitest tests

e2e/                            # Playwright E2E tests
tests/                          # Python backend tests
config/                         # YAML configuration
docs/                           # Documentation
```

## Contributing

1. Follow the guidelines in [CLAUDE.md](./CLAUDE.md)
2. Ensure all tests pass: `make test-all`
3. Type check: `make typecheck`
4. Lint: `make lint`

## License

Proprietary
