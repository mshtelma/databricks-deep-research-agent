# Deep Research Agent Development Guidelines

## Project Overview

Deep research agent with multi-agent architecture (Coordinator, Planner, Researcher, Reflector, Synthesizer + specialized variants), step-by-step reflection, web search via Brave API, streaming chat UI with persistence on Databricks Lakebase.

**Key Design Decisions:**
- **Plain Async Python** orchestration (NOT LangGraph/DSPy/AutoGen)
- **Step-by-step reflection** after EACH research step (CONTINUE/ADJUST/COMPLETE)
- **Tiered model routing**: simple (fast), analytical (balanced), complex (reasoning)

## Quick Reference - Make Commands

### Development
| Command | Description |
|---------|-------------|
| `make dev` | Run backend (:8000) + frontend (:5173) with hot reload |
| `make dev-backend` | Backend only with hot reload |
| `make dev-frontend` | Frontend only with Vite |
| `make install` | Install all dependencies (backend + frontend + e2e) |
| `make quickstart` | Set up local development environment |

### Database
| Command | Description |
|---------|-------------|
| `make db` | Start local PostgreSQL via Docker + run migrations |
| `make db-stop` | Stop local PostgreSQL |
| `make db-reset` | Reset schema (drops ALL data, recreates tables) |
| `make clean_db` | Delete all chats/messages (preserves schema) |
| `make db-migrate-remote TARGET=dev` | Run migrations on deployed Lakebase |

### Testing
| Command | Description |
|---------|-------------|
| `make test` | Unit tests only (fast, mocked, no credentials) |
| `make test-integration` | Integration tests (real LLM/Brave, requires creds) |
| `make test-complex` | Long-running tests (production config) |
| `make test-all` | All Python + Frontend tests |
| `make test-frontend` | Frontend tests only |
| `make e2e` | Build + run Playwright E2E tests |
| `make e2e-ui` | E2E tests with Playwright UI |

### Quality & Build
| Command | Description |
|---------|-------------|
| `make typecheck` | Type check backend (mypy) + frontend (tsc) |
| `make lint` | Lint backend (ruff) + frontend (eslint) |
| `make format` | Format code (ruff + prettier) |
| `make build` | Build frontend to `static/` |
| `make prod` | Build + run unified production server (:8000) |
| `make clean` | Remove build artifacts |

### Deployment (Databricks Apps)
| Command | Description |
|---------|-------------|
| `make deploy TARGET=dev` | Full deployment (build, migrate, grant, start) |
| `make deploy TARGET=ais` | Deploy to AIS workspace |
| `make logs TARGET=dev` | Download app logs |
| `make logs TARGET=dev FOLLOW=-f` | Follow logs in real-time |
| `make logs TARGET=dev SEARCH="--search ERROR"` | Filter logs |
| `make requirements` | Generate requirements.txt from pyproject.toml |
| `make bundle-validate` | Validate Databricks bundle config |
| `make bundle-summary` | Show deployment summary |

### Direct Commands
```bash
# Pytest with markers
uv run pytest -m "unit"          # Unit tests
uv run pytest -m "integration"   # Integration tests
uv run pytest -m "complex"       # Complex tests

# Individual tools
uv run mypy src/deep_research --strict
uv run ruff check src/deep_research
cd frontend && npm run typecheck

# View logs
tail -f /tmp/deep-research-dev.log   # Dev server logs
tail -f /tmp/deep-research-prod.log  # Prod server logs
```

## Tech Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Backend | Python | 3.11+ |
| Frontend | TypeScript + React | 5.x / 18.x |
| Agent Orchestration | Plain Async Python | - |
| LLM Client | openai (AsyncOpenAI) | 1.10+ |
| Backend Framework | FastAPI | 0.109+ |
| Database | Databricks Lakebase (PostgreSQL) | Preview |
| Database Client | psycopg3 / asyncpg | 3.1+ / 0.29+ |
| Observability | MLflow | 3.8+ |
| UI | Tailwind CSS, TanStack Query | 3.4+ / 5.x |

## Project Structure

```text
src/deep_research/               # Python package (pip-installable)
├── agent/
│   ├── orchestrator.py          # Main async pipeline
│   ├── state.py                 # ResearchState model
│   ├── config.py                # Agent configuration accessors
│   ├── persistence.py           # DB persistence layer
│   ├── nodes/                   # Agent nodes
│   │   ├── coordinator.py       # Query classification
│   │   ├── background.py        # Quick background search
│   │   ├── planner.py           # Research plan generation
│   │   ├── researcher.py        # Classic step execution
│   │   ├── react_researcher.py  # ReAct-style researcher
│   │   ├── reflector.py         # Step-by-step decisions
│   │   ├── synthesizer.py       # Final report generation
│   │   ├── citation_synthesizer.py   # Citation-aware synthesis
│   │   └── react_synthesizer.py      # ReAct synthesis mode
│   ├── prompts/                 # Agent prompt templates
│   ├── tools/                   # web_search, web_crawler, registries
│   └── pipeline/                # Pipeline components
├── api/v1/                      # FastAPI routes
│   ├── utils/                   # Shared auth & transformers
│   └── *.py                     # Endpoint files
├── models/                      # SQLAlchemy models
├── schemas/                     # Pydantic request/response schemas
├── services/                    # Business logic
│   ├── base.py                  # BaseRepository pattern
│   ├── llm/                     # LLM client, rate limiter
│   ├── search/                  # Brave Search API
│   ├── citation/                # Citation pipeline
│   └── *_service.py             # Domain services
├── core/                        # Config, auth, tracing
├── db/                          # Database, migrations
└── main.py                      # FastAPI app entry point

frontend/src/                    # React frontend
├── components/                  # UI components
│   ├── chat/                    # Chat UI components
│   ├── research/                # Research panel components
│   ├── citations/               # Citation display components
│   └── common/                  # Shared components
├── hooks/                       # Custom React hooks
├── pages/                       # Page components
├── api/                         # API client
└── types/                       # TypeScript types

e2e/                             # Playwright E2E tests
├── tests/                       # Test spec files
├── pages/                       # Page Object Models
└── fixtures/                    # Test fixtures

tests/                           # Python tests (3-tier)
├── unit/                        # Fast, mocked tests
├── integration/                 # Real LLM/Brave tests
└── complex/                     # Long-running tests

config/                          # Configuration files
├── app.yaml                     # Production config
├── app.test.yaml                # Test config (minimal iterations)
├── app.e2e.yaml                 # E2E test config
└── app.example.yaml             # Documented example
```

## Key Configuration Files

| File | Purpose |
|------|---------|
| `config/app.yaml` | Central config (endpoints, models, agents, search) |
| `config/app.test.yaml` | Test config (fast models, minimal iterations) |
| `.env` | Environment variables (credentials, secrets) |
| `pyproject.toml` | Python package definition |
| `databricks.yml` | Databricks Asset Bundle config |

## Constitution Principles (MUST FOLLOW)

1. **LLM Calls**: All LLM calls MUST use OpenAI client via WorkspaceClient
2. **Type Annotations**: ALL functions MUST have type annotations
3. **Pydantic Models**: Use for data structures and validation
4. **No Runtime Introspection**: No hasattr/isinstance for type safety
5. **Linting**: mypy strict + ruff MUST pass before merge

## Code Patterns

### Python Backend
- Async/await for all I/O operations
- Pydantic models for request/response schemas
- Dependency injection via `FastAPI Depends()`
- Services extend `BaseRepository[T]` from `services/base.py`
- Use eager-loading from `services/loading.py` to prevent N+1

### API Layer
- Use auth utilities from `api/v1/utils/authorization`
- Use response transformers from `api/v1/utils/transformers`
- Never duplicate `_verify_*` functions in endpoints

### TypeScript Frontend
- Strict mode enabled
- TanStack Query for data fetching
- SSE for streaming (not WebSockets)
- Use `formatActivityLabel()` from `@/utils/activityLabels`

## Authentication

### Local Development
```bash
# .env - Profile-based (recommended)
DATABRICKS_CONFIG_PROFILE=your-profile-name
LAKEBASE_INSTANCE_NAME=your-instance-name
LAKEBASE_DATABASE=deep_research

# OR direct token
DATABRICKS_HOST=https://your-workspace.databricks.com
DATABRICKS_TOKEN=your-token

# OR local PostgreSQL
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/postgres
```

### Lakebase OAuth
- Tokens have 1-hour lifetime with 5-minute refresh buffer
- Username is always `"token"` for OAuth connections
- Host: `{LAKEBASE_INSTANCE_NAME}.database.cloud.databricks.com`

## Deployment Architecture

### Two-Phase Deployment
1. **Bootstrap**: Deploy with `postgres` database (always exists)
2. **Wait**: Lakebase instance becomes ready (~30-60s)
3. **Create**: `deep_research` database via script
4. **Complete**: Re-deploy with `deep_research` as configured database

### Permission Model
```
Developer runs migrations → Tables owned by developer
App has CAN_CONNECT_AND_CREATE → But cannot SELECT/INSERT/UPDATE
Solution: GRANT ALL to app service principal after migrations
```

### Profile Mapping
| TARGET | Profile | Workspace |
|--------|---------|-----------|
| dev | e2-demo-west | E2 Demo West |
| ais | ais | AIS Production |

## YAML Configuration

```yaml
# Model endpoints with rate limits
endpoints:
  databricks-llama-70b:
    endpoint_identifier: databricks-meta-llama-3-1-70b-instruct
    max_context_window: 128000
    tokens_per_minute: 200000

# Model tiers with fallback
models:
  analytical:
    endpoints: [databricks-llama-70b]
    temperature: 0.7
    fallback_on_429: true

# Research depth profiles
research_types:
  light:
    steps: {min: 1, max: 3}
    researcher: {mode: classic}
  extended:
    steps: {min: 5, max: 10}
    researcher: {mode: react, max_tool_calls: 20}
```

### Config Access in Code
```python
from deep_research.agent.config import (
    get_research_type_config,
    get_step_limits,
    get_researcher_config_for_depth,
)
from deep_research.core.app_config import get_app_config
```

## Known Issues

### Web Search Mode Incomplete
- **Status**: Design complete, implementation pending
- **Current**: Falls through to full Deep Research pipeline
- **Expected**: Single-step researcher with limited budget (~15s)

## Researcher Modes
- `classic`: Single-pass with fixed searches/crawls per step (faster)
- `react`: ReAct loop where LLM controls tool calls with budget (more intelligent)

<!-- MANUAL ADDITIONS START -->
<!-- Add project-specific notes here that should persist across updates -->
<!-- MANUAL ADDITIONS END -->