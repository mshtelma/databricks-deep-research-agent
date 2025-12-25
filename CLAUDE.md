# Deep Research Agent Development Guidelines

Auto-generated from feature plans. Last updated: 2025-12-21

## Project Overview

Deep research agent with **5-agent architecture** (Coordinator, Planner, Researcher, Reflector, Synthesizer), step-by-step reflection, web search via Brave API, streaming chat UI with persistence on Databricks Lakebase.

**Key Design Decisions:**
- **Plain Async Python** orchestration (NOT LangGraph/DSPy/AutoGen)
- **Step-by-step reflection** after EACH research step (CONTINUE/ADJUST/COMPLETE)
- **Tiered model routing**: simple (fast), analytical (balanced), complex (reasoning)

## Active Technologies
- Python 3.11+ (backend), TypeScript 5.x (frontend) + FastAPI, httpx, openai (with custom base_url), Pydantic v2, React 18, TanStack Query, trafilatura (001-deep-research-agent)
- Databricks Lakebase (PostgreSQL) via psycopg3/asyncpg (001-deep-research-agent)
- TypeScript 5.x (Playwright tests) + Playwright (latest), @playwright/tes (001-deep-research-agent)
- N/A (tests run against existing application) (001-deep-research-agent)
- Python 3.11+ (backend), TypeScript 5.x (frontend, E2E) (001-deep-research-agent)
- Mocked (unit tests), Real PostgreSQL/Lakebase (integration tests optional) (001-deep-research-agent)
- Python 3.11+ + PyYAML, Pydantic v2, FastAPI (001-deep-research-agent)
- YAML file (`config/app.yaml`), Pydantic models for validation (001-deep-research-agent)

| Component | Technology | Version |
|-----------|------------|---------|
| Backend Language | Python | 3.11+ |
| Frontend Language | TypeScript | 5.x |
| Agent Orchestration | Plain Async Python | N/A |
| LLM Client | openai (AsyncOpenAI) | 1.10+ |
| HTTP Client | httpx | 0.26+ |
| Backend Framework | FastAPI | 0.109+ |
| Frontend Framework | React | 18.x |
| Database | Databricks Lakebase (PostgreSQL) | Preview |
| Database Client | psycopg3/asyncpg | 3.1+/0.29+ |
| Observability | MLflow | 3.8+ |
| UI Components | Tailwind CSS | 3.4+ |
| State Management | TanStack Query | 5.x |
| JSON Repair | json-repair | 0.25+ |
| Web Content Extraction | trafilatura | 2.0+ |

## Project Structure

```text
src/                            # Python backend source
├── agent/
│   ├── orchestrator.py         # Main async pipeline
│   ├── state.py                # ResearchState model
│   ├── nodes/                  # 5 agent nodes
│   │   ├── coordinator.py      # Query classification
│   │   ├── background.py       # Quick web search
│   │   ├── planner.py          # Research plan
│   │   ├── researcher.py       # Step execution
│   │   ├── reflector.py        # Step-by-step decisions
│   │   └── synthesizer.py      # Final report
│   ├── prompts/                # Agent prompt templates
│   └── tools/                  # web_search, web_crawler
├── api/                        # FastAPI routes
├── models/                     # Pydantic models
├── services/
│   ├── llm/                    # LLM client, rate limiter, model router
│   ├── search/                 # Brave Search API
│   └── storage/                # Session/message repos
├── core/                       # Config, auth, tracing
├── static_files.py             # SPA serving for production
├── db/                         # Database connection
└── main.py                     # FastAPI app entry point

frontend/                       # React frontend source
├── src/
│   ├── components/
│   │   ├── chat/               # ChatContainer, MessageList, etc.
│   │   ├── research/           # ReasoningPanel, PlanProgress
│   │   └── common/             # Shared components
│   ├── pages/
│   ├── hooks/                  # useStreamingQuery, useSession
│   ├── services/               # API client, SSE handling
│   └── types/
└── tests/                      # Vitest unit tests

e2e/                            # Playwright E2E tests (full-stack)
├── tests/                      # Test spec files
├── pages/                      # Page Object Models
├── fixtures/                   # Test fixtures
└── playwright.config.ts

tests/                          # Python unit tests
static/                         # Built frontend (gitignored, created by `make build`)
```

**Deployment**: Frontend builds to `static/` and is served by FastAPI. No Node.js runtime in production.

## Commands

```bash
# Development (two terminals)
make dev                         # Terminal 1: Backend with hot reload (:8000)
make dev-frontend                # Terminal 2: Frontend with hot reload (:5173)

# Production build
make build                       # Build frontend to static/
make prod                        # Run unified server on :8000

# E2E Testing (single command!)
make e2e                         # Build + start server + run Playwright tests
make e2e-ui                      # Run E2E tests with Playwright UI

# Quality checks
make typecheck                   # Type check backend + frontend
make lint                        # Lint backend + frontend
make test                        # Run Python tests
make test-frontend               # Run frontend tests

# Individual tools (if needed)
uv run mypy src --strict
uv run ruff check src
cd frontend && npm run typecheck
```

## Constitution Principles (MUST FOLLOW)

### I. Clients and Workspace Integration
- All LLM calls MUST use OpenAI client via WorkspaceClient
- All Databricks access MUST use WorkspaceClient
- No direct API calls bypassing these clients

### II. Typing-First Python
- ALL functions MUST have type annotations
- Use Pydantic models for data structures
- Use TypedDict/dataclasses for complex types

### III. Avoid Runtime Introspection
- NO hasattr/isinstance for type safety
- Use Pydantic for validation at boundaries
- Prefer explicit interfaces/Protocols

### IV. Linting and Static Type Enforcement
- mypy strict mode MUST pass before merge
- ruff MUST pass with no errors
- `# type: ignore` requires justification comment

## Code Style

### Python (Backend)
- Use async/await for all I/O operations
- Pydantic models for all request/response schemas
- Dependency injection via FastAPI Depends()
- SQLAlchemy async for database operations

### TypeScript (Frontend)
- Strict mode enabled
- Use TanStack Query for data fetching
- SSE for streaming (not WebSockets)
- Shadcn/UI components for accessibility
- Activity labels: Use `formatActivityLabel()` and `getActivityColor()` from `@/utils/activityLabels` for event display

## Key Files

- `config/app.yaml` - Central configuration (endpoints, models, agents, search)
- `config/app.example.yaml` - Documented example configuration
- `src/core/app_config.py` - Pydantic configuration models
- `specs/001-deep-research-agent/spec.md` - Feature specification
- `specs/001-deep-research-agent/plan.md` - Implementation plan
- `specs/001-deep-research-agent/data-model.md` - Entity definitions
- `specs/001-deep-research-agent/contracts/openapi.yaml` - API contract
- `.specify/memory/constitution.md` - Project principles

## Central YAML Configuration

All model endpoints, roles, agent limits, and search settings are configured in `config/app.yaml`.

### Configuration Structure

```yaml
# Default model role (simple, analytical, complex, or custom)
default_role: analytical

# Model endpoints with rate limits and capabilities
endpoints:
  databricks-llama-70b:
    endpoint_identifier: databricks-meta-llama-3-1-70b-instruct
    max_context_window: 128000
    tokens_per_minute: 200000
    supports_structured_output: true

# Model roles (tiers) with priority-ordered endpoints
models:
  analytical:
    endpoints:
      - databricks-llama-70b
    temperature: 0.7
    max_tokens: 8000
    reasoning_effort: medium
    fallback_on_429: true

# Agent configuration
agents:
  researcher:
    max_search_queries: 2
    max_urls_to_crawl: 3
  planner:
    max_plan_iterations: 3
  coordinator:
    enable_clarification: true

# Search configuration
search:
  brave:
    requests_per_second: 1.0
    default_result_count: 10
```

### Environment Variable Interpolation

```yaml
# Required variable (fails if not set)
endpoint_identifier: ${MODEL_ENDPOINT}

# Optional with default
endpoint_identifier: ${MODEL_ENDPOINT:-databricks-llama-70b}
```

### Accessing Configuration in Code

```python
# LLM service loads from central config automatically
from src.services.llm.config import ModelConfig
model_config = ModelConfig()  # Loads from app.yaml

# Agent configuration accessors
from src.agent.config import get_researcher_config, get_planner_config
researcher_config = get_researcher_config()
max_queries = researcher_config.max_search_queries

# Direct app config access
from src.core.app_config import get_app_config
app_config = get_app_config()
default_role = app_config.default_role
```

## Authentication Configuration

### Profile-Based Auth (Recommended)

```bash
# .env - Profile-based authentication (preferred)
DATABRICKS_CONFIG_PROFILE=your-profile-name

# Lakebase Configuration
LAKEBASE_INSTANCE_NAME=your-instance-name  # NOT the full hostname
LAKEBASE_DATABASE=deep_research
```

### Direct Token Auth (Fallback)

```bash
# .env - Direct token authentication (fallback)
DATABRICKS_HOST=https://your-workspace.databricks.com
DATABRICKS_TOKEN=your-personal-access-token

# For local PostgreSQL (when Lakebase not configured)
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/deep_research
```

### Key Authentication Patterns

**LLM Client** (`src/services/llm/client.py`):
- Supports both `DATABRICKS_TOKEN` and profile-based auth
- Profile auth: `WorkspaceClient(profile=...).config.authenticate()` then `oauth_token().access_token`
- Uses `AsyncOpenAI` with custom `base_url` pointing to Databricks serving endpoints

**Lakebase** (`src/db/lakebase_auth.py`):
- Requires `DATABRICKS_CONFIG_PROFILE` for OAuth token generation
- Uses `WorkspaceClient.database.generate_database_credential(instance_names=[...])`
- OAuth username is always `"token"`, actual token is the password
- Tokens auto-refresh (1-hour lifetime, 5-minute buffer)
- Host derived from instance name: `{LAKEBASE_INSTANCE_NAME}.database.cloud.databricks.com`

## Recent Changes
- 001-deep-research-agent: Cross-cutting backend services (2025-12-25)
  - Added `PreferencesService` for managing user preferences (get/update)
  - Added `FeedbackService` for message feedback with MLflow trace correlation
  - Added `ExportService` for chat export to Markdown/JSON formats
  - Added system_instructions support in ResearchState and OrchestrationConfig
  - System instructions from user preferences now applied to all agent prompts
  - Added `build_system_prompt()` utility for consistent instruction injection
  - Updated research API endpoints to fetch and pass user preferences

- 001-deep-research-agent: LLM context truncation (2025-12-25)
  - Added `src/services/llm/truncation.py` with intelligent message truncation
  - Preserves system prompt and recent messages when context exceeds limits
  - `truncate_messages()` and `truncate_text()` with configurable limits
  - `get_context_window_for_request()` for endpoint-aware truncation

- 001-deep-research-agent: Researcher validation error resilience (2025-12-25)
  - Added default value to ResearcherOutput.observation field for validation resilience
  - Enhanced researcher prompt with explicit "ALWAYS REQUIRED" observation guidance
  - Added instructions for handling limited/empty search results
  - Updated output schema comments to emphasize observation is required
  - Fixes: LLM validation errors when search results are limited or empty

- 001-deep-research-agent: Multi-entity query decomposition (2025-12-25)
  - Enhanced planner prompt with entity-by-entity decomposition guidance
  - Added "Multi-Entity Query Handling" section with CRITICAL RULE to never bundle entities
  - Increased step limit from 2-8 to 2-15 for multi-entity comparisons
  - Updated researcher search guidelines with "Entity-Focused Search Strategy"
  - Enhanced SEARCH_QUERY_PROMPT with entity isolation rules and diverse examples
  - Fixes: broad multi-entity queries now decomposed into focused entity-specific steps

- 001-deep-research-agent: MLflow 3.8+ upgrade and streaming trace fix (2025-12-25)
  - Upgraded MLflow from 2.10+ to 3.8+ for production-grade tracing support
  - Added `@mlflow.trace` decorator to `stream_research()` in orchestrator.py
  - Fixed traces not appearing from web app - `stream_research()` now has root span like `run_research()`
  - Added root span attributes (session_id, query, max_iterations, streaming) for trace correlation
  - `mlflow.update_current_trace()` now called inside span context (was outside before)
  - Added `mlflow.flush_trace_async_logging()` after streaming completes in API layer
  - Fixes: orphan traces, context propagation issues, traces not flushed before response ends

- 001-deep-research-agent: LLM rate limit retry with configurable backoff (2025-12-25)
  - Added automatic retry with backoff when rate limits are hit at the LLM client layer
  - All agents benefit automatically without code changes
  - Configurable via `rate_limiting` section in `config/app.yaml`
  - Supports both exponential (`2^n`) and linear (`n+1`) backoff strategies
  - Smart wait time calculation using endpoint health state
  - New `RateLimitingConfig` model with `calculate_delay()` method
  - Refactored `LLMClient.complete()` and `LLMClient.stream()` with retry wrappers

- 001-deep-research-agent: MLflow trace session grouping (2025-12-24)
  - Added `user_id` and `chat_id` parameters to `run_research()` and `stream_research()` functions
  - Traces from the same chat conversation are now grouped together via `mlflow.update_current_trace()`
  - Uses `mlflow.trace.user` (user_id) and `mlflow.trace.session` (chat_id) metadata
  - API layer passes user context to orchestrator for trace correlation
  - New FR-099 in spec.md documenting the requirement

- 001-deep-research-agent: MLflow tracing fixes (2025-12-24)
  - Enabled async logging via `mlflow.config.enable_async_logging(True)` for FastAPI context
  - Added MLflow spans to `LLMClient.complete()` with tier, endpoint, and token metrics
  - Added root span attributes (session_id, query, max_iterations) in orchestrator
  - LLM calls now visible in MLflow traces as `llm_simple`, `llm_analytical`, `llm_complex` spans

- 001-deep-research-agent: Chat UX improvements (2025-12-24)
  - Auto-select first chat or create new one when navigating to home page without a chat
  - Changed "Untitled Chat" label to "New chat..." with italic/muted styling
  - Fixed plan step status flickering by clearing currentStepIndex on step completion
  - New FR-093, FR-094, FR-095 in spec.md

- 001-deep-research-agent: LLM-based search query generation in Background Investigator (2025-12-24)
  - Fixed HTTP 422 errors from Brave Search when user queries are too long
  - Added `BACKGROUND_SEARCH_PROMPT` in `src/agent/prompts/background.py`
  - Background Investigator now uses LLM to generate 2-3 focused search queries
  - Added `BackgroundConfig` in `src/core/app_config.py` with configurable limits
  - Added background config section to `config/app.yaml`
  - Fallback to truncated query (200 chars) if LLM query generation fails
  - New FR-092 in spec.md documenting the requirement

- 001-deep-research-agent: Markdown rendering in agent messages (2025-12-24)
  - Added `frontend/src/components/common/MarkdownRenderer.tsx` - reusable markdown component
  - Uses `react-markdown` with `remark-gfm` for GitHub Flavored Markdown support
  - Added `react-syntax-highlighter` for code block syntax highlighting (vscDarkPlus/vs themes)
  - Updated `AgentMessage.tsx` to render content with full markdown support
  - Enhanced CSS prose styles for tables, blockquotes, headings, strikethrough
  - Links open in new tabs with `target="_blank" rel="noopener noreferrer"`
  - Streaming content renders as markdown in real-time

- 001-deep-research-agent: Research Activity panel improvements (2025-12-24)
  - Added `frontend/src/utils/activityLabels.ts` with event formatting utilities
  - `formatActivityLabel()`: Converts raw event types to human-readable labels with emojis
  - `getActivityColor()`: Returns Tailwind color classes (green/amber/blue/red) by event status
  - Updated `ChatPage.tsx` to use the new formatters for the activity log

- 001-deep-research-agent: Central YAML configuration (2025-12-24)
  - Added `config/app.yaml` for all model endpoints, roles, agents, and search settings
  - Environment variable interpolation: `${VAR}` and `${VAR:-default}`
  - Pydantic v2 models for configuration validation in `src/core/app_config.py`
  - Agent config accessors in `src/agent/config.py`
  - Startup validation fails fast if config is invalid
  - All hardcoded values removed from agent nodes and services

- 001-deep-research-agent: Critical codebase review and fixes (2025-12-24)
  - **Security**: Added authorization checks to all 9 API endpoints (messages.py, research.py)
  - **Data integrity**: Fixed model/migration column name mismatches (Source, UserPreferences, MessageFeedback, AuditLog)
  - **Transaction safety**: Added flush after delete operations, rollback on errors
  - **Bug fixes**: Fixed UUID collision risk (now uses uuid5), state mutation during streaming
  - **Race conditions**: Added refresh after JSONB updates in research_session_service.py
  - **Deprecations**: Replaced all datetime.utcnow() with datetime.now(UTC) across codebase
  - **Schema updates**: FeedbackRating now uses string values ("positive"/"negative") matching migration

- 001-deep-research-agent: Added Python 3.11+ (backend), TypeScript 5.x (frontend, E2E)
  - Moved `backend/src/` to `/src/` (Python at root)
  - Moved `backend/tests/` to `/tests/`
  - Moved `frontend/e2e/` to `/e2e/` (full-stack E2E tests at root)
  - Single `make e2e` command for E2E testing
  - Simplified imports: `from src.xxx` instead of `from backend.src.xxx`

  - LLM client supports profile-based auth via `WorkspaceClient.config.oauth_token()`
  - Lakebase uses `WorkspaceClient.database.generate_database_credential()`
  - Host derived from `LAKEBASE_INSTANCE_NAME` (not separate host config)
  - Fallback to direct `DATABASE_URL` for local development

  - TypeScript 5.x + Playwright for browser-based e2e tests
  - Page Object Model architecture
  - Tests for User Story 9 acceptance scenarios

  - Frontend builds to `static/` via Vite
  - FastAPI serves static files in production (SERVE_STATIC=true)
  - Single container deployment, no Node.js runtime
  - Use `make build` and `make prod` for unified deployment

  - **DECIDED**: Plain Async Python for agent orchestration (NOT LangGraph)
  - 5-agent architecture: Coordinator, Planner, Researcher, Reflector, Synthesizer
  - Step-by-step reflection after EACH research step
  - Tiered model routing (simple/analytical/complex)

<!-- MANUAL ADDITIONS START -->
<!-- Add project-specific notes here that should persist across updates -->
<!-- MANUAL ADDITIONS END -->
