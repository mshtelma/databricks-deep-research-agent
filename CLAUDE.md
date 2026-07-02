# Deep Research Agent Development Guidelines

## Project Overview

Deep research agent with multi-agent architecture (Coordinator, Planner, Researcher, Reflector, Synthesizer + specialized variants), step-by-step reflection, web search via Brave API, streaming chat UI with persistence on Databricks Lakebase.

**Key Design Decisions:**
- **Plain Async Python** orchestration (NOT LangGraph/DSPy/AutoGen)
- **Step-by-step reflection** after EACH research step (CONTINUE/ADJUST/COMPLETE)
- **Tiered model routing**: simple (fast), analytical (balanced), complex (reasoning)

## Quick Reference — Make Commands

### Development
| Command | Description |
|---------|-------------|
| `make dev` | Run backend (:8000) + frontend (:5173) with hot reload |
| `make dev-backend` | Backend only with hot reload |
| `make dev-frontend` | Frontend only with Vite |
| `make install` | Install all dependencies (backend + frontend + e2e) |
| `make quickstart` | Set up local development environment |
| `make worktree BRANCH=<name>` | Create worktree from current branch with .env symlinks |
| `make worktree BRANCH=<name> INSTALL=1` | Create worktree + install dependencies |
| `make worktree-list` | List all worktrees |
| `make worktree-remove BRANCH=<name>` | Remove a worktree |
| `make worktree-link BRANCH=<name>` | Re-link .env files in existing worktree |

### Database
| Command | Description |
|---------|-------------|
| `make db-provision TARGET=local-dev` | Full Autoscaling setup, writes `.env.local-dev` |
| `make db-provision TARGET=e2e` | Full Autoscaling setup for E2E, writes `.env.e2e` |
| `make db-cleanup TARGET=dev` | Remove orphaned resources from interrupted deploys |
| `make db-migrate` | Run migrations on configured Lakebase |
| `make db-migrate DB_SUFFIX=dev` | Run migrations on `deep_research_dev` database |
| `make db-status` | Check current migration status |
| `make db-migrate-remote TARGET=dev` | Run migrations on deployed Lakebase instance |
| `make db-reset` | Reset schema (drops ALL data, recreates tables) |
| `make clean_db` | Delete all chats/messages (preserves schema) |
| `make db-local` | Start local PostgreSQL via Docker (fallback only) |
| `make db-local-stop` | Stop local PostgreSQL |

**Note:** Local development uses **remote Lakebase** by default. Use `DB_SUFFIX` to isolate databases (e.g., `DB_SUFFIX=dev` targets `deep_research_dev`).

### Testing
| Command | Description |
|---------|-------------|
| `make test` | Unit tests — both framework + app (fast, mocked) |
| `make test-framework` | Framework unit tests only |
| `make test-app` | App unit tests only |
| `make test-integration` | Integration tests (real LLM/Brave, requires creds) |
| `make test-complex` | Long-running tests (production config) |
| `make test-all` | All Python + Frontend tests |
| `make test-frontend` | Frontend tests only |

### E2E Testing (Playwright)
| Command | Description |
|---------|-------------|
| `make e2e` | Build + run all E2E tests |
| `make e2e-fast` | Basic UI tests, no research (~1 min) |
| `make e2e-medium` | Light research operations (2-5 min) |
| `make e2e-slow` | Full research with verification (~10 min) |
| `make e2e-super-slow` | Parallel research sessions (~15 min) |
| `make e2e-custom-agents` | Custom agent CRUD + research (~30 min) |
| `make e2e-all` | All E2E categories |
| `make e2e-ui` | E2E with Playwright UI |

### Quality & Build
| Command | Description |
|---------|-------------|
| `make typecheck` | Type check both projects (mypy + tsc) |
| `make lint` | Lint both projects (ruff + eslint) |
| `make format` | Format both projects (ruff + prettier) |
| `make build` | Build frontend to `static/` |
| `make prod` | Build + run unified production server (:8000) |
| `make clean` | Remove build artifacts |

### Deployment (Databricks Apps)
| Command | Description |
|---------|-------------|
| `make deploy TARGET=dev` | Full deployment (build, migrate, grant, start) — gated by `typecheck-framework` |
| `make deploy TARGET=ais` | Deploy to AIS workspace — gated by `typecheck-framework` |
| `make deploy-unchecked TARGET=ais` | Emergency revert / typecheck-baseline catch-up only. Skips mypy. |
| `make app-deploy TARGET=ais` | Fast app-only redeploy — gated by `typecheck-framework` |
| `make app-deploy-unchecked TARGET=ais` | Fast redeploy, skips mypy. Emergency use only. |
| `make logs TARGET=dev` | Download app logs |
| `make logs TARGET=dev FOLLOW=-f` | Follow logs in real-time |
| `make logs TARGET=dev SEARCH="--search ERROR"` | Filter logs |
| `make requirements` | Generate requirements.txt from pyproject.toml |
| `make bundle-validate` | Validate Databricks bundle config |
| `make bundle-summary` | Show deployment summary |

**Deploy gate (added 2026-05-25):** `make deploy` and `make app-deploy` depend on
`typecheck-framework` (strict mypy on the framework code). This catches
attribute-name typos, missing kwargs, and signature drift before they reach
production — the class of regression that crashed the synthesizer's
`InterleavedGenerator` on 2026-05-25. Use the `*-unchecked` variants only for
emergency reverts where typecheck cannot pass.

### Framework
| Command | Description |
|---------|-------------|
| `make run-example WORKFLOW=simple_research QUERY="What is AI?"` | Run a framework example workflow |

### Direct Commands
```bash
uv run pytest -m "unit"          # Unit tests
uv run pytest -m "integration"   # Integration tests
uv run mypy src/deep_research --strict
uv run ruff check src/deep_research
cd frontend && npm run typecheck
tail -f /tmp/deep-research-dev.log   # Dev server logs
```

## Git Worktree Workflow

**Always use worktrees for new features and bugfixes.** This keeps the main checkout clean and allows parallel development.

### Creating a Worktree
```bash
# Branch from current branch (default)
make worktree BRANCH=feature-new-tool

# Branch from a specific ref
make worktree BRANCH=fix-bug BASE=main

# Create + install dependencies in one step
make worktree BRANCH=feature-new-tool INSTALL=1
```

Worktrees are created at `../.worktrees/<branch>/`. All gitignored `.env*` files are automatically symlinked from the main worktree so secrets, benchmark configs, and local overrides are shared.

### Working in a Worktree
```bash
cd ../.worktrees/feature-new-tool
make install    # first time only (or use INSTALL=1 above)
make dev        # start dev server
make test       # run tests
```

To run a dev server on a different port (avoids collision with main): `PORT=8001 make dev`

### Cleanup
```bash
make worktree-remove BRANCH=feature-new-tool                  # keep branch
make worktree-remove BRANCH=feature-new-tool DELETE_BRANCH=1  # remove branch too
```

### Re-linking Env Files
If you add a new `.env.*` file to the main worktree, re-link all worktrees:
```bash
make worktree-link BRANCH=feature-new-tool
```

## Project Structure (uv Workspace Monorepo)

This is a **uv workspace monorepo** with two packages:
- `databricks-deep-research` — standalone framework (PyPI-publishable)
- `databricks-deep-research-app` — the app that uses the framework

```text
pyproject.toml                        # Root workspace config

databricks-deep-research/             # Framework package
├── src/databricks_deep_research/
│   ├── workflow/                      # DAG engine: definition, executor, state, conditions, validation
│   │   ├── runtime/                   # Plan-and-execute orchestration (planning, step exec, recovery)
│   │   └── runtime_core/             # RuntimeState, WorkflowRunRequest/Result, store, selectors
│   ├── agents/                        # Agent harness, ReAct loop, grounding, isolation, query policy
│   │   ├── builtins/                  # 6 subtypes: coordinator, planner, researcher, reflector, synthesizer, background
│   │   ├── execution/                 # Pool/state projection, output normalization
│   │   └── prompts/                   # Per-subtype prompt templates
│   ├── tools/                         # ResearchTool protocol, ToolResolver, UrlRegistry
│   │   ├── builtins/                  # web_search, web_crawl, brave_search, file_search, vector_search, genie, knowledge_assistant
│   │   └── factories/                 # Builtin + Databricks tool factories
│   ├── pools/                         # Shared research pools (dedup, capacity, BM25 search)
│   ├── llm/                           # FrameworkLLMClient, ModelTier routing, token budget
│   ├── citation/                      # 7-stage verification pipeline (12 modules)
│   ├── events/                        # StreamEvent discriminated union
│   ├── templates/                     # Safe Jinja2 template rendering
│   ├── runner.py                      # WorkflowRunner high-level API
│   ├── tracing.py                     # MLflow tracing integration
│   └── errors.py                      # WorkflowError hierarchy
├── docs/                              # Framework documentation (43 files)
├── examples/                          # YAML workflow examples
└── tests/

databricks-deep-research-app/         # Application package
├── src/deep_research/
│   ├── agent/
│   │   ├── orchestrator.py            # Main async pipeline (legacy path)
│   │   ├── framework_orchestrator.py  # Framework path (use_framework=True)
│   │   ├── adapters/                  # Framework ↔ App adapters (llm, config, domain, tool, checkpoint)
│   │   ├── pipeline/                  # Phase-based execution pipeline
│   │   ├── workflows/                 # YAML workflow definitions + builder
│   │   ├── nodes/                     # Agent nodes (coordinator, planner, researcher, etc.)
│   │   ├── prompts/                   # App-specific prompt templates
│   │   ├── tools/                     # App-specific tools (enterprise, vector search, genie)
│   │   └── utils/                     # Agent utilities (conversation history)
│   ├── agent_server/                  # Databricks agent serving endpoint
│   ├── api/v1/                        # FastAPI routes (15 modules + utils/)
│   ├── middleware/                     # Auth, CSRF, security headers, audit, logging
│   ├── plugins/                       # Plugin system (base, manager, discovery, lifecycle/)
│   ├── conversation/                  # Intent classification + conversation routing
│   ├── output/                        # Output protocol + format registry
│   ├── models/                        # SQLAlchemy models
│   ├── schemas/                       # Pydantic schemas
│   ├── services/                      # Business logic (llm/, search/, citation/, storage/)
│   ├── core/                          # Config, auth, tracing, dependencies
│   ├── deployment/                    # Lakebase provisioning, migration runner, permission grants
│   ├── db/                            # Database session, migrations/ (20 versions)
│   ├── cli/                           # CLI provisioning tools
│   ├── jobs/                          # Background job definitions
│   └── main.py                        # FastAPI entry point
├── frontend/src/                      # React frontend
├── e2e/                               # Playwright E2E tests
├── tests/                             # App tests (unit/integration/complex)
├── scripts/                           # Utility scripts
└── config/                            # YAML config files
```

### Key Configuration Files
| File | Purpose |
|------|---------|
| `config/app.yaml` | Central config (endpoints, models, agents, search) |
| `config/app.test.yaml` | Test config (fast models, minimal iterations) |
| `.env` | Shared secrets (BRAVE_API_KEY, etc.) — not touched by provisioning |
| `pyproject.toml` | Python package definition |
| `databricks.yml` | Databricks Asset Bundle config |

### Env File Layout
All env files live in the **repository root**, not inside `databricks-deep-research-app/`.

| File | Written by | Purpose |
|------|-----------|---------|
| `.env` | Manual | Shared secrets (BRAVE_API_KEY, workspace creds) |
| `.env.local-dev` | `make db-provision TARGET=local-dev` | Local dev Lakebase connection |
| `.env.e2e` | `make db-provision TARGET=e2e` | E2E test Lakebase connection |
| `.env.ais` | `make deploy TARGET=ais` | AIS deployment config |
| `.env.example` | Checked in | Template for new developers |

## Constitution Principles (MUST FOLLOW)

1. **LLM Calls**: All LLM calls MUST use OpenAI client via WorkspaceClient
2. **Type Annotations**: ALL functions MUST have type annotations
3. **Pydantic Models**: Use for data structures and validation
4. **No Runtime Introspection**: No hasattr/isinstance for type safety
5. **Linting**: mypy strict + ruff MUST pass before merge
6. **Env Files**: All `.env*` files live in repo root, not inside app directory
7. **Target Env Loading**: Use `load-env` pattern for target-specific env loading

## Architecture

### Framework (`databricks-deep-research`)
Standalone multi-agent orchestration library (see `databricks-deep-research/docs/` for full reference):
- **Workflow Engine**: YAML-defined DAG with 8 node types (sequence, parallel, loop, conditional, agent, tool, subworkflow, plan_and_execute)
- **Plan-and-Execute Runtime**: `runtime/` handles planning, step execution, evaluation, recovery; `runtime_core/` provides `RuntimeState` (append-only), `WorkflowRunRequest`/`WorkflowRunResult`
- **Agent Harness**: prompt→LLM→parse→state cycle with ReAct tool-calling loop; `execution/` pipeline handles pool/state projection and output normalization
- **Builtin Subtypes**: coordinator, planner, researcher, reflector, synthesizer, background (registered via `register_builtin()`)
- **Tool System**: `ResearchTool` protocol with `SourceKind` enum (web, vector_index, sql_analytics, qa_assistant, file, builtin); `ToolResolver` chains overrides → cache → factories → registry; `UrlRegistry` prevents hallucinated URLs
- **7 Builtin Tools**: web_search, web_crawl, brave_search, file_search, vector_search, genie, knowledge_assistant
- **Pool System**: Shared research pools with dedup, capacity limits, BM25+vector hybrid search
- **Citation Pipeline**: 7-stage verification (evidence selection → interleaved generation → confidence → NLI verification → correction → numeric QA → ARE retrieval)
- **Public API**: `WorkflowRunner`, `WorkflowResult`, `RuntimeState`, `ExecutionContext`, `FrameworkLLMClient`, `ModelTier`, `ResearchTool`, `ToolFactoryContext`, `StreamEvent`

### App Integration
- **Feature Flag**: `OrchestrationConfig.use_framework=True` routes through `framework_orchestrator.py` instead of legacy `orchestrator.py`
- **Adapter Pattern**: `llm_adapter.py`, `config_translator.py`, `domain_context.py`, `tool_adapter.py`, `checkpoint_adapter.py` bridge framework ↔ app
- **Middleware Stack**: Auth → CSRF → SecurityHeaders → AuditLog → RequestLogging (in `middleware/`)
- **Plugin System**: `PluginManager` discovers tool/prompt providers + lifecycle hooks via entry points
- **Conversation Handler**: Intent classification and routing in `conversation/`
- **Output Protocol**: Extensible output format registry in `output/`
- **Deployment Helpers**: Lakebase provisioning, migration runner, permission grants in `deployment/`

### Designer Topologies
The Designer scaffolds workflows into one of these structural shapes (selected
deterministically by `task_signature.select_topology`; an explicit
`coordination_pattern` wins via rule-0). Each topology is one
`topology_registry.TopologySpec` (name → builder + structural_family) plus a
`workflow_builder._build_<name>_workflow`; the enum-parity test keeps
`TopologyName` / `TopologyKind` / `TOPOLOGIES` / the registry in lockstep.

| Topology | Shape | When | Coordination axes |
|----------|-------|------|-------------------|
| `single_agent` | coordinator → 1 agent | bounded single-pass lookup | — |
| `parallel_lanes` (default) | coordinator → parallel(N lanes) → draft synth → coverage reflector → final synth | 2+ independent lanes | `independent_workstreams_count` |
| `plan_and_execute` | coordinator → plan_and_execute(planner → body → reflector, looped) → synth | steps depend on earlier findings | `step_dependencies_present`/`iteration_required` |
| `best_of_n` | coordinator → parallel(evidence) → parallel(N candidate synths) → judge | generate N candidates, pick best | `coordination_candidate_count`, `coordination_judge_tier` |
| `iterative_refinement` | coordinator → parallel(evidence) → loop(draft → critic) → finalizer | draft, critique, improve until good | `refine_participants`, `refine_max_iterations`, `proposer_families` |
| `router` | classifier → conditional(branch_1..M, default) | classify + branch to distinct sub-pipelines | `router_cases` |
| `tree_search` | coordinator → parallel(level-1: B) → [gap reflector → parallel(narrowed level)]* → synth | survey breadth, then go deeper on gaps | `tree_breadth` (2-6), `tree_depth` (1-3) |

`tree_search` is a **static unroll** over depth D (no runtime recursion): breadth
narrows per level (`next = max(2, breadth // 2)`), and each between-level gap
reflector is UPSTREAM of the next level so `{level{i}_review}` is a normal upstream
output_key (passes the prompt-var whitelist). The probe (`ast_introspection`/`probe`)
detects it at the root BEFORE the generic parallel-recursion via the builder's
`l{N}_research-level` parallel-id convention.

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

### Framework Integration
- Import public API from `databricks_deep_research` (see `__init__.py` for full list)
- Tools implement `ResearchTool` protocol; use `ToolFactoryContext` for DI (workspace_client, user_token)
- `SourceKind` enum drives query generation and result admission per tool type
- `UrlRegistry` maps integer indices → URLs; LLM never sees raw URLs
- `RuntimeState` is append-only — never mutate, always extend

## Authentication

### Local Development (Lakebase — Standard)
```bash
# Profile-based (RECOMMENDED)
DATABRICKS_CONFIG_PROFILE=your-profile-name
LAKEBASE_INSTANCE_NAME=your-instance-name
LAKEBASE_DATABASE=deep_research

# Autoscaling backend (written by db-provision)
ENDPOINT_NAME=your-endpoint-name
PGHOST=your-pghost.database.cloud.databricks.com
```

### Local PostgreSQL (Fallback Only)
```bash
# Only for offline development — requires: make db-local
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/postgres
```

### Lakebase OAuth
- Tokens have 1-hour lifetime with 5-minute refresh buffer
- Username is always `"token"` for OAuth connections
- Auth path auto-detected: `ENDPOINT_NAME` → Autoscaling, `PGHOST` → Provisioned, `LAKEBASE_INSTANCE_NAME` → Legacy

## Deployment Architecture

### Autoscaling Pipeline
1. Provision Lakebase endpoint via `make db-provision TARGET=<target>`
2. Wait for endpoint ready state
3. Create database, run migrations
4. Grant permissions to app service principal
5. Deploy app via `make deploy TARGET=<target>`

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
| fevm | fevm | FEVM Serverless Stable |

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

### Citation Pipeline Tuning

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `citation_verification.max_evidence_chars` | int (200-10000) | 3000 | Pipeline-wide cap on evidence quote length, applied to all 5 truncation sites (evidence selection, claim generation prompt, single-claim NLI, batch verification, retry verification). Supersedes the legacy `evidence_preselection.max_span_length`. Override per-agent via custom-agent YAML: `config.citation_pipeline.max_evidence_chars`. |

```yaml
# Project-wide default in app.yaml
citation_verification:
  max_evidence_chars: 3000

# Per-agent override in custom-agent YAML
- node_type: agent
  subtype: synthesizer
  config:
    citation_pipeline:
      max_evidence_chars: 5000   # raise for richer tabular corpora
```

Feature flag `CITATION_SOFT_WARN_ENABLED` (default `true`): when the verifier
cannot produce real entailment judgments (all claims abstained or NLI-crashed),
the synthesizer emits the LLM-written report with a `> ⚠️ Grounding warning`
banner instead of the canned "Insufficient Evidence" template. Set to `false`
to revert to the legacy hard-fail behavior.

### Search Providers
The builtin `web_search` tool's backend is selected by `search.provider` in
`app.yaml` (default `databricks` — works out-of-the-box on a Databricks workspace
with NO external search subscription; **Brave is opt-in and never assumed/required**).
All providers implement the framework `SearchClient` protocol and feed the same
pool → crawl → citation pipeline.

| Provider | Backend | Notes |
|----------|---------|-------|
| `databricks` (default) | Model-serving **built-in web search** | A *billed model generation* per query (latency/cost ≫ Brave; ~8–16 billed searches per deep-research run). Pay-per-token endpoints only; unavailable on provisioned-throughput / HIPAA-BAA / cross-region-disabled workspaces (set `provider: brave`/`jina` there). |
| `brave` (opt-in) | Brave Web Search API | Fast REST search; needs `BRAVE_API_KEY`. |
| `jina` (opt-in) | Jina Search API | Returns full page content; `JINA_API_KEY` optional. |

`databricks` provider (`config/app.yaml` → `search.databricks`):
- **Gemini endpoint** (`databricks-gemini-3-1-flash-lite`, default) — native
  `generateContent` grounding; single fast call; redirect URLs auto-resolved to
  canonical. **OpenAI endpoint** (`databricks-gpt-5`) — Responses API; direct URLs +
  real titles, but agentic/slower (can exceed the `web_search`-mode timeout).
- Family auto-detected from the endpoint name (override via `model_family`).
- Reuses the framework LLM client's serving-endpoints connection (OBO identity).
- Concurrency capped by env `DBX_WEBSEARCH_MAX_CONCURRENCY` (from `search.databricks.max_concurrency`).
- Implementation: `databricks-deep-research/.../tools/builtins/databricks_web_search.py`.

**Provider precedence (high → low):** per-tool `config.provider` → global
`app.yaml search.provider` → built-in `databricks` (`DEFAULT_SEARCH_PROVIDER` in
`core/app_config.py`, the single source; `resolve_effective_provider` centralizes
the rule). A web tool with **no** `provider` **inherits at runtime** (never
stamped/baked on save), so changing `app.yaml search.provider` /
`search.databricks.endpoint` re-points every inheriting agent — a live global
lever. databricks sub-config (`model`/`model_family`/`timeout_seconds`/`max_results`/
`resolve_redirects`): explicit per-tool value → else `search.databricks` defaults
(auto-filled) → else framework defaults.

**Per-tool provider — three authoring surfaces:**
- *YAML* — set `config.provider` (and for databricks `config.model`/tuning) on a
  `web_search`/`web_research` decl. Honored by the framework factory
  (`_resolve_search_provider`); the app-side precedence guard
  `framework_orchestrator._tool_names_with_explicit_provider` stops the auto-injected
  global tool from shadowing it.
- *Designer UI inspector* — `agent_designer/registry._web_provider_properties` merges a
  `provider` dropdown + databricks knobs into the `web_search`/`web_research`
  `config_schema`; `SchemaField` renders them with **no React change**. New tools start
  `provider`-absent (= inherit) because `frontend/src/lib/jsonSchema.defaultConfigForSchema`
  seeds only explicit-default/required fields.
- *Designer chat* — the architect (`designer_architect.yaml`) sets `config.provider` per
  lane/tool only when the user asks for a specific backend; otherwise omits it (inherit).

**How the default reaches tools:**
- *Inherited tools (no per-tool `provider`)* use `ctx.search_client`, set by
  `workflow_runner_factory._apply_default_search_client` to the global-provider
  backend (databricks by default, via the shared `tool_adapter._build_web_search_client`).
  This single point covers every entry path (main-chat, agent-serving, designer),
  because the framework factory's no-provider branch reads `ctx.search_client`.
  Inherited config is **never** stamped/baked, keeping the app.yaml lever live.
- *Explicit per-tool `databricks`* is made self-describing (endpoint/tuning filled
  from `search.databricks`; `max_results` floored to `web_research`'s `total_results`)
  at persist time (`ast_normalizer.apply_web_search_provider_defaults`, shared by the
  designer normalizer + agent-save — **explicit-databricks only, never inherited**)
  and as a runtime net (`framework_orchestrator._fill_databricks_tool_defaults`). Both
  reuse `core/app_config.fill_databricks_search_defaults`. Explicit `brave`/`jina` are
  left untouched.
- *Deploy (shell-app)* binds the Brave secret only when a web tool **explicitly**
  pins `provider: brave` (`shell_app._definition_uses_brave_web_search`;
  `databricks.yml.j2` / `app.yaml` gate the secret on `brave_secret_scope`), so a
  default (databricks) shell-app needs no Brave secret.

Design-time validation: `semantic_validation` rejects an out-of-enum
`provider`/`model_family`; the framework `_resolve_search_provider` is the runtime
backstop. Exercised by `test_app_config`, `test_ast_normalizer`,
`test_workflow_runner_factory_search_client`, `test_shell_app_exporter`,
`test_provider_agnostic_retriever`, `test_registry_web_provider`,
`test_tool_config_enum_validation`, and `investment_research_databricks_*`
scaffold-and-run cases.

**Domain allowlist push-down (added 2026-06-07).** A per-agent INCLUDE allowlist
(`DomainFilterConfig`) used to be enforced ONLY post-hoc — the grounded LLM searched the
open web and the filter dropped everything off-domain, so allowlists often returned zero
sources. Now the allowlist is also pushed into the engine:
- **OpenAI** (`databricks-gpt-5*`, Responses API): `tools=[{"type":"web_search","filters":
  {"allowed_domains":[…]}}]` — bare domains, subdomains auto-included. Verified honored +
  confined on AIS. **All-or-nothing**: pushed only if every pattern reduces to a bare
  registrable domain (≥2 labels) within the limit (`*.gov`/`news.*`/over-limit ⇒ no push,
  fall back to hint+post-hoc). Graceful per-run fallback: a 400 on `filters` latches it off
  and retries without (so an unsupporting proxy degrades, never hard-fails).
- **Both families**: a *soft* domain scope hint is appended to the search instruction
  (Gemini's only lever; lists raw patterns incl. wildcards). Hard enforcement is the engine
  filter or the post-hoc `url_allowed` — never the instruction text (avoids self-censoring
  to zero).
- **Subdomain-inclusive matching**: a bare `reuters.com` now also matches `www.reuters.com`
  in BOTH matchers (`domain_filter.match_domain_pattern`, `web_search._domain_matches`) —
  aligns post-hoc with OpenAI's subdomain semantics; affects all providers + exclude mode.
- Gate: `search.databricks.push_allowed_domains` (default `true`); EXCLUDE mode and Gemini
  are unaffected by the gate. Wired in `tool_adapter._build_web_search_client`
  (`_allowlist_patterns`) + `factories/builtin._build_databricks_search_provider`; derived in
  `databricks_web_search._pushable_allowed_domains` / `_domain_scope_clause`. The per-agent
  filter reaches runtime via the `framework_orchestrator` ToolResolver override (shadows the
  factory's shared `ctx.search_client`). Exercised by `test_databricks_web_search`,
  `test_domain_filter`, `test_tool_adapter_provider`, `test_app_config`.

### Query Modes
- `simple` — Direct LLM response, no research
- `web_search` — Lightweight pipeline: 2-5 sources, 15-20s timeout, falls back to simple on timeout
- `deep_research` — Full multi-step pipeline with reflection

### Researcher Modes
- `classic`: Single-pass with fixed searches/crawls per step (faster)
- `react`: ReAct loop where LLM controls tool calls with budget (more intelligent)

### Config Access in Code
```python
from deep_research.agent.config import (
    get_research_type_config,
    get_step_limits,
    get_researcher_config_for_depth,
)
from deep_research.core.app_config import get_app_config
```
