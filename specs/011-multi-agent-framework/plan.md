# Implementation Plan: Multi-Agent Framework Extraction

**Branch**: `011-multi-agent-framework` | **Date**: 2026-03-08 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/011-multi-agent-framework/spec.md`

## Session Decisions (2026-03-08)

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | **Full replacement** — no feature flag, orchestrator rewritten | Project not released, no rollback needed |
| D2 | **Full port** — all production agents move to framework builtins | No dual mode, no simplified-vs-production |
| D3 | **AsyncOpenAI directly** — no LLM Protocol abstraction | Databricks standardizes on OpenAI |
| D4 | **Coordinator + Background in framework** — 6 builtin subtypes | Essential agentic routing + background investigation patterns |
| D5 | **YAML is first-class** — framework fully supports YAML load/save | Internal data model, YAML is serialization |
| D6 | **Drop classic researcher** — only ReAct mode | Always use ReAct in practice |
| D7 | **Drop simple synthesizer** — only ReAct + Reclaim modes | Simple mode unused |
| D8 | **Prompt customization** — `system_prompt` + `user_prompt_template` overrides in YAML | Main pair only, internal prompts stay internal |

## Summary

Extract a standalone, PyPI-publishable multi-agent orchestration framework (`databricks-deep-research`, import: `databricks_deep_research`) from the existing Deep Research project. The framework provides composable workflow trees with 8 node types, append-only immutable state, shared research pools, declarative conditions, 6 standard agent subtypes with full production-quality builtin implementations, builtin tools (web search, web crawl, file search), and streaming execution events. The framework uses `openai.AsyncOpenAI` directly (Databricks standardizes on OpenAI-compatible endpoints).

The Deep Research application becomes a consumer that declares the framework as a dependency. The app's `config_translator.py` builds `WorkflowDefinition` programmatically from `OrchestrationConfig`; standalone users can load workflows from YAML. The old orchestrator is **fully replaced** — no feature flag, no parallel coexistence.

Technical approach: Monorepo with two subdirectory projects — `databricks-deep-research/` (framework) and `databricks-deep-research-app/` (application). A root `pyproject.toml` with `[tool.uv.workspace]` ties them together for local development. Framework is a Python library with zero database/HTTP/UI dependencies (depends on `openai`, `pydantic`, `pyyaml`, optional `httpx`/`trafilatura`/`bm25s`). App retains FastAPI, database, frontend, and provides the `AsyncOpenAI` client + model mapping via adapter.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: Framework: `openai>=1.10.0`, Pydantic 2.x, PyYAML, asyncio. Optional: `httpx` (web search), `trafilatura` (web crawl), `bm25s+numpy` (pool search). App: FastAPI, SQLAlchemy, asyncpg, databricks-sdk, mlflow (all existing).
**Storage**: Framework: None (stateless library). App: Databricks Lakebase (PostgreSQL) — existing.
**Testing**: pytest + pytest-asyncio. Framework tests self-contained with mock LLM. App tests cover integration layer.
**Target Platform**: Linux server (Databricks Apps), local development on macOS/Linux.
**Project Type**: Monorepo with two subdirectory projects — `databricks-deep-research/` (framework library) + `databricks-deep-research-app/` (web application). Root `pyproject.toml` with `uv` workspace.
**Performance Goals**: Workflow execution overhead < 50ms per node (excluding LLM calls). Event streaming with < 100ms latency from node completion to SSE emission.
**Constraints**: Framework must have zero dependencies on app code, database, or HTTP. Must pass mypy strict + ruff.
**Scale/Scope**: Framework: ~38 source files, ~9000 LOC. App: existing codebase moved into `databricks-deep-research-app/`, ~20 files modified, ~2500 LOC adapter code. Root workspace config + Makefile.

## Constitution Check

### Principle I: Clients and Workspace Integration

**Status**: DEVIATION — Justified

The framework imports `openai.AsyncOpenAI` directly as a dependency. This deviates from the principle that frameworks should use WorkspaceClient. **Justification**: Databricks standardizes on OpenAI-compatible endpoints — all model serving endpoints expose the OpenAI API. A Protocol abstraction would add complexity without practical value (no non-OpenAI backends planned). The app still uses `WorkspaceClient` to obtain the `AsyncOpenAI` client; the framework just consumes it.

### Principle II: Typing-First Python

**Status**: PASS

All framework entities are Pydantic models or typed dataclasses. All functions have full type annotations. Node configs use discriminated unions. The `WorkflowState` uses typed `StateEntry` records. Pool tools are typed. Streaming events are Pydantic BaseModel with `event_type: Literal[...]` discriminator.

### Principle III: Avoid Runtime Introspection

**Status**: PASS

Node type dispatch uses an explicit `NodeType` enum → handler mapping (not isinstance chains). Agent subtype resolution uses a typed defaults dict. Condition evaluation uses discriminated union types. Tool resolution uses typed `ToolRef` models. No `hasattr`, no duck-typing introspection.

### Principle IV: Linting and Static Type Enforcement

**Status**: PASS

Both projects will have mypy strict + ruff configurations. The framework's `pyproject.toml` will mirror the existing strict settings. All `# type: ignore` comments will include justification. The framework's public API surface will be fully annotated.

### Post-Design Re-evaluation

- Principle I deviation acknowledged — AsyncOpenAI is a concrete dependency, not a protocol.
- The executor dispatch uses enum mapping, not introspection (III).
- The checkpoint handler is protocol-based (III).
- All data models are Pydantic with full annotations (II, IV).
- Events use Pydantic BaseModel with Literal discriminator (II).

## Project Structure

### Documentation (this feature)

```text
specs/011-multi-agent-framework/
├── plan.md              # This file
├── research.md          # Architectural decisions
├── data-model.md        # All framework entities
├── quickstart.md        # Getting started guide
├── architecture.md      # Full architecture (~3700 lines)
├── contracts/           # Public API contracts (Python)
│   ├── executor_api.py
│   ├── llm_client.py   # AsyncOpenAI wrapper + LLMResponse + ModelTier
│   ├── tool_protocol.py
│   ├── events.py
│   └── checkpoint_protocol.py
└── checklists/
    ├── requirements.md
    └── behavioral_parity.md   # P0c migration parity checklist
```

### Source Code (monorepo root)

```text
databricks-deep-research-agent/          # Repository root
├── pyproject.toml                       # uv workspace definition
├── CLAUDE.md                            # Dev guidelines
├── README.md
├── Makefile                             # Root Makefile delegating to sub-projects
├── specs/                               # Shared specs
│
├── databricks-deep-research/            # ── FRAMEWORK (PyPI-publishable) ──
│   ├── pyproject.toml                   # Package: databricks-deep-research
│   ├── src/
│   │   └── databricks_deep_research/    # Import: databricks_deep_research
│   │       ├── __init__.py              # Public API exports
│   │       ├── workflow/                # Core orchestration engine
│   │       │   ├── definition.py        # WorkflowDefinition, WorkflowNode, NodeType
│   │       │   ├── state.py             # WorkflowState, StateEntry, append-only log
│   │       │   ├── executor.py          # WorkflowExecutor — tree walker (all 8 types)
│   │       │   ├── conditions.py        # StateCondition, combinators (all_of, any_of, negate)
│   │       │   ├── validation.py        # Load-time structural validation
│   │       │   ├── loader.py            # YAML from_yaml/to_yaml
│   │       │   └── context.py           # ExecutionContext
│   │       ├── pools/                   # Shared research pools
│   │       │   ├── pool_state.py        # PoolState, PoolConfig (dedup, capacity, locks)
│   │       │   ├── pool_tools.py        # Auto-generated search/retrieval tools
│   │       │   └── pool_registry.py     # Pool name → PoolState mapping
│   │       ├── agents/                  # Agent system
│   │       │   ├── config.py            # AgentNodeConfig, subtype defaults
│   │       │   ├── isolation.py         # AgentInput, AgentOutput
│   │       │   ├── harness.py           # Agent execution harness
│   │       │   ├── react_loop.py        # Generic ReAct execution loop
│   │       │   ├── prompts/             # Production prompt templates
│   │       │   │   ├── coordinator.py
│   │       │   │   ├── researcher.py
│   │       │   │   ├── planner.py
│   │       │   │   ├── reflector.py
│   │       │   │   ├── synthesizer.py
│   │       │   │   └── background.py
│   │       │   ├── output_models.py     # Typed output models for all 6 subtypes
│   │       │   └── builtins/            # Full production implementations
│   │       │       ├── coordinator.py   # Query classification + routing
│   │       │       ├── researcher.py    # ReAct researcher (only mode)
│   │       │       ├── planner.py       # Research plan generation
│   │       │       ├── reflector.py     # CONTINUE/ADJUST/COMPLETE reflection
│   │       │       ├── synthesizer.py   # ReAct + Reclaim (P0d) synthesis
│   │       │       └── background.py    # Background investigation + query decomposition
│   │       ├── tools/                   # Tool system
│   │       │   ├── protocol.py          # ResearchTool protocol, ToolDefinition, ToolResult
│   │       │   ├── registry.py          # Tool resolution and caching
│   │       │   ├── url_registry.py      # Index→URL mapping for security
│   │       │   └── builtins/            # Builtin tools
│   │       │       ├── web_search.py    # Brave Search (constructor DI, optional [web])
│   │       │       ├── web_crawl.py     # Web crawler (constructor DI, optional [crawl])
│   │       │       └── file_search.py   # BM25 file search (optional [search])
│   │       ├── citation/               # 7-stage verification pipeline (P0d)
│   │       │   ├── types.py             # EvidenceInfo, ClaimInfo, etc.
│   │       │   ├── config.py            # CitationConfig (per-depth tuning)
│   │       │   ├── pipeline.py          # Stateless orchestrator
│   │       │   ├── evidence_selector.py # Stage 1 (merged ContentEvaluator)
│   │       │   ├── claim_generator.py   # Stage 2 (ReClaim interleaved)
│   │       │   ├── confidence_classifier.py # Stage 3
│   │       │   ├── isolated_verifier.py # Stage 4 (NLI entailment)
│   │       │   ├── citation_corrector.py # Stage 5
│   │       │   ├── numeric_verifier.py  # Stage 6
│   │       │   ├── atomic_decomposer.py # Stage 7a (FActScore)
│   │       │   ├── verification_retriever.py # Stage 7b (ARE)
│   │       │   └── citation_keys.py     # Marker mapping
│   │       ├── templates/
│   │       │   └── renderer.py          # SafeTemplateRenderer
│   │       ├── llm/                     # LLM client
│   │       │   ├── client.py            # AsyncOpenAI wrapper + LLMResponse + ModelTier
│   │       │   ├── budget.py            # TokenBudget
│   │       │   └── rate_limiter.py      # Rate limiting
│   │       ├── events/
│   │       │   └── types.py             # Pydantic StreamEvent with event_type discriminator
│   │       └── errors.py               # WorkflowError, ValidationError, CancelledError
│   └── tests/
│       ├── test_definition.py
│       ├── test_state.py
│       ├── test_executor.py
│       ├── test_conditions.py
│       ├── test_pools.py
│       ├── test_harness.py
│       ├── test_react_loop.py
│       ├── test_researcher.py
│       ├── test_planner.py
│       ├── test_reflector.py
│       ├── test_synthesizer.py
│       ├── test_renderer.py
│       ├── test_events.py
│       ├── test_yaml_roundtrip.py
│       └── conftest.py
│
└── databricks-deep-research-app/        # ── APP (moved from root) ──
    ├── pyproject.toml                   # Package: databricks-deep-research-app
    ├── Makefile                          # App-specific make targets
    ├── databricks.yml
    ├── alembic.ini
    ├── src/
    │   └── deep_research/               # Import: deep_research (unchanged)
    │       ├── agent/
    │       │   ├── adapters/            # NEW: Framework integration
    │       │   │   ├── llm_adapter.py   # App LLMClient → AsyncOpenAI + model mapping
    │       │   │   ├── domain_context.py # DomainContextTracker: unified event translation + state
    │       │   │   ├── config_translator.py # OrchestrationConfig → WorkflowDefinition
    │       │   │   ├── tool_adapter.py  # Enterprise tools → framework protocol
    │       │   │   └── checkpoint_adapter.py # DB persistence → CheckpointHandler
    │       │   ├── workflows/           # App-specific workflow definitions
    │       │   │   ├── deep_research.yaml   # Reference deep research pipeline
    │       │   │   ├── web_search.yaml  # Web search mode
    │       │   │   └── builder.py       # Programmatic tree builders
    │       │   ├── orchestrator.py      # REWRITTEN: Uses framework executor
    │       │   ├── config.py            # App-level config accessors (unchanged)
    │       │   ├── pipeline/            # Plugin system (UNCHANGED)
    │       │   └── tools/               # Enterprise tools (STAY IN APP)
    │       │       ├── genie.py         # Databricks Genie
    │       │       ├── knowledge_assistant.py # KA queries
    │       │       ├── user_vector_search.py  # User doc search
    │       │       ├── vector_search.py # Generic vector search
    │       │       └── factory.py       # Enterprise tool loading
    │       ├── api/v1/                  # UNCHANGED
    │       ├── models/                  # UNCHANGED
    │       ├── schemas/                 # UNCHANGED
    │       ├── services/
    │       │   └── llm/client.py        # App's LLMClient (stays, adapted)
    │       ├── db/                      # UNCHANGED
    │       ├── core/                    # UNCHANGED
    │       ├── plugins/                 # UNCHANGED
    │       └── main.py                  # UNCHANGED
    ├── frontend/                        # UNCHANGED
    ├── tests/
    │   └── unit/agent/adapters/         # Adapter tests
    ├── e2e/                             # E2E tests
    ├── config/                          # YAML configs
    └── static/                          # Built frontend
```

### Root Workspace Configuration

```toml
# Root pyproject.toml (NOT a package — workspace only)
[project]
name = "databricks-deep-research-workspace"
version = "0.0.0"
requires-python = ">=3.11"

[tool.uv.workspace]
members = ["databricks-deep-research", "databricks-deep-research-app"]

[tool.uv.sources]
databricks-deep-research = { workspace = true }
```

### Framework Optional Extras

```toml
# databricks-deep-research/pyproject.toml
[project.optional-dependencies]
web = ["httpx>=0.24"]
crawl = ["trafilatura>=1.6"]
search = ["bm25s>=0.1", "numpy>=1.24"]
all = ["databricks-deep-research[web,crawl,search]"]
```

Pool BM25 search gracefully falls back to keyword match without `bm25s`. Clear `ImportError` with install instructions when dep missing.

## Implementation Phases

### P0a: Monorepo Structure + Framework Skeleton (~1,800 LOC)

**Goal**: Framework package exists with all PUBLIC TYPE definitions. No business logic — just types, protocols, and the package structure. App builds and all existing tests pass.

**PR Strategy**: Two PRs for reviewability:
1. **PR 1: Structural move** — `git mv` files into `databricks-deep-research-app/`, update all path references, create root workspace + Makefiles. NO new code.
2. **PR 2: Framework skeleton** — Create `databricks-deep-research/` with all type definitions, framework tests, workspace dependency wiring.

**Step 1 — Move app into subdirectory:**

```bash
mkdir databricks-deep-research-app
git mv src/ frontend/ tests/ e2e/ config/ static/ databricks.yml alembic.ini pyproject.toml databricks-deep-research-app/
# Keep at root: Makefile (rewritten), specs/, README.md, .env files, .github/
```

**Path reference audit:**

| File | Changes needed |
|------|---------------|
| `databricks-deep-research-app/pyproject.toml` | Rename package to `databricks-deep-research-app`. Add `databricks-deep-research` dependency. |
| `databricks-deep-research-app/databricks.yml` | Update `source_code_path`, resource paths. |
| `databricks-deep-research-app/alembic.ini` | Verify `script_location` relative paths. |
| `databricks-deep-research-app/Makefile` | NEW: All existing targets with paths relative to app dir. |
| `databricks-deep-research-app/e2e/playwright.config.ts` | Verify base URL and test paths. |
| Root `Makefile` | NEW: Delegates to sub-project Makefiles. |
| Root `pyproject.toml` | NEW: Workspace config. |

**Step 2 — Create framework skeleton:**

| File | LOC | Contents |
|------|-----|---------|
| `workflow/definition.py` | 150 | `WorkflowNode`, `WorkflowDefinition`, `NodeType` (8 types), `ConditionalBranch` |
| `workflow/state.py` | 200 | `WorkflowState` — append-only log, `get(key)`, `get_all(key)`, `StateEntry`, async locks |
| `workflow/context.py` | 60 | `ExecutionContext` — LLM client, tools, config |
| `workflow/conditions.py` | 80 | `StateCondition`, combinators (`all_of`, `any_of`, `negate`) |
| `agents/config.py` | 100 | `AgentNodeConfig` — subtype, model tier, input/output keys, pool refs |
| `agents/isolation.py` | 80 | `AgentInput`, `AgentOutput` — typed I/O contracts |
| `tools/protocol.py` | 120 | `ResearchTool` protocol (`definition` property, `validate_arguments`, `execute`), `ToolDefinition`, `ToolResult` (`success`, `data`), `ToolContext`, `UrlRegistry` |
| `tools/url_registry.py` | 80 | Index→URL mapping for security. Created per workflow execution, shared across tool calls. |
| `pools/pool_state.py` | 120 | `PoolState` — typed accumulation with dedup, max capacity, async locks |
| `llm/client.py` | 200 | AsyncOpenAI wrapper + `LLMResponse` (with `structured: Any | None`) + `ModelTier` enum + `ModelTierConfig` + `EndpointHealth` + retry/fallback methods + `embed()`/`embed_single()`/`supports_embeddings` |
| `events/types.py` | 150 | Pydantic `BaseModel` events with `event_type: Literal[...]` discriminator. Includes verification event stubs for P0d. |
| `errors.py` | 30 | `WorkflowError`, `WorkflowValidationError`, `WorkflowCancelledError` |
| Various `__init__.py` | 50 | Subpackage inits + public API exports |
| `tests/test_definition.py` | 100 | Tree construction, validation tests |
| `tests/test_state.py` | 100 | WorkflowState get/get_all/append tests |
| `tests/test_pool.py` | 80 | PoolState dedup, capacity tests |
| `tests/conftest.py` | 20 | Test fixtures |

**Verification**: `uv sync` resolves workspace, `make test` passes, `make typecheck` passes, `make lint` passes, `make dev` starts backend+frontend, all Python imports unchanged.

### P0b-infra: Execution Engine (~2,800 LOC)

**Goal**: Framework can execute workflow trees with mock agents. All 8 node types work.

| File | LOC | Contents |
|------|-----|---------|
| `workflow/executor.py` | 500 | `WorkflowExecutor` — tree walker. All 8 types: `sequence` (serial), `parallel` (asyncio.gather), `loop` (condition exit, min/max, LoopBreakSignal), `conditional` (branch eval), `agent` (calls harness), `tool` (calls tool directly), `subworkflow` (deferred to P2), `plan_and_execute` (plan→execute→evaluate cycle with continue/replan/complete). Yields `StreamEvent`s. |
| `workflow/loader.py` | 150 | YAML → `WorkflowDefinition` parser with validation |
| `workflow/validation.py` | 120 | Load-time: duplicate IDs, leaf-node checks, children count per type |
| `agents/harness.py` | 300 | Constructs `AgentInput` from state + pools, calls agent, parses `AgentOutput`, writes to state + pools, emits events. Supports structured output. |
| `agents/react_loop.py` | 250 | Generic ReAct loop: message construction, tool call dispatch, budget enforcement, conversation management. Reused by researcher + synthesizer. |
| `pools/pool_registry.py` | 60 | Pool name → PoolState mapping, auto-tool generation |
| `pools/pool_tools.py` | 100 | Auto-generated `pool_search`, `pool_retrieve`, `pool_count`, `pool_topics`, `pool_get_by_index`. BM25 optional (fallback to keyword match). |
| `tools/registry.py` | 80 | Tool name resolution + caching |
| `templates/renderer.py` | 100 | `SafeTemplateRenderer` — `{variable}` substitution. No Jinja2. |
| `llm/budget.py` | 80 | `TokenBudget` — tracks usage across agents, enforces limits |
| `llm/rate_limiter.py` | 200 | EndpointHealth, health-based endpoint selection, TPM tracking, exponential backoff with jitter, 429 fallback logic |
| Tests | 600 | test_executor.py, test_loader.py, test_harness.py, test_react_loop.py, test_pools.py |

### P0b-builtins: Full Production Agents + Tools (~4,320 LOC)

**Goal**: Framework ships with production-quality agent implementations. All current agent logic ported.

**Agent implementations** (full port from app's `agent/nodes/`):

| File | LOC | Source | Contents |
|------|-----|--------|---------|
| `agents/builtins/coordinator.py` | 400 | `nodes/coordinator.py` (390) | Query classification, simple query detection, depth recommendation, follow-up detection. Structured output via `CoordinatorOutput`. Emits `CoordinatorClassifiedEvent`. |
| `agents/builtins/researcher.py` | 800 | `nodes/react_researcher.py` (1228) | **ReAct only** (D6). LLM-controlled tool calls, observation synthesis, source tracking, token budget awareness. Uses `react_loop.py`. |
| `agents/builtins/planner.py` | 500 | `nodes/planner.py` (558) | Plan generation, `has_enough_context` detection, depth-aware step generation (min/max). Source-aware mode (checks data_landscape from background). Emits `PlanCreatedEvent`. |
| `agents/builtins/reflector.py` | 250 | `nodes/reflector.py` (253) | Coverage analysis, CONTINUE/ADJUST/COMPLETE decision. Uses `pool_inject` for observations (small during early steps). Emits `ReflectionDecisionEvent`. |
| `agents/builtins/synthesizer.py` | 700 | `nodes/react_synthesizer.py` (1404) | **ReAct mode** (D7): LLM controls evidence retrieval during generation. Streaming via `AgentStreamChunkEvent`. Structured output. Emits `SynthesisStartedEvent`. **Reclaim mode** added in P0d. |
| `agents/builtins/background.py` | 500 | `nodes/background.py` (1074) | Query decomposition, data landscape assessment, enterprise source discovery (5s timeout). SIMPLE tier. Emits `BackgroundCompletedEvent`. |
| `agents/output_models.py` | 120 | — | Typed output models (`PlanOutput`, `ReflectionOutput`, `CoordinatorOutput`, `ResearcherOutput`, `SynthesizerOutput`, `BackgroundOutput`) for all 6 subtypes |

**Prompt templates** (moved from app's `agent/prompts/`):

| File | LOC | Contents |
|------|-----|---------|
| `agents/prompts/coordinator.py` | 100 | `COORDINATOR_SYSTEM_PROMPT`, `COORDINATOR_USER_PROMPT`, `SIMPLE_QUERY_SYSTEM_PROMPT` |
| `agents/prompts/researcher.py` | 180 | `RESEARCHER_SYSTEM_PROMPT`, `SEARCH_QUERY_PROMPT`, background investigation prompts (merged) |
| `agents/prompts/planner.py` | 400 | `PLANNER_SYSTEM_PROMPT`, `PLANNER_USER_PROMPT`, source-aware variants |
| `agents/prompts/reflector.py` | 100 | `REFLECTOR_SYSTEM_PROMPT`, `REFLECTOR_USER_PROMPT` |
| `agents/prompts/synthesizer.py` | 150 | `SYNTHESIZER_SYSTEM_PROMPT`, `SYNTHESIZER_USER_PROMPT`, ReAct synthesis prompts |
| `agents/prompts/background.py` | 100 | Background investigation prompts, query decomposition instructions |

**Builtin tools** (constructor DI, optional deps):

| File | LOC | Source | Contents |
|------|-----|--------|---------|
| `tools/builtins/web_search.py` | 200 | `tools/web_search.py` (336) | Brave Search. `WebSearchTool(search_client=...)`. Optional `[web]`. |
| `tools/builtins/web_crawl.py` | 150 | `tools/web_crawler.py` (676) | Trafilatura. `WebCrawlTool(crawler=...)`. Optional `[crawl]`. |
| `tools/builtins/file_search.py` | 150 | `tools/file_search.py` (419) | BM25. Optional `[search]`. |

**Dead code (NOT ported)**:
- `nodes/researcher.py` (727 LOC) — classic mode, dropped per D6
- `nodes/synthesizer.py` (717 LOC) — simple mode, dropped per D7

**Verification**: `make test-framework` passes with mock LLM, standalone demo works (YAML → events), `make typecheck` + `make lint` pass.

### P0c: App Integration (~4,300 LOC)

**Goal**: App uses the framework. Orchestrator fully rewritten (D1). All existing E2E tests pass.

**Adapter layer** (`src/deep_research/agent/adapters/`):

| File | LOC | Contents |
|------|-----|---------|
| `llm_adapter.py` | 170 | Wraps app's `LLMClient` (health tracking, fallback, OAuth) → `AsyncOpenAI` + model tier mapping + `embedding_model` for framework. |
| `domain_context.py` | 200-300 | `DomainContextTracker`: event-forwarding adapter (NOT state reconstructor). Methods: `process_event(StreamEvent) -> list[AppSSEEvent]`, `get_persistence_delta() -> PersistenceDelta`, `should_persist() -> bool`. Each handler is 5-15 lines of pattern matching — enriched domain events are self-contained with all metadata the app needs. Accumulates `PersistenceDelta` (sources, observations, step updates) incrementally. |
| `config_translator.py` | 300-400 | `OrchestrationConfig` → `WorkflowDefinition` tree builder. Must handle ALL modes (no bail — D1). Maps: `enable_background_investigation`, `source_scope`, `output_format`, `output_schema`, `file_ids`, `synthesis_mode`. |
| `checkpoint_adapter.py` | 100 | Wraps app's DB persistence as `CheckpointHandler` protocol. |
| `tool_adapter.py` | 150 | Constructor DI factory: creates framework tools once per workflow execution. Injects search clients, domain filters, user tokens at construction time. Wraps enterprise tools (Genie, KA, VectorSearch) for framework's `ResearchTool` protocol. |

**Orchestrator rewrite** (`orchestrator.py`):
- Old 3769 LOC monolith → thin wrapper around framework executor
- `config_translator.translate(orchestration_config)` → `WorkflowDefinition`
- `WorkflowExecutor(context).execute(definition)` → yields `StreamEvent`
- Pipeline: `tracker = DomainContextTracker(config)`; for each event: `tracker.process_event(event)` → SSE events; `tracker.should_persist()` → `persist(tracker.get_persistence_delta())`
- Implementation guided by `checklists/behavioral_parity.md`

**Workflow definitions** (`agent/workflows/`):
- `deep_research.yaml` (80 LOC) — reference YAML for deep research pipeline
- `web_search.yaml` (30 LOC) — web search mode
- `builder.py` (100 LOC) — `preset_steps_to_tree()` for Level 1 abstraction

**Plugin compatibility**: When `plugin_manager.has_custom_phase_mode()`, the config_translator builds tree nodes that call the existing `PhaseExecutor`. PhaseExecutor, PipelineCustomization unchanged.

**Tests**:
- `test_llm_adapter.py` (100), `test_domain_context.py` (200), `test_config_translator.py` (300)

**Config translator test matrix** — all valid `OrchestrationConfig` combinations that produce distinct workflow trees:

| source_scope | synthesis_mode | workflow_mode | background | enterprise_tools | file_ids | Expected Tree Shape |
|---|---|---|---|---|---|---|
| web_only | react | PLANNER | true | none | none | coord → bg → plan_and_execute → synth |
| web_only | react | PLANNER | false | none | none | coord → plan_and_execute → synth |
| web_only | reclaim | PLANNER | true | none | none | coord → bg → plan_and_execute → reclaim_synth |
| enterprise_only | react | PLANNER | true | [genie] | none | coord → bg → plan_and_execute(enterprise tools) → synth |
| both | react | PLANNER | true | [vector] | [file1] | coord → bg → plan_and_execute(all tools) → synth |
| web_only | react | MANUAL | false | none | none | coord → plan_and_execute(preset_steps) → synth |
| web_only | react | HYBRID | false | none | none | coord → plan_and_execute(partial_preset) → synth |
| none | react | PLANNER | false | none | none | coord → plan_and_execute(no search tools) → synth |
| web_only | react | PLANNER | true | none | [f1,f2] | coord → bg → plan_and_execute(+file_search) → synth |

Each row becomes a parameterized unit test asserting the tree structure.

**Deployment**: `make deploy` builds framework wheel → includes in bundle.

**Verification**: `make test` passes (all unit), `make e2e` passes, `make deploy TARGET=dev` succeeds. Reclaim mode temporarily unavailable (added in P0d).

### P0d: Citation Verification Pipeline (~7,500 LOC)

**Goal**: 7-stage citation verification pipeline is a framework feature. Reclaim synthesis mode fully works.

**Data structures** (`citation/types.py`, `citation/config.py` — ~300 LOC):
- `EvidenceInfo`, `ClaimInfo`, `VerificationSummaryInfo`, `RankedEvidence`, `InterleavedClaim`, `CorrectionResult`, `NumericVerificationResult`
- `CitationConfig` — per-depth tuning (evidence count, token budgets, softening strategy)

**Pipeline stages** (`citation/` — ~5,000 LOC):

| File | LOC | Stage | Contents |
|------|-----|-------|---------|
| `pipeline.py` | 800 | Orchestrator | Stateless: `(llm, config, sources, observations, query) → AsyncIterator[(content, events)]` |
| `evidence_selector.py` | 400 | 1 | LLM-based span extraction + quality filtering (ContentEvaluator merged) |
| `claim_generator.py` | 400 | 2 | ReClaim interleaved generation with citation markers |
| `confidence_classifier.py` | 150 | 3 | Rule-based confidence routing |
| `isolated_verifier.py` | 500 | 4 | NLI entailment, batch 10 pairs, verification cache (MD5) |
| `citation_corrector.py` | 400 | 5 | Post-hoc citation correction (keep/replace/remove/add_alternate) |
| `numeric_verifier.py` | 400 | 6 | QA-based numeric verification with normalization |
| `atomic_decomposer.py` | 600 | 7a | FActScore atomic fact decomposition, batch 5 claims |
| `verification_retriever.py` | 800 | 7b | ARE: decompose → external search → soften unsupported claims |
| `citation_keys.py` | 150 | Util | Citation marker mapping (claim → source key) |

**Integration** (~1,000 LOC):
- `agents/builtins/synthesizer.py` modified (+300): reclaim mode delegates to pipeline
- `events/types.py` modified (+100): `ClaimGeneratedEvent`, `ClaimVerifiedEvent`, `CitationCorrectedEvent`, `NumericClaimDetectedEvent`, `VerificationSummaryEvent`

**App updates** (~500 LOC):
- `config_translator.py` (+100): map `synthesis_mode`, `enable_citation_verification` to synthesizer config
- `domain_context.py` (+300): map verification events → app SSE events, extract claims/evidence/summary from state

**Tests** (~800 LOC): `test_pipeline.py`, `test_verifier.py`, `test_interleaved.py`, `test_are.py`

**Verification**: `make test-framework` passes, E2E with `synthesis_mode: "reclaim"` works, verification events stream to UI.

## Code Migration Map

### Moves to framework (full port)

| Source (app) | Destination (framework) | LOC |
|---|---|---|
| `nodes/coordinator.py` (390) | `agents/builtins/coordinator.py` | ~400 |
| `nodes/planner.py` (558) | `agents/builtins/planner.py` | ~500 |
| `nodes/react_researcher.py` (1228) | `agents/builtins/researcher.py` | ~800 |
| `nodes/reflector.py` (253) | `agents/builtins/reflector.py` | ~250 |
| `nodes/react_synthesizer.py` (1404) | `agents/builtins/synthesizer.py` | ~700 |
| `nodes/citation_synthesizer.py` (254) | `agents/builtins/synthesizer.py` | +400 (P0d) |
| `prompts/*.py` (~974) | `agents/prompts/*.py` | ~930 |
| `tools/web_search.py` (336) | `tools/builtins/web_search.py` | ~200 |
| `tools/web_crawler.py` (676) | `tools/builtins/web_crawl.py` | ~150 |
| `tools/file_search.py` (419) | `tools/builtins/file_search.py` | ~150 |
| `tools/base.py` (193) | `tools/protocol.py` | ~120 |
| `services/citation/` (~9500) | `citation/` (12 files) | ~5000 (P0d) |
| `nodes/background.py` (1074) | `agents/builtins/background.py` | ~500 |

### Dead code (not ported)

| Source | LOC | Reason |
|--------|-----|--------|
| `nodes/researcher.py` | 727 | D6: classic researcher dropped |
| `nodes/synthesizer.py` | 717 | D7: simple synthesizer dropped |

### Stays in app

| Source | LOC | Reason |
|--------|-----|--------|
| `tools/genie.py` | 645 | Enterprise, Databricks SDK |
| `tools/knowledge_assistant.py` | 856 | Enterprise |
| `tools/user_vector_search.py` | 702 | Enterprise |
| `tools/vector_search.py` | 280 | Enterprise |
| `tools/factory.py` | 414 | Enterprise tool loading |
| `nodes/custom_phase_executor.py` | 346 | Plugin system (unchanged) |
| `nodes/source_routing.py` | 416 | Absorbed into config_translator |

## Key Design Decisions

| # | Decision | Choice | Rationale | See |
|---|----------|--------|-----------|-----|
| 1 | Code boundary | Framework = orchestration + agents + tools + citation; App = HTTP + DB + UI + enterprise tools | "Can it run without a server?" test | research.md §1 |
| 2 | LLM client | `openai.AsyncOpenAI` directly (D3) | Databricks standardizes on OpenAI; protocol adds complexity without value | research.md §2 |
| 3 | State model | Append-only log + separate pools | Full audit trail, serializable | research.md §3 |
| 4 | Config format | YAML first-class (D5); app uses programmatic construction | Human-readable + flexible | research.md §4 |
| 5 | Event bridge | Pydantic BaseModel with `event_type: Literal[...]` discriminator | Type-safe, serializable | research.md §5 |
| 6 | Prompt management | Defaults per subtype; `system_prompt` + `user_prompt_template` YAML overrides (D8) | Customizable without code changes | research.md §6 |
| 7 | Testing | Separate suites; framework uses mock LLM | Project independence | research.md §7 |
| 8 | Migration | Full replacement, no feature flag (D1) | Not released, clean cut | research.md §8 |
| 9 | Plugin system | Stays in app; config_translator wraps custom phases | Plugins are app-coupled | research.md §9 |
| 10 | Database | Zero DB in framework; CheckpointHandler protocol | Persistence-agnostic | research.md §10 |
| 11 | Agent port | Full port to framework builtins (D2) | No dual mode | — |
| 12 | Researcher mode | ReAct only (D6) | Classic unused | — |
| 13 | Synthesizer mode | ReAct + Reclaim only (D7) | Simple unused | — |
| 14 | Coordinator | Framework builtin (D4) | Essential routing | — |
| 15 | Background investigator | Framework builtin (6th subtype) | Specialized behavior warrants distinct subtype | research.md §15 |
| 16 | plan_and_execute | 8th primitive for plan-execute-evaluate | Plan-execute-evaluate cycle too complex for generic loop+conditional | architecture.md §4.6 |
| 17 | Typed output models | Per-subtype Pydantic contracts + domain events | Simplifies event mapper to trivial forwarder | architecture.md §7.6 |
| 18 | Domain context tracker | Unified event_mapper + state_bridge | Eliminates synchronization issues | architecture.md §15.3 |

## Implementation Order

1. **P0a** → Framework package exists, workspace works, all existing tests pass
2. **P0b-infra** → Executor walks trees, all 8 node types (incl. plan_and_execute), pools, conditions work
3. **P0b-builtins** → Production agents (6 subtypes) + tools, standalone demo works
4. **P0c adapters** → llm_adapter, domain_context, tool_adapter
5. **P0c config_translator** → OrchestrationConfig → WorkflowDefinition (all modes)
6. **P0c orchestrator rewrite** → Full replacement, framework executor
7. **P0c deploy** → Wheel building, bundle inclusion
8. **P0c verification** → E2E tests pass, plugin compat verified
9. **P0d-types** → Citation data structures + config
10. **P0d-pipeline** → 7-stage verification (Stages 1-7)
11. **P0d-integration** → Synthesizer reclaim mode + verification events
12. **P0d-app** → Update domain_context tracker for verification events
13. **P0d verification** → E2E with reclaim mode, verification events in UI

## Total Scope

| Phase | New LOC | Modified LOC | Delta from original |
|-------|---------|-------------|---------------------|
| P0a (PR1: move) | ~200 (Makefiles) | ~500 (path refs) | +80 (UrlRegistry) |
| P0a (PR2: skeleton) | ~1,600 | ~50 | |
| P0b-infra | ~2,800 | 0 | +300 (plan_and_execute executor + rate limiting) |
| P0b-builtins | ~4,320 | 0 | +820 (background + output models + domain events) |
| P0c | ~3,900 | ~500 | -600 (domain_context reduced from 600-800 to 200-300, tool_adapter +50) |
| P0d | ~7,500 | ~400 | 0 |
| **Total** | **~20,370** | **~1,450** | **+1,050** |

## Critical Files

| File | Role |
|------|------|
| `src/deep_research/agent/orchestrator.py` | 3769 LOC monolith — rewritten in P0c |
| `src/deep_research/agent/state.py` | 1187 LOC ResearchState — domain_context target |
| `src/deep_research/agent/nodes/react_researcher.py` | 1228 LOC — primary researcher port source |
| `src/deep_research/agent/nodes/react_synthesizer.py` | 1404 LOC — primary synthesizer port source |
| `src/deep_research/agent/pipeline/protocols.py` | Plugin protocols — must not be modified |
| `src/deep_research/schemas/streaming.py` | SSE event types — domain_context target |
| `src/deep_research/services/llm/client.py` | App's LLMClient — llm_adapter source |
| `src/deep_research/services/citation/pipeline.py` | 2703 LOC — P0d porting source |
| `src/deep_research/services/citation/verification_retriever.py` | 1637 LOC — P0d highest-complexity |
| `config/app.yaml` | Config that drives config_translator |

## Deferred Beyond P0

| Feature | Original Phase | Deferred To |
|---------|---------------|-------------|
| Subworkflow execution | Phase 2 | P2 |
| Parameterized templates (BestOfN, SelfCritique, Debate, MajorityVote) | Phase 2 | P2 |
| Conversation compaction (2-phase) | Phase 3 | Future |
| Tool call deduplication (SHA-256) | Phase 1 | Future |
| Transform builtins (majority_vote, concatenate, etc.) | Phase 1 | Future |
| Checkpointing and recovery | Phase 3 | Future |
| Human-in-the-loop gates | Phase 3 | Future |
| DataFlowGraph static analysis | Phase 3 | Future |
| MLflow tracing integration | Phase 3 | Future |
| Dynamic subworkflows | Phase 3 | Future |

## Complexity Tracking

| Aspect | Complexity | Justification |
|--------|-----------|---------------|
| Monorepo with two projects | Moderate | Required by FR-001. `uv` workspace handles resolution. |
| Full agent port to framework | High | 6 production agents (~4320 LOC) with tuned prompts + typed output models. Most complex part of P0b. |
| AsyncOpenAI direct dependency | Low | Single client type, no protocol dispatch. Constitution I deviation justified. |
| Domain context tracker (event forwarding + persistence delta) | Moderate | ~200-300 LOC, risk reduced by enriched self-contained domain events. Pattern-matching forwarder with PersistenceDelta accumulation. |
| Config translator (all modes) | High | Must handle every OrchestrationConfig permutation. No bail-to-legacy fallback. |
| 7-stage citation pipeline in framework | High | 12 files, ~5000 LOC. Stateless redesign from app's stateful pipeline. |
| ReAct loop extraction | Moderate | Generic reusable component from two app-specific implementations. |
