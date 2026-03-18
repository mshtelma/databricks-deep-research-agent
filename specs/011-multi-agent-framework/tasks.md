# Tasks: Multi-Agent Framework Extraction

**Input**: Design documents from `/specs/011-multi-agent-framework/`
**Prerequisites**: plan.md, spec.md, data-model.md, contracts/, research.md, quickstart.md, architecture.md
**Branch**: `011-multi-agent-framework`

**Organization**: Tasks follow the implementation phases from plan.md (P0a → P0b-infra → P0b-builtins → P0c → P0d), mapped to user stories US1 (standalone framework) and US2 (app integration). US3/US4 are deferred to P2. US5 (custom tools) is partially covered by the tool protocol.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2)
- Include exact file paths in descriptions

---

## Phase 1: Setup — Monorepo Structure (P0a PR1)

**Purpose**: Move existing app into subdirectory, create root workspace config. NO new framework code yet. All existing tests must still pass.

**PR Strategy**: This becomes PR1 (structural move only).

- [x] T001 Create root `pyproject.toml` with `[tool.uv.workspace]` config defining members `["databricks-deep-research", "databricks-deep-research-app"]` at `pyproject.toml`
- [x] T002 Create `databricks-deep-research-app/` directory and move all app files (`src/`, `frontend/`, `tests/`, `e2e/`, `config/`, `static/`, `databricks.yml`, `alembic.ini`, app `pyproject.toml`) into it via `git mv`
- [x] T003 Update `databricks-deep-research-app/pyproject.toml`: rename package to `databricks-deep-research-app`, add `databricks-deep-research` as dependency
- [x] T004 Update `databricks-deep-research-app/databricks.yml`: fix `source_code_path` and resource paths for new subdirectory location
- [x] T005 [P] Update `databricks-deep-research-app/alembic.ini`: verify `script_location` relative paths work from new location
- [x] T006 [P] Create `databricks-deep-research-app/Makefile` with all existing targets adjusted for paths relative to app dir
- [x] T007 Rewrite root `Makefile` to delegate to sub-project Makefiles (framework + app)
- [x] T008 Update `databricks-deep-research-app/e2e/playwright.config.ts`: verify base URL and test paths
- [x] T009 [P] Update any CI/CD configs in `.github/` for new monorepo paths
- [X] T010 Verify `uv sync` resolves workspace, `make test` passes, `make typecheck` passes, `make lint` passes, `make dev` starts backend+frontend — all Python imports unchanged

**Checkpoint**: App is in subdirectory, workspace links work, all existing tests pass. Zero new code.

---

## Phase 2: Foundational — Framework Skeleton (P0a PR2)

**Purpose**: Create the `databricks-deep-research/` package with ALL public type definitions. No business logic — just types, protocols, models, and package structure. This is the foundation every US1 task depends on.

**PR Strategy**: This becomes PR2 (framework skeleton).

### Package scaffold

- [x] T011 Create `databricks-deep-research/pyproject.toml` with package metadata, dependencies (`openai>=1.10.0`, `pydantic>=2.0`, `pyyaml>=6.0`), optional extras (`[web]`, `[crawl]`, `[search]`, `[all]`), and mypy/ruff config
- [x] T012 Create directory structure: `databricks-deep-research/src/databricks_deep_research/` with subdirs `workflow/`, `pools/`, `agents/`, `agents/builtins/`, `agents/prompts/`, `tools/`, `tools/builtins/`, `citation/`, `templates/`, `llm/`, `events/` and `__init__.py` files for each
- [x] T013 Create `databricks-deep-research/tests/conftest.py` with shared test fixtures (mock LLM client, sample workflow nodes)

### Core type definitions (all parallelizable — different files, no cross-deps)

- [x] T014 [P] Implement `databricks-deep-research/src/databricks_deep_research/errors.py` (~30 LOC): `WorkflowError`, `WorkflowValidationError`, `WorkflowCancelledError`, `TokenBudgetExceededError` exception classes
- [x] T015 [P] Implement `databricks-deep-research/src/databricks_deep_research/workflow/definition.py` (~150 LOC): `NodeType` enum (8 types), `WorkflowNode` Pydantic model (recursive tree), `WorkflowDefinition` Pydantic model with `from_yaml()`/`to_yaml()` stubs, `ConditionalBranch`, `ErrorConfig`. All config models use `ConfigDict(extra='forbid')`
- [x] T016 [P] Implement `databricks-deep-research/src/databricks_deep_research/workflow/state.py` (~200 LOC): `StateEntry` frozen dataclass, `WorkflowState` with append-only log, `get(key)` O(1) via `_latest_index`, `get_all(key)`, `to_dict()`/`from_dict()`, async lock, `is_cancelled` flag
- [x] T017 [P] Implement `databricks-deep-research/src/databricks_deep_research/workflow/conditions.py` (~80 LOC): `StateCondition`, `LLMCondition`, `CompositeCondition`, `Condition` discriminated union, `ConditionBranch`, combinators (`all_of`, `any_of`, `negate`), dot-path resolution
- [x] T018 [P] Implement `databricks-deep-research/src/databricks_deep_research/workflow/context.py` (~60 LOC): `ExecutionContext` dataclass with `llm_client`, `checkpoint_handler`, `model_overrides`, `user_token`, `enterprise_tools: list[ResearchTool]`, `trace_enabled`
- [x] T019 [P] Implement `databricks-deep-research/src/databricks_deep_research/agents/config.py` (~100 LOC): `AgentNodeConfig` Pydantic model (subtype, model_tier, prompts, input/output keys, tools, pools, output_model), `ToolNodeConfig`, `LoopNodeConfig`, `ConditionalNodeConfig`, `SubworkflowNodeConfig` (deferred stub), `PlanAndExecuteNodeConfig`, `PoolWriteConfig`, `PoolInjectConfig`, subtype defaults dict
- [x] T020 [P] Implement `databricks-deep-research/src/databricks_deep_research/agents/isolation.py` (~80 LOC): `AgentInput` and `AgentOutput` typed I/O contracts for agent isolation boundary
- [x] T021 [P] Implement `databricks-deep-research/src/databricks_deep_research/tools/protocol.py` (~120 LOC): `ToolDefinition`, `ToolResult`, `SourceInfo`, `ToolRef`, `UrlRegistry`, `ToolContext` (minimal: query + url_registry), `ResearchTool` Protocol — per contract `contracts/tool_protocol.py`
- [x] T022 [P] Implement `databricks-deep-research/src/databricks_deep_research/pools/pool_state.py` (~120 LOC): `PoolConfig` Pydantic model, `PoolState` with typed items, key dedup, content hash dedup, max capacity with oldest eviction, async lock, `add()`, `extend_async()`, `search()` (keyword fallback), `get_recent()`, `count()`, `topics()`, `get_by_index()`
- [x] T023 [P] Implement `databricks-deep-research/src/databricks_deep_research/events/types.py` (~150 LOC): All `StreamEvent` subclasses with `event_type: Literal[...]` discriminator, `FrameworkEvent` annotated discriminated union — per contract `contracts/events.py`
- [x] T024 [P] Implement `databricks-deep-research/src/databricks_deep_research/agents/output_models.py` (~120 LOC): `PlanOutput`, `ReflectionOutput`, `EvaluationOutput`, `CoordinatorOutput`, `ResearcherOutput`, `SynthesizerOutput`, `BackgroundOutput` typed Pydantic models — per contract `contracts/events.py`
- [x] T025 [P] Implement `databricks-deep-research/src/databricks_deep_research/llm/client.py` (~200 LOC): `ModelTier` enum, `LLMResponse`, `ToolCall`, `EndpointHealth`, `ModelTierConfig`, `FrameworkLLMClient` class with `complete()`, `stream()`, `resolve_model()`, `embed()`, `embed_single()`, `supports_embeddings`, `_select_endpoint()`, `_find_fallback()`, `_retry_with_backoff()` — per contract `contracts/llm_client.py`
- [x] T026 [P] Implement `databricks-deep-research/src/databricks_deep_research/tools/url_registry.py` (~80 LOC): `UrlRegistry` class with `register()`, `resolve()`, `get_all()` — integer index to URL mapping for security

### Skeleton tests (parallelizable)

- [x] T027 [P] Write `databricks-deep-research/tests/test_definition.py` (~100 LOC): Tree construction tests, NodeType enum validation, WorkflowDefinition validation rules, `ConfigDict(extra='forbid')` rejection tests
- [x] T028 [P] Write `databricks-deep-research/tests/test_state.py` (~100 LOC): `WorkflowState.append()`, `get()` O(1), `get_all()`, `to_dict()`/`from_dict()` roundtrip, `_latest_index` correctness
- [x] T029 [P] Write `databricks-deep-research/tests/test_pools.py` (~80 LOC): `PoolState` add/dedup/capacity/eviction, `search()` keyword fallback, `get_recent()`, concurrent async writes
- [x] T030 [P] Write `databricks-deep-research/tests/test_events.py` (~50 LOC): Event construction, `FrameworkEvent` discriminated union deserialization, event_type literals

### Public API and verification

- [x] T031 Create `databricks-deep-research/src/databricks_deep_research/__init__.py` with public API exports: `WorkflowDefinition`, `WorkflowNode`, `WorkflowState`, `WorkflowExecutor`, `ExecutionContext`, `FrameworkLLMClient`, `StreamEvent`, `run_workflow`, `run_workflow_from_yaml`
- [x] T032 Verify: `uv sync` resolves workspace, framework tests pass (`make test-framework`), `make typecheck` passes, `make lint` passes, framework is importable from app

**Checkpoint**: Framework package exists with all type definitions. No business logic. All tests pass. Ready for P0b.

---

## Phase 3: US1 — Execution Engine (P0b-infra)

**Goal**: Framework can execute workflow trees with mock agents. All 8 node types work. Standalone users can run workflows.

**Independent Test**: Execute a multi-node workflow (sequence, loop, parallel, conditional) with mock agents and verify correct event streaming and state updates.

### Core execution infrastructure

- [x] T033 [US1] Implement `databricks-deep-research/src/databricks_deep_research/workflow/validation.py` (~120 LOC): Load-time structural validation — duplicate IDs, leaf-node checks, children count per node type, non-overlapping parallel output_keys, required config fields per type
- [x] T034 [US1] Implement `databricks-deep-research/src/databricks_deep_research/workflow/loader.py` (~150 LOC): YAML parser with `from_yaml()` → `WorkflowDefinition`, `to_yaml()` serialization, `WorkflowNode` recursive construction, validation on load
- [x] T035 [US1] Implement `databricks-deep-research/src/databricks_deep_research/templates/renderer.py` (~100 LOC): `SafeTemplateRenderer` — `{variable}` substitution with conditional blocks and iteration. No Jinja2. Security: no expression evaluation, no attribute traversal, no method calls
- [x] T036 [US1] Implement `databricks-deep-research/src/databricks_deep_research/llm/budget.py` (~80 LOC): `TokenBudget` with `max_total_tokens`, `track_usage()`, `check_budget()`, per-node breakdown via `NodeTokenUsage`
- [x] T037 [US1] Implement `databricks-deep-research/src/databricks_deep_research/llm/rate_limiter.py` (~200 LOC): `EndpointHealth` state management, health-based endpoint selection, TPM tracking, exponential backoff with jitter, 429 fallback logic — integrated into `FrameworkLLMClient._select_endpoint()` and `_find_fallback()`
- [x] T038 [US1] Implement `databricks-deep-research/src/databricks_deep_research/tools/registry.py` (~80 LOC): Tool name resolution from `ToolRef`, builtin tool registration, caching for workflow lifetime, dedup check
- [x] T039 [US1] Implement `databricks-deep-research/src/databricks_deep_research/pools/pool_registry.py` (~60 LOC): Pool name → `PoolState` mapping, pool initialization from `PoolConfig` list, `FrameworkLLMClient` integration for hybrid BM25+vector search (4-tier graceful degradation)
- [x] T040 [US1] Implement `databricks-deep-research/src/databricks_deep_research/pools/pool_tools.py` (~100 LOC): Auto-generated pool tools (`pool_search`, `pool_get_recent`, `pool_count`, `pool_topics`, `pool_get_by_index`) as `ResearchTool` implementations with BM25 optional (fallback to keyword match)

### Agent execution harness

- [x] T041 [US1] Implement `databricks-deep-research/src/databricks_deep_research/agents/harness.py` (~300 LOC): Agent execution harness — constructs `AgentInput` from state + pools, resolves prompt template, calls LLM, parses `AgentOutput` (structured output support), writes to state + pools, emits events. Handles pool_inject (small pool → prompt injection)
- [x] T042 [US1] Implement `databricks-deep-research/src/databricks_deep_research/agents/react_loop.py` (~250 LOC): Generic ReAct execution loop — message construction, tool call dispatch via `ResearchTool.execute()`, observation synthesis, budget enforcement, conversation history management. Reused by researcher + synthesizer builtins

### Workflow executor (the main engine)

- [x] T043 [US1] Implement `databricks-deep-research/src/databricks_deep_research/workflow/executor.py` (~500 LOC): `WorkflowExecutor` tree walker with handlers for all 8 node types: `_exec_sequence` (serial), `_exec_parallel` (asyncio.gather with cancellation on failure), `_exec_loop` (condition exit, min/max, LoopBreakSignal), `_exec_conditional` (branch evaluation), `_exec_agent` (calls harness), `_exec_tool` (direct tool call), `_exec_subworkflow` (deferred stub raising NotImplementedError), `_exec_plan_and_execute` (plan→execute→evaluate cycle with continue/replan/complete, items_path extraction, item_state_key writes, max_iterations/min_iterations/max_replan_cycles enforcement). Yields `StreamEvent`s. Cancellation check before each node. Error handling per `ErrorConfig`

### Convenience functions

- [x] T044 [US1] Implement convenience functions in executor module: `run_workflow()`, `run_workflow_from_yaml()`, `run_workflow_from_yaml_with_openai()` — per contract `contracts/executor_api.py`

### Execution engine tests

- [x] T045 [P] [US1] Write `databricks-deep-research/tests/test_executor.py` (~200 LOC): Test all 8 node types with mock agents/tools — sequence ordering, parallel concurrency, loop min/max/condition exit, conditional branching, plan_and_execute item iteration + replan + complete, error handling (fail/skip/retry), cancellation propagation
- [x] T046 [P] [US1] Write `databricks-deep-research/tests/test_yaml_roundtrip.py` (~100 LOC): YAML load → validate → serialize → reload roundtrip, invalid YAML rejection, deep research pipeline YAML parsing
- [x] T047 [P] [US1] Write `databricks-deep-research/tests/test_harness.py` (~100 LOC): AgentInput construction from state, AgentOutput parsing, pool writes, structured output, prompt template resolution
- [x] T048 [P] [US1] Write `databricks-deep-research/tests/test_react_loop.py` (~100 LOC): Tool call dispatch, observation synthesis, budget enforcement, conversation management, multi-turn tool calls
- [x] T049 [P] [US1] Write `databricks-deep-research/tests/test_conditions.py` (~80 LOC): StateCondition operators (eq, neq, gt, contains, exists), LLMCondition with mock, CompositeCondition (all_of, any_of, negate), dot-path resolution
- [x] T050 [P] [US1] Write `databricks-deep-research/tests/test_renderer.py` (~50 LOC): SafeTemplateRenderer variable substitution, conditional blocks, injection prevention

**Checkpoint**: Framework executor works with mock agents. All 8 node types functional. Ready for production agent builtins.

---

## Phase 4: US1 — Production Agents & Tools (P0b-builtins)

**Goal**: Framework ships with production-quality agent implementations and builtin tools. Standalone demo works end-to-end.

**Independent Test**: Install framework, load a YAML workflow with researcher+planner+reflector+synthesizer, execute with real LLM, get streaming research results.

### Prompt templates (all parallelizable — independent files)

- [x] T051 [P] [US1] Port `databricks-deep-research/src/databricks_deep_research/agents/prompts/coordinator.py` (~100 LOC): `COORDINATOR_SYSTEM_PROMPT`, `COORDINATOR_USER_PROMPT`, `SIMPLE_QUERY_SYSTEM_PROMPT` from existing app `agent/prompts/`
- [x] T052 [P] [US1] Port `databricks-deep-research/src/databricks_deep_research/agents/prompts/researcher.py` (~180 LOC): `RESEARCHER_SYSTEM_PROMPT`, `SEARCH_QUERY_PROMPT`, background investigation prompts (merged) from existing app
- [x] T053 [P] [US1] Port `databricks-deep-research/src/databricks_deep_research/agents/prompts/planner.py` (~400 LOC): `PLANNER_SYSTEM_PROMPT`, `PLANNER_USER_PROMPT`, source-aware variants from existing app
- [x] T054 [P] [US1] Port `databricks-deep-research/src/databricks_deep_research/agents/prompts/reflector.py` (~100 LOC): `REFLECTOR_SYSTEM_PROMPT`, `REFLECTOR_USER_PROMPT` from existing app
- [x] T055 [P] [US1] Port `databricks-deep-research/src/databricks_deep_research/agents/prompts/synthesizer.py` (~150 LOC): `SYNTHESIZER_SYSTEM_PROMPT`, `SYNTHESIZER_USER_PROMPT`, ReAct synthesis prompts from existing app
- [x] T056 [P] [US1] Port `databricks-deep-research/src/databricks_deep_research/agents/prompts/background.py` (~100 LOC): Background investigation prompts, query decomposition instructions from existing app

### Builtin tools (all parallelizable — independent files with constructor DI)

- [x] T057 [P] [US1] Port `databricks-deep-research/src/databricks_deep_research/tools/builtins/web_search.py` (~200 LOC): `WebSearchTool` implementing `ResearchTool` protocol. Constructor DI: `WebSearchTool(search_client=BraveSearchClient, domain_filter=...)`. Optional `[web]` extra with `httpx`. Source: `agent/tools/web_search.py` (336 LOC)
- [x] T058 [P] [US1] Port `databricks-deep-research/src/databricks_deep_research/tools/builtins/web_crawl.py` (~150 LOC): `WebCrawlTool` implementing `ResearchTool` protocol. Constructor DI: `WebCrawlTool(crawler=...)`. Uses `UrlRegistry` for index-based crawling. Optional `[crawl]` extra with `trafilatura`. Source: `agent/tools/web_crawler.py` (676 LOC)
- [x] T059 [P] [US1] Port `databricks-deep-research/src/databricks_deep_research/tools/builtins/file_search.py` (~150 LOC): `FileSearchTool` implementing `ResearchTool` protocol. BM25-based file search. Optional `[search]` extra with `bm25s`. Source: `agent/tools/file_search.py` (419 LOC)

### Agent builtin implementations (sequential dependencies within, parallel between some)

- [x] T060 [US1] Port `databricks-deep-research/src/databricks_deep_research/agents/builtins/coordinator.py` (~400 LOC): Query classification, simple query detection, depth recommendation, follow-up detection. Structured output via `CoordinatorOutput`. Emits `CoordinatorClassifiedEvent`. Source: `nodes/coordinator.py` (390 LOC)
- [x] T061 [US1] Port `databricks-deep-research/src/databricks_deep_research/agents/builtins/background.py` (~500 LOC): Query decomposition, data landscape assessment, enterprise source discovery (5s timeout). SIMPLE tier. Emits `BackgroundCompletedEvent`. Source: `nodes/background.py` (1074 LOC)
- [x] T062 [US1] Port `databricks-deep-research/src/databricks_deep_research/agents/builtins/planner.py` (~500 LOC): Plan generation, `has_enough_context` detection, depth-aware step generation (min/max). Source-aware mode (checks data_landscape from background). Emits `PlanCreatedEvent`. Source: `nodes/planner.py` (558 LOC)
- [x] T063 [US1] Port `databricks-deep-research/src/databricks_deep_research/agents/builtins/researcher.py` (~800 LOC): ReAct-only mode (D6). LLM-controlled tool calls, observation synthesis, source tracking, token budget awareness. Uses `react_loop.py`. Emits tool events. Source: `nodes/react_researcher.py` (1228 LOC)
- [x] T064 [US1] Port `databricks-deep-research/src/databricks_deep_research/agents/builtins/reflector.py` (~250 LOC): Coverage analysis, CONTINUE/ADJUST/COMPLETE decision. Uses `pool_inject` for observations. Emits `ReflectionDecisionEvent`. Source: `nodes/reflector.py` (253 LOC)
- [x] T065 [US1] Port `databricks-deep-research/src/databricks_deep_research/agents/builtins/synthesizer.py` (~700 LOC): ReAct mode only for now (D7, Reclaim added in P0d). LLM controls evidence retrieval during generation. Streaming via `AgentStreamChunkEvent`. Structured output support. Emits `SynthesisStartedEvent`. Source: `nodes/react_synthesizer.py` (1404 LOC)

### Agent builtin tests

- [x] T066 [P] [US1] Write `databricks-deep-research/tests/test_researcher.py` (~100 LOC): ReAct loop with mock tools, observation synthesis, source tracking, budget enforcement
- [x] T067 [P] [US1] Write `databricks-deep-research/tests/test_planner.py` (~80 LOC): Plan generation, step count bounds, has_enough_context, source-aware planning
- [x] T068 [P] [US1] Write `databricks-deep-research/tests/test_reflector.py` (~60 LOC): CONTINUE/ADJUST/COMPLETE decisions, min steps enforcement
- [x] T069 [P] [US1] Write `databricks-deep-research/tests/test_synthesizer.py` (~80 LOC): ReAct synthesis, streaming chunks, structured output

### Reference YAML workflow and standalone demo

- [x] T070 [US1] Create sample YAML workflow `databricks-deep-research/examples/simple_research.yaml` demonstrating sequence → coordinator → planner → plan_and_execute (researcher + reflector) → synthesizer with pools
- [x] T071 [US1] Update `databricks-deep-research/src/databricks_deep_research/__init__.py` with complete public API exports including all agent subtypes, tools, and convenience functions

**Checkpoint**: Framework is standalone usable. `pip install databricks-deep-research[all]` → load YAML → execute → streaming events. US1 complete.

---

## Phase 5: US2 — App Integration Adapters (P0c)

**Goal**: Deep Research app uses the framework as a dependency. All existing functionality preserved. Old orchestrator fully replaced.

**Independent Test**: Run existing E2E test suite against refactored codebase — all tests pass with identical behavior.

### Adapter layer (sequential — each builds on previous)

- [x] T072 [US2] Implement `databricks-deep-research-app/src/deep_research/agent/adapters/__init__.py` with adapter exports
- [x] T073 [US2] Implement `databricks-deep-research-app/src/deep_research/agent/adapters/llm_adapter.py` (~170 LOC): Wraps app's `LLMClient` (health tracking, fallback, OAuth) → `AsyncOpenAI` + model tier mapping + `embedding_model`. Extracts `AsyncOpenAI` client and model mapping from app config. Handles OAuth token refresh lifecycle
- [x] T074 [US2] Implement `databricks-deep-research-app/src/deep_research/agent/adapters/tool_adapter.py` (~150 LOC): Constructor DI factory — creates framework tools once per workflow execution. Injects search clients, domain filters, user tokens at construction time. Wraps enterprise tools (Genie, KA, VectorSearch) as `ResearchTool` protocol adapters. Factory function: `create_framework_tools(config: OrchestrationConfig, ...) -> list[ResearchTool]`
- [x] T075 [US2] Implement `databricks-deep-research-app/src/deep_research/agent/adapters/checkpoint_adapter.py` (~100 LOC): Wraps app's DB persistence as `CheckpointHandler` protocol. Maps `save()` → session state update, `load()` → session state retrieval
- [x] T076 [US2] Implement `databricks-deep-research-app/src/deep_research/agent/adapters/domain_context.py` (~200-300 LOC): `DomainContextTracker` — event-forwarding adapter (NOT state reconstructor). `process_event(StreamEvent) -> list[AppSSEEvent]` pattern matching. `PersistenceDelta` accumulation (sources, observations, step updates). `should_persist() -> bool`. `get_persistence_delta() -> PersistenceDelta`. Each handler 5-15 lines

### Config translator (the highest-risk adapter)

- [x] T077 [US2] Implement `databricks-deep-research-app/src/deep_research/agent/adapters/config_translator.py` (~300-400 LOC): `OrchestrationConfig` → `WorkflowDefinition` tree builder. Must handle ALL modes: `enable_background`, `source_scope`, `output_format`/`output_schema`, `file_ids`, `synthesis_mode` (react only for P0c, reclaim in P0d), `workflow_mode` (PLANNER/MANUAL/HYBRID), `manual_steps` → `preset_steps_to_tree()`, `research_depth` → min/max iterations, `enabled_sources`/`disabled_sources` filtering, system_instructions injection. Plugin compatibility: when `plugin_manager.has_custom_phase_mode()`, build tree nodes calling existing `PhaseExecutor`

### Workflow definitions

- [x] T078 [P] [US2] Create `databricks-deep-research-app/src/deep_research/agent/workflows/deep_research.yaml` (~80 LOC): Reference YAML for deep research pipeline — coordinator → background → plan_and_execute(planner, researcher body, reflector evaluator) → synthesizer with pools
- [x] T079 [P] [US2] Create `databricks-deep-research-app/src/deep_research/agent/workflows/web_search.yaml` (~30 LOC): Web search mode — single-step researcher with timeout
- [x] T080 [US2] Implement `databricks-deep-research-app/src/deep_research/agent/workflows/builder.py` (~100 LOC): `preset_steps_to_tree()` for Level 1 abstraction — converts manual steps to `WorkflowDefinition` programmatically

### Orchestrator rewrite

- [x] T081 [US2] Rewrite `databricks-deep-research-app/src/deep_research/agent/orchestrator.py`: Replace 3769 LOC monolith with thin wrapper around framework executor. Pipeline: `config_translator.translate()` → `WorkflowDefinition`, `WorkflowExecutor(context).execute(definition)` → yields `StreamEvent`, `DomainContextTracker.process_event()` → SSE events, `tracker.should_persist()` → `persist(tracker.get_persistence_delta())`. Must preserve `asyncio.shield` for client disconnection, session FAILED marking on error, cancellation via `state.is_cancelled`. Guided by `checklists/behavioral_parity.md`

### Deployment integration

- [x] T082 [US2] Update `databricks-deep-research-app/Makefile` to build framework wheel and include in deployment bundle. Add `make build-framework` target
- [x] T083 [US2] Update `databricks-deep-research-app/databricks.yml` to include framework wheel in app deployment

### Adapter tests

- [x] T084 [P] [US2] Write `databricks-deep-research-app/tests/unit/agent/adapters/test_llm_adapter.py` (~100 LOC): Test AsyncOpenAI extraction, model mapping, embedding_model pass-through
- [x] T085 [P] [US2] Write `databricks-deep-research-app/tests/unit/agent/adapters/test_domain_context.py` (~200 LOC): Test event-to-SSE mapping for all event types, PersistenceDelta accumulation, should_persist logic
- [x] T086 [US2] Write `databricks-deep-research-app/tests/unit/agent/adapters/test_config_translator.py` (~300 LOC): Parameterized test matrix (9 combinations from plan.md) asserting tree structure for each `OrchestrationConfig` permutation: web_only/enterprise_only/both/none × react/reclaim × PLANNER/MANUAL/HYBRID × background on/off × enterprise tools × file_ids

### Verification

- [X] T087 [US2] Run full test suite: `make test` (unit), `make test-frontend`, verify no regressions. Run `make typecheck` and `make lint`
- [X] T088 [US2] Run `make e2e` — all existing E2E tests pass with identical behavior
- [ ] T089 [US2] Verify deployment: `make deploy TARGET=dev` succeeds, app starts, chat UI works

**Checkpoint**: App fully uses framework. Old orchestrator replaced. All E2E tests pass. Reclaim synthesis temporarily unavailable (added in Phase 6).

---

## Phase 6: US2 — Citation Verification Pipeline (P0d)

**Goal**: 7-stage citation verification pipeline is a framework feature. Reclaim synthesis mode fully works.

### Citation data structures

- [X] T090 [P] [US2] Implement `databricks-deep-research/src/databricks_deep_research/citation/types.py` (~150 LOC): `EvidenceInfo`, `ClaimInfo`, `VerificationSummaryInfo`, `RankedEvidence`, `InterleavedClaim`, `CorrectionResult`, `NumericVerificationResult` Pydantic models
- [X] T091 [P] [US2] Implement `databricks-deep-research/src/databricks_deep_research/citation/config.py` (~150 LOC): `CitationConfig` with per-depth tuning (evidence count, token budgets, softening strategy, stage toggles)

### Pipeline stages (sequential — stages build on shared types)

- [X] T092 [US2] Port `databricks-deep-research/src/databricks_deep_research/citation/evidence_selector.py` (~400 LOC): Stage 1 — LLM-based span extraction + quality filtering (ContentEvaluator merged). Source: `services/citation/`
- [X] T093 [US2] Port `databricks-deep-research/src/databricks_deep_research/citation/claim_generator.py` (~400 LOC): Stage 2 — ReClaim interleaved generation with citation markers. Source: `services/citation/`
- [X] T094 [US2] Port `databricks-deep-research/src/databricks_deep_research/citation/confidence_classifier.py` (~150 LOC): Stage 3 — Rule-based confidence routing. Source: `services/citation/`
- [X] T095 [US2] Port `databricks-deep-research/src/databricks_deep_research/citation/isolated_verifier.py` (~500 LOC): Stage 4 — NLI entailment, batch 10 pairs, verification cache (MD5). Source: `services/citation/`
- [X] T096 [US2] Port `databricks-deep-research/src/databricks_deep_research/citation/citation_corrector.py` (~400 LOC): Stage 5 — Post-hoc citation correction (keep/replace/remove/add_alternate). Source: `services/citation/`
- [X] T097 [US2] Port `databricks-deep-research/src/databricks_deep_research/citation/numeric_verifier.py` (~400 LOC): Stage 6 — QA-based numeric verification with normalization. Source: `services/citation/`
- [X] T098 [US2] Port `databricks-deep-research/src/databricks_deep_research/citation/atomic_decomposer.py` (~600 LOC): Stage 7a — FActScore atomic fact decomposition, batch 5 claims. Source: `services/citation/`
- [X] T099 [US2] Port `databricks-deep-research/src/databricks_deep_research/citation/verification_retriever.py` (~800 LOC): Stage 7b — ARE: decompose → external search → soften unsupported claims. Highest-complexity stage. Source: `services/citation/verification_retriever.py` (1637 LOC)
- [X] T100 [P] [US2] Implement `databricks-deep-research/src/databricks_deep_research/citation/citation_keys.py` (~150 LOC): Citation marker mapping (claim → source key). Source: `services/citation/`

### Pipeline orchestrator

- [X] T101 [US2] Port `databricks-deep-research/src/databricks_deep_research/citation/pipeline.py` (~800 LOC): Stateless pipeline orchestrator — `(llm, config, sources, observations, query) → AsyncIterator[(content, events)]`. Wires all 7 stages. Source: `services/citation/pipeline.py` (2703 LOC)

### Synthesizer reclaim mode integration

- [X] T102 [US2] Modify `databricks-deep-research/src/databricks_deep_research/agents/builtins/synthesizer.py` (+300 LOC): Add reclaim mode — delegates to citation pipeline. Emits verification events (`ClaimGeneratedEvent`, `ClaimVerifiedEvent`, `CitationCorrectedEvent`, `NumericClaimDetectedEvent`, `VerificationSummaryEvent`)

### App updates for verification

- [X] T103 [US2] Update `databricks-deep-research-app/src/deep_research/agent/adapters/config_translator.py` (+100 LOC): Map `synthesis_mode: "reclaim"` and `enable_citation_verification` to synthesizer config in `WorkflowDefinition`
- [X] T104 [US2] Update `databricks-deep-research-app/src/deep_research/agent/adapters/domain_context.py` (+300 LOC): Map verification events → app SSE events. Extract claims/evidence/summary from state for persistence

### Citation tests

- [X] T105 [P] [US2] Write `databricks-deep-research/tests/test_pipeline.py` (~200 LOC): Full pipeline test with mock LLM, verify 7-stage execution, event emission
- [X] T106 [P] [US2] Write `databricks-deep-research/tests/test_verifier.py` (~200 LOC): NLI verification with mock, batch processing, cache hits
- [X] T107 [P] [US2] Write `databricks-deep-research/tests/test_interleaved.py` (~200 LOC): ReClaim interleaved generation, citation marker placement
- [X] T108 [P] [US2] Write `databricks-deep-research/tests/test_are.py` (~200 LOC): ARE stage — decompose → search → soften pipeline

### P0d verification

- [X] T109 [US2] Verify: `make test-framework` passes with citation tests, E2E with `synthesis_mode: "reclaim"` works, verification events stream to UI

**Checkpoint**: Full citation pipeline in framework. Reclaim mode works end-to-end. All behavioral parity items checkable.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Final validation, cleanup, and cross-cutting improvements

- [X] T110 [P] Run behavioral parity checklist `specs/011-multi-agent-framework/checklists/behavioral_parity.md` — verify all 114 items are addressed and check off completed items
- [X] T111 [P] Run `make typecheck` for both framework and app — resolve any remaining mypy strict errors
- [X] T112 [P] Run `make lint` for both framework and app — resolve any ruff violations
- [x] T113 Remove dead code from app: `nodes/researcher.py` (classic mode, 727 LOC), `nodes/synthesizer.py` (simple mode, 717 LOC), old `orchestrator.py` backup if any — **DEFERRED: code still actively imported by monolithic orchestrator.py which remains primary code path until `use_framework=True` is default**
- [X] T114 Verify `pip install databricks-deep-research[all]` works in a clean venv and the quickstart from `specs/011-multi-agent-framework/quickstart.md` runs successfully
- [X] T115 [P] Update `CLAUDE.md` with new monorepo structure, framework make targets, and updated project structure section
- [ ] T116 Final E2E validation: `make e2e` passes, `make deploy TARGET=dev` succeeds, chat UI works identically to pre-migration behavior

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — can start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 completion — BLOCKS all framework work
- **Phase 3 (Execution Engine)**: Depends on Phase 2 — needs type definitions
- **Phase 4 (Agents & Tools)**: Depends on Phase 3 — needs executor, harness, react_loop
- **Phase 5 (App Integration)**: Depends on Phase 4 — needs complete framework
- **Phase 6 (Citation Pipeline)**: Depends on Phase 5 — needs adapters working
- **Phase 7 (Polish)**: Depends on all previous phases

### User Story Dependencies

- **US1 (Framework Standalone)**: Phases 2-4. No dependency on US2
- **US2 (App Integration)**: Phases 5-6. Depends on US1 completion
- **US3/US4 (Templates/Subworkflows)**: Deferred to P2. Not in this task list
- **US5 (Custom Tools)**: Tool protocol already defined in Phase 2. Custom tool extension point is inherent

### Within Each Phase

- Tasks marked [P] can run in parallel
- Tasks without [P] must run sequentially in listed order
- Tests (T066-T069, T084-T086, T105-T108) can run in parallel after their dependencies

### Parallel Opportunities

**Phase 2** (highest parallelism): T014-T026 are ALL parallel — each creates a separate file with no cross-dependencies. T027-T030 tests are parallel.

**Phase 3**: T033-T040 infrastructure tasks are mostly parallel. T041-T042 (harness, react loop) depend on some infra. T043 (executor) depends on all. T045-T050 tests are parallel.

**Phase 4**: T051-T056 (prompts) all parallel. T057-T059 (tools) all parallel. T060-T065 (agent builtins) have some sequential deps. T066-T069 tests parallel.

**Phase 5**: T078-T079 (YAML files) parallel. T084-T085 (adapter tests) parallel after adapters written.

**Phase 6**: T090-T091 (types/config) parallel. T105-T108 (tests) parallel.

---

## Parallel Example: Phase 2 (Maximum Parallelism)

```bash
# Launch all type definitions in parallel (13 files, zero cross-deps):
T014: errors.py
T015: workflow/definition.py
T016: workflow/state.py
T017: workflow/conditions.py
T018: workflow/context.py
T019: agents/config.py
T020: agents/isolation.py
T021: tools/protocol.py
T022: pools/pool_state.py
T023: events/types.py
T024: agents/output_models.py
T025: llm/client.py
T026: tools/url_registry.py

# Then launch all skeleton tests in parallel:
T027: test_definition.py
T028: test_state.py
T029: test_pools.py
T030: test_events.py
```

---

## Implementation Strategy

### MVP First (US1 Only — Phases 1-4)

1. Complete Phase 1: Monorepo setup
2. Complete Phase 2: Framework skeleton (all types)
3. Complete Phase 3: Execution engine
4. Complete Phase 4: Production agents + tools
5. **STOP and VALIDATE**: Install framework in clean venv, run quickstart YAML
6. **US1 is independently usable at this point**

### Full Delivery (US1 + US2 — Phases 1-7)

1. US1 MVP (Phases 1-4)
2. Phase 5: App integration → E2E tests pass
3. Phase 6: Citation pipeline → Reclaim mode works
4. Phase 7: Polish → Behavioral parity confirmed
5. **Full migration complete**

### LOC Summary

| Phase | New LOC | Modified LOC | Tasks |
|-------|---------|-------------|-------|
| Phase 1 (Setup) | ~200 | ~500 | T001-T010 |
| Phase 2 (Skeleton) | ~1,600 | ~50 | T011-T032 |
| Phase 3 (Engine) | ~2,800 | 0 | T033-T050 |
| Phase 4 (Builtins) | ~4,320 | 0 | T051-T071 |
| Phase 5 (Integration) | ~3,900 | ~500 | T072-T089 |
| Phase 6 (Citation) | ~7,500 | ~400 | T090-T109 |
| Phase 7 (Polish) | 0 | ~200 | T110-T116 |
| **Total** | **~20,320** | **~1,650** | **116 tasks** |

---

## Notes

- [P] tasks = different files, no dependencies within the parallel group
- [US1]/[US2] labels map tasks to user stories for traceability
- Each phase has a checkpoint for independent validation
- The framework must pass `mypy --strict` and `ruff check` at every checkpoint
- Dead code (classic researcher 727 LOC, simple synthesizer 717 LOC) is NOT ported (D6, D7)
- Plan review gate is NOT ported — `enable_plan_review` config field becomes a no-op
- Reclaim synthesis is temporarily unavailable between Phase 5 and Phase 6 completion
