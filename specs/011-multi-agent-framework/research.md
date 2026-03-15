# Research: Multi-Agent Framework Extraction

**Feature**: 011-multi-agent-framework | **Date**: 2026-03-08
**Input**: spec.md + architecture.md

## Decision 1: Code Boundary Split

**Decision**: The framework (`databricks-deep-research`) gets all orchestration, agent implementations (full production port), tools (core builtins), state, pools, conditions, template renderer, LLM client wrapper, citation/verification pipeline, and streaming event code. The app retains FastAPI, database, persistence, middleware, API endpoints, schemas, frontend, deployment, plugin system, and enterprise tools (Genie, KA, VectorSearch).

**Rationale**: The boundary follows the "can it run without a database or web server?" test. Orchestration, agents, tools, and state management are domain-independent. Database persistence, HTTP APIs, UI, and enterprise integrations are application concerns.

**Enterprise tools stay in app**: Genie, KnowledgeAssistant, UserVectorSearch, VectorSearch depend on Databricks SDK and data source configurations. They are wrapped via `tool_adapter.py` for the framework's `ResearchTool` protocol.

**Alternatives considered**:
- **Thin framework (only tree executor + state)**: Rejected — would force every consumer to reimplement agents, tools, search, verification.
- **Fat framework (includes FastAPI/DB/enterprise)**: Rejected — would make framework unusable outside Deep Research context.

## Decision 2: LLM Client — AsyncOpenAI Directly

**Decision**: The framework imports `openai.AsyncOpenAI` as a direct dependency. `llm/client.py` provides a thin concrete wrapper with model tier mapping (`ModelTier` enum → model name), structured output support (`LLMResponse.structured: Any | None`), and token usage tracking. No `typing.Protocol` abstraction.

**Rationale**: Databricks standardizes on OpenAI-compatible endpoints — all model serving endpoints expose the OpenAI API. A Protocol abstraction would add complexity without practical value since no non-OpenAI backends are planned. The app wraps its existing `LLMClient` (health tracking, fallback, OAuth refresh) via `llm_adapter.py` to provide the `AsyncOpenAI` instance + model mapping.

**Constitution Principle I deviation**: Acknowledged. The framework depends on `openai` directly rather than using WorkspaceClient. Justified because: (1) Databricks standardizes on OpenAI, (2) the app still uses WorkspaceClient to obtain the client, (3) protocol abstraction adds complexity without practical benefit.

**Rate limiting**: Endpoint health tracking (consecutive errors, rate limit windows), TPM budget enforcement, 429 fallback to alternate endpoints, and exponential backoff with jitter are ported INTO `FrameworkLLMClient`. Standalone users get production-quality resilience without needing the app's `LLMClient`. OAuth token refresh remains in the app's `llm_adapter.py` — it depends on Databricks SDK and is not a framework concern.

**Alternatives considered**:
- **`typing.Protocol` with `complete()` method**: Original plan. Rejected — adds indirection for a single concrete client type.
- **Abstract base class**: Rejected — Protocol is more Pythonic but unnecessary when there's one implementation.

## Decision 3: State Model Design

**Decision**: `WorkflowState` uses an append-only log of `StateEntry` records (node_id, key, value, timestamp) with two read patterns: `get(key)` returns latest, `get_all(key)` returns accumulated list. Shared pools (`PoolState`) are separate from the state log, with typed items, dedup, max capacity, and async locks.

**Rationale**: Append-only state provides full audit trail, supports both replace and accumulate semantics, and is trivially serializable for checkpointing. Pools need dedup, search, capacity limits — different from simple state entries.

**No change from original design.**

## Decision 4: Configuration Strategy — YAML First-Class

**Decision**: YAML workflow definitions loaded via `WorkflowDefinition.from_yaml(path)`. Framework provides `WorkflowNode`, `NodeType`, and type-specific config models. YAML is a first-class framework feature for standalone users.

The Deep Research app uses `config_translator.py` to build `WorkflowDefinition` programmatically from `OrchestrationConfig` (because the app's config has many dynamic fields). This is an app-level choice, not a framework limitation.

**Rationale**: YAML is human-readable, diffable, and version-controllable. The framework's internal data model (`WorkflowDefinition`) can be constructed from YAML or programmatically — both are equally valid.

## Decision 5: Event Bridge (Framework → App Streaming)

**Decision**: The framework executor yields `StreamEvent` instances (Pydantic `BaseModel` with `event_type: Literal[...]` discriminator) via async generator. The app's `event_mapper.py` translates these to SSE events for the frontend.

**Rationale**: Async generator is the natural Python pattern for streaming results. Pydantic BaseModel with literal discriminator enables type-safe serialization and deserialization. The event_mapper is acknowledged as the highest-risk component (~500-700 LOC of semantic translation).

**Change from original**: Events are Pydantic `BaseModel` (not `dataclass(frozen=True)`) with `event_type: Literal[...]` discriminator field on every event.

## Decision 6: Prompt Management

**Decision**: Framework ships production-quality default prompts per subtype in `agents/prompts/`. Prompts are Python string constants. Users can override the main prompt pair via YAML:

```yaml
config:
  subtype: researcher
  system_prompt: "Custom system prompt..."        # Override default
  user_prompt_template: "Custom template..."      # Override default
```

If `system_prompt` / `user_prompt_template` are null, subtype defaults are used. Internal prompts (search query generation, source-aware planner variants) stay internal — not exposed in YAML.

**Rationale**: Prompts evolve with agent logic. Exposing only the main pair keeps YAML config manageable while allowing customization. `SafeTemplateRenderer` prevents injection in user-provided templates.

## Decision 7: Testing Strategy

**Decision**: Two test suites. Framework tests (`databricks-deep-research/tests/`) are self-contained with mock LLM. App tests (`tests/unit/agent/adapters/`) cover the integration layer (event mapping, config translation, state bridge). Existing unit/integration/complex tiers preserved for app.

**No change from original design.**

## Decision 8: Migration Strategy — Full Replacement

**Decision**: The app's orchestrator is fully replaced — no feature flag, no dual-path code. The old 3769 LOC `orchestrator.py` is rewritten as a thin wrapper around the framework executor. `config_translator.py` must handle ALL `OrchestrationConfig` modes.

**Rationale**: Project is not released. No need for a rollback path. A feature flag would add maintenance burden and delay the clean cut. Full replacement forces all edge cases to be handled upfront.

**Reclaim mode gap**: Between P0c and P0d, reclaim synthesis is temporarily unavailable. Acceptable since the project is in development.

**Change from original**: The original architecture §14.2 described gradual migration with a nullable `workflow_tree` JSONB column. Removed — no gradual migration.

## Decision 9: Plugin System

**Decision**: The app's existing plugin system stays in the app. When plugins define custom phases, the `config_translator` builds workflow tree nodes that call the existing `PhaseExecutor`. `PhaseExecutor`, `PipelineCustomization`, and all phase implementations are completely unchanged.

**No change from original design.**

## Decision 10: Database Layer

**Decision**: All database code stays in the app. Framework has zero database dependencies. Checkpointing is via a `CheckpointHandler` protocol — framework defines it, app provides the DB implementation.

**No change from original design.**

## Decision 11: Agent Port Strategy — Full Port

**Decision**: All production agent logic moves into framework `agents/builtins/`. This includes all prompts, error handling, edge cases, and tuning. The app's `agent/nodes/` directory is emptied. No "simplified builtins + app overrides" pattern.

**Rationale**: No dual mode. Framework builtins ARE the production agents. For user-defined subtypes without a builtin, the framework uses the generic agent harness with ReAct loop.

## Decision 12: Researcher Mode — ReAct Only

**Decision**: Only ReAct researcher mode is ported. Classic researcher (`nodes/researcher.py`, 727 LOC with fixed search/crawl budget) is dead code — not ported.

**Rationale**: Classic mode is unused in practice. ReAct (LLM-controlled tool calls) is strictly more capable. Dropping classic simplifies the researcher builtin.

## Decision 13: Synthesizer Mode — ReAct + Reclaim Only

**Decision**: Only ReAct and Reclaim synthesis modes. Simple synthesizer (`nodes/synthesizer.py`, 717 LOC) is dead code — not ported.

- **ReAct** (default): LLM controls evidence retrieval during generation. From `react_synthesizer.py`.
- **Reclaim** (P0d): 7-stage citation verification pipeline. From `citation_synthesizer.py` + `services/citation/`.

**Rationale**: Simple mode is unused. ReAct provides evidence-grounded synthesis. Reclaim adds formal verification.

## Decision 14: Coordinator — Framework Builtin

**Decision**: Coordinator is a framework builtin subtype (alongside researcher, planner, reflector, synthesizer, background). It handles query classification, depth recommendation, simple query detection, and follow-up detection.

**Rationale**: Query routing is a general-purpose agentic pattern, not app-specific logic. Every multi-agent system needs an entry point that decides what to do with the input. The specific routing categories (simple/web_search/deep_research) are just configuration.

## Decision 15: Background Investigator — Framework Builtin

**Decision**: The background investigator is ported as the 6th builtin subtype (`background`), not configured as a generic researcher node. It performs query decomposition, data landscape assessment, and enterprise source discovery with a 5-second timeout.

**Rationale**: The background investigator has fundamentally different behavior from a researcher:
- Uses SIMPLE model tier (researcher uses ANALYTICAL)
- Performs query decomposition (generates 2-3 sub-queries)
- Probes enterprise sources in parallel with a hard 5s timeout
- Produces structured `BackgroundOutput` (data_landscape, summary, query_decomposition)
- Feeds source-aware planning via data landscape output

These specialized behaviors warrant a distinct subtype rather than overloading the researcher with conditional logic. The `background` subtype also emits its own domain event (`BackgroundCompletedEvent`), keeping the event system clean.
