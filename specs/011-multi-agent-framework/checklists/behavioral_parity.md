# Behavioral Parity Checklist: orchestrator.py → Framework Migration

**Purpose**: Exhaustive checklist of `orchestrator.py` behaviors that must be preserved during P0c migration. Each item references source line ranges and the target component responsible.

**Source**: `src/deep_research/agent/orchestrator.py` (~3769 LOC)
**Target**: Framework executor + app adapters (config_translator, domain_context, llm_adapter, tool_adapter)

---

## 1. Configuration & Initialization (~20 items)

- [X] `max_plan_iterations` limits ADJUST replan cycles → handled by: `PlanAndExecuteNodeConfig.max_replan_cycles`
- [X] `enable_background` toggles background investigation phase → handled by: `config_translator` (conditional node)
- [X] `enable_clarification` toggles coordinator clarification rounds → handled by: `config_translator` (conditional node)
- [X] `timeout` enforces overall research timeout → handled by: `WorkflowDefinition.timeout_seconds`
- [X] `query_mode` routes to simple/web_search/deep_research/custom → handled by: `config_translator` (top-level conditional)
- [X] `research_depth` controls step count min/max → handled by: `PlanAndExecuteNodeConfig.min_iterations/max_iterations`
- [X] `system_instructions` injected into all agent prompts → handled by: `config_translator` (system_prompt override per agent)
- [X] `source_scope` (SourceScopeConfig) wired after state creation → handled by: `config_translator` + `ToolContext`
- [X] `enabled_sources` / `disabled_sources` filter enterprise tools → handled by: `tool_adapter` filtering
- [X] `file_ids` loads file content for research → handled by: `config_translator` (file search tool creation)
- [X] `agent_id` loads custom agent config from DB → handled by: app-level, before `config_translator`
- [X] `model_overrides` maps tiers to specific endpoints → handled by: `ExecutionContext.model_overrides`
- [X] `domain_filter` restricts search scope → handled by: constructor DI on tool instances (not ToolContext)
- [X] `workflow_mode` selects pipeline variant → handled by: `config_translator`
- [X] `manual_steps` converted to plan → handled by: `config_translator` (preset_steps_to_tree)
- [X] `synthesis_mode` selects react/reclaim → handled by: `config_translator` (synthesizer config)
- [X] `enable_post_verification` triggers citation pipeline → handled by: synthesizer `verification` config
- [X] `output_format` / `output_schema` for structured output → handled by: `AgentNodeConfig.output_format/output_schema`
- [x] `enable_plan_review` / `require_plan_approval` / `plan_review_timeout` → **REMOVED — not ported. Plan review gate deferred indefinitely. Field is a no-op (logs deprecation warning).**
- [X] State creation via `_create_initial_state()` helper → handled by: `WorkflowState` construction in executor
- [X] Source scope wiring (`state.source_scope_config = ...`) → handled by: `config_translator` + `ToolContext`
- [X] Custom agent application (merge priority, preset steps, model overrides, domain filter) → handled by: app-level merge before `config_translator`

## 2. Query Handling & Routing (~10 items)

- [X] Simple mode short-circuit: coordinator → direct response, no research → handled by: `config_translator` (conditional: is_simple → agent-only)
- [X] Web search mode: 1-step plan, timeout+retry, fallback to simple → handled by: `web_search.yaml` workflow
- [X] Deep research mode: full pipeline (background → plan → plan_and_execute → synthesize) → handled by: `deep_research.yaml` workflow
- [X] Custom phase mode: plugin-driven pipeline → handled by: `config_translator` (PhaseExecutor integration)
- [X] Coordinator clarification rounds (multi-turn before routing) → handled by: conditional loop in workflow tree
- [X] Follow-up detection (coordinator identifies follow-up queries) → handled by: coordinator builtin
- [X] Direct response for simple queries (no research needed) → handled by: coordinator output + conditional
- [X] Web search timeout with graceful fallback → handled by: error_handling config on web search node
- [X] Query context extraction (LLM + regex fallback) for plugins → handled by: stays in app (plugin system)
- [X] Depth recommendation from coordinator feeds planner config → handled by: state flow (coordinator output_key → planner input_key)

## 3. Planning (~8 items)

- [X] Plan creation with depth-aware step generation (min/max from research_depth) → handled by: planner builtin + `PlanAndExecuteNodeConfig`
- [x] Plan review/approval with timeout → **REMOVED — not ported. Plan review gate deferred indefinitely.**
- [X] Early completion (`has_enough_context` = true → skip research) → handled by: planner builtin + conditional node
- [X] Plan adjustment (ADJUST preserves completed steps, adds new) → handled by: `plan_and_execute` node (re-runs planner with context)
- [X] Source-aware planning (data_landscape from background feeds planner) → handled by: planner builtin reads `background` state key
- [X] Plan title and thought extraction for UI display → handled by: `PlanCreatedEvent` domain event
- [X] Step-level plan detail (title, description, search hints per step) → handled by: `PlanOutput` typed model
- [X] Iteration tracking (which replan cycle we're on) → handled by: `PlanAndExecuteNodeConfig` + `ReplanTriggeredEvent`

## 4. Research Loop (~10 items)

- [X] Step iteration: while `has_more_steps` in plan → handled by: `plan_and_execute` node iterates plan steps
- [X] Researcher mode selection (ReAct only per D6) → handled by: researcher builtin (always ReAct)
- [X] ReactResearchEvent → StreamEvent conversion → handled by: researcher builtin emits native `StreamEvent`s
- [X] Reflection after each step (CONTINUE/ADJUST/COMPLETE) → handled by: `plan_and_execute` reflector phase
- [X] CONTINUE → next step → handled by: `plan_and_execute` execution semantics
- [X] ADJUST → break step loop, re-run planner → handled by: `plan_and_execute` + `ReplanTriggeredEvent`
- [X] COMPLETE → break step loop, proceed to synthesis → handled by: `plan_and_execute` exit
- [X] Min steps enforcement (prevent premature COMPLETE) → handled by: `PlanAndExecuteNodeConfig.min_iterations`
- [X] Max plan iterations (prevent infinite ADJUST loops) → handled by: `PlanAndExecuteNodeConfig.max_replan_cycles`
- [X] Step exhaustion (all steps done without COMPLETE) → handled by: `plan_and_execute` natural exit

## 5. Source Management (~12 items)

- [X] File content loading — inline strategy (small files injected into prompt) → handled by: `config_translator` (input_keys for file content)
- [X] File content loading — hybrid strategy (summary + retrieval tool) → handled by: `config_translator` (file_search tool + summary)
- [X] File content loading — retrieval strategy (tool-only for large files) → handled by: `config_translator` (file_search tool)
- [X] File search tool creation (explicit file IDs) → handled by: `tool_adapter` wraps file_search builtin
- [X] File auto-discovery (no explicit IDs, search available files) → handled by: stays in app
- [X] 3-tier enterprise tool loading (DB → cache → direct source IDs) → handled by: stays in app (`factory.py`)
- [X] Tool deduplication (no duplicate tool names) → handled by: `tools/registry.py` dedup
- [X] Source scope enforcement — web search blocked when scope excludes web → handled by: `ToolContext` + tool filtering
- [X] Source scope enforcement — enterprise blocked when scope excludes enterprise → handled by: `ToolContext` + tool filtering
- [X] Follow-up source loading from chat history → handled by: stays in app (session-aware)
- [X] Enterprise tool parallel execution by source type → handled by: researcher builtin (ReAct handles tool calls)
- [X] Source tracking (`enterprise://{tool_name}` URL scheme) → handled by: `SourceInfo.source_type` + tool metadata

## 6. Streaming Events (~25 items)

- [X] `lifecycle:start` event at research begin → handled by: `WorkflowStartedEvent`
- [X] `lifecycle:complete` event at research end → handled by: `WorkflowCompletedEvent`
- [X] `agent:coordinator` event with classification result → handled by: `CoordinatorClassifiedEvent`
- [X] `planning:start` / `planning:complete` events → handled by: `NodeStartedEvent` / `NodeCompletedEvent` on planner node
- [X] `planning:plan_created` with title, steps → handled by: `PlanCreatedEvent`
- [X] `step:start` with step index and title → handled by: `ItemStartedEvent`
- [X] `step:complete` with observation summary → handled by: `ItemCompletedEvent`
- [X] `reflection:decision` with CONTINUE/ADJUST/COMPLETE → handled by: `ReflectionDecisionEvent`
- [X] `tool:search_start` / `tool:search_result` → handled by: `ToolCallEvent` / `ToolResultEvent`
- [X] `tool:crawl_start` / `tool:crawl_result` → handled by: `ToolCallEvent` / `ToolResultEvent`
- [X] `synthesis:start` event → handled by: `SynthesisStartedEvent`
- [X] `synthesis:chunk` for streaming tokens → handled by: `AgentStreamChunkEvent`
- [X] `synthesis:complete` event → handled by: `NodeCompletedEvent` on synthesizer node
- [X] `citation:claim_generated` events → handled by: `ClaimGeneratedEvent` (P0d)
- [X] `citation:claim_verified` events → handled by: `ClaimVerifiedEvent` (P0d)
- [X] `citation:corrected` events → handled by: `CitationCorrectedEvent` (P0d)
- [X] `citation:numeric_detected` events → handled by: `NumericClaimDetectedEvent` (P0d)
- [X] `citation:summary` event → handled by: `VerificationSummaryEvent` (P0d)
- [X] `custom_phase:start` / `custom_phase:complete` events → handled by: stays in app (plugin events)
- [X] `error` event on exception → handled by: `NodeErrorEvent`
- [X] Event buffering (accumulate before flush) → handled by: domain_context tracker
- [X] Event flushing on each yield → handled by: async generator yield
- [X] `background:complete` with sources discovered → handled by: `BackgroundCompletedEvent`
- [X] `replan` event on ADJUST decision → handled by: `ReplanTriggeredEvent`
- [X] Persistence-related events (session update triggers) → handled by: domain_context tracker `should_persist()`

## 7. Persistence (~10 items)

- [X] Two-phase model: session start (IN_PROGRESS) → session complete (COMPLETED) → handled by: domain_context tracker
- [X] Simple mode persistence — pre-created message path → handled by: domain_context tracker
- [X] Simple mode persistence — new message path → handled by: domain_context tracker
- [X] Web search persistence (intermediate state) → handled by: domain_context tracker
- [X] `asyncio.shield` for client disconnection resilience → handled by: orchestrator wrapper (stays in app)
- [X] Session FAILED marking on error → handled by: domain_context tracker error handling
- [X] Incremental state persistence (plan, steps, sources) → handled by: domain_context `get_persistence_delta()`
- [X] Research state reconstruction for persistence → handled by: domain_context `get_research_state()`
- [X] Message content update with final report → handled by: domain_context tracker
- [X] Session metadata update (duration, token count) → handled by: domain_context tracker

## 8. Synthesis & Output (~8 items)

- [X] JSON/structured output with `output_format` and `output_schema` → handled by: synthesizer builtin + `AgentNodeConfig`
- [X] Structured synthesis error fallback (retry without schema) → handled by: synthesizer builtin (StructuredSynthesisError handling)
- [X] Citation verification mode (6-stage pipeline events) → handled by: synthesizer builtin reclaim mode (P0d)
- [X] Plain synthesis streaming (token-by-token) → handled by: `AgentStreamChunkEvent`
- [X] Plugin lifecycle hooks — `synthesis_config` → handled by: stays in app (plugin system)
- [X] Plugin lifecycle hooks — `synthesis_started` → handled by: stays in app (plugin system)
- [X] Plugin lifecycle hooks — `validation_error` → handled by: stays in app (plugin system)
- [X] ReAct synthesis (LLM controls evidence retrieval during generation) → handled by: synthesizer builtin

## 9. Error Handling (~5 items)

- [X] Cancellation via `state.is_cancelled` check → handled by: `WorkflowState.is_cancelled` + executor checks
- [X] Exception → `StreamErrorEvent` (mapped to SSE error) → handled by: `NodeErrorEvent` → domain_context → SSE
- [X] `CancelledError` not re-raised (graceful degradation) → handled by: executor cancellation handling
- [X] Session FAILED marking on unhandled exception → handled by: domain_context tracker
- [X] Timeout enforcement (per-workflow and per-step) → handled by: executor timeout + `WorkflowDefinition.timeout_seconds`

## 10. Plugin Integration (~6 items)

- [X] Custom phase routing (query_mode = "custom") → handled by: `config_translator` builds PhaseExecutor nodes
- [X] Query context extraction (LLM + regex fallback) → handled by: stays in app (plugin system)
- [X] PhaseExecutor integration (custom phases execute within workflow) → handled by: `config_translator` wraps PhaseExecutor as tool/agent node
- [X] Plugin lifecycle hooks (multiple hook points) → handled by: stays in app (orchestrator wrapper)
- [X] Plugin-provided synthesis config → handled by: stays in app (plugin system)
- [X] Plugin validation error handling → handled by: stays in app (plugin system)

---

## Migration Progress Tracking

| Category | Total Items | Completed | Notes |
|----------|-------------|-----------|-------|
| Configuration & Initialization | 22 | 22 | All covered by config_translator + executor |
| Query Handling & Routing | 10 | 10 | Workflow YAMLs + config_translator |
| Planning | 8 | 8 | Planner builtin + plan_and_execute node |
| Research Loop | 10 | 10 | plan_and_execute execution semantics |
| Source Management | 12 | 12 | Tool adapters + registry |
| Streaming Events | 25 | 25 | Full event type coverage |
| Persistence | 10 | 10 | domain_context tracker |
| Synthesis & Output | 8 | 8 | Synthesizer builtin + reclaim mode |
| Error Handling | 5 | 5 | Executor + workflow state |
| Plugin Integration | 6 | 6 | Stays in app (orchestrator wrapper) |
| **Total** | **116** | **116** | **All items addressed** |
