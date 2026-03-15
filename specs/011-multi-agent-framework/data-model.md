# Data Model: Multi-Agent Framework Extraction

**Feature**: 011-multi-agent-framework | **Date**: 2026-03-08
**Input**: spec.md + architecture.md (sections 3-12)

> **Revision (2026-03-08 session decisions)**:
> - LLM Client: Removed `LLMClient` protocol. Framework uses `FrameworkLLMClient` wrapping `AsyncOpenAI` directly.
> - Tool Protocol: Updated to `definition` property (single), `validate_arguments()`, `ToolResult.success`, `data` not `metadata`.
> - Agent Subtypes: 6 subtypes (researcher, planner, reflector, synthesizer, coordinator, background). Removed `verifier` as standalone. Classic researcher and simple synthesizer dropped.
> - Events: Pydantic `BaseModel` with `event_type: Literal[...]` discriminator (not dataclass).
> - Verification events added for P0d citation pipeline.
> - Templates, SubworkflowNodeConfig, GateConfig, DataFlowGraph: Deferred beyond P0.
> - CheckpointConfig/CheckpointState: Protocol defined in P0, full auto-checkpoint deferred.

## Entity Relationship Overview

```
WorkflowDefinition
├── root: WorkflowNode (recursive tree)
│   ├── config: AgentNodeConfig | ToolNodeConfig | LoopNodeConfig | ...
│   ├── children: list[WorkflowNode]
│   └── error_handling: ErrorConfig | None
├── pools: list[PoolConfig]
├── token_budget: int
├── required_inputs: list[str]
└── output_keys: list[str]

WorkflowState (runtime)
├── log: list[StateEntry]         # Append-only
├── pools: dict[str, PoolState]   # Shared accumulation
├── token_budget: TokenBudget | None
├── context: ExecutionContext      # Auth, model overrides, etc.
└── is_cancelled: bool

WorkflowExecutor
├── llm: FrameworkLLMClient       # AsyncOpenAI wrapper (not a protocol)
├── resolved_tools: dict[str, ResearchTool]
├── checkpoint_handler: CheckpointHandler | None (protocol)
└── execute(definition, state) → AsyncGenerator[StreamEvent]
```

## Core Entities

### 1. WorkflowDefinition

The top-level container for a complete workflow specification.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| id | str | Yes | -- | Unique workflow identifier |
| name | str | Yes | -- | Human-readable name |
| description | str | No | "" | Purpose description |
| version | int | No | 1 | Schema version for migration |
| root | WorkflowNode | Yes | -- | Root node of the workflow tree |
| pools | list[PoolConfig] | No | [] | Shared pool declarations |
| required_inputs | list[str] | No | ["query"] | Keys that must be provided at execution start |
| output_keys | list[str] | No | ["output"] | Keys produced by the workflow |
| token_budget | int | No | 0 | Max total tokens (0 = unlimited) |
| timeout_seconds | int | No | 1800 | Workflow-level timeout |

**Validation rules**: Unique pool names, root node passes structural validation, required_inputs non-empty.

**Serialization**: YAML <-> Pydantic model via `from_yaml(path)` / `to_yaml(path)`. YAML is a first-class framework feature.

### 2. WorkflowNode

A single node in the workflow tree. Recursive structure.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| id | str | Yes | -- | Unique within the workflow tree |
| type | NodeType | Yes | -- | One of 8 types (see enum below) |
| label | str | Yes | -- | Human-readable display name |
| config | dict[str, Any] | No | {} | Type-specific configuration (validated per type) |
| children | list[WorkflowNode] | No | [] | Child nodes (empty for leaves) |
| error_handling | ErrorConfig | None | No | None | Per-node error handling override |

**NodeType enum**: `agent`, `tool`, `sequence`, `parallel`, `loop`, `conditional`, `subworkflow`, `plan_and_execute`

**P0 scope**: `subworkflow` type is defined but implementation deferred beyond P0.

**Structural validation rules**:
- `agent`: children must be empty (leaf)
- `tool`: children must be empty (leaf)
- `sequence`: >= 1 child
- `parallel`: >= 2 children, non-overlapping output_keys
- `loop`: exactly 1 child
- `conditional`: >= 1 child (conditions list determines branching)
- `subworkflow`: children must be empty (leaf-like, delegates to resolved definition)
- `plan_and_execute`: children must be empty; config must have `planner`, `body` inline configs; `evaluator` is optional

### 3. Type-Specific Node Configs

#### AgentNodeConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| subtype | str | None | Standard subtype (researcher, synthesizer, planner, reflector, coordinator) |
| model_tier | str | "analytical" | LLM tier: simple, analytical, complex |
| system_prompt | str | None | Override system prompt (uses subtype default if None) |
| user_prompt_template | str | None | Override user prompt template (uses subtype default if None) |
| input_keys | list[str] | ["query"] | State keys to read as context |
| output_key | str | "output" | State key to write result |
| output_mode | str | "replace" | "replace" (latest wins) or "append" (accumulate) |
| output_format | str | "text" | "text", "json", "markdown" |
| output_schema | dict | None | JSON schema for structured output |
| tools | list[ToolRef] | [] | Tool references for this agent |
| pool_writes | list[PoolWriteConfig] | [] | Pools to write results into |
| pool_tools | list[str] | [] | Pool names to generate search tools for |
| max_tool_calls | int | 10 | Max tool calls per execution |
| max_retries | int | 2 | Max retries on parse failure |
| conversation_budget | int | 0 | Max conversation tokens before compaction |
| pool_inject | list[PoolInjectConfig] | [] | Pools to inject directly into prompt when small (see §5.3) |
| output_model | type | None | Typed Pydantic output model class (set by subtype defaults) |

**Prompt customization**: When `system_prompt` or `user_prompt_template` is null, the subtype's built-in default prompt is used. Only the main prompt pair (system + user template) is configurable via YAML. Internal prompts (search query generation, source-aware planner variants) stay internal.

#### ToolNodeConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| ref | ToolRef | -- | Tool reference (type + name) |
| input_mapping | dict[str, str] | {} | Parameter name -> state key mapping |
| output_key | str | "output" | State key for tool result |

#### LoopNodeConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| until | Condition | -- | Exit condition (evaluated after each iteration) |
| min_iterations | int | 1 | Minimum iterations before checking condition |
| max_iterations | int | 10 | Safety limit |

#### ConditionalNodeConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| conditions | list[ConditionBranch] | -- | Ordered condition -> child_index mapping |
| default_branch | int | None | Index of default child if no condition matches |

#### SubworkflowNodeConfig (deferred beyond P0)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| ref | str | -- | Workflow ID or template name |
| params | dict[str, Any] | {} | Template parameters |
| input_mapping | dict[str, str] | {} | Parent state key -> child state key |
| output_mapping | dict[str, str] | {} | Child state key -> parent state key |
| output_key | str | "output" | Fallback output key if no output_mapping |
| pool_mode | str | "shared" | "shared" or "isolated" |

#### PlanAndExecuteNodeConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| planner | AgentNodeConfig | -- | Produces structured output with items list |
| items_path | str | "steps" | Dot-path to items in planner output |
| item_state_key | str | "current_item" | State key for current item |
| body | WorkflowNode | -- | Processes each item |
| evaluator | AgentNodeConfig or None | None | Optional: continue/replan/complete |
| max_iterations | int | 10 | Total body executions across all cycles |
| min_iterations | int | 1 | Before "complete" is honored |
| max_replan_cycles | int | 3 | Max planner re-invocations |
| complete_on_exhaustion | bool | True | Exit when all items done (vs. re-plan) |

#### PoolInjectConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| pool | str | -- | Pool name |
| threshold | int | 10 | Inject into prompt if pool.count() <= this |
| format | str | "numbered" | Injection format: "numbered", "bullet", "json" |

### 4. WorkflowState

Runtime state flowing through the workflow tree.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| query | str | -- | Original user query |
| log | list[StateEntry] | [] | Append-only state log |
| pools | dict[str, PoolState] | {} | Shared research pools (name -> pool) |
| token_budget | TokenBudget | None | Token usage tracker |
| model_overrides | dict[str, str] | {} | Model tier -> endpoint overrides |
| enterprise_tools | list[ResearchTool] | [] | Enterprise tool instances |
| user_token | str | None | OBO token for enterprise tools |
| domain_filter | str | None | Domain filter for search |
| is_cancelled | bool | False | Cancellation flag |
| _latest_index | dict[str, int] | {} | Internal index for O(1) get() lookups |

**Methods**:
- `append(node_id, key, value)` -> adds StateEntry to log, updates `_latest_index` for O(1) lookup
- `get(key)` -> returns latest value for key via `_latest_index` (O(1) instead of O(n))
- `get_all(key)` -> returns all values for key (accumulated)
- `to_dict()` / `from_dict()` -> serialization for checkpointing

### 5. StateEntry

Immutable record of a single state write.

| Field | Type | Description |
|-------|------|-------------|
| node_id | str | ID of the node that wrote this entry |
| key | str | State key |
| value | Any | The value written |
| timestamp | str | ISO 8601 timestamp |

### 6. PoolConfig

Declaration of a shared pool in a workflow definition.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| name | str | -- | Pool name (unique within workflow) |
| item_type | str | "text" | Type of items: text, source, claim, evidence |
| dedup_key | str | None | Field name for key-based dedup |
| dedup_content_hash | bool | True | Enable content-hash dedup |
| max_items | int | 0 | Max capacity (0 = unlimited) |

### 7. PoolState

Runtime pool -- multi-producer accumulation with typed items.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| config | PoolConfig | -- | Pool configuration |
| items | list[Any] | [] | Accumulated items |
| seen_keys | set[str] | set() | Keys seen for dedup |
| seen_hashes | set[str] | set() | Content hashes for dedup |
| lock | asyncio.Lock | -- | Async lock for concurrent writes |

**Methods**:
- `add(item)` -> add with dedup check, evict oldest if at capacity
- `extend_async(items)` -> bulk add with lock
- `search(query, limit)` -> search items by relevance (BM25 optional, fallback to keyword)
- `get_recent(n)` -> last N items
- `count()` -> total items
- `topics()` -> unique topic labels
- `get_by_index(i)` -> item at index

### 8. Condition System

#### StateCondition

| Field | Type | Description |
|-------|------|-------------|
| key | str | Dot-path state key to evaluate |
| operator | str | eq, neq, gt, lt, gte, lte, contains, in, exists, not_exists |
| value | Any | Comparison value |

#### LLMCondition

| Field | Type | Description |
|-------|------|-------------|
| prompt_template | str | Template evaluated with state context |
| model_tier | str | LLM tier for evaluation |
| expected_output | str | Expected response for "true" |

#### CompositeCondition

| Field | Type | Description |
|-------|------|-------------|
| operator | str | all, any, not |
| conditions | list[Condition] | Sub-conditions |

#### ConditionBranch

| Field | Type | Description |
|-------|------|-------------|
| condition | Condition | Condition to evaluate |
| child_index | int | Index of child to execute if condition is true |

**Condition** = `StateCondition | LLMCondition | CompositeCondition` (discriminated union)

### 9. Tool System

#### ToolRef

| Field | Type | Description |
|-------|------|-------------|
| type | str | "builtin", "uc_function", "uc_tool", "enterprise" |
| name | str | Tool identifier |

#### ResearchTool (Protocol)

| Method | Signature | Description |
|--------|-----------|-------------|
| definition | property -> ToolDefinition | Single property combining name, description, parameters schema |
| validate_arguments | (args) -> dict | Validate and transform arguments before execution |
| execute | async (args, context) -> ToolResult | Execute the tool |

#### ToolDefinition

| Field | Type | Description |
|-------|------|-------------|
| name | str | Tool name |
| description | str | Tool description for LLM function calling |
| parameters | dict[str, Any] | JSON Schema of parameters |
| source_type | str | "builtin", "uc_function", "uc_tool", "enterprise" |

#### ToolResult

| Field | Type | Description |
|-------|------|-------------|
| content | str | Tool output content |
| success | bool | Whether execution succeeded |
| sources | list[SourceInfo] | Source references |
| data | dict[str, Any] | Additional structured data |
| error | str | None | Error message on failure |

#### SourceInfo

| Field | Type | Description |
|-------|------|-------------|
| url | str | Source URL |
| title | str | Source title |
| snippet | str | Relevant snippet |
| source_type | str | "web", "enterprise", "file", etc. |

#### UrlRegistry

| Method | Signature | Description |
|--------|-----------|-------------|
| register | (url: str) -> int | Register a URL, return its integer index |
| resolve | (index: int) -> str or None | Resolve index back to URL |
| get_all | () -> list[tuple[int, str]] | Return all (index, url) pairs |

Created per workflow execution, shared across all tool calls. Added to `ToolContext` as `url_registry: UrlRegistry | None`.

#### ToolContext

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| query | str | "" | Current query (changes per tool call) |
| url_registry | UrlRegistry or None | None | URL index → URL mapping (security) |

**Constructor DI pattern**: Tool dependencies (search clients, domain filters, user tokens) are injected at tool creation time, not passed per-call. Only per-call values belong in ToolContext.

#### PoolWriteConfig

| Field | Type | Description |
|-------|------|-------------|
| pool | str | Target pool name |
| extract | str | None | Dot-path to extract from output (None = whole output) |
| transform | str | None | Transform function name |

### 10. Streaming Events

Pydantic `BaseModel` with `event_type: Literal[...]` discriminator on every event. All events carry `node_id` and `timestamp`.

| Event Type | event_type literal | Key Fields | Emitted By |
|------------|-------------------|------------|------------|
| NodeStartedEvent | "node_started" | node_type, label | All nodes |
| NodeCompletedEvent | "node_completed" | duration_ms | All nodes |
| NodeErrorEvent | "node_error" | error_message, will_retry | All nodes |
| NodeSkippedEvent | "node_skipped" | reason | On error_handling=skip |
| LoopIterationEvent | "loop_iteration" | iteration, max_iterations | Loop |
| LoopExitEvent | "loop_exit" | reason, total_iterations | Loop |
| BranchSelectedEvent | "branch_selected" | branch_index, condition_summary | Conditional |
| ToolCallEvent | "tool_call" | tool_name, arguments | Agent/Tool |
| ToolResultEvent | "tool_result" | tool_name, result_summary | Agent/Tool |
| ToolCacheHitEvent | "tool_cache_hit" | tool_name, cache_key | Agent/Tool |
| AgentOutputEvent | "agent_output" | output_key, output_preview | Agent |
| AgentStreamChunkEvent | "agent_stream_chunk" | chunk, subtype | Agent (streaming) |
| CheckpointSavedEvent | "checkpoint_saved" | checkpoint_size | Executor |
| CheckpointResumedEvent | "checkpoint_resumed" | resumed_from | Executor |
| TokenUsageEvent | "token_usage" | prompt_tokens, completion_tokens | Executor |
| TokenBudgetExceededEvent | "token_budget_exceeded" | used, limit | Executor |
| ConversationCompactedEvent | "conversation_compacted" | tokens_saved | Executor |
| WorkflowStartedEvent | "workflow_started" | workflow_id, workflow_name | Executor |
| WorkflowCompletedEvent | "workflow_completed" | workflow_id, duration_ms, final_report, structured_output, total_sources, total_steps_executed | Executor |

**Domain events (builtin subtypes)**:

| Event Type | event_type literal | Key Fields | Emitted By |
|------------|-------------------|------------|------------|
| PlanCreatedEvent | "plan_created" | plan_id, title, thought, steps, iteration, has_enough_context | planner builtin |
| ReflectionDecisionEvent | "reflection_decision" | decision, reasoning, suggested_changes | reflector builtin |
| ItemsExtractedEvent | "items_extracted" | total_items, items_path, cycle | plan_and_execute |
| ItemStartedEvent | "item_started" | item_index, item_summary, total_items | plan_and_execute |
| ItemCompletedEvent | "item_completed" | item_index, items_processed | plan_and_execute |
| EvaluationDecisionEvent | "evaluation_decision" | decision, reasoning, items_processed | plan_and_execute |
| ReplanTriggeredEvent | "replan_triggered" | cycle, reason, items_remaining | plan_and_execute |
| PlanAndExecuteExitEvent | "plan_and_execute_exit" | reason, total_items_processed, replan_cycles | plan_and_execute |
| CoordinatorClassifiedEvent | "coordinator_classified" | complexity, recommended_depth, is_simple, direct_response, follow_up_type, reasoning | coordinator builtin |
| BackgroundCompletedEvent | "background_completed" | sources_discovered, data_landscape_summary, data_landscape, query_decomposition | background builtin |
| SynthesisStartedEvent | "synthesis_started" | total_observations, total_sources | synthesizer builtin |

**Typed output models** (per-subtype Pydantic contracts for AgentOutput.content):

| Model | Subtype | Key Fields |
|-------|---------|------------|
| PlanOutput | planner | title, thought, steps, has_enough_context, iteration |
| ReflectionOutput | reflector | decision (continue/adjust/complete), reasoning, suggested_changes |
| EvaluationOutput | evaluator (plan_and_execute) | decision (continue/replan/complete), reasoning, suggested_changes |
| CoordinatorOutput | coordinator | complexity, is_simple, recommended_depth, direct_response |
| ResearcherOutput | researcher | findings, sources_found |
| SynthesizerOutput | synthesizer | report, structured_output |
| BackgroundOutput | background | data_landscape, summary, query_decomposition |

**Verification events (P0d)**:

| Event Type | event_type literal | Key Fields |
|------------|-------------------|------------|
| ClaimGeneratedEvent | "claim_generated" | claim_text, claim_index, citation_keys |
| ClaimVerifiedEvent | "claim_verified" | claim_index, verdict, confidence |
| CitationCorrectedEvent | "citation_corrected" | claim_index, action |
| NumericClaimDetectedEvent | "numeric_claim_detected" | claim_index, numeric_value |
| VerificationSummaryEvent | "verification_summary" | total_claims, verified_claims, overall_confidence |

**Deferred events** (beyond P0): GateWaitingEvent, GateResumedEvent, GateTimeoutEvent.

### 11. Execution Context

Passed to the executor at startup. Contains auth and global config.

| Field | Type | Description |
|-------|------|-------------|
| llm_client | FrameworkLLMClient | AsyncOpenAI wrapper with model tier mapping |
| checkpoint_handler | CheckpointHandler | None | Persistence handler (protocol) |
| model_overrides | dict[str, str] | {} | Model tier overrides |
| user_token | str | None | OBO token |
| enterprise_tools | list[ResearchTool] | [] | Pre-resolved enterprise tools |
| trace_enabled | bool | True | MLflow tracing |

### 12. Support Entities

#### ErrorConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| on_error | str | "fail" | "fail", "skip", "retry" |
| max_retries | int | 2 | Retries before fallback |
| retry_delay_seconds | float | 1.0 | Base delay (exponential backoff) |

#### TokenBudget

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| max_total_tokens | int | -- | Budget ceiling (0 = unlimited) |
| total_tokens | int | 0 | Running total |
| total_prompt_tokens | int | 0 | Running prompt total |
| total_completion_tokens | int | 0 | Running completion total |
| node_usage | dict[str, NodeTokenUsage] | {} | Per-node breakdown |

### 13. Agent Subtypes (Builtin Defaults)

| Subtype | model_tier | output_key | output_format | tools | pool_writes | pool_tools | output_model | Notes |
|---------|-----------|------------|---------------|-------|-------------|------------|--------------|-------|
| coordinator | simple | coordination | json | [] | [] | [] | CoordinatorOutput | Query classification, depth recommendation, follow-up detection |
| researcher | analytical | findings | text | [web_search, web_crawl] | [observations, sources] | [observations, sources] | ResearcherOutput | ReAct mode only (classic dropped) |
| planner | analytical | plan | json | [] | [observations] | [] | PlanOutput | Structured plan generation |
| reflector | simple | reflection | json | [] | [] | [observations] | ReflectionOutput | CONTINUE/ADJUST/COMPLETE decision |
| synthesizer | complex | report | text | [] | [claims] | [observations, sources] | SynthesizerOutput | ReAct (default) + Reclaim (P0d) modes. Simple mode dropped. |
| background | simple | background | json | [web_search] | [sources] | [] | BackgroundOutput | Query decomposition, data landscape, enterprise discovery (5s timeout) |

**Prompt customization**: Each subtype ships with production-quality default prompts. Users may override the main pair via YAML:
```yaml
config:
  subtype: researcher
  system_prompt: "Custom system prompt..."
  user_prompt_template: "Custom user template with {query}..."
```
Internal prompts (search query generation, source-aware planner variants, etc.) are not exposed in YAML.

### 14. LLM Client

The framework uses `FrameworkLLMClient` which wraps `openai.AsyncOpenAI` directly. **No Protocol abstraction.**

| Field/Method | Type | Description |
|--------------|------|-------------|
| openai_client | property -> AsyncOpenAI | Underlying OpenAI client |
| resolve_model(tier) | (str) -> str | Map ModelTier to concrete model name |
| complete(messages, tier, ...) | async -> LLMResponse | Chat completion |
| stream(messages, tier, ...) | async generator -> str or ToolCall | Streaming completion |
| embed(texts, model) | async -> list[list[float]] | Batch embed via OpenAI embeddings.create() |
| embed_single(text) | async -> list[float] | Convenience for single text |
| supports_embeddings | property -> bool | Whether embedding model is configured |

**ModelTier enum**: `simple`, `analytical`, `complex`

**LLMResponse fields**: `content`, `tool_calls`, `usage`, `model`, `finish_reason`, `structured` (parsed Pydantic model or dict for structured output).

**Rate limiting**: `FrameworkLLMClient` accepts `dict[str, str | ModelTierConfig]` for `model_mapping`. When a `ModelTierConfig` is provided, the client performs endpoint health tracking, TPM budget enforcement, 429 fallback, and exponential backoff with jitter.

#### EndpointHealth

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| is_healthy | bool | True | Whether endpoint is accepting requests |
| consecutive_errors | int | 0 | Error counter (reset on success) |
| rate_limited_until | float | 0.0 | Timestamp when rate limit expires |
| tokens_used_this_minute | int | 0 | TPM tracking counter |
| minute_started_at | float | 0.0 | Start of current minute window |

**Methods**: `mark_success()`, `mark_failure(rate_limited)`, `can_handle_request(estimated_tokens, tpm_limit)`

#### ModelTierConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| endpoints | list[str] | -- | Priority-ordered endpoint names |
| fallback_on_429 | bool | True | Try next endpoint on rate limit |
| rotation_strategy | str | "PRIORITY" | "PRIORITY" or "ROUND_ROBIN" |
| tokens_per_minute | int | 0 | TPM limit (0 = unlimited) |

**Rationale**: Databricks standardizes on OpenAI-compatible endpoints. A Protocol would add complexity without practical value since no non-OpenAI backends are planned. Rate limiting is ported into the framework so standalone users get production-quality resilience. OAuth token refresh remains in the app's `llm_adapter.py`.

**Embedding support**: `FrameworkLLMClient` accepts an optional `embedding_model: str | None` at construction. When configured, `embed()` and `embed_single()` delegate to `openai.embeddings.create()`. Used by `PoolRegistry` for hybrid BM25+vector search. Graceful degradation when no embedding model is configured.

### 15. Protocols (Framework-Defined)

| Protocol | Methods | Implementor |
|----------|---------|-------------|
| CheckpointHandler | save(execution_id, workflow_id, state), load(execution_id, workflow_id) | App provides (DB-backed). Protocol in P0, full auto-checkpoint deferred. |
| ResearchTool | definition, validate_arguments(), execute() | Framework builtins + app custom tools |

**Removed**: `LLMClient` protocol -- replaced by concrete `FrameworkLLMClient` class.

### Deferred Beyond P0

The following entities are **defined in the data model** but implementation is deferred:

| Entity | Reason |
|--------|--------|
| SubworkflowNodeConfig | Composable subworkflows are P2 |
| GateConfig | HITL gates are P2 |
| Parameterized Templates (BestOfN, SelfCritique, Debate, MajorityVote) | Template registry is P2 |
| DataFlowGraph | Static analysis is P2 |
| CheckpointConfig / CheckpointState | Protocol defined in P0; automatic checkpoint granularity deferred |
| VerificationConfig | Verification is integrated into synthesizer config, not a separate entity |
