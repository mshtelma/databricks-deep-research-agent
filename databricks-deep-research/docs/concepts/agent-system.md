# Agent System

> How agents are configured, executed, and extended.

## Overview

Agents are the primary computation units in the framework. Each agent node in a workflow runs through the **agent harness** (`execute_agent()`), which orchestrates the full lifecycle: build prompt, call LLM, parse output, write state. The harness is the single point of interaction between the workflow executor and an agent node -- no agent touches global state directly.

Every agent receives an immutable `AgentInput` envelope and produces a mutable `AgentOutput` envelope. This isolation boundary ensures that agents are composable, testable, and safe from side effects.

**Source:** `databricks_deep_research/agents/harness.py`

## The Agent Harness (`execute_agent`)

The `execute_agent()` function is the core entry point. It accepts a `node_id`, an `AgentNodeConfig`, the current `WorkflowState`, an `FrameworkLLMClient`, resolved tool instances, and the pool registry. Here is the step-by-step execution flow:

### Step 0a: Builtin Enrichment

The harness looks up the agent's `subtype` in the builtin registry via `get_builtin()`. If a builtin is registered and provides an `enrich_config` hook, the config is enriched before anything else happens. This is how builtin subtypes inject default prompts, output models, and format overrides without requiring explicit YAML configuration.

```
builtin = get_builtin(config.subtype)
if builtin and builtin.enrich_config:
    config = builtin.enrich_config(config, state)
```

### Step 0b: Input Key Auto-Detection

The harness scans both `system_prompt` and `user_prompt_template` using `SafeTemplateRenderer.extract_variables()` to discover all `{variable}` references. Any detected variables that are not already in `config.input_keys` are merged in. This means you rarely need to specify `input_keys` manually -- the harness infers them from your prompt templates.

```
detected_keys = (
    renderer.extract_variables(config.system_prompt)
    | renderer.extract_variables(config.user_prompt_template)
)
merged_keys = sorted(set(existing_keys) | detected_keys)
```

### Step 1: Build AgentInput

The `_build_input()` helper constructs the `AgentInput` envelope:

1. **State reading**: Each key in `input_keys` is resolved from the workflow state. Dot-notation keys (e.g., `plan.steps`) use `state.get_nested()` for deep access; simple keys use `state.get()`.

2. **Serialization**: Resolved values are serialized to strings via `_serialize_for_context()`, which handles Pydantic models (JSON), dicts (JSON), lists (JSON), and primitives (str).

3. **Template rendering**: `SafeTemplateRenderer.render()` fills `{variable}` placeholders in both `system_prompt` and `user_prompt_template`. The variable `query` (from `state.query`) is always available. Context keys only override when they resolve to a non-None value.

4. **Pool injection**: For each entry in `config.pool_inject`, the harness reads recent items from the named pool (up to `max_items`), optionally truncates each item to `max_item_chars`, and attaches the result to `AgentInput.pool_content`.

### Step 2: Build Messages

The `_build_messages()` helper converts the `AgentInput` into OpenAI-format messages:

- The `system_prompt` becomes a `{"role": "system", ...}` message.
- The `user_prompt` (or `query` as fallback) forms the core of the user message.
- Pool content is appended as formatted sections (e.g., `## pool_name\n- item1\n- item2`).
- Any `conversation_history` entries are inserted before the user message.

### Step 3: Execute (Simple or ReAct)

The harness chooses one of three execution paths:

1. **Builtin custom execution**: If the builtin provides an `execute` hook (e.g., the synthesizer in `reclaim` grounding mode), it takes full control and returns an `AgentOutput` directly.

2. **ReAct loop**: If `tools` are provided and `max_tool_calls > 0`, the harness delegates to `ReactLoop.execute()`. This is the path used by researchers and any agent that needs tool calling.

3. **Simple LLM call**: Otherwise, a single `llm_client.complete()` call is made with the constructed messages, using the configured `model_tier`. If `output_model` is set (a Pydantic class), structured output parsing is requested.

A `ToolContext` is built before execution with the original query, the shared URL registry, current step info, background summary, recent observations, and discovered sources from pools.

### Step 4: Parse Output

The `_parse_output()` function handles three output formats:

- **Non-string content** (already a Pydantic model from structured output): returned as-is.
- **`output_format: "json"`**: Attempts `json.loads()`. If that fails, tries extracting JSON from markdown code blocks (` ```json ... ``` `). As a final fallback, uses `json_repair` if available.
- **`output_format: "text"` or `"markdown"`**: Returned as-is.

After parsing, `_enrich_parsed_output()` augments the output for `researcher` and `background` subtypes by attaching sources collected during the ReAct loop, normalizing field names (`findings`, `observation`, `research_status`), and building data landscape summaries.

### Step 5: Write to State

The parsed output is written to the workflow state under the configured `output_key`:

```python
state.append(node_id, config.output_key, state_output)
```

For researchers specifically, structured metadata fields (`research_status`, `blocking_reason`, `search_queries`, `key_points`, `sources_used`) are extracted and written as separate state entries.

### Step 6: Pool Writes

For each `PoolWriteConfig` in `config.pool_writes`, the harness extracts items from the parsed output using the dot-path specified in `extract`. Items are then appended to the target pool. If extraction finds nothing but the ReAct loop collected sources separately (common for `extract: "sources"`), those ReAct-collected sources are used as a fallback.

### Step 7: Emit Output Event

An `AgentOutputEvent` is emitted with the `node_id`, `output_key`, and a preview of the output (first 200 characters).

### Step 8: Builtin Post-Processing

If the builtin provides a `post_process` hook, it is called with the final output. This is where subtypes emit domain-specific events (e.g., `CoordinatorClassifiedEvent`, `PlanCreatedEvent`, `ReflectionDecisionEvent`).

### Return

The harness returns an `AgentOutput` containing the content, output key, pool writes, sources, token usage, and all accumulated events.

## AgentNodeConfig

The `AgentNodeConfig` Pydantic model defines every aspect of an agent node's behavior. All fields have sensible defaults; most agents only need `subtype` and optionally a custom prompt.

**Source:** `databricks_deep_research/agents/config.py`

### Identity

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `subtype` | `str` | (required) | Agent type: `coordinator`, `planner`, `researcher`, `reflector`, `synthesizer`, `evaluator`, or a custom registered subtype. |
| `model_tier` | `str` | `"analytical"` | LLM tier to use: `simple` (fast), `analytical` (balanced), `complex` (reasoning). |

### Prompts

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `system_prompt` | `str` | `""` | System prompt template. Supports `{variable}` placeholders. If empty, the builtin subtype provides a default. |
| `user_prompt_template` | `str` | `""` | User prompt template. Supports `{variable}` placeholders. If empty, the builtin subtype provides a default. |

### Input / Output

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `input_keys` | `list[str]` | `[]` | State keys to read as context. Auto-detected from prompt templates at runtime; explicit values override auto-detection. Dot-notation supported (e.g., `plan.steps`). |
| `output_key` | `str` | `"output"` | State key where the parsed output is written. |
| `output_mode` | `str` | `"text"` | Output handling mode: `text`, `json`, `structured`. |
| `output_format` | `str` | `"text"` | Expected output format for parsing: `text`, `markdown`, `json`. |
| `output_schema` | `dict \| None` | `None` | JSON schema for output validation. |
| `output_model` | `Any` | `None` | Pydantic model class for structured output. When set, the LLM is asked to produce output conforming to this schema, and the response is parsed into an instance of this model. |

### Tools

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tools` | `list[str \| dict]` | `[]` | Tool names or tool config dicts to make available to this agent. |
| `max_tool_calls` | `int` | `0` | Maximum number of tool calls in a ReAct loop. `0` means no tool calling (simple LLM call). |
| `max_result_chars` | `int` | `4000` | Maximum characters per tool result. Old results are truncated during conversation compaction. `0` = unlimited. |

### Pools

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `pool_writes` | `list[PoolWriteConfig]` | `[]` | Rules for writing agent output to shared pools. Each entry specifies a target pool and a dot-path extraction expression. |
| `pool_tools` | `list[str]` | `[]` | Pool-based tools to expose (e.g., `pool_search`). |
| `pool_inject` | `list[PoolInjectConfig]` | `[]` | Rules for injecting pool contents into the agent's prompt. Each entry specifies a source pool, max items, and truncation settings. |

#### PoolWriteConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `pool` | `str` | (required) | Target pool name. |
| `extract` | `str` | (required) | Dot-path expression to extract items from agent output. |
| `transform` | `str \| None` | `None` | Optional transformation template. |

#### PoolInjectConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `pool` | `str` | (required) | Source pool name. |
| `threshold` | `float` | `0.0` | BM25/relevance threshold. |
| `format` | `str` | `"text"` | Formatting mode: `text`, `json`, `markdown`. |
| `max_items` | `int` | `20` | Maximum items to inject into the prompt. |
| `max_item_chars` | `int` | `0` | Maximum characters per item. `0` = unlimited. |

### Grounding

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `grounding_mode` | `"none" \| "classical_lite" \| "reclaim" \| None` | `None` | Citation grounding strategy. `none` = no grounding. `classical_lite` = lightweight post-hoc citation. `reclaim` = full interleaved citation generation with multi-stage verification. |

### Budget and Retries

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `conversation_budget` | `int \| None` | `None` | Max tokens for LLM response (passed as `max_tokens`). |
| `max_retries` | `int` | `2` | Number of retries on failure. |

## ReAct Tool-Calling Loop

The `ReactLoop` class manages iterative LLM-tool interaction for agents that need to call tools (primarily the researcher, but available to any agent with `tools` configured and `max_tool_calls > 0`).

**Source:** `databricks_deep_research/agents/react_loop.py`

### When Triggered

The ReAct loop is activated when the harness detects that `tools` is non-empty and `config.max_tool_calls > 0`. Otherwise, the harness falls through to a simple single LLM call.

### Loop Structure

```
while True:
    1. Compact old tool results (if not first iteration)
    2. Call LLM with messages + tool definitions
    3. Track token usage
    4. If no tool_calls in response OR max_tool_calls reached -> return final content
    5. Append assistant message (with tool_calls) to conversation
    6. Phase 1: Classify each tool_call as cached or uncached
    7. Phase 2: Execute uncached calls in parallel
    8. Phase 3: Reassemble results in original order, append tool messages
    9. Continue loop
```

### Step-by-Step Detail

**Tool Selection**: Before the loop starts, `_apply_step_tool_selection()` partitions tools into *active* (preferred for the current step) and *fallback* (available if active tools return no results). This is driven by the source-aware planning layer which matches step source hints against tool capabilities.

**LLM Call**: Each iteration calls `llm_client.complete()` (or `stream()` on the first iteration if streaming is enabled) with the accumulated messages and current tool definitions.

**Phase 1 -- Parse and Classify**: Each tool call from the LLM response is checked against the `ToolCallCache`. Cache hits emit a `ToolCacheHitEvent` and skip execution. New calls are queued for execution with a `ToolCallEvent`.

**Phase 2 -- Parallel Execution**: All uncached tool calls are executed concurrently via `asyncio.gather()`. Each execution goes through:

1. `plan_tool_arguments()` -- rewrites queries using source-aware strategies (step context, root query, background summary).
2. `tool.validate_arguments()` -- validates and sanitizes arguments.
3. `tool.execute()` -- runs the actual tool against its data source.
4. `admit_tool_result()` -- filters results by relevance, tracking accepted/rejected counts.

**Phase 3 -- Reassemble**: Results are placed back in original tool call order to maintain conversation coherence. Executed results are cached for future deduplication. `ToolResultEvent` is emitted for each completed execution with source counts and success/error metadata.

**Fallback Widening**: If a round of tool calls returns zero accepted sources and fallback tools exist, the loop enables fallback tools and injects a system message explaining that preferred sources yielded no evidence. If the LLM still produces no tool calls after fallback is enabled, a second nudge is injected once.

### Conversation Compaction

To prevent unbounded prompt growth, `_compact_old_tool_results()` truncates tool result messages from prior iterations to `max_result_chars`. Only the most recent iteration's results are preserved intact. This keeps the conversation window manageable across many tool-calling rounds.

### Tool Result Caching

The `ToolCallCache` deduplicates identical tool calls within a research session. It stores both the result content and any source metadata. Cache keys are scoped by the current step (or the root query if no step is active), so the same tool call in different research steps is not incorrectly cached.

### Exit Conditions

The loop terminates when any of these conditions is met:

1. **No tool calls**: The LLM responds without requesting any tool calls, indicating it has enough information.
2. **Max tool calls reached**: The `call_count` reaches `max_tool_calls`.
3. **Budget exceeded**: Implicit via the token tracking -- the LLM naturally stops requesting tools as context fills up.

### Events Emitted

| Event | When | Key Fields |
|-------|------|------------|
| `ToolCallEvent` | Before executing an uncached tool call | `tool_name`, `arguments` |
| `ToolResultEvent` | After a tool execution completes | `tool_name`, `result_summary`, `source_count`, `accepted_source_count`, `rejected_source_count`, `tool_success`, `tool_error` |
| `ToolCacheHitEvent` | When a cached result is reused | `tool_name`, `cache_key` |
| `AgentStreamChunkEvent` | During streaming on the first LLM call | `chunk`, `subtype` |

## Builtin Subtypes

The framework ships with six builtin agent subtypes. Each registers default prompts, an output model, and lifecycle hooks. Subtypes are registered at import time -- importing `databricks_deep_research.agents.builtins` triggers all registrations.

| Subtype | Purpose | Default Tier | Output Key | Output Model | Domain Event |
|---------|---------|-------------|------------|--------------|--------------|
| `coordinator` | Classify query complexity and route to appropriate depth | `simple` | `coordination` | `CoordinatorOutput` | `CoordinatorClassifiedEvent` |
| `background` | Quick context discovery and data landscape assessment | `simple` | `background` | `BackgroundOutput` | `BackgroundCompletedEvent` |
| `planner` | Create structured research plan with ordered steps | `analytical` | `plan` | `PlanOutput` | `PlanCreatedEvent` |
| `researcher` | Execute a research step via ReAct tool-calling loop | `analytical` | `findings` | `ResearcherOutput` | (tool events from ReAct loop) |
| `reflector` | Evaluate progress, decide CONTINUE/ADJUST/COMPLETE | `analytical` | `reflection` | `ReflectionOutput` | `ReflectionDecisionEvent` |
| `synthesizer` | Generate the final research report with optional citation grounding | `complex` | `report` | `SynthesizerOutput` | `SynthesisCompletedEvent` |

There is also an `evaluator` subtype defined in `SUBTYPE_DEFAULTS` (used by the `plan_and_execute` meta-node) with the same default tier and input keys as the reflector, but with an `EvaluationOutput` model that supports `continue`, `replan`, and `complete` decisions.

For detailed prompt templates and behavior, see [Builtin Agents Guide](../guides/builtin-agents.md).

## Subtype Registration

Subtypes are registered via `register_builtin()`, which stores a `BuiltinSubtype` entry in a global registry. Each entry can provide three optional hooks:

```python
register_builtin(
    "coordinator",
    post_process=_post_process,     # Emit domain events after output
    enrich_config=_enrich_config,   # Inject default prompts/models
    execute=None,                   # Custom execution (optional)
    output_model=CoordinatorOutput, # Default Pydantic output model
)
```

**Hook signatures:**

- **`enrich_config(config, state) -> AgentNodeConfig`**: Called before input construction. Fills in default prompts, output models, and format overrides when the YAML config leaves them unset. Must return a (possibly modified) config.

- **`post_process(node_id, output, config, state) -> list[StreamEvent]`**: Called after state writes. Inspects the final output and emits domain-specific events (e.g., `PlanCreatedEvent`). May also write additional state entries (as the background subtype does with `background_summary` and `data_landscape`).

- **`execute(node_id, config, state, llm_client, tools, pools, agent_input, messages, tool_ctx) -> AgentOutput | None`**: Full custom execution that replaces the default simple/ReAct path. If it returns `None`, the harness proceeds with the default execution. Used by the synthesizer for `reclaim` grounding mode.

**Custom subtypes** follow the same pattern. Register your subtype at module load time, and it becomes available to any workflow that references it:

```python
from databricks_deep_research.agents.builtins.registry import register_builtin

def my_enrich(config, state):
    # inject custom prompts, etc.
    return config

def my_post_process(node_id, output, config, state):
    # emit custom events
    return []

register_builtin(
    "my_custom_agent",
    enrich_config=my_enrich,
    post_process=my_post_process,
)
```

Look up registered subtypes with `get_builtin(name)` or list all with `list_builtins()`.

## Output Models

Typed Pydantic models define the structured output for each builtin subtype. When set as `output_model`, the LLM is asked to produce JSON conforming to the model's schema, and the harness parses the response into a validated instance.

**Source:** `databricks_deep_research/agents/output_models.py`

### CoordinatorOutput

Classifies the incoming query and determines routing.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `complexity` | `str` | (required) | Complexity classification (e.g., `"simple"`, `"moderate"`, `"complex"`). |
| `is_simple` | `bool` | `False` | Whether the query can be answered directly without research. |
| `recommended_depth` | `str` | `"standard"` | Suggested research depth (e.g., `"light"`, `"standard"`, `"extended"`). |
| `direct_response` | `str \| None` | `None` | Direct answer for simple queries (skips research pipeline). |
| `follow_up_type` | `str \| None` | `None` | Follow-up classification if applicable. |

### BackgroundOutput

Initial context gathering and data landscape discovery.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `data_landscape` | `dict` | `{}` | Grouped source summary for the planner (source names, types, counts, sample titles). |
| `summary` | `str` | `""` | Natural-language summary of available data. |
| `query_decomposition` | `list[str]` | `[]` | Sub-questions derived from the original query. |
| `discovered_sources` | `list[dict]` | `[]` | Individual source records discovered during background search. |

### PlanOutput

A structured research plan with ordered steps.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `title` | `str` | (required) | Plan title. |
| `thought` | `str` | (required) | Planner's reasoning about the approach. |
| `steps` | `list[PlanStepOutput]` | (required) | Ordered list of research steps. |
| `has_enough_context` | `bool` | `False` | Whether existing context is sufficient (skip research). |
| `iteration` | `int` | `1` | Replan iteration counter. |

Each `PlanStepOutput` contains:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | `str` | (required) | Unique step identifier. |
| `title` | `str` | (required) | Step title. |
| `description` | `str` | `""` | Detailed step description. |
| `step_type` | `"research" \| "analysis"` | `"research"` | Whether this step requires external data or is purely analytical. |
| `needs_search` | `bool` | `True` | Whether the step requires search tool calls. |
| `source_hints` | `list[SourceHintOutput]` | `[]` | Hints for which data sources to prefer and how to query them. |
| `exclude_sources` | `list[str]` | `[]` | Sources to skip for this step. |

### ReflectionOutput

Step-by-step progress evaluation.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `decision` | `"continue" \| "adjust" \| "complete"` | (required) | Whether to continue, adjust the plan, or complete research. |
| `reasoning` | `str` | (required) | Explanation for the decision. |
| `suggested_changes` | `list[str] \| None` | `None` | Suggested plan adjustments (when decision is `"adjust"`). |

### EvaluationOutput

Used by the `plan_and_execute` meta-node evaluator. Similar to `ReflectionOutput` but supports replanning.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `decision` | `"continue" \| "replan" \| "complete"` | (required) | Whether to continue, trigger a replan, or complete. |
| `reasoning` | `str` | (required) | Explanation for the decision. |
| `suggested_changes` | `list[str] \| None` | `None` | Suggested changes for replanning. |

### ResearcherOutput

Findings from a single research step.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `search_queries` | `list[str]` | `[]` | Queries used during research. |
| `observation` | `str` | `""` | Main observation/finding text. |
| `key_points` | `list[str]` | `[]` | Extracted key points. |
| `sources_used` | `list[str]` | `[]` | Source identifiers used. |
| `research_status` | `"ok" \| "blocked" \| "insufficient_data"` | `"ok"` | Status of the research step. |
| `blocking_reason` | `str \| None` | `None` | Explanation if the step was blocked. |
| `findings` | `str` | `""` | Alias for observation text. |
| `sources_found` | `int` | `0` | Number of sources discovered. |

### SynthesizerOutput

The final research report.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `report` | `str` | (required) | Full markdown report. |
| `structured_output` | `Any \| None` | `None` | Optional structured data alongside the report. |

## Pool Integration Flow

Pools are shared data stores that enable agents to pass accumulated knowledge across workflow steps. The flow through a single agent execution is:

```
Pool Inject --> Build Prompt --> LLM Call --> Parse Output --> Pool Writes
```

**Pool Inject** (`config.pool_inject`): Before the LLM call, the harness reads recent items from specified pools and formats them as sections in the user message. For example, a synthesizer can inject all accumulated findings from the `sources` pool. Each `PoolInjectConfig` controls the pool name, maximum items, and per-item character limits.

**Build Prompt**: Injected pool items are appended to the user message as formatted lists under section headers (e.g., `## sources`). This gives the LLM access to knowledge accumulated by prior agents without those agents needing to know about each other.

**LLM Call**: The model processes the prompt (which now includes pool context) and generates its response. For ReAct agents, tool results also contribute to the conversation context.

**Parse Output**: The raw LLM response is parsed according to `output_format` (text, JSON, or structured). The harness normalizes fields for researcher and background subtypes.

**Pool Writes** (`config.pool_writes`): After parsing, the harness extracts items from the output using dot-path expressions and appends them to target pools. For example, a researcher with `pool_writes: [{pool: sources, extract: sources}]` writes discovered source records to the shared `sources` pool. If the extract path yields nothing but the ReAct loop collected sources, those are used as a fallback.

This inject/write cycle means each agent in a workflow can read from and contribute to shared pools, building a cumulative research context across the entire pipeline.

## See Also

- [Architecture](architecture.md)
- [Builtin Agents Guide](../guides/builtin-agents.md)
- [Custom Agents Guide](../guides/custom-agents.md)
- [Tool System](tool-system.md)
- [Agent Config Reference](../reference/agent-config-reference.md)
