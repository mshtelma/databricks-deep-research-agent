# Agent Config Reference

> Complete field-by-field reference for AgentNodeConfig and related models.

## AgentNodeConfig

Full configuration for a single agent (LLM) node.

### Identity

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `subtype` | `str` | **required** | Agent subtype name (coordinator, planner, researcher, reflector, synthesizer, evaluator) |
| `model_tier` | `str` | `"analytical"` | LLM tier: `simple` (fast), `analytical` (balanced), `complex` (reasoning) |

### Prompts

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `system_prompt` | `str` | `""` | System prompt template. Supports Jinja-style variable interpolation |
| `user_prompt_template` | `str` | `""` | User prompt template. Variables are auto-detected by the harness via `SafeTemplateRenderer.extract_variables()` |

### Input/Output

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `input_keys` | `list[str]` | `[]` | State keys to read. Auto-detected from prompt templates at runtime; explicit values in YAML override auto-detection |
| `output_key` | `str` | `"output"` | State key to write the agent's result into |
| `output_mode` | `str` | `"text"` | Output parsing mode: `text`, `json`, `structured` |
| `output_format` | `str` | `"text"` | Format hint for the output: `text`, `markdown`, `json` |
| `output_schema` | `dict[str, Any] \| None` | `None` | JSON Schema for validating structured output |
| `output_model` | `Any \| None` | `None` | Pydantic model class for structured output parsing |

### Tools

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tools` | `list[str \| dict[str, Any]]` | `[]` | Tool names (or inline tool configs) to make available to the agent |
| `max_tool_calls` | `int` | `0` | Max tool calls in a ReAct loop. `0` means no tool calling (single-pass) |
| `max_result_chars` | `int` | `4000` | Max chars per tool result. `0` = unlimited; `>0` truncates older tool results |

### Pools

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `pool_writes` | `list[PoolWriteConfig]` | `[]` | How the agent writes items to shared pools after execution |
| `pool_tools` | `list[str]` | `[]` | Pool tool names available to the agent (e.g., `pool_search`, `pool_get_recent`) |
| `pool_inject` | `list[PoolInjectConfig]` | `[]` | How pool contents are injected into the agent's prompt before execution |

### Synthesis Context

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `synthesis_context` | `SynthesisContextConfig \| None` | `None` | Controls how accumulated research is injected into synthesizer prompts |

### Grounding

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `grounding_mode` | `"none" \| "classical_lite" \| "reclaim" \| None` | `None` | Citation grounding strategy: `none` (no grounding), `classical_lite` (lightweight), `reclaim` (full retrieval-augmented claim verification) |

### Budget

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `conversation_budget` | `int \| None` | `None` | Max conversation tokens. `None` = no limit |
| `max_retries` | `int` | `2` | Max retries on LLM call failure |

---

## PoolWriteConfig

Describes how an agent writes items to a shared pool after execution.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `pool` | `str` | **required** | Target pool name |
| `extract` | `str` | **required** | Jinja / dot-path expression applied to the agent output to extract items |
| `transform` | `str \| None` | `None` | Optional transformation template applied to each extracted item |

---

## PoolInjectConfig

Describes how pool contents are injected into an agent's prompt before execution.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `pool` | `str` | **required** | Source pool name |
| `threshold` | `float` | `0.0` | BM25 / relevance score threshold for filtering items |
| `format` | `str` | `"text"` | Injection format: `text`, `json`, `markdown` |
| `max_items` | `int` | `20` | Maximum number of items to inject |
| `max_item_chars` | `int` | `0` | Max chars per item. `0` = unlimited; `>0` truncates each item |
| `compaction` | `PromptCompactionConfig \| None` | `None` | Compaction strategy for managing large pool injections |

---

## PromptCompactionConfig

Controls prompt-context compaction for injected pool content. Used by `PoolInjectConfig.compaction` and `SynthesisContextFieldConfig.compaction`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | `"none" \| "dedupe" \| "summarize" \| "auto"` | `"none"` | Compaction strategy |
| `max_total_chars` | `int` | `0` | Hard cap on total injected chars (`0` = unlimited) |
| `target_chars` | `int` | `0` | Target chars for summarization mode |
| `summary_model_tier` | `str` | `"simple"` | Model tier for summarization |
| `dedupe_key` | `"auto" \| "url" \| "title" \| "text"` | `"auto"` | Key for dedup-based compaction |

---

## SynthesisContextConfig

Controls synthesizer-specific context materialization. Each field configures how one category of accumulated research is prepared before injection into the synthesizer prompt.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `observations` | `SynthesisContextFieldConfig \| None` | `None` | Controls observation injection |
| `sources` | `SynthesisContextFieldConfig \| None` | `None` | Controls source injection |
| `fallback_discovery_sources` | `SynthesisContextFieldConfig \| None` | `None` | Controls fallback discovery source injection |

### SynthesisContextFieldConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_items` | `int` | `20` | Maximum number of items to inject |
| `max_item_chars` | `int` | `0` | Max chars per item (`0` = unlimited) |
| `compaction` | `PromptCompactionConfig \| None` | `None` | Optional compaction strategy |

---

## ToolNodeConfig

Configuration for a pure-tool (no LLM) node.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `ref` | `dict[str, Any]` | **required** | Tool reference descriptor (identifies the tool to invoke) |
| `input_mapping` | `dict[str, str]` | `{}` | Maps state keys to tool input parameter names |
| `output_key` | `str` | `"tool_result"` | State key to write the tool result into |

---

## LoopNodeConfig

Configuration for a loop control node.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `until` | `dict[str, Any]` | **required** | Serialized condition (StateCondition, LLMCondition, or Composite) that terminates the loop |
| `min_iterations` | `int` | `1` | Minimum iterations before the `until` condition is checked |
| `max_iterations` | `int` | `10` | Hard upper bound on iterations |

---

## ConditionalNodeConfig

Configuration for a conditional branching node.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `conditions` | `list[dict[str, Any]]` | **required** | List of serialized `ConditionBranch` objects, evaluated in order |
| `default_branch` | `int` | `0` | Index of the branch to take when no condition matches |

---

## SubworkflowNodeConfig

Configuration for invoking a nested workflow.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `ref` | `str` | **required** | Workflow name or file path to invoke |
| `params` | `dict[str, Any]` | `{}` | Static parameters passed to the subworkflow |
| `input_mapping` | `dict[str, str]` | `{}` | Maps parent state keys to subworkflow input keys |
| `output_mapping` | `dict[str, str]` | `{}` | Maps subworkflow output keys back to parent state keys |
| `output_key` | `str` | `"subworkflow_result"` | State key to write the subworkflow result into |
| `pool_mode` | `str` | `"inherit"` | Pool sharing strategy: `inherit` (share parent pools), `isolate` (fresh pools), `merge` (merge on completion) |

---

## PlanAndExecuteNodeConfig

Configuration for the plan-and-execute meta-node. The planner agent generates a dynamic plan, and the body nodes execute each plan item in a loop with optional re-planning.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `planner` | `dict[str, Any]` | **required** | Serialized `AgentNodeConfig` for the planner agent |
| `items_path` | `str` | `"steps"` | Dot-path into the planner output to extract the iterable of plan items |
| `item_state_key` | `str` | `"current_step"` | State key where each plan item is placed during execution |
| `body` | `dict[str, Any]` | `{}` | Serialized child node(s) to run for each plan item |
| `evaluator` | `dict[str, Any] \| None` | `None` | Optional serialized `AgentNodeConfig` for an evaluator agent that reviews each iteration |
| `max_iterations` | `int` | `10` | Hard upper bound on plan item executions |
| `min_iterations` | `int` | `1` | Minimum iterations before early stopping is allowed |
| `max_replan_cycles` | `int` | `3` | Maximum number of times the planner can regenerate the plan |
| `complete_on_exhaustion` | `bool` | `True` | Whether to proceed to synthesis when max iterations are reached (vs. raising an error) |
| `planner_guidance` | `str` | `""` | Free-text guidance injected into the planner prompt |
| `synthesis_metadata` | `dict[str, str]` | `{}` | Key-value pairs written to state for the downstream synthesizer |

---

## Subtype Defaults

Each builtin subtype overrides specific `AgentNodeConfig` defaults. Fields not listed use the `AgentNodeConfig` defaults shown above.

> **Note:** `input_keys` values below are documentation only. At runtime, input keys are auto-detected from prompt templates by the harness. Explicit `input_keys` in YAML workflow definitions always override auto-detection.

| Field | `coordinator` | `planner` | `researcher` | `reflector` | `synthesizer` | `evaluator` |
|-------|--------------|-----------|-------------|------------|--------------|------------|
| `model_tier` | `simple` | `analytical` | `analytical` | `analytical` | `complex` | `analytical` |
| `output_key` | `coordination` | `plan` | `findings` | `reflection` | `report` | `evaluation` |
| `output_format` | `json` | `json` | `json` | `json` | `markdown` | `json` |
| `input_keys` | `query` | `query`, `background` | `query`, `current_step`, `plan` | `query`, `plan_summary`, `findings`, `current_step`, `remaining_steps`, `total_steps`, `steps_completed`, `min_steps`, `step_title`, `iteration`, `observation`, `all_observations`, `sources_count`, `source_topics`, `source_quality` | `query`, `plan` | *(same as reflector)* |
| `tools` | `[]` | `[]` | `web_search`, `web_crawl` | `[]` | `[]` | `[]` |
| `pool_writes` | `[]` | `[]` | `[{pool: sources, extract: sources}]` | `[]` | `[]` | `[]` |
| `pool_tools` | `[]` | `[]` | `pool_search` | `[]` | `pool_search` | `[]` |

---

## See Also

- [Agent System](../concepts/agent-system.md)
- [Builtin Agents](../guides/builtin-agents.md)
- [Custom Agents](../guides/custom-agents.md)
