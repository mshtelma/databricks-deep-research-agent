# Workflow Definition Schema

> Complete YAML schema reference for workflow definitions.

All models use `extra="forbid"` -- unknown fields cause validation errors.

---

## Top-Level Fields

These are the fields of `WorkflowDefinition`, the root object of every workflow YAML file.

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `id` | `str` | -- | **yes** | Unique workflow identifier. |
| `name` | `str` | -- | **yes** | Human-readable name shown in UIs and logs. |
| `description` | `str` | `""` | no | Prose description of the workflow's purpose. |
| `version` | `int` | `1` | no | Schema version for forward-compatible migration. |
| `root` | `WorkflowNode` | -- | **yes** | The root node of the execution tree. |
| `tools` | `list[ToolDeclaration]` | `[]` | no | Top-level tool declarations referenced by agent nodes. |
| `pools` | `list[dict]` | `[]` | no | Pool declarations (validated as `PoolConfig`). |
| `sources` | `list[SourceDefinition]` | `[]` | no | Data-source declarations for source-aware planning. |
| `models` | `dict[str, str \| dict]` | `{}` | no | Per-tier model endpoint configuration. Keys are tier names; values are endpoint strings or dicts with `endpoints`, `fallback_on_429`, `rotation_strategy`, `tokens_per_minute`. See [Model Configuration](../guides/model-configuration.md). |
| `required_inputs` | `list[str]` | `["query"]` | no | State keys that **must** be present before execution begins. |
| `output_keys` | `list[str]` | `["output"]` | no | State keys the workflow is expected to produce. |
| `token_budget` | `int` | `0` | no | Maximum total tokens across all LLM calls. `0` means unlimited. |
| `timeout_seconds` | `int` | `1800` | no | Hard wall-clock timeout for the entire workflow execution. |

---

## WorkflowNode Fields

Every node in the workflow tree is a `WorkflowNode`.

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `id` | `str` | -- | **yes** | Unique node identifier within the workflow. |
| `type` | `NodeType` | -- | **yes** | Discriminator selecting the node kind (see enum below). |
| `label` | `str` | -- | **yes** | Human-readable label for UIs and log output. |
| `config` | `dict[str, Any]` | `{}` | no | Type-specific configuration (see *Node Config by Type*). |
| `children` | `list[WorkflowNode]` | `[]` | no | Child nodes. Required for composite types; must be empty for leaf types. |
| `error_handling` | `ErrorConfig \| null` | `null` | no | Per-node error policy. Inherits default (`fail`) when absent. |

**Leaf types** (`agent`, `tool`, `subworkflow`) must have no children.
**Composite types** (`sequence`, `parallel`, `loop`, `conditional`, `plan_and_execute`) carry child nodes.

---

## NodeType Enum Values

| Value | Kind | Description |
|-------|------|-------------|
| `agent` | leaf | An LLM call with prompt, tools, and structured output. |
| `tool` | leaf | A pure-tool invocation (no LLM). |
| `sequence` | composite | Run children one after another, in order. |
| `parallel` | composite | Run children concurrently (all must complete). |
| `loop` | composite | Repeat children until a condition is met. |
| `conditional` | composite | Branch to one child based on conditions. |
| `subworkflow` | leaf | Invoke a nested workflow definition. |
| `plan_and_execute` | composite | Planner agent generates a plan; body executes each step with optional re-planning. |

---

## ToolDeclaration Fields

Declared in the top-level `tools:` section. Agent nodes reference tools by `name`.

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `name` | `str` | -- | **yes** | Unique tool name. Agent configs reference this string. |
| `kind` | `str` | -- | **yes** | Tool kind (`web_search`, `web_crawl`, `vector_search`, or a custom string). |
| `config` | `dict[str, Any]` | `{}` | no | Kind-specific configuration (e.g., `index_name`, `num_results`). |
| `description` | `str` | `""` | no | Human-readable description injected into the tool definition. |

**Example:**

```yaml
tools:
  - name: earnings_index
    kind: vector_search
    config:
      index_name: prod_catalog.finance.earnings_idx
      num_results: 10
    description: "Quarterly earnings filings"
```

---

## SourceDefinition Fields

Declared in the top-level `sources:` section. Used by the planner for source-aware research plans. **Not** a tool declaration -- tools are registered separately.

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `name` | `str` | -- | **yes** | Unique ID, typically matching a tool name. |
| `kind` | `str` | -- | **yes** | Source kind (e.g., `vector_search`, `genie`, `knowledge_assistant`). |
| `endpoint` | `str` | `""` | no | Index name, Genie space ID, or similar endpoint identifier. |
| `description` | `str` | `""` | no | Human-readable description for planner context. |
| `query_strategy` | `dict[str, Any]` | `{}` | no | Kind-specific query configuration. |
| `metadata` | `dict[str, Any]` | `{}` | no | Extra metadata (columns, schema, etc.). |

---

## ErrorConfig Fields

Per-node error-handling policy. Attached via the `error_handling` field on any `WorkflowNode`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `on_error` | `str` | `"fail"` | Error strategy: `fail` (propagate exception), `skip` (emit `NodeSkippedEvent` and continue), or `retry`. |
| `max_retries` | `int` | `2` | Maximum retry attempts when `on_error` is `retry`. |
| `retry_delay_seconds` | `float` | `1.0` | Initial delay between retries (exponential back-off). |

**Example:**

```yaml
error_handling:
  on_error: retry
  max_retries: 3
  retry_delay_seconds: 2.0
```

---

## PoolConfig Fields

Declared in the top-level `pools:` section. Pools are shared, append-only collections used to pass data between agents.

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `name` | `str` | -- | **yes** | Unique pool identifier. Agents reference this in `pool_writes`, `pool_inject`, and `pool_tools`. |
| `item_type` | `str` | `"text"` | no | Semantic type of stored items: `text`, `source`, `claim`, `evidence`. |
| `dedup_key` | `str \| null` | `null` | no | Field name for key-based deduplication. Items with duplicate key values are rejected. |
| `dedup_content_hash` | `bool` | `true` | no | When `true`, items with identical content hashes are rejected. |
| `max_items` | `int` | `0` | no | Maximum pool capacity. `0` means unlimited. When full, oldest items are evicted. |

**Example:**

```yaml
pools:
  - name: sources
    dedup_key: url
    max_items: 200
  - name: observations
    dedup_key: content_hash
    max_items: 100
```

---

## Node Config by Type

The `config` dict on each `WorkflowNode` is validated according to its `type`. Below are the expected keys for each node type.

### agent config

Full configuration model: `AgentNodeConfig`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `subtype` | `str` | -- (**required**) | Agent subtype: `coordinator`, `planner`, `researcher`, `reflector`, `synthesizer`, `evaluator`, or a custom string. |
| `model_tier` | `str` | `"analytical"` | LLM tier: `simple` (fast), `analytical` (balanced), `complex` (reasoning). |
| `system_prompt` | `str` | `""` | System prompt. If empty, the builtin subtype default is used. |
| `user_prompt_template` | `str` | `""` | Jinja2 user prompt template. Variables are auto-detected or specified via `input_keys`. |
| `input_keys` | `list[str]` | `[]` | State keys to extract and inject into the prompt. Auto-detected from template when empty. |
| `output_key` | `str` | `"output"` | State key where the agent's output is stored. |
| `output_mode` | `str` | `"text"` | Output parsing mode: `text`, `json`, `structured`. |
| `output_format` | `str` | `"text"` | Output serialization format: `text`, `markdown`, `json`. |
| `output_schema` | `dict \| null` | `null` | JSON schema for structured output validation. |
| `grounding_mode` | `"none" \| "classical_lite" \| "reclaim" \| null` | `null` | Citation grounding strategy. |
| `tools` | `list[str \| dict]` | `[]` | Tools available to this agent. Each entry is a tool name string or a `{type, name}` dict. |
| `pool_writes` | `list[PoolWriteConfig]` | `[]` | Pools this agent writes results into after execution. |
| `pool_tools` | `list[str]` | `[]` | Pool names exposed as search tools (e.g., `pool_search`). |
| `pool_inject` | `list[PoolInjectConfig]` | `[]` | Pool contents injected into the prompt before execution. |
| `max_tool_calls` | `int` | `0` | Maximum tool calls in a ReAct loop. `0` means no tool calling. |
| `max_retries` | `int` | `2` | Maximum LLM call retries on transient failures. |
| `max_result_chars` | `int` | `4000` | Truncation limit for tool results. `0` means unlimited. |
| `conversation_budget` | `int \| null` | `null` | Token budget for the conversation. `null` means unlimited. |
| `output_model` | `Any` | `null` | Pydantic model class for structured output (programmatic only, not YAML). |

**PoolWriteConfig** (used in `pool_writes`):

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `pool` | `str` | -- (**required**) | Target pool name. |
| `extract` | `str` | -- (**required**) | Jinja/dot-path expression to extract items from agent output. |
| `transform` | `str \| null` | `null` | Optional transformation template applied to each extracted item. |

**PoolInjectConfig** (used in `pool_inject`):

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `pool` | `str` | -- (**required**) | Source pool name. |
| `threshold` | `float` | `0.0` | BM25/relevance threshold. |
| `format` | `str` | `"text"` | Injection format: `text`, `json`, `markdown`. |
| `max_items` | `int` | `20` | Maximum items to inject. |
| `max_item_chars` | `int` | `0` | Truncation per item. `0` means unlimited. |

**Example:**

```yaml
config:
  subtype: researcher
  model_tier: analytical
  output_key: findings
  tools:
    - type: builtin
      name: web_search
    - type: builtin
      name: web_crawl
  pool_writes:
    - pool: observations
      extract: findings
    - pool: sources
      extract: sources
  max_tool_calls: 15
```

---

### tool config

Full configuration model: `ToolNodeConfig`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `ref` | `dict[str, Any]` | -- (**required**) | Tool reference descriptor (e.g., `{type: builtin, name: web_search}`). |
| `input_mapping` | `dict[str, str]` | `{}` | Maps state keys to tool argument names. |
| `output_key` | `str` | `"tool_result"` | State key where the tool result is stored. |

---

### sequence config

No additional config keys. Children are executed in order.

```yaml
- id: main
  type: sequence
  label: Main Pipeline
  children:
    - ...
    - ...
```

---

### parallel config

No additional config keys. Children are executed concurrently; all must complete.

```yaml
- id: parallel_research
  type: parallel
  label: Parallel Research
  children:
    - ...
    - ...
```

---

### loop config

Full configuration model: `LoopNodeConfig`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `until` | `dict[str, Any]` | -- (**required**) | Exit condition (serialised `StateCondition`, `LLMCondition`, or `CompositeCondition`). |
| `min_iterations` | `int` | `1` | Minimum iterations before the exit condition is checked. |
| `max_iterations` | `int` | `10` | Hard upper bound on iterations. |

**Condition types** (used in `until`):

*StateCondition* -- evaluate against a state key:

| Key | Type | Description |
|-----|------|-------------|
| `type` | `"state"` | Discriminator. |
| `key` | `str` | Dot-path state key (e.g., `research.step_count`). |
| `operator` | `str` | One of: `eq`, `neq`, `gt`, `lt`, `gte`, `lte`, `contains`, `in`, `exists`, `not_exists`. |
| `value` | `Any` | Comparison value (not required for `exists`/`not_exists`). |

*LLMCondition* -- ask an LLM a yes/no question:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `type` | `"llm"` | -- | Discriminator. |
| `prompt_template` | `str` | -- | Jinja2 prompt template for the LLM. |
| `model_tier` | `str` | `"simple"` | LLM tier to use. |
| `expected_output` | `str` | `"yes"` | The LLM response that means "condition is true". |

*CompositeCondition* -- boolean combination:

| Key | Type | Description |
|-----|------|-------------|
| `type` | `"composite"` | Discriminator. |
| `operator` | `str` | Boolean operator: `all` (AND), `any` (OR), `not` (NOT). |
| `conditions` | `list[Condition]` | Nested conditions. |

**Example:**

```yaml
config:
  until:
    type: state
    key: reflection.decision
    operator: eq
    value: COMPLETE
  min_iterations: 2
  max_iterations: 10
```

---

### conditional config

Full configuration model: `ConditionalNodeConfig`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `conditions` | `list[ConditionBranch]` | -- (**required**) | List of condition-to-child-index mappings, evaluated in order. |
| `default_branch` | `int` | `0` | Index of the child to execute when no condition matches. |

Each `ConditionBranch` is:

| Key | Type | Description |
|-----|------|-------------|
| `condition` | `StateCondition \| LLMCondition \| CompositeCondition` | The condition to evaluate. |
| `child_index` | `int` | Index of the child node to execute if the condition is true. |

---

### subworkflow config

Full configuration model: `SubworkflowNodeConfig`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `ref` | `str` | -- (**required**) | Workflow name or file path to the nested workflow definition. |
| `params` | `dict[str, Any]` | `{}` | Static parameters passed to the sub-workflow. |
| `input_mapping` | `dict[str, str]` | `{}` | Maps parent state keys to sub-workflow input keys. |
| `output_mapping` | `dict[str, str]` | `{}` | Maps sub-workflow output keys back to parent state keys. |
| `output_key` | `str` | `"subworkflow_result"` | Default parent state key for the sub-workflow result. |
| `pool_mode` | `str` | `"inherit"` | Pool sharing strategy: `inherit` (shared pools), `isolate` (separate pools), `merge` (merge on completion). |

---

### plan_and_execute config

Full configuration model: `PlanAndExecuteNodeConfig`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `planner` | `dict[str, Any]` | -- (**required**) | Serialised `AgentNodeConfig` for the planner agent. |
| `items_path` | `str` | `"steps"` | Dot-path into planner output for the iterable of plan items. |
| `item_state_key` | `str` | `"current_step"` | State key where each plan item is placed during execution. |
| `body` | `dict[str, Any]` | `{}` | Serialised child node(s) to run per plan item. |
| `evaluator` | `dict[str, Any] \| null` | `null` | Optional evaluator/reflector agent config, run after each step. |
| `max_iterations` | `int` | `10` | Maximum total steps to execute. |
| `min_iterations` | `int` | `1` | Minimum steps before the evaluator can signal completion. |
| `max_replan_cycles` | `int` | `3` | Maximum times the planner can regenerate the plan mid-execution. |
| `complete_on_exhaustion` | `bool` | `true` | Whether to proceed to synthesis when all steps are exhausted (even without explicit COMPLETE). |
| `planner_guidance` | `str` | `""` | Free-text guidance injected into the planner prompt. |
| `synthesis_metadata` | `dict[str, str]` | `{}` | Key-value pairs written to state for the synthesizer. |

---

## Complete Example

A full workflow YAML with annotations for every field.

```yaml
# ---- Top-level WorkflowDefinition fields ----
id: deep_research                      # Unique workflow ID
name: Deep Research                    # Human-readable name
description: "Full multi-agent deep research pipeline"  # Optional
version: 1                             # Schema version
required_inputs: [query]               # State keys required before start
output_keys: [report]                  # State keys produced
token_budget: 0                        # 0 = unlimited
timeout_seconds: 1800                  # 30-minute hard timeout

# ---- Pool declarations ----
pools:
  - name: sources                      # Pool name
    item_type: source                  # Semantic type
    dedup_key: url                     # Key-based dedup on "url" field
    dedup_content_hash: true           # Also hash-dedup (default)
    max_items: 200                     # Evict oldest when full
  - name: observations
    dedup_key: content_hash
    max_items: 100

# ---- Tool declarations ----
tools:
  - name: earnings_search
    kind: vector_search
    config:
      index_name: prod.finance.earnings_idx
      num_results: 10
    description: "Quarterly earnings filings"

# ---- Source declarations ----
sources:
  - name: earnings_search
    kind: vector_search
    endpoint: prod.finance.earnings_idx
    description: "SEC 10-Q and 10-K filings"
    query_strategy:
      rewrite: true
      expand_synonyms: true
    metadata:
      columns: [filing_date, ticker, content]

# ---- Workflow tree ----
root:
  id: main
  type: sequence
  label: Deep Research Pipeline
  error_handling:                       # ErrorConfig on the root node
    on_error: fail
    max_retries: 2
    retry_delay_seconds: 1.0
  children:

    # --- Agent node: coordinator ---
    - id: coordinator
      type: agent
      label: Query Classifier
      config:
        subtype: coordinator
        model_tier: simple
        output_key: coordination

    # --- Agent node: background ---
    - id: background
      type: agent
      label: Background Investigator
      config:
        subtype: background
        model_tier: simple
        output_key: background
        tools:
          - type: builtin
            name: web_search
          - type: builtin
            name: web_crawl
        max_tool_calls: 5

    # --- Plan-and-execute node: research cycle ---
    - id: research_cycle
      type: plan_and_execute
      label: Research Cycle
      config:
        planner:                        # AgentNodeConfig for planner
          subtype: planner
          model_tier: analytical
          output_key: plan
        items_path: steps               # Dot-path into planner output
        item_state_key: current_step    # State key per iteration
        body:                           # Child node executed per step
          id: researcher
          type: agent
          label: Researcher
          config:
            subtype: researcher
            model_tier: analytical
            output_key: findings
            tools:
              - type: builtin
                name: web_search
              - type: builtin
                name: web_crawl
              - earnings_search         # Reference to declared tool
            pool_writes:
              - pool: observations
                extract: findings
              - pool: sources
                extract: sources
            max_tool_calls: 15
            max_result_chars: 4000
        evaluator:                      # Optional reflector agent
          subtype: reflector
          model_tier: analytical
          output_key: evaluation
          pool_inject:
            - pool: observations
              threshold: 0
              format: text
              max_items: 20
        max_iterations: 7
        min_iterations: 2
        max_replan_cycles: 3
        complete_on_exhaustion: true
        planner_guidance: "Focus on primary sources and recent data."
        synthesis_metadata:
          depth: extended

    # --- Agent node: synthesizer ---
    - id: synthesizer
      type: agent
      label: Report Synthesizer
      config:
        subtype: synthesizer
        model_tier: complex
        output_key: report
        output_format: markdown
        pool_inject:
          - pool: observations
            threshold: 0
          - pool: sources
            threshold: 0
        pool_tools:
          - observations
          - sources
        max_tool_calls: 10
```

---

## See Also

- [Workflow Engine](../concepts/workflow-engine.md)
- [Node Types Reference](node-types-reference.md)
- [YAML Workflow Authoring](../guides/yaml-workflow-authoring.md)
