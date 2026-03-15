# Workflow Engine

> How workflows are defined, loaded, validated, and executed.

## Overview

The workflow engine is the core of the framework. It:

1. Defines workflows as recursive trees of typed nodes (`WorkflowDefinition`)
2. Loads and saves them from YAML
3. Validates structure and references at load time
4. Executes them by walking the tree depth-first, yielding `StreamEvent` objects

The schema layer (`definition.py`) is intentionally free of runtime behaviour. Loading (`loader.py`) handles YAML deserialization and triggers validation (`validation.py`). Execution (`executor.py`) walks the validated tree.

## WorkflowDefinition

The top-level, serializable description of a complete workflow.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | `str` | *required* | Unique workflow identifier |
| `name` | `str` | *required* | Human-readable name shown in UIs and logs |
| `description` | `str` | `""` | Prose description of the workflow's purpose |
| `version` | `int` | `1` | Schema version for forward-compatible migration |
| `root` | `WorkflowNode` | *required* | Root node of the execution tree |
| `tools` | `list[ToolDeclaration]` | `[]` | Tool declarations available to agent nodes |
| `pools` | `list[dict[str, Any]]` | `[]` | Pool declarations (validated later via `PoolConfig`) |
| `sources` | `list[SourceDefinition]` | `[]` | Data source declarations for planner context |
| `required_inputs` | `list[str]` | `["query"]` | State keys that must be present before execution |
| `output_keys` | `list[str]` | `["output"]` | State keys the workflow is expected to produce |
| `token_budget` | `int` | `0` | Max total tokens across all LLM calls (`0` = unlimited) |
| `timeout_seconds` | `int` | `1800` | Hard wall-clock timeout for the entire execution |

`WorkflowDefinition` uses Pydantic's `extra="forbid"` config, so unrecognized fields in YAML cause an immediate error rather than being silently ignored.

## WorkflowNode

A single node in the workflow tree. The recursive `children` field enables the tree structure.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | `str` | *required* | Unique identifier within the workflow |
| `type` | `NodeType` | *required* | Discriminator for the node kind |
| `label` | `str` | *required* | Human-readable label for logs and events |
| `config` | `dict[str, Any]` | `{}` | Free-form config; validated per type at load/execution time |
| `children` | `list[WorkflowNode]` | `[]` | Child nodes (empty for leaf types, populated for composites) |
| `error_handling` | `ErrorConfig \| None` | `None` | Per-node error policy; inherits parent behavior when `None` |
| `budget_seconds` | `float \| None` | `None` | Per-node wall-clock time limit in seconds. Raises `NodeBudgetExceededError` when exceeded. `None` = no limit. |

The `config` dict is intentionally untyped at the definition layer. Concrete per-type config models (`AgentNodeConfig`, `LoopNodeConfig`, `ConditionalNodeConfig`, `ToolNodeConfig`, `PlanAndExecuteNodeConfig`) are validated by the executor when the node type is known.

## NodeType Enum

Eight node types, split into leaf and composite categories:

| Type | Category | Description |
|------|----------|-------------|
| `agent` | Leaf | Runs an LLM agent (prompt -> LLM -> parse -> state). Requires `config.subtype`. |
| `tool` | Leaf | Invokes a single tool by reference. Requires `config.ref`. |
| `subworkflow` | Leaf | Delegates to another `WorkflowDefinition` |
| `sequence` | Composite | Runs children one after another in order |
| `parallel` | Composite | Runs children concurrently; children must have non-overlapping `output_key` values |
| `loop` | Composite | Repeats children until a condition is met or max iterations reached |
| `conditional` | Composite | Evaluates conditions and selects a branch; requires at least 2 children |
| `plan_and_execute` | Composite | Dynamically plans items, executes each, reflects, and optionally replans. Children must be empty (body lives in config). |

For detailed behavior and config options per type, see [Node Types Reference](../reference/node-types-reference.md).

## ToolDeclaration

Declares a tool in the workflow's top-level `tools:` section. Agent nodes reference tools by `name`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | *required* | Unique tool name, referenced in agent configs |
| `kind` | `str` | *required* | Tool kind (e.g., `web_search`, `vector_search`, `genie`) |
| `config` | `dict[str, Any]` | `{}` | Kind-specific configuration |
| `description` | `str` | `""` | Human-readable description, injected into tool definitions |

Example:

```yaml
tools:
  - name: earnings_index
    kind: vector_search
    config:
      index_name: prod_catalog.finance.earnings_idx
      num_results: 10
    description: "Quarterly earnings filings"
```

## SourceDefinition

Declares a data source available to the workflow. Used by the planner to generate source-aware research plans. This is separate from tool declarations -- sources describe *what data is available*, tools describe *how to access it*.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | *required* | Unique ID, typically matching a tool name |
| `kind` | `str` | *required* | Source kind value |
| `endpoint` | `str` | `""` | Index name, Genie space ID, etc. |
| `description` | `str` | `""` | Human-readable, included in planner context |
| `query_strategy` | `dict[str, Any]` | `{}` | Kind-specific query configuration |
| `metadata` | `dict[str, Any]` | `{}` | Extra info (columns, schema, etc.) |

When `sources` is omitted from YAML but `tools` are present, the loader automatically derives source definitions from tool declarations using a kind-based mapping.

## ErrorConfig

Per-node error-handling policy.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `on_error` | `str` | `"fail"` | Policy: `fail` (propagate), `skip` (emit `NodeSkippedEvent` and continue), or `retry` |
| `max_retries` | `int` | `2` | Maximum retry attempts when `on_error` is `retry` |
| `retry_delay_seconds` | `float` | `1.0` | Delay between retries with exponential back-off |

## YAML Loading

Three public functions handle serialization:

### `load_workflow(path)`

Loads a workflow from a YAML file on disk.

```python
from databricks_deep_research.workflow.loader import load_workflow

defn = load_workflow("workflows/deep_research.yaml")
```

Raises `FileNotFoundError` if the path does not exist. Raises `WorkflowValidationError` if the YAML fails structural validation.

### `load_workflow_from_string(yaml_content)`

Parses a workflow from a raw YAML string. Useful for tests and dynamically generated workflows.

```python
from databricks_deep_research.workflow.loader import load_workflow_from_string

defn = load_workflow_from_string(yaml_text)
```

### `save_workflow(definition, path)`

Serializes a `WorkflowDefinition` to a YAML file. Parent directories must already exist.

```python
from databricks_deep_research.workflow.loader import save_workflow

save_workflow(defn, "out.yaml")
```

### Convenience methods

The loader module also wires up `WorkflowDefinition.from_yaml()` and `WorkflowDefinition.to_yaml()` as convenience methods that delegate to the functions above:

```python
defn = WorkflowDefinition.from_yaml("workflows/deep_research.yaml")
defn.to_yaml("out.yaml")
```

### Deserialization pipeline

The loading process follows these steps:

1. Parse YAML text into a raw Python dict via `yaml.safe_load`
2. Validate that the root is a mapping and required top-level fields (`id`, `name`, `root`) are present
3. Recursively build `WorkflowNode` objects from the `root` dict, validating required fields (`id`, `type`, `label`) and recognized `NodeType` values at each level
4. Parse `ToolDeclaration` objects from the `tools` list
5. Parse `SourceDefinition` objects from `sources`, or auto-derive them from tool declarations when `sources` is empty
6. Construct the `WorkflowDefinition` Pydantic model
7. Run `validate_workflow()` for structural validation
8. Return the validated definition

## Validation

`validate_workflow()` performs load-time structural checks on a `WorkflowDefinition`. It raises `WorkflowValidationError` with a list of all errors found (not just the first one).

### Checks performed

**Top-level constraints:**
- `required_inputs` must be a non-empty list
- `output_keys` must be a non-empty list

**Node tree constraints (applied recursively):**
- All node IDs must be unique across the entire tree
- Leaf types (`agent`, `tool`, `subworkflow`) must have no children
- Composite types (`sequence`, `parallel`, `loop`) must have at least 1 child
- `conditional` nodes must have at least 2 children (branches)
- `plan_and_execute` nodes must have exactly 0 children (the body lives in config)
- `parallel` node children must have non-overlapping `output_key` values
- `agent` nodes must include `subtype` in their config
- `tool` nodes must include `ref` in their config

**Warnings (logged, not errors):**
- Agent nodes with `pool_writes.extract` that won't match the `output_key` for `text`/`markdown` output formats

## Execution Model

The `WorkflowExecutor` walks the validated tree depth-first, yielding `StreamEvent` objects as an async generator.

- Each of the 8 node types has a dedicated execution handler
- The executor takes a `WorkflowDefinition`, a `FrameworkLLMClient`, and optional tool resolver/registry/factories
- Events are yielded throughout execution: `WorkflowStartedEvent`, `NodeStartedEvent`, `NodeCompletedEvent`, `LoopIterationEvent`, `BranchSelectedEvent`, `ToolResultEvent`, `WorkflowCompletedEvent`, and others
- `WorkflowState` (append-only log) is mutated as nodes complete
- Conditions (for `conditional` and `loop` nodes) are evaluated against a materialized snapshot of latest state values
- Token budget and timeout are enforced globally
- Error handling follows the per-node `ErrorConfig` policy

For a detailed walkthrough of the execution architecture, see [Architecture](architecture.md).

## Example

A minimal research workflow in YAML:

```yaml
id: minimal-research
name: minimal-research
version: 1
tools:
  - name: web_search
    kind: web_search
pools:
  - name: sources
    item_type: source
    dedup_key: url
root:
  id: research
  type: sequence
  label: Research pipeline
  children:
    - id: plan
      type: agent
      label: Plan research steps
      config:
        subtype: planner
    - id: research-step
      type: agent
      label: Execute research
      config:
        subtype: researcher
        tools: [web_search]
    - id: synthesize
      type: agent
      label: Synthesize findings
      config:
        subtype: synthesizer
```

**Breakdown:**

- **`id` / `name`**: Identify the workflow in logs and UIs.
- **`tools`**: Declares `web_search` as available. Agent nodes reference it by name.
- **`pools`**: A `sources` pool with URL-based deduplication for collected research sources.
- **`root`**: A `sequence` node that runs three agent children in order:
  1. **`plan`** -- a `planner` agent that generates research steps.
  2. **`research-step`** -- a `researcher` agent that executes search queries using the `web_search` tool.
  3. **`synthesize`** -- a `synthesizer` agent that combines findings into the final output.

Since `sources` is omitted, the loader auto-derives a source definition from the `web_search` tool declaration.

## See Also

- [Architecture](architecture.md)
- [Node Types Reference](../reference/node-types-reference.md)
- [YAML Workflow Authoring](../guides/yaml-workflow-authoring.md)
- [State Management](state-management.md)
