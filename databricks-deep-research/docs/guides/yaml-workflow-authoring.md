# YAML Workflow Authoring

> Write custom research workflows from scratch using YAML definitions.

## Overview

A workflow YAML file is a declarative blueprint for a multi-agent research pipeline. It describes **what** agents to run, in **what order**, with **which tools** and **shared data pools** -- without writing any Python. The framework's executor reads the YAML, validates it, and walks the tree depth-first, yielding streaming events along the way.

This guide walks you through writing YAML workflows from simple to complex. By the end, you will be able to compose any multi-agent research pipeline.

---

## Minimal Workflow

Start with the absolute minimum -- a single agent node:

```yaml
id: hello_research
name: Hello Research
version: 1
required_inputs: [query]
output_keys: [report]

root:
  id: answer
  type: agent
  label: Direct Answer
  config:
    subtype: synthesizer
    user_prompt_template: "Answer this question: {{ query }}"
    output_key: report
```

This workflow has:

- **`id`** / **`name`** -- unique identifier and human-readable name.
- **`version`** -- schema version (always `1` for now).
- **`required_inputs`** -- state keys that must be provided before execution. The executor will refuse to start without them.
- **`output_keys`** -- state keys the workflow is expected to produce. Used by `WorkflowResult.output` to find the final text.
- **`root`** -- the single top-level node. Every node needs `id`, `type`, and `label`.

The agent `config` says "use the built-in `synthesizer` subtype" (which comes with a default system prompt for report writing) and "write output to the state key `report`". The `{{ query }}` syntax is a Jinja2 template variable -- it is filled from the workflow state at runtime.

### Running it

```python
from databricks_deep_research import WorkflowRunner

runner = WorkflowRunner.from_databricks()
result = await runner.run("hello_research.yaml", query="What is quantum computing?")
print(result.output)
```

Or with the lower-level API:

```python
from databricks_deep_research import run_workflow_from_yaml, FrameworkLLMClient

client = FrameworkLLMClient.from_databricks(model="databricks-claude-sonnet-4")
state, events = await run_workflow_from_yaml(
    "hello_research.yaml",
    client,
    initial_state={"query": "What is quantum computing?"},
)
print(state.get("report"))
```

---

## Adding Tools

Tools are declared at the top level of the workflow and referenced by name inside agent nodes.

```yaml
tools:
  - name: web_search
    kind: web_search
    config:
      brave_api_key: "${BRAVE_API_KEY}"

  - name: web_crawl
    kind: web_crawl
```

Each tool declaration has:

| Field | Required | Description |
|-------|----------|-------------|
| `name` | yes | Unique name. Agent nodes reference this string. |
| `kind` | yes | Tool kind. Built-in kinds: `web_search`, `web_crawl`, `file_search`, `vector_search`, `genie`, `knowledge_assistant`, `custom`. |
| `config` | no | Kind-specific configuration (API keys, index names, etc.). Supports `${ENV_VAR}` syntax. |
| `description` | no | Human-readable text injected into the tool definition the LLM sees. |

### Enterprise tool kinds

| Kind | Config keys | Description |
|------|-------------|-------------|
| `vector_search` | `index_name`, `num_results` | Databricks Vector Search index |
| `genie` | `space_id` | Databricks Genie (data warehouse natural language) |
| `knowledge_assistant` | `endpoint_name` | AI-powered Q&A on internal docs |

Example with enterprise tools:

```yaml
tools:
  - name: earnings_index
    kind: vector_search
    config:
      index_name: prod_catalog.finance.earnings_idx
      num_results: 10
    description: "Quarterly earnings filings and financial data"

  - name: genie
    kind: genie
    config:
      space_id: 01ef8d7a-0000-0000-0000-000000000000
    description: "Enterprise data warehouse -- operational metrics and KPIs"
```

---

## Adding Pools

Pools are shared, append-only data stores that let agents pass information to each other without direct coupling. A researcher writes sources into a pool; the synthesizer reads them out.

```yaml
pools:
  - name: sources
    dedup_key: url
    max_items: 100

  - name: observations
    dedup_content_hash: true
    max_items: 50
```

| Field | Default | Description |
|-------|---------|-------------|
| `name` | (required) | Unique pool name. Must match `pool_writes` and `pool_inject` references. |
| `dedup_key` | `null` | Field name for key-based deduplication (e.g. `url`). Items with the same key value are dropped. |
| `dedup_content_hash` | `true` | Hash-based dedup. Prevents identical items regardless of key. |
| `max_items` | `0` (unlimited) | Capacity limit. When full, the oldest item is evicted. |

Agents interact with pools in two ways:

1. **`pool_writes`** -- the agent's output is extracted and added to a pool after execution.
2. **`pool_inject`** -- pool contents are injected into the agent's prompt before execution.

Both are configured inside the agent's `config` block (see [Pool Configuration](#pool-configuration) below).

---

## Model Configuration

Workflow YAML files can optionally define their own model tier mappings in a top-level `models:` section. This makes the workflow self-contained — no Python `model_mapping` needed.

```yaml
models:
  simple: databricks-claude-haiku-4-5
  analytical:
    endpoints: [databricks-claude-haiku-4-5, databricks-gpt-5-mini]
    fallback_on_429: true
  complex: databricks-claude-opus-4-5
```

For the full story — multi-endpoint failover, rotation strategies, custom tiers — see [Model Configuration](model-configuration.md).

---

## Building a Sequence

The most common composite node is `sequence` -- it runs children one after another. Here is a three-agent pipeline:

```yaml
id: basic_pipeline
name: Basic Pipeline
version: 1
required_inputs: [query]
output_keys: [report]

pools:
  - name: sources
    dedup_key: url
    max_items: 100
  - name: observations
    dedup_content_hash: true
    max_items: 50

root:
  id: main
  type: sequence
  label: Main Pipeline
  children:
    # 1. Classify the query
    - id: coordinator
      type: agent
      label: Query Classifier
      config:
        subtype: coordinator
        model_tier: simple
        output_key: coordination

    # 2. Research
    - id: researcher
      type: agent
      label: Researcher
      config:
        subtype: researcher
        model_tier: analytical
        output_key: findings
        tools: [web_search, web_crawl]
        pool_writes:
          - pool: observations
            extract: findings
          - pool: sources
            extract: sources
        max_tool_calls: 10

    # 3. Synthesize
    - id: synthesizer
      type: agent
      label: Report Synthesizer
      config:
        subtype: synthesizer
        model_tier: complex
        output_key: report
        pool_inject:
          - pool: observations
            threshold: 0
          - pool: sources
            threshold: 0
```

Key points:

- Each child's `output_key` writes to state. Subsequent children can read those keys via prompt templates.
- The coordinator writes to `coordination`; the researcher can see `{{ coordination }}` in its prompt.
- The synthesizer uses `pool_inject` to pull all observations and sources into its prompt context.

---

## The Plan-and-Execute Pattern

This is the core pattern for deep research. A planner creates a list of research steps, a researcher executes each one, and a reflector evaluates progress -- deciding whether to continue, adjust the plan, or complete early.

```yaml
- id: research_cycle
  type: plan_and_execute
  label: Research Cycle
  config:
    # --- The planner agent ---
    planner:
      subtype: planner
      model_tier: analytical
      output_key: plan

    # --- How to extract steps from the plan ---
    items_path: steps          # dot-path into planner output for the iterable
    item_state_key: current_step  # state key each item is written to

    # --- The body node (runs once per step) ---
    body:
      id: researcher
      type: agent
      label: Researcher
      config:
        subtype: researcher
        model_tier: analytical
        output_key: findings
        tools: [web_search, web_crawl]
        pool_writes:
          - pool: observations
            extract: findings
          - pool: sources
            extract: sources
        max_tool_calls: 15

    # --- The evaluator agent (runs after each step) ---
    evaluator:
      subtype: reflector
      model_tier: analytical
      output_key: evaluation
      pool_inject:
        - pool: observations
          threshold: 0

    # --- Iteration controls ---
    min_iterations: 2       # run at least this many steps
    max_iterations: 10      # hard cap on steps
    max_replan_cycles: 3    # max times the plan can be regenerated
```

### How it works

1. **Planner runs first.** It receives the query and any prior context, and outputs a structured plan with a `steps` array.
2. **For each step**, the `body` node executes. The current step object is written to `current_step` in state, so the researcher's prompt can reference `{{ current_step }}`.
3. **After each step**, the `evaluator` (reflector) runs. It sees the accumulated observations and decides:
   - **CONTINUE** -- proceed to the next step.
   - **ADJUST** -- the current plan is inadequate; trigger replanning. The planner runs again with all context so far, producing a new set of steps. This counts toward `max_replan_cycles`.
   - **COMPLETE** -- enough evidence has been gathered; exit the loop early.
4. **Iteration guards** ensure the loop terminates:
   - `min_iterations` prevents premature completion (the reflector cannot say COMPLETE before this many steps).
   - `max_iterations` is the hard ceiling.
   - `max_replan_cycles` caps how many times replanning can happen.

### Planner guidance

For workflows with non-standard tools (e.g. enterprise-only, no web search), you can inject free-text guidance into the planner's prompt:

```yaml
config:
  planner_guidance: |
    Available tools for the researcher:
    - genie: Query enterprise data warehouse (financial data, KPIs)
    - vector_search: Semantic search over internal documents
    There is NO web search available. Design steps for these sources only.
  planner:
    subtype: planner
    model_tier: analytical
    output_key: plan
  # ... rest of config
```

---

## Adding Parallel Research

The `parallel` node runs all its children concurrently. This is useful for querying multiple independent sources at once.

```yaml
- id: parallel_research
  type: parallel
  label: Parallel Researchers
  children:
    - id: researcher_web
      type: agent
      label: Web Researcher
      config:
        subtype: researcher
        model_tier: analytical
        output_key: web_findings
        tools: [web_search, web_crawl]
        pool_writes:
          - pool: observations
            extract: web_findings
          - pool: sources
            extract: sources
        max_tool_calls: 5

    - id: researcher_enterprise
      type: agent
      label: Enterprise Researcher
      config:
        subtype: researcher
        model_tier: analytical
        output_key: enterprise_findings
        tools: [vector_search, knowledge_assistant]
        pool_writes:
          - pool: observations
            extract: enterprise_findings
          - pool: sources
            extract: sources
        max_tool_calls: 5
```

Rules for `parallel` nodes:

- Each child **must** have a unique `output_key`. The validator rejects duplicate output keys among siblings.
- All children write to the **same shared pools**, so observations from both researchers are available to the synthesizer.
- Children execute as concurrent `asyncio` tasks. The parallel node completes when all children finish.

---

## Conditional Branching

The `conditional` node picks one of its children to execute based on state conditions. This lets you route simple queries to a fast path and complex queries to full research.

```yaml
- id: branch
  type: conditional
  label: Complexity Branch
  config:
    conditions:
      - key: coordination.is_simple
        operator: eq
        value: true
    default_branch: 1
  children:
    # Branch 0: Simple query -- direct synthesis
    - id: simple_synth
      type: agent
      label: Simple Synthesizer
      config:
        subtype: synthesizer
        model_tier: simple
        output_key: report
        max_tool_calls: 0

    # Branch 1 (default): Complex query -- full research
    - id: full_research
      type: sequence
      label: Full Research Path
      children:
        - id: background
          type: agent
          label: Background Investigator
          config:
            subtype: background
            model_tier: simple
            output_key: background
            tools: [web_search]
            max_tool_calls: 3

        # ... research cycle, synthesizer, etc.
```

The `conditions` list is evaluated in order. Each condition specifies:

| Field | Description |
|-------|-------------|
| `key` | Dot-path into state (e.g. `coordination.is_simple`). |
| `operator` | Comparison: `eq`, `neq`, `gt`, `lt`, `gte`, `lte`, `contains`, `in`, `exists`, `not_exists`. |
| `value` | Value to compare against. |

If the first condition matches, branch 0 (the first child) runs. If no condition matches, `default_branch` selects the fallback child (index-based, starting at 0).

A `conditional` node must have **at least 2 children**.

---

## Custom Prompts

Every agent node uses default prompts from its subtype, but you can override them with `system_prompt` and `user_prompt_template`:

```yaml
config:
  subtype: researcher
  model_tier: analytical
  system_prompt: |
    You are a specialized {{ domain }} researcher.
    Focus on finding quantitative data and primary sources.
    Always cite your sources with URLs.
  user_prompt_template: |
    Research step: {{ current_step }}
    Query: {{ query }}
    Previous findings: {{ observations }}
```

### Template variables

Templates use Jinja2 syntax. Available variables include:

- **`query`** -- always available (from `required_inputs`).
- **Any state key** -- anything written by a previous node's `output_key` (e.g. `coordination`, `background`, `plan`, `findings`).
- **Pool contents** -- injected via `pool_inject` as formatted text.
- **`current_step`** -- inside `plan_and_execute` body nodes, the current step object.
- **Custom inputs** -- anything passed in `initial_state`.

Variables that are missing from state are rendered as empty strings, not errors.

---

## Pool Configuration

### Writing to pools (`pool_writes`)

After an agent executes, `pool_writes` extracts data from its output and adds it to the named pool:

```yaml
config:
  subtype: researcher
  output_key: findings
  pool_writes:
    - pool: observations
      extract: findings       # dot-path into the agent's output
    - pool: sources
      extract: sources
```

| Field | Description |
|-------|-------------|
| `pool` | Name of the target pool (must be declared in top-level `pools`). |
| `extract` | Dot-path expression on the agent's output to extract items. |
| `transform` | (Optional) Jinja template to transform each extracted item. |

### Reading from pools (`pool_inject`)

Before an agent executes, `pool_inject` retrieves items from pools and injects them into the prompt:

```yaml
config:
  subtype: synthesizer
  pool_inject:
    - pool: observations
      threshold: 0            # BM25 relevance threshold (0 = all items)
      format: text            # text, json, or markdown
      max_items: 20           # max items to inject
    - pool: sources
      threshold: 0
```

| Field | Default | Description |
|-------|---------|-------------|
| `pool` | (required) | Name of the source pool. |
| `threshold` | `0.0` | BM25/relevance threshold. `0` means inject all items. |
| `format` | `text` | Serialization format: `text`, `json`, `markdown`. |
| `max_items` | `20` | Cap on injected items. |
| `max_item_chars` | `0` | Per-item character truncation (`0` = unlimited). |

### Pool tools (`pool_tools`)

Some agents benefit from being able to _search_ pools interactively (rather than getting a bulk injection). The `pool_tools` field gives the agent search tools for the named pools:

```yaml
config:
  subtype: synthesizer
  pool_tools:
    - observations
    - sources
  max_tool_calls: 10
```

This registers `pool_search` and `pool_get_recent` tools scoped to the listed pools, so the LLM can query them during its ReAct loop.

---

## Error Handling

Every node supports an optional `error_handling` block:

```yaml
- id: enrichment
  type: agent
  label: Optional Enrichment
  config:
    subtype: researcher
    model_tier: simple
    output_key: enrichment
    tools: [web_search]
    max_tool_calls: 3
  error_handling:
    on_error: skip           # skip this node on failure
    max_retries: 0
```

| Field | Default | Description |
|-------|---------|-------------|
| `on_error` | `fail` | Policy: `fail` (propagate exception), `skip` (emit `NodeSkippedEvent` and continue), `retry` (retry with exponential back-off). |
| `max_retries` | `2` | Maximum retry attempts (only used when `on_error: retry`). |
| `retry_delay_seconds` | `1.0` | Base delay between retries in seconds (doubles each attempt). |

Use `skip` for optional enrichment steps that should not abort the entire workflow. Use `retry` for transient failures (rate limits, network errors).

---

## Complete Example: Full Research Pipeline

This example combines all the patterns above into a production-grade research workflow:

```yaml
# Full Research Pipeline
#
# 1. Coordinator classifies the query
# 2. Background investigator gathers initial context
# 3. Plan-and-execute cycle: planner creates steps, researcher executes
#    each step, reflector evaluates progress (continue/replan/complete)
# 4. Synthesizer generates the final report
#
# Usage:
#   from databricks_deep_research import WorkflowRunner
#
#   runner = WorkflowRunner.from_databricks()
#   result = await runner.run("full_research.yaml", query="Your question here")
#   print(result.output)

id: full_research
name: Full Research Pipeline
version: 1
required_inputs: [query]
output_keys: [report]

# -- Shared data pools --------------------------------------------------------
pools:
  - name: sources            # URLs and metadata from search results
    dedup_key: url
    max_items: 200
  - name: observations       # Key findings extracted by researchers
    dedup_key: content_hash
    max_items: 100

# -- Node tree -----------------------------------------------------------------
root:
  id: main
  type: sequence
  label: Main Pipeline
  children:

    # Step 1: Classify the query complexity and intent
    - id: coordinator
      type: agent
      label: Query Classifier
      config:
        subtype: coordinator
        model_tier: simple         # fast model -- classification is easy
        output_key: coordination

    # Step 2: Quick background investigation
    - id: background
      type: agent
      label: Background Investigator
      config:
        subtype: background
        model_tier: simple
        output_key: background
        tools: [web_search]
        max_tool_calls: 5
      error_handling:
        on_error: skip             # non-critical -- continue without background

    # Step 3: Research cycle (plan -> execute -> evaluate)
    - id: research_cycle
      type: plan_and_execute
      label: Research Cycle
      config:
        planner:
          subtype: planner
          model_tier: analytical
          output_key: plan
        items_path: steps
        item_state_key: current_step
        body:
          id: researcher
          type: agent
          label: Researcher
          config:
            subtype: researcher
            model_tier: analytical
            output_key: findings
            tools: [web_search, web_crawl]
            pool_writes:
              - pool: observations
                extract: findings
              - pool: sources
                extract: sources
            max_tool_calls: 15
        evaluator:
          subtype: reflector
          model_tier: analytical
          output_key: evaluation
          pool_inject:
            - pool: observations
              threshold: 0
        max_iterations: 10
        min_iterations: 2
        max_replan_cycles: 3

    # Step 4: Generate final report from all accumulated evidence
    - id: synthesizer
      type: agent
      label: Report Synthesizer
      config:
        subtype: synthesizer
        model_tier: complex        # strongest model for final output
        output_key: report
        pool_inject:
          - pool: observations
            threshold: 0
          - pool: sources
            threshold: 0
        pool_tools:                # let synthesizer search pools interactively
          - observations
          - sources
        max_tool_calls: 10
```

---

## Complete Example: Enterprise-Only Research

This workflow uses only enterprise data sources -- no web search at all. The `planner_guidance` field tells the planner what tools are available so it generates appropriate research steps.

```yaml
id: enterprise_research
name: Enterprise Research Pipeline
description: Research using only enterprise data sources (no web search)
version: 1
required_inputs: [query]
output_keys: [report]

# -- Enterprise tool declarations -----------------------------------------------
tools:
  - name: genie
    kind: genie
    config:
      space_id: 01ef8d7a-0000-0000-0000-000000000000
    description: "Enterprise data warehouse -- financial data, operational metrics, KPIs"

  - name: vector_search
    kind: vector_search
    config:
      index_name: prod_catalog.docs.internal_docs_idx
      num_results: 10
    description: "Internal documents -- architecture reviews, technical specs, policies"

  - name: knowledge_assistant
    kind: knowledge_assistant
    config:
      endpoint_name: internal-docs-assistant
    description: "AI-powered Q&A on internal documentation and runbooks"

# -- Pools (smaller caps for enterprise data) -----------------------------------
pools:
  - name: sources
    dedup_key: url
    max_items: 50
  - name: observations
    dedup_content_hash: true
    max_items: 30

# -- Node tree -------------------------------------------------------------------
root:
  id: main
  type: sequence
  label: Enterprise Research Pipeline
  children:
    # Step 1: Classify the query
    - id: coordinator
      type: agent
      label: Query Classifier
      config:
        subtype: coordinator
        model_tier: simple
        output_key: coordination

    # Step 2: Research cycle with enterprise tools only
    - id: research_cycle
      type: plan_and_execute
      label: Enterprise Research Cycle
      config:
        planner_guidance: |
          Available tools for the researcher:
          - genie: Query enterprise data warehouse using natural language (financial data, operational metrics, KPIs)
          - vector_search: Semantic search over internal documents (architecture reviews, technical specs, policies)
          - knowledge_assistant: AI-powered Q&A on internal documentation and runbooks
          There is NO web search available. Design all research steps for these enterprise data sources.
          For financial queries, plan steps that ask specific questions to the genie tool.
          For technical/documentation queries, plan steps that search the vector index or ask the knowledge assistant.
        planner:
          subtype: planner
          model_tier: analytical
          output_key: plan
        items_path: steps
        item_state_key: current_step
        body:
          id: researcher
          type: agent
          label: Enterprise Researcher
          config:
            subtype: researcher
            model_tier: analytical
            output_key: findings
            tools: [genie, vector_search, knowledge_assistant]
            pool_writes:
              - pool: observations
                extract: findings
              - pool: sources
                extract: sources
            max_tool_calls: 8
        evaluator:
          subtype: reflector
          model_tier: analytical
          output_key: evaluation
          pool_inject:
            - pool: observations
              threshold: 0
        max_iterations: 3
        min_iterations: 1
        max_replan_cycles: 1

    # Step 3: Synthesize final report
    - id: synthesizer
      type: agent
      label: Report Synthesizer
      config:
        subtype: synthesizer
        model_tier: analytical
        output_key: report
        pool_inject:
          - pool: observations
            threshold: 0
          - pool: sources
            threshold: 0
        max_tool_calls: 0
```

---

## Validation

The framework validates your YAML at load time, before any execution begins. Validation catches:

- **Missing required fields** -- every node must have `id`, `type`, and `label`. Agent nodes must have `subtype` in their config.
- **Unknown node types** -- the `type` field must be one of: `agent`, `tool`, `sequence`, `parallel`, `loop`, `conditional`, `subworkflow`, `plan_and_execute`.
- **Structural violations** -- leaf nodes (`agent`, `tool`, `subworkflow`) must not have `children`. Composite nodes (`sequence`, `parallel`, `loop`) must have at least one child. `conditional` must have at least two children.
- **Duplicate node IDs** -- every `id` across the entire tree must be unique.
- **Duplicate output keys in parallel** -- children of a `parallel` node cannot share the same `output_key`.
- **Pool write mismatches** -- warns when `pool_writes.extract` does not match the agent's `output_key` for text/markdown output formats.
- **Empty required_inputs or output_keys** -- the top-level lists must be non-empty.

### Debugging validation errors

Validation errors are raised as `WorkflowValidationError` with a list of human-readable messages:

```python
from databricks_deep_research import load_workflow, WorkflowValidationError

try:
    definition = load_workflow("my_workflow.yaml")
except WorkflowValidationError as e:
    for error in e.errors:
        print(f"  - {error}")
```

You can also validate a YAML string without a file:

```python
from databricks_deep_research import load_workflow_from_string

definition = load_workflow_from_string(yaml_text)
```

---

## Tips and Gotchas

1. **Define tools before referencing them.** Agent nodes reference tools by name (e.g. `tools: [web_search]`). If you use custom tools, declare them in the top-level `tools:` section. Built-in kinds (`web_search`, `web_crawl`) are auto-registered even without an explicit declaration.

2. **Pool names must match.** The `pool` field in `pool_writes` and `pool_inject` must exactly match a name in the top-level `pools:` list. Typos fail silently at runtime (the pool just stays empty).

3. **Template variables must exist in state or be provided as inputs.** Missing variables render as empty strings rather than raising errors, which can produce confusing agent behavior. Use `required_inputs` to document what the workflow needs.

4. **Use `error_handling: skip` for optional enrichment steps.** A failing background investigator should not abort the entire pipeline.

5. **Use `required_inputs` to document workflow inputs.** This is both documentation and a runtime guard -- the executor will refuse to start if any required input is missing from state.

6. **`model_tier` controls cost and capability.** Use `simple` for classification and quick tasks, `analytical` for research and reflection, `complex` for final synthesis. The actual model behind each tier is set when you create the `FrameworkLLMClient` or via the YAML `models:` section. See [Model Configuration](model-configuration.md) for details.

7. **`max_tool_calls: 0` disables tools.** Set this on synthesizer nodes that should only use injected pool data, not make new tool calls.

8. **Keep `max_items` reasonable.** Large pools consume prompt tokens. Start with 50-100 for sources and observations. The synthesizer can use `pool_tools` to search instead of injecting everything.

9. **`output_key` determines state writes.** Every agent writes its output to `state[output_key]`. Later agents can read it via `{{ output_key }}` in templates. Choose descriptive, unique keys.

10. **Order matters in sequences.** Each child runs after the previous one completes. An agent can only reference state keys written by nodes that ran before it.

---

## See Also

- [Workflow Engine](../concepts/workflow-engine.md) -- how the executor walks the tree
- [Node Types Reference](../reference/node-types-reference.md) -- all 8 node types with full config schemas
- [Workflow Definition Schema](../reference/workflow-definition-schema.md) -- complete field reference
- [Builtin Agents](builtin-agents.md) -- coordinator, planner, researcher, reflector, synthesizer subtypes
- [Example Walkthroughs](../examples/walkthrough-simple-research.md) -- step-by-step walkthrough of the example YAMLs
