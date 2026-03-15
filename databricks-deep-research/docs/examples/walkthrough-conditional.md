# Walkthrough: Conditional Branching

> Route queries to different pipelines based on complexity classification.

## Overview

This walkthrough shows how `conditional` nodes let you branch workflow execution based on state values -- e.g., routing simple queries to a direct answer and complex queries to full research.

`conditional_research.yaml` lives at `examples/conditional_research.yaml` in the framework package. It builds on the same four-stage pattern from the simple research walkthrough but adds a branching layer between the coordinator and the rest of the pipeline. The coordinator classifies the incoming query, a conditional node inspects the classification, and execution forks:

- **Simple path** -- a single synthesizer agent produces an answer immediately (no research, no tools).
- **Complex path** -- background investigation, a plan-and-execute research cycle, and a final synthesizer.

This pattern is useful whenever you want to avoid spending time and tokens on a full research loop for questions the LLM can answer directly.

---

## The Complete YAML

```yaml
id: conditional_research
name: Conditional Research Pipeline
description: Conditional branching — simple queries get direct answers, complex ones get full research
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
  label: Conditional Research Pipeline
  children:
    # Step 1: Classify the query
    - id: coordinator
      type: agent
      label: Query Classifier
      config:
        subtype: coordinator
        model_tier: simple
        output_key: coordination

    # Step 2: Branch based on coordinator classification
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
        # Branch 0: Simple query — direct synthesis (no research)
        - id: simple_synth
          type: agent
          label: Simple Synthesizer
          config:
            subtype: synthesizer
            model_tier: simple
            output_key: report
            max_tool_calls: 0

        # Branch 1 (default): Complex query — full research
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
                    max_tool_calls: 8
                evaluator:
                  subtype: reflector
                  model_tier: analytical
                  output_key: evaluation
                max_iterations: 3
                min_iterations: 1
                max_replan_cycles: 1

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

## How Conditional Nodes Work

The conditional node is the routing mechanism at the center of this workflow. Here is how it operates at execution time:

1. **Coordinator classifies the query first.** The coordinator agent runs before the conditional node and writes a structured classification (including an `is_simple` boolean) to `state["coordination"]`.
2. **Conditional node evaluates state conditions in order.** Each entry in the `conditions` list is checked sequentially against the current workflow state. The framework resolves dot-paths like `coordination.is_simple` by walking into nested dicts.
3. **First matching condition selects the branch.** Conditions map to children by index -- if condition 0 matches, child 0 (the simple synthesizer) executes. If condition 1 matched, child 1 would execute, and so on.
4. **If no condition matches, `default_branch` is used.** In this workflow, `default_branch: 1` means the complex research path runs when the query is not classified as simple. This is the safe default: when in doubt, do the full research.

The conditional node emits a `BranchSelectedEvent` before executing the chosen child, so you can observe which path was taken in the event stream.

---

## Section-by-Section Annotation

### Metadata

```yaml
id: conditional_research
name: Conditional Research Pipeline
description: Conditional branching — simple queries get direct answers, complex ones get full research
version: 1
required_inputs: [query]
output_keys: [report]
```

| Field | Purpose |
|-------|---------|
| `id` | Machine-readable identifier used in logging and event correlation. |
| `name` | Human-readable label shown in UIs and trace output. |
| `description` | Free-text summary of what makes this workflow different from the simple pipeline. |
| `version` | Integer version. Bump when the workflow structure changes. |
| `required_inputs` | The executor raises a validation error if `query` is missing from the initial state. |
| `output_keys` | Both branches write to `report`, so the workflow always produces this key on success. |

---

### Pools

```yaml
pools:
  - name: sources
    dedup_key: url
    max_items: 100
  - name: observations
    dedup_content_hash: true
    max_items: 50
```

Pools are shared, append-only collections that pass data between agents. Only the complex path writes to or reads from these pools -- the simple path bypasses them entirely.

- **`sources`** -- every web page or document the researcher encounters. Deduplication by `url` prevents the same page from appearing twice.
- **`observations`** -- distilled findings extracted by the researcher. `dedup_content_hash: true` collapses identical observations across overlapping searches.

The caps (100 sources, 50 observations) are lower than the simple research walkthrough because this workflow has tighter iteration limits (`max_iterations: 3`), so fewer items accumulate.

---

### Root Sequence

```yaml
root:
  id: main
  type: sequence
  label: Conditional Research Pipeline
  children:
    - ...  # coordinator
    - ...  # branch (conditional)
```

The root is a **sequence** with two children: the coordinator and the conditional branch. The coordinator always runs first. Then the conditional node routes to one of its two children based on the classification.

---

### Step 1: Coordinator

```yaml
- id: coordinator
  type: agent
  label: Query Classifier
  config:
    subtype: coordinator
    model_tier: simple
    output_key: coordination
```

**What it does:** Classifies the incoming query as simple or complex. The builtin coordinator prompt asks the LLM to return a structured JSON object that includes an `is_simple` boolean, complexity level, and domain tags.

**Key details:**

- **`model_tier: simple`** -- classification does not need a reasoning model; the cheapest tier is sufficient.
- **`output_key: coordination`** -- the classification result is written to `state["coordination"]`. The conditional node reads from this key in the next step.

**Events emitted:** `node_started`, `coordinator_classified`, `node_completed`.

---

### Step 2: Conditional Branch

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
    - ...  # Branch 0: simple_synth
    - ...  # Branch 1: full_research (default)
```

This is the key node in the workflow. It has one condition and two children:

| Condition index | Condition | Selects child |
|----------------|-----------|---------------|
| 0 | `coordination.is_simple == true` | Branch 0 (`simple_synth`) |
| *(no match)* | `default_branch: 1` | Branch 1 (`full_research`) |

The `default_branch: 1` setting means the complex path is the fallback. If the coordinator fails to produce an `is_simple` field, or if the value is `false`, the workflow defaults to full research. This is intentional -- it is safer to over-research than to skip research on a question that needed it.

**Events emitted:** `node_started`, `branch_selected` (includes which branch index was chosen), `node_completed`.

---

### Branch 0: Simple Synthesizer

```yaml
- id: simple_synth
  type: agent
  label: Simple Synthesizer
  config:
    subtype: synthesizer
    model_tier: simple
    output_key: report
    max_tool_calls: 0
```

**What it does:** Generates a direct answer from the LLM's training knowledge, with no research step and no tool calls.

**Key details:**

- **`model_tier: simple`** -- the fastest, cheapest tier. Simple questions do not need a powerful model.
- **`max_tool_calls: 0`** -- no tools, no web search, no pool access. The synthesizer answers purely from context.
- **`output_key: report`** -- writes to the same key as the complex path's synthesizer, so downstream consumers see a uniform output regardless of which branch ran.

This branch typically completes in under 2 seconds -- the entire cost of the workflow is the coordinator call plus this single synthesis call.

---

### Branch 1: Full Research Path

```yaml
- id: full_research
  type: sequence
  label: Full Research Path
  children:
    - id: background
      ...
    - id: research_cycle
      ...
    - id: synthesizer
      ...
```

The complex path is a three-step **sequence** that mirrors the structure from the simple research walkthrough: background investigation, plan-and-execute research cycle, and final synthesis.

#### Background

```yaml
- id: background
  type: agent
  label: Background Investigator
  config:
    subtype: background
    model_tier: simple
    output_key: background
    tools: [web_search]
    max_tool_calls: 3
```

A quick, broad web search to establish baseline context. With `max_tool_calls: 3`, the background agent runs a few searches and finishes fast, giving the planner something to work with.

#### Research Cycle

```yaml
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
        max_tool_calls: 8
    evaluator:
      subtype: reflector
      model_tier: analytical
      output_key: evaluation
    max_iterations: 3
    min_iterations: 1
    max_replan_cycles: 1
```

The plan-and-execute node runs the core research loop: the planner creates steps, the researcher executes each step with web search and crawl, and the reflector evaluates progress after each step.

| Limit | Value | Rationale |
|-------|-------|-----------|
| `max_iterations: 3` | Tighter than a standalone deep research workflow. Since simple queries are already filtered out, complex queries still get thorough treatment in 3 focused steps. |
| `min_iterations: 1` | The reflector can declare COMPLETE after the first step if evidence is sufficient. |
| `max_replan_cycles: 1` | One replan is allowed if the initial plan proves inadequate. |
| `max_tool_calls: 8` | Per-step tool budget. Enough for several searches and targeted crawls. |

#### Final Synthesizer

```yaml
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

The synthesizer receives all collected observations and sources via `pool_inject` (threshold 0 = include everything) and produces the final report. `max_tool_calls: 0` means it writes only -- no additional searches during synthesis.

---

## Condition Operators

The `conditions` list in a conditional node uses `StateCondition` entries. Each condition has a `key` (dot-path into state), an `operator`, and a `value`. Here is a quick reference of supported operators:

| Operator | Description | Example |
|----------|-------------|---------|
| `eq` | Equal | `operator: eq, value: true` |
| `neq` | Not equal | `operator: neq, value: "skip"` |
| `gt` | Greater than | `operator: gt, value: 5` |
| `gte` | Greater than or equal | `operator: gte, value: 3` |
| `lt` | Less than | `operator: lt, value: 10` |
| `lte` | Less than or equal | `operator: lte, value: 100` |
| `in` | Value is a member of the list | `operator: in, value: [complex, analytical]` |
| `contains` | Left side contains value on the right | `operator: contains, value: "error"` |
| `exists` | Key exists in state (no `value` needed) | `operator: exists` |
| `not_exists` | Key does not exist in state | `operator: not_exists` |

You can also use **composite conditions** (`all`, `any`, `not`) and **LLM conditions** for more advanced branching. See the [Conditions and Branching](../guides/conditions-and-branching.md) guide for full details.

---

## Expected Event Stream

### Simple Path (e.g., "What is the capital of France?")

```
workflow_started
node_started (main)
node_started (coordinator)
coordinator_classified {is_simple: true, complexity: "simple"}
node_completed (coordinator)
node_started (branch)
branch_selected {branch_index: 0, condition: "coordination.is_simple == true"}
node_started (simple_synth)
agent_stream_chunk {chunk: "The capital of France is Paris..."}
agent_output {key: "report"}
node_completed (simple_synth)
node_completed (branch)
node_completed (main)
workflow_completed {total_tokens: ~800}
```

The simple path completes in two agent calls (coordinator + synthesizer) and typically finishes in 1--3 seconds.

### Complex Path (e.g., "What are the economic implications of quantum computing on global supply chains?")

```
workflow_started
node_started (main)
node_started (coordinator)
coordinator_classified {is_simple: false, complexity: "complex"}
node_completed (coordinator)
node_started (branch)
branch_selected {branch_index: 1, condition: "default_branch"}
node_started (full_research)
node_started (background)
tool_call {web_search}
tool_result {3 sources}
background_completed
node_completed (background)
node_started (research_cycle)
plan_created {steps: ["Step 1: ...", "Step 2: ...", "Step 3: ..."]}
item_started {step: 1}
tool_call {web_search}
tool_result {...}
tool_call {web_crawl}
tool_result {...}
item_completed {step: 1}
evaluation_decision {CONTINUE}
item_started {step: 2}
tool_call {web_search}
tool_result {...}
item_completed {step: 2}
evaluation_decision {COMPLETE}
plan_and_execute_exit {reason: "evaluator_complete", items_processed: 2}
node_completed (research_cycle)
node_started (synthesizer)
agent_stream_chunk {chunk: "# Economic Implications of Quantum Computing\n\n..."}
agent_output {key: "report"}
node_completed (synthesizer)
node_completed (full_research)
node_completed (branch)
node_completed (main)
workflow_completed {total_tokens: ~12000}
```

The complex path runs 5+ agent calls (coordinator, background, planner, researchers, reflectors, synthesizer) and typically takes 30--90 seconds depending on query difficulty and model latency.

---

## Running It

```python
import asyncio
from openai import AsyncOpenAI
from databricks_deep_research import WorkflowRunner, load_workflow

async def main():
    workflow = load_workflow("examples/conditional_research.yaml")
    runner = WorkflowRunner(
        workflow,
        AsyncOpenAI(),
        model_mapping={
            "simple": "gpt-4o-mini",
            "analytical": "gpt-4o",
            "complex": "gpt-4o",
        },
    )

    # Simple query — takes the fast path
    result = await runner.run(query="What is the capital of France?")
    print(result.state["report"])

    # Complex query — takes the full research path
    result = await runner.run(
        query="What are the economic implications of quantum computing on global supply chains?"
    )
    print(result.state["report"])

asyncio.run(main())
```

The `model_mapping` dict maps the `model_tier` values used in the YAML (`simple`, `analytical`, `complex`) to actual model identifiers your OpenAI-compatible endpoint understands. The conditional branching is fully transparent to the caller -- both queries return a `report` key regardless of which path executed.

---

## See Also

- [Conditions and Branching](../guides/conditions-and-branching.md)
- [Node Types Reference - conditional](../reference/node-types-reference.md#6-conditional)
- [YAML Workflow Authoring](../guides/yaml-workflow-authoring.md)
