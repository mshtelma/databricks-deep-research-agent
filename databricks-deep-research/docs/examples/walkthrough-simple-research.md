# Walkthrough: Simple Research Pipeline

> Line-by-line walkthrough of the primary example workflow.

## Overview

`simple_research.yaml` is the canonical example showing the full research pipeline: coordinator, background, plan-and-execute, synthesizer. It lives at `examples/simple_research.yaml` in the framework package and demonstrates every major node type you will use in a typical deep-research workflow.

The pipeline follows a four-stage pattern:

1. **Coordinator** -- classify the incoming query so downstream agents know how to handle it.
2. **Background** -- fire off a quick web search to build initial context before the heavy research loop.
3. **Plan-and-Execute** -- the core research cycle where a planner creates steps, a researcher executes each step, and a reflector decides whether to continue, adjust, or complete.
4. **Synthesizer** -- consume every observation and source gathered so far and produce the final report.

---

## The Complete YAML

```yaml
# Simple Deep Research Workflow
#
# Demonstrates the full research pipeline:
# 1. Coordinator classifies the query
# 2. Background investigator gathers initial context
# 3. Plan-and-execute cycle: planner creates steps, researcher executes
#    each step, reflector evaluates progress (continue/replan/complete)
# 4. Synthesizer generates the final report
#
# Usage:
#   from databricks_deep_research import run_workflow_from_yaml
#
#   state, events = await run_workflow_from_yaml(
#       "examples/simple_research.yaml",
#       openai_client=client,
#       model_mapping={"simple": "gpt-4o-mini", "analytical": "gpt-4o", "complex": "gpt-4o"},
#       initial_state={"query": "What are the latest advances in quantum computing?"},
#   )
#   print(state.get("report"))

id: simple_research
name: Simple Research Pipeline
version: 1
required_inputs: [query]
output_keys: [report]

pools:
  - name: sources
    dedup_key: url
    max_items: 200
  - name: observations
    dedup_key: content_hash
    max_items: 100

root:
  id: main
  type: sequence
  label: Main Pipeline
  children:
    # Step 1: Classify the query
    - id: coordinator
      type: agent
      label: Query Classifier
      config:
        subtype: coordinator
        model_tier: simple
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

    # Step 4: Generate final report
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
        pool_tools:
          - observations
          - sources
        max_tool_calls: 10
```

---

## Section-by-Section Annotation

### Metadata

```yaml
id: simple_research
name: Simple Research Pipeline
version: 1
required_inputs: [query]
output_keys: [report]
```

| Field | Purpose |
|-------|---------|
| `id` | Machine-readable identifier. Used in logging, tracing, and event correlation. |
| `name` | Human-readable label shown in UIs and log output. |
| `version` | Integer version. Bump this when the workflow structure changes so consumers can detect incompatibilities. |
| `required_inputs` | State keys that **must** be present before execution starts. The executor raises a validation error if `query` is missing from the initial state. |
| `output_keys` | State keys the workflow guarantees to produce. After a successful run, `state.get("report")` is always populated. |

---

### Pools

```yaml
pools:
  - name: sources
    dedup_key: url
    max_items: 200
  - name: observations
    dedup_key: content_hash
    max_items: 100
```

Pools are **shared, append-only collections** that any node in the workflow can write to or read from. They serve as the primary coordination mechanism between agents that do not share direct state.

**`sources` pool**

- Holds every web page, document, or API result the researcher encounters.
- **Dedup by `url`**: if two research steps find the same page, it is stored only once.
- **Cap at 200 items**: prevents unbounded growth during long research cycles.

**`observations` pool**

- Holds distilled findings: facts, quotes, and data points extracted by the researcher.
- **Dedup by `content_hash`**: identical observations from overlapping searches are collapsed.
- **Cap at 100 items**: keeps the synthesizer's context window manageable.

Pools support hybrid BM25 + vector search at query time, so the synthesizer can pull the most relevant subset of observations rather than dumping everything into the prompt.

---

### Root Sequence

```yaml
root:
  id: main
  type: sequence
  label: Main Pipeline
  children:
    - ...  # 4 children
```

The root node is a **sequence** -- it executes its children one after another, top to bottom. Each child runs to completion before the next starts. The four children form the complete research pipeline.

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

**What it does:** The coordinator is the first agent to see the user's query. It classifies the query's complexity (simple, analytical, or complex), identifies the domain, and decides which research depth the pipeline should use.

**Key details:**

- **`subtype: coordinator`** -- loads the builtin coordinator prompt, which asks the LLM to return a structured JSON classification.
- **`model_tier: simple`** -- uses the cheapest/fastest model tier. Classification does not need a reasoning model.
- **`output_key: coordination`** -- the classification result (complexity level, domain tags, suggested depth) is written to `state["coordination"]`. Downstream agents read this key to adapt their behavior.

**Events emitted:** `node_started`, `coordinator_classified` (contains the complexity classification), `node_completed`.

---

### Step 2: Background

```yaml
- id: background
  type: agent
  label: Background Investigator
  config:
    subtype: background
    model_tier: simple
    output_key: background
    tools: [web_search]
    max_tool_calls: 5
```

**What it does:** Before the heavy research loop begins, the background agent runs a quick, broad web search to establish baseline context. This gives the planner something to work with when creating the research plan.

**Key details:**

- **`subtype: background`** -- uses the builtin background prompt, tuned for breadth over depth.
- **`tools: [web_search]`** -- only web search is enabled; no crawling at this stage to keep it fast.
- **`max_tool_calls: 5`** -- hard cap on tool invocations. The background step should finish in a few seconds, not minutes.
- **`output_key: background`** -- the gathered context is written to `state["background"]`.

The background agent's results (sources and observations) flow into the pools, seeding them for the research cycle.

**Events emitted:** `node_started`, `tool_call` (web_search), `tool_result`, `background_completed`, `node_completed`.

---

### Step 3: Plan-and-Execute

```yaml
- id: research_cycle
  type: plan_and_execute
  label: Research Cycle
  config:
    planner: ...
    body: ...
    evaluator: ...
    max_iterations: 10
    min_iterations: 2
    max_replan_cycles: 3
```

This is the heart of the workflow. The `plan_and_execute` node type implements a loop with three phases per iteration: **plan**, **execute**, **evaluate**.

#### Planner

```yaml
planner:
  subtype: planner
  model_tier: analytical
  output_key: plan
items_path: steps
item_state_key: current_step
```

- **`subtype: planner`** -- uses the builtin planner prompt. The LLM receives the query, coordination result, and background context, then outputs a structured plan.
- **`model_tier: analytical`** -- needs a more capable model than the coordinator since it must reason about what information is still missing.
- **`output_key: plan`** -- the plan object (with a `steps` array) is written to `state["plan"]`.
- **`items_path: steps`** -- tells the executor to iterate over `plan.steps`.
- **`item_state_key: current_step`** -- each iteration writes the current step to `state["current_step"]` so the body agent knows what to research.

#### Body (Researcher)

```yaml
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
```

The body runs **once per plan step**. Each iteration:

1. The researcher reads `state["current_step"]` to know what to investigate.
2. It uses `web_search` and `web_crawl` to gather information (up to 15 tool calls).
3. Findings are written to `state["findings"]` and also appended to the `observations` pool.
4. Discovered sources are appended to the `sources` pool.

The **`pool_writes`** configuration is how data flows from agent output into shared pools:
- `extract: findings` -- the executor extracts the `findings` field from the agent's structured output and appends each item to the `observations` pool.
- `extract: sources` -- same pattern for sources.

#### Evaluator (Reflector)

```yaml
evaluator:
  subtype: reflector
  model_tier: analytical
  output_key: evaluation
  pool_inject:
    - pool: observations
      threshold: 0
```

After each plan step completes, the reflector evaluates progress:

- **CONTINUE** -- proceed to the next step in the plan.
- **ADJUST** -- the plan needs modification (triggers a replan).
- **COMPLETE** -- enough information has been gathered; exit the loop early.

The reflector receives the full `observations` pool via `pool_inject` (threshold 0 means include all items). This lets it judge whether the accumulated evidence is sufficient.

#### Iteration Limits

```yaml
max_iterations: 10
min_iterations: 2
max_replan_cycles: 3
```

| Limit | Purpose |
|-------|---------|
| `min_iterations: 2` | The reflector cannot return COMPLETE until at least 2 research steps have run. Prevents premature exits on easy-looking queries. |
| `max_iterations: 10` | Hard ceiling. Even if the reflector keeps returning CONTINUE, the loop exits after 10 steps. |
| `max_replan_cycles: 3` | If the reflector returns ADJUST, the planner re-runs. This caps how many times the plan can be rewritten to prevent infinite replanning. |

---

### Step 4: Synthesizer

```yaml
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
    pool_tools:
      - observations
      - sources
    max_tool_calls: 10
```

**What it does:** Consumes all gathered observations and sources, then generates the final research report.

**Key details:**

- **`model_tier: complex`** -- uses the most capable (and typically most expensive) model tier. The synthesizer must produce a coherent, well-structured report from potentially dozens of observations, so reasoning ability matters.
- **`output_key: report`** -- the final report text is written to `state["report"]`, which matches the workflow's declared `output_keys`.
- **`pool_inject`** -- both `observations` and `sources` pools are injected into the agent's context at threshold 0 (include everything).
- **`pool_tools`** -- the synthesizer can also **search** the pools via tool calls. This is useful when the injected context exceeds the model's window: the agent can use BM25+vector search to pull specific observations on demand.
- **`max_tool_calls: 10`** -- allows up to 10 pool search queries during synthesis.

**Events emitted:** `node_started`, `synthesis_started`, `agent_stream_chunk` (streamed report tokens), `agent_output`, `node_completed`.

---

## Expected Event Stream

Below is the approximate sequence of events emitted during a typical run. The exact number of research iterations and tool calls varies by query.

```
workflow_started
node_started (coordinator)
coordinator_classified {complexity: "complex"}
node_completed (coordinator)
node_started (background)
tool_call {web_search}
tool_result {5 sources}
background_completed
node_completed (background)
node_started (plan_and_execute)
plan_created {steps: ["Step 1: ...", "Step 2: ...", ...]}
item_started {step: 1}
tool_call {web_search}
tool_result {...}
tool_call {web_crawl}
tool_result {...}
item_completed {step: 1}
reflection_decision {CONTINUE}
item_started {step: 2}
...
reflection_decision {COMPLETE}
plan_and_execute_exit
node_completed (plan_and_execute)
node_started (synthesizer)
agent_stream_chunk {chunk: "# Report Title\n\n..."}
...
node_completed (synthesizer)
workflow_completed {total_tokens: ~15000}
```

Every event carries a `node_id`, timestamp, and correlation metadata, so you can reconstruct a full trace of the pipeline after the fact.

---

## Running It

```python
from databricks_deep_research import WorkflowRunner, load_workflow
from openai import AsyncOpenAI

workflow = load_workflow("examples/simple_research.yaml")
runner = WorkflowRunner(workflow, AsyncOpenAI(), model_mapping={
    "simple": "gpt-4o-mini",
    "analytical": "gpt-4o",
    "complex": "gpt-4o",
})
result = await runner.run(query="What are the latest advances in quantum computing?")

print(result.state["report"])
```

The `model_mapping` dict maps the `model_tier` values used in the YAML (`simple`, `analytical`, `complex`) to actual model identifiers your OpenAI-compatible endpoint understands.

---

## Customizing

### 1. Change Iteration Limits

Tighten the research loop for faster results on simpler queries:

```yaml
# In the plan_and_execute config:
max_iterations: 5    # was 10
min_iterations: 1    # was 2
max_replan_cycles: 1 # was 3
```

### 2. Add a Pool

Add a `hypotheses` pool for the researcher to track working theories:

```yaml
pools:
  - name: sources
    dedup_key: url
    max_items: 200
  - name: observations
    dedup_key: content_hash
    max_items: 100
  - name: hypotheses          # new pool
    dedup_key: content_hash
    max_items: 50
```

Then reference it in the researcher's `pool_writes` and the synthesizer's `pool_inject`.

### 3. Change Model Tiers

Promote the background agent to the `analytical` tier for deeper initial context:

```yaml
- id: background
  type: agent
  label: Background Investigator
  config:
    subtype: background
    model_tier: analytical  # was "simple"
    output_key: background
    tools: [web_search]
    max_tool_calls: 5
```

Or downgrade the synthesizer to `analytical` if you want faster (cheaper) report generation at the cost of some quality:

```yaml
- id: synthesizer
  type: agent
  label: Report Synthesizer
  config:
    subtype: synthesizer
    model_tier: analytical  # was "complex"
    output_key: report
```

---

## See Also

- [YAML Workflow Authoring](../guides/yaml-workflow-authoring.md)
- [Builtin Agents](../guides/builtin-agents.md)
- [Quick Start](../getting-started/quickstart.md)
