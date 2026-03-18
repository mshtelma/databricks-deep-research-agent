# Walkthrough: Parallel Research

> Run multiple researchers concurrently to cover different aspects of a query.

## Overview

`parallel_research.yaml` demonstrates how to use `parallel` nodes to execute multiple research agents concurrently, each writing to shared pools. It lives at `examples/parallel_research.yaml` in the framework package.

The pipeline follows a three-stage pattern:

1. **Coordinator** -- classify the incoming query so downstream agents know how to handle it.
2. **Parallel Researchers** -- a web researcher and an enterprise researcher run concurrently, each contributing to the same shared pools.
3. **Synthesizer** -- consume every observation and source gathered by both researchers and produce the final report.

This pattern is useful when the query spans multiple information domains (public web + internal documents) and you want to reduce wall-clock time by running the searches in parallel rather than sequentially.

---

## The Complete YAML

```yaml
id: parallel_research
name: Parallel Research Pipeline
description: Two parallel researchers with shared pool, followed by synthesizer
version: 1
required_inputs: [query]
output_keys: [report]

tools:
  - name: web_search
    kind: web_search

  - name: web_crawl
    kind: web_crawl

  - name: vector_search
    kind: vector_search
    config:
      index_name: prod_catalog.docs.internal_docs_idx
      num_results: 10
    description: "Internal document search"

  - name: knowledge_assistant
    kind: knowledge_assistant
    config:
      endpoint_name: internal-docs-assistant
    description: "AI-powered Q&A on internal documentation"

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
  label: Parallel Research Pipeline
  children:
    # Step 1: Classify the query
    - id: coordinator
      type: agent
      label: Query Classifier
      config:
        subtype: coordinator
        model_tier: simple
        output_key: coordination

    # Step 2: Two researchers in parallel
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

    # Step 3: Synthesize from both research tracks
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

## How Parallel Execution Works

The `parallel` node type is the key to this workflow. When the executor reaches a `parallel` node, it launches all children concurrently using `asyncio.gather` and waits for every child to finish before moving to the next node in the parent sequence.

The concurrency model has four properties:

1. **Children run concurrently via asyncio** -- each child is scheduled as an independent coroutine. On a single event loop this means they interleave at every `await` point (tool calls, LLM requests), giving true concurrent I/O without threading.
2. **Shared pools are thread-safe** -- pools use `asyncio.Lock` internally, so multiple researchers can append items to the same pool without data races or lost writes.
3. **Events from all children are interleaved in the stream** -- the event stream contains events from both researchers mixed together. Each event carries a `node_id` field (`researcher_web` or `researcher_enterprise`) so consumers can separate them.
4. **All children must complete before the parent completes** -- the `parallel` node does not finish until every child has returned. If one child fails, the error propagates and the other children are cancelled.

---

## Section-by-Section Annotation

### Metadata

```yaml
id: parallel_research
name: Parallel Research Pipeline
description: Two parallel researchers with shared pool, followed by synthesizer
version: 1
required_inputs: [query]
output_keys: [report]
```

| Field | Purpose |
|-------|---------|
| `id` | Machine-readable identifier. Used in logging, tracing, and event correlation. |
| `name` | Human-readable label shown in UIs and log output. |
| `description` | Optional longer description of the workflow's purpose. |
| `version` | Integer version. Bump when the workflow structure changes. |
| `required_inputs` | State keys that **must** be present before execution starts. The executor raises a validation error if `query` is missing. |
| `output_keys` | State keys the workflow guarantees to produce. After a successful run, `state.get("report")` is always populated. |

---

### Tools

```yaml
tools:
  - name: web_search
    kind: web_search

  - name: web_crawl
    kind: web_crawl

  - name: vector_search
    kind: vector_search
    config:
      index_name: prod_catalog.docs.internal_docs_idx
      num_results: 10
    description: "Internal document search"

  - name: knowledge_assistant
    kind: knowledge_assistant
    config:
      endpoint_name: internal-docs-assistant
    description: "AI-powered Q&A on internal documentation"
```

This workflow declares four tools split across two categories:

**Web tools** (`web_search`, `web_crawl`) -- used by the web researcher. These are builtin tools that need no additional configuration.

**Enterprise tools** (`vector_search`, `knowledge_assistant`) -- used by the enterprise researcher. These require deployment-specific configuration:

| Tool | Config Key | Purpose |
|------|-----------|---------|
| `vector_search` | `index_name` | Points to a Databricks Vector Search index containing internal documents. `num_results` controls how many chunks are returned per query. |
| `knowledge_assistant` | `endpoint_name` | Points to a Databricks model serving endpoint that runs an AI assistant over internal documentation. |

Tools are referenced by `name` in each agent's `tools` list. This decouples tool definitions from agent configuration, making it easy to swap implementations without changing the workflow structure.

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

Pools are **shared, append-only collections** that serve as the coordination mechanism between the parallel researchers and the downstream synthesizer.

**`sources` pool**

- Holds every web page, document, or API result encountered by either researcher.
- **Dedup by `url`**: if both researchers find the same source, it is stored only once.
- **Cap at 100 items**: prevents unbounded growth when parallel agents are both actively searching.

**`observations` pool**

- Holds distilled findings: facts, quotes, and data points extracted by each researcher.
- **Dedup by content hash**: identical observations from overlapping searches are collapsed. This is especially important in parallel execution, where two researchers may find the same information through different paths.
- **Cap at 50 items**: keeps the synthesizer's context window manageable.

The pool caps are lower than in the `simple_research.yaml` example (100/50 vs 200/100) because parallel researchers tend to produce a higher volume of items in the same wall-clock time.

---

### Root Sequence

```yaml
root:
  id: main
  type: sequence
  label: Parallel Research Pipeline
  children:
    - ...  # 3 children
```

The root is a **sequence** -- it executes its three children (coordinator, parallel researchers, synthesizer) one after another, top to bottom.

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

**What it does:** The coordinator classifies the incoming query's complexity, domain, and suggested research depth.

**Key details:**

- **`subtype: coordinator`** -- loads the builtin coordinator prompt, which returns a structured JSON classification.
- **`model_tier: simple`** -- uses the cheapest/fastest model tier. Classification does not need a reasoning model.
- **`output_key: coordination`** -- the classification result is written to `state["coordination"]`. Both parallel researchers read this to adapt their search strategy.

---

### Step 2: Parallel Researchers

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

This is the core of the workflow. The `parallel` node launches both researchers concurrently.

**Web Researcher (`researcher_web`)**

- Searches the public web using `web_search` and `web_crawl`.
- Writes findings to `state["web_findings"]`.
- Appends extracted observations and sources to the shared pools.

**Enterprise Researcher (`researcher_enterprise`)**

- Searches internal data using `vector_search` and `knowledge_assistant`.
- Writes findings to `state["enterprise_findings"]`.
- Appends extracted observations and sources to the same shared pools.

**Critical design points:**

| Aspect | Detail |
|--------|--------|
| **Separate `output_key` values** | `web_findings` vs `enterprise_findings`. Each researcher writes to a different state key to avoid collisions. If both wrote to the same key, the last writer would overwrite the first. |
| **Shared pools** | Both write to `observations` and `sources`. Pool writes are append-only and lock-protected, so concurrent writes are safe. |
| **Identical `max_tool_calls`** | Both are capped at 5 tool calls. Since they run concurrently, the total tool calls across both is up to 10 in the same wall-clock window. |
| **Separate tool sets** | Each researcher has access only to the tools relevant to its domain. The web researcher cannot query enterprise data and vice versa. |

---

### Step 3: Synthesizer

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

**What it does:** Consumes all observations and sources gathered by both researchers and produces the final report.

**Key details:**

- **`model_tier: analytical`** -- uses the balanced model tier. Unlike the simple_research example (which uses `complex`), this workflow keeps the synthesizer at `analytical` since the parallel researchers have already done focused, tool-assisted research.
- **`output_key: report`** -- the final report is written to `state["report"]`, matching the workflow's declared `output_keys`.
- **`pool_inject`** -- both pools are injected into the synthesizer's context at threshold 0 (include all items). The synthesizer sees the combined output of both researchers without needing to know which researcher produced which item.
- **`max_tool_calls: 0`** -- the synthesizer has no tools; it operates purely on the injected pool contents. This is a simpler configuration than the `simple_research.yaml` synthesizer, which has pool search tools for handling larger context.

---

## When to Use Parallel Research

Parallel research is the right pattern when:

- **Multiple independent aspects to research** -- the query touches distinct domains (e.g., technical details and market analysis) that can be investigated independently.
- **Different source types** -- you want to combine web search results with enterprise data (vector search indexes, knowledge assistants, Genie spaces) in a single pipeline.
- **Speed optimization** -- wall-clock time matters more than token efficiency. Two researchers running concurrently finish in roughly the time of one, at the cost of double the concurrent token usage.
- **Breadth over depth** -- you want broad coverage across sources rather than deep iterative investigation of a single thread.

Parallel research is **not** the right pattern when:

- Steps depend on each other (step 2 needs step 1's results). Use a `sequence` or `plan_and_execute` instead.
- You need iterative refinement with reflection between steps. Use `plan_and_execute` with an evaluator.
- Token budget is tight. Parallel agents consume tokens concurrently, which can hit rate limits faster.

---

## Caveats

### State Key Collisions

All parallel children share the same workflow state. If two children write to the same `output_key`, the last writer wins and the first writer's output is silently overwritten. Always use distinct `output_key` values for parallel children:

```yaml
# CORRECT: separate output keys
- id: researcher_web
  config:
    output_key: web_findings      # unique
- id: researcher_enterprise
  config:
    output_key: enterprise_findings  # unique

# WRONG: collision
- id: researcher_web
  config:
    output_key: findings          # collision!
- id: researcher_enterprise
  config:
    output_key: findings          # collision!
```

### Token Budget Sharing

Parallel agents consume tokens concurrently. If your LLM endpoint has a tokens-per-minute rate limit, two parallel researchers can hit that limit twice as fast as a single sequential researcher. Consider:

- Lowering `max_tool_calls` on each parallel child.
- Using endpoints with higher rate limits for parallel workflows.
- Enabling `fallback_on_429: true` in the model configuration to handle rate limit errors gracefully.

### Pool Dedup Under Concurrency

Pool dedup handles duplicate sources found by multiple researchers. If the web researcher and the enterprise researcher both discover the same URL, the `dedup_key: url` setting ensures it is stored only once. The second write is silently dropped, and the researcher that produced it is not notified of the dedup. This is the desired behavior -- downstream consumers (the synthesizer) see a clean, deduplicated collection.

---

## Expected Event Stream

Below is the approximate sequence of events emitted during a typical parallel research run. Events from the two researchers are **interleaved** -- the exact ordering depends on which researcher's I/O calls return first.

```
workflow_started
node_started (coordinator)
coordinator_classified {complexity: "analytical"}
node_completed (coordinator)
node_started (parallel_research)
  node_started (researcher_web)            # both start ~simultaneously
  node_started (researcher_enterprise)     #
  tool_call (researcher_web) {web_search}
  tool_call (researcher_enterprise) {vector_search}
  tool_result (researcher_enterprise) {3 documents}
  tool_result (researcher_web) {5 sources}
  tool_call (researcher_web) {web_crawl}
  tool_call (researcher_enterprise) {knowledge_assistant}
  tool_result (researcher_enterprise) {answer}
  tool_result (researcher_web) {page content}
  pool_write (observations) {from: researcher_enterprise}
  pool_write (observations) {from: researcher_web}
  pool_write (sources) {from: researcher_web}
  pool_write (sources) {from: researcher_enterprise}
  ...
  node_completed (researcher_enterprise)   # may finish before web
  node_completed (researcher_web)          # or after
node_completed (parallel_research)         # waits for both
node_started (synthesizer)
agent_stream_chunk {chunk: "# Report Title\n\n..."}
...
node_completed (synthesizer)
workflow_completed {total_tokens: ~8000}
```

Every event carries a `node_id`, timestamp, and correlation metadata. You can filter the stream by `node_id` to reconstruct each researcher's activity independently.

---

## Running It

```python
from databricks_deep_research import WorkflowRunner, load_workflow
from openai import AsyncOpenAI

workflow = load_workflow("examples/parallel_research.yaml")
runner = WorkflowRunner(workflow, AsyncOpenAI(), model_mapping={
    "simple": "gpt-4o-mini",
    "analytical": "gpt-4o",
    "complex": "gpt-4o",
})
result = await runner.run(query="How does our internal auth system compare to industry best practices?")

print(result.state["report"])
```

The `model_mapping` dict maps the `model_tier` values used in the YAML (`simple`, `analytical`, `complex`) to actual model identifiers your OpenAI-compatible endpoint understands. Both parallel researchers use the `analytical` tier in this workflow, so they will both be served by the same model.

To stream events as they arrive (useful for showing real-time progress from both researchers):

```python
async for event in runner.stream(query="How does our internal auth system compare to industry best practices?"):
    print(f"[{event.node_id}] {event.type}: {event.summary}")
```

---

## See Also

- [Node Types Reference - parallel](../reference/node-types-reference.md#4-parallel)
- [YAML Workflow Authoring](../guides/yaml-workflow-authoring.md)
- [Pool Configuration](../guides/pool-configuration.md)
