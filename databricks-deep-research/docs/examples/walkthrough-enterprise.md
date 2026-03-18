# Walkthrough: Enterprise Research

> Using Databricks enterprise data sources (Vector Search, Genie) in workflows.

## Overview

This walkthrough shows how to build workflows that query internal enterprise data instead of (or alongside) web sources. We will walk through `examples/enterprise_research.yaml` section by section, explaining how enterprise tools are declared, how the planner is guided toward those tools, and how results flow through pools into the final synthesis.

By the end you will understand how to:

- Declare Vector Search, Genie, and Knowledge Assistant tools in YAML
- Tell the planner which tools are available (and that web search is not)
- Wire the researcher to use only enterprise tools
- Run the workflow with an OBO token for user-scoped permissions
- Mix enterprise and web sources in a single workflow

## The Complete YAML

```yaml
id: enterprise_research
name: Enterprise Research Pipeline
description: Research using only enterprise data sources (no web search)
version: 1
required_inputs: [query]
output_keys: [report]

tools:
  - name: genie
    kind: genie
    config:
      space_id: 01ef8d7a-0000-0000-0000-000000000000
    description: "Enterprise data warehouse — financial data, operational metrics, KPIs"

  - name: vector_search
    kind: vector_search
    config:
      index_name: prod_catalog.docs.internal_docs_idx
      num_results: 10
    description: "Internal documents — architecture reviews, technical specs, policies"

  - name: knowledge_assistant
    kind: knowledge_assistant
    config:
      endpoint_name: internal-docs-assistant
    description: "AI-powered Q&A on internal documentation and runbooks"

pools:
  - name: sources
    dedup_key: url
    max_items: 50
  - name: observations
    dedup_content_hash: true
    max_items: 30

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

## Section-by-Section Annotation

### Enterprise Tool Declarations

The top-level `tools:` section registers three enterprise data sources. Each declaration tells the framework how to instantiate the tool at execution time via the `DatabricksToolFactory`.

```yaml
tools:
  - name: genie
    kind: genie
    config:
      space_id: 01ef8d7a-0000-0000-0000-000000000000
    description: "Enterprise data warehouse — financial data, operational metrics, KPIs"
```

**Genie** connects to a Databricks Genie space. The LLM sends a natural-language question, Genie translates it to SQL, runs the query, and returns tabular results. Configuration is minimal -- only `space_id` is required. The `description` is included in the function-calling schema so the LLM knows *what kind of data* lives behind this tool.

```yaml
  - name: vector_search
    kind: vector_search
    config:
      index_name: prod_catalog.docs.internal_docs_idx
      num_results: 10
    description: "Internal documents — architecture reviews, technical specs, policies"
```

**Vector Search** queries a Unity Catalog Vector Search index. `index_name` is the fully qualified three-part name (`catalog.schema.index`). `num_results` sets the default hit count per query. You can optionally add `columns`, `query_type` (e.g., `HYBRID`), and `filters_json` for static filters.

```yaml
  - name: knowledge_assistant
    kind: knowledge_assistant
    config:
      endpoint_name: internal-docs-assistant
    description: "AI-powered Q&A on internal documentation and runbooks"
```

**Knowledge Assistant** wraps a Databricks serving endpoint that implements a Q&A interface. The LLM sends a question; the endpoint returns an answer. This works with RAG chains, fine-tuned models, or any custom deployment behind a serving endpoint.

All three tools require a `WorkspaceClient` for authentication. The framework injects this automatically through the `ToolFactoryContext` at execution time.

### Source Definitions (Pools)

```yaml
pools:
  - name: sources
    dedup_key: url
    max_items: 50
  - name: observations
    dedup_content_hash: true
    max_items: 30
```

Pools are shared, append-only stores that pass data between agents. This workflow uses two:

- **`sources`** -- Collects metadata about every data source the researcher touches. `dedup_key: url` ensures the same source URL is never stored twice, regardless of which tool produced it. Enterprise results use URL schemes like `enterprise://genie/{space_id}` and `enterprise://vector_search/{tool_name}/{idx}`.
- **`observations`** -- Collects the actual findings and evidence. `dedup_content_hash: true` prevents duplicate observations by hashing content. The `max_items: 30` cap keeps the synthesizer prompt within token budget.

Both pools are written to by the researcher (via `pool_writes`) and read by the reflector and synthesizer (via `pool_inject`).

### Planner Guidance

```yaml
planner_guidance: |
  Available tools for the researcher:
  - genie: Query enterprise data warehouse using natural language (financial data, operational metrics, KPIs)
  - vector_search: Semantic search over internal documents (architecture reviews, technical specs, policies)
  - knowledge_assistant: AI-powered Q&A on internal documentation and runbooks
  There is NO web search available. Design all research steps for these enterprise data sources.
  For financial queries, plan steps that ask specific questions to the genie tool.
  For technical/documentation queries, plan steps that search the vector index or ask the knowledge assistant.
```

The `planner_guidance` field is injected into the planner agent's system prompt. This is critical for enterprise-only workflows because the planner has no other way to know which tools exist. Without guidance, the planner may generate steps like "Search the web for Q3 revenue data" -- steps the researcher cannot fulfill.

Good planner guidance should:

1. **List every available tool** with a short description of what data it covers
2. **State what is NOT available** (here: no web search)
3. **Give routing hints** -- which tool to use for which kind of question

### Enterprise Researcher

```yaml
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
```

The `tools:` list on the researcher node controls which tools appear in the LLM's function-calling schema. Only the three enterprise tools are listed -- no `web_search` or `web_crawl`. The researcher agent will only see (and can only call) these tools.

Key configuration choices:

- **`max_tool_calls: 8`** -- Enterprise tools, especially Genie, can be slower than web search (Genie has a 180-second timeout per query). Eight calls provides enough budget for 2--3 queries per tool while keeping total latency bounded.
- **`pool_writes`** -- After each research step, findings go to the `observations` pool and source metadata goes to the `sources` pool. The `extract` field names the key within the researcher's output that contains the relevant data.
- **`model_tier: analytical`** -- The balanced tier is a good default for enterprise research. Use `complex` (reasoning model) only for queries that require multi-step logical inference.

### Pool Integration

The reflector and synthesizer both read from pools via `pool_inject`:

```yaml
evaluator:
  subtype: reflector
  model_tier: analytical
  output_key: evaluation
  pool_inject:
    - pool: observations
      threshold: 0
```

The reflector receives all collected observations (`threshold: 0` means "inject everything"). It uses this accumulated evidence to decide whether to CONTINUE researching, ADJUST the plan, or declare COMPLETE.

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

The synthesizer receives both `observations` and `sources`. It generates the final report by weaving together findings from all enterprise tools. `max_tool_calls: 0` ensures the synthesizer only writes -- it cannot make additional tool calls.

## Running with OBO Token

When deployed as a Databricks App, enterprise tools need to run with the requesting user's permissions. Pass the on-behalf-of (OBO) token through the `ToolFactoryContext`:

```python
import asyncio
from openai import AsyncOpenAI
from databricks_deep_research import WorkflowRunner, load_workflow
from databricks_deep_research.tools.factory import ToolFactoryContext

async def main():
    client = AsyncOpenAI(
        base_url="https://your-workspace.databricks.com/serving-endpoints",
        api_key=obo_token,
    )
    workflow = load_workflow("examples/enterprise_research.yaml")

    # Create a tool context with the user's OBO token
    tool_ctx = ToolFactoryContext.from_defaults(user_token=obo_token)

    runner = WorkflowRunner(
        workflow,
        client,
        model_mapping={
            "simple": "databricks-meta-llama-3-1-70b-instruct",
            "analytical": "databricks-meta-llama-3-1-70b-instruct",
        },
        tool_factory_context=tool_ctx,
    )

    result = await runner.run(
        query="What were Q3 2024 revenue trends?",
    )
    print(result.output)

asyncio.run(main())
```

The token flows through the factory chain:

```
request.state.obo_token -> ToolFactoryContext.user_token -> WorkspaceClient -> enterprise tool API calls
```

This ensures that Genie queries, vector search requests, and knowledge assistant calls all execute with the requesting user's Unity Catalog permissions.

## Mixing Enterprise and Web

To combine both source types, declare web tools alongside enterprise tools and give the researcher access to all of them. A parallel node lets web and enterprise research run concurrently:

```yaml
tools:
  - name: web_search
    kind: web_search
    config:
      max_results: 10
  - name: web_crawl
    kind: web_crawl
  - name: genie
    kind: genie
    config:
      space_id: 01ef8d7a-0000-0000-0000-000000000000
    description: "Enterprise data warehouse analytics"
  - name: vector_search
    kind: vector_search
    config:
      index_name: prod_catalog.docs.internal_docs_idx
      num_results: 10
    description: "Internal document search"

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
  label: Parallel Mixed Pipeline
  children:
    - id: coordinator
      type: agent
      label: Query Classifier
      config:
        subtype: coordinator
        model_tier: simple
        output_key: coordination

    # Run web and enterprise research in parallel
    - id: parallel_research
      type: parallel
      label: Parallel Research
      children:
        - id: web_researcher
          type: agent
          label: Web Researcher
          config:
            subtype: researcher
            model_tier: analytical
            output_key: web_findings
            tools: [web_search, web_crawl]
            pool_writes:
              - pool: observations
                extract: findings
              - pool: sources
                extract: sources
            max_tool_calls: 8

        - id: enterprise_researcher
          type: agent
          label: Enterprise Researcher
          config:
            subtype: researcher
            model_tier: analytical
            output_key: enterprise_findings
            tools: [genie, vector_search]
            pool_writes:
              - pool: observations
                extract: findings
              - pool: sources
                extract: sources
            max_tool_calls: 6

    # Synthesize from all sources
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

Both researchers write to the same `sources` and `observations` pools. Pool deduplication prevents duplicates regardless of which tool produced the result. The synthesizer sees all collected evidence uniformly through `pool_inject`, merging web context with internal data in the final report.

This parallel pattern reduces total wall-clock time because web and enterprise research run concurrently. Increase `max_items` on pools when combining sources to accommodate the higher volume.

## See Also

- [Enterprise Data Sources](../guides/enterprise-data-sources.md)
- [Builtin Tools](../guides/builtin-tools.md)
- [Authentication](../getting-started/authentication.md)
