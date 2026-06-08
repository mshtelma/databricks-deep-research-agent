# Enterprise Data Sources

> Connect workflows to Databricks Vector Search, Genie, and Knowledge Assistant.

## Overview

The framework integrates with three Databricks enterprise data sources for searching internal data alongside web sources. Each enterprise tool implements the `ResearchTool` protocol and is created automatically by the `DatabricksToolFactory` when declared in a workflow YAML file.

Enterprise tools require a `WorkspaceClient` for authentication and are resolved through the factory chain: YAML `tools:` declaration -> `DatabricksToolFactory.create()` -> concrete tool instance.

**Supported kinds:**

| Kind | Tool class | Source kind | Use case |
|------|-----------|-------------|----------|
| `vector_search` | `DatabricksVectorSearchTool` | `vector_index` | Semantic search over embeddings |
| `genie` | `DatabricksGenieTool` | `sql_analytics` | Natural language to SQL |
| `knowledge_assistant` | `DatabricksKnowledgeAssistantTool` | `qa_assistant` | Q&A over serving endpoints |

## Vector Search

Queries Unity Catalog Vector Search indexes using semantic similarity. The tool wraps the Databricks SDK `VectorSearchIndexes.query_index()` API.

### Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `index_name` | string | yes | -- | Fully qualified index name (`catalog.schema.index`) |
| `columns` | list[string] | no | auto-discovered | Columns to return from the index |
| `num_results` | integer | no | `10` | Number of results per query |
| `query_type` | string | no | -- | Query type (e.g., `HYBRID` for hybrid search) |
| `filters_json` | string | no | -- | Static JSON filter applied to all queries |

### YAML declaration

```yaml
tools:
  - name: product_docs
    kind: vector_search
    config:
      index_name: catalog.schema.product_docs_index
      columns: [content, url, title]
      num_results: 10
      query_type: HYBRID
    description: "Product documentation and knowledge base articles"
```

### Tool parameters (LLM-facing)

The LLM receives a function with these parameters:

- `query` (string, required) -- the search query text (max 1000 characters)
- `num_results` (integer, optional) -- override the default result count
- `filters` (object, optional) -- dynamic filter conditions per query

### Column auto-discovery

When `columns` is omitted, the tool calls `VectorSearchIndexes.get_index()` at first execution to discover the primary key and embedding source columns automatically. Explicitly setting `columns` avoids this extra API call.

### Source tracking

Results use the URL scheme `enterprise://vector_search/{tool_name}/{row_index}`. If a result row contains a column named `url`, `source_url`, or `doc_url`, that value is used as the `canonical_url` for citation linking.

### Use cases

- Search internal documentation, knowledge bases, and support tickets
- Semantic retrieval over product specs, architecture docs, or policies
- Hybrid (keyword + vector) search when `query_type: HYBRID` is set

---

## Genie (NL -> SQL)

Queries structured data using natural language via Databricks Genie spaces. The tool starts a Genie conversation, waits for the SQL to execute (up to 180 seconds), and formats the tabular results.

### Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `space_id` | string | yes | -- | Genie space identifier |

### YAML declaration

```yaml
tools:
  - name: sales_data
    kind: genie
    config:
      space_id: "01ef8d7a-0000-0000-0000-000000000000"
    description: "Enterprise data warehouse -- revenue, operational metrics, KPIs"
```

### Tool parameters (LLM-facing)

- `question` (string, required) -- natural language question about the data (max 2000 characters)

### Result formatting

Genie results are formatted as pipe-delimited tables with up to 50 rows. The tool extracts:

1. Text content from message attachments
2. Generated SQL queries and their descriptions
3. Tabular data from statement responses (column headers + rows)

### Timeout handling

Genie queries have a 180-second timeout. On timeout, the tool returns a non-fatal `ToolResult` with `success=False` and `error="timeout"`, allowing the researcher agent to try an alternative approach.

### Source tracking

Results use the URL scheme `enterprise://genie/{space_id}`.

### Use cases

- Query revenue, financial metrics, and KPIs using natural language
- Analyze operational data (customer counts, usage statistics)
- Generate ad-hoc reports from data warehouses

---

## Knowledge Assistant

Queries Databricks serving endpoints that implement a Q&A interface. The tool sends a question to the endpoint and extracts the answer from the response.

### Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `endpoint_name` | string | yes | -- | Databricks serving endpoint name |

### YAML declaration

```yaml
tools:
  - name: internal_qa
    kind: knowledge_assistant
    config:
      endpoint_name: my-assistant-endpoint
    description: "AI-powered Q&A on internal documentation and runbooks"
```

### Tool parameters (LLM-facing)

- `question` (string, required) -- question to ask the knowledge assistant (max 2000 characters)

### Response extraction

The tool handles multiple serving endpoint response formats:

1. **Predictions format** -- `response.predictions` (DataframeSplitInput)
2. **Outputs format** -- `response.outputs`
3. **Chat format** -- `response.choices[0].message.content`
4. **Fallback** -- `str(response)`

### Source tracking

Results use the URL scheme `enterprise://knowledge_assistant/{endpoint_name}`.

### Use cases

- Q&A over curated knowledge bases
- Query internal runbooks and operational documentation
- Ask questions about company policies and procedures

---

## Authentication

All enterprise tools require a Databricks `WorkspaceClient` for API access. The client is injected through the `ToolFactoryContext` at tool creation time.

### Token flow

```
ToolFactoryContext.workspace_client
    |
    v
DatabricksToolFactory.create(decl, ctx)
    |
    v
DatabricksVectorSearchTool(workspace_client=ctx.workspace_client, ...)
DatabricksGenieTool(workspace_client=ctx.workspace_client, ...)
DatabricksKnowledgeAssistantTool(workspace_client=ctx.workspace_client, ...)
```

### Local development

Set credentials via environment variables or a Databricks config profile:

```bash
# Option 1: Direct token
export DATABRICKS_HOST=https://your-workspace.databricks.com
export DATABRICKS_TOKEN=your-token

# Option 2: Config profile
export DATABRICKS_CONFIG_PROFILE=your-profile-name
```

### Databricks Apps (OBO)

When running as a Databricks App, the on-behalf-of (OBO) token from the request context propagates through:

```
request.state.obo_token -> ToolFactoryContext.user_token -> WorkspaceClient
```

This ensures enterprise tool calls are made with the requesting user's permissions.

### Factory context setup

```python
from databricks_deep_research.tools.factory import ToolFactoryContext

# Auto-detect workspace client and search client
ctx = ToolFactoryContext.from_defaults(
    brave_api_key="...",       # optional — only for the Brave web_search backend (see Search Providers)
    user_token="...",          # optional, for OBO
)

# Or inject explicitly
from databricks.sdk import WorkspaceClient

ctx = ToolFactoryContext(
    workspace_client=WorkspaceClient(),
    user_token=obo_token,
)
```

---

## Enterprise-Only Workflow

A complete workflow that uses only enterprise sources (no web search). This is the recommended pattern when all data lives inside your Databricks workspace.

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
          - genie: Query enterprise data warehouse (financial data, metrics, KPIs)
          - vector_search: Semantic search over internal documents
          - knowledge_assistant: AI-powered Q&A on runbooks and documentation
          There is NO web search available. Design all research steps for these enterprise data sources.
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

Key points for enterprise-only workflows:

- **`planner_guidance`** -- Tell the planner which tools exist and that web search is unavailable. Without this, the planner may generate steps that assume web access.
- **`tools:` list on researcher** -- Only list the enterprise tool names. The researcher agent will only see these tools in its function-calling schema.
- **`max_tool_calls: 8`** -- Enterprise tools (especially Genie) can be slower than web search. Allow enough calls but keep the budget bounded.

---

## Mixing Enterprise and Web Sources

Combine web and enterprise sources in a single workflow for comprehensive research. The researcher agent decides which tool to call based on the question context.

```yaml
id: mixed_sources
name: Mixed Sources Research Pipeline
description: Research using both web and enterprise tools
version: 1
required_inputs: [query]
output_keys: [report]

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
  label: Mixed Sources Pipeline
  children:
    - id: coordinator
      type: agent
      label: Query Classifier
      config:
        subtype: coordinator
        model_tier: simple
        output_key: coordination

    # Quick web background scan
    - id: background
      type: agent
      label: Background Investigator
      config:
        subtype: background
        model_tier: simple
        output_key: background
        tools: [web_search]
        max_tool_calls: 2

    # Research cycle with all tools available
    - id: research_cycle
      type: plan_and_execute
      label: Mixed Research Cycle
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
          label: Mixed Researcher
          config:
            subtype: researcher
            model_tier: analytical
            output_key: findings
            tools: [web_search, web_crawl, genie, vector_search]
            pool_writes:
              - pool: observations
                extract: findings
              - pool: sources
                extract: sources
            max_tool_calls: 10
        evaluator:
          subtype: reflector
          model_tier: analytical
          output_key: evaluation
          pool_inject:
            - pool: observations
              threshold: 0
        max_iterations: 4
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

### Shared pools for unified source collection

Both web and enterprise results flow into the same `sources` and `observations` pools. Pool deduplication (`dedup_key: url` for sources, `dedup_content_hash: true` for observations) prevents duplicate entries regardless of the source type. The synthesizer sees all collected evidence uniformly through `pool_inject`.

### Parallel web + enterprise pattern

For workflows where web and enterprise research can run independently, use a `parallel` node:

```yaml
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

This pattern runs web and enterprise research concurrently, reducing total wall-clock time. Both researchers write to the same shared pools, and the synthesizer merges all findings.

---

## Source Routing

The framework routes queries to appropriate sources based on the tool declarations and agent configuration in the workflow YAML.

### How routing works

1. **Tool declarations** (top-level `tools:` section) define which data sources are available and how to connect to them.
2. **Agent `tools:` lists** (inside `config:`) control which subset of declared tools each agent node can use.
3. **The `ToolResolver`** resolves tool names at execution time through a priority chain: overrides -> cache -> factory chain -> legacy registry.
4. **The researcher agent** decides which tool to call based on the current research step and the available tool definitions in its function-calling schema.

### Source kinds and query strategies

Each tool kind maps to a `SourceKind` that influences how the agent formulates queries:

| Tool kind | Source kind | Query strategy |
|-----------|-----------|----------------|
| `web_search` | `web` | Keyword/BM25 search queries |
| `vector_search` | `vector_index` | Semantic similarity queries |
| `genie` | `sql_analytics` | Natural language data questions |
| `knowledge_assistant` | `qa_assistant` | Direct questions |
| `file_search` | `file` | Keyword search over uploaded files |

### Planner guidance

For enterprise-only or mixed workflows, use `planner_guidance` in the `plan_and_execute` node to inform the planner about available tools. Without guidance, the planner may generate steps that assume tools are available when they are not.

```yaml
config:
  planner_guidance: |
    Available tools: genie (financial data), vector_search (internal docs).
    No web search is available.
```

---

## See Also

- [Builtin Agents](builtin-agents.md)
- [YAML Workflow Authoring](yaml-workflow-authoring.md)
- [Search Providers](search-providers.md) -- web_search backends (databricks / brave / jina)
- [Tool System](../concepts/tool-system.md)
- [Authentication](../getting-started/authentication.md)
- [Example: Enterprise Research](../../examples/enterprise_research.yaml)
- [Example: Mixed Sources](../../examples/mixed_sources.yaml)
- [Example: Verified Enterprise Research](../../examples/verified_enterprise_research.yaml)
- [Example: Multi-Source Research](../../examples/multi_source_research.yaml)
