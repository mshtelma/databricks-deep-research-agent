# Builtin Tools

> Detailed guide to all builtin tools shipped with the framework.

## Overview

The framework includes 6 builtin tools covering web search, crawling, file search, and Databricks enterprise data sources. All tools implement the `ResearchTool` protocol, which requires three members:

- **`definition`** -- a `ToolDefinition` with name, description, and JSON Schema parameters for LLM function calling.
- **`validate_arguments(arguments)`** -- validates and cleans raw LLM arguments; raises `ValueError` on invalid input.
- **`execute(arguments, context)`** -- async execution returning a `ToolResult`.

Dependencies are constructor-injected -- the tool constructor receives clients, tokens, and configuration, while `ToolContext` carries only per-call values (current query, shared `UrlRegistry`).

---

## 1. `web_search`

Search the web through the configured search provider (Databricks built-in web search by default in the app; Brave or Jina also supported -- see [Search Providers](search-providers.md)). Returns numbered results with titles and snippets. Discovered URLs are registered in the shared `UrlRegistry` so downstream tools (like `web_crawl`) can resolve integer indices back to URLs without the LLM ever seeing raw URLs.

| Property | Value |
|----------|-------|
| **Class** | `WebSearchTool` |
| **ToolKind** | `web_search` |
| **SourceKind** | `web` |
| **Required extra** | `web` (`pip install 'databricks-deep-research[web]'` -- installs `httpx`) |
| **Import** | `from databricks_deep_research.tools.builtins.web_search import WebSearchTool` |

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `search_client` | `SearchClient` (protocol) | *required* | Any async backend satisfying `search(query, *, count, freshness) -> list[SearchResult]` |
| `domain_filter` | `list[str] \| None` | `None` | Allowed domain patterns (e.g. `["*.gov", "reuters.com"]`). Results not matching any pattern are dropped. Supports `*.suffix` wildcards. |
| `max_results` | `int` | `5` | Default result count when the LLM does not specify one. Clamped to 1--20. |

### LLM Parameters (JSON Schema)

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | `string` | Yes | A specific, focused search query. Max 500 characters. |
| `count` | `integer` | No | Number of results to return (default: constructor `max_results`, max: 20). |
| `freshness` | `string` | No | Time filter: `"pd"` (past day), `"pw"` (past week), `"pm"` (past month), or omit for any time. |

### Return Value

`ToolResult` with:
- **`content`**: Formatted lines like `[0] **Title**\n    Snippet`, one per result. Returns `"No search results found. Try a different query."` when empty.
- **`sources`**: List of `SourceInfo` with `url`, `title`, `snippet`, `source_type="web"`.
- **`data`**: `{"query": str, "total_results": int, "count": int}`.
- **`success`**: `False` on exception, with `error` populated.

### YAML Declaration

```yaml
tools:
  - name: web_search
    kind: web_search
    config:
      brave_api_key: "${BRAVE_API_KEY}"
      max_results: 10
      domain_filter:
        - "*.gov"
        - "reuters.com"
        - "*.edu"
```

### Search Client: BraveSearchAdapter

`BraveSearchAdapter` (`tools.builtins.brave_search`) is one of the shipped `SearchClient` backends for `WebSearchTool`. It wraps the Brave Web Search API with `httpx`. The framework package auto-creates it when `BRAVE_API_KEY` is set; the app defaults to Databricks built-in web search instead. See [Search Providers](search-providers.md) for the full backend list and precedence rules.

```python
from databricks_deep_research.tools.builtins.brave_search import BraveSearchAdapter

adapter = BraveSearchAdapter(api_key="your-brave-key")
results = await adapter.search("NVIDIA revenue 2025", count=5)
```

`ToolFactoryContext.from_defaults()` auto-creates a `BraveSearchAdapter` when `BRAVE_API_KEY` is set.

### Usage Tips

- Include entities, dates, or metrics in queries for best results (e.g. "Apple Q4 2024 revenue earnings report").
- Use `freshness: "pd"` for breaking news, `"pw"` for recent developments.
- The `domain_filter` is applied post-search -- results outside allowed domains are silently dropped.
- Result indices are registered in the shared `UrlRegistry` and should be passed to `web_crawl` for full-page extraction.

---

## 2. `web_crawl`

Fetch a web page and extract readable text. The LLM provides an integer `url_index` (from prior `web_search` results) which is resolved via the shared `UrlRegistry` -- the LLM never passes raw URLs directly, preventing hallucinated-URL injection.

| Property | Value |
|----------|-------|
| **Class** | `WebCrawlTool` |
| **ToolKind** | `web_crawl` |
| **SourceKind** | `web` (definition) / `builtin` (routing) |
| **Required extra** | `web` (`httpx`) + `crawl` (`trafilatura`) |
| **Import** | `from databricks_deep_research.tools.builtins.web_crawl import WebCrawlTool` |

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `crawler` | `ContentCrawler \| None` | `None` | Custom async callable `(url) -> (text, title)`. When `None`, the built-in httpx + trafilatura pipeline is used. |
| `timeout` | `float` | `30.0` | HTTP timeout in seconds (used by the default crawler). |
| `max_content_length` | `int` | `50_000` | Maximum extracted text length in characters. Content beyond this is truncated. |

### LLM Parameters (JSON Schema)

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `url_index` | `integer` | Yes | Index number of the source from search results (0, 1, 2, ...). Stringified integers are coerced automatically. Must be non-negative. |

### Return Value

`ToolResult` with:
- **`content`**: Formatted as `# Title\n\nURL: url\n\ncontent_text`. On failure: descriptive error message.
- **`sources`**: Single `SourceInfo` with `url`, `title`, `snippet` (first 200 chars), `source_type="web"`.
- **`data`**: `{"url": str, "title": str|None, "content_length": int, "url_index": int}`.
- **`success`**: `False` when URL cannot be resolved, page returns non-HTML content, content is too short (< 100 chars, likely CAPTCHA/error page), or fetching fails.

### YAML Declaration

```yaml
tools:
  - name: web_crawl
    kind: web_crawl
    config:
      timeout: 30
      max_content_length: 50000
```

### Usage Tips

- Always call `web_search` first -- `web_crawl` requires valid indices from the shared `UrlRegistry`.
- Pages shorter than 100 characters are rejected (likely blocked or error pages).
- The default crawler rotates user-agents to reduce bot-detection blocking.
- Without `trafilatura` installed, a crude HTML tag-stripping fallback is used. Install the `[crawl]` extra for proper extraction.
- Supports only `http` and `https` URL schemes.

---

## 3. `file_search`

Search through user-provided documents and files for relevant passages. Uses BM25 ranking when the `[search]` extra is installed (`bm25s`), falling back to simple keyword overlap otherwise.

| Property | Value |
|----------|-------|
| **Class** | `FileSearchTool` |
| **ToolKind** | `file_search` |
| **SourceKind** | `file` |
| **Required extra** | `search` (`bm25s`) for BM25 ranking; works without it via keyword fallback |
| **Import** | `from databricks_deep_research.tools.builtins.file_search import FileSearchTool` |

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_index` | `FileIndex` (protocol) | *required* | Pre-built index implementing `get_chunks() -> list[dict]`. Each chunk must have `content` (str) and `source` (str). Optional: `chunk_index`, `page_number`, `section`, `file_id`. |
| `top_k` | `int` | `5` | Default number of results to return. |

### LLM Parameters (JSON Schema)

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | `string` | Yes | Search query to find relevant content in files. Must be 2--500 characters. |
| `top_k` | `integer` | No | Maximum results to return (default: constructor `top_k`, max: 50). |

### Return Value

`ToolResult` with:
- **`content`**: Formatted lines like `[0] filename.pdf (page 3) (score: 0.85)\n    ...highlighted snippet...`, one per result.
- **`sources`**: List of `SourceInfo` with `url` as `file://{file_id}#chunk-{idx}`, `title`, `snippet` (first 300 chars), `source_type="file"`.
- **`data`**: `{"query": str, "num_results": int}`.
- **`success`**: Always `True` (defaults); returns "No files available" or "No results found" messages when empty.

### YAML Declaration

```yaml
tools:
  - name: file_search
    kind: file_search
    config:
      top_k: 10
```

### Usage Tips

- The `FileIndex` protocol is intentionally minimal -- implement `get_chunks()` to return your pre-chunked documents.
- BM25 ranking (`bm25s` library) gives significantly better results than the keyword fallback. Install the `[search]` extra for production use.
- Results include keyword highlighting -- a snippet is centered around the first query term match with configurable context (150 chars on each side).
- Only results with a positive score are returned (BM25 > 0, or keyword overlap > 0).

---

## 4. `vector_search`

Query a Databricks Unity Catalog Vector Search index. Wraps the Databricks SDK `VectorSearchIndexes.query_index()` API. Configuration (index name, columns, query type) is injected via the constructor; the LLM only provides a query string.

| Property | Value |
|----------|-------|
| **Class** | `DatabricksVectorSearchTool` |
| **ToolKind** | `vector_search` |
| **SourceKind** | `vector_index` |
| **Required extra** | `integration` (`databricks-sdk`) |
| **Import** | `from databricks_deep_research.tools.builtins.vector_search import DatabricksVectorSearchTool` |

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `workspace_client` | `Any` | *required* | Databricks `WorkspaceClient` instance. |
| `name` | `str` | *required* | Tool name as seen by the LLM. |
| `index_name` | `str` | *required* | Fully qualified index name (e.g. `catalog.schema.index`). |
| `columns` | `list[str] \| None` | `None` | Columns to return. When `None`, auto-discovered from index metadata (primary key + embedding source columns). |
| `num_results` | `int` | `10` | Default number of results. |
| `query_type` | `str \| None` | `None` | Query type override (e.g. `"ANN"`, `"HYBRID"`). |
| `filters_json` | `str \| None` | `None` | Default filter conditions as a JSON string. |
| `description` | `str` | `""` | Custom description; defaults to `"Vector search over {index_name}"`. |

### LLM Parameters (JSON Schema)

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | `string` | Yes | The search query text. Max 1000 characters. |
| `num_results` | `integer` | No | Number of results to return (default: constructor `num_results`). |
| `filters` | `object` | No | Optional filter conditions (serialized to JSON for the API). |

### Return Value

`ToolResult` with:
- **`content`**: Numbered lines like `[1] col_name: value; col_name: value`, one per result row.
- **`sources`**: List of `SourceInfo` with `url` as `enterprise://vector_search/{tool_name}/{idx}`, `canonical_url` extracted from URL columns if present, `title`, `snippet` (first 1500 chars of content), `source_type="enterprise"`, `source_kind="vector_index"`, `relevance_score`.
- **`data`**: `{"result_count": int, "source_kind": "vector_index", "empty_result": bool}`.
- **`success`**: `False` on exception, with `error` populated.

### YAML Declaration

```yaml
tools:
  - name: my_vector_index
    kind: vector_search
    config:
      index_name: catalog.schema.earnings_index
      num_results: 10
      query_type: "ANN"
      columns:
        - id
        - content
        - title
        - url
```

### Usage Tips

- If `columns` is not specified, the tool auto-discovers a minimal column set from index metadata (primary key + embedding source columns).
- Content is extracted by scanning for common column names: `content`, `text`, `chunk_text`, `page_content`.
- Title is extracted by scanning: `title`, `source_title`, `doc_title`, `name`.
- URL columns (`url`, `source_url`, `doc_url`) are detected and stored as `canonical_url` on the source.
- Filters from LLM arguments take precedence over the constructor `filters_json`.

---

## 5. `genie`

Natural language SQL analytics via the Databricks Genie API. Wraps the conversation lifecycle (start conversation, create message, poll until complete, format results) into a single tool call.

| Property | Value |
|----------|-------|
| **Class** | `DatabricksGenieTool` |
| **ToolKind** | `genie` |
| **SourceKind** | `sql_analytics` |
| **Required extra** | `integration` (`databricks-sdk`) |
| **Import** | `from databricks_deep_research.tools.builtins.genie import DatabricksGenieTool` |

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `workspace_client` | `Any` | *required* | Databricks `WorkspaceClient` instance. |
| `name` | `str` | *required* | Tool name as seen by the LLM. |
| `space_id` | `str` | *required* | Genie space ID to query. |
| `description` | `str` | `""` | Custom description; defaults to `"Natural language SQL via Genie space {space_id}"`. |

### LLM Parameters (JSON Schema)

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `question` | `string` | Yes | Natural language question about the data. Max 2000 characters. |

### Return Value

`ToolResult` with:
- **`content`**: Formatted text including SQL query, description, and a pipe-delimited table (up to 50 rows, with a `"... (N more rows)"` trailer). Returns `"No results returned from Genie."` when empty.
- **`sources`**: Single `SourceInfo` with `url` as `enterprise://genie/{space_id}`, `title` as `"Genie: {question}"`, `snippet` (first 300 chars), `source_type="enterprise"`, `source_kind="sql_analytics"`.
- **`data`**: `{"space_id": str, "conversation_id": str, "source_kind": "sql_analytics", "empty_result": bool}`.
- **`success`**: `False` on timeout (180s max wait) or other exceptions.

### YAML Declaration

```yaml
tools:
  - name: sales_analytics
    kind: genie
    config:
      space_id: "01ef1234abcd5678"
      description: "Query sales data using natural language"
```

### Usage Tips

- Genie has a maximum wait time of 180 seconds. Complex queries on large datasets may time out.
- The tool starts a new conversation per call via `start_conversation_and_wait()`.
- Results include both the generated SQL and the data table for transparency.
- Table output is formatted as pipe-delimited text, limited to 50 rows. Larger result sets show a row count trailer.
- The tool extracts results from attachments (text, inline query tables, statement responses) with deduplication of repeated statement IDs.

---

## 6. `knowledge_assistant`

Query a Databricks serving endpoint that implements a Q&A interface. Wraps the `serving_endpoints.query()` API and handles multiple response formats (predictions, outputs, choices).

| Property | Value |
|----------|-------|
| **Class** | `DatabricksKnowledgeAssistantTool` |
| **ToolKind** | `knowledge_assistant` |
| **SourceKind** | `qa_assistant` |
| **Required extra** | `integration` (`databricks-sdk`) |
| **Import** | `from databricks_deep_research.tools.builtins.knowledge_assistant import DatabricksKnowledgeAssistantTool` |

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `workspace_client` | `Any` | *required* | Databricks `WorkspaceClient` instance. |
| `name` | `str` | *required* | Tool name as seen by the LLM. |
| `endpoint_name` | `str` | *required* | Name of the serving endpoint to query. |
| `description` | `str` | `""` | Custom description; defaults to `"Knowledge assistant via {endpoint_name}"`. |

### LLM Parameters (JSON Schema)

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `question` | `string` | Yes | Question to ask the knowledge assistant. Max 2000 characters. |

### Return Value

`ToolResult` with:
- **`content`**: The extracted answer text. Format depends on the endpoint response type (see below).
- **`sources`**: Single `SourceInfo` with `url` as `enterprise://knowledge_assistant/{endpoint_name}`, `title` as `"Knowledge Assistant: {question}"`, `snippet` (first 300 chars), `source_type="enterprise"`, `source_kind="qa_assistant"`.
- **`data`**: `{"endpoint_name": str, "source_kind": "qa_assistant", "empty_result": bool}`.
- **`success`**: `False` on exception.

The answer extraction handles multiple response formats in order:
1. `response.predictions` -- DataframeSplitInput format
2. `response.outputs` -- generic output format
3. `response.choices[0].text` or `response.choices[0].message.content` -- chat completion format
4. Fallback: `str(response)`

### YAML Declaration

```yaml
tools:
  - name: company_kb
    kind: knowledge_assistant
    config:
      endpoint_name: "my-knowledge-endpoint"
      description: "Ask questions about company policies and procedures"
```

### Usage Tips

- The endpoint must accept `inputs=[{"query": question}]` format.
- Works with any Databricks serving endpoint -- RAG chains, fine-tuned models, or custom deployments.
- The flexible response parsing handles the most common Databricks endpoint response formats automatically.

---

## The `table_*` Tool Family

Beyond the six tools above, the framework ships a family of **table tools** for structured research directly over the rows of a bound Delta table -- the "the answer is a cell, a row, or a group total in this table" use case, without natural-language-to-SQL. All six kinds map to `SourceKind.text_table` and are backed by the `tools/builtins/text_table/` package. They are Databricks-bound (need a workspace client + SQL warehouse), so an OBO user identity is required in a deployed app.

| ToolKind | Class | Purpose |
|----------|-------|---------|
| `table_discovery` | `TableDiscoveryTool` | List exposed tables; register DISCOVERED bindings. |
| `table_search` | `TableSearchTool` | Substring (`LIKE`) search over a binding's content column. |
| `table_read` | `TableReadTool` | Filter / project / order / paginate rows. |
| `table_neighbors` | `TableNeighborsTool` | Sibling rows around an anchor by partition + order. |
| `table_load` | `TableLoadTool` | Materialize specific row(s) into the compute namespace. |
| `table_aggregate` | `TableAggregateTool` | `count`/`sum`/`avg`/`min`/`max` with optional `GROUP BY`. |

See [SQL / Table Tools](sql-table-tools.md) for full parameters and the `text_table` internals.

---

## Declaring Tools in YAML

Tools are declared in the top-level `tools:` section of a workflow YAML file. Agent nodes reference tools by name.

```yaml
tools:
  # Web tools
  - name: web_search
    kind: web_search
    config:
      brave_api_key: "${BRAVE_API_KEY}"
      max_results: 10
      domain_filter:
        - "*.gov"
        - "*.edu"

  - name: web_crawl
    kind: web_crawl
    config:
      max_content_length: 50000
      timeout: 30

  # File search
  - name: file_search
    kind: file_search
    config:
      top_k: 10

  # Enterprise data sources
  - name: earnings_index
    kind: vector_search
    config:
      index_name: prod_catalog.finance.earnings_idx
      num_results: 10
      columns: [id, content, title, url]

  - name: sales_genie
    kind: genie
    config:
      space_id: "01ef1234abcd5678"

  - name: policy_kb
    kind: knowledge_assistant
    config:
      endpoint_name: "policy-qa-endpoint"
```

Agent nodes reference tools by name:

```yaml
nodes:
  - id: researcher
    type: agent
    config:
      subtype: researcher
      tools: [web_search, web_crawl, earnings_index, sales_genie]
```

### Legacy Syntax

The older `type`/`name` syntax is still supported for backward compatibility:

```yaml
tools:
  - type: builtin
    name: web_search
  - type: enterprise
    name: my_vector_search
```

## ToolKind to SourceKind Mapping

The framework maps each `ToolKind` to a `SourceKind` that controls query generation strategy, admission policy, and result formatting:

| ToolKind | SourceKind | Query Style |
|----------|------------|-------------|
| `web_search` | `web` | Keyword/BM25 search |
| `web_crawl` | `builtin` | Not a data source (fetches known URLs) |
| `file_search` | `file` | Keyword search over uploaded files |
| `vector_search` | `vector_index` | Semantic embedding queries |
| `genie` | `sql_analytics` | NL-to-SQL, structured tabular results |
| `knowledge_assistant` | `qa_assistant` | NL question to NL answer |
| `table_discovery` | `text_table` | List/register bound Delta tables |
| `table_search` | `text_table` | Substring search over a content column |
| `table_read` | `text_table` | Filter / project / paginate rows |
| `table_neighbors` | `text_table` | Sibling rows around an anchor |
| `table_load` | `text_table` | Materialize rows for downstream compute |
| `table_aggregate` | `text_table` | Aggregations with optional `GROUP BY` |

## See Also

- [Tool System](../concepts/tool-system.md)
- [Search Providers](search-providers.md)
- [SQL / Table Tools](sql-table-tools.md)
- [Custom Tools](custom-tools.md)
- [Enterprise Data Sources](enterprise-data-sources.md)
- [Tool Protocol Reference](../reference/tool-protocol-reference.md)
