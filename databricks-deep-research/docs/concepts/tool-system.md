# Tool System

> The protocol, registry, and factories that power tool execution.

## Overview

Tools are how agents interact with the outside world. The framework defines a `ResearchTool` protocol that all tools must implement, a `ToolResolver` for unified lookup, a `ToolRegistry` for builtin/external tool storage, and a `ToolFactory` protocol for constructing tools from YAML declarations.

## ResearchTool Protocol

Every tool -- builtin, UC, or enterprise -- implements the `ResearchTool` protocol (`tools/protocol.py`). It is a `@runtime_checkable` protocol with three members:

| Member | Signature | Purpose |
|--------|-----------|---------|
| `definition` | `@property -> ToolDefinition` | Identity + JSON Schema for LLM function-calling |
| `validate_arguments` | `(arguments: dict) -> dict` | Clean/transform raw LLM args before execution; raises `ValueError` on invalid input |
| `execute` | `async (arguments: dict, context: ToolContext) -> ToolResult` | Async execution returning structured results |

Tool *dependencies* (search clients, tokens, domain filters) are injected via the tool constructor -- **not** via `ToolContext`. `ToolContext` carries only per-call values that change between invocations.

### ToolDefinition

Combines identity and schema for LLM function calling. Frozen dataclass.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | -- | Unique tool identifier |
| `description` | `str` | -- | Human-readable description for LLM |
| `parameters` | `dict[str, Any]` | -- | JSON Schema for tool arguments |
| `source_type` | `str` | `"builtin"` | Origin category: `builtin`, `uc_function`, `uc_tool`, `enterprise` |
| `source_kind` | `str` | `"builtin"` | Query modality (`SourceKind` value) |
| `metadata` | `dict[str, Any]` | `{}` | Arbitrary extra metadata |

### ToolResult

Returned by `execute()`. Frozen dataclass.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `content` | `str` | -- | Textual content for the LLM |
| `success` | `bool` | `True` | Whether execution succeeded |
| `sources` | `list[SourceInfo]` | `[]` | Source references discovered |
| `data` | `dict[str, Any]` | `{}` | Structured data (tables, metadata) |
| `error` | `str \| None` | `None` | Error message on failure |

### SourceInfo

A source reference attached to a `ToolResult`. Frozen dataclass.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `url` | `str` | -- | Source URL |
| `canonical_url` | `str \| None` | `None` | Canonical/deduplicated URL |
| `title` | `str` | `""` | Source title |
| `snippet` | `str` | `""` | Short excerpt |
| `content` | `str \| None` | `None` | Full content if available |
| `source_type` | `str` | `"web"` | Category: `web`, `enterprise`, `file`, etc. |
| `source_kind` | `str \| None` | `None` | `SourceKind` value; preferred over `source_type` for routing/admission |
| `relevance_score` | `float \| None` | `None` | Upstream relevance score |

### ToolContext

Per-call context passed to `execute()`. Frozen dataclass.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | `str` | `""` | Current research query |
| `url_registry` | `UrlRegistry \| None` | `None` | Shared URL index for the workflow run |
| `current_step` | `Any \| None` | `None` | Current research step metadata |
| `background_summary` | `str` | `""` | Background context summary |
| `recent_observations` | `list[str]` | `[]` | Recent findings from previous steps |
| `discovered_sources` | `list[Any]` | `[]` | Sources found so far |

### ToolRef

Reference to a tool by type and name, used in legacy YAML workflow configs:

```python
ToolRef(type="builtin", name="web_search")
ToolRef(type="uc_function", name="catalog.schema.my_function")
```

## Enums

### SourceKind

Defines how a tool should be queried and how its results should be interpreted. Drives query generation strategy, admission policy, and result formatting.

| Value | Description |
|-------|-------------|
| `web` | Keyword/BM25 search (Brave, Google) -- synthetic relevance scores |
| `vector_index` | Semantic embedding queries -- trusts upstream `relevance_score` |
| `sql_analytics` | NL-to-SQL (Genie) -- structured tabular results |
| `qa_assistant` | NL question to NL answer (Knowledge Assistant, endpoints) -- prose answers |
| `file` | Keyword search over uploaded files |
| `builtin` | Framework internals (pool tools, crawl) -- not a data source |

### ToolKind

Well-known tool kinds for YAML `tools:` declarations. Maps 1:1 with concrete tool implementations. Custom kinds (not in this enum) are also supported since the `kind` field on `ToolDeclaration` is typed as `str`, not constrained to this enum.

| Value | Maps to SourceKind |
|-------|--------------------|
| `web_search` | `web` |
| `web_crawl` | `builtin` |
| `file_search` | `file` |
| `vector_search` | `vector_index` |
| `genie` | `sql_analytics` |
| `knowledge_assistant` | `qa_assistant` |
| `custom` | `builtin` (default for unknown kinds) |

The mapping is performed by `tool_kind_to_source_kind(kind)`, which returns `"builtin"` for unknown kinds.

## URL Registry

`UrlRegistry` (`tools/protocol.py`) is a lightweight integer-to-URL map shared across all tool calls within a single workflow run. Its purpose is **security**: the LLM sees integer indices only, never raw URLs, which prevents hallucinated URL injection.

### How It Works

Internally backed by a list (O(1) index lookup) and a reverse dict (deduplication):

| Method | Signature | Behavior |
|--------|-----------|----------|
| `register(url)` | `str -> int` | Register a URL and return its integer index. Returns the existing index if the URL was already registered (deduplication). |
| `resolve(index)` | `int -> str \| None` | Resolve an index back to a URL. Returns `None` if not found. |
| `get_all()` | `-> list[tuple[int, str]]` | All `(index, url)` pairs in registration order. |
| `__len__` | `-> int` | Number of registered URLs |
| `__contains__` | `url in registry` | Membership test by URL string |

### Preventing URL Hallucination

1. `web_search` discovers URLs and registers them via `register(url)`, getting back integer indices.
2. The LLM sees results with `[0]`, `[1]`, `[2]` markers -- never raw URLs.
3. When the LLM asks to crawl a page, it passes the integer index.
4. `web_crawl` calls `resolve(index)` to get the real URL for fetching.
5. If the LLM invents an index that was never registered, `resolve()` returns `None` and the tool can reject the request.

The registry is created once per workflow execution and shared across all tool calls for consistency.

## Tool Resolver

`ToolResolver` (`tools/resolver.py`) is the single entry point for tool resolution in the executor. It handles both new-style name strings (from YAML `tools:` declarations) and legacy `{type, name}` dicts for backward compatibility.

### Resolution Order

When `resolve(ref)` is called, it checks sources in this priority:

1. **Overrides** -- app-injected tools registered via `override(name, tool)` (highest priority).
2. **Cache** -- previously resolved declarations (avoids re-creating tools).
3. **Declarations** -- created via the factory chain from the YAML `tools:` section. Each factory is consulted in order; the first one that `supports(kind)` and successfully `create()`s the tool wins.
4. **Legacy fallback** -- `ToolRegistry` for old `{type, name}` dicts.

If no source can resolve the tool, a `ValueError` is raised with a message listing declared tools, overrides, and registered factories.

### Key Methods

| Method | Description |
|--------|-------------|
| `override(name, tool)` | Register a runtime override (highest priority) |
| `resolve(ref)` | Resolve a single tool name or legacy dict to a `ResearchTool` |
| `resolve_many(refs)` | Resolve multiple refs; collects errors instead of failing on first |
| `initialize()` | Eagerly create all declared tools (optional pre-warming) |
| `list_available()` | Return all resolvable tool names (overrides + declarations + cache + legacy builtins) |

### Constructor

```python
ToolResolver(
    declarations: list[ToolDeclaration] | None,  # From YAML tools: section
    factories: list[ToolFactory] | None,          # Ordered factory chain
    factory_context: ToolFactoryContext | None,    # Runtime dependencies
    legacy_registry: ToolRegistry | None,          # Backward compat
)
```

## Tool Registry (Legacy)

`ToolRegistry` (`tools/registry.py`) is the lower-level store for tool instances. It separates builtin and external tools and caches resolved instances for the lifetime of a workflow execution.

| Method | Description |
|--------|-------------|
| `register_builtin(name, tool)` | Register a builtin tool |
| `register_external(name, tool)` | Register an external (enterprise/UC) tool |
| `resolve(ref: ToolRef)` | Resolve a `ToolRef` to a concrete instance (with caching) |
| `resolve_many(refs)` | Resolve a list of tool-ref dicts |
| `get_all_builtins()` | Return all builtin tools as `dict[str, ResearchTool]` |
| `has(name)` | Check if a tool is registered (builtin or external) |

The `ToolResolver` wraps `ToolRegistry` as a fallback and adds factory-based creation on top.

## Tool Factories

Factories create `ResearchTool` instances from `ToolDeclaration` entries in the YAML `tools:` section.

### ToolFactory Protocol

```python
class ToolFactory(Protocol):
    def supports(self, kind: str) -> bool: ...
    async def create(self, declaration: ToolDeclaration, context: ToolFactoryContext) -> ResearchTool: ...
```

Each factory declares which `kind` values it supports. The `ToolResolver` consults factories in registration order when a tool name maps to a `ToolDeclaration`.

### ToolFactoryContext

Runtime dependencies available to factories at tool creation time. Fields are optional; factories validate the ones they need and raise `ValueError` with a clear message if a required dependency is missing.

| Field | Type | Description |
|-------|------|-------------|
| `workspace_client` | `Any \| None` | `databricks.sdk.WorkspaceClient` for Databricks API calls |
| `user_token` | `str \| None` | OBO token for authenticated calls |
| `search_client` | `Any \| None` | Search client (e.g., Brave) for `web_search` |
| `crawler` | `Any \| None` | Content crawler for `web_crawl` |
| `file_index` | `Any \| None` | File index for `file_search` |
| `extras` | `dict[str, Any]` | App-specific dependencies |

The convenience constructor `ToolFactoryContext.from_defaults()` auto-detects:
- `workspace_client` from `databricks.sdk.WorkspaceClient()` defaults
- `search_client` from a `brave_api_key` parameter or the `BRAVE_API_KEY` environment variable (creates a `BraveSearchAdapter`)
- `crawler` is always `None` -- `WebCrawlTool` uses its built-in httpx + trafilatura pipeline

All auto-detection is wrapped in try/except so missing dependencies result in `None` fields. Errors surface later only if a factory actually needs the missing dependency.

### ToolDeclaration

A Pydantic model from the YAML `tools:` section that feeds into factory creation:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | -- | Unique tool name, referenced in agent configs |
| `kind` | `str` | -- | `ToolKind` value or custom string |
| `config` | `dict[str, Any]` | `{}` | Kind-specific configuration (e.g., `index_name`, `num_results`) |
| `description` | `str` | `""` | Human-readable, injected into the tool definition |

## Builtin Tools

| Tool | Kind | Source Kind | Required Config | Description |
|------|------|-------------|-----------------|-------------|
| web_search | `web_search` | `web` | `BRAVE_API_KEY` env var | Brave Search API |
| web_crawl | `web_crawl` | `builtin` | -- | trafilatura HTML extraction via httpx |
| file_search | `file_search` | `file` | `file_index` in context | Search uploaded files |
| vector_search | `vector_search` | `vector_index` | `index_name` in config | UC Vector Search index |
| genie | `genie` | `sql_analytics` | `space_id` in config | NL-to-SQL via Genie |
| knowledge_assistant | `knowledge_assistant` | `qa_assistant` | `endpoint_name` in config | Knowledge Assistant endpoint |

See [Builtin Tools Guide](../guides/builtin-tools.md) for detailed usage and configuration.

## Tool Execution Flow

```
Agent (ReAct loop)
  |
  v
ToolResolver.resolve(tool_name)        -- find the tool by name
  |
  v
tool.validate_arguments(raw_args)       -- clean / transform LLM args
  |
  v
tool.execute(validated_args, context)   -- async execution
  |                    |
  |                    v
  |          UrlRegistry.register(source.url)   -- register discovered URLs
  |                    |
  |                    v
  |          ToolResult { content, sources, data }
  |
  v
Return to agent with [idx] references  -- LLM sees indices, not URLs
```

1. The ReAct loop receives a tool call from the LLM with a tool name and arguments.
2. `ToolResolver.resolve()` walks the resolution chain (overrides, cache, declarations/factories, legacy registry) to find the `ResearchTool` instance.
3. `validate_arguments()` cleans and transforms the raw LLM arguments, raising `ValueError` if they are invalid.
4. `execute()` runs the tool asynchronously, receiving validated arguments and a `ToolContext` with the current query and shared `UrlRegistry`.
5. During execution, the tool registers any discovered source URLs in the `UrlRegistry`, getting back integer indices.
6. The `ToolResult` is returned with content formatted using `[idx]` markers that reference the registered URLs.
7. The agent incorporates the result and continues its reasoning loop.

## See Also

- [Agent System](agent-system.md) -- How agents call tools
- [Builtin Tools Guide](../guides/builtin-tools.md)
- [Custom Tools Guide](../guides/custom-tools.md)
- [Tool Protocol Reference](../reference/tool-protocol-reference.md)
