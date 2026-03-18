# Tool Protocol Reference

> Complete field-by-field reference for all tool-related types.

All types are defined in `databricks_deep_research.tools.protocol` unless noted otherwise.

---

## ResearchTool Protocol

`ResearchTool` is a `typing.Protocol` (decorated with `@runtime_checkable`) that every tool -- builtin, UC function, or enterprise connector -- must implement. Tool dependencies (search clients, tokens, domain filters) are injected via the tool constructor, **not** via `ToolContext`.

```python
@runtime_checkable
class ResearchTool(Protocol):

    @property
    def definition(self) -> ToolDefinition:
        """Tool definition combining name, description, and parameter schema."""
        ...

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Validate and potentially transform raw LLM arguments before execution.

        Returns:
            Validated / transformed arguments dict.

        Raises:
            ValueError: If arguments are invalid.
        """
        ...

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the tool with validated arguments.

        Args:
            arguments: Validated arguments matching ``self.definition.parameters``.
            context: Execution context (query, URL registry).

        Returns:
            ToolResult with content, success status, optional sources / data.
        """
        ...
```

### Method Summary

| Method | Kind | Returns | Description |
|--------|------|---------|-------------|
| `definition` | `@property` | `ToolDefinition` | Identity and JSON Schema for LLM function-calling. |
| `validate_arguments(arguments)` | sync | `dict[str, Any]` | Clean / transform raw LLM args before execution. Raises `ValueError` on invalid input. |
| `execute(arguments, context)` | `async` | `ToolResult` | Run the tool with validated arguments and per-call context. |

---

## ToolDefinition

Frozen dataclass. Combines tool identity with a JSON Schema for LLM function-calling.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | *(required)* | Unique tool name. Used in LLM tool-call payloads and YAML references. |
| `description` | `str` | *(required)* | Human-readable description. Included in the LLM system prompt for tool selection. |
| `parameters` | `dict[str, Any]` | *(required)* | JSON Schema describing the tool's input parameters. Passed to the LLM for function-calling. |
| `source_type` | `str` | `"builtin"` | Origin category: `"builtin"`, `"uc_function"`, `"uc_tool"`, or `"enterprise"`. |
| `source_kind` | `str` | `"builtin"` | Query modality. Should be a `SourceKind` value (e.g. `"web"`, `"vector_index"`). Drives query generation strategy and admission policy. |
| `metadata` | `dict[str, Any]` | `{}` | Arbitrary key-value metadata attached to the tool definition. |

---

## ToolResult

Frozen dataclass. Returned by `ResearchTool.execute()`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `content` | `str` | *(required)* | Primary textual output of the tool execution. Consumed by the agent as an observation. |
| `success` | `bool` | `True` | Whether the tool execution succeeded. `False` triggers error-handling paths in the agent. |
| `sources` | `list[SourceInfo]` | `[]` | Source references discovered during execution (URLs, documents, etc.). |
| `data` | `dict[str, Any]` | `{}` | Structured data payload (e.g. tabular results from Genie, parsed JSON). |
| `error` | `str \| None` | `None` | Human-readable error message when `success` is `False`. |

---

## SourceInfo

Frozen dataclass. Represents a single source reference within a `ToolResult`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `url` | `str` | *(required)* | Primary URL or URI of the source. |
| `canonical_url` | `str \| None` | `None` | Canonical (deduplicated) URL, if different from `url`. |
| `title` | `str` | `""` | Display title of the source. |
| `snippet` | `str` | `""` | Short text excerpt or summary from the source. |
| `content` | `str \| None` | `None` | Full text content, if fetched. |
| `source_type` | `str` | `"web"` | Origin category string: `"web"`, `"enterprise"`, `"file"`, etc. |
| `source_kind` | `str \| None` | `None` | `SourceKind` value. Preferred over `source_type` for routing and admission decisions. |
| `relevance_score` | `float \| None` | `None` | Upstream relevance/similarity score (0.0--1.0). Meaning depends on `source_kind`. |

---

## ToolContext

Frozen dataclass. Per-call context passed to `ResearchTool.execute()`. Only values that change between invocations belong here; tool dependencies are constructor-injected.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | `str` | `""` | The current research query or sub-question being investigated. |
| `url_registry` | `UrlRegistry \| None` | `None` | Shared URL-to-index map for the current workflow run. Tools register URLs and return indices; the LLM never sees raw URLs. |
| `current_step` | `Any \| None` | `None` | The current research step / plan item, if available. |
| `background_summary` | `str` | `""` | Background context summary from prior research. |
| `recent_observations` | `list[str]` | `[]` | Recent tool observations for context continuity. |
| `discovered_sources` | `list[Any]` | `[]` | Sources discovered so far in this research session. |

---

## ToolRef

Frozen dataclass. A reference to a tool by type and name, used in legacy YAML workflow configs.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | `str` | *(required)* | Tool origin: `"builtin"`, `"uc_function"`, `"uc_tool"`, or `"enterprise"`. |
| `name` | `str` | *(required)* | Tool identifier. Must match a name registered in `ToolRegistry`. |

**Examples:**

```python
ToolRef(type="builtin", name="web_search")
ToolRef(type="uc_function", name="catalog.schema.my_function")
```

> **Note:** New-style YAML workflows reference tools by plain name strings. `ToolRef` is retained for backward compatibility with `{type, name}` dict syntax.

---

## ToolFactoryContext

*Defined in `databricks_deep_research.tools.factory`.*

Mutable dataclass. Runtime dependencies available to tool factories at tool creation time. Fields are optional; factories validate the ones they need and raise `ValueError` with a clear message if a required dependency is missing.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `workspace_client` | `Any \| None` | `None` | Databricks `WorkspaceClient` instance for SDK calls. |
| `user_token` | `str \| None` | `None` | OBO (on-behalf-of) token for authenticated user calls. |
| `search_client` | `Any \| None` | `None` | Web search client implementing the `SearchClient` protocol (e.g. Brave). |
| `crawler` | `Any \| None` | `None` | Content crawler implementing the `ContentCrawler` protocol. |
| `file_index` | `Any \| None` | `None` | `FileIndex` instance for `file_search` tool. |
| `extras` | `dict[str, Any]` | `{}` | App-specific dependencies not covered by the named fields. |

### Class Method: `from_defaults`

```python
@classmethod
def from_defaults(
    cls,
    *,
    workspace_client: Any | None = None,
    user_token: str | None = None,
    brave_api_key: str | None = None,
    extras: dict[str, Any] | None = None,
) -> ToolFactoryContext:
```

Creates a context with auto-detected defaults:

- **`workspace_client`** -- If not provided, attempts `WorkspaceClient()` from the Databricks SDK (falls back to `None`).
- **`search_client`** -- Created as `BraveSearchAdapter` from the `brave_api_key` parameter or the `BRAVE_API_KEY` environment variable.
- **`crawler`** -- Always `None`; `WebCrawlTool` uses its built-in httpx + trafilatura pipeline when no crawler is injected.

All auto-detection is wrapped in try/except so missing dependencies result in `None` fields. Errors surface later only if a factory actually needs the missing dependency.

---

## UrlRegistry

Class (not a dataclass). Maps integer indices to URLs. Created per workflow execution, shared across all tool calls within a single run.

The LLM sees integer indices only -- never raw URLs -- which prevents hallucinated URL injection. Internally backed by a list for O(1) index lookup and a reverse dict for deduplication.

### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `__init__` | `() -> None` | -- | Create an empty registry. |
| `register` | `(url: str) -> int` | `int` | Register a URL and return its integer index. Returns the existing index if the URL was already registered (deduplication). |
| `resolve` | `(index: int) -> str \| None` | `str \| None` | Resolve an index back to its URL. Returns `None` if out of range. |
| `get_all` | `() -> list[tuple[int, str]]` | `list[tuple[int, str]]` | Return all `(index, url)` pairs in registration order. |
| `__len__` | `() -> int` | `int` | Number of registered URLs. |
| `__contains__` | `(url: str) -> bool` | `bool` | Check whether a URL is already registered. |
| `__repr__` | `() -> str` | `str` | E.g. `UrlRegistry(count=42)`. |

---

## Enums

### SourceKind

`StrEnum`. Describes how a tool should be queried and how its results should be interpreted. Drives query generation strategy, admission policy, and result formatting.

| Value | String | Description |
|-------|--------|-------------|
| `web` | `"web"` | Keyword / BM25 search (Brave, Google). Produces synthetic relevance scores. |
| `vector_index` | `"vector_index"` | Semantic embedding queries. Trust upstream `relevance_score`. |
| `sql_analytics` | `"sql_analytics"` | NL-to-SQL via Genie. Produces structured tabular results. |
| `qa_assistant` | `"qa_assistant"` | NL question to NL answer (Knowledge Assistants, endpoints). Returns prose answers. |
| `file` | `"file"` | Keyword search over uploaded files. |
| `builtin` | `"builtin"` | Framework internals (pool tools, crawl). Not a user-facing data source. |

### ToolKind

`StrEnum`. Well-known tool kinds for YAML `tools:` declarations. Maps 1:1 with concrete tool implementations. Custom kinds (not in this enum) are supported -- the `kind` field on `ToolDeclaration` is typed as `str`, not constrained to this enum.

| Value | String | Description |
|-------|--------|-------------|
| `web_search` | `"web_search"` | Web search via configured search client (e.g. Brave API). |
| `web_crawl` | `"web_crawl"` | Fetch and extract content from a URL by index. |
| `file_search` | `"file_search"` | Search over user-uploaded files. |
| `vector_search` | `"vector_search"` | Databricks Vector Search index query. |
| `genie` | `"genie"` | Databricks Genie NL-to-SQL analytics. |
| `knowledge_assistant` | `"knowledge_assistant"` | Databricks Knowledge Assistant endpoint. |
| `custom` | `"custom"` | Catch-all for externally defined tool kinds. |

---

## Type Aliases

### `tool_kind_to_source_kind`

```python
def tool_kind_to_source_kind(kind: str) -> str
```

Maps a `ToolKind` value to its corresponding `SourceKind` value. Returns `"builtin"` for unknown kinds.

**Mapping table:**

| ToolKind | SourceKind |
|----------|------------|
| `web_search` | `web` |
| `web_crawl` | `builtin` |
| `file_search` | `file` |
| `vector_search` | `vector_index` |
| `genie` | `sql_analytics` |
| `knowledge_assistant` | `qa_assistant` |

---

## See Also

- [Tool System](../concepts/tool-system.md)
- [Custom Tools](../guides/custom-tools.md)
- [Builtin Tools](../guides/builtin-tools.md)
