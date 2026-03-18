# Custom Tools

> Implement the `ResearchTool` protocol to add custom data sources.

## Overview

Any Python class implementing the `ResearchTool` protocol can be used as a tool
in workflows. Tools are discovered through YAML `tools:` declarations and
created at runtime by `ToolFactory` instances registered with the `ToolResolver`.

## The ResearchTool Protocol

The protocol is defined in `databricks_deep_research.tools.protocol` and has
three members:

```python
@runtime_checkable
class ResearchTool(Protocol):
    @property
    def definition(self) -> ToolDefinition:
        """Tool definition combining name, description, and parameter schema."""
        ...

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Validate and potentially transform arguments before execution."""
        ...

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the tool with validated arguments."""
        ...
```

Key design points:

- **Constructor DI** -- tool dependencies (search clients, tokens, domain
  filters) are injected at construction time, *not* via `ToolContext`.
- **validate then execute** -- `validate_arguments` returns the canonical input
  to `execute()`, combining validation and transformation so uncleaned args are
  never passed to execution.
- **Async execution** -- `execute()` is an `async` method. All I/O should use
  `await`.

## Step-by-Step Example

This walkthrough creates a Confluence wiki search tool.

### Step 1: Define the tool class

```python
from typing import Any

from databricks_deep_research.tools.protocol import (
    ResearchTool,
    SourceInfo,
    SourceKind,
    ToolContext,
    ToolDefinition,
    ToolResult,
)


class ConfluenceSearchTool:
    """Search Confluence wiki pages."""

    def __init__(self, base_url: str, api_token: str) -> None:
        self._base_url = base_url
        self._api_token = api_token

    # -- ResearchTool protocol ------------------------------------------------

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="confluence_search",
            description="Search Confluence wiki pages for relevant content",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "space": {
                        "type": "string",
                        "description": "Confluence space key (optional)",
                    },
                },
                "required": ["query"],
            },
            source_type="confluence",
            source_kind=SourceKind.web,
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = arguments.get("query")
        if not query or not isinstance(query, str):
            raise ValueError("'query' is required and must be a string")

        cleaned: dict[str, Any] = {"query": query.strip()}
        space = arguments.get("space")
        if space and isinstance(space, str):
            cleaned["space"] = space.strip()
        return cleaned

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        query = arguments["query"]
        space = arguments.get("space")

        try:
            results = await self._search_confluence(query, space)
        except Exception as e:
            return ToolResult(
                content=f"Confluence search failed: {e}",
                success=False,
                error=str(e),
            )

        # Register URLs in the shared registry so downstream tools can
        # resolve them by index (LLM never sees raw URLs).
        sources: list[SourceInfo] = []
        lines: list[str] = []
        for r in results:
            idx: int
            if context.url_registry is not None:
                idx = context.url_registry.register(r["url"])
            else:
                idx = len(lines)

            sources.append(
                SourceInfo(
                    url=r["url"],
                    title=r["title"],
                    snippet=r["excerpt"],
                    source_kind=SourceKind.web,
                )
            )
            lines.append(f"[{idx}] **{r['title']}**\n    {r['excerpt']}")

        content = "\n\n".join(lines) if lines else "No results found."
        return ToolResult(
            content=content,
            success=True,
            sources=sources,
            data={"query": query, "total_results": len(results)},
        )

    # -- internal helpers (not part of the protocol) --------------------------

    async def _search_confluence(
        self, query: str, space: str | None
    ) -> list[dict[str, str]]:
        """Call the Confluence REST API.  Implementation omitted for brevity."""
        ...
```

### Step 2: Register via ToolFactory (preferred)

A `ToolFactory` creates `ResearchTool` instances from YAML `ToolDeclaration`
objects. Implement the two-method protocol:

```python
from databricks_deep_research.tools.factory import (
    ToolFactory,
    ToolFactoryContext,
)
from databricks_deep_research.tools.protocol import ResearchTool
from databricks_deep_research.workflow.definition import ToolDeclaration


class ConfluenceToolFactory:
    """Factory that creates ConfluenceSearchTool from YAML declarations."""

    def supports(self, kind: str) -> bool:
        return kind == "confluence"

    async def create(
        self,
        declaration: ToolDeclaration,
        context: ToolFactoryContext,
    ) -> ResearchTool:
        config = declaration.config
        base_url = config.get("base_url")
        api_token = config.get("api_token")
        if not base_url or not api_token:
            raise ValueError(
                "confluence tool requires 'base_url' and 'api_token' in config"
            )
        return ConfluenceSearchTool(base_url=base_url, api_token=api_token)
```

Then pass the factory when building a `ToolResolver`:

```python
from databricks_deep_research.tools.resolver import ToolResolver

resolver = ToolResolver(
    declarations=workflow.tools,       # from parsed YAML
    factories=[ConfluenceToolFactory()],
    factory_context=ToolFactoryContext.from_defaults(),
)
```

The resolver will call your factory whenever it encounters a tool declaration
with `kind: confluence`.

### Step 3: Or inject directly via ToolResolver.override()

If you already have a constructed tool instance (for example in tests or when
wiring app-specific dependencies), bypass the factory chain entirely:

```python
tool = ConfluenceSearchTool(
    base_url="https://wiki.example.com",
    api_token="my-token",
)

resolver = ToolResolver(declarations=workflow.tools)
resolver.override("confluence_search", tool)
```

Overrides have the **highest priority** in the resolution order, ahead of
factory-created tools and legacy registry lookups.

You can also pass pre-built tools on `ExecutionContext.enterprise_tools`:

```python
from databricks_deep_research.workflow.context import ExecutionContext

ctx = ExecutionContext(
    llm_client=my_client,
    enterprise_tools=[tool],
)
```

### Step 4: Declare in YAML

Add the tool to the workflow's top-level `tools:` section. Agent nodes reference
it by `name`:

```yaml
tools:
  - name: confluence_search
    kind: confluence
    config:
      base_url: "https://wiki.example.com"
      api_token: "${CONFLUENCE_TOKEN}"
    description: "Search internal Confluence wiki"

nodes:
  - id: researcher
    type: agent
    config:
      subtype: researcher
      tools: [web_search, confluence_search]
```

Environment variable interpolation (`${VAR}`) is handled by the YAML loader.

## ToolContext

`ToolContext` is a frozen dataclass passed to `execute()` on every call. It
carries **per-call** values that change between invocations. Tool dependencies
that are stable for the lifetime of the tool belong in the constructor instead.

| Field | Type | Description |
|-------|------|-------------|
| `query` | `str` | The original user query (may be rewritten) |
| `url_registry` | `UrlRegistry \| None` | Shared index-to-URL map for the workflow run |
| `current_step` | `Any \| None` | Current research step metadata |
| `background_summary` | `str` | Summary from the background research phase |
| `recent_observations` | `list[str]` | Recent tool observations for context |
| `discovered_sources` | `list[Any]` | Sources discovered in earlier steps |

### UrlRegistry

The `UrlRegistry` is a lightweight integer-to-URL map shared across all tool
calls within a single workflow run. The LLM sees integer indices only -- never
raw URLs -- which prevents hallucinated URL injection.

```python
# In your tool's execute() method:
if context.url_registry is not None:
    idx = context.url_registry.register("https://example.com/page")
    # idx is a stable integer; use it in formatted output for the LLM
```

Key methods:
- `register(url) -> int` -- register a URL, returns its index (deduplicates)
- `resolve(index) -> str | None` -- look up a URL by index
- `get_all() -> list[tuple[int, str]]` -- all `(index, url)` pairs

## ToolResult

The return value of `execute()`. All fields except `content` have defaults.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `content` | `str` | *(required)* | Human-readable summary for the LLM |
| `success` | `bool` | `True` | Whether the tool call succeeded |
| `sources` | `list[SourceInfo]` | `[]` | Structured source metadata for citation tracking |
| `data` | `dict[str, Any]` | `{}` | Arbitrary structured data for downstream processing |
| `error` | `str \| None` | `None` | Error message when `success=False` |

Best practice: always return a `ToolResult` even on failure -- set
`success=False` and populate `error` rather than raising an exception. This lets
the LLM recover gracefully (e.g., try a different query).

## SourceInfo

Source metadata attached to `ToolResult.sources`. Used by the citation pipeline
to track provenance.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `url` | `str` | *(required)* | Primary URL of the source |
| `canonical_url` | `str \| None` | `None` | De-duplicated canonical URL |
| `title` | `str` | `""` | Human-readable title |
| `snippet` | `str` | `""` | Short excerpt / summary |
| `content` | `str \| None` | `None` | Full content (when available) |
| `source_type` | `str` | `"web"` | Source type label (web, enterprise, file, etc.) |
| `source_kind` | `str \| None` | `None` | `SourceKind` value; preferred over `source_type` for routing and admission |
| `relevance_score` | `float \| None` | `None` | Upstream relevance score (trusted for vector/semantic sources) |

## SourceKind

The `SourceKind` enum drives query generation strategy, admission policy, and
result formatting:

| Value | Use case | Example tools |
|-------|----------|---------------|
| `web` | Keyword / BM25 search | Brave, Google |
| `vector_index` | Semantic embedding queries | Databricks Vector Search |
| `sql_analytics` | NL-to-SQL | Genie |
| `qa_assistant` | NL question to NL answer | Knowledge Assistant |
| `file` | Keyword search over uploaded files | File search |
| `builtin` | Framework internals | Pool tools, web crawl |

## ToolFactoryContext

Runtime dependencies available to factories at tool creation time. All fields
are optional -- factories validate the ones they need and raise `ValueError` if
a required dependency is missing.

| Field | Type | Description |
|-------|------|-------------|
| `workspace_client` | `Any \| None` | Databricks `WorkspaceClient` |
| `user_token` | `str \| None` | OBO token for authenticated calls |
| `search_client` | `Any \| None` | `SearchClient` protocol (for web search) |
| `crawler` | `Any \| None` | `ContentCrawler` protocol (for web crawl) |
| `file_index` | `Any \| None` | `FileIndex` for file search |
| `extras` | `dict[str, Any]` | App-specific dependencies |

Use `ToolFactoryContext.from_defaults()` for auto-detection of workspace client
and Brave search credentials from environment variables.

## Best Practices

1. **Register URLs with the shared `url_registry`** -- the LLM should see
   integer indices, not raw URLs. This prevents URL hallucination and enables
   the `web_crawl` tool to resolve indices back to URLs.

2. **Return meaningful `content` for LLM consumption** -- the `content` string
   is what the LLM reads. Format it clearly with indices, titles, and snippets.

3. **Include `SourceInfo` for citation tracking** -- the citation verification
   pipeline uses `sources` to attribute claims. Without them, your tool's
   results cannot be cited.

4. **Handle errors gracefully** -- return `ToolResult(success=False, error=...)`
   instead of raising exceptions. This lets the LLM retry with different
   arguments or fall back to another tool.

5. **Use `validate_arguments` to catch bad LLM arguments** -- LLMs sometimes
   produce malformed tool calls. Validate eagerly and raise `ValueError` with
   clear messages.

6. **Inject dependencies via the constructor** -- keep `execute()` stateless
   aside from `ToolContext`. This makes tools testable with mock dependencies.

7. **Set `source_kind` on your `ToolDefinition`** -- this tells the framework
   how to route queries and format results for your tool's modality.

## See Also

- [Tool System](../concepts/tool-system.md)
- [Builtin Tools](builtin-tools.md)
- [Tool Protocol Reference](../reference/tool-protocol-reference.md)
