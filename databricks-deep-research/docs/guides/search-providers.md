# Search Providers

> How the `web_search` / `web_research` tools obtain results, the three shipped
> backends, and the precedence rules that select one.

## Overview

The web tools (`web_search`, `web_research`) are **backend-agnostic**. Each takes a
single dependency — a `SearchClient` — injected through its constructor. The tool
never talks to an HTTP API directly; it only calls `SearchClient.search(...)`.
Three backends ship with the framework (`databricks`, `brave`, `jina`), and you
can supply your own.

Whatever the backend, results flow into the **same** downstream machinery —
`SearchClient.search()` → `WebSearchTool` / `WebResearchTool` (dedup, URL
registry, table extraction) → pool → `web_crawl` → citation pipeline. The choice
of provider affects latency, cost, and whether full page bodies arrive inline —
but not the shape of the research run.

---

## Provider comparison

| Provider | Backend | Latency / cost | Returns | Credentials |
|----------|---------|----------------|---------|-------------|
| `databricks` | Model-serving **built-in web search** (`DatabricksWebSearchAdapter`) | A *billed model generation* per query — far heavier than a search REST call (~8–16 billed searches per deep-research run) | Snippets + grounding citations (`content=None`; `web_crawl` fetches bodies) | Workspace serving endpoint (OBO / app identity); no external key |
| `brave` | Brave Web Search API (`BraveSearchAdapter`) | Fast REST call | Snippets only (`content=None`) | `BRAVE_API_KEY` (required) |
| `jina` | Jina Search API (`JinaSearchAdapter`) | REST call; runs Jina Reader per result | Snippets **and full page content** inline | `JINA_API_KEY` (optional — works key-less with per-IP limits) |

Notes:

- `databricks` is **pay-per-token endpoints only** — unavailable on
  provisioned-throughput / HIPAA-BAA / cross-region-disabled workspaces (pin
  `brave` or `jina` there).
- `jina` populates `SearchResult.content`, so `web_research` skips the crawl step
  for those hits; `brave` and `databricks` return snippet-only rows that
  `web_research` / `web_crawl` then fetch.

---

## The `SearchClient` protocol

Every backend satisfies one `@runtime_checkable` protocol, defined in
`databricks-deep-research/src/databricks_deep_research/tools/builtins/web_search.py`:

```python
class SearchClient(Protocol):
    async def search(
        self,
        query: str,
        *,
        count: int = 10,
        freshness: str | None = None,   # "pd" | "pw" | "pm" | None
    ) -> list[SearchResult]: ...
```

A `SearchResult` is a frozen dataclass: `url`, `title`, `snippet`,
`relevance_score` (default `0.5`), and an optional `content` field carrying full
page text when the provider returns it.

To plug in your own backend, implement `search(...)` returning
`list[SearchResult]` and pass the instance as `search_client` to `WebSearchTool` /
`WebResearchTool`, or expose it on `ToolFactoryContext.search_client`. `freshness`
may be accepted-and-ignored if your backend has no time filter (as
`JinaSearchAdapter` does).

---

## Provider precedence

When a web tool is created, the effective provider is resolved high → low:

1. **Per-tool `config.provider`** on the `web_search` / `web_research`
   declaration.
2. **Global `app.yaml` `search.provider`** (the workspace default).
3. **Built-in `databricks` default** — `DEFAULT_SEARCH_PROVIDER` in the app's
   `core/app_config.py`, centralized by `resolve_effective_provider(...)`.

> **Framework vs. app default — get this right.** The *application*
> (`databricks-deep-research-app`) defaults `search.provider` to `databricks`
> (built-in web search, no external subscription needed). The *framework package
> itself does not default to databricks*: `ToolFactoryContext.from_defaults()`
> auto-creates a `BraveSearchAdapter` **only when `BRAVE_API_KEY` is set**,
> leaving `search_client = None` otherwise. The framework simply accepts any
> `SearchClient`; the app layer is what selects `databricks` by default.

**Inherited vs. explicit, in the factory** (`tools/factories/builtin.py`,
`BuiltinToolFactory.create` + `_resolve_search_provider`):

- A web tool with **no** `config.provider` is *inherited*: the factory uses the
  pre-built `ctx.search_client`. The app points that client at the global
  provider's backend at runtime, so changing `app.yaml search.provider` /
  `search.databricks.endpoint` re-points every inheriting tool. Inherited config
  is **never stamped/baked** onto the tool — it stays a live global lever.
- A web tool with an **explicit** `config.provider` is resolved by
  `_resolve_search_provider(provider, ctx, config)`, which constructs the named
  backend (`brave` / `jina` / `databricks`) from `config` + factory context.
  Unknown providers raise `ValueError` (supported set: `{brave, jina, databricks}`).

---

## The `databricks` sub-config

When `provider: databricks` is selected, `DatabricksWebSearchAdapter`
(`tools/builtins/databricks_web_search.py`) runs the search as a model-serving
call. Two transports are auto-selected by **model family** (inferred from the
endpoint name; override with `model_family`):

- **OpenAI** (`databricks-gpt-5*`) — the **Responses API**
  (`responses.create(tools=[{"type": "web_search"}])`). Citations carry **direct,
  crawlable URLs**. Agentic / slower.
- **Gemini** (`databricks-gemini-*`, default) — the **native `generateContent`**
  surface (`tools=[{"google_search": {}}]`). A single fast call. Citations are
  `vertexaisearch.cloud.google.com/...` **redirects**, resolved to canonical
  publisher URLs via a cheap 302 `Location` lookup (see `resolve_redirects`).

> The Gemini *OpenAI-compatible* `chat.completions` path returns no grounding
> metadata and is deliberately unsupported for sourcing.

Sub-config keys (per-tool `config`, defaulting from the app's `search.databricks`
block via `fill_databricks_search_defaults`):

| Key | Type | Default | Notes |
|-----|------|---------|-------|
| `model` | `str` | `databricks-gemini-3-1-flash-lite` (app `search.databricks.endpoint`) | Serving endpoint that performs the search. Required if no app default / `DATABRICKS_WEB_SEARCH_ENDPOINT` env var. |
| `model_family` | `"openai" \| "gemini"` | auto-detected from `model` | Must match the endpoint family, or the wrong transport hits the endpoint (hard 400). |
| `timeout_seconds` | `float` | `30.0` | Per-call wall-clock budget; the call returns `[]` on timeout. |
| `max_results` | `int` (1–20) | `10` | Hard cap on returned rows. `web_research` floors this to its `total_results`. |
| `resolve_redirects` | `bool` | `true` | Resolve Gemini grounding-redirect URLs to canonical (no-op for OpenAI). |
| `max_concurrency` | `int` (1–32) | `4` | Process-wide cap on concurrent built-in-search generations. Exported as env `DBX_WEBSEARCH_MAX_CONCURRENCY`, read lazily by the adapter's semaphore. |

`fill_databricks_search_defaults` only fills **absent** keys — it never
overwrites an explicit per-tool value (e.g. a deliberate `resolve_redirects:
false` or a smaller `timeout_seconds`).

---

## Configuration examples

**Inherit the workspace provider** (recommended — keeps the global lever live):

```yaml
tools:
  - name: web_search
    kind: web_search
    config:
      max_results: 10        # no `provider` => uses ctx.search_client
```

**Pin to Brave** (fast REST; needs `BRAVE_API_KEY`):

```yaml
tools:
  - name: web_search
    kind: web_search
    config:
      provider: brave
      max_results: 10
      domain_filter:
        - "*.gov"
        - "reuters.com"
```

**Pin to Databricks built-in web search** with an explicit OpenAI endpoint:

```yaml
tools:
  - name: web_research
    kind: web_research
    config:
      provider: databricks
      model: databricks-gpt-5
      model_family: openai      # match the endpoint; else auto-detected
      timeout_seconds: 45
      total_results: 12
```

---

## Credentials

| Provider | Requirement |
|----------|-------------|
| `databricks` | None beyond the workspace serving endpoint. Built-in search runs as the **app / service-principal** identity (the same one LLM calls use), via `ctx.serving_client_provider` or a `workspace_client`-derived client — not the OBO user, since it queries the public web. |
| `brave` | `BRAVE_API_KEY` env var (or `api_keys["brave"]` on `ToolFactoryContext`). Resolution raises `ValueError` if missing. |
| `jina` | `JINA_API_KEY` optional — without it, rate limits are per-IP rather than per-key. |

A default (databricks) deployment needs **no Brave secret**: the Brave key is
bound at deploy time only when a web tool *explicitly* pins `provider: brave`.

---

## See also

- [Builtin Tools](builtin-tools.md) — `web_search`, `web_crawl`, `web_research`
  constructor parameters and return shapes.
- [Tool Protocol Reference](../reference/tool-protocol-reference.md) — the
  `ResearchTool` protocol, `ToolDefinition`, `ToolResult`, and `ToolFactoryContext`.
- [Tool System](../concepts/tool-system.md) — resolver, registry, and factory chain.
