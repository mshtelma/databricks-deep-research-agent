# Web Search Providers

The builtin `web_search` tool can run on three interchangeable backends behind one
`SearchClient` protocol. All of them feed the same pool → crawl → citation pipeline.

The backend is chosen by `search.provider` in `app.yaml`.

| Provider | Backend | Notes |
|----------|---------|-------|
| **`databricks`** *(default)* | Model-serving **built-in web search** | Works out-of-the-box on a Databricks workspace with **no external search subscription**. Costs a billed model generation per query. |
| **`brave`** *(opt-in)* | Brave Web Search API | Fast REST search; requires `BRAVE_API_KEY`. |
| **`jina`** *(opt-in)* | Jina Search API | Returns full page content; `JINA_API_KEY` optional. |

!!! tip "Default needs no key"
    Because `databricks` is the default, a fresh deployment requires **no Brave or
    Jina account and no secret**. Brave/Jina are strictly opt-in.

## The `databricks` provider

```yaml
search:
  provider: databricks
  databricks:
    endpoint: databricks-gemini-3-1-flash-lite   # default; fast native grounding
    # endpoint: databricks-gpt-5                  # alternative: OpenAI Responses API
    max_concurrency: 4
```

- **Gemini endpoint** (default) — native `generateContent` grounding in a single fast
  call; redirect URLs are auto-resolved to canonical.
- **OpenAI endpoint** — Responses API; returns direct URLs and real titles, but is more
  agentic and slower.
- The model family is auto-detected from the endpoint name and reuses the framework
  LLM client's serving connection (OBO identity).

!!! warning "Workspace availability"
    The built-in search uses pay-per-token endpoints and is unavailable on
    provisioned-throughput, HIPAA-BAA, or cross-region-disabled workspaces. Set
    `provider: brave` or `jina` there.

## Provider precedence

From highest to lowest priority:

1. **Per-tool** `config.provider` on a `web_search` / `web_research` declaration
2. **Global** `app.yaml` → `search.provider`
3. **Built-in default** — `databricks`

A web tool with **no** `provider` *inherits at runtime* — so changing
`app.yaml`'s `search.provider` (or the databricks endpoint) re-points every inheriting
agent at once. Inherited config is never baked in on save, keeping the global lever live.

## Per-agent domain allowlists

An agent can constrain web search to a set of domains. The allowlist is both **pushed
into the engine** (for the OpenAI Responses API, via `allowed_domains`) and enforced
**post-hoc** as a safety net, so off-domain results are dropped even when the engine
can't filter natively. A soft domain-scope hint is also appended to the search
instruction. A bare `reuters.com` also matches `www.reuters.com`.

## Go deeper

- [Search providers (framework docs)](https://github.com/mshtelma/databricks-deep-research-agent/blob/main/databricks-deep-research/docs/guides/search-providers.md)
- [Configuration](configuration.md)
