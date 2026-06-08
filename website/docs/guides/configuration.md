# Configuration

Almost everything is centralized in `databricks-deep-research-app/config/app.yaml`.
A separate `config/app.test.yaml` holds fast models and minimal iterations for tests.

## Model endpoints and tiers

Declare endpoints (with rate limits), then group them into tiers with fallback:

```yaml
# Model endpoints with rate limits
endpoints:
  databricks-llama-70b:
    endpoint_identifier: databricks-meta-llama-3-1-70b-instruct
    max_context_window: 128000
    tokens_per_minute: 200000

# Model tiers with fallback
models:
  simple:
    endpoints: [databricks-gemini-flash]
  analytical:
    endpoints: [databricks-claude-sonnet]
    temperature: 0.7
    fallback_on_429: true
  complex:
    endpoints: [databricks-claude-sonnet-er]
```

The three tiers — **simple**, **analytical**, **complex** — back the
[tiered routing](../concepts/architecture.md#tiered-model-routing) described in the
architecture. Each can list multiple endpoints for automatic failover on rate limits.

## Research depth profiles

```yaml
research_types:
  light:
    steps: { min: 1, max: 3 }
    researcher: { mode: classic }
  medium:
    steps: { min: 3, max: 6 }
    researcher: { mode: react }
  extended:
    steps: { min: 5, max: 10 }
    researcher: { mode: react, max_tool_calls: 20 }
```

See [research depths and researcher modes](../getting-started/quickstart.md#pick-a-research-depth)
for what these mean in practice.

## Search provider

```yaml
search:
  provider: databricks   # databricks (default) | brave | jina
```

Full details — including per-tool overrides and domain allowlists — are on the
[Web Search Providers](web-search-providers.md) page.

## Citation pipeline tuning

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `citation_verification.max_evidence_chars` | int (200–10000) | 3000 | Pipeline-wide cap on evidence quote length, applied to all truncation sites |

```yaml
citation_verification:
  max_evidence_chars: 3000   # raise for richer tabular corpora
```

This can be overridden per agent in a custom-agent YAML via
`config.citation_pipeline.max_evidence_chars`.

## Environment variables

Set in `.env` for local development; in `app.yaml` for deployed apps:

| Variable | Purpose |
|----------|---------|
| `DATABRICKS_CONFIG_PROFILE` | Profile-based auth (recommended) |
| `DATABRICKS_HOST` / `DATABRICKS_TOKEN` | Token-based auth (alternative) |
| `LAKEBASE_INSTANCE_NAME` / `LAKEBASE_DATABASE` | Lakebase connection |
| `BRAVE_API_KEY` | Only for `provider: brave` |
| `APP_CONFIG_PATH` | Path to `app.yaml` (default `config/app.yaml`) |
| `LOG_LEVEL` | Logging verbosity |
| `MLFLOW_TRACKING_URI` | `databricks` enables automatic tracing |

## Go deeper

- [Configuration (full docs)](https://github.com/mshtelma/databricks-deep-research-agent/blob/main/docs/configuration.md)
- [Model configuration (framework)](https://github.com/mshtelma/databricks-deep-research-agent/blob/main/databricks-deep-research/docs/guides/model-configuration.md)
