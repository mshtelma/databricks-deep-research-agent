# Pool Configuration

> Advanced pool setup: dedup strategies, capacity limits, injection formatting, and pool tools.

## Overview
Pools are the shared memory that connects agents. This guide covers advanced configuration.

## Pool Declaration
```yaml
pools:
  - name: sources
    item_type: source
    dedup_key: url
    max_items: 100
```

### PoolConfig Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | *(required)* | Unique pool identifier. Referenced by `pool_writes`, `pool_inject`, and `pool_tools` elsewhere in the workflow. |
| `item_type` | `str` | `"text"` | Semantic label for items in the pool. Common values: `text`, `source`, `claim`, `evidence`. Used for documentation and logging only; does not enforce a schema. |
| `dedup_key` | `str \| null` | `null` | Field name to use for key-based dedup. When set, the pool extracts this field from each incoming item (dict key or object attribute) and rejects items whose key has already been seen. Set to `null` to disable key-based dedup. |
| `dedup_content_hash` | `bool` | `true` | When `true`, the pool computes an MD5 hash of the JSON-serialised item and rejects exact content duplicates. Works independently of `dedup_key`. |
| `max_items` | `int` | `0` | Maximum number of items the pool will hold. `0` means unlimited. When the limit is reached, the oldest item is evicted (FIFO) before the new item is appended. |

## Dedup Strategies

### Key-Based Dedup
- Set `dedup_key` to a field name (e.g., `"url"`)
- Extracts that field from each item (supports both dicts and objects with attributes), rejects duplicates
- O(1) via set lookup on the stringified key value

```yaml
pools:
  - name: sources
    dedup_key: url     # Items with duplicate "url" values are silently dropped
```

### Content-Hash Dedup
- Set `dedup_content_hash: true` (this is the default)
- MD5 hash of the JSON-serialised item (keys sorted, `default=str`)
- Catches semantically identical items even when no single field uniquely identifies them

```yaml
pools:
  - name: observations
    dedup_content_hash: true   # Default; shown here for clarity
```

### Both Together
You can use both simultaneously for belt-and-suspenders dedup. The key check runs first; if the key is new, the content-hash check runs second. An item must pass both to be added.

```yaml
pools:
  - name: sources
    dedup_key: url
    dedup_content_hash: true   # Both active
```

### Disabling All Dedup
```yaml
pools:
  - name: raw_chunks
    dedup_key: null
    dedup_content_hash: false  # Accept everything, including duplicates
```

## Capacity Limits
- `max_items: <int>` (0 = unlimited)
- When the limit is reached, the **oldest** item is evicted (FIFO) before the new item is appended
- Important for long research runs with many sources to keep memory bounded

```yaml
pools:
  - name: sources
    max_items: 100    # Keep the 100 most recent sources
```

## Pool Writes (PoolWriteConfig)
Pool writes define how an agent pushes items into a pool after its LLM call completes.

```yaml
pool_writes:
  - pool: sources
    extract: sources      # Field to extract from agent output
    transform: null       # Optional transformation
  - pool: observations
    extract: findings
    transform: text
```

### PoolWriteConfig Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `pool` | `str` | *(required)* | Name of the target pool. Must match a pool declared in the `pools` section. |
| `extract` | `str` | *(required)* | Jinja / dot-path expression evaluated against the agent's output. The result should be a list of items (or a single item) to write into the pool. |
| `transform` | `str \| null` | `null` | Optional transformation applied to each extracted item before writing. Set to `"text"` to convert items to their string representation. `null` writes items as-is. |

### How It Works
1. After the agent produces output, the framework evaluates the `extract` expression against the output.
2. If the result is a list, each element is written individually; if a single value, it is written as one item.
3. If `transform` is set to `"text"`, each item is converted to a string before being added to the pool.
4. Dedup and capacity rules on the target pool apply to each write.

### Subtype Defaults
The built-in `researcher` subtype writes sources by default:
```yaml
pool_writes:
  - pool: sources
    extract: sources
```
Other subtypes (`coordinator`, `planner`, `reflector`, `synthesizer`, `evaluator`) have no default pool writes.

## Pool Injection (PoolInjectConfig)
Pool injection controls how pool contents are inserted into an agent's prompt before the LLM call.

```yaml
pool_inject:
  - pool: sources
    format: markdown
    max_items: 50
    max_item_chars: 500
    threshold: 0.0
```

### PoolInjectConfig Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `pool` | `str` | *(required)* | Name of the source pool. Must match a pool declared in the `pools` section. |
| `format` | `str` | `"text"` | How items are rendered into the prompt. One of: `"text"` (plain string), `"json"` (JSON array), `"markdown"` (markdown-formatted list). |
| `max_items` | `int` | `20` | Maximum number of items to inject. Items are selected by relevance (if a query is available) or recency. |
| `max_item_chars` | `int` | `0` | Per-item character truncation limit. `0` means no truncation; values greater than 0 truncate each item to that many characters. Useful for keeping prompt size bounded. |
| `threshold` | `float` | `0.0` | Minimum relevance score for search-based injection. Items scoring below this threshold are excluded. `0.0` includes all items. Only applies when the injection uses search (BM25 / hybrid) rather than chronological retrieval. |

### Format Options
- **`text`**: Each item is converted to its string representation, one per line.
- **`json`**: Items are serialised as a JSON array.
- **`markdown`**: Items are formatted as a markdown list with key fields highlighted.

## Pool Tools
When an agent declares `pool_tools`, the framework auto-generates tools that the agent can call during ReAct loops. Each declared pool name produces **five tools**, all prefixed with the pool name:

```yaml
config:
  subtype: researcher
  pool_tools:
    - sources    # Generates: sources_search, sources_get_recent, sources_count, sources_topics, sources_get_by_index
```

### Generated Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `{name}_search` | `query: str` (required), `limit: int` (default 10) | Search the pool by keyword query. Uses BM25+vector hybrid search when available, falls back to keyword overlap, then to chronological retrieval. |
| `{name}_get_recent` | `n: int` (default 10) | Get the N most recently added items from the pool. |
| `{name}_count` | *(none)* | Get the current number of items in the pool. |
| `{name}_topics` | *(none)* | Get unique topic/title labels extracted from pool items. Looks for `"topic"` or `"title"` fields in dict items. |
| `{name}_get_by_index` | `index: int` (required) | Get a specific item by its zero-based index. Returns an error if the index is out of range. |

### Search Degradation Tiers
The `{name}_search` tool uses the `PoolRegistry` for search when available. Search quality degrades gracefully across four tiers:

1. **Full hybrid** (BM25 + vector): `bm25s` installed and an embedding model is configured. BM25 retrieves candidates, then vector cosine similarity re-ranks them with configurable alpha weighting.
2. **BM25 only**: `bm25s` installed but no embedding model. Pure BM25 retrieval.
3. **Keyword fallback**: Neither `bm25s` nor embeddings available. Simple word-overlap scoring.
4. **Chronological**: Empty results from all above methods fall back to `get_recent()`.

### Subtype Defaults
- **`researcher`**: `pool_tools: ["pool_search"]` (searches the sources pool)
- **`synthesizer`**: `pool_tools: ["pool_search"]` (searches the sources pool)
- All other subtypes: no pool tools by default

## Common Patterns

### Sources + Observations
The standard two-pool pattern: `sources` for URL/title/snippet data, `observations` for extracted findings.

```yaml
pools:
  - name: sources
    item_type: source
    dedup_key: url
    max_items: 100
  - name: observations
    item_type: text
    dedup_content_hash: true

nodes:
  - name: researcher
    type: agent
    config:
      subtype: researcher
      pool_writes:
        - pool: sources
          extract: sources
        - pool: observations
          extract: findings
          transform: text
      pool_tools:
        - sources

  - name: synthesizer
    type: agent
    config:
      subtype: synthesizer
      pool_inject:
        - pool: sources
          format: markdown
          max_items: 50
        - pool: observations
          format: text
          max_items: 30
      pool_tools:
        - sources
        - observations
```

### Claims Pool
For the citation pipeline: verified claims with citation keys.

```yaml
pools:
  - name: claims
    item_type: claim
    dedup_key: claim_text
    max_items: 200
```

### Evidence Pool
For synthesis: ranked evidence spans that the synthesizer can search and cite.

```yaml
pools:
  - name: evidence
    item_type: evidence
    dedup_content_hash: true
    max_items: 150
```

## See Also
- [Pool System](../concepts/pool-system.md)
- [Agent System](../concepts/agent-system.md)
- [YAML Workflow Authoring](yaml-workflow-authoring.md)
