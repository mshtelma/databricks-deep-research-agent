# Pool System

> Shared memory pools for cross-agent knowledge accumulation with dedup and search.

## Overview
Pools are named collections that agents write to and read from during workflow execution. They enable knowledge sharing between agents---a researcher writes sources and observations, and the synthesizer reads them.

## PoolConfig
`PoolConfig` is a Pydantic `BaseModel` (with `extra="forbid"`) that defines pool behavior:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | *(required)* | Unique pool identifier used to reference the pool in workflows and tools. |
| `item_type` | `str` | `"text"` | Semantic type hint for pool contents. Common values: `text`, `source`, `claim`, `evidence`. |
| `dedup_key` | `str \| None` | `None` | Field name for key-based deduplication. When set, the named field is extracted from each item and checked against a set of already-seen keys. |
| `dedup_content_hash` | `bool` | `True` | Enable content-hash deduplication. Each item is JSON-serialized (with sorted keys) and MD5-hashed; duplicates are rejected. |
| `max_items` | `int` | `0` | Capacity limit. `0` means unlimited. When exceeded, the oldest item is evicted (FIFO). |

## PoolState
`PoolState` holds the runtime data for a single pool. One instance is created per `PoolConfig`.

**Internal storage:**
- `items: list[Any]` -- ordered list of all pool items (append order).
- `seen_keys: set[str]` -- set of dedup key values already added (used when `dedup_key` is set).
- `seen_hashes: set[str]` -- set of MD5 content hashes already added (used when `dedup_content_hash` is `True`).
- `_lock: asyncio.Lock` -- protects bulk operations (`extend_async`).

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `add` | `(item: Any) -> bool` | Add a single item with dedup checks. Returns `True` if added, `False` if rejected as duplicate. Evicts the oldest item if `max_items` is exceeded. |
| `extend_async` | `async (items: list[Any]) -> int` | Bulk add under an async lock. Returns the count of items actually added (after dedup filtering). |
| `search` | `(query: str, limit: int = 10) -> list[Any]` | Keyword-based search using word overlap scoring. Splits query and item text into word sets and scores by intersection ratio. This is the fallback when BM25 is unavailable. |
| `get_recent` | `(n: int = 10) -> list[Any]` | Return the last `n` items in insertion order. |
| `count` | `() -> int` | Number of items currently in the pool. |
| `topics` | `() -> list[str]` | Extract unique topic labels. Looks for `"topic"` or `"title"` keys in dict items and returns a sorted list of unique values. |
| `get_by_index` | `(index: int) -> Any \| None` | Direct access by zero-based index. Returns `None` if out of range. |

## Deduplication
Two dedup strategies can operate independently or together:

1. **Key-based**: Extract the value of a named field (e.g., `"url"`) from the item (supports both `dict` and attribute access). Check against the `seen_keys` set. O(1) lookup.
2. **Content-hash**: JSON-serialize the item with `sort_keys=True` and `default=str`, compute an MD5 hex digest, and check against the `seen_hashes` set. O(1) lookup.
3. Both strategies can be active simultaneously. An item is rejected if *either* check finds a duplicate.

Example:
```yaml
pools:
  - name: sources
    item_type: source
    dedup_key: url           # Won't add duplicate URLs
    max_items: 100           # FIFO eviction after 100
  - name: observations
    item_type: text
    dedup_content_hash: true  # Won't add duplicate content
```

## PoolRegistry
`PoolRegistry` manages the mapping from pool names to `PoolState` instances and provides search with graceful degradation.

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm_client` | `Any` | `None` | Optional `FrameworkLLMClient` for embedding-based vector search. Typed as `Any` to avoid circular imports. |
| `alpha` | `float` | `0.6` | Weight for BM25 vs vector in hybrid search. `0.0` = vector only, `1.0` = BM25 only. |

**Methods:**

| Method | Description |
|--------|-------------|
| `initialize_from_configs(configs)` | Create pools eagerly from a list of config dicts (typically from `WorkflowDefinition.pools`). Skips duplicates with a warning. |
| `get(name) -> PoolState` | Get pool by name. Raises `KeyError` if not found. |
| `get_or_create(name, **kwargs) -> PoolState` | Lazy creation: returns an existing pool or creates one with default config. |
| `has(name) -> bool` | Check whether a pool exists. |
| `all_pools() -> dict[str, PoolState]` | Shallow copy of all registered pools. |
| `search(pool_name, query, top_k) -> list[Any]` | Search a pool using the best available method (see search tiers below). |

### Search Tiers (Graceful Degradation)
The registry attempts search in this order, falling back if a tier is unavailable or yields no results:

| Tier | Requirements | Method |
|------|-------------|--------|
| **1. Full hybrid** | `bm25s` + `numpy` installed, `llm_client` with `supports_embeddings` | BM25 retrieval, then re-rank with cosine similarity. Score: `alpha * bm25_rank + (1 - alpha) * cosine`. |
| **2. BM25 only** | `bm25s` + `numpy` installed | BM25 retrieval via `bm25s.BM25`. Tokenizes corpus and query, retrieves top-k by BM25 score. |
| **3. Keyword fallback** | None (always available) | `PoolState.search()` -- word-overlap scoring between query and JSON-serialized items. |
| **4. Chronological** | None (always available) | `PoolState.get_recent()` -- returns the most recent items regardless of query. |

### How the BM25 Index Works
The BM25 index is built on-the-fly for each search call:
1. Each pool item is converted to a text string (strings pass through, dicts are JSON-serialized).
2. The corpus is tokenized via `bm25s.tokenize()`.
3. A `bm25s.BM25` retriever is created and indexed.
4. The query is tokenized and retrieved against the corpus.
5. Results with score > 0 are returned sorted by descending score.

For hybrid search (Tier 1), the BM25 results are further re-ranked:
1. All BM25 result texts plus the query are embedded via `llm_client.embed()`.
2. Cosine similarity is computed between the query vector and each result vector.
3. BM25 rank scores are normalized to `[0, 1]` based on position.
4. Final score = `alpha * bm25_rank_score + (1 - alpha) * cosine_score`.
5. Results are sorted by final hybrid score.

## Pool Tools
Each pool gets five auto-generated tools, prefixed with the pool name. These tools implement the `ResearchTool` protocol and are injected into agent context via `create_pool_tools()`.

For a pool named `sources`, the tools are:

| Tool | Parameters | Description |
|------|-----------|-------------|
| `sources_search` | `query: str` (required), `limit: int` (default 10) | Search the pool by keyword query. Uses `PoolRegistry.search()` if a registry is provided (hybrid/BM25), otherwise falls back to `PoolState.search()` (keyword overlap). |
| `sources_get_recent` | `n: int` (default 10) | Get the N most recent items from the pool. |
| `sources_count` | *(none)* | Get the number of items currently in the pool. |
| `sources_topics` | *(none)* | Get unique topic labels extracted from pool items (looks for `"topic"` or `"title"` keys). |
| `sources_get_by_index` | `index: int` (required) | Get a specific item by zero-based index. Returns an error if the index is out of range. |

All tools return a `ToolResult` with JSON-serialized content and metadata in the `data` field.

The factory function:
```python
create_pool_tools(pool_name: str, pool: PoolState, *, registry: PoolRegistry | None = None) -> list[ResearchTool]
```

## Writing to Pools
Agents write to pools via `PoolWriteConfig`, which extracts fields from the agent's output and adds them to the target pool:

```yaml
config:
  subtype: researcher
  pool_writes:
    - pool: sources
      extract: sources        # Extract 'sources' field from output
    - pool: observations
      extract: findings
      transform: text         # Convert to plain text
```

## Reading from Pools (Injection)
Agents read from pools via `PoolInjectConfig`, which injects pool content into the agent's prompt before execution:

```yaml
config:
  subtype: synthesizer
  pool_inject:
    - pool: sources
      format: markdown
      max_items: 50
      max_item_chars: 500
      threshold: 0.0          # Minimum relevance (0 = all)
    - pool: observations
      format: text
      max_items: 30
```

## Typical Pool Patterns
1. **Sources pool**: Researcher writes URLs, titles, and snippets. Synthesizer reads for citation. Configured with `dedup_key: url` to avoid duplicate sources.
2. **Observations pool**: Researcher writes findings. Reflector reads for progress assessment. Synthesizer reads for report content. Configured with `dedup_content_hash: true` to avoid duplicate observations.
3. **Claims pool**: Citation pipeline writes verified claims. Configured with a `max_items` limit to bound memory usage.

## See Also
- [Agent System](agent-system.md) -- Pool writes and injection
- [Pool Configuration Guide](../guides/pool-configuration.md)
- [State Management](state-management.md)
