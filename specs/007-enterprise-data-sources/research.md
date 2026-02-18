# Research: Enterprise Data Sources

**Date**: 2026-02-04
**Status**: COMPLETE
**Updated**: 2026-02-04 (Added parallel tool execution research and SOTA multi-source discovery findings)

---

## 0. Multi-Source Discovery & Planning Research (2025-2026)

### 0.1 Databricks Instructed Retriever (January 2026)

The Databricks Instructed Retriever represents a significant advancement in RAG systems, achieving **70% improvement** over naive RAG on enterprise tasks. This research informs our multi-source discovery architecture.

**Key Capabilities:**

1. **Query Decomposition**: Break complex queries into a "search plan" with multiple subqueries
   - Complex queries like "Compare our Q3 revenue to competitors in the APAC region" become:
     - Subquery 1: "Our company Q3 revenue" → Vector Search (internal data)
     - Subquery 2: "Competitor revenue APAC Q3" → Web Search (external data)
   - Each subquery targets the most appropriate data source

2. **Metadata Reasoning**: Translate natural language to precise filters
   - "From last year" → `timestamp > 2025-01-01`
   - "By the engineering team" → `department = 'engineering'`
   - Critical for Vector Search indexes with rich metadata

3. **Contextual Relevance**: Use system instructions to boost relevant docs beyond keyword matching
   - LLM understands domain context and can identify relevant docs that don't match keywords
   - Example: "security vulnerabilities" also matches "CVE reports", "penetration testing results"

**Key Insight from Databricks Research:**
> "Errors come not from agents failing to reason, but failing to retrieve the right data"

This validates our approach of investing in better discovery and routing rather than more sophisticated reasoning.

**Architecture Implications:**
- Run discovery BEFORE planning to understand data landscape
- Generate source-specific subqueries, not one-size-fits-all queries
- Use metadata filtering aggressively for enterprise sources

### 0.2 Agentic RAG Best Practices (2025-2026)

Modern agentic RAG systems have evolved beyond simple retrieve-and-generate patterns. Key patterns relevant to our multi-source architecture:

**Query Planning Agent:**
- Dedicated agent that decomposes complex queries into parallelizable subqueries
- Each subquery specifies: target source, query text, expected result type
- Enables parallel execution across heterogeneous sources (SQL, vector, web)

**Multi-Agent Routing:**
- Different agents specialize in different sources
  - SQL Agent: Understands table schemas, can generate complex joins
  - Vector Agent: Understands semantic similarity, metadata filtering
  - Web Agent: Understands search engine behavior, result freshness
- Coordinator routes queries to appropriate specialist agents

**Dynamic Tool Selection:**
- Not all tools available for all queries
- Coordinator decides which tools/sources to invoke based on:
  - Query intent (factual vs. analytical vs. comparative)
  - Data freshness requirements
  - Source reliability for the domain
  - Previous results in the conversation

**Relevance Feedback Loop:**
- After each retrieval, evaluate result quality
- If insufficient, try alternative sources or reformulated queries
- Track which sources provided useful results per topic area

### 0.3 Deep Research SOTA (DeepPlanner, Step-DeepResearch)

Recent research on deep research systems provides guidance for our planning and execution architecture.

**DeepPlanner (arXiv 2510.12979):**
- High-level planning dramatically improves output quality vs. granular next-step prediction
- Plans should be "actionable but flexible" - specific enough to guide execution, general enough to adapt
- Key finding: 23% improvement when planner has source awareness

**Step-DeepResearch (arXiv 2512.20491v1):**
- Source selection should happen at planning time, not execution time
- Each step should specify which sources to consult with rationale
- "Discovery phase" before detailed planning improves plan quality by 18%
- Recommended flow: Discovery → Planning → Per-Step Execution → Reflection

**Multi-Source Fusion Patterns:**
- **Early Fusion**: Combine sources before LLM synthesis (our RRF approach)
- **Late Fusion**: Let LLM synthesize from multiple source outputs separately
- **Hybrid Fusion**: Early fusion for similar sources, late fusion for different types
- Recommendation: Hybrid - RRF within source types, LLM synthesis across types

### 0.4 Parallel Tool Execution Research (2025-2026)

Modern research agents and LLM tool-calling systems have evolved to support parallel tool execution, enabling latency reduction. This research informs our parallel execution strategy.

#### 0.4.1 Existing Architecture Analysis (CRITICAL)

**Before designing parallel execution, we must understand existing constraints:**

1. **WebCrawler ALREADY has internal parallelism**:
   ```python
   # web_crawler.py line 145
   self._semaphore = asyncio.Semaphore(max_concurrent)  # Default: 5

   # line 396 - crawl() method uses gather internally
   async def crawl(self, urls: list[str]) -> CrawlOutput:
       tasks = [self._fetch_url(url) for url in urls]
       results = await asyncio.gather(*tasks)  # Already parallel!
   ```
   **Implication**: Calling `web_crawl` for 3 URLs sequentially vs parallel makes NO difference - the crawler already parallelizes internally.

2. **Rate limiting serializes requests**:
   ```python
   # brave.py line 82, 92
   self._lock = asyncio.Lock()
   async def _rate_limit(self):
       async with self._lock:  # Only ONE request proceeds at a time
           ...

   # web_crawler.py line 153, 165
   self._rate_lock = asyncio.Lock()
   async def _rate_limit(self):
       async with self._rate_lock:  # Serializes crawl requests
           ...
   ```
   **Implication**: Even with `asyncio.gather()`, requests queue up on rate limit locks. Real-world speedup is LIMITED.

3. **LLM already returns multiple tool calls**:
   ```python
   # client.py line 1451-1481 - already accumulates multiple tool calls
   completed_tool_calls = []
   for tcc in tool_call_chunks.values():
       completed_tool_calls.append(ToolCall(...))
   ```
   **Implication**: The `parallel_tool_calls` API parameter just hints to the model. Our code already handles multiple tool calls.

4. **UrlRegistry is already thread-safe**:
   ```python
   # url_registry.py line 60
   self._lock = Lock()  # threading.Lock, already safe
   ```
   **Implication**: No changes needed for UrlRegistry.

#### 0.4.2 Realistic Performance Expectations

**CORRECTED Performance Analysis (accounting for rate limiting):**

| Scenario | Sequential | Parallel | Real Improvement | Notes |
|----------|------------|----------|------------------|-------|
| 3 web_search (1.5s API + 1s rate limit each) | 7.5s | ~4.5s | 40% | Rate limit dominates |
| 5 web_crawl (2s each, 0.5s rate limit) | 12.5s | ~4s | 68% | Semaphore helps |
| Mixed (2 search + 3 crawl) | 10s | ~6s | 40% | Dependencies + rate limits |

**Revised Expected Impact**: 20-40% latency reduction (not 30-50%), primarily from:
- Overlapping network I/O wait times
- WebCrawler's internal semaphore (already parallel)
- Batching same-type tools reduces LLM round-trips

**Where parallel execution DOES help:**
- Enterprise sources (Vector Search, Genie) have DIFFERENT rate limiters
- Cross-source parallelism (web search + VS search simultaneously)
- CPU-bound operations (JSON parsing, content processing)

#### 0.4.3 Lock Type Selection (CORRECTED)

**CRITICAL CORRECTION: Use asyncio.Lock, NOT threading.RLock**

The original plan suggested `threading.RLock`. This is WRONG because:

1. Tool execution is async (`await _execute_tool(...)`)
2. `threading.RLock` in async code causes issues when tasks yield control
3. `asyncio.Lock` is designed for async/await patterns

**Correct approach:**

```python
# ResearchState mutations happen from async context
# Use asyncio.Lock wrapped in sync helper

import asyncio
from contextlib import contextmanager

class ResearchState:
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    # For sync callers (existing code compatibility)
    def add_source(self, source: SourceInfo) -> None:
        """Sync-compatible source addition."""
        # Get or create event loop
        try:
            loop = asyncio.get_running_loop()
            # If we're in async context, schedule coroutine
            if loop.is_running():
                asyncio.create_task(self._add_source_async(source))
                return
        except RuntimeError:
            pass
        # Sync fallback (no event loop)
        self._add_source_sync(source)

    async def _add_source_async(self, source: SourceInfo) -> None:
        async with self._lock:
            if not any(s.url == source.url for s in self.sources):
                self.sources.append(source)

    def _add_source_sync(self, source: SourceInfo) -> None:
        # Simple sync version for non-async contexts
        if not any(s.url == source.url for s in self.sources):
            self.sources.append(source)
```

**BETTER ALTERNATIVE: Per-collection locks (granular locking)**

```python
class ResearchState:
    _sources_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _claims_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _evidence_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # No contention between add_source() and add_claim()
```

#### 0.4.4 Missing Critical Features

**Features NOT in original plan that are REQUIRED:**

1. **Timeout handling per tool**:
   ```python
   async def execute_with_timeout(coro, timeout_seconds: float):
       try:
           return await asyncio.wait_for(coro, timeout=timeout_seconds)
       except asyncio.TimeoutError:
           return f"Tool timed out after {timeout_seconds}s"
   ```

2. **Budget atomicity** (reserve before parallel execution):
   ```python
   async def reserve_budget(react_state, count: int) -> bool:
       async with react_state._budget_lock:
           if react_state.tool_call_count + count > max_budget:
               return False
           react_state.tool_call_count += count  # Reserve atomically
           return True
   ```

3. **Streaming events during parallel execution**:
   ```python
   # Events should be yielded as tools complete, not batched
   async def execute_tools_parallel_with_events(...):
       for coro in asyncio.as_completed(tasks):
           result = await coro
           yield ReactResearchEvent(event_type="tool_result", ...)
   ```

4. **MLflow tracing for parallel spans**:
   ```python
   # Each tool gets its own span, all under a parent batch span
   async with safe_tool_span("parallel_batch") as batch_span:
       for tc in tool_calls:
           tasks.append(_execute_with_span(tc, parent=batch_span))
   ```

5. **Per-source rate limiters for enterprise sources**:
   ```python
   class SourceRateLimiter:
       _limiters: dict[str, asyncio.Lock] = {}

       def get_limiter(self, source_name: str) -> asyncio.Lock:
           if source_name not in self._limiters:
               self._limiters[source_name] = asyncio.Lock()
           return self._limiters[source_name]
   ```

6. **Context window budget tracking**:
   ```python
   # Parallel results expand context - track and limit
   total_result_tokens = sum(len(r) // 4 for r in results)
   if total_result_tokens > context_budget:
       # Truncate or drop lowest-relevance results
   ```

#### 0.4.5 Tool Dependency Graph

```
web_search ──────────────────────────────────────┐
     │                                           │
     │ Registers URLs in UrlRegistry             │
     ▼                                           │
web_crawl (index) ←── Requires URL registry ─────┘
     │
     │ CANNOT run before web_search completes
     ▼

vector_search ──┬── Independent (different rate limiter)
                │
genie_query ────┼── Independent (different rate limiter)
                │
assistant_query ┴── Independent (different rate limiter)
```

**Key Insight**: Enterprise sources (VS, Genie, Assistants) are independent of web sources and each other. TRUE parallelism benefit comes from cross-source execution.

#### 0.4.6 Model-Specific Considerations

| Model Tier | `parallel_tool_calls` | Rationale |
|------------|----------------------|-----------|
| simple (Haiku) | `true` | Fast models benefit from parallelism |
| analytical (Haiku/GPT-5-mini) | `true` | Standard research tier |
| complex (Claude Opus) | `true` | High-quality synthesis |
| o-series (reasoning) | `false` | May not support parallel calls |

**Note**: The `parallel_tool_calls` parameter is a HINT to the model, not a guarantee. Models may return multiple tool calls regardless of this setting.

### 0.5 Architecture Decisions for Multi-Source Discovery & Parallel Execution

Based on the research above, we make the following architecture decisions:

**Decision A1: Discovery Before Planning (Inspired by Instructed Retriever)**
- **Rationale**: Planner needs to know what data exists to make intelligent routing decisions
- **Trade-off**: Adds ~2-5 seconds to research start, but dramatically improves routing quality
- **Implementation**: `run_background_discovery()` before `run_planner()`

**Decision A2: Source Hints, Not Hard Requirements**
- **Rationale**: Steps specify `source_hints` with priority, not hard requirements
- **Why**: Allows graceful degradation if a source is unavailable or returns no results
- **Alternative Rejected**: Hard source requirements would cause step failures

**Decision A3: Parallel Discovery Across Sources**
- **Rationale**: Query all sources simultaneously during discovery to minimize latency
- **Why**: RRF fusion handles combining results; parallel execution critical for UX
- **Trade-off**: Higher initial resource usage, but dramatically faster than sequential

**Decision A4: Source Budgets Per Research Type**
- **Rationale**: Different depths should have different source query limits
- **Why**: Extended research can query more; light research should be selective
- **Implementation**: Configure in `research_types.{type}.source_budgets`

**Decision A5: Per-Step Tool Filtering**
- **Rationale**: Not all tools should be available for all steps
- **Why**: Reduces LLM confusion, prevents wasteful queries to irrelevant sources
- **Implementation**: `filter_tools_for_step()` based on `step.source_hints`

**Decision A6: Cross-Source Parallel Execution (PRIMARY BENEFIT)**
- **Rationale**: Enterprise sources (VS, Genie) have independent rate limiters from web search
- **Why**: True parallelism benefit (40%+ improvement) comes from querying different sources simultaneously
- **Implementation**:
  1. Group by SOURCE TYPE (not tool type): web_sources, enterprise_sources
  2. Execute source groups in parallel with `asyncio.gather()`
  3. Within web_sources: rate limiter serializes anyway (minimal benefit)
  4. Cross-source: VS + Genie + Web can run truly parallel
- **Key Insight**: Don't over-engineer same-source parallelism; focus on cross-source

**Decision A7: Async-Safe State with Granular Locks**
- **Rationale**: Race conditions in ResearchState must be fixed BEFORE enabling parallelism
- **Why**: `add_source()` has check-then-act race; concurrent calls can create duplicates
- **CORRECTED Implementation**:
  1. Use `asyncio.Lock` (NOT threading.RLock - we're in async context!)
  2. Use GRANULAR locks per collection (sources, claims, evidence) to reduce contention
  3. Sync callers use event loop detection for compatibility
- **Why NOT RLock**: Tool execution is async (`await _execute_tool`); RLock can deadlock

**Decision A8: Configurable Parallel Tool Calls per Model Tier**
- **Rationale**: Not all models support or benefit equally from parallel tool calls
- **Why**: o-series reasoning models may not support `parallel_tool_calls` parameter
- **Implementation**:
  ```yaml
  models:
    simple:
      parallel_tool_calls: true
    analytical:
      parallel_tool_calls: true
    complex:
      parallel_tool_calls: true  # Claude Opus supports it
    # o-series would be: parallel_tool_calls: false
  ```

**Decision A9: Timeout and Budget Management for Parallel Execution**
- **Rationale**: Parallel execution can hang on slow tools or exceed budgets
- **Why**: Without timeouts, one slow tool blocks entire batch; without budget checks, overruns
- **Implementation**:
  1. Per-tool timeout with `asyncio.wait_for()`
  2. Atomic budget reservation BEFORE starting parallel batch
  3. Context window tracking (parallel results expand context)
- **Trade-off**: Complexity vs reliability; worth it for production robustness

**Decision A10: Event Streaming During Parallel Execution**
- **Rationale**: Users expect real-time feedback on tool progress
- **Why**: Batching events until all tools complete degrades UX
- **Implementation**:
  1. Use `asyncio.as_completed()` to yield events as tools finish
  2. Each tool completion yields its own `ReactResearchEvent`
  3. Preserve logical ordering in final message history (not event order)
- **Trade-off**: More complex event handling vs better UX

---

## 1. RAG Retrieval Best Practices (2025-2026)

### Hybrid Search Architecture
- **Production standard**: BM25 + dense embeddings working together
- **Pure vector search fails on**: SKUs, product numbers, codenames, exact phrases
- **Databricks**: Use `query_type="hybrid"` parameter (built-in, not separate)

### Score Fusion
- **Databricks handles fusion internally** with `query_type="hybrid"`
- **For cross-source fusion** (VS + Genie + files): Use RRF (Reciprocal Rank Fusion)
- **RRF formula**: `score = 1/(k+r)`, k≈60

### Reranking
- **Databricks has built-in reranker** - NOT a separate service/endpoint
- **API**: `reranker=DatabricksReranker(columns_to_rerank=[...])`
- **Performance**: ~1.5s for 50 docs
- **Improvement**: +15 percentage points on enterprise benchmarks

---

## 2. Databricks Vector Search Capabilities

### Hybrid Search
```python
results = index.similarity_search(
    query_text="How to create a Vector Search index",
    columns=["id", "text", "parent_doc_summary", "date"],
    num_results=10,
    query_type="hybrid",  # Enables BM25 + vector fusion (max 200 results)
)
```

- **Parameter**: `query_type="hybrid"`
- **Max results**: 200 (when hybrid enabled)
- **Combines**: keyword (BM25) + semantic similarity

### Built-in Reranking (NOT separate endpoint)

```python
from databricks.vector_search.reranker import DatabricksReranker

results = index.similarity_search(
    query_text="How to create a Vector Search index",
    columns=["id", "text", "parent_doc_summary", "date"],
    num_results=10,
    query_type="hybrid",
    reranker=DatabricksReranker(columns_to_rerank=["text", "parent_doc_summary"])
)
```

- **Class**: `from databricks.vector_search.reranker import DatabricksReranker`
- **Pass to**: `similarity_search(..., reranker=DatabricksReranker(...))`
- **SDK requirement**: `databricks-vectorsearch >= 0.57`
- **Column order matters**: First 2000 chars evaluated
- **Debug output**: `reranker_time` in response shows timing

### Schema Auto-Detection

```python
# Get index metadata to detect available columns
vs_client = obo_client.vector_search_indexes
index_info = vs_client.get_index(index_name)

# Columns available in delta_sync_index_spec or direct_access_index_spec
schema = index_info.delta_sync_index_spec or index_info.direct_access_index_spec
columns = [col.name for col in schema.columns_to_sync or []]
```

- Use `vs_client.get_index(index_name)` to get index metadata
- Columns available in `delta_sync_index_spec.columns_to_sync` or `direct_access_index_spec`
- Text columns auto-detected for reranking (string/text types)

### Limits
- **Filter by ID**: max 1,024 IDs per query
- **Hybrid search**: max 200 results

---

## 3. Existing Codebase Infrastructure (MUST REUSE)

| Component | Location | Capability |
|-----------|----------|------------|
| HybridSearchIndex | evidence_registry.py | BM25 + GTE fusion (α=0.6) - for local/file search |
| ContentEvaluator | content_evaluator.py | Quality scoring, paywall detection |
| chunk_content() | evidence_selector.py | 8000 char chunks, 1000 overlap |
| EvidenceRegistry | evidence_registry.py | Index-based evidence tracking |
| ChatSourcePoolService | chat_source_pool_service.py | Source accumulation + search |
| UrlRegistry | url_registry.py | URL deduplication |

---

## 4. Architecture Decisions

### Decision 1: Use Databricks Native Hybrid Search + Reranking
**Rationale**: Built-in, no extra infrastructure, +15% quality improvement

**Implementation**:
```python
# CORRECT - use built-in parameters
results = vs_client.query_index(
    index_name=self._index_name,
    query_text=arguments["query"],
    columns=self._columns,
    num_results=10,
    query_type="hybrid",
    reranker=DatabricksReranker(columns_to_rerank=self._columns_to_rerank)
)
```

### Decision 2: Auto-Detect Schema from Index
**Rationale**: Users add own indexes (FR-016b), can't hardcode columns in app.yaml

**Implementation**:
```python
# When user adds a Vector Search source:
# 1. Get OBO client to query index metadata
# 2. Auto-detect columns from index schema
# 3. Identify text columns for reranking
# 4. Store detected config in UserDataSource.config JSONB
```

### Decision 3: HybridSearchIndex for Local Files
**Rationale**: Uploaded files aren't in Vector Search, need local hybrid. Reuse existing code.

**Implementation**:
```python
from deep_research.agent.tools.evidence_registry import HybridSearchIndex

# Reuse existing BM25 + GTE hybrid for file chunks
index = HybridSearchIndex(embedder=self._embedder)
for chunk in chunks:
    index.add(content=chunk.content, metadata={...})
results = index.search(query=query, alpha=0.6, top_k=10)
```

### Decision 4: RRF for Cross-Source Fusion
**Rationale**: When combining VS results, Genie results, and file results, use RRF.

**Implementation**:
```python
def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    k: int = 60,
) -> list[SearchResult]:
    """RRF fusion for combining results from multiple sources."""
    scores: dict[str, float] = {}
    for results in result_lists:
        for rank, result in enumerate(results):
            key = _get_result_key(result)
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
    # Sort by fused score, deduplicate
    ...
```

### Decision 5: Configurable Document Parsing (Databricks Default + Docling Fallback)

**CORRECTION**: pyxtxt is NOT recommended. Use configurable approach with Databricks native as default.

| Library | Best For | Speed | Quality | Role |
|---------|----------|-------|---------|------|
| **ai_parse_document** | Databricks-native | ~5s | Excellent | **DEFAULT** |
| **Docling** (IBM) | Local/air-gapped | 38s | Best | **FALLBACK** |
| **PyMuPDF4LLM** | Speed-critical | 5s | Poor tables | Not used |
| **MarkItDown** (MS) | Non-PDF | Fast | Weak PDF | Not used |

**Architecture: Configurable with Automatic Fallback**

```yaml
# config/app.yaml
file_upload:
  parsing:
    parser: databricks  # Default: "databricks", Alternative: "docling"
```

**DEFAULT: Databricks `ai_parse_document`**

```python
# Automatically used when parser: databricks (default)
result = spark.sql("""
    SELECT ai_parse_document(
        content,
        format => 'PDF',
        output_format => 'MARKDOWN'
    ) as parsed
    FROM read_files(?)
""")
```

**Why ai_parse_document as default:**
- Native Unity Catalog integration
- Captures tables, figures, diagrams with AI descriptions
- Can chain with `ai_extract`, `ai_classify`, `ai_summarize`
- 3-5x lower cost than competitors (per Databricks)
- Supports PDF, JPG/JPEG, PNG, DOC/DOCX, PPT/PPTX
- Requires DBR 17.1+ or serverless environment v3+

**FALLBACK: Docling** (automatic when Databricks unavailable)

```python
# Automatically used when:
# 1. parser: docling in config
# 2. Databricks parser fails
# 3. DBR < 17.1
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert(file_path)
markdown = result.document.export_to_markdown()
```

**Why Docling as fallback:**
- Best quality for tables and document structure (uses DocLayNet + TableFormer)
- Open source (MIT license)
- Supports PDF, DOCX, PPTX, XLSX, HTML, images, audio
- Integrates with LangChain, LlamaIndex
- Good for local development and air-gapped deployments

**Why NOT pyxtxt:**
- Obscure library, not in any RAG benchmarks
- Beta status (v0.3.5)
- No quality comparisons available
- Databricks native solution is superior

### Decision 6: Row-Based Chunking for Tabular Data
**Rationale**: Tables are atomic units - never split without headers (per RAG best practices)

**Implementation**:
```python
# Each row becomes a chunk WITH headers prepended for context
content = f"Headers: {header_line}\nRow {i+1}: {row_line}"
```

---

## 5. OBO Token Handling (Fix Required)

**Current Issue**: OBO token extracted in middleware but not preserved for later use

**Fix Required**:
```python
# In middleware/auth.py
obo_token = extract_obo_token(dict(request.headers))
if obo_token:
    request.state.obo_token = obo_token  # PRESERVE for later use
    user_client = get_user_workspace_client(obo_token)
    request.state.user_workspace_client = user_client

# In ResearchContext (state.py)
@dataclass
class ResearchContext:
    user_token: str | None = None  # Add this field

# In OBODatabricksClient
class OBODatabricksClient:
    async def get_client(self, user_token: str | None) -> WorkspaceClient:
        if user_token:
            return get_user_workspace_client(user_token)
        return get_workspace_client()
```

---

## 6. UserDataSource Config Schema

For Vector Search sources, the `config` JSONB column stores:

```python
{
    "endpoint_name": "vs-endpoint-prod",
    "index_name": "catalog.schema.product_docs_index",
    "columns": ["id", "title", "content", "url"],  # AUTO-DETECTED from index
    "columns_to_rerank": ["content", "title"],     # AUTO-DETECTED text columns
    "enable_hybrid": True,                          # User can toggle
    "enable_reranking": True,                       # User can toggle
    "num_results": 10,                              # User can configure
}
```

This allows:
1. Users to add any Vector Search index they have access to
2. Schema auto-detection on source creation
3. User customization of search parameters
4. No hardcoded columns in app.yaml

---

## 7. Cross-Source Deduplication Strategy

When combining results from multiple sources (VS + Genie + Files), deduplication key priority:

1. **Explicit ID** (most reliable)
2. **URL + optional chunk info** (for web/VS sources)
3. **Content fingerprint** (fallback for Genie/generated content)

```python
def _get_result_key(result: SearchResult) -> str:
    if result.id:
        return f"id:{result.id}"
    if result.url:
        chunk_suffix = f"#{result.chunk_index}" if result.chunk_index else ""
        return f"url:{result.url}{chunk_suffix}"
    # Fallback: first 2000 chars (matches Databricks reranker limit)
    return f"content:{_compute_content_fingerprint(result.content)}"
```

---

## Sources

### Parallel Tool Execution & LLM Tool Calling
- [OpenAI Parallel Tool Calls Documentation](https://platform.openai.com/docs/guides/function-calling)
- [Databricks Instructed Retriever - Parallel Query Execution](https://www.databricks.com/blog/instructed-retriever-unlocking-system-level-reasoning-search-agents)
- [Python asyncio.gather() Best Practices](https://docs.python.org/3/library/asyncio-task.html#asyncio.gather)
- [Thread Safety with RLock](https://docs.python.org/3/library/threading.html#rlock-objects)

### Multi-Source Discovery & Agentic RAG
- [Databricks Instructed Retriever Blog](https://www.databricks.com/blog/instructed-retriever-unlocking-system-level-reasoning-search-agents)
- [VentureBeat: Instructed Retriever 70% Improvement](https://venturebeat.com/data/databricks-instructed-retriever-beats-traditional-rag-data-retrieval-by-70)
- [Agentic RAG Survey (arXiv 2501.09136)](https://arxiv.org/abs/2501.09136)
- [DeepPlanner: Scaling Planning (arXiv 2510.12979)](https://arxiv.org/html/2510.12979)
- [Step-DeepResearch Technical Report](https://arxiv.org/html/2512.20491v1)
- [What Is Agentic RAG? (Aisera 2025)](https://aisera.com/blog/agentic-rag/)
- [Azure AI Search Agentic Retrieval](https://learn.microsoft.com/en-us/azure/search/retrieval-augmented-generation-overview)

### Databricks Vector Search & Reranking
- [Databricks Reranking Blog](https://www.databricks.com/blog/reranking-mosaic-ai-vector-search-faster-smarter-retrieval-rag-agents)
- [Query Vector Search Docs](https://docs.databricks.com/aws/en/vector-search/query-vector-search)
- [Vector Search Retrieval Quality Guide](https://docs.databricks.com/aws/en/vector-search/vector-search-retrieval-quality)

### RAG Best Practices
- [Hybrid RAG Best Practices](https://superlinked.com/vectorhub/articles/optimizing-rag-with-hybrid-search-reranking)
- [RAG Chunking for Excel/CSV](https://ragaboutit.com/mastering-document-chunking-for-non-standard-excel-files-a-software-engineers-guide/)
- [Best Chunking Strategies for RAG 2025](https://www.firecrawl.dev/blog/best-chunking-strategies-rag-2025)

### Document Parsing Libraries
- [Databricks ai_parse_document Docs](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_parse_document)
- [Databricks Document Intelligence Blog](https://www.databricks.com/blog/pdfs-production-announcing-state-art-document-intelligence-databricks)
- [PDF Data Extraction Benchmark 2025](https://procycons.com/en/blogs/pdf-data-extraction-benchmark/)
- [Docling vs LlamaParse vs Unstructured Comparison](https://llms.reducto.ai/document-parser-comparison)
- [Python PDF Libraries 2026 Evaluation](https://unstract.com/blog/evaluating-python-pdf-to-text-libraries/)
- [7 Python PDF Extractors Tested (2025)](https://dev.to/onlyoneaman/i-tested-7-python-pdf-extractors-so-you-dont-have-to-2025-edition-akm)
- [Docling GitHub](https://github.com/docling-project/docling)
- [Microsoft MarkItDown GitHub](https://github.com/microsoft/markitdown)
- [PyMuPDF4LLM Documentation](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/)

---

## 8. Databricks API Discovery Methods (US9a/US9b)

### 8.1 Vector Search Discovery API

| Method | Parameters | Returns | Purpose |
|--------|------------|---------|---------|
| `w.vector_search_endpoints.list_endpoints()` | `page_token` (optional) | `Iterator[EndpointInfo]` | List all VS endpoints |
| `w.vector_search_indexes.list_indexes(endpoint_name)` | `endpoint_name`, `page_token` (optional) | `Iterator[MiniVectorIndex]` | List indexes per endpoint |
| `w.vector_search_indexes.get_index(index_name)` | `index_name`, `ensure_reranker_compatible` (optional) | `VectorIndex` | Get full index metadata |

**VectorIndex Data Structure:**
```python
@dataclass
class VectorIndex:
    name: str | None                    # Index identifier
    creator: str | None                 # User who created
    endpoint_name: str | None           # Associated endpoint
    primary_key: str | None             # Primary key field
    index_type: VectorIndexType | None  # DELTA_SYNC or DIRECT_ACCESS
    status: VectorIndexStatus | None    # Operational state
    delta_sync_index_spec: DeltaSyncVectorIndexSpecResponse | None
    direct_access_index_spec: DirectAccessVectorIndexSpec | None
```

### 8.2 Genie Discovery API

| Method | Parameters | Returns | Purpose |
|--------|------------|---------|---------|
| `w.genie.list_spaces()` | `page_size`, `page_token` (optional) | `GenieListSpacesResponse` | List all Genie spaces |
| `w.genie.get_space(space_id)` | `space_id`, `include_serialized_space` | `GenieSpace` | Get space details |

**GenieSpace Data Structure:**
```python
@dataclass
class GenieSpace:
    id: str
    title: str | None
    description: str | None
    warehouse_id: str | None
    creator: str | None
```

### 8.3 Serving Endpoints Discovery API

| Method | Parameters | Returns | Purpose |
|--------|------------|---------|---------|
| `w.serving_endpoints.list()` | None | `Iterator[ServingEndpoint]` | List all serving endpoints |

**Knowledge Assistant Identification Heuristics:**
- Tags contain: `assistant`, `knowledge`, `expert`
- Name patterns: Contains `assistant`, `expert`, `helper`
- Endpoint type: `CUSTOM` or `EXTERNAL_MODEL`
- Allow manual classification as fallback

### 8.4 OBO Authentication Pattern

```python
from databricks.sdk import WorkspaceClient
from databricks_ai_bridge import ModelServingUserCredentials

# Initialize inside predict() - user identity only known at runtime
user_client = WorkspaceClient(
    credentials_strategy=ModelServingUserCredentials()
)
```

**Required API Scopes:**
| Resource | Scope |
|----------|-------|
| Vector Search | `vectorsearch.vector-search-indexes` |
| Serving Endpoints | `serving.serving-endpoints` |
| Genie | `dashboards.genie` |

### 8.5 Discovery Implementation Pattern

```python
async def discover_all_sources(user_token: str | None) -> list[DiscoveredSource]:
    """Parallel discovery of all Databricks data sources."""
    w = WorkspaceClient(credentials_strategy=ModelServingUserCredentials())

    # Execute discovery in parallel
    vs_task = asyncio.to_thread(_discover_vector_search_sync, w)
    genie_task = asyncio.to_thread(_discover_genie_sync, w)
    serving_task = asyncio.to_thread(_discover_serving_sync, w)

    results = await asyncio.gather(vs_task, genie_task, serving_task, return_exceptions=True)

    # Flatten and filter errors
    sources = []
    for result in results:
        if isinstance(result, Exception):
            logger.warning(f"Discovery failed: {result}")
            continue
        sources.extend(result)

    return sources
```

### 8.6 Sources
- [Vector Search SDK](https://databricks-sdk-py.readthedocs.io/en/latest/workspace/vectorsearch/vector_search_indexes.html)
- [Genie SDK](https://databricks-sdk-py.readthedocs.io/en/stable/workspace/dashboards/genie.html)
- [Serving Endpoints SDK](https://databricks-sdk-py.readthedocs.io/en/latest/workspace/serving/serving_endpoints.html)
- [Agent Authentication](https://docs.databricks.com/aws/en/generative-ai/agent-framework/agent-authentication)
