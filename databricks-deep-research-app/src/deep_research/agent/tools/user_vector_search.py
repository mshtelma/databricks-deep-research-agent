"""User-configured Vector Search Tool with OBO authentication.

Implements VectorSearchTool for user-added indexes with:
- OBO authentication (uses user's permissions via WorkspaceClient)
- Hybrid search (BM25 + vectors)
- Built-in reranking (via SDK-native columns_to_rerank)
- Metadata filtering
- Dynamic tool definition from config
- MLflow tracing for observability
- Multi-query execution with Reciprocal Rank Fusion (RRF)

Part of 007-enterprise-data-sources feature (T012, T013).
"""

import asyncio
import time
from typing import Any

from deep_research.agent.tools.base import (
    ResearchContext,
    ToolDefinition,
    ToolResult,
)
from deep_research.core.logging_utils import get_logger
from deep_research.models.data_source import UserDataSource
from deep_research.schemas.query_config import (
    FilterExpression,
    FilterSyntax,
    QueryType,
    VectorSearchQueryConfig,
)
from deep_research.services.metrics import record_source_query
from deep_research.services.obo_client import OBODatabricksClient
from deep_research.services.vector_search_query import (
    ColumnRoles,
    VectorSearchQueryService,
    VectorSearchResult,
    extract_queryable_columns,
)

logger = get_logger(__name__)

# Re-export for backward compatibility
VectorSearchResultItem = VectorSearchResult


def reciprocal_rank_fusion(
    result_sets: list[list[VectorSearchResult]],
    k: int = 60,
) -> list[VectorSearchResult]:
    """Merge multiple ranked result lists using Reciprocal Rank Fusion.

    score(d) = sum(1 / (k + rank_i)) for each result set where d appears.
    k=60 matches Databricks HYBRID internal RRF parameter.

    Args:
        result_sets: List of ranked result lists to merge.
        k: RRF constant (higher = less weight to top ranks).

    Returns:
        Merged and re-ranked list of results.
    """
    import hashlib

    scores: dict[str, float] = {}
    docs: dict[str, VectorSearchResult] = {}

    for results in result_sets:
        for rank, result in enumerate(results):
            # Dedup key: prefer URL, fall back to content hash
            if result.url:
                doc_key = result.url
            else:
                content_preview = result.content[:200] if result.content else result.id
                doc_key = hashlib.md5(content_preview.encode()).hexdigest()[:16]

            scores[doc_key] = scores.get(doc_key, 0.0) + 1.0 / (k + rank + 1)
            if doc_key not in docs or result.score > docs[doc_key].score:
                docs[doc_key] = result  # Keep highest-scoring version

    # Sort by fused score descending
    ranked_keys = sorted(docs.keys(), key=lambda key: scores[key], reverse=True)
    return [docs[key] for key in ranked_keys]


class UserVectorSearchTool:
    """Vector Search tool for user-configured indexes with OBO authentication.

    Key features:
    - Uses user's OBO token for queries (respects their permissions)
    - Supports hybrid search (BM25 + vector fusion)
    - Supports built-in reranking for improved relevance
    - Supports metadata filtering from arguments
    - Dynamic tool definition based on index schema

    Configuration is loaded from UserDataSource.config JSONB, including
    auto-detected columns from index metadata.
    """

    def __init__(
        self,
        obo_client: OBODatabricksClient,
        data_source: UserDataSource | None = None,
        endpoint_name: str = "",
        index_name: str = "",
        columns: list[str] | None = None,
        columns_to_rerank: list[str] | None = None,
        enable_hybrid: bool = True,
        enable_reranking: bool = True,
        num_results: int = 10,
        description: str | None = None,
        query_config: VectorSearchQueryConfig | None = None,
        source_name: str | None = None,
        column_roles: ColumnRoles | None = None,
    ) -> None:
        """Initialize the Vector Search tool.

        Args:
            obo_client: OBO client for user authentication.
            data_source: UserDataSource configuration from database (optional if source_name given).
            endpoint_name: Vector Search endpoint name.
            index_name: Fully qualified index name (catalog.schema.index).
            columns: Columns to return (auto-detected if not provided).
            columns_to_rerank: Text columns for reranking.
            enable_hybrid: Enable hybrid search (BM25 + vectors).
            enable_reranking: Enable reranking (requires databricks-vectorsearch >= 0.57).
            num_results: Default number of results.
            description: Tool description for LLM.
            query_config: Query configuration from US9b (overrides other params if provided).
            source_name: Display name for the source (alternative to data_source).
        """
        self._obo_client = obo_client
        self._source_name = source_name or (data_source.name if data_source else index_name)
        self._endpoint_name = endpoint_name
        self._index_name = index_name

        # Apply query config if provided (US9b - T010u)
        self._query_config = query_config
        if query_config:
            # Query config takes precedence over individual params
            self._columns = query_config.columns or columns or []
            self._columns_to_rerank = query_config.columns_to_rerank or columns_to_rerank or []
            self._enable_hybrid = query_config.query_type in (QueryType.HYBRID, QueryType.FULL_TEXT)
            self._enable_reranking = query_config.enable_reranking
            self._num_results = query_config.num_results
            self._score_threshold = query_config.score_threshold
            self._query_type = query_config.query_type
            self._filter_syntax = query_config.filter_syntax
            self._default_filters = query_config.filters
        else:
            self._columns = columns or []
            self._columns_to_rerank = columns_to_rerank or []
            self._enable_hybrid = enable_hybrid
            self._enable_reranking = enable_reranking
            self._num_results = num_results
            self._score_threshold = None
            self._query_type = QueryType.HYBRID if enable_hybrid else QueryType.ANN
            self._filter_syntax = FilterSyntax.SQL
            self._default_filters = []

        # Generate tool name from index (unique per source)
        safe_name = index_name.replace(".", "_").replace("-", "_")
        self._tool_name = f"search_{safe_name}"

        # Description
        self._description = description or (
            f"Search '{self._source_name}' for semantically similar documents. "
            f"Returns relevant passages from the index."
        )

        # Shared query service
        self._query_service = VectorSearchQueryService()

        # Column role mapping — from enrichment or populated by _discover_columns()
        self._column_roles: ColumnRoles | None = column_roles

    @property
    def definition(self) -> ToolDefinition:
        """Return tool definition for LLM function calling.

        Includes dynamic filter schema based on available columns.
        """
        params: dict[str, Any] = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Semantic search query in natural language.",
                },
                "num_results": {
                    "type": "integer",
                    "description": f"Number of results to return (default: {self._num_results}).",
                    "default": self._num_results,
                },
            },
            "required": ["query"],
        }

        # Add filters parameter if we have filterable columns
        filterable_columns = [c for c in self._columns if c not in ("id", "content", "score")]
        if filterable_columns:
            params["properties"]["filters"] = {
                "type": "object",
                "description": (
                    f"Optional metadata filters. Available columns: {', '.join(filterable_columns)}. "
                    "Example: {\"department\": \"engineering\"}"
                ),
            }

        return ToolDefinition(
            name=self._tool_name,
            description=self._description,
            parameters=params,
            source_type="vector_search",
        )

    async def _execute_single_query(
        self,
        client: Any,
        query: str,
        num_results: int,
        filters: dict[str, Any] | None,
    ) -> list[VectorSearchResult]:
        """Execute a single vector search query.

        Args:
            client: OBO-authenticated WorkspaceClient.
            query: Search query text.
            num_results: Number of results to return.
            filters: Optional runtime filters.

        Returns:
            List of VectorSearchResult from the query service.
        """
        query_type_str = self._resolve_query_type()
        filters_json = self._build_filters_json(filters)
        rerank_cols = (
            self._columns_to_rerank
            if self._enable_reranking and self._columns_to_rerank
            else None
        )

        return await self._query_service.query(
            client=client,
            index_name=self._index_name,
            query_text=query,
            columns=self._columns,
            num_results=num_results,
            query_type=query_type_str,
            filters_json=filters_json,
            score_threshold=self._score_threshold,
            columns_to_rerank=rerank_cols,
            column_roles=self._column_roles,
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ResearchContext,
    ) -> ToolResult:
        """Execute vector search with OBO authentication.

        Supports multi-query execution with Reciprocal Rank Fusion (RRF).
        When `_alternate_queries` is present in arguments (injected by the
        query rewriter), executes each alternate query sequentially and
        merges results using RRF for improved recall and relevance.

        Args:
            arguments: Tool arguments containing 'query', optional 'num_results', 'filters'.
                May also contain '_alternate_queries' from the query rewriter.
            context: Research context with user_token for OBO.

        Returns:
            ToolResult with formatted search results and source tracking.
        """
        query = arguments.get("query", "")
        num_results = arguments.get("num_results", self._num_results)
        filters = arguments.get("filters")
        alternate_queries: list[str] = arguments.pop("_alternate_queries", [])

        logger.info(
            "VECTOR_SEARCH_TOOL_EXECUTE",
            tool_name=self._tool_name,
            index_name=self._index_name,
            endpoint_name=self._endpoint_name,
            query=query[:100],
            num_results=num_results,
            has_filters=filters is not None,
            alternate_query_count=len(alternate_queries),
            obo_authenticated=context.user_token is not None,
        )

        if not query:
            return ToolResult(
                content="Error: 'query' is required.",
                success=False,
                error="Missing required argument: query",
            )

        start_time = time.perf_counter()
        result_count = 0

        try:
            # Get OBO-authenticated WorkspaceClient
            client = await self._obo_client.get_client(context.user_token)

            # Discover columns on first execute if not already known
            if not self._columns:
                await self._discover_columns(client)
                if not self._columns:
                    return ToolResult(
                        content=(
                            f"Cannot query index '{self._index_name}': unable to determine "
                            f"available columns. The index may not be accessible or may not "
                            f"have an embedding configuration."
                        ),
                        success=False,
                        error="Column discovery failed",
                    )
            elif not self._column_roles:
                # Columns known (from enrichment/config) but roles missing.
                # Discover roles so the parser maps content correctly.
                await self._discover_columns(client)

            logger.info(
                "VECTOR_SEARCH_USING_WORKSPACE_CLIENT",
                index=self._index_name,
                query=query[:80],
                query_type=self._resolve_query_type(),
                columns=self._columns[:5] if self._columns else [],
                content_column=self._column_roles.content_column if self._column_roles else None,
                has_column_roles=self._column_roles is not None,
                obo_authenticated=context.user_token is not None,
            )

            # Execute primary query
            primary_results = await self._execute_single_query(
                client, query, num_results, filters,
            )

            # Execute alternate queries sequentially and merge with RRF
            if alternate_queries and primary_results:
                all_result_sets: list[list[VectorSearchResult]] = [primary_results]
                for alt_query in alternate_queries[:5]:  # Cap at 5 alternates
                    try:
                        alt_results = await self._execute_single_query(
                            client, alt_query, num_results, filters,
                        )
                        if alt_results:
                            all_result_sets.append(alt_results)
                    except Exception as e:
                        logger.warning(
                            "VS_ALTERNATE_QUERY_FAILED",
                            query=alt_query[:80],
                            error=str(e)[:100],
                        )
                        continue  # Skip failed alternates

                if len(all_result_sets) > 1:
                    results = reciprocal_rank_fusion(all_result_sets)[:num_results]
                    logger.info(
                        "VS_MULTI_QUERY_RRF_COMPLETE",
                        query_count=len(all_result_sets),
                        fused_result_count=len(results),
                    )
                else:
                    results = primary_results
            else:
                results = primary_results

            if not results:
                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.info(
                    "VECTOR_SEARCH_SPAN_ATTRS",
                    duration_ms=duration_ms,
                    result_count=0,
                    success=True,
                )
                # Record metrics for monitoring (T108)
                record_source_query(
                    source_type="vector_search",
                    source_name=self._source_name,
                    latency_ms=duration_ms,
                    success=True,
                )
                return ToolResult(
                    content="No results found matching your query.",
                    success=True,
                    sources=[],
                    data={"query": query, "num_results": 0},
                )

            # Deduplicate and format results
            unique_results = self._deduplicate_results(results)
            result_count = len(unique_results)

            logger.info(
                "VECTOR_SEARCH_RAW_RESULTS",
                index=self._index_name,
                query=query[:80],
                raw_count=len(unique_results),
            )

            # Build sources for citation tracking
            sources: list[dict[str, Any]] = []
            formatted_results: list[str] = []

            for idx, result in enumerate(unique_results):
                # Skip results with empty content (edge case: content_column
                # exists but individual rows have NULL content).
                if not result.content or not result.content.strip():
                    continue

                # Build navigable workspace URL
                if result.url:
                    source_url = result.url
                else:
                    from deep_research.core.auth import get_workspace_host
                    workspace_host = get_workspace_host()
                    if workspace_host:
                        # Parse catalog.schema.table for catalog explorer URL
                        parts = self._index_name.split(".")
                        if len(parts) == 3:
                            base_url = f"{workspace_host}/explore/data/{parts[0]}/{parts[1]}/{parts[2]}"
                        else:
                            base_url = f"{workspace_host}/compute/vector-search"
                        source_url = f"{base_url}#{result.id or idx}"
                    else:
                        source_url = f"vs://{self._endpoint_name}/{self._index_name}/{result.id}"

                # Title fallback: result metadata → source name → last index segment
                source_title = result.title or self._source_name or self._index_name.rsplit(".", 1)[-1]

                sources.append({
                    "type": "vector_search",
                    "source_name": self._source_name,
                    "index_name": self._index_name,
                    "endpoint_name": self._endpoint_name,
                    "url": source_url,
                    "title": source_title,
                    "content": result.content[:1000] if result.content else "",
                    "relevance_score": result.score,
                    "search_index": idx,
                    "metadata": result.metadata,
                })

                # Format for LLM
                url_display = f"\nURL: {result.url}" if result.url else ""
                formatted_results.append(
                    f"[{idx + 1}] **{result.title}** (score: {result.score:.3f}){url_display}\n"
                    f"    {result.content[:400]}..."
                    if len(result.content) > 400 else
                    f"[{idx + 1}] **{result.title}** (score: {result.score:.3f}){url_display}\n"
                    f"    {result.content}"
                )

            # Accurate count after filtering empty results
            result_count = len(sources)
            if result_count < len(unique_results):
                logger.warning(
                    "VECTOR_SEARCH_EMPTY_RESULTS_FILTERED",
                    index=self._index_name,
                    query=query[:80],
                    total=len(unique_results),
                    skipped=len(unique_results) - result_count,
                    content_column=self._column_roles.content_column if self._column_roles else None,
                    has_column_roles=self._column_roles is not None,
                )

            logger.info(
                "VECTOR_SEARCH_RESULTS_ACCEPTED",
                index=self._index_name,
                query=query[:80],
                accepted=result_count,
                total=len(unique_results),
            )

            content = f"Found {result_count} results from {self._source_name}:\n\n"
            content += "\n\n".join(formatted_results)

            # Log final metrics
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "VECTOR_SEARCH_SPAN_ATTRS",
                duration_ms=duration_ms,
                result_count=result_count,
                success=True,
            )

            # Record metrics for monitoring (T108)
            record_source_query(
                source_type="vector_search",
                source_name=self._source_name,
                latency_ms=duration_ms,
                success=True,
            )

            return ToolResult(
                content=content,
                success=True,
                sources=sources,
                data={
                    "query": query,
                    "num_results": len(unique_results),
                    "source_name": self._source_name,
                    "index_name": self._index_name,
                },
            )

        except Exception as e:
            error_msg = str(e)
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Provide helpful error messages
            if "PERMISSION_DENIED" in error_msg or "403" in error_msg:
                error_msg = (
                    f"Permission denied: You don't have access to index '{self._index_name}'. "
                    "Please verify your permissions."
                )
            elif "NOT_FOUND" in error_msg or "404" in error_msg:
                error_msg = f"Index not found: '{self._index_name}' does not exist."

            # Log error info
            logger.info(
                "VECTOR_SEARCH_SPAN_ATTRS",
                duration_ms=duration_ms,
                result_count=0,
                success=False,
                error_type=type(e).__name__,
            )

            # Record error metrics for monitoring (T108)
            record_source_query(
                source_type="vector_search",
                source_name=self._source_name,
                latency_ms=duration_ms,
                success=False,
                error=error_msg[:200],
            )

            logger.error(
                "VECTOR_SEARCH_ERROR",
                error=error_msg,
                error_type=type(e).__name__,
                index=self._index_name,
                duration_ms=duration_ms,
                exc_info=True,
            )

            return ToolResult(
                content=f"Search failed: {error_msg[:500]}",
                success=False,
                error=error_msg,
            )

    async def _discover_columns(self, client: Any) -> None:
        """Discover queryable columns via get_index() on first execute.

        Called when self._columns is empty (no column info from discovery or config).
        Populates both self._columns and self._column_roles only when a content
        column is found — otherwise the tool stays disabled (empty self._columns
        triggers the guard at execute()).

        One-time cost: ~1s per index. Results are cached on the tool instance.
        """
        try:
            loop = asyncio.get_event_loop()
            index = await loop.run_in_executor(
                None,
                lambda: client.vector_search_indexes.get_index(self._index_name),
            )
            roles = extract_queryable_columns(index)
            if roles and roles.all_columns:
                if roles.content_column is None:
                    logger.warning(
                        "VECTOR_SEARCH_SOURCE_DISABLED",
                        index=self._index_name,
                        reason="no_content_column",
                        columns=roles.all_columns[:10],
                        id_column=roles.id_column,
                    )
                    return  # self._columns stays empty → guard at execute() fires
                if not self._columns:
                    self._columns = roles.all_columns
                self._column_roles = roles
                logger.info(
                    "VECTOR_SEARCH_COLUMNS_ACTIVE",
                    index=self._index_name,
                    content_column=roles.content_column,
                    id_column=roles.id_column,
                    columns=roles.all_columns[:10],
                )
            else:
                logger.warning(
                    "VECTOR_SEARCH_NO_COLUMNS_FOUND",
                    index=self._index_name,
                )
        except Exception as e:
            logger.warning(
                "VECTOR_SEARCH_COLUMN_DISCOVERY_FAILED",
                index=self._index_name,
                error=str(e)[:200],
            )

    def _resolve_query_type(self) -> str | None:
        """Map internal QueryType enum to SDK query_type string."""
        if self._query_type == QueryType.HYBRID:
            return "HYBRID"
        elif self._query_type == QueryType.FULL_TEXT:
            return "HYBRID"  # SDK uses HYBRID for full-text as well
        # ANN is the default — no explicit query_type needed
        return None

    def _build_filters_json(
        self,
        runtime_filters: dict[str, Any] | None,
    ) -> str | None:
        """Build filters_json string for query_index() API.

        Combines default config filters with runtime filters.

        Args:
            runtime_filters: Filters passed at query time from tool arguments.

        Returns:
            JSON string for filters_json param, or None.
        """
        combined_expressions: list[FilterExpression] = list(self._default_filters)

        if runtime_filters:
            for col, val in runtime_filters.items():
                parts = col.split(" ", 1)
                if len(parts) == 2:
                    from deep_research.schemas.query_config import FilterOperator
                    column = parts[0]
                    op_str = parts[1].strip()
                    op_map = {
                        "=": FilterOperator.EQ,
                        "!=": FilterOperator.NE,
                        "<": FilterOperator.LT,
                        "<=": FilterOperator.LE,
                        ">": FilterOperator.GT,
                        ">=": FilterOperator.GE,
                        "LIKE": FilterOperator.LIKE,
                        "NOT LIKE": FilterOperator.NOT_LIKE,
                        "IN": FilterOperator.IN,
                    }
                    operator = op_map.get(op_str.upper(), FilterOperator.EQ)
                    combined_expressions.append(FilterExpression(
                        column=column,
                        operator=operator,
                        value=val,
                    ))
                else:
                    from deep_research.schemas.query_config import FilterOperator
                    combined_expressions.append(FilterExpression(
                        column=col,
                        operator=FilterOperator.EQ,
                        value=val,
                    ))

        if not combined_expressions:
            return None

        # For query_index() API, build a filters_json string.
        # Use SQL syntax (the API accepts SQL filter strings directly).
        if self._filter_syntax == FilterSyntax.SQL:
            return VectorSearchQueryService.build_filters_json(
                filters_sql=" AND ".join(f.to_sql() for f in combined_expressions),
            )
        else:
            result: dict[str, Any] = {}
            for f in combined_expressions:
                result.update(f.to_dict())
            return VectorSearchQueryService.build_filters_json(filters_dict=result)

    @staticmethod
    def _deduplicate_results(
        results: list[VectorSearchResult],
    ) -> list[VectorSearchResult]:
        """Deduplicate results within a single call by content hash.

        Uses a call-local set — intentionally does NOT accumulate across calls.
        Cross-node dedup is handled by the pool system (PoolState.seen_hashes).
        Within-call dedup catches duplicates from multi-query RRF fusion that
        share content but differ in URL/ID.
        """
        import hashlib

        seen: set[str] = set()
        unique: list[VectorSearchResult] = []
        for result in results:
            content_preview = result.content[:500] if result.content else result.id
            content_hash = hashlib.md5(content_preview.encode()).hexdigest()[:16]
            if content_hash not in seen:
                seen.add(content_hash)
                unique.append(result)
        return unique

    def validate_arguments(self, arguments: dict[str, Any]) -> list[str]:
        """Validate search arguments.

        Args:
            arguments: Raw arguments from LLM.

        Returns:
            List of error messages (empty if valid).
        """
        errors: list[str] = []

        # Required: query
        query = arguments.get("query")
        if not query:
            errors.append("'query' is required")
        elif not isinstance(query, str):
            errors.append("'query' must be a string")
        elif len(query) > 2000:
            errors.append("'query' must be 2000 characters or less")

        # Optional: num_results
        num_results = arguments.get("num_results")
        if num_results is not None:
            if not isinstance(num_results, int):
                errors.append("'num_results' must be an integer")
            elif num_results < 1 or num_results > 200:
                errors.append("'num_results' must be between 1 and 200")

        # Optional: filters
        filters = arguments.get("filters")
        if filters is not None and not isinstance(filters, dict):
            errors.append("'filters' must be an object")

        return errors
