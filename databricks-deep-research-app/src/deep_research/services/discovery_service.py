"""Data Source Discovery Service.

This module implements automatic discovery of data sources available to the user
via Databricks SDK APIs using On-Behalf-Of (OBO) authentication.

Discovered source types:
- Vector Search indexes (via vector_search_endpoints and vector_search_indexes APIs)
- Genie spaces (via genie API)
- Knowledge Assistants (via serving_endpoints API, filtered by type/tags)

Key Features:
- Parallel discovery across all source types
- Caching with 5-minute TTL per user
- Graceful handling of partial failures
- Metadata extraction for query configuration

SDK Integration (from research.md):
- w.vector_search_endpoints.list_endpoints() → discover endpoints
- w.vector_search_indexes.list_indexes(endpoint_name) → discover indexes
- w.genie.list_spaces() → discover Genie spaces
- w.serving_endpoints.list() → discover Knowledge Assistants
- All use ModelServingUserCredentials() for OBO authentication
"""

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

from databricks.sdk import WorkspaceClient

from deep_research.core.auth import get_user_workspace_client, get_workspace_client
from deep_research.core.logging_utils import get_logger
from deep_research.schemas.data_source import DataSourceType
from deep_research.schemas.discovery import (
    DiscoveredSource,
    DiscoveryError,
    DiscoveryResponse,
    DiscoveryStatus,
    FilterColumnInfo,
    GenieSpaceMetadata,
    QueryType,
    ServingEndpointMetadata,
    SourceMetadataResponse,
    VectorSearchMetadata,
)
from deep_research.services.discovery_cache import DiscoveryCache, get_discovery_cache
from deep_research.services.mcp_discovery import discover_mcp_connections

logger = get_logger(__name__)

# Discovery configuration - per-type timeouts for graceful degradation
DISCOVERY_TIMEOUT_VS = timedelta(seconds=15)  # Vector Search can be slow
DISCOVERY_TIMEOUT_GENIE = timedelta(seconds=10)
DISCOVERY_TIMEOUT_SERVING = timedelta(seconds=10)
DISCOVERY_TIMEOUT_MCP = timedelta(seconds=5)
CACHE_TTL = timedelta(minutes=5)  # Cache TTL

# Genie pagination - SDK does NOT auto-paginate list_spaces()
GENIE_PAGE_SIZE = 100   # Spaces per page request
GENIE_MAX_PAGES = 50    # Safety limit: 50 × 100 = 5,000 spaces max

# Heuristics for identifying Knowledge Assistants
ASSISTANT_NAME_PATTERNS = ["assistant", "expert", "advisor", "knowledge", "agent"]
ASSISTANT_TAG_KEYS = ["type", "assistant_type", "knowledge_assistant"]


class DiscoveryService:
    """Service for discovering available data sources via Databricks SDK.

    Uses OBO authentication to discover sources the user has access to.
    Results are cached per-user with configurable TTL.

    Example:
        service = DiscoveryService()

        # Discover all sources
        response = await service.discover_all(user_token)

        # Get specific source metadata
        metadata = await service.get_source_metadata(user_token, "vs:catalog.schema.index")

        # Force refresh
        response = await service.refresh(user_token)
    """

    def __init__(self, cache: DiscoveryCache | None = None) -> None:
        """Initialize the discovery service.

        Args:
            cache: Optional cache instance. Uses global cache if not provided.
        """
        self._cache = cache or get_discovery_cache()

    async def _get_client(self, user_token: str | None) -> WorkspaceClient:
        """Get WorkspaceClient with appropriate authentication.

        Args:
            user_token: User's OAuth token for OBO. If None, uses service principal.

        Returns:
            Configured WorkspaceClient.
        """
        if user_token:
            return get_user_workspace_client(user_token)
        return get_workspace_client()

    # =========================================================================
    # Vector Search Discovery (T010c)
    # =========================================================================

    async def discover_vector_search_sources(
        self,
        user_token: str | None,
    ) -> tuple[list[DiscoveredSource], DiscoveryError | None]:
        """Discover Vector Search indexes accessible to the user.

        Uses list_endpoints() and list_indexes() for fast discovery.
        Detailed metadata (embedding columns, filter columns) is fetched
        on-demand via get_source_metadata().

        APIs:
        - w.vector_search_endpoints.list_endpoints() → Iterator[EndpointInfo]
        - w.vector_search_indexes.list_indexes(endpoint_name) → Iterator[MiniVectorIndex]

        MiniVectorIndex fields: name, endpoint_name, primary_key, index_type

        Args:
            user_token: User's OAuth token for OBO authentication. If None, uses
                       workspace client (local dev with user's profile).

        Returns:
            Tuple of (discovered sources, error if any).
        """
        sources: list[DiscoveredSource] = []

        try:
            client = await self._get_client(user_token)
            loop = asyncio.get_event_loop()

            # 1. List all VS endpoints (single API call)
            endpoints = await loop.run_in_executor(
                None,
                lambda: list(client.vector_search_endpoints.list_endpoints()),
            )

            logger.debug("VS_DISCOVERY_ENDPOINTS", count=len(endpoints))

            # 2. List indexes for each endpoint (parallel)
            async def list_indexes_for_endpoint(ep_name: str) -> list[tuple[str, Any]]:
                """List indexes for one endpoint, returns (endpoint_name, mini_index) tuples."""
                try:
                    def list_fn() -> list[Any]:
                        return list(client.vector_search_indexes.list_indexes(ep_name))

                    mini_indexes = await loop.run_in_executor(None, list_fn)
                    return [(ep_name, idx) for idx in mini_indexes if idx.name]
                except Exception as e:
                    logger.warning(
                        "VS_DISCOVERY_ENDPOINT_ERROR",
                        endpoint=ep_name,
                        error=str(e)[:200],
                    )
                    return []

            # Run list_indexes for all endpoints in parallel
            endpoint_names = [ep.name for ep in endpoints if ep.name]
            index_results = await asyncio.gather(
                *[list_indexes_for_endpoint(name) for name in endpoint_names],
                return_exceptions=True,
            )

            # Flatten results
            for result in index_results:
                if isinstance(result, Exception):
                    logger.warning("VS_DISCOVERY_BATCH_ERROR", error=str(result)[:200])
                    continue

                # result is list[tuple[str, Any]] at this point
                if not isinstance(result, list):
                    continue

                for ep_name, mini_index in result:
                    index_name = mini_index.name

                    # Use MiniVectorIndex directly - NO get_index() call
                    primary_key = getattr(mini_index, "primary_key", None) or "id"
                    index_type_raw = getattr(mini_index, "index_type", None)
                    index_type_str = str(index_type_raw.value) if index_type_raw and hasattr(index_type_raw, "value") else str(index_type_raw or "DELTA_SYNC")
                    # Map to valid Literal values
                    index_type: str = "DELTA_SYNC" if index_type_str not in ("DELTA_SYNC", "DIRECT_ACCESS") else index_type_str

                    # Build minimal metadata - detailed fields populated on-demand
                    metadata = VectorSearchMetadata(
                        index_name=index_name,
                        endpoint_name=ep_name,
                        primary_key=primary_key,
                        index_type=index_type,
                        # Detailed fields - populated on-demand via get_source_metadata()
                        embedding_columns=[],  # Empty list, populated on-demand
                        embedding_dimension=None,
                        embedding_model=None,
                        filter_columns=[],  # Empty list, populated on-demand
                        supported_query_types=[QueryType.ANN],  # Safe default
                        supports_reranking=False,
                        row_count=None,
                        is_ready=True,  # Assume ready since it's listed
                    )

                    # Extract table name from full index path
                    parts = index_name.split(".")
                    display_name = parts[-1] if parts else index_name

                    source = DiscoveredSource(
                        source_id=f"vs:{index_name}",
                        source_type=DataSourceType.VECTOR_SEARCH,
                        name=display_name,
                        endpoint_name=ep_name,
                        description=f"Vector Search index: {index_name}",
                        status=DiscoveryStatus.READY,
                        capabilities=["ann"],  # Minimal - detailed via metadata
                        metadata=metadata.model_dump(),
                        discovered_at=datetime.now(UTC),
                    )
                    sources.append(source)

            if sources:
                sources = await self._enrich_with_columns(sources, user_token)

            logger.info("VS_DISCOVERY_COMPLETE", source_count=len(sources))
            return sources, None

        except Exception as e:
            error_msg = str(e)
            error_code = "UNKNOWN_ERROR"

            if "PERMISSION_DENIED" in error_msg or "403" in error_msg:
                error_code = "PERMISSION_DENIED"
            elif "unavailable" in error_msg.lower() or "503" in error_msg:
                error_code = "SERVICE_UNAVAILABLE"

            logger.error("VS_DISCOVERY_FAILED", error=error_msg[:200])
            return sources, DiscoveryError(
                source_type=DataSourceType.VECTOR_SEARCH,
                error_code=error_code,
                error_message=f"Failed to discover Vector Search sources: {error_msg[:200]}",
                retryable=error_code != "PERMISSION_DENIED",
            )

    async def _enrich_with_columns(
        self,
        sources: list[DiscoveredSource],
        user_token: str | None,
    ) -> list[DiscoveredSource]:
        """Validate and enrich all VS sources with column info.

        Every source is checked via get_index(). Sources without a discoverable
        content column are filtered — they produce empty results and "Untitled"
        entries in the citation pipeline.
        """
        from deep_research.services.vector_search_query import extract_queryable_columns

        client = await self._get_client(user_token)
        loop = asyncio.get_event_loop()
        semaphore = asyncio.Semaphore(10)

        async def enrich_one(source: DiscoveredSource) -> DiscoveredSource | None:
            index_name = source.metadata.get("index_name", "")
            if not index_name:
                return source
            async with semaphore:
                try:
                    idx_name = index_name

                    index = await loop.run_in_executor(
                        None,
                        lambda: client.vector_search_indexes.get_index(idx_name),
                    )
                    roles = extract_queryable_columns(index)
                    if roles and roles.all_columns:
                        if roles.content_column is None:
                            logger.warning(
                                "VS_SOURCE_NO_CONTENT_COLUMN",
                                index=index_name,
                                columns=roles.all_columns[:10],
                                id_column=roles.id_column,
                            )
                            return None
                        metadata = dict(source.metadata)
                        metadata["queryable_columns"] = roles.all_columns
                        metadata["content_column"] = roles.content_column
                        return source.model_copy(update={"metadata": metadata})
                    return None
                except Exception as e:
                    logger.debug(
                        "VS_ENRICHMENT_SKIP",
                        index=index_name,
                        error=str(e)[:100],
                    )
            return None

        enriched = await asyncio.gather(
            *[enrich_one(s) for s in sources],
            return_exceptions=True,
        )
        result: list[DiscoveredSource] = []
        filtered_count = 0
        for r in enriched:
            if r is None or isinstance(r, BaseException):
                filtered_count += 1
                continue
            result.append(r)
        logger.info(
            "VS_ENRICHMENT_COMPLETE",
            enriched=len(result),
            filtered=filtered_count,
            total=len(sources),
        )
        return result

    def _extract_vector_search_metadata(
        self,
        index: Any,
        endpoint_name: str,
    ) -> VectorSearchMetadata:
        """Extract metadata from a Vector Search index.

        Args:
            index: VectorIndex object from SDK.
            endpoint_name: Name of the VS endpoint.

        Returns:
            VectorSearchMetadata with extracted information.
        """
        # Default values
        index_name = getattr(index, "name", "unknown")
        primary_key = getattr(index, "primary_key", "id")

        # Determine index type
        index_type = "DELTA_SYNC"
        if hasattr(index, "index_type") and index.index_type:
            index_type = str(index.index_type.value) if hasattr(index.index_type, "value") else str(index.index_type)

        # Extract embedding columns and filter columns from spec
        embedding_columns: list[str] = []
        filter_columns: list[FilterColumnInfo] = []
        embedding_dimension: int | None = None
        embedding_model: str | None = None

        spec = getattr(index, "delta_sync_index_spec", None) or getattr(index, "direct_access_index_spec", None)

        if spec:
            # Get embedding source columns
            if hasattr(spec, "embedding_source_columns") and spec.embedding_source_columns:
                for col in spec.embedding_source_columns:
                    if hasattr(col, "name"):
                        embedding_columns.append(col.name)
                    if hasattr(col, "embedding_model_endpoint_name") and col.embedding_model_endpoint_name:
                        embedding_model = col.embedding_model_endpoint_name

            # Get embedding dimensions
            if hasattr(spec, "embedding_vector_columns") and spec.embedding_vector_columns:
                for col in spec.embedding_vector_columns:
                    if hasattr(col, "embedding_dimension"):
                        embedding_dimension = col.embedding_dimension

            # Extract filterable columns from columns_to_sync or schema
            if hasattr(spec, "columns_to_sync") and spec.columns_to_sync:
                for col in spec.columns_to_sync:
                    col_name = col.name if hasattr(col, "name") else str(col)
                    data_type = "string"  # Default

                    # Try to get data type
                    if hasattr(col, "data_type"):
                        dtype = str(col.data_type).lower()
                        if "int" in dtype:
                            data_type = "integer"
                        elif "float" in dtype or "double" in dtype:
                            data_type = "float"
                        elif "bool" in dtype:
                            data_type = "boolean"
                        elif "timestamp" in dtype or "date" in dtype:
                            data_type = "timestamp"

                    # Determine operators based on type
                    operators = ["=", "!="]
                    if data_type in ("integer", "float", "timestamp"):
                        operators.extend(["<", "<=", ">", ">="])
                    if data_type == "string":
                        operators.extend(["LIKE", "NOT LIKE", "IN"])

                    # data_type is known to be a valid Literal value at this point
                    filter_columns.append(
                        FilterColumnInfo(
                            name=col_name,
                            data_type=data_type,
                            operators=operators,
                        )
                    )

        # Determine supported query types
        supported_query_types = [QueryType.ANN]  # Always supported

        # HYBRID requires text columns (simplified heuristic)
        has_text_columns = any(c.data_type == "string" for c in filter_columns)
        if has_text_columns or embedding_columns:
            supported_query_types.append(QueryType.HYBRID)
            # FULL_TEXT is beta and typically available if HYBRID is
            supported_query_types.append(QueryType.FULL_TEXT)

        # Check if index is ready
        is_ready = True
        if hasattr(index, "status") and index.status:
            if hasattr(index.status, "ready"):
                is_ready = bool(index.status.ready)
            elif hasattr(index.status, "index_status"):
                is_ready = str(index.status.index_status).upper() == "ONLINE"

        # Try to get row count
        row_count = None
        if (
            hasattr(index, "status") and index.status
            and hasattr(index.status, "num_of_source_rows")
        ):
            row_count = index.status.num_of_source_rows

        # Extract queryable columns via shared utility
        from deep_research.services.vector_search_query import extract_queryable_columns
        roles = extract_queryable_columns(index)
        queryable_cols = roles.all_columns if roles else []

        return VectorSearchMetadata(
            index_name=index_name,
            endpoint_name=endpoint_name,
            primary_key=primary_key,
            index_type=index_type,
            embedding_columns=embedding_columns,
            embedding_dimension=embedding_dimension,
            embedding_model=embedding_model,
            queryable_columns=queryable_cols,
            filter_columns=filter_columns,
            supported_query_types=supported_query_types,
            supports_reranking=bool(filter_columns),  # Reranking needs text columns
            row_count=row_count,
            is_ready=is_ready,
        )

    # =========================================================================
    # Genie Discovery (T010d)
    # =========================================================================

    async def discover_genie_spaces(
        self,
        user_token: str | None,
    ) -> tuple[list[DiscoveredSource], DiscoveryError | None]:
        """Discover Genie spaces accessible to the user.

        Uses only list_spaces() for fast discovery. Detailed metadata
        (owner, created_at) is fetched on-demand via get_source_metadata().

        API: w.genie.list_spaces() → GenieListSpacesResponse
        Response fields: space_id, title, description, warehouse_id

        Args:
            user_token: User's OAuth token for OBO authentication. If None, uses
                       workspace client (local dev with user's profile).

        Returns:
            Tuple of (discovered sources, error if any).
        """
        sources: list[DiscoveredSource] = []

        try:
            client = await self._get_client(user_token)
            loop = asyncio.get_event_loop()

            # Paginate through all Genie spaces — SDK does NOT auto-paginate
            def _list_all_spaces() -> list[Any]:
                """Fetch all Genie spaces with manual pagination (blocking, runs in executor)."""
                all_spaces: list[Any] = []
                seen_ids: set[str] = set()
                page_token: str | None = None

                for page_num in range(1, GENIE_MAX_PAGES + 1):
                    response = client.genie.list_spaces(
                        page_size=GENIE_PAGE_SIZE,
                        page_token=page_token,
                    )

                    if not response or not hasattr(response, "spaces"):
                        break

                    page_spaces = response.spaces or []

                    # Dedup by space_id to guard against API edge cases
                    for space in page_spaces:
                        sid = getattr(space, "space_id", None) or getattr(space, "id", None)
                        if sid and sid not in seen_ids:
                            seen_ids.add(sid)
                            all_spaces.append(space)

                    logger.debug(
                        "GENIE_DISCOVERY_PAGE",
                        page=page_num,
                        page_count=len(page_spaces),
                        total_so_far=len(all_spaces),
                    )

                    # Follow next_page_token or stop
                    next_token = getattr(response, "next_page_token", None)
                    if not next_token:
                        break
                    page_token = next_token

                return all_spaces

            try:
                spaces = await loop.run_in_executor(None, _list_all_spaces)
            except AttributeError:
                # Genie API may not be available in all SDK versions
                logger.warning("GENIE_API_NOT_AVAILABLE")
                return sources, DiscoveryError(
                    source_type=DataSourceType.GENIE,
                    error_code="API_NOT_AVAILABLE",
                    error_message="Genie API not available in current SDK version",
                    retryable=False,
                )

            logger.debug("GENIE_DISCOVERY_SPACES", count=len(spaces))

            for space_summary in spaces:
                # Extract space_id - handle both naming conventions
                space_id = getattr(space_summary, "space_id", None) or getattr(space_summary, "id", None)
                if not space_id:
                    logger.warning("GENIE_DISCOVERY_SKIP_NO_ID")
                    continue

                # Use summary fields directly - NO get_space() call needed
                title = getattr(space_summary, "title", None) or space_id
                description = getattr(space_summary, "description", None)
                warehouse_id = getattr(space_summary, "warehouse_id", None)

                # Build metadata with available fields
                # owner and created_at will be fetched on-demand via get_source_metadata()
                metadata = GenieSpaceMetadata(
                    space_id=space_id,
                    title=title,
                    description=description,
                    warehouse_id=warehouse_id,
                    owner=None,  # Populated on-demand
                    created_at=None,  # Populated on-demand
                )

                source = DiscoveredSource(
                    source_id=f"genie:{space_id}",
                    source_type=DataSourceType.GENIE,
                    name=title,
                    endpoint_name=space_id,
                    description=description,
                    status=DiscoveryStatus.READY,
                    capabilities=["sql", "conversation", "follow_up"],
                    metadata=metadata.model_dump(),
                    discovered_at=datetime.now(UTC),
                )
                sources.append(source)

            logger.info("GENIE_DISCOVERY_COMPLETE", source_count=len(sources))
            return sources, None

        except Exception as e:
            error_msg = str(e)
            error_code = "UNKNOWN_ERROR"

            if "PERMISSION_DENIED" in error_msg or "403" in error_msg:
                error_code = "PERMISSION_DENIED"
            elif "unavailable" in error_msg.lower() or "503" in error_msg:
                error_code = "SERVICE_UNAVAILABLE"

            logger.error("GENIE_DISCOVERY_FAILED", error=error_msg[:200])
            return sources, DiscoveryError(
                source_type=DataSourceType.GENIE,
                error_code=error_code,
                error_message=f"Failed to discover Genie spaces: {error_msg[:200]}",
                retryable=error_code != "PERMISSION_DENIED",
            )

    # =========================================================================
    # Serving Endpoint Discovery (T010e)
    # =========================================================================

    async def discover_serving_endpoints(
        self,
        user_token: str | None,
        include_all: bool = False,
    ) -> tuple[list[DiscoveredSource], DiscoveryError | None]:
        """Discover serving endpoints that are likely Knowledge Assistants.

        Uses:
        - w.serving_endpoints.list() → Iterator[ServingEndpoint]

        Filters by heuristics:
        - Endpoint name patterns (assistant, expert, knowledge, etc.)
        - Tags indicating Knowledge Assistant type
        - Endpoint type

        When include_all=True, includes all serving endpoints (not just detected KAs).

        Args:
            user_token: User's OAuth token for OBO authentication. If None, uses
                       workspace client (local dev with user's profile).
            include_all: If True, include all serving endpoints, not just detected KAs.

        Returns:
            Tuple of (discovered sources, error if any).
        """
        sources: list[DiscoveredSource] = []

        try:
            client = await self._get_client(user_token)
            loop = asyncio.get_event_loop()

            # List all serving endpoints
            endpoints = await loop.run_in_executor(
                None,
                lambda: list(client.serving_endpoints.list()),
            )

            logger.debug("SERVING_DISCOVERY_ENDPOINTS", count=len(endpoints))

            for endpoint in endpoints:
                if not endpoint.name:
                    continue

                # Check if this is likely a Knowledge Assistant
                is_ka = self._is_knowledge_assistant(endpoint)

                # Skip if not a KA and we're not including all endpoints
                if not include_all and not is_ka:
                    continue

                # Extract metadata
                endpoint_type = "CUSTOM"
                if hasattr(endpoint, "endpoint_type") and endpoint.endpoint_type:
                    endpoint_type = str(endpoint.endpoint_type)

                state = "READY"
                if hasattr(endpoint, "state") and endpoint.state:
                    state_val = endpoint.state
                    if hasattr(state_val, "value"):
                        state = state_val.value
                    elif hasattr(state_val, "ready"):
                        state = "READY" if state_val.ready else "NOT_READY"
                    else:
                        state = str(state_val).upper()

                tags: dict[str, str] = {}
                if hasattr(endpoint, "tags") and endpoint.tags:
                    tags = dict(endpoint.tags) if isinstance(endpoint.tags, dict) else {}

                creator = getattr(endpoint, "creator", None)

                # Determine assistant type from tags
                assistant_type = None
                for key in ASSISTANT_TAG_KEYS:
                    if key in tags:
                        assistant_type = tags[key]
                        break

                metadata = ServingEndpointMetadata(
                    endpoint_name=endpoint.name,
                    endpoint_type=endpoint_type,
                    state=state if state in ("READY", "NOT_READY", "PENDING") else "NOT_READY",
                    tags=tags,
                    is_knowledge_assistant=is_ka,
                    assistant_type=assistant_type,
                    creator=creator,
                )

                # Determine status
                status = DiscoveryStatus.READY
                if state != "READY":
                    status = DiscoveryStatus.UNAVAILABLE

                # Set description based on whether it's a KA or generic endpoint
                if is_ka:
                    description = f"Knowledge Assistant endpoint ({assistant_type or 'general'})"
                else:
                    description = f"Serving endpoint ({endpoint_type})"

                source = DiscoveredSource(
                    source_id=f"assistant:{endpoint.name}",
                    source_type=DataSourceType.KNOWLEDGE_ASSISTANT,
                    name=endpoint.name,
                    endpoint_name=endpoint.name,
                    description=description,
                    status=status,
                    capabilities=["chat", "context"],
                    metadata=metadata.model_dump(),
                    discovered_at=datetime.now(UTC),
                )
                sources.append(source)

            logger.info("SERVING_DISCOVERY_COMPLETE", source_count=len(sources))
            return sources, None

        except Exception as e:
            error_msg = str(e)
            error_code = "UNKNOWN_ERROR"

            if "PERMISSION_DENIED" in error_msg or "403" in error_msg:
                error_code = "PERMISSION_DENIED"
            elif "unavailable" in error_msg.lower() or "503" in error_msg:
                error_code = "SERVICE_UNAVAILABLE"

            logger.error("SERVING_DISCOVERY_FAILED", error=error_msg[:200])
            return sources, DiscoveryError(
                source_type=DataSourceType.KNOWLEDGE_ASSISTANT,
                error_code=error_code,
                error_message=f"Failed to discover serving endpoints: {error_msg[:200]}",
                retryable=error_code != "PERMISSION_DENIED",
            )

    def _is_knowledge_assistant(self, endpoint: Any) -> bool:
        """Determine if a serving endpoint is likely a Knowledge Assistant.

        Uses heuristics:
        - Name patterns (assistant, expert, knowledge, agent)
        - Tags indicating assistant type
        - Endpoint type (CUSTOM endpoints more likely)

        Args:
            endpoint: ServingEndpoint object from SDK.

        Returns:
            True if endpoint is likely a Knowledge Assistant.
        """
        name = getattr(endpoint, "name", "").lower()

        # Check name patterns
        for pattern in ASSISTANT_NAME_PATTERNS:
            if pattern in name:
                return True

        # Check tags
        tags = getattr(endpoint, "tags", {}) or {}
        if isinstance(tags, dict):
            for key in ASSISTANT_TAG_KEYS:
                if key in tags:
                    return True

            # Check for explicit knowledge_assistant=true tag
            if tags.get("knowledge_assistant", "").lower() == "true":
                return True

        return False

    # =========================================================================
    # MCP Server Discovery
    # =========================================================================

    async def discover_mcp_server_sources(
        self,
        user_token: str | None,
    ) -> tuple[list[DiscoveredSource], DiscoveryError | None]:
        """Discover MCP servers visible to the user.

        MCP servers are surfaced as normal selectable sources in the chat source
        browser. The runtime still attaches them through ``enabled_mcp_servers``;
        discovery only provides stable ``mcp:<connection_name>`` source IDs for
        selection.
        """
        sources: list[DiscoveredSource] = []

        try:
            client = await self._get_client(user_token)
            loop = asyncio.get_event_loop()
            servers = await loop.run_in_executor(
                None,
                lambda: discover_mcp_connections(client),
            )

            for server in servers:
                connection_name = (server.connection_name or server.name).strip()
                if not connection_name:
                    continue

                metadata = dict(server.metadata or {})
                metadata.update(
                    {
                        "client_kind": server.client_kind,
                        "connection_name": server.connection_name,
                        "managed_target": server.managed_target,
                    }
                )

                sources.append(
                    DiscoveredSource(
                        source_id=f"mcp:{connection_name}",
                        source_type=DataSourceType.MCP_SERVER,
                        name=server.name or connection_name,
                        endpoint_name=connection_name,
                        description=server.description or "MCP server",
                        status=DiscoveryStatus.READY,
                        capabilities=["mcp", "tools"],
                        metadata=metadata,
                        discovered_at=datetime.now(UTC),
                    )
                )

            logger.info("MCP_DISCOVERY_COMPLETE", source_count=len(sources))
            return sources, None

        except Exception as e:
            error_msg = str(e)
            error_code = "UNKNOWN_ERROR"

            if "PERMISSION_DENIED" in error_msg or "403" in error_msg:
                error_code = "PERMISSION_DENIED"
            elif "unavailable" in error_msg.lower() or "503" in error_msg:
                error_code = "SERVICE_UNAVAILABLE"

            logger.error("MCP_DISCOVERY_FAILED", error=error_msg[:200])
            return sources, DiscoveryError(
                source_type=DataSourceType.MCP_SERVER,
                error_code=error_code,
                error_message=f"Failed to discover MCP server sources: {error_msg[:200]}",
                retryable=error_code != "PERMISSION_DENIED",
            )

    # =========================================================================
    # Main Discovery Methods
    # =========================================================================

    async def discover_all(
        self,
        user_id: str,
        user_token: str | None = None,
        force_refresh: bool = False,
        source_types: list[DataSourceType] | None = None,
        include_all_endpoints: bool = False,
    ) -> DiscoveryResponse:
        """Discover all data sources with graceful degradation.

        Each discovery type runs with independent timeout. Partial results
        are returned even if some types fail or timeout.

        Args:
            user_id: User ID for cache keying (always available after auth).
            user_token: Optional OBO token for user-level access in Databricks Apps.
                       If None, uses workspace client (local dev with user's profile).
            force_refresh: If True, bypass cache and re-discover.
            source_types: Optional list of specific types to discover.
            include_all_endpoints: If True, include all serving endpoints, not just KAs.

        Returns:
            DiscoveryResponse with all discovered sources.
        """
        # Check cache first (unless force refresh)
        if not force_refresh:
            cached = await self._cache.get(user_id=user_id)
            if cached:
                logger.debug(
                    "DISCOVERY_CACHE_HIT",
                    user_id=user_id[:8] if len(user_id) > 8 else user_id,
                    source_count=len(cached),
                )
                return self._build_response(cached, from_cache=True)

        types_to_discover = source_types or [
            DataSourceType.VECTOR_SEARCH,
            DataSourceType.GENIE,
            DataSourceType.KNOWLEDGE_ASSISTANT,
            DataSourceType.MCP_SERVER,
        ]

        # Helper function to run discovery with per-type timeout
        async def run_with_timeout(
            coro: Any,
            timeout: timedelta,
            source_type: DataSourceType,
        ) -> tuple[list[DiscoveredSource], DiscoveryError | None]:
            """Run discovery with timeout, returning partial results on timeout."""
            try:
                return await asyncio.wait_for(coro, timeout=timeout.total_seconds())
            except TimeoutError:
                logger.warning(
                    "DISCOVERY_TYPE_TIMEOUT",
                    source_type=source_type.value,
                    timeout_seconds=timeout.total_seconds(),
                )
                return [], DiscoveryError(
                    source_type=source_type,
                    error_code="TIMEOUT",
                    error_message=f"Discovery timed out after {timeout.total_seconds()}s",
                    retryable=True,
                )

        # Build tasks with per-type timeouts
        tasks = []
        if DataSourceType.VECTOR_SEARCH in types_to_discover:
            tasks.append(
                run_with_timeout(
                    self.discover_vector_search_sources(user_token),
                    DISCOVERY_TIMEOUT_VS,
                    DataSourceType.VECTOR_SEARCH,
                )
            )
        if DataSourceType.GENIE in types_to_discover:
            tasks.append(
                run_with_timeout(
                    self.discover_genie_spaces(user_token),
                    DISCOVERY_TIMEOUT_GENIE,
                    DataSourceType.GENIE,
                )
            )
        if DataSourceType.KNOWLEDGE_ASSISTANT in types_to_discover:
            tasks.append(
                run_with_timeout(
                    self.discover_serving_endpoints(user_token, include_all=include_all_endpoints),
                    DISCOVERY_TIMEOUT_SERVING,
                    DataSourceType.KNOWLEDGE_ASSISTANT,
                )
            )
        if DataSourceType.MCP_SERVER in types_to_discover:
            tasks.append(
                run_with_timeout(
                    self.discover_mcp_server_sources(user_token),
                    DISCOVERY_TIMEOUT_MCP,
                    DataSourceType.MCP_SERVER,
                )
            )

        # Execute all tasks in parallel - no outer timeout!
        # Each task has its own timeout and returns errors instead of raising
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results (now handles partial success)
        all_sources: list[DiscoveredSource] = []
        errors: list[DiscoveryError] = []

        for result in results:
            if isinstance(result, Exception):
                logger.error("DISCOVERY_TASK_EXCEPTION", error=str(result)[:200])
                continue

            if isinstance(result, tuple):
                sources, error = result
                all_sources.extend(sources)
                if error:
                    errors.append(error)

        # Cache results even if partial (with error indicators)
        if all_sources:  # Only cache if we got something
            await self._cache.set(user_id=user_id, sources=all_sources, ttl=CACHE_TTL)

        logger.info(
            "DISCOVERY_COMPLETE",
            user_id=user_id[:8] if len(user_id) > 8 else user_id,
            total_sources=len(all_sources),
            error_count=len(errors),
        )

        return self._build_response(all_sources, errors=errors if errors else None)

    def _build_response(
        self,
        sources: list[DiscoveredSource],
        from_cache: bool = False,
        errors: list[DiscoveryError] | None = None,
    ) -> DiscoveryResponse:
        """Build a DiscoveryResponse from discovered sources.

        Args:
            sources: List of discovered sources.
            from_cache: Whether results came from cache.
            errors: Optional list of discovery errors.

        Returns:
            DiscoveryResponse object.
        """
        # Group by type
        by_type: dict[str, list[DiscoveredSource]] = {}
        for source in sources:
            type_key = source.source_type.value if isinstance(source.source_type, DataSourceType) else source.source_type
            if type_key not in by_type:
                by_type[type_key] = []
            by_type[type_key].append(source)

        now = datetime.now(UTC)

        return DiscoveryResponse(
            sources=sources,
            total_count=len(sources),
            by_type=by_type,
            discovered_at=now,
            cached=from_cache,
            cache_expires_at=now + CACHE_TTL if from_cache else None,
            errors=errors,
        )

    async def get_source_metadata(
        self,
        user_id: str,
        user_token: str | None = None,
        source_id: str = "",
    ) -> SourceMetadataResponse | None:
        """Get detailed metadata for a specific discovered source.

        Fetches live metadata from Databricks APIs for fields not
        available in the discovery summary.

        Args:
            user_id: User ID (for cache lookup of base source info).
            user_token: Optional OBO token for OBO authentication.
            source_id: Source ID (e.g., 'genie:space_id', 'vs:index_name').

        Returns:
            SourceMetadataResponse with detailed metadata, or None if not found.
        """
        # Parse source type and identifier
        if source_id.startswith("genie:"):
            space_id = source_id[6:]
            return await self._fetch_genie_metadata(user_token, space_id)
        elif source_id.startswith("vs:"):
            index_name = source_id[3:]
            return await self._fetch_vector_search_metadata(user_token, index_name)
        elif source_id.startswith("assistant:"):
            endpoint_name = source_id[10:]
            return await self._fetch_serving_metadata(user_id, user_token, endpoint_name)

        return None

    async def _fetch_genie_metadata(
        self,
        user_token: str | None,
        space_id: str,
    ) -> SourceMetadataResponse | None:
        """Fetch detailed Genie space metadata via get_space() API."""
        try:
            client = await self._get_client(user_token)
            loop = asyncio.get_event_loop()

            space = await loop.run_in_executor(
                None,
                lambda: client.genie.get_space(space_id),
            )

            if not space:
                return None

            # Extract all available metadata
            title = getattr(space, "title", None) or space_id
            description = getattr(space, "description", None)
            warehouse_id = getattr(space, "warehouse_id", None)
            owner = getattr(space, "creator", None)
            created_at = getattr(space, "created_at", None)

            metadata = GenieSpaceMetadata(
                space_id=space_id,
                title=title,
                description=description,
                warehouse_id=warehouse_id,
                owner=owner,
                created_at=created_at,
            )

            source = DiscoveredSource(
                source_id=f"genie:{space_id}",
                source_type=DataSourceType.GENIE,
                name=title,
                endpoint_name=space_id,
                description=description,
                status=DiscoveryStatus.READY,
                capabilities=["sql", "conversation", "follow_up"],
                metadata=metadata.model_dump(),
                discovered_at=datetime.now(UTC),
            )

            return SourceMetadataResponse(source=source, genie=metadata)

        except Exception as e:
            logger.warning("GENIE_METADATA_FETCH_ERROR", space_id=space_id, error=str(e)[:200])
            return None

    async def _fetch_vector_search_metadata(
        self,
        user_token: str | None,
        index_name: str,
    ) -> SourceMetadataResponse | None:
        """Fetch detailed Vector Search index metadata via get_index() API."""
        try:
            client = await self._get_client(user_token)
            loop = asyncio.get_event_loop()

            index = await loop.run_in_executor(
                None,
                lambda: client.vector_search_indexes.get_index(index_name),
            )

            if not index:
                return None

            # Extract endpoint name
            endpoint_name = getattr(index, "endpoint_name", "")

            # Use existing helper to extract full metadata
            metadata = self._extract_vector_search_metadata(index, endpoint_name)

            # Determine capabilities from metadata
            capabilities = ["ann"]
            if QueryType.HYBRID in metadata.supported_query_types:
                capabilities.append("hybrid")
            if QueryType.FULL_TEXT in metadata.supported_query_types:
                capabilities.append("full_text")
            if metadata.supports_reranking:
                capabilities.append("reranking")

            status = DiscoveryStatus.READY if metadata.is_ready else DiscoveryStatus.SYNCING

            parts = index_name.split(".")
            display_name = parts[-1] if parts else index_name

            source = DiscoveredSource(
                source_id=f"vs:{index_name}",
                source_type=DataSourceType.VECTOR_SEARCH,
                name=display_name,
                endpoint_name=endpoint_name,
                description=f"Vector Search index: {index_name}",
                status=status,
                capabilities=capabilities,
                metadata=metadata.model_dump(),
                discovered_at=datetime.now(UTC),
            )

            return SourceMetadataResponse(source=source, vector_search=metadata)

        except Exception as e:
            logger.warning("VS_METADATA_FETCH_ERROR", index_name=index_name, error=str(e)[:200])
            return None

    async def _fetch_serving_metadata(
        self,
        user_id: str,
        user_token: str | None,
        endpoint_name: str,
    ) -> SourceMetadataResponse | None:
        """Fetch serving endpoint metadata (from cached discovery)."""
        # Serving endpoints don't need additional API calls - all metadata is in list()
        # Just look up from cached discovery
        response = await self.discover_all(user_id=user_id, user_token=user_token)

        for source in response.sources:
            if source.source_id == f"assistant:{endpoint_name}":
                return SourceMetadataResponse(
                    source=source,
                    serving_endpoint=ServingEndpointMetadata(**source.metadata),
                )

        return None

    async def refresh(
        self,
        user_id: str,
        user_token: str | None = None,
        source_types: list[DataSourceType] | None = None,
    ) -> DiscoveryResponse:
        """Force refresh discovery cache.

        Args:
            user_id: User ID for cache keying.
            user_token: Optional OBO token for OBO authentication.
            source_types: Optional list of specific types to refresh.

        Returns:
            Fresh DiscoveryResponse.
        """
        # Invalidate cache first
        await self._cache.invalidate(user_id=user_id)

        # Re-discover
        return await self.discover_all(
            user_id=user_id,
            user_token=user_token,
            force_refresh=True,
            source_types=source_types,
        )


# Singleton instance
_discovery_service: DiscoveryService | None = None


def get_discovery_service() -> DiscoveryService:
    """Get the global discovery service instance.

    Returns:
        The singleton DiscoveryService instance.
    """
    global _discovery_service
    if _discovery_service is None:
        _discovery_service = DiscoveryService()
    return _discovery_service


def reset_discovery_service() -> None:
    """Reset the global discovery service (for testing)."""
    global _discovery_service
    _discovery_service = None
