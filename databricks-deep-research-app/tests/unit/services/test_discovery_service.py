"""Unit tests for DiscoveryService.

Tests for:
- Mock WorkspaceClient responses (T010o)
- Parallel discovery (T010o)
- Cache TTL and invalidation (T010o)
- Error handling for partial failures (T010o)
- Graceful degradation with per-task timeouts
- Simplified discovery (no get_space, no get_index)
"""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_research.schemas.data_source import DataSourceType
from deep_research.schemas.discovery import (
    DiscoveredSource,
    DiscoveryStatus,
)
from deep_research.services.discovery_cache import DiscoveryCache
from deep_research.services.discovery_service import (
    ASSISTANT_NAME_PATTERNS,
    DISCOVERY_TIMEOUT_GENIE,
    DISCOVERY_TIMEOUT_SERVING,
    DISCOVERY_TIMEOUT_VS,
    GENIE_MAX_PAGES,
    GENIE_PAGE_SIZE,
    DiscoveryService,
    get_discovery_service,
    reset_discovery_service,
)


@pytest.fixture
def mock_cache() -> MagicMock:
    """Create a mock cache for testing."""
    cache = MagicMock(spec=DiscoveryCache)
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock()
    cache.invalidate = AsyncMock(return_value=1)
    return cache


@pytest.fixture
def discovery_service(mock_cache: MagicMock) -> DiscoveryService:
    """Create a discovery service with mocked cache."""
    return DiscoveryService(cache=mock_cache)


@pytest.fixture
def mock_vs_endpoint() -> MagicMock:
    """Create a mock Vector Search endpoint."""
    endpoint = MagicMock()
    endpoint.name = "test-endpoint"
    endpoint.endpoint_type = "STANDARD"
    return endpoint


@pytest.fixture
def mock_vs_index() -> MagicMock:
    """Create a mock Vector Search index."""
    index = MagicMock()
    index.name = "catalog.schema.test_index"
    index.primary_key = "id"
    index.index_type = MagicMock(value="DELTA_SYNC")

    # Mock status
    status = MagicMock()
    status.ready = True
    status.num_of_source_rows = 10000
    index.status = status

    # Mock spec with properly configured columns
    # Use spec_set or configure_mock to properly set 'name' attribute
    embedding_col = MagicMock()
    embedding_col.name = "content"
    embedding_col.embedding_model_endpoint_name = "databricks-gte-large"

    vector_col = MagicMock()
    vector_col.embedding_dimension = 1024

    # Create column objects with name as a real string attribute
    col_id = MagicMock()
    col_id.name = "id"
    col_id.data_type = "string"

    col_content = MagicMock()
    col_content.name = "content"
    col_content.data_type = "string"

    col_timestamp = MagicMock()
    col_timestamp.name = "timestamp"
    col_timestamp.data_type = "timestamp"

    spec = MagicMock()
    spec.embedding_source_columns = [embedding_col]
    spec.embedding_vector_columns = [vector_col]
    spec.columns_to_sync = [col_id, col_content, col_timestamp]
    index.delta_sync_index_spec = spec
    index.direct_access_index_spec = None

    return index


@pytest.fixture
def mock_genie_space() -> MagicMock:
    """Create a mock Genie space."""
    space = MagicMock()
    space.id = "space123"
    space.space_id = "space123"
    space.title = "Test Genie Space"
    space.description = "A test Genie space for SQL queries"
    space.warehouse_id = "warehouse123"
    space.creator = "test@example.com"
    space.created_at = "2024-01-01T00:00:00Z"
    return space


@pytest.fixture
def mock_serving_endpoint() -> MagicMock:
    """Create a mock serving endpoint (Knowledge Assistant)."""
    endpoint = MagicMock()
    endpoint.name = "my-knowledge-assistant"
    endpoint.endpoint_type = "CUSTOM"
    endpoint.state = MagicMock(value="READY")
    endpoint.tags = {"type": "knowledge_assistant", "department": "engineering"}
    endpoint.creator = "admin@example.com"
    return endpoint


def _make_enrichable_index(
    name: str = "catalog.schema.test_index",
    primary_key: str = "id",
    content_column: str = "content",
) -> MagicMock:
    """Create a mock VectorIndex that passes enrichment (has content_column)."""
    index = MagicMock()
    index.name = name
    index.primary_key = primary_key
    index.direct_access_index_spec = None

    embedding_col = MagicMock()
    embedding_col.name = content_column

    spec = MagicMock()
    spec.embedding_source_columns = [embedding_col]
    spec.embedding_vector_columns = []
    spec.schema_json = None
    spec.columns_to_sync = None
    index.delta_sync_index_spec = spec

    return index


class TestVectorSearchDiscovery:
    """Tests for Vector Search source discovery."""

    @pytest.mark.asyncio
    async def test_discover_vector_search_sources(
        self,
        discovery_service: DiscoveryService,
        mock_vs_endpoint: MagicMock,
        mock_vs_index: MagicMock,
    ) -> None:
        """Test successful Vector Search discovery with column enrichment."""
        # Mock the workspace client
        mock_client = MagicMock()

        # Configure endpoint listing
        mock_client.vector_search_endpoints.list_endpoints.return_value = [mock_vs_endpoint]

        # Configure index listing - use MiniVectorIndex format
        mock_mini_index = MagicMock()
        mock_mini_index.name = mock_vs_index.name
        mock_mini_index.primary_key = "id"
        mock_mini_index.index_type = MagicMock(value="DELTA_SYNC")
        mock_client.vector_search_indexes.list_indexes.return_value = [mock_mini_index]

        # Configure get_index for enrichment (returns full VectorIndex)
        mock_client.vector_search_indexes.get_index.return_value = mock_vs_index

        with patch.object(discovery_service, "_get_client", return_value=mock_client):
            sources, error = await discovery_service.discover_vector_search_sources("test-token")

        assert error is None
        assert len(sources) == 1

        source = sources[0]
        assert source.source_id == f"vs:{mock_vs_index.name}"
        assert source.source_type == DataSourceType.VECTOR_SEARCH
        assert source.status == DiscoveryStatus.READY
        assert "ann" in source.capabilities
        assert source.endpoint_name == mock_vs_endpoint.name

        # get_index() is called during enrichment (bounded, best-effort)
        mock_client.vector_search_indexes.get_index.assert_called_once()
        # Enrichment should populate queryable_columns in metadata
        assert "queryable_columns" in source.metadata
        assert len(source.metadata["queryable_columns"]) > 0

    @pytest.mark.asyncio
    async def test_discover_vector_search_basic(
        self,
        discovery_service: DiscoveryService,
        mock_vs_endpoint: MagicMock,
    ) -> None:
        """Test basic Vector Search discovery returns minimal metadata."""
        mock_client = MagicMock()
        mock_client.vector_search_endpoints.list_endpoints.return_value = [mock_vs_endpoint]

        mock_mini_index = MagicMock()
        mock_mini_index.name = "catalog.schema.test_idx"
        mock_mini_index.primary_key = "id"
        mock_mini_index.index_type = MagicMock(value="DELTA_SYNC")
        mock_client.vector_search_indexes.list_indexes.return_value = [mock_mini_index]

        # Enrichment calls get_index — provide a valid index with content column
        mock_client.vector_search_indexes.get_index.return_value = _make_enrichable_index(
            name="catalog.schema.test_idx",
        )

        with patch.object(discovery_service, "_get_client", return_value=mock_client):
            sources, error = await discovery_service.discover_vector_search_sources("test-token")

        # Should return source with minimal metadata
        assert len(sources) == 1
        assert error is None
        # Status is always READY in simplified discovery (detailed status via metadata)
        assert sources[0].status == DiscoveryStatus.READY

    @pytest.mark.asyncio
    async def test_discover_vector_search_permission_error(
        self,
        discovery_service: DiscoveryService,
    ) -> None:
        """Test handling of permission denied error."""
        mock_client = MagicMock()
        mock_client.vector_search_endpoints.list_endpoints.side_effect = Exception(
            "PERMISSION_DENIED: User not authorized"
        )

        with patch.object(discovery_service, "_get_client", return_value=mock_client):
            sources, error = await discovery_service.discover_vector_search_sources("test-token")

        assert len(sources) == 0
        assert error is not None
        assert error.error_code == "PERMISSION_DENIED"
        assert not error.retryable


class TestGenieDiscovery:
    """Tests for Genie space discovery."""

    @pytest.mark.asyncio
    async def test_discover_genie_spaces(
        self,
        discovery_service: DiscoveryService,
        mock_genie_space: MagicMock,
    ) -> None:
        """Test successful Genie discovery."""
        mock_client = MagicMock()

        # Mock list response
        list_response = MagicMock()
        list_response.spaces = [mock_genie_space]
        list_response.next_page_token = None
        mock_client.genie.list_spaces.return_value = list_response

        # Mock get_space
        mock_client.genie.get_space.return_value = mock_genie_space

        with patch.object(discovery_service, "_get_client", return_value=mock_client):
            sources, error = await discovery_service.discover_genie_spaces("test-token")

        assert error is None
        assert len(sources) == 1

        source = sources[0]
        assert source.source_id == f"genie:{mock_genie_space.space_id}"
        assert source.source_type == DataSourceType.GENIE
        assert source.status == DiscoveryStatus.READY
        assert "sql" in source.capabilities

    @pytest.mark.asyncio
    async def test_discover_genie_api_not_available(
        self,
        discovery_service: DiscoveryService,
    ) -> None:
        """Test handling when Genie API is not available."""
        mock_client = MagicMock()
        mock_client.genie.list_spaces.side_effect = AttributeError("'WorkspaceClient' has no attribute 'genie'")

        with patch.object(discovery_service, "_get_client", return_value=mock_client):
            sources, error = await discovery_service.discover_genie_spaces("test-token")

        assert len(sources) == 0
        assert error is not None
        assert error.error_code == "API_NOT_AVAILABLE"


class TestServingEndpointDiscovery:
    """Tests for serving endpoint (Knowledge Assistant) discovery."""

    @pytest.mark.asyncio
    async def test_discover_serving_endpoints(
        self,
        discovery_service: DiscoveryService,
        mock_serving_endpoint: MagicMock,
    ) -> None:
        """Test successful Knowledge Assistant discovery."""
        mock_client = MagicMock()
        mock_client.serving_endpoints.list.return_value = [mock_serving_endpoint]

        with patch.object(discovery_service, "_get_client", return_value=mock_client):
            sources, error = await discovery_service.discover_serving_endpoints("test-token")

        assert error is None
        assert len(sources) == 1

        source = sources[0]
        assert source.source_id == f"assistant:{mock_serving_endpoint.name}"
        assert source.source_type == DataSourceType.KNOWLEDGE_ASSISTANT
        assert source.status == DiscoveryStatus.READY

    @pytest.mark.asyncio
    async def test_filter_non_assistant_endpoints(
        self,
        discovery_service: DiscoveryService,
    ) -> None:
        """Test that non-assistant endpoints are filtered out."""
        mock_client = MagicMock()

        # Create endpoint without assistant indicators
        non_assistant = MagicMock()
        non_assistant.name = "my-model-endpoint"
        non_assistant.tags = {}
        mock_client.serving_endpoints.list.return_value = [non_assistant]

        with patch.object(discovery_service, "_get_client", return_value=mock_client):
            sources, error = await discovery_service.discover_serving_endpoints("test-token")

        assert len(sources) == 0

    def test_is_knowledge_assistant_by_name(
        self,
        discovery_service: DiscoveryService,
    ) -> None:
        """Test assistant detection by name patterns."""
        for pattern in ASSISTANT_NAME_PATTERNS:
            endpoint = MagicMock()
            endpoint.name = f"my-{pattern}-v1"
            endpoint.tags = {}

            assert discovery_service._is_knowledge_assistant(endpoint)

    def test_is_knowledge_assistant_by_tags(
        self,
        discovery_service: DiscoveryService,
    ) -> None:
        """Test assistant detection by tags."""
        endpoint = MagicMock()
        endpoint.name = "generic-endpoint"
        endpoint.tags = {"type": "knowledge_assistant"}

        assert discovery_service._is_knowledge_assistant(endpoint)


class TestParallelDiscovery:
    """Tests for parallel discovery execution."""

    @pytest.mark.asyncio
    async def test_discover_all_parallel(
        self,
        discovery_service: DiscoveryService,
        mock_vs_endpoint: MagicMock,
        mock_genie_space: MagicMock,
        mock_serving_endpoint: MagicMock,
    ) -> None:
        """Test that discover_all runs all discovery types in parallel."""
        mock_client = MagicMock()

        # Setup VS - use mini index format
        mock_client.vector_search_endpoints.list_endpoints.return_value = [mock_vs_endpoint]
        mock_mini_index = MagicMock()
        mock_mini_index.name = "catalog.schema.test_index"
        mock_mini_index.primary_key = "id"
        mock_mini_index.index_type = MagicMock(value="DELTA_SYNC")
        mock_client.vector_search_indexes.list_indexes.return_value = [mock_mini_index]

        # Enrichment needs a valid index with content column
        mock_client.vector_search_indexes.get_index.return_value = _make_enrichable_index()

        # Setup Genie - only list_spaces needed now
        list_response = MagicMock()
        list_response.spaces = [mock_genie_space]
        list_response.next_page_token = None
        mock_client.genie.list_spaces.return_value = list_response

        # Setup Serving
        mock_client.serving_endpoints.list.return_value = [mock_serving_endpoint]

        with patch.object(discovery_service, "_get_client", return_value=mock_client):
            response = await discovery_service.discover_all(
                user_id="test-user-id",
                user_token="test-token",
                force_refresh=True,
            )

        assert response.total_count == 3
        assert len(response.sources) == 3

        # Verify grouping
        assert "vector_search" in response.by_type
        assert "genie" in response.by_type
        assert "knowledge_assistant" in response.by_type

    @pytest.mark.asyncio
    async def test_discover_all_with_specific_types(
        self,
        discovery_service: DiscoveryService,
        mock_vs_endpoint: MagicMock,
    ) -> None:
        """Test discover_all with filtered source types."""
        mock_client = MagicMock()

        # Setup VS - use mini index format
        mock_client.vector_search_endpoints.list_endpoints.return_value = [mock_vs_endpoint]
        mock_mini_index = MagicMock()
        mock_mini_index.name = "catalog.schema.test_index"
        mock_mini_index.primary_key = "id"
        mock_mini_index.index_type = MagicMock(value="DELTA_SYNC")
        mock_client.vector_search_indexes.list_indexes.return_value = [mock_mini_index]

        # Enrichment needs a valid index with content column
        mock_client.vector_search_indexes.get_index.return_value = _make_enrichable_index()

        with patch.object(discovery_service, "_get_client", return_value=mock_client):
            response = await discovery_service.discover_all(
                user_id="test-user-id",
                user_token="test-token",
                force_refresh=True,
                source_types=[DataSourceType.VECTOR_SEARCH],
            )

        # Only VS should be discovered
        assert response.total_count == 1
        assert "vector_search" in response.by_type


class TestCacheIntegration:
    """Tests for cache integration."""

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_response(
        self,
        mock_cache: MagicMock,
    ) -> None:
        """Test that cache hit returns cached response without re-discovering."""
        cached_sources = [
            DiscoveredSource(
                source_id="vs:cached_index",
                source_type=DataSourceType.VECTOR_SEARCH,
                name="cached_index",
                endpoint_name="endpoint",
                status=DiscoveryStatus.READY,
                capabilities=["ann"],
                metadata={},
                discovered_at=datetime.now(UTC),
            )
        ]
        mock_cache.get = AsyncMock(return_value=cached_sources)

        service = DiscoveryService(cache=mock_cache)

        response = await service.discover_all(user_id="test-user-id", user_token="test-token")

        assert response.cached is True
        assert response.total_count == 1
        mock_cache.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_force_refresh_bypasses_cache(
        self,
        mock_cache: MagicMock,
    ) -> None:
        """Test that force_refresh bypasses cache."""
        service = DiscoveryService(cache=mock_cache)

        mock_client = MagicMock()
        mock_client.vector_search_endpoints.list_endpoints.return_value = []
        mock_client.genie.list_spaces.side_effect = AttributeError()
        mock_client.serving_endpoints.list.return_value = []

        with patch.object(service, "_get_client", return_value=mock_client):
            response = await service.discover_all(
                user_id="test-user-id",
                user_token="test-token",
                force_refresh=True,
            )

        # Cache should not be checked on force refresh
        assert response.cached is False

    @pytest.mark.asyncio
    async def test_refresh_invalidates_cache(
        self,
        mock_cache: MagicMock,
    ) -> None:
        """Test that refresh invalidates cache before re-discovering."""
        service = DiscoveryService(cache=mock_cache)

        mock_client = MagicMock()
        mock_client.vector_search_endpoints.list_endpoints.return_value = []
        mock_client.genie.list_spaces.side_effect = AttributeError()
        mock_client.serving_endpoints.list.return_value = []

        with patch.object(service, "_get_client", return_value=mock_client):
            await service.refresh(user_id="test-user-id", user_token="test-token")

        mock_cache.invalidate.assert_called_once_with(user_id="test-user-id")


class TestErrorHandling:
    """Tests for partial failure handling."""

    @pytest.mark.asyncio
    async def test_partial_failures_dont_block_other_types(
        self,
        discovery_service: DiscoveryService,
        mock_serving_endpoint: MagicMock,
    ) -> None:
        """Test that errors in one type don't prevent other types from being discovered."""
        mock_client = MagicMock()

        # VS fails
        mock_client.vector_search_endpoints.list_endpoints.side_effect = Exception("VS Error")

        # Genie fails
        mock_client.genie.list_spaces.side_effect = Exception("Genie Error")

        # Serving succeeds
        mock_client.serving_endpoints.list.return_value = [mock_serving_endpoint]

        with patch.object(discovery_service, "_get_client", return_value=mock_client):
            response = await discovery_service.discover_all(
                user_id="test-user-id",
                user_token="test-token",
                force_refresh=True,
            )

        # Should have 1 source from serving + 2 errors
        assert response.total_count == 1
        assert len(response.errors) == 2

    @pytest.mark.asyncio
    async def test_error_details_preserved(
        self,
        discovery_service: DiscoveryService,
    ) -> None:
        """Test that error details are preserved in response."""
        mock_client = MagicMock()
        mock_client.vector_search_endpoints.list_endpoints.side_effect = Exception(
            "503 Service Unavailable"
        )
        mock_client.genie.list_spaces.side_effect = AttributeError()
        mock_client.serving_endpoints.list.return_value = []

        with patch.object(discovery_service, "_get_client", return_value=mock_client):
            response = await discovery_service.discover_all(
                user_id="test-user-id",
                user_token="test-token",
                force_refresh=True,
            )

        # Find VS error
        vs_error = next((e for e in response.errors if e.source_type == DataSourceType.VECTOR_SEARCH), None)
        assert vs_error is not None
        assert vs_error.error_code == "SERVICE_UNAVAILABLE"
        assert vs_error.retryable is True


class TestMetadataExtraction:
    """Tests for metadata extraction from SDK responses."""

    def test_extract_vector_search_metadata(
        self,
        discovery_service: DiscoveryService,
        mock_vs_index: MagicMock,
    ) -> None:
        """Test Vector Search metadata extraction."""
        metadata = discovery_service._extract_vector_search_metadata(mock_vs_index, "test-endpoint")

        assert metadata.index_name == mock_vs_index.name
        assert metadata.endpoint_name == "test-endpoint"
        assert metadata.is_ready is True
        assert metadata.row_count == 10000
        assert len(metadata.filter_columns) > 0
        assert metadata.embedding_model == "databricks-gte-large"


class TestGlobalServiceInstance:
    """Tests for global service singleton."""

    def test_get_discovery_service_singleton(self) -> None:
        """Test that get_discovery_service returns singleton."""
        reset_discovery_service()

        service1 = get_discovery_service()
        service2 = get_discovery_service()

        assert service1 is service2

        reset_discovery_service()

    def test_reset_discovery_service(self) -> None:
        """Test resetting the global service."""
        reset_discovery_service()

        service1 = get_discovery_service()
        reset_discovery_service()
        service2 = get_discovery_service()

        assert service1 is not service2

        reset_discovery_service()


class TestGracefulDegradation:
    """Tests for graceful degradation with per-task timeouts."""

    @pytest.mark.asyncio
    async def test_discovery_returns_partial_on_genie_timeout(
        self,
        mock_cache: MagicMock,
        mock_serving_endpoint: MagicMock,
    ) -> None:
        """Genie timeout should still return VS and Serving results."""
        service = DiscoveryService(cache=mock_cache)
        mock_client = MagicMock()

        # VS returns quickly with empty results
        mock_client.vector_search_endpoints.list_endpoints.return_value = []

        # Genie times out (simulate slow API)
        async def slow_genie() -> None:
            await asyncio.sleep(100)  # Will timeout

        # We patch at a higher level - make the discover_genie_spaces slow
        async def timeout_genie(token: str | None) -> tuple:
            await asyncio.sleep(100)
            return [], None

        # Serving succeeds
        mock_client.serving_endpoints.list.return_value = [mock_serving_endpoint]

        with patch.object(service, "_get_client", return_value=mock_client), patch.object(service, "discover_genie_spaces", side_effect=timeout_genie):
                response = await service.discover_all(
                    user_id="test-user-id",
                    user_token="test-token",
                    force_refresh=True,
                )

        # Should have 1 source from serving + 1 error from Genie timeout
        assert response.total_count == 1  # Only serving endpoint
        assert len(response.errors) == 1
        assert response.errors[0].source_type == DataSourceType.GENIE
        assert response.errors[0].error_code == "TIMEOUT"
        assert response.errors[0].retryable is True

    @pytest.mark.asyncio
    async def test_per_task_timeouts_are_independent(
        self,
        mock_cache: MagicMock,
    ) -> None:
        """Each discovery type should have independent timeout."""
        _ = DiscoveryService(cache=mock_cache)

        # Verify timeout constants are defined
        assert timedelta(seconds=15) == DISCOVERY_TIMEOUT_VS
        assert timedelta(seconds=10) == DISCOVERY_TIMEOUT_GENIE
        assert timedelta(seconds=10) == DISCOVERY_TIMEOUT_SERVING

    @pytest.mark.asyncio
    async def test_partial_results_are_cached(
        self,
        mock_cache: MagicMock,
        mock_serving_endpoint: MagicMock,
    ) -> None:
        """Partial results should still be cached."""
        service = DiscoveryService(cache=mock_cache)
        mock_client = MagicMock()

        # VS fails
        mock_client.vector_search_endpoints.list_endpoints.side_effect = Exception("VS Error")

        # Genie fails
        mock_client.genie.list_spaces.side_effect = Exception("Genie Error")

        # Serving succeeds
        mock_client.serving_endpoints.list.return_value = [mock_serving_endpoint]

        with patch.object(service, "_get_client", return_value=mock_client):
            _ = await service.discover_all(
                user_id="test-user-id",
                user_token="test-token",
                force_refresh=True,
            )

        # Cache should be called with partial results
        mock_cache.set.assert_called_once()
        call_kwargs = mock_cache.set.call_args.kwargs
        assert len(call_kwargs["sources"]) == 1  # Only serving endpoint

    @pytest.mark.asyncio
    async def test_empty_results_not_cached(
        self,
        mock_cache: MagicMock,
    ) -> None:
        """Empty results should not be cached."""
        service = DiscoveryService(cache=mock_cache)
        mock_client = MagicMock()

        # All fail
        mock_client.vector_search_endpoints.list_endpoints.side_effect = Exception("VS Error")
        mock_client.genie.list_spaces.side_effect = Exception("Genie Error")
        mock_client.serving_endpoints.list.side_effect = Exception("Serving Error")

        with patch.object(service, "_get_client", return_value=mock_client):
            _ = await service.discover_all(
                user_id="test-user-id",
                user_token="test-token",
                force_refresh=True,
            )

        # Cache should not be called for empty results
        mock_cache.set.assert_not_called()


class TestSimplifiedGenieDiscovery:
    """Tests for simplified Genie discovery without get_space() calls."""

    @pytest.mark.asyncio
    async def test_genie_discovery_no_get_space_calls(
        self,
        discovery_service: DiscoveryService,
    ) -> None:
        """Genie discovery should NOT call get_space() for each space."""
        mock_client = MagicMock()

        # Create space summary with all fields we need
        space_summary = MagicMock()
        space_summary.space_id = "space123"
        space_summary.id = "space123"
        space_summary.title = "Test Space"
        space_summary.description = "Test description"
        space_summary.warehouse_id = "warehouse123"

        list_response = MagicMock()
        list_response.spaces = [space_summary]
        list_response.next_page_token = None
        mock_client.genie.list_spaces.return_value = list_response

        with patch.object(discovery_service, "_get_client", return_value=mock_client):
            sources, error = await discovery_service.discover_genie_spaces("test-token")

        # list_spaces() called once with pagination params
        mock_client.genie.list_spaces.assert_called_once_with(
            page_size=GENIE_PAGE_SIZE, page_token=None
        )

        # get_space() should NOT be called
        mock_client.genie.get_space.assert_not_called()

        # Should still return source
        assert len(sources) == 1
        assert error is None
        assert sources[0].source_id == "genie:space123"

    @pytest.mark.asyncio
    async def test_genie_discovery_uses_summary_fields(
        self,
        discovery_service: DiscoveryService,
    ) -> None:
        """Genie discovery should use fields from list_spaces() response."""
        mock_client = MagicMock()

        space_summary = MagicMock()
        space_summary.space_id = "space456"
        space_summary.title = "My Genie Space"
        space_summary.description = "Space for SQL analytics"
        space_summary.warehouse_id = "wh789"

        list_response = MagicMock()
        list_response.spaces = [space_summary]
        list_response.next_page_token = None
        mock_client.genie.list_spaces.return_value = list_response

        with patch.object(discovery_service, "_get_client", return_value=mock_client):
            sources, error = await discovery_service.discover_genie_spaces("test-token")

        source = sources[0]
        assert source.name == "My Genie Space"
        assert source.description == "Space for SQL analytics"

        # Metadata should have available fields, but owner/created_at should be None
        metadata = source.metadata
        assert metadata["space_id"] == "space456"
        assert metadata["warehouse_id"] == "wh789"
        assert metadata["owner"] is None
        assert metadata["created_at"] is None

    @pytest.mark.asyncio
    async def test_genie_pagination_multi_page(
        self,
        discovery_service: DiscoveryService,
    ) -> None:
        """Genie discovery should follow next_page_token across multiple pages."""
        mock_client = MagicMock()

        # Page 1: 2 spaces + next_page_token
        space1 = MagicMock()
        space1.space_id = "space_a"
        space1.title = "Space A"
        space1.description = "First space"
        space1.warehouse_id = "wh1"

        space2 = MagicMock()
        space2.space_id = "space_b"
        space2.title = "Space B"
        space2.description = "Second space"
        space2.warehouse_id = "wh2"

        page1 = MagicMock()
        page1.spaces = [space1, space2]
        page1.next_page_token = "token_page2"

        # Page 2: 1 space + no next_page_token
        space3 = MagicMock()
        space3.space_id = "space_c"
        space3.title = "Space C"
        space3.description = "Third space"
        space3.warehouse_id = "wh3"

        page2 = MagicMock()
        page2.spaces = [space3]
        page2.next_page_token = None

        mock_client.genie.list_spaces.side_effect = [page1, page2]

        with patch.object(discovery_service, "_get_client", return_value=mock_client):
            sources, error = await discovery_service.discover_genie_spaces("test-token")

        assert error is None
        assert len(sources) == 3

        source_ids = {s.source_id for s in sources}
        assert source_ids == {"genie:space_a", "genie:space_b", "genie:space_c"}

        # Verify pagination calls
        assert mock_client.genie.list_spaces.call_count == 2
        mock_client.genie.list_spaces.assert_any_call(
            page_size=GENIE_PAGE_SIZE, page_token=None
        )
        mock_client.genie.list_spaces.assert_any_call(
            page_size=GENIE_PAGE_SIZE, page_token="token_page2"
        )

    @pytest.mark.asyncio
    async def test_genie_pagination_max_pages_safety(
        self,
        discovery_service: DiscoveryService,
    ) -> None:
        """Genie pagination should stop at GENIE_MAX_PAGES even if next_page_token persists."""
        mock_client = MagicMock()

        def make_page(*_args: object, **_kwargs: object) -> MagicMock:
            space = MagicMock()
            space.space_id = f"space_{mock_client.genie.list_spaces.call_count}"
            space.title = f"Space {mock_client.genie.list_spaces.call_count}"
            space.description = None
            space.warehouse_id = "wh1"

            page = MagicMock()
            page.spaces = [space]
            page.next_page_token = "more"  # Always returns more
            return page

        mock_client.genie.list_spaces.side_effect = make_page

        with patch.object(discovery_service, "_get_client", return_value=mock_client):
            sources, error = await discovery_service.discover_genie_spaces("test-token")

        assert error is None
        assert mock_client.genie.list_spaces.call_count == GENIE_MAX_PAGES
        assert len(sources) == GENIE_MAX_PAGES

    @pytest.mark.asyncio
    async def test_genie_pagination_empty_first_page(
        self,
        discovery_service: DiscoveryService,
    ) -> None:
        """Genie discovery should handle empty first page gracefully."""
        mock_client = MagicMock()

        page = MagicMock()
        page.spaces = []
        page.next_page_token = None
        mock_client.genie.list_spaces.return_value = page

        with patch.object(discovery_service, "_get_client", return_value=mock_client):
            sources, error = await discovery_service.discover_genie_spaces("test-token")

        assert error is None
        assert len(sources) == 0


class TestSimplifiedVectorSearchDiscovery:
    """Tests for simplified Vector Search discovery with parallel listing."""

    @pytest.mark.asyncio
    async def test_vs_discovery_enriches_all(
        self,
        discovery_service: DiscoveryService,
        mock_vs_endpoint: MagicMock,
    ) -> None:
        """VS discovery should call get_index() for every discovered source."""
        mock_client = MagicMock()

        # Setup endpoints
        mock_client.vector_search_endpoints.list_endpoints.return_value = [mock_vs_endpoint]

        # Setup multiple mini indexes to verify all are enriched
        mini_indexes = []
        for i in range(15):
            idx = MagicMock()
            idx.name = f"catalog.schema.test_index_{i}"
            idx.primary_key = "id"
            idx.index_type = MagicMock(value="DELTA_SYNC")
            mini_indexes.append(idx)
        mock_client.vector_search_indexes.list_indexes.return_value = mini_indexes

        # Enrichment needs a valid index with content column
        mock_client.vector_search_indexes.get_index.side_effect = lambda name: _make_enrichable_index(name=name)

        with patch.object(discovery_service, "_get_client", return_value=mock_client):
            sources, error = await discovery_service.discover_vector_search_sources("test-token")

        # list_endpoints() called once
        mock_client.vector_search_endpoints.list_endpoints.assert_called_once()

        # list_indexes() called once per endpoint
        mock_client.vector_search_indexes.list_indexes.assert_called_once()

        # get_index() called for EVERY source, not just first N
        assert mock_client.vector_search_indexes.get_index.call_count == 15

        # All sources should be returned
        assert len(sources) == 15
        assert error is None

    @pytest.mark.asyncio
    async def test_vs_discovery_parallel_index_listing(
        self,
        discovery_service: DiscoveryService,
    ) -> None:
        """VS discovery should list indexes in parallel for all endpoints."""
        mock_client = MagicMock()

        # Create 3 endpoints
        endpoints = []
        for i in range(3):
            ep = MagicMock()
            ep.name = f"endpoint-{i}"
            endpoints.append(ep)

        mock_client.vector_search_endpoints.list_endpoints.return_value = endpoints

        # Track call order and timing
        call_times: list[float] = []
        import time

        def make_list_fn(ep_name: str) -> list:
            call_times.append(time.time())
            mini_index = MagicMock()
            mini_index.name = f"catalog.schema.{ep_name}_index"
            mini_index.primary_key = "id"
            mini_index.index_type = MagicMock(value="DELTA_SYNC")
            return [mini_index]

        mock_client.vector_search_indexes.list_indexes.side_effect = make_list_fn

        # Enrichment needs a valid index with content column for each discovered index
        mock_client.vector_search_indexes.get_index.side_effect = lambda name: _make_enrichable_index(name=name)

        with patch.object(discovery_service, "_get_client", return_value=mock_client):
            sources, error = await discovery_service.discover_vector_search_sources("test-token")

        # All 3 list_indexes calls should have been made
        assert mock_client.vector_search_indexes.list_indexes.call_count == 3

        # All 3 sources should be discovered
        assert len(sources) == 3
        assert error is None

    @pytest.mark.asyncio
    async def test_vs_discovery_uses_mini_index_fields(
        self,
        discovery_service: DiscoveryService,
        mock_vs_endpoint: MagicMock,
    ) -> None:
        """VS discovery should use fields from MiniVectorIndex."""
        mock_client = MagicMock()
        mock_client.vector_search_endpoints.list_endpoints.return_value = [mock_vs_endpoint]

        mini_index = MagicMock()
        mini_index.name = "catalog.schema.my_index"
        mini_index.primary_key = "doc_id"
        mini_index.index_type = MagicMock(value="DIRECT_ACCESS")
        mock_client.vector_search_indexes.list_indexes.return_value = [mini_index]

        # Enrichment needs a valid index with content column
        mock_client.vector_search_indexes.get_index.return_value = _make_enrichable_index(
            name="catalog.schema.my_index", primary_key="doc_id",
        )

        with patch.object(discovery_service, "_get_client", return_value=mock_client):
            sources, error = await discovery_service.discover_vector_search_sources("test-token")

        source = sources[0]
        assert source.source_id == "vs:catalog.schema.my_index"
        assert source.name == "my_index"  # Display name from last part

        # Metadata should have basic fields
        metadata = source.metadata
        assert metadata["index_name"] == "catalog.schema.my_index"
        assert metadata["primary_key"] == "doc_id"
        assert metadata["index_type"] == "DIRECT_ACCESS"
        # Detailed fields should be empty lists (populated on-demand)
        assert metadata["embedding_columns"] == []
        assert metadata["embedding_dimension"] is None


class TestOnDemandMetadataFetching:
    """Tests for on-demand metadata fetching."""

    @pytest.mark.asyncio
    async def test_get_genie_metadata_fetches_on_demand(
        self,
        discovery_service: DiscoveryService,
    ) -> None:
        """get_source_metadata for Genie should call get_space() on-demand."""
        mock_client = MagicMock()

        space = MagicMock()
        space.title = "Full Title"
        space.description = "Full description"
        space.warehouse_id = "wh123"
        space.creator = "owner@example.com"
        space.created_at = "2024-01-01T00:00:00Z"
        mock_client.genie.get_space.return_value = space

        with patch.object(discovery_service, "_get_client", return_value=mock_client):
            result = await discovery_service.get_source_metadata(
                user_id="test-user",
                user_token="test-token",
                source_id="genie:space123",
            )

        # get_space() should be called
        mock_client.genie.get_space.assert_called_once_with("space123")

        assert result is not None
        assert result.source.source_id == "genie:space123"
        assert result.source.name == "Full Title"
        assert result.source.capabilities == ["sql", "conversation", "follow_up"]
        assert result.genie is not None
        assert result.genie.space_id == "space123"
        assert result.genie.title == "Full Title"
        assert result.genie.warehouse_id == "wh123"
        assert result.genie.owner == "owner@example.com"
        # created_at gets parsed by Pydantic, so just verify it's not None
        assert result.genie.created_at is not None

    @pytest.mark.asyncio
    async def test_get_vs_metadata_fetches_on_demand(
        self,
        discovery_service: DiscoveryService,
        mock_vs_index: MagicMock,
    ) -> None:
        """get_source_metadata for VS should call get_index() on-demand."""
        mock_client = MagicMock()
        mock_vs_index.endpoint_name = "test-endpoint"
        mock_client.vector_search_indexes.get_index.return_value = mock_vs_index

        with patch.object(discovery_service, "_get_client", return_value=mock_client):
            result = await discovery_service.get_source_metadata(
                user_id="test-user",
                user_token="test-token",
                source_id="vs:catalog.schema.test_index",
            )

        # get_index() should be called
        mock_client.vector_search_indexes.get_index.assert_called_once_with(
            "catalog.schema.test_index"
        )

        assert result is not None
        assert result.source.source_id == "vs:catalog.schema.test_index"
        assert result.source.name == "test_index"
        assert result.vector_search is not None
        assert result.vector_search.is_ready is True
        assert result.vector_search.index_name == "catalog.schema.test_index"
        assert result.vector_search.endpoint_name == "test-endpoint"
        assert "ann" in result.source.capabilities

    @pytest.mark.asyncio
    async def test_get_serving_metadata_uses_cache(
        self,
        mock_cache: MagicMock,
        mock_serving_endpoint: MagicMock,
    ) -> None:
        """get_source_metadata for serving should use cached discovery."""
        service = DiscoveryService(cache=mock_cache)
        mock_client = MagicMock()

        # Setup - make serving endpoints return data
        mock_client.vector_search_endpoints.list_endpoints.return_value = []
        mock_client.genie.list_spaces.side_effect = AttributeError()
        mock_client.serving_endpoints.list.return_value = [mock_serving_endpoint]

        with patch.object(service, "_get_client", return_value=mock_client):
            result = await service.get_source_metadata(
                user_id="test-user",
                user_token="test-token",
                source_id=f"assistant:{mock_serving_endpoint.name}",
            )

        assert result is not None
        assert result.serving_endpoint is not None

    @pytest.mark.asyncio
    async def test_get_metadata_returns_none_for_unknown_source(
        self,
        discovery_service: DiscoveryService,
    ) -> None:
        """get_source_metadata should return None for unknown source types."""
        result = await discovery_service.get_source_metadata(
            user_id="test-user",
            user_token="test-token",
            source_id="unknown:something",
        )

        assert result is None
