"""Unit tests for VectorSearchTool class."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from deep_research.agent.tools.base import ResearchContext, ResearchTool
from deep_research.agent.tools.vector_search import (
    VectorSearchTool,
    create_vector_search_tools_from_config,
)
from deep_research.services.vector_search_query import (
    ColumnRoles,
    VectorSearchQueryService,
    VectorSearchResult,
    extract_queryable_columns,
)


def create_test_context() -> ResearchContext:
    """Create a test ResearchContext."""
    return ResearchContext(
        chat_id=uuid4(),
        user_id="test-user",
        research_type="medium",
    )


def make_mock_query_results(
    results: list[list[Any]] | None = None,
) -> list[VectorSearchResult]:
    """Create mock VectorSearchResult list from raw rows."""
    if results is None:
        return [
            VectorSearchResult(
                id="doc_0",
                title="Product Guide",
                content="Content about products...",
                url="https://example.com/1",
                score=0.95,
                metadata={},
            ),
            VectorSearchResult(
                id="doc_1",
                title="API Reference",
                content="API documentation...",
                url="https://example.com/2",
                score=0.87,
                metadata={},
            ),
        ]
    return [
        VectorSearchResult(
            id=f"doc_{i}",
            title=row[0] if len(row) > 0 else "Untitled",
            content=row[1] if len(row) > 1 else "",
            url=row[2] if len(row) > 2 else None,
            score=row[3] if len(row) > 3 else 0.0,
            metadata={},
        )
        for i, row in enumerate(results)
    ]


class TestVectorSearchToolDefinition:
    """Tests for VectorSearchTool definition property."""

    def test_implements_research_tool_protocol(self) -> None:
        """VectorSearchTool should implement ResearchTool protocol."""
        tool = VectorSearchTool(
            endpoint_name="test-endpoint",
            index_name="catalog.schema.test_index",
        )
        assert isinstance(tool, ResearchTool)

    def test_definition_has_generated_name(self) -> None:
        """Tool name should be generated from endpoint name."""
        tool = VectorSearchTool(
            endpoint_name="product-docs",
            index_name="catalog.schema.test_index",
        )
        assert tool.definition.name == "search_product_docs"

    def test_definition_with_custom_name(self) -> None:
        """Should use custom tool name when provided."""
        tool = VectorSearchTool(
            endpoint_name="test-endpoint",
            index_name="catalog.schema.test_index",
            tool_name="custom_search",
        )
        assert tool.definition.name == "custom_search"

    def test_definition_has_description(self) -> None:
        """Tool definition should have a description."""
        tool = VectorSearchTool(
            endpoint_name="test-endpoint",
            index_name="catalog.schema.test_index",
        )
        assert tool.definition.description
        assert "catalog.schema.test_index" in tool.definition.description

    def test_definition_with_custom_description(self) -> None:
        """Should use custom description when provided."""
        tool = VectorSearchTool(
            endpoint_name="test-endpoint",
            index_name="catalog.schema.test_index",
            description="Custom search description",
        )
        assert tool.definition.description == "Custom search description"

    def test_definition_has_query_parameter(self) -> None:
        """Tool definition should require 'query' parameter."""
        tool = VectorSearchTool(
            endpoint_name="test-endpoint",
            index_name="catalog.schema.test_index",
        )
        params = tool.definition.parameters
        assert "query" in params["properties"]
        assert params["required"] == ["query"]

    def test_definition_has_num_results_parameter(self) -> None:
        """Tool definition should have optional 'num_results' parameter."""
        tool = VectorSearchTool(
            endpoint_name="test-endpoint",
            index_name="catalog.schema.test_index",
            num_results=10,
        )
        props = tool.definition.parameters["properties"]
        assert "num_results" in props
        assert props["num_results"]["type"] == "integer"
        assert props["num_results"]["default"] == 10


class TestVectorSearchToolValidation:
    """Tests for VectorSearchTool argument validation."""

    def test_valid_query_only(self) -> None:
        """Should accept valid query-only arguments."""
        tool = VectorSearchTool(
            endpoint_name="test-endpoint",
            index_name="catalog.schema.test_index",
        )
        errors = tool.validate_arguments({"query": "test search"})
        assert errors == []

    def test_valid_full_arguments(self) -> None:
        """Should accept all valid arguments."""
        tool = VectorSearchTool(
            endpoint_name="test-endpoint",
            index_name="catalog.schema.test_index",
        )
        errors = tool.validate_arguments({
            "query": "test search",
            "num_results": 10,
        })
        assert errors == []

    def test_missing_query(self) -> None:
        """Should reject missing query."""
        tool = VectorSearchTool(
            endpoint_name="test-endpoint",
            index_name="catalog.schema.test_index",
        )
        errors = tool.validate_arguments({})
        assert len(errors) == 1
        assert "query" in errors[0]

    def test_empty_query(self) -> None:
        """Should reject empty query."""
        tool = VectorSearchTool(
            endpoint_name="test-endpoint",
            index_name="catalog.schema.test_index",
        )
        errors = tool.validate_arguments({"query": ""})
        assert len(errors) == 1
        assert "query" in errors[0]

    def test_query_too_long(self) -> None:
        """Should reject query over 1000 characters."""
        tool = VectorSearchTool(
            endpoint_name="test-endpoint",
            index_name="catalog.schema.test_index",
        )
        errors = tool.validate_arguments({"query": "x" * 1001})
        assert len(errors) == 1
        assert "1000" in errors[0]

    def test_invalid_num_results_type(self) -> None:
        """Should reject non-integer num_results."""
        tool = VectorSearchTool(
            endpoint_name="test-endpoint",
            index_name="catalog.schema.test_index",
        )
        errors = tool.validate_arguments({"query": "test", "num_results": "ten"})
        assert len(errors) == 1
        assert "integer" in errors[0]

    def test_num_results_out_of_range(self) -> None:
        """Should reject num_results outside 1-100 range."""
        tool = VectorSearchTool(
            endpoint_name="test-endpoint",
            index_name="catalog.schema.test_index",
        )

        errors = tool.validate_arguments({"query": "test", "num_results": 0})
        assert len(errors) == 1
        assert "between" in errors[0]

        errors = tool.validate_arguments({"query": "test", "num_results": 101})
        assert len(errors) == 1
        assert "between" in errors[0]


class TestVectorSearchToolExecution:
    """Tests for VectorSearchTool execute method using WorkspaceClient."""

    @pytest.mark.asyncio
    async def test_successful_search(self) -> None:
        """Should execute search via query service and return formatted results."""
        tool = VectorSearchTool(
            endpoint_name="test-endpoint",
            index_name="catalog.schema.test_index",
        )

        mock_results = make_mock_query_results()

        with patch.object(tool._query_service, "query", return_value=mock_results), \
             patch("deep_research.agent.tools.vector_search.get_workspace_client") as mock_get_client:
            mock_get_client.return_value = MagicMock()

            context = create_test_context()
            result = await tool.execute({"query": "test query"}, context)

        assert result.success
        assert "[0]" in result.content
        assert "[1]" in result.content
        assert "Product Guide" in result.content
        assert "API Reference" in result.content
        assert "0.95" in result.content

    @pytest.mark.asyncio
    async def test_search_returns_sources(self) -> None:
        """Should include sources for citation tracking."""
        tool = VectorSearchTool(
            endpoint_name="test-endpoint",
            index_name="catalog.schema.test_index",
        )

        mock_results = make_mock_query_results()

        with patch.object(tool._query_service, "query", return_value=mock_results), \
             patch("deep_research.agent.tools.vector_search.get_workspace_client") as mock_get_client:
            mock_get_client.return_value = MagicMock()

            context = create_test_context()
            result = await tool.execute({"query": "test query"}, context)

        assert result.sources is not None
        assert len(result.sources) == 2
        assert result.sources[0]["type"] == "vector_search"
        assert result.sources[0]["url"] == "https://example.com/1"
        assert result.sources[0]["index_name"] == "catalog.schema.test_index"

    @pytest.mark.asyncio
    async def test_search_returns_data(self) -> None:
        """Should include data with query and counts."""
        tool = VectorSearchTool(
            endpoint_name="test-endpoint",
            index_name="catalog.schema.test_index",
        )

        mock_results = make_mock_query_results()

        with patch.object(tool._query_service, "query", return_value=mock_results), \
             patch("deep_research.agent.tools.vector_search.get_workspace_client") as mock_get_client:
            mock_get_client.return_value = MagicMock()

            context = create_test_context()
            result = await tool.execute({"query": "test query", "num_results": 5}, context)

        assert result.data is not None
        assert result.data["query"] == "test query"
        assert result.data["num_results"] == 2
        assert result.data["index_name"] == "catalog.schema.test_index"

    @pytest.mark.asyncio
    async def test_empty_results(self) -> None:
        """Should handle empty search results."""
        tool = VectorSearchTool(
            endpoint_name="test-endpoint",
            index_name="catalog.schema.test_index",
        )

        with patch.object(tool._query_service, "query", return_value=[]), \
             patch("deep_research.agent.tools.vector_search.get_workspace_client") as mock_get_client:
            mock_get_client.return_value = MagicMock()

            context = create_test_context()
            result = await tool.execute({"query": "no results query"}, context)

        assert result.success
        assert "No results found" in result.content

    @pytest.mark.asyncio
    async def test_search_with_custom_num_results(self) -> None:
        """Should pass num_results to query service."""
        tool = VectorSearchTool(
            endpoint_name="test-endpoint",
            index_name="catalog.schema.test_index",
        )

        with patch.object(tool._query_service, "query", return_value=[]) as mock_query, \
             patch("deep_research.agent.tools.vector_search.get_workspace_client") as mock_get_client:
            mock_get_client.return_value = MagicMock()

            context = create_test_context()
            await tool.execute({"query": "test", "num_results": 15}, context)

        mock_query.assert_called_once()
        call_kwargs = mock_query.call_args.kwargs
        assert call_kwargs["num_results"] == 15

    @pytest.mark.asyncio
    async def test_search_error_handling(self) -> None:
        """Should handle search errors gracefully."""
        tool = VectorSearchTool(
            endpoint_name="test-endpoint",
            index_name="catalog.schema.test_index",
        )

        with patch.object(tool._query_service, "query", side_effect=Exception("Search failed")), \
             patch("deep_research.agent.tools.vector_search.get_workspace_client") as mock_get_client:
            mock_get_client.return_value = MagicMock()

            context = create_test_context()
            result = await tool.execute({"query": "test"}, context)

        assert not result.success
        assert result.error is not None
        assert "Search failed" in result.error

    @pytest.mark.asyncio
    async def test_missing_columns_handled(self) -> None:
        """Should handle results with missing columns."""
        tool = VectorSearchTool(
            endpoint_name="test-endpoint",
            index_name="catalog.schema.test_index",
        )

        mock_results = [
            VectorSearchResult(
                id="doc_0",
                title="Only Title",
                content="",
                url=None,
                score=0.9,
                metadata={},
            ),
        ]

        with patch.object(tool._query_service, "query", return_value=mock_results), \
             patch("deep_research.agent.tools.vector_search.get_workspace_client") as mock_get_client:
            mock_get_client.return_value = MagicMock()

            context = create_test_context()
            result = await tool.execute({"query": "test"}, context)

        assert result.success
        assert "Only Title" in result.content


class TestVectorSearchToolConfiguration:
    """Tests for VectorSearchTool configuration."""

    def test_uses_configured_columns(self) -> None:
        """Should use configured columns for search."""
        tool = VectorSearchTool(
            endpoint_name="test-endpoint",
            index_name="catalog.schema.test_index",
            columns=["title", "body", "link"],
        )
        assert tool._columns == ["title", "body", "link"]

    def test_uses_configured_filters(self) -> None:
        """Should apply configured filters to searches."""
        tool = VectorSearchTool(
            endpoint_name="test-endpoint",
            index_name="catalog.schema.test_index",
            filters={"category": "docs"},
        )
        assert tool._filters == {"category": "docs"}


class TestCreateVectorSearchToolsFromConfig:
    """Tests for create_vector_search_tools_from_config function."""

    def test_returns_empty_when_disabled(self) -> None:
        """Should return empty list when VS is disabled."""
        mock_config = MagicMock()
        mock_config.enabled = False

        tools = create_vector_search_tools_from_config(mock_config)
        assert tools == []

    def test_returns_empty_when_no_config(self) -> None:
        """Should return empty list when config is None."""
        tools = create_vector_search_tools_from_config(None)
        assert tools == []

    def test_creates_tools_from_endpoints(self) -> None:
        """Should create a tool for each enabled endpoint."""
        mock_endpoint1 = MagicMock()
        mock_endpoint1.endpoint_name = "endpoint1"
        mock_endpoint1.index_name = "catalog.schema.index1"
        mock_endpoint1.enabled = True
        mock_endpoint1.columns = ["title", "content"]
        mock_endpoint1.tool_name = None
        mock_endpoint1.description = None
        mock_endpoint1.num_results = 5
        mock_endpoint1.filters = None

        mock_endpoint2 = MagicMock()
        mock_endpoint2.endpoint_name = "endpoint2"
        mock_endpoint2.index_name = "catalog.schema.index2"
        mock_endpoint2.enabled = True
        mock_endpoint2.columns = None
        mock_endpoint2.tool_name = "custom_search"
        mock_endpoint2.description = "Custom description"
        mock_endpoint2.num_results = 10
        mock_endpoint2.filters = None

        mock_config = MagicMock()
        mock_config.enabled = True
        mock_config.endpoints = {
            "ep1": mock_endpoint1,
            "ep2": mock_endpoint2,
        }

        tools = create_vector_search_tools_from_config(mock_config)

        assert len(tools) == 2
        assert tools[0].definition.name == "search_endpoint1"
        assert tools[1].definition.name == "custom_search"

    def test_skips_disabled_endpoints(self) -> None:
        """Should skip disabled endpoints."""
        mock_endpoint = MagicMock()
        mock_endpoint.enabled = False

        mock_config = MagicMock()
        mock_config.enabled = True
        mock_config.endpoints = {"ep1": mock_endpoint}

        tools = create_vector_search_tools_from_config(mock_config)
        assert tools == []

    def test_handles_endpoint_creation_error(self) -> None:
        """Should continue after endpoint creation error."""
        # Create an endpoint that will raise an AttributeError when accessed
        mock_endpoint1 = MagicMock()
        # Force an error by making endpoint_name property raise
        type(mock_endpoint1).endpoint_name = property(
            lambda self: (_ for _ in ()).throw(ValueError("Bad endpoint"))
        )
        mock_endpoint1.enabled = True

        mock_endpoint2 = MagicMock()
        mock_endpoint2.endpoint_name = "good-endpoint"
        mock_endpoint2.index_name = "catalog.schema.good_index"
        mock_endpoint2.enabled = True
        mock_endpoint2.columns = None
        mock_endpoint2.tool_name = None
        mock_endpoint2.description = None
        mock_endpoint2.num_results = 5
        mock_endpoint2.filters = None

        mock_config = MagicMock()
        mock_config.enabled = True
        mock_config.endpoints = {
            "bad": mock_endpoint1,
            "good": mock_endpoint2,
        }

        tools = create_vector_search_tools_from_config(mock_config)
        assert len(tools) == 1
        assert tools[0].definition.name == "search_good_endpoint"


class TestVectorSearchQueryService:
    """Tests for the VectorSearchQueryService."""

    def test_parse_response_sdk_objects(self) -> None:
        """Should parse SDK-style response with attribute access."""
        from deep_research.services.vector_search_query import VectorSearchQueryService

        # Create mock SDK response with attribute access
        col1 = MagicMock()
        col1.name = "title"
        col2 = MagicMock()
        col2.name = "content"
        col3 = MagicMock()
        col3.name = "url"
        col4 = MagicMock()
        col4.name = "score"

        response = MagicMock()
        response.manifest.columns = [col1, col2, col3, col4]
        response.result.data_array = [
            ["Product Guide", "Content...", "https://example.com/1", 0.95],
            ["API Docs", "API info...", "https://example.com/2", 0.87],
        ]

        service = VectorSearchQueryService()
        results = service._parse_response(response, ["title", "content", "url"])

        assert len(results) == 2
        assert results[0].title == "Product Guide"
        assert results[0].url == "https://example.com/1"
        assert results[0].score == 0.95
        assert results[1].title == "API Docs"

    def test_parse_response_dict_format(self) -> None:
        """Should parse dict-style response (legacy/testing)."""
        from deep_research.services.vector_search_query import VectorSearchQueryService

        response = {
            "manifest": {
                "columns": [
                    {"name": "title"},
                    {"name": "content"},
                    {"name": "score"},
                ],
            },
            "result": {
                "data_array": [
                    ["Title 1", "Content 1", 0.9],
                ],
            },
        }

        service = VectorSearchQueryService()
        results = service._parse_response(response, ["title", "content"])

        assert len(results) == 1
        assert results[0].title == "Title 1"
        assert results[0].score == 0.9

    def test_parse_response_empty(self) -> None:
        """Should return empty list for empty data_array."""
        from deep_research.services.vector_search_query import VectorSearchQueryService

        response = MagicMock()
        response.manifest.columns = []
        response.result.data_array = []

        service = VectorSearchQueryService()
        results = service._parse_response(response, [])

        assert results == []

    def test_build_filters_json_from_dict(self) -> None:
        """Should convert dict filters to JSON string."""
        from deep_research.services.vector_search_query import VectorSearchQueryService

        result = VectorSearchQueryService.build_filters_json(
            filters_dict={"category": "docs", "year >": 2023}
        )
        assert result is not None
        import json
        parsed = json.loads(result)
        assert parsed["category"] == "docs"
        assert parsed["year >"] == 2023

    def test_build_filters_json_from_sql(self) -> None:
        """Should pass through SQL filter string."""
        from deep_research.services.vector_search_query import VectorSearchQueryService

        result = VectorSearchQueryService.build_filters_json(
            filters_sql="category = 'docs' AND year > 2023"
        )
        assert result == "category = 'docs' AND year > 2023"

    def test_build_filters_json_none(self) -> None:
        """Should return None when no filters provided."""
        from deep_research.services.vector_search_query import VectorSearchQueryService

        result = VectorSearchQueryService.build_filters_json()
        assert result is None


class TestExtractQueryableColumns:
    """Tests for extract_queryable_columns utility function."""

    def test_delta_sync_index(self) -> None:
        """Should extract columns from DELTA_SYNC index with embedding_source_columns."""
        # Mock embedding source column
        source_col = MagicMock()
        source_col.name = "text_content"

        # Mock embedding vector column (should be excluded)
        vector_col = MagicMock()
        vector_col.name = "text_content_vector"

        # Mock DELTA_SYNC spec
        spec = MagicMock()
        spec.embedding_source_columns = [source_col]
        spec.embedding_vector_columns = [vector_col]
        spec.schema_json = None

        # Mock VectorIndex
        index = MagicMock()
        index.primary_key = "doc_id"
        index.delta_sync_index_spec = spec
        index.direct_access_index_spec = None

        roles = extract_queryable_columns(index)

        assert roles is not None
        assert roles.id_column == "doc_id"
        assert roles.content_column == "text_content"
        assert "doc_id" in roles.all_columns
        assert "text_content" in roles.all_columns
        assert "text_content_vector" not in roles.all_columns

    def test_direct_access_with_schema_json(self) -> None:
        """Should extract columns from DIRECT_ACCESS index with schema_json."""
        # Mock embedding source column
        source_col = MagicMock()
        source_col.name = "passage"

        # Mock embedding vector column
        vector_col = MagicMock()
        vector_col.name = "embedding"

        spec = MagicMock()
        spec.embedding_source_columns = [source_col]
        spec.embedding_vector_columns = [vector_col]
        spec.schema_json = '{"row_id": "int", "passage": "string", "embedding": "array<float>", "source_url": "string"}'

        index = MagicMock()
        index.primary_key = "row_id"
        index.delta_sync_index_spec = None
        index.direct_access_index_spec = spec

        roles = extract_queryable_columns(index)

        assert roles is not None
        assert roles.id_column == "row_id"
        assert roles.content_column == "passage"
        # schema_json columns minus vector column
        assert "row_id" in roles.all_columns
        assert "passage" in roles.all_columns
        assert "source_url" in roles.all_columns
        assert "embedding" not in roles.all_columns

    def test_no_spec(self) -> None:
        """Should return just primary_key when no spec available."""
        index = MagicMock()
        index.primary_key = "pk_col"
        index.delta_sync_index_spec = None
        index.direct_access_index_spec = None

        roles = extract_queryable_columns(index)

        assert roles is not None
        assert roles.id_column == "pk_col"
        assert roles.content_column is None
        assert roles.all_columns == ["pk_col"]

    def test_no_primary_key(self) -> None:
        """Should return None when no primary_key."""
        index = MagicMock()
        index.primary_key = None

        roles = extract_queryable_columns(index)

        assert roles is None

    def test_no_embedding_source_columns(self) -> None:
        """Should handle index with no embedding_source_columns gracefully."""
        spec = MagicMock()
        spec.embedding_source_columns = None
        spec.embedding_vector_columns = None
        spec.schema_json = None

        index = MagicMock()
        index.primary_key = "id"
        index.delta_sync_index_spec = spec
        index.direct_access_index_spec = None

        roles = extract_queryable_columns(index)

        assert roles is not None
        assert roles.id_column == "id"
        assert roles.content_column is None
        assert roles.all_columns == ["id"]


class TestParseResponseWithColumnRoles:
    """Tests for _parse_response with ColumnRoles (deterministic mapping)."""

    def test_with_column_roles(self) -> None:
        """Should map columns deterministically using ColumnRoles."""
        roles = ColumnRoles(
            id_column="doc_id",
            content_column="text_body",
            all_columns=["doc_id", "text_body", "source_path"],
        )

        # Mock response with non-standard column names
        col1 = MagicMock()
        col1.name = "doc_id"
        col2 = MagicMock()
        col2.name = "text_body"
        col3 = MagicMock()
        col3.name = "source_path"
        col4 = MagicMock()
        col4.name = "score"

        response = MagicMock()
        response.manifest.columns = [col1, col2, col3, col4]
        response.result.data_array = [
            ["abc123", "This is the document content about Python.", "/docs/python.md", 0.92],
        ]

        service = VectorSearchQueryService()
        results = service._parse_response(
            response,
            ["doc_id", "text_body", "source_path"],
            column_roles=roles,
        )

        assert len(results) == 1
        assert results[0].id == "abc123"
        assert results[0].content == "This is the document content about Python."
        assert results[0].score == 0.92
        # source_path is metadata, not a mapped field
        assert results[0].metadata.get("source_path") == "/docs/python.md"
        # Title derived from content since no "title" column
        assert "Python" in results[0].title

    def test_without_column_roles_legacy(self) -> None:
        """Should use legacy hardcoded names when column_roles is None."""
        col1 = MagicMock()
        col1.name = "id"
        col2 = MagicMock()
        col2.name = "title"
        col3 = MagicMock()
        col3.name = "content"
        col4 = MagicMock()
        col4.name = "score"

        response = MagicMock()
        response.manifest.columns = [col1, col2, col3, col4]
        response.result.data_array = [
            ["doc_0", "My Title", "My Content", 0.85],
        ]

        service = VectorSearchQueryService()
        results = service._parse_response(response, ["id", "title", "content"])

        assert len(results) == 1
        assert results[0].id == "doc_0"
        assert results[0].title == "My Title"
        assert results[0].content == "My Content"
        assert results[0].score == 0.85

    def test_column_roles_no_content_column(self) -> None:
        """Should handle ColumnRoles with content_column=None."""
        roles = ColumnRoles(
            id_column="pk",
            content_column=None,
            all_columns=["pk"],
        )

        col1 = MagicMock()
        col1.name = "pk"
        col2 = MagicMock()
        col2.name = "score"

        response = MagicMock()
        response.manifest.columns = [col1, col2]
        response.result.data_array = [
            ["row_1", 0.77],
        ]

        service = VectorSearchQueryService()
        results = service._parse_response(response, ["pk"], column_roles=roles)

        assert len(results) == 1
        assert results[0].id == "row_1"
        assert results[0].content == ""
        assert results[0].title == "Untitled"

    def test_title_derived_from_content(self) -> None:
        """Should derive title from content when no title column exists."""
        roles = ColumnRoles(
            id_column="id",
            content_column="body",
            all_columns=["id", "body"],
        )

        col1 = MagicMock()
        col1.name = "id"
        col2 = MagicMock()
        col2.name = "body"
        col3 = MagicMock()
        col3.name = "score"

        response = MagicMock()
        response.manifest.columns = [col1, col2, col3]
        long_content = "A" * 100
        response.result.data_array = [
            ["d1", long_content, 0.9],
        ]

        service = VectorSearchQueryService()
        results = service._parse_response(response, ["id", "body"], column_roles=roles)

        assert len(results) == 1
        # Title should be first 80 chars + "..."
        assert results[0].title.endswith("...")
        assert len(results[0].title) <= 84  # 80 + "..."


class TestUserVectorSearchToolColumnDiscovery:
    """Tests for UserVectorSearchTool lazy column discovery."""

    @pytest.mark.asyncio
    async def test_discovers_columns_on_first_execute(self) -> None:
        """Should call get_index() and populate columns on first execute."""
        from deep_research.agent.tools.user_vector_search import UserVectorSearchTool

        # Mock OBO client
        mock_obo_client = MagicMock()

        # Mock the index returned by get_index()
        source_col = MagicMock()
        source_col.name = "chunk_text"
        vector_col = MagicMock()
        vector_col.name = "chunk_embedding"

        mock_spec = MagicMock()
        mock_spec.embedding_source_columns = [source_col]
        mock_spec.embedding_vector_columns = [vector_col]
        mock_spec.schema_json = None

        mock_index = MagicMock()
        mock_index.primary_key = "chunk_id"
        mock_index.delta_sync_index_spec = mock_spec
        mock_index.direct_access_index_spec = None

        # Mock WorkspaceClient
        mock_client = MagicMock()
        mock_client.vector_search_indexes.get_index.return_value = mock_index

        # Mock query service to return results
        mock_results = [
            VectorSearchResult(
                id="chunk_1",
                title="Test",
                content="Test content",
                url=None,
                score=0.9,
                metadata={},
            )
        ]

        mock_obo_client.get_client = AsyncMock(return_value=mock_client)

        tool = UserVectorSearchTool(
            obo_client=mock_obo_client,
            source_name="test_source",
            endpoint_name="test_ep",
            index_name="catalog.schema.test_index",
            # columns=None -> empty list -> triggers discovery
        )

        # Verify columns start empty
        assert tool._columns == []
        assert tool._column_roles is None

        context = create_test_context()

        with patch.object(tool._query_service, "query", return_value=mock_results):
            result = await tool.execute({"query": "test query"}, context)

        assert result.success
        # Columns should be populated after discovery
        assert "chunk_id" in tool._columns
        assert "chunk_text" in tool._columns
        assert "chunk_embedding" not in tool._columns
        assert tool._column_roles is not None
        assert tool._column_roles.id_column == "chunk_id"
        assert tool._column_roles.content_column == "chunk_text"

        # get_index was called once
        mock_client.vector_search_indexes.get_index.assert_called_once_with(
            "catalog.schema.test_index"
        )

    @pytest.mark.asyncio
    async def test_skips_discovery_when_columns_provided(self) -> None:
        """Should NOT call get_index() when columns are pre-provided."""
        from deep_research.agent.tools.user_vector_search import UserVectorSearchTool

        mock_obo_client = MagicMock()
        mock_client = MagicMock()
        mock_obo_client.get_client = AsyncMock(return_value=mock_client)

        mock_results = [
            VectorSearchResult(
                id="doc_0",
                title="Test",
                content="Test content",
                url=None,
                score=0.9,
                metadata={},
            )
        ]

        tool = UserVectorSearchTool(
            obo_client=mock_obo_client,
            source_name="test_source",
            endpoint_name="test_ep",
            index_name="catalog.schema.test_index",
            columns=["doc_id", "text_content", "source_url"],
        )

        context = create_test_context()

        with patch.object(tool._query_service, "query", return_value=mock_results):
            result = await tool.execute({"query": "test query"}, context)

        assert result.success
        # get_index should NOT have been called
        mock_client.vector_search_indexes.get_index.assert_not_called()

    @pytest.mark.asyncio
    async def test_discovery_failure_returns_error(self) -> None:
        """Should return error ToolResult when column discovery fails."""
        from deep_research.agent.tools.user_vector_search import UserVectorSearchTool

        mock_obo_client = MagicMock()
        mock_client = MagicMock()
        mock_client.vector_search_indexes.get_index.side_effect = Exception("PERMISSION_DENIED")
        mock_obo_client.get_client = AsyncMock(return_value=mock_client)

        tool = UserVectorSearchTool(
            obo_client=mock_obo_client,
            source_name="test_source",
            endpoint_name="test_ep",
            index_name="catalog.schema.test_index",
            # No columns -> discovery needed -> will fail
        )

        context = create_test_context()
        result = await tool.execute({"query": "test query"}, context)

        assert not result.success
        assert "unable to determine" in result.content
