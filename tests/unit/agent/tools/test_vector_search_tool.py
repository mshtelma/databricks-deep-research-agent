"""Unit tests for VectorSearchTool class."""

from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from deep_research.agent.tools.base import ResearchContext, ResearchTool
from deep_research.agent.tools.vector_search import (
    VectorSearchTool,
    create_vector_search_tools_from_config,
)


def create_test_context() -> ResearchContext:
    """Create a test ResearchContext."""
    return ResearchContext(
        chat_id=uuid4(),
        user_id="test-user",
        research_type="medium",
    )


def create_mock_index(results: list[list[Any]] | None = None) -> MagicMock:
    """Create a mock Vector Search index."""
    mock_index = MagicMock()

    if results is None:
        # Default test results
        results = [
            ["Product Guide", "Content about products...", "https://example.com/1", 0.95],
            ["API Reference", "API documentation...", "https://example.com/2", 0.87],
        ]

    mock_response = {
        "manifest": {
            "column_count": 4,
            "columns": [
                {"name": "title", "type": "string"},
                {"name": "content", "type": "string"},
                {"name": "url", "type": "string"},
                {"name": "score", "type": "double"},
            ],
        },
        "result": {
            "row_count": len(results),
            "data_array": results,
        },
    }

    mock_index.similarity_search.return_value = mock_response
    return mock_index


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
    """Tests for VectorSearchTool execute method."""

    @pytest.mark.asyncio
    async def test_successful_search(self) -> None:
        """Should execute search and return formatted results."""
        tool = VectorSearchTool(
            endpoint_name="test-endpoint",
            index_name="catalog.schema.test_index",
        )

        mock_index = create_mock_index()
        tool._index = mock_index  # Inject mock

        context = create_test_context()
        result = await tool.execute({"query": "test query"}, context)

        assert result.success
        assert "[0]" in result.content
        assert "[1]" in result.content
        assert "Product Guide" in result.content
        assert "API Reference" in result.content
        assert "0.95" in result.content  # Score displayed

    @pytest.mark.asyncio
    async def test_search_returns_sources(self) -> None:
        """Should include sources for citation tracking."""
        tool = VectorSearchTool(
            endpoint_name="test-endpoint",
            index_name="catalog.schema.test_index",
        )

        mock_index = create_mock_index()
        tool._index = mock_index

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

        mock_index = create_mock_index()
        tool._index = mock_index

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

        mock_index = create_mock_index(results=[])
        tool._index = mock_index

        context = create_test_context()
        result = await tool.execute({"query": "no results query"}, context)

        assert result.success
        assert "No results found" in result.content

    @pytest.mark.asyncio
    async def test_search_with_custom_num_results(self) -> None:
        """Should pass num_results to search."""
        tool = VectorSearchTool(
            endpoint_name="test-endpoint",
            index_name="catalog.schema.test_index",
        )

        mock_index = create_mock_index()
        tool._index = mock_index

        context = create_test_context()
        await tool.execute({"query": "test", "num_results": 15}, context)

        mock_index.similarity_search.assert_called_once()
        call_kwargs = mock_index.similarity_search.call_args.kwargs
        assert call_kwargs["num_results"] == 15

    @pytest.mark.asyncio
    async def test_search_error_handling(self) -> None:
        """Should handle search errors gracefully."""
        tool = VectorSearchTool(
            endpoint_name="test-endpoint",
            index_name="catalog.schema.test_index",
        )

        mock_index = create_mock_index()
        mock_index.similarity_search.side_effect = Exception("Search failed")
        tool._index = mock_index

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

        # Results with only title and score, missing content and url
        mock_response = {
            "manifest": {
                "columns": [
                    {"name": "title", "type": "string"},
                    {"name": "score", "type": "double"},
                ],
            },
            "result": {
                "row_count": 1,
                "data_array": [["Only Title", 0.9]],
            },
        }
        mock_index = MagicMock()
        mock_index.similarity_search.return_value = mock_response
        tool._index = mock_index

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
