"""Unit tests for WebSearchTool class."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from deep_research.agent.tools.base import ResearchContext, ResearchTool, ToolDefinition
from deep_research.agent.tools.web_search import (
    WebSearchOutput,
    WebSearchResult,
    WebSearchTool,
)


def create_test_context() -> ResearchContext:
    """Create a test ResearchContext."""
    return ResearchContext(
        chat_id=uuid4(),
        user_id="test-user",
        research_type="medium",
    )


def create_mock_client(results: list[WebSearchResult] | None = None) -> MagicMock:
    """Create a mock BraveSearchClient."""
    mock_client = MagicMock()
    mock_response = MagicMock()

    if results is None:
        results = [
            WebSearchResult(
                url="https://example.com/1",
                title="Example Page 1",
                snippet="This is the first example result.",
                relevance_score=0.9,
            ),
            WebSearchResult(
                url="https://example.com/2",
                title="Example Page 2",
                snippet="This is the second example result.",
                relevance_score=0.8,
            ),
        ]

    # Convert to the format BraveSearchClient returns
    mock_response.results = [
        MagicMock(
            url=r.url,
            title=r.title,
            snippet=r.snippet,
            relevance_score=r.relevance_score,
        )
        for r in results
    ]

    mock_client.search = AsyncMock(return_value=mock_response)
    return mock_client


class TestWebSearchToolDefinition:
    """Tests for WebSearchTool definition property."""

    def test_implements_research_tool_protocol(self) -> None:
        """WebSearchTool should implement ResearchTool protocol."""
        mock_client = create_mock_client()
        tool = WebSearchTool(mock_client)
        assert isinstance(tool, ResearchTool)

    def test_definition_has_correct_name(self) -> None:
        """Tool definition should have name 'web_search'."""
        mock_client = create_mock_client()
        tool = WebSearchTool(mock_client)
        assert tool.definition.name == "web_search"

    def test_definition_has_description(self) -> None:
        """Tool definition should have a description."""
        mock_client = create_mock_client()
        tool = WebSearchTool(mock_client)
        assert tool.definition.description
        assert "search" in tool.definition.description.lower()

    def test_definition_has_required_query_parameter(self) -> None:
        """Tool definition should require 'query' parameter."""
        mock_client = create_mock_client()
        tool = WebSearchTool(mock_client)
        params = tool.definition.parameters
        assert params["required"] == ["query"]

    def test_definition_has_optional_count_parameter(self) -> None:
        """Tool definition should have optional 'count' parameter."""
        mock_client = create_mock_client()
        tool = WebSearchTool(mock_client)
        props = tool.definition.parameters["properties"]
        assert "count" in props
        assert props["count"]["type"] == "integer"

    def test_definition_has_optional_freshness_parameter(self) -> None:
        """Tool definition should have optional 'freshness' parameter."""
        mock_client = create_mock_client()
        tool = WebSearchTool(mock_client)
        props = tool.definition.parameters["properties"]
        assert "freshness" in props
        assert "enum" in props["freshness"]


class TestWebSearchToolValidation:
    """Tests for WebSearchTool argument validation."""

    def test_valid_query_only(self) -> None:
        """Should accept valid query-only arguments."""
        mock_client = create_mock_client()
        tool = WebSearchTool(mock_client)
        errors = tool.validate_arguments({"query": "test search"})
        assert errors == []

    def test_valid_full_arguments(self) -> None:
        """Should accept all valid arguments."""
        mock_client = create_mock_client()
        tool = WebSearchTool(mock_client)
        errors = tool.validate_arguments({
            "query": "test search",
            "count": 10,
            "freshness": "pw",
        })
        assert errors == []

    def test_missing_query(self) -> None:
        """Should reject missing query."""
        mock_client = create_mock_client()
        tool = WebSearchTool(mock_client)
        errors = tool.validate_arguments({})
        assert len(errors) == 1
        assert "query" in errors[0]

    def test_empty_query(self) -> None:
        """Should reject empty query."""
        mock_client = create_mock_client()
        tool = WebSearchTool(mock_client)
        errors = tool.validate_arguments({"query": ""})
        assert len(errors) == 1
        assert "query" in errors[0]

    def test_non_string_query(self) -> None:
        """Should reject non-string query."""
        mock_client = create_mock_client()
        tool = WebSearchTool(mock_client)
        errors = tool.validate_arguments({"query": 123})
        assert len(errors) == 1
        assert "string" in errors[0]

    def test_query_too_long(self) -> None:
        """Should reject query over 500 characters."""
        mock_client = create_mock_client()
        tool = WebSearchTool(mock_client)
        errors = tool.validate_arguments({"query": "x" * 501})
        assert len(errors) == 1
        assert "500" in errors[0]

    def test_invalid_count_type(self) -> None:
        """Should reject non-integer count."""
        mock_client = create_mock_client()
        tool = WebSearchTool(mock_client)
        errors = tool.validate_arguments({"query": "test", "count": "five"})
        assert len(errors) == 1
        assert "integer" in errors[0]

    def test_count_out_of_range(self) -> None:
        """Should reject count outside 1-20 range."""
        mock_client = create_mock_client()
        tool = WebSearchTool(mock_client)

        errors = tool.validate_arguments({"query": "test", "count": 0})
        assert len(errors) == 1
        assert "between" in errors[0]

        errors = tool.validate_arguments({"query": "test", "count": 21})
        assert len(errors) == 1
        assert "between" in errors[0]

    def test_invalid_freshness(self) -> None:
        """Should reject invalid freshness value."""
        mock_client = create_mock_client()
        tool = WebSearchTool(mock_client)
        errors = tool.validate_arguments({"query": "test", "freshness": "invalid"})
        assert len(errors) == 1
        assert "freshness" in errors[0]


class TestWebSearchToolExecution:
    """Tests for WebSearchTool execute method."""

    @pytest.mark.asyncio
    async def test_successful_search(self) -> None:
        """Should execute search and return formatted results."""
        mock_client = create_mock_client()
        tool = WebSearchTool(mock_client)
        context = create_test_context()

        result = await tool.execute({"query": "test search"}, context)

        assert result.success
        assert "[0]" in result.content
        assert "[1]" in result.content
        assert "Example Page 1" in result.content
        assert "Example Page 2" in result.content

    @pytest.mark.asyncio
    async def test_search_returns_sources(self) -> None:
        """Should include sources for citation tracking."""
        mock_client = create_mock_client()
        tool = WebSearchTool(mock_client)
        context = create_test_context()

        result = await tool.execute({"query": "test search"}, context)

        assert result.sources is not None
        assert len(result.sources) == 2
        assert result.sources[0]["type"] == "web"
        assert result.sources[0]["url"] == "https://example.com/1"
        assert result.sources[0]["title"] == "Example Page 1"

    @pytest.mark.asyncio
    async def test_search_returns_data(self) -> None:
        """Should include data with query and counts."""
        mock_client = create_mock_client()
        tool = WebSearchTool(mock_client)
        context = create_test_context()

        result = await tool.execute({"query": "test search", "count": 10}, context)

        assert result.data is not None
        assert result.data["query"] == "test search"
        assert result.data["count"] == 10
        assert result.data["total_results"] == 2

    @pytest.mark.asyncio
    async def test_empty_results(self) -> None:
        """Should handle empty search results."""
        mock_client = create_mock_client(results=[])
        tool = WebSearchTool(mock_client)
        context = create_test_context()

        result = await tool.execute({"query": "test search"}, context)

        assert result.success
        assert "No search results found" in result.content

    @pytest.mark.asyncio
    async def test_search_passes_parameters(self) -> None:
        """Should pass all parameters to search client."""
        mock_client = create_mock_client()
        tool = WebSearchTool(mock_client)
        context = create_test_context()

        await tool.execute({
            "query": "test query",
            "count": 15,
            "freshness": "pd",
        }, context)

        mock_client.search.assert_called_once()
        call_kwargs = mock_client.search.call_args.kwargs
        assert call_kwargs["query"] == "test query"
        assert call_kwargs["count"] == 15
        assert call_kwargs["freshness"] == "pd"

    @pytest.mark.asyncio
    async def test_search_error_handling(self) -> None:
        """Should handle search errors gracefully."""
        mock_client = create_mock_client()
        mock_client.search.side_effect = Exception("API error")
        tool = WebSearchTool(mock_client)
        context = create_test_context()

        result = await tool.execute({"query": "test search"}, context)

        assert not result.success
        assert result.error is not None
        assert "API error" in result.error

    @pytest.mark.asyncio
    async def test_urls_not_shown_in_content(self) -> None:
        """Should NOT include URLs in content (security feature)."""
        mock_client = create_mock_client()
        tool = WebSearchTool(mock_client)
        context = create_test_context()

        result = await tool.execute({"query": "test search"}, context)

        # URLs should be in sources, NOT in content
        assert "https://example.com" not in result.content
        assert result.sources is not None
        assert "https://example.com/1" in result.sources[0]["url"]
