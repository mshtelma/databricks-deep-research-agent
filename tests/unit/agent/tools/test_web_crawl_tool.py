"""Unit tests for WebCrawlTool class."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from deep_research.agent.tools.base import ResearchContext, ResearchTool
from deep_research.agent.tools.web_crawler import (
    CrawlOutput,
    CrawlResult,
    WebCrawler,
    WebCrawlTool,
)


def create_test_context(url_registry: dict[str, Any] | None = None) -> ResearchContext:
    """Create a test ResearchContext with optional url_registry."""
    return ResearchContext(
        chat_id=uuid4(),
        user_id="test-user",
        research_type="medium",
        url_registry=url_registry or {},
    )


def create_mock_crawler(results: list[CrawlResult] | None = None) -> MagicMock:
    """Create a mock WebCrawler."""
    mock_crawler = MagicMock(spec=WebCrawler)

    if results is None:
        results = [
            CrawlResult(
                url="https://example.com/page",
                title="Example Page",
                content="This is the page content.",
                success=True,
            )
        ]

    mock_output = CrawlOutput(
        results=results,
        successful_count=sum(1 for r in results if r.success),
        failed_count=sum(1 for r in results if not r.success),
    )

    mock_crawler.crawl = AsyncMock(return_value=mock_output)
    return mock_crawler


class TestWebCrawlToolDefinition:
    """Tests for WebCrawlTool definition property."""

    def test_implements_research_tool_protocol(self) -> None:
        """WebCrawlTool should implement ResearchTool protocol."""
        mock_crawler = create_mock_crawler()
        tool = WebCrawlTool(mock_crawler)
        assert isinstance(tool, ResearchTool)

    def test_definition_has_correct_name(self) -> None:
        """Tool definition should have name 'web_crawl'."""
        mock_crawler = create_mock_crawler()
        tool = WebCrawlTool(mock_crawler)
        assert tool.definition.name == "web_crawl"

    def test_definition_has_description(self) -> None:
        """Tool definition should have a description."""
        mock_crawler = create_mock_crawler()
        tool = WebCrawlTool(mock_crawler)
        assert tool.definition.description
        assert "content" in tool.definition.description.lower()

    def test_definition_has_index_parameter(self) -> None:
        """Tool definition should have 'index' parameter."""
        mock_crawler = create_mock_crawler()
        tool = WebCrawlTool(mock_crawler)
        props = tool.definition.parameters["properties"]
        assert "index" in props
        assert props["index"]["type"] == "integer"

    def test_definition_has_url_parameter(self) -> None:
        """Tool definition should have optional 'url' parameter."""
        mock_crawler = create_mock_crawler()
        tool = WebCrawlTool(mock_crawler)
        props = tool.definition.parameters["properties"]
        assert "url" in props
        assert props["url"]["type"] == "string"

    def test_no_required_parameters(self) -> None:
        """Neither index nor url should be required (validation handles this)."""
        mock_crawler = create_mock_crawler()
        tool = WebCrawlTool(mock_crawler)
        assert tool.definition.parameters.get("required", []) == []


class TestWebCrawlToolValidation:
    """Tests for WebCrawlTool argument validation."""

    def test_valid_index(self) -> None:
        """Should accept valid index argument."""
        mock_crawler = create_mock_crawler()
        tool = WebCrawlTool(mock_crawler)
        errors = tool.validate_arguments({"index": 0})
        assert errors == []

    def test_valid_url(self) -> None:
        """Should accept valid url argument."""
        mock_crawler = create_mock_crawler()
        tool = WebCrawlTool(mock_crawler)
        errors = tool.validate_arguments({"url": "https://example.com"})
        assert errors == []

    def test_valid_both(self) -> None:
        """Should accept both index and url."""
        mock_crawler = create_mock_crawler()
        tool = WebCrawlTool(mock_crawler)
        errors = tool.validate_arguments({"index": 0, "url": "https://example.com"})
        assert errors == []

    def test_missing_both(self) -> None:
        """Should reject missing both index and url."""
        mock_crawler = create_mock_crawler()
        tool = WebCrawlTool(mock_crawler)
        errors = tool.validate_arguments({})
        assert len(errors) == 1
        assert "index" in errors[0] or "url" in errors[0]

    def test_invalid_index_type(self) -> None:
        """Should reject non-integer index."""
        mock_crawler = create_mock_crawler()
        tool = WebCrawlTool(mock_crawler)
        errors = tool.validate_arguments({"index": "zero"})
        assert len(errors) == 1
        assert "integer" in errors[0]

    def test_negative_index(self) -> None:
        """Should reject negative index."""
        mock_crawler = create_mock_crawler()
        tool = WebCrawlTool(mock_crawler)
        errors = tool.validate_arguments({"index": -1})
        assert len(errors) == 1
        assert "non-negative" in errors[0]

    def test_invalid_url_type(self) -> None:
        """Should reject non-string url."""
        mock_crawler = create_mock_crawler()
        tool = WebCrawlTool(mock_crawler)
        errors = tool.validate_arguments({"url": 123})
        assert len(errors) == 1
        assert "string" in errors[0]

    def test_invalid_url_scheme(self) -> None:
        """Should reject url without http/https scheme."""
        mock_crawler = create_mock_crawler()
        tool = WebCrawlTool(mock_crawler)
        errors = tool.validate_arguments({"url": "ftp://example.com"})
        assert len(errors) == 1
        assert "http" in errors[0]


class TestWebCrawlToolExecutionWithUrl:
    """Tests for WebCrawlTool execute with direct URL."""

    @pytest.mark.asyncio
    async def test_successful_crawl_with_url(self) -> None:
        """Should execute crawl and return content."""
        mock_crawler = create_mock_crawler()
        tool = WebCrawlTool(mock_crawler)
        context = create_test_context()

        result = await tool.execute({"url": "https://example.com/page"}, context)

        assert result.success
        assert "Example Page" in result.content
        assert "This is the page content." in result.content

    @pytest.mark.asyncio
    async def test_crawl_returns_sources(self) -> None:
        """Should include sources for citation tracking."""
        mock_crawler = create_mock_crawler()
        tool = WebCrawlTool(mock_crawler)
        context = create_test_context()

        result = await tool.execute({"url": "https://example.com/page"}, context)

        assert result.sources is not None
        assert len(result.sources) == 1
        assert result.sources[0]["type"] == "web"
        assert result.sources[0]["url"] == "https://example.com/page"

    @pytest.mark.asyncio
    async def test_crawl_returns_data(self) -> None:
        """Should include data with URL and content info."""
        mock_crawler = create_mock_crawler()
        tool = WebCrawlTool(mock_crawler)
        context = create_test_context()

        result = await tool.execute({"url": "https://example.com/page"}, context)

        assert result.data is not None
        assert result.data["url"] == "https://example.com/page"
        assert result.data["title"] == "Example Page"
        assert result.data["content_length"] > 0

    @pytest.mark.asyncio
    async def test_crawl_failure(self) -> None:
        """Should handle crawl failures gracefully."""
        mock_crawler = create_mock_crawler(results=[
            CrawlResult(
                url="https://example.com/page",
                title=None,
                content="",
                success=False,
                error="HTTP 404",
            )
        ])
        tool = WebCrawlTool(mock_crawler)
        context = create_test_context()

        result = await tool.execute({"url": "https://example.com/page"}, context)

        assert not result.success
        assert result.error is not None
        assert "404" in result.error


class TestWebCrawlToolExecutionWithIndex:
    """Tests for WebCrawlTool execute with index (from search results)."""

    @pytest.mark.asyncio
    async def test_crawl_with_index_dict_registry(self) -> None:
        """Should resolve URL from dict-based url_registry."""
        mock_crawler = create_mock_crawler()
        tool = WebCrawlTool(mock_crawler)

        # Create a dict-based registry simulating search results
        url_registry = {
            "entry_0": {
                "index": 0,
                "url": "https://example.com/page",
                "title": "Registered Page",
            }
        }
        context = create_test_context(url_registry=url_registry)

        result = await tool.execute({"index": 0}, context)

        assert result.success
        mock_crawler.crawl.assert_called_once()
        # Check that the correct URL was passed
        call_args = mock_crawler.crawl.call_args
        assert "https://example.com/page" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_crawl_with_invalid_index(self) -> None:
        """Should handle invalid index gracefully."""
        mock_crawler = create_mock_crawler()
        tool = WebCrawlTool(mock_crawler)

        url_registry = {
            "entry_0": {"index": 0, "url": "https://example.com/page"}
        }
        context = create_test_context(url_registry=url_registry)

        result = await tool.execute({"index": 99}, context)

        assert not result.success
        assert "No URL found for index 99" in result.content

    @pytest.mark.asyncio
    async def test_crawl_with_empty_registry(self) -> None:
        """Should handle empty registry."""
        mock_crawler = create_mock_crawler()
        tool = WebCrawlTool(mock_crawler)
        context = create_test_context(url_registry={})

        result = await tool.execute({"index": 0}, context)

        assert not result.success
        # Empty dict is falsy, so it triggers "No URL registry available"
        assert "registry" in result.content.lower() or "No URL found" in result.content


class TestWebCrawlToolEdgeCases:
    """Edge case tests for WebCrawlTool."""

    @pytest.mark.asyncio
    async def test_missing_both_parameters(self) -> None:
        """Should return error when neither index nor url provided."""
        mock_crawler = create_mock_crawler()
        tool = WebCrawlTool(mock_crawler)
        context = create_test_context()

        result = await tool.execute({}, context)

        assert not result.success
        assert "must be provided" in result.content

    @pytest.mark.asyncio
    async def test_empty_crawl_results(self) -> None:
        """Should handle empty crawl results."""
        mock_crawler = create_mock_crawler(results=[])
        tool = WebCrawlTool(mock_crawler)
        context = create_test_context()

        result = await tool.execute({"url": "https://example.com/page"}, context)

        assert not result.success
        assert "Failed to crawl" in result.content

    @pytest.mark.asyncio
    async def test_exception_handling(self) -> None:
        """Should handle unexpected exceptions."""
        mock_crawler = create_mock_crawler()
        mock_crawler.crawl.side_effect = Exception("Unexpected error")
        tool = WebCrawlTool(mock_crawler)
        context = create_test_context()

        result = await tool.execute({"url": "https://example.com/page"}, context)

        assert not result.success
        assert result.error is not None
        assert "Unexpected error" in result.error
