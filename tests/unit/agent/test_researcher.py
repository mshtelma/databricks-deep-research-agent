"""Unit tests for Researcher agent."""

from unittest.mock import AsyncMock

import pytest

from src.agent.nodes.researcher import SearchQueriesOutput, _generate_search_queries
from src.services.llm.types import LLMResponse


@pytest.fixture
def mock_llm_client() -> AsyncMock:
    """Create a mock LLM client."""
    return AsyncMock()


class TestGenerateSearchQueries:
    """Tests for _generate_search_queries function."""

    @pytest.mark.asyncio
    async def test_uses_structured_output(self, mock_llm_client: AsyncMock):
        """Test that structured output is used when available."""
        # Arrange
        queries_output = SearchQueriesOutput(
            queries=["quantum computing papers 2024", "quantum computing breakthroughs"]
        )

        mock_llm_client.complete = AsyncMock(
            return_value=LLMResponse(
                content="{}",  # Empty content, structured is used
                usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                endpoint_id="test-endpoint",
                duration_ms=200.0,
                structured=queries_output,
            )
        )

        # Act
        result = await _generate_search_queries(
            llm=mock_llm_client,
            step_title="Research quantum computing",
            step_description="Find recent papers on quantum computing",
            query="What are the latest developments in quantum computing?",
            max_generated_queries=3,
        )

        # Assert
        assert result == ["quantum computing papers 2024", "quantum computing breakthroughs"]

    @pytest.mark.asyncio
    async def test_fallback_to_json_parsing(self, mock_llm_client: AsyncMock):
        """Test fallback to JSON parsing when structured output not available."""
        # Arrange
        mock_llm_client.complete = AsyncMock(
            return_value=LLMResponse(
                content='["query 1", "query 2", "query 3"]',
                usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                endpoint_id="test-endpoint",
                duration_ms=200.0,
                structured=None,  # No structured output
            )
        )

        # Act
        result = await _generate_search_queries(
            llm=mock_llm_client,
            step_title="Test step",
            step_description="Test description",
            query="Test query",
            max_generated_queries=3,
        )

        # Assert
        assert result == ["query 1", "query 2", "query 3"]

    @pytest.mark.asyncio
    async def test_fallback_json_with_queries_key(self, mock_llm_client: AsyncMock):
        """Test fallback JSON parsing when response has 'queries' key."""
        # Arrange
        mock_llm_client.complete = AsyncMock(
            return_value=LLMResponse(
                content='{"queries": ["query A", "query B"]}',
                usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                endpoint_id="test-endpoint",
                duration_ms=200.0,
                structured=None,
            )
        )

        # Act
        result = await _generate_search_queries(
            llm=mock_llm_client,
            step_title="Test step",
            step_description="Test description",
            query="Test query",
            max_generated_queries=3,
        )

        # Assert
        assert result == ["query A", "query B"]

    @pytest.mark.asyncio
    async def test_empty_response_uses_fallback(self, mock_llm_client: AsyncMock):
        """Test that empty response falls back to step description query."""
        # Arrange - Simulate the error that was happening in production
        mock_llm_client.complete = AsyncMock(
            return_value=LLMResponse(
                content="",  # Empty response
                usage={"prompt_tokens": 100, "completion_tokens": 0, "total_tokens": 100},
                endpoint_id="test-endpoint",
                duration_ms=200.0,
                structured=None,
            )
        )

        # Act
        result = await _generate_search_queries(
            llm=mock_llm_client,
            step_title="Research topic",
            step_description="Find information about the topic",
            query="What is the topic?",
            max_generated_queries=3,
        )

        # Assert - Should use fallback: step title + description
        assert len(result) == 1
        assert "Research topic" in result[0]

    @pytest.mark.asyncio
    async def test_malformed_json_uses_fallback(self, mock_llm_client: AsyncMock):
        """Test that malformed JSON falls back to step description query."""
        # Arrange
        mock_llm_client.complete = AsyncMock(
            return_value=LLMResponse(
                content="Here are some queries:\n- query 1\n- query 2",  # Not JSON
                usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                endpoint_id="test-endpoint",
                duration_ms=200.0,
                structured=None,
            )
        )

        # Act
        result = await _generate_search_queries(
            llm=mock_llm_client,
            step_title="Search step",
            step_description="Search for relevant information",
            query="Find info",
            max_generated_queries=3,
        )

        # Assert - Should use fallback
        assert len(result) == 1
        assert "Search step" in result[0]

    @pytest.mark.asyncio
    async def test_respects_max_generated_queries_limit(self, mock_llm_client: AsyncMock):
        """Test that result is limited to max_generated_queries."""
        # Arrange
        queries_output = SearchQueriesOutput(
            queries=["query 1", "query 2", "query 3", "query 4", "query 5"]
        )

        mock_llm_client.complete = AsyncMock(
            return_value=LLMResponse(
                content="{}",
                usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                endpoint_id="test-endpoint",
                duration_ms=200.0,
                structured=queries_output,
            )
        )

        # Act
        result = await _generate_search_queries(
            llm=mock_llm_client,
            step_title="Test",
            step_description="Test",
            query="Test",
            max_generated_queries=2,  # Limit to 2
        )

        # Assert
        assert len(result) == 2
        assert result == ["query 1", "query 2"]

    @pytest.mark.asyncio
    async def test_llm_exception_uses_fallback(self, mock_llm_client: AsyncMock):
        """Test that LLM exceptions result in fallback query."""
        # Arrange
        mock_llm_client.complete = AsyncMock(side_effect=Exception("API Error"))

        # Act
        result = await _generate_search_queries(
            llm=mock_llm_client,
            step_title="Error test",
            step_description="Test handling of errors",
            query="Test query",
            max_generated_queries=3,
        )

        # Assert - Should use fallback
        assert len(result) == 1
        assert "Error test" in result[0]


class TestSearchQueriesOutput:
    """Tests for SearchQueriesOutput model."""

    def test_model_creation(self):
        """Test basic model creation."""
        output = SearchQueriesOutput(queries=["query 1", "query 2"])
        assert output.queries == ["query 1", "query 2"]

    def test_empty_queries_list(self):
        """Test model with empty queries list."""
        output = SearchQueriesOutput(queries=[])
        assert output.queries == []

    def test_model_serialization(self):
        """Test JSON serialization of SearchQueriesOutput."""
        output = SearchQueriesOutput(queries=["a", "b", "c"])
        json_str = output.model_dump_json()
        parsed = SearchQueriesOutput.model_validate_json(json_str)
        assert parsed.queries == ["a", "b", "c"]
