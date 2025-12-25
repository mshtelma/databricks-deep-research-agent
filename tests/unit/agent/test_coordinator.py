"""Unit tests for Coordinator agent."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.agent.nodes.coordinator import (
    CoordinatorOutput,
    handle_simple_query,
    run_coordinator,
)
from src.agent.state import ResearchState
from src.services.llm.types import LLMResponse


@pytest.fixture
def research_state() -> ResearchState:
    """Create a research state for testing."""
    return ResearchState(
        query="What are the latest developments in quantum computing?",
        conversation_history=[],
        session_id=uuid4(),
    )


@pytest.fixture
def research_state_with_history() -> ResearchState:
    """Create a research state with conversation history."""
    return ResearchState(
        query="Can you elaborate on that?",
        conversation_history=[
            {"role": "user", "content": "What is quantum computing?"},
            {"role": "assistant", "content": "Quantum computing uses quantum mechanics..."},
        ],
        session_id=uuid4(),
    )


@pytest.fixture
def mock_llm_client() -> AsyncMock:
    """Create a mock LLM client."""
    client = AsyncMock()
    return client


class TestRunCoordinator:
    """Tests for run_coordinator function."""

    @pytest.mark.asyncio
    async def test_classify_complex_query(
        self, research_state: ResearchState, mock_llm_client: AsyncMock
    ):
        """Test classification of a complex research query."""
        # Arrange
        coordinator_output = CoordinatorOutput(
            complexity="complex",
            follow_up_type="new_topic",
            is_ambiguous=False,
            clarifying_questions=[],
            recommended_depth="deep",
            reasoning="This is a research query requiring multiple sources.",
            is_simple_query=False,
            direct_response=None,
        )

        mock_llm_client.complete = AsyncMock(
            return_value=LLMResponse(
                content=coordinator_output.model_dump_json(),
                usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                endpoint_id="test-endpoint",
                duration_ms=500.0,
                structured=coordinator_output,
            )
        )

        # Act
        result = await run_coordinator(research_state, mock_llm_client)

        # Assert
        assert result.query_classification is not None
        assert result.query_classification.complexity == "complex"
        assert result.query_classification.follow_up_type == "new_topic"
        assert result.is_simple_query is False
        assert result.direct_response is None

    @pytest.mark.asyncio
    async def test_classify_simple_query(
        self, mock_llm_client: AsyncMock
    ):
        """Test classification of a simple query with direct response."""
        # Arrange
        state = ResearchState(
            query="What is 2 + 2?",
            conversation_history=[],
            session_id=uuid4(),
        )

        coordinator_output = CoordinatorOutput(
            complexity="simple",
            follow_up_type="new_topic",
            is_ambiguous=False,
            clarifying_questions=[],
            recommended_depth="shallow",
            reasoning="This is a simple arithmetic question.",
            is_simple_query=True,
            direct_response="2 + 2 equals 4.",
        )

        mock_llm_client.complete = AsyncMock(
            return_value=LLMResponse(
                content=coordinator_output.model_dump_json(),
                usage={"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
                endpoint_id="test-endpoint",
                duration_ms=200.0,
                structured=coordinator_output,
            )
        )

        # Act
        result = await run_coordinator(state, mock_llm_client)

        # Assert
        assert result.query_classification is not None
        assert result.query_classification.complexity == "simple"
        assert result.is_simple_query is True
        assert result.direct_response == "2 + 2 equals 4."

    @pytest.mark.asyncio
    async def test_classify_ambiguous_query(
        self, mock_llm_client: AsyncMock
    ):
        """Test classification of an ambiguous query that needs clarification."""
        # Arrange
        state = ResearchState(
            query="Tell me about that",
            conversation_history=[],
            session_id=uuid4(),
        )

        coordinator_output = CoordinatorOutput(
            complexity="moderate",
            follow_up_type="clarification",
            is_ambiguous=True,
            clarifying_questions=[
                "Could you specify what topic you'd like to know about?",
                "Are you referring to something we discussed earlier?",
            ],
            recommended_depth="auto",
            reasoning="The query is too vague without context.",
            is_simple_query=False,
            direct_response=None,
        )

        mock_llm_client.complete = AsyncMock(
            return_value=LLMResponse(
                content=coordinator_output.model_dump_json(),
                usage={"prompt_tokens": 50, "completion_tokens": 50, "total_tokens": 100},
                endpoint_id="test-endpoint",
                duration_ms=300.0,
                structured=coordinator_output,
            )
        )

        # Act
        result = await run_coordinator(state, mock_llm_client)

        # Assert
        assert result.query_classification is not None
        assert result.query_classification.is_ambiguous is True
        assert len(result.query_classification.clarifying_questions) == 2

    @pytest.mark.asyncio
    async def test_classify_follow_up_query(
        self, research_state_with_history: ResearchState, mock_llm_client: AsyncMock
    ):
        """Test classification of a follow-up query."""
        # Arrange
        coordinator_output = CoordinatorOutput(
            complexity="moderate",
            follow_up_type="clarification",
            is_ambiguous=False,
            clarifying_questions=[],
            recommended_depth="moderate",
            reasoning="This is a follow-up to the previous quantum computing discussion.",
            is_simple_query=False,
            direct_response=None,
        )

        mock_llm_client.complete = AsyncMock(
            return_value=LLMResponse(
                content=coordinator_output.model_dump_json(),
                usage={"prompt_tokens": 120, "completion_tokens": 60, "total_tokens": 180},
                endpoint_id="test-endpoint",
                duration_ms=400.0,
                structured=coordinator_output,
            )
        )

        # Act
        result = await run_coordinator(research_state_with_history, mock_llm_client)

        # Assert
        assert result.query_classification is not None
        assert result.query_classification.follow_up_type == "clarification"
        mock_llm_client.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling_returns_default_classification(
        self, research_state: ResearchState, mock_llm_client: AsyncMock
    ):
        """Test that errors result in default classification."""
        # Arrange
        mock_llm_client.complete = AsyncMock(side_effect=Exception("LLM API error"))

        # Act
        result = await run_coordinator(research_state, mock_llm_client)

        # Assert
        assert result.query_classification is not None
        assert result.query_classification.complexity == "moderate"
        assert result.query_classification.follow_up_type == "new_topic"
        assert result.query_classification.is_ambiguous is False
        assert "error" in result.query_classification.reasoning.lower()

    @pytest.mark.asyncio
    async def test_uses_structured_output_from_response(
        self, research_state: ResearchState, mock_llm_client: AsyncMock
    ):
        """Test that structured output from response is used when available."""
        # Arrange
        coordinator_output = CoordinatorOutput(
            complexity="complex",
            follow_up_type="new_topic",
            is_ambiguous=False,
            clarifying_questions=[],
            recommended_depth="deep",
            reasoning="Using structured output directly.",
            is_simple_query=False,
            direct_response=None,
        )

        mock_llm_client.complete = AsyncMock(
            return_value=LLMResponse(
                content="{}",  # Empty content, structured is used
                usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                endpoint_id="test-endpoint",
                duration_ms=500.0,
                structured=coordinator_output,
            )
        )

        # Act
        result = await run_coordinator(research_state, mock_llm_client)

        # Assert
        assert result.query_classification is not None
        assert result.query_classification.reasoning == "Using structured output directly."


class TestHandleSimpleQuery:
    """Tests for handle_simple_query function."""

    @pytest.mark.asyncio
    async def test_yields_direct_response_when_available(
        self, mock_llm_client: AsyncMock
    ):
        """Test that direct response is yielded when available."""
        # Arrange
        state = ResearchState(
            query="What is 2 + 2?",
            conversation_history=[],
            session_id=uuid4(),
            direct_response="2 + 2 equals 4.",
        )

        # Act
        chunks = []
        async for chunk in handle_simple_query(state, mock_llm_client):
            chunks.append(chunk)

        # Assert
        assert len(chunks) == 1
        assert chunks[0] == "2 + 2 equals 4."
        mock_llm_client.stream.assert_not_called()

    @pytest.mark.asyncio
    async def test_streams_response_when_no_direct_response(
        self, mock_llm_client: AsyncMock
    ):
        """Test that LLM stream is used when no direct response available."""
        # Arrange
        state = ResearchState(
            query="Hello!",
            conversation_history=[],
            session_id=uuid4(),
            direct_response=None,  # No direct response
        )

        async def mock_stream(*args, **kwargs):
            for chunk in ["Hello", "!", " How", " can", " I", " help?"]:
                yield chunk

        mock_llm_client.stream = mock_stream

        # Act
        chunks = []
        async for chunk in handle_simple_query(state, mock_llm_client):
            chunks.append(chunk)

        # Assert
        assert len(chunks) == 6
        assert "".join(chunks) == "Hello! How can I help?"


class TestCoordinatorOutput:
    """Tests for CoordinatorOutput model."""

    def test_model_defaults(self):
        """Test default values for CoordinatorOutput."""
        output = CoordinatorOutput(
            complexity="moderate",
            follow_up_type="new_topic",
            is_ambiguous=False,
            reasoning="Test reasoning",
        )

        assert output.clarifying_questions == []
        assert output.recommended_depth == "auto"
        assert output.is_simple_query is False
        assert output.direct_response is None

    def test_model_serialization(self):
        """Test JSON serialization of CoordinatorOutput."""
        output = CoordinatorOutput(
            complexity="simple",
            follow_up_type="clarification",
            is_ambiguous=True,
            clarifying_questions=["Question 1?"],
            recommended_depth="shallow",
            reasoning="Test reasoning",
            is_simple_query=True,
            direct_response="Direct answer",
        )

        json_str = output.model_dump_json()
        parsed = CoordinatorOutput.model_validate_json(json_str)

        assert parsed.complexity == "simple"
        assert parsed.is_simple_query is True
        assert parsed.direct_response == "Direct answer"
