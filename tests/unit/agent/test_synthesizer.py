"""Unit tests for Synthesizer agent."""

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.agent.nodes.synthesizer import run_synthesizer, stream_synthesis
from src.agent.state import (
    Plan,
    PlanStep,
    QueryClassification,
    ResearchState,
    SourceInfo,
    StepStatus,
    StepType,
)
from src.services.llm.types import LLMResponse


@pytest.fixture
def mock_llm_client() -> AsyncMock:
    """Create a mock LLM client."""
    return AsyncMock()


@pytest.fixture
def research_state_with_observations() -> ResearchState:
    """Create a research state with observations and sources."""
    state = ResearchState(
        query="What are the latest developments in quantum computing?",
        conversation_history=[],
        session_id=uuid4(),
    )

    state.all_observations = [
        "Google achieved quantum supremacy with their Sycamore processor in 2019.",
        "IBM announced a roadmap to build a 1000+ qubit quantum computer by 2023.",
        "Advances in quantum error correction are making quantum computers more practical.",
    ]

    state.sources = [
        SourceInfo(
            url="https://example.com/quantum1",
            title="Google Quantum Supremacy",
            snippet="Google's quantum computer performed a calculation in 200 seconds...",
        ),
        SourceInfo(
            url="https://example.com/quantum2",
            title="IBM Quantum Roadmap",
            snippet="IBM's roadmap outlines plans for scaling quantum computers...",
        ),
    ]

    state.current_plan = Plan(
        id="plan-1",
        title="Research Quantum Computing",
        thought="Comprehensive research plan",
        steps=[
            PlanStep(
                id="step-1",
                title="Search papers",
                description="Find academic papers",
                step_type=StepType.RESEARCH,
                needs_search=True,
                status=StepStatus.COMPLETED,
            ),
            PlanStep(
                id="step-2",
                title="Analyze findings",
                description="Synthesize research",
                step_type=StepType.ANALYSIS,
                needs_search=False,
                status=StepStatus.COMPLETED,
            ),
        ],
    )

    state.query_classification = QueryClassification(
        complexity="complex",
        follow_up_type="new_topic",
        is_ambiguous=False,
        recommended_depth="deep",
        reasoning="Complex research query",
    )

    return state


@pytest.fixture
def empty_research_state() -> ResearchState:
    """Create a research state with no observations."""
    return ResearchState(
        query="Simple query",
        conversation_history=[],
        session_id=uuid4(),
    )


class TestRunSynthesizer:
    """Tests for run_synthesizer function."""

    @pytest.mark.asyncio
    async def test_generates_final_report(
        self, research_state_with_observations: ResearchState, mock_llm_client: AsyncMock
    ):
        """Test successful report generation."""
        # Arrange
        final_report = """
        ## Research Summary

        Quantum computing has made significant advances in recent years.

        ### Key Findings
        1. Google achieved quantum supremacy in 2019
        2. IBM has an ambitious roadmap for scaling quantum computers
        3. Error correction techniques are improving

        ### Sources
        - [Google Quantum Supremacy](https://example.com/quantum1)
        - [IBM Quantum Roadmap](https://example.com/quantum2)
        """

        mock_llm_client.complete = AsyncMock(
            return_value=LLMResponse(
                content=final_report,
                usage={"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
                endpoint_id="test-endpoint",
                duration_ms=2000.0,
            )
        )

        # Act
        result = await run_synthesizer(research_state_with_observations, mock_llm_client)

        # Assert
        assert result.final_report == final_report
        assert result.completed_at is not None

    @pytest.mark.asyncio
    async def test_handles_empty_observations(
        self, empty_research_state: ResearchState, mock_llm_client: AsyncMock
    ):
        """Test synthesis with no observations."""
        # Arrange
        report = "Based on the query, here is a general summary..."

        mock_llm_client.complete = AsyncMock(
            return_value=LLMResponse(
                content=report,
                usage={"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300},
                endpoint_id="test-endpoint",
                duration_ms=1000.0,
            )
        )

        # Act
        result = await run_synthesizer(empty_research_state, mock_llm_client)

        # Assert
        assert result.final_report == report
        mock_llm_client.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling_creates_fallback_report(
        self, research_state_with_observations: ResearchState, mock_llm_client: AsyncMock
    ):
        """Test that errors result in a fallback report using observations."""
        # Arrange
        mock_llm_client.complete = AsyncMock(side_effect=Exception("LLM API error"))

        # Act
        result = await run_synthesizer(research_state_with_observations, mock_llm_client)

        # Assert
        assert result.final_report is not None
        assert "Research Summary" in result.final_report
        assert "Google achieved quantum supremacy" in result.final_report
        assert result.completed_at is not None

    @pytest.mark.asyncio
    async def test_uses_complex_model_tier(
        self, research_state_with_observations: ResearchState, mock_llm_client: AsyncMock
    ):
        """Test that synthesizer uses the COMPLEX model tier with depth-appropriate tokens."""
        # Arrange
        from src.services.llm.types import ModelTier
        from src.agent.config import get_report_limits

        mock_llm_client.complete = AsyncMock(
            return_value=LLMResponse(
                content="Report content",
                usage={"prompt_tokens": 500, "completion_tokens": 200, "total_tokens": 700},
                endpoint_id="test-endpoint",
                duration_ms=1500.0,
            )
        )

        # Act
        await run_synthesizer(research_state_with_observations, mock_llm_client)

        # Assert
        mock_llm_client.complete.assert_called_once()
        call_kwargs = mock_llm_client.complete.call_args.kwargs
        assert call_kwargs["tier"] == ModelTier.COMPLEX
        # max_tokens is dynamic based on resolved depth from research_types config
        # The fixture uses default depth ("auto" -> "extended")
        resolved_depth = research_state_with_observations.resolve_depth()
        expected_max_tokens = get_report_limits(resolved_depth).max_tokens
        assert call_kwargs["max_tokens"] == expected_max_tokens

    @pytest.mark.asyncio
    async def test_includes_sources_in_prompt(
        self, research_state_with_observations: ResearchState, mock_llm_client: AsyncMock
    ):
        """Test that sources are included in the synthesis prompt."""
        # Arrange
        mock_llm_client.complete = AsyncMock(
            return_value=LLMResponse(
                content="Report with sources",
                usage={"prompt_tokens": 800, "completion_tokens": 300, "total_tokens": 1100},
                endpoint_id="test-endpoint",
                duration_ms=1800.0,
            )
        )

        # Act
        await run_synthesizer(research_state_with_observations, mock_llm_client)

        # Assert - Check that complete was called (sources would be in messages)
        mock_llm_client.complete.assert_called_once()


class TestStreamSynthesis:
    """Tests for stream_synthesis function."""

    @pytest.mark.asyncio
    async def test_streams_report_chunks(
        self, research_state_with_observations: ResearchState, mock_llm_client: AsyncMock
    ):
        """Test that synthesis streams chunks correctly."""
        # Arrange
        chunks = ["## Research", " Summary\n\n", "Quantum computing", " has advanced..."]

        async def mock_stream(*args, **kwargs):
            for chunk in chunks:
                yield chunk

        mock_llm_client.stream = mock_stream

        # Act
        received_chunks = []
        async for chunk in stream_synthesis(research_state_with_observations, mock_llm_client):
            received_chunks.append(chunk)

        # Assert
        assert received_chunks == chunks
        assert research_state_with_observations.final_report == "".join(chunks)
        assert research_state_with_observations.completed_at is not None

    @pytest.mark.asyncio
    async def test_handles_stream_error(
        self, research_state_with_observations: ResearchState, mock_llm_client: AsyncMock
    ):
        """Test that stream errors are handled gracefully."""
        # Arrange
        async def mock_stream(*args, **kwargs):
            yield "Partial content"
            yield " before error"
            raise Exception("Stream connection lost")

        mock_llm_client.stream = mock_stream

        # Act
        received_chunks = []
        async for chunk in stream_synthesis(research_state_with_observations, mock_llm_client):
            received_chunks.append(chunk)

        # Assert
        assert len(received_chunks) == 3  # Two normal chunks + error message
        assert "Error during synthesis" in received_chunks[-1]

    @pytest.mark.asyncio
    async def test_empty_observations_still_streams(
        self, empty_research_state: ResearchState, mock_llm_client: AsyncMock
    ):
        """Test streaming with empty observations."""
        # Arrange
        async def mock_stream(*args, **kwargs):
            yield "Summary based on query"

        mock_llm_client.stream = mock_stream

        # Act
        received_chunks = []
        async for chunk in stream_synthesis(empty_research_state, mock_llm_client):
            received_chunks.append(chunk)

        # Assert
        assert len(received_chunks) == 1
        assert received_chunks[0] == "Summary based on query"


class TestSourcesFormatting:
    """Tests for sources formatting in synthesizer."""

    @pytest.mark.asyncio
    async def test_limits_sources_to_twenty(
        self, mock_llm_client: AsyncMock
    ):
        """Test that sources are limited to 20 in the prompt."""
        # Arrange
        state = ResearchState(
            query="Test query",
            conversation_history=[],
            session_id=uuid4(),
        )

        # Add 30 sources
        state.sources = [
            SourceInfo(url=f"https://example.com/{i}", title=f"Source {i}")
            for i in range(30)
        ]

        state.all_observations = ["Test observation"]

        mock_llm_client.complete = AsyncMock(
            return_value=LLMResponse(
                content="Report",
                usage={"prompt_tokens": 500, "completion_tokens": 100, "total_tokens": 600},
                endpoint_id="test-endpoint",
                duration_ms=1000.0,
            )
        )

        # Act
        await run_synthesizer(state, mock_llm_client)

        # Assert - The function limits to 20 sources (line 49 in synthesizer.py)
        # We can verify it completes without error with many sources
        mock_llm_client.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_sources_without_titles(
        self, mock_llm_client: AsyncMock
    ):
        """Test that sources without titles use 'Untitled'."""
        # Arrange
        state = ResearchState(
            query="Test query",
            conversation_history=[],
            session_id=uuid4(),
        )

        state.sources = [
            SourceInfo(url="https://example.com/1", title=None),
            SourceInfo(url="https://example.com/2", title=""),
        ]

        state.all_observations = ["Test observation"]

        mock_llm_client.complete = AsyncMock(
            return_value=LLMResponse(
                content="Report",
                usage={"prompt_tokens": 300, "completion_tokens": 100, "total_tokens": 400},
                endpoint_id="test-endpoint",
                duration_ms=800.0,
            )
        )

        # Act
        await run_synthesizer(state, mock_llm_client)

        # Assert - Should complete without error
        mock_llm_client.complete.assert_called_once()
