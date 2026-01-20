"""Unit tests for Reflector agent."""

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from deep_research.agent.nodes.reflector import ReflectorOutput, run_reflector
from deep_research.agent.state import (
    Plan,
    PlanStep,
    ReflectionDecision,
    ResearchState,
    StepStatus,
    StepType,
)
from deep_research.services.llm.types import LLMResponse


@pytest.fixture
def mock_llm_client() -> AsyncMock:
    """Create a mock LLM client."""
    return AsyncMock()


@pytest.fixture
def plan_with_steps() -> Plan:
    """Create a research plan with multiple steps."""
    return Plan(
        id="plan-1",
        title="Research Quantum Computing",
        thought="Comprehensive research on quantum computing developments.",
        steps=[
            PlanStep(
                id="step-1",
                title="Find recent papers",
                description="Search for recent academic papers on quantum computing.",
                step_type=StepType.RESEARCH,
                needs_search=True,
                status=StepStatus.COMPLETED,
                observation="Found 5 relevant papers from 2024.",
            ),
            PlanStep(
                id="step-2",
                title="Analyze industry developments",
                description="Research industry announcements and developments.",
                step_type=StepType.RESEARCH,
                needs_search=True,
                status=StepStatus.IN_PROGRESS,
            ),
            PlanStep(
                id="step-3",
                title="Synthesize findings",
                description="Combine all research into a comprehensive summary.",
                step_type=StepType.ANALYSIS,
                needs_search=False,
                status=StepStatus.PENDING,
            ),
        ],
    )


@pytest.fixture
def research_state_with_plan(plan_with_steps: Plan) -> ResearchState:
    """Create a research state with an active plan."""
    state = ResearchState(
        query="What are the latest developments in quantum computing?",
        conversation_history=[],
        session_id=uuid4(),
    )
    state.current_plan = plan_with_steps
    state.current_step_index = 1  # On step 2
    state.all_observations = ["Found 5 relevant papers from 2024."]
    state.last_observation = "Industry report shows major investments."
    return state


class TestRunReflector:
    """Tests for run_reflector function."""

    @pytest.mark.asyncio
    async def test_continue_decision(
        self, research_state_with_plan: ResearchState, mock_llm_client: AsyncMock
    ):
        """Test reflector decides to continue to next step."""
        # Arrange
        reflector_output = ReflectorOutput(
            decision="continue",
            reasoning="Good progress, should continue with remaining steps.",
            suggested_changes=None,
        )

        mock_llm_client.complete = AsyncMock(
            return_value=LLMResponse(
                content=reflector_output.model_dump_json(),
                usage={"prompt_tokens": 200, "completion_tokens": 50, "total_tokens": 250},
                endpoint_id="test-endpoint",
                duration_ms=400.0,
                structured=reflector_output,
            )
        )

        # Act
        result = await run_reflector(research_state_with_plan, mock_llm_client)

        # Assert
        assert result.last_reflection is not None
        assert result.last_reflection.decision == ReflectionDecision.CONTINUE
        assert len(result.reflection_history) == 1

    @pytest.mark.asyncio
    async def test_adjust_decision(
        self, research_state_with_plan: ResearchState, mock_llm_client: AsyncMock
    ):
        """Test reflector decides to adjust the plan."""
        # Arrange
        reflector_output = ReflectorOutput(
            decision="adjust",
            reasoning="The current plan is missing important aspects. Need to revise.",
            suggested_changes=[
                "Add a step to research quantum error correction",
                "Include more recent industry announcements",
            ],
        )

        mock_llm_client.complete = AsyncMock(
            return_value=LLMResponse(
                content=reflector_output.model_dump_json(),
                usage={"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280},
                endpoint_id="test-endpoint",
                duration_ms=450.0,
                structured=reflector_output,
            )
        )

        # Act
        result = await run_reflector(research_state_with_plan, mock_llm_client)

        # Assert
        assert result.last_reflection is not None
        assert result.last_reflection.decision == ReflectionDecision.ADJUST
        assert result.last_reflection.suggested_changes is not None
        assert len(result.last_reflection.suggested_changes) == 2

    @pytest.mark.asyncio
    async def test_complete_decision(
        self, research_state_with_plan: ResearchState, mock_llm_client: AsyncMock
    ):
        """Test reflector decides to complete early."""
        # Arrange
        reflector_output = ReflectorOutput(
            decision="complete",
            reasoning="We have gathered sufficient information to answer the query.",
            suggested_changes=None,
        )

        mock_llm_client.complete = AsyncMock(
            return_value=LLMResponse(
                content=reflector_output.model_dump_json(),
                usage={"prompt_tokens": 200, "completion_tokens": 40, "total_tokens": 240},
                endpoint_id="test-endpoint",
                duration_ms=350.0,
                structured=reflector_output,
            )
        )

        # Act
        result = await run_reflector(research_state_with_plan, mock_llm_client)

        # Assert
        assert result.last_reflection is not None
        assert result.last_reflection.decision == ReflectionDecision.COMPLETE

    @pytest.mark.asyncio
    async def test_no_plan_returns_early(
        self, mock_llm_client: AsyncMock
    ):
        """Test that reflector returns early when no plan exists."""
        # Arrange
        state = ResearchState(
            query="Test query",
            conversation_history=[],
            session_id=uuid4(),
        )
        state.current_plan = None

        # Act
        result = await run_reflector(state, mock_llm_client)

        # Assert
        assert result.last_reflection is None
        mock_llm_client.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_handling_defaults_to_continue(
        self, research_state_with_plan: ResearchState, mock_llm_client: AsyncMock
    ):
        """Test that errors result in default continue decision."""
        # Arrange
        mock_llm_client.complete = AsyncMock(side_effect=Exception("LLM API error"))

        # Act
        result = await run_reflector(research_state_with_plan, mock_llm_client)

        # Assert
        assert result.last_reflection is not None
        assert result.last_reflection.decision == ReflectionDecision.CONTINUE
        assert "error" in result.last_reflection.reasoning.lower()

    @pytest.mark.asyncio
    async def test_reflection_history_accumulates(
        self, research_state_with_plan: ResearchState, mock_llm_client: AsyncMock
    ):
        """Test that multiple reflections accumulate in history."""
        # Arrange
        reflector_output = ReflectorOutput(
            decision="continue",
            reasoning="Continuing research.",
            suggested_changes=None,
        )

        mock_llm_client.complete = AsyncMock(
            return_value=LLMResponse(
                content=reflector_output.model_dump_json(),
                usage={"prompt_tokens": 200, "completion_tokens": 30, "total_tokens": 230},
                endpoint_id="test-endpoint",
                duration_ms=300.0,
                structured=reflector_output,
            )
        )

        # Act - Run reflector multiple times
        result1 = await run_reflector(research_state_with_plan, mock_llm_client)
        result2 = await run_reflector(result1, mock_llm_client)
        result3 = await run_reflector(result2, mock_llm_client)

        # Assert
        assert len(result3.reflection_history) == 3

    @pytest.mark.asyncio
    async def test_uses_structured_output(
        self, research_state_with_plan: ResearchState, mock_llm_client: AsyncMock
    ):
        """Test that structured output from LLM is used when available."""
        # Arrange
        reflector_output = ReflectorOutput(
            decision="continue",
            reasoning="Using structured output.",
            suggested_changes=["Suggestion 1"],
        )

        mock_llm_client.complete = AsyncMock(
            return_value=LLMResponse(
                content="{}",  # Empty content, structured is used
                usage={"prompt_tokens": 200, "completion_tokens": 50, "total_tokens": 250},
                endpoint_id="test-endpoint",
                duration_ms=400.0,
                structured=reflector_output,
            )
        )

        # Act
        result = await run_reflector(research_state_with_plan, mock_llm_client)

        # Assert
        assert result.last_reflection is not None
        assert result.last_reflection.reasoning == "Using structured output."
        assert result.last_reflection.suggested_changes == ["Suggestion 1"]


    @pytest.mark.asyncio
    async def test_uppercase_decision_normalized(
        self, research_state_with_plan: ResearchState, mock_llm_client: AsyncMock
    ):
        """Test that uppercase decision values are normalized to lowercase."""
        # Arrange - LLM returns uppercase decision (edge case we're fixing)
        ReflectorOutput(
            decision="continue",  # Pydantic validates as lowercase
            reasoning="Should continue with research.",
            suggested_changes=None,
        )

        # Simulate LLM returning uppercase via raw JSON parsing
        mock_llm_client.complete = AsyncMock(
            return_value=LLMResponse(
                content='{"decision": "CONTINUE", "reasoning": "Test", "suggested_changes": null}',
                usage={"prompt_tokens": 200, "completion_tokens": 50, "total_tokens": 250},
                endpoint_id="test-endpoint",
                duration_ms=400.0,
                structured=None,  # Force fallback to JSON parsing
            )
        )

        # Act
        result = await run_reflector(research_state_with_plan, mock_llm_client)

        # Assert - should succeed with lowercase normalization
        assert result.last_reflection is not None
        assert result.last_reflection.decision == ReflectionDecision.CONTINUE


class TestReflectorOutput:
    """Tests for ReflectorOutput model."""

    def test_model_defaults(self):
        """Test default values for ReflectorOutput."""
        output = ReflectorOutput(
            decision="continue",
            reasoning="Test reasoning",
        )

        assert output.suggested_changes is None

    def test_model_serialization(self):
        """Test JSON serialization of ReflectorOutput."""
        output = ReflectorOutput(
            decision="adjust",
            reasoning="Plan needs adjustment.",
            suggested_changes=["Change 1", "Change 2"],
        )

        json_str = output.model_dump_json()
        parsed = ReflectorOutput.model_validate_json(json_str)

        assert parsed.decision == "adjust"
        assert parsed.suggested_changes == ["Change 1", "Change 2"]

    def test_literal_rejects_invalid_decision(self):
        """Test that Literal type rejects invalid decision values."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ReflectorOutput(
                decision="invalid_decision",
                reasoning="This should fail validation.",
            )
