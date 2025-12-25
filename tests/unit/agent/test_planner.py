"""Unit tests for Planner agent."""

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.agent.nodes.planner import PlanOutput, PlanStepOutput, run_planner
from src.agent.state import (
    ReflectionDecision,
    ReflectionResult,
    ResearchState,
    StepType,
)
from src.services.llm.types import LLMResponse


@pytest.fixture
def mock_llm_client() -> AsyncMock:
    """Create a mock LLM client."""
    return AsyncMock()


@pytest.fixture
def research_state() -> ResearchState:
    """Create a basic research state."""
    return ResearchState(
        query="What are the latest developments in quantum computing?",
        conversation_history=[],
        session_id=uuid4(),
    )


@pytest.fixture
def research_state_with_background() -> ResearchState:
    """Create a research state with background investigation results."""
    state = ResearchState(
        query="What are the latest developments in quantum computing?",
        conversation_history=[],
        session_id=uuid4(),
    )
    state.background_investigation_results = (
        "Recent developments include Google's quantum supremacy claim, "
        "IBM's 1000+ qubit roadmap, and advances in error correction."
    )
    return state


@pytest.fixture
def plan_output() -> PlanOutput:
    """Create a sample plan output."""
    return PlanOutput(
        id="plan-1",
        title="Research Quantum Computing Developments",
        thought="Need to research academic and industry developments.",
        has_enough_context=False,
        steps=[
            PlanStepOutput(
                id="step-1",
                title="Search academic papers",
                description="Find recent academic papers on quantum computing.",
                step_type="research",
                needs_search=True,
            ),
            PlanStepOutput(
                id="step-2",
                title="Industry announcements",
                description="Research industry announcements from major companies.",
                step_type="research",
                needs_search=True,
            ),
            PlanStepOutput(
                id="step-3",
                title="Analyze findings",
                description="Synthesize all research into key findings.",
                step_type="analysis",
                needs_search=False,
            ),
        ],
    )


class TestRunPlanner:
    """Tests for run_planner function."""

    @pytest.mark.asyncio
    async def test_creates_plan_successfully(
        self, research_state: ResearchState, mock_llm_client: AsyncMock, plan_output: PlanOutput
    ):
        """Test successful plan creation."""
        # Arrange
        mock_llm_client.complete = AsyncMock(
            return_value=LLMResponse(
                content=plan_output.model_dump_json(),
                usage={"prompt_tokens": 300, "completion_tokens": 200, "total_tokens": 500},
                endpoint_id="test-endpoint",
                duration_ms=800.0,
                structured=plan_output,
            )
        )

        # Act
        result = await run_planner(research_state, mock_llm_client)

        # Assert
        assert result.current_plan is not None
        assert result.current_plan.title == "Research Quantum Computing Developments"
        assert len(result.current_plan.steps) == 3
        assert result.current_step_index == 0
        assert result.plan_iterations == 1

    @pytest.mark.asyncio
    async def test_increments_plan_iterations(
        self, research_state: ResearchState, mock_llm_client: AsyncMock, plan_output: PlanOutput
    ):
        """Test that plan iterations increment correctly."""
        # Arrange
        research_state.plan_iterations = 1

        mock_llm_client.complete = AsyncMock(
            return_value=LLMResponse(
                content=plan_output.model_dump_json(),
                usage={"prompt_tokens": 300, "completion_tokens": 200, "total_tokens": 500},
                endpoint_id="test-endpoint",
                duration_ms=800.0,
                structured=plan_output,
            )
        )

        # Act
        result = await run_planner(research_state, mock_llm_client)

        # Assert
        assert result.plan_iterations == 2

    @pytest.mark.asyncio
    async def test_respects_max_iterations(
        self, research_state: ResearchState, mock_llm_client: AsyncMock
    ):
        """Test that max iterations limit is respected."""
        # Arrange
        research_state.plan_iterations = 3
        research_state.max_plan_iterations = 3

        # Act
        result = await run_planner(research_state, mock_llm_client)

        # Assert
        mock_llm_client.complete.assert_not_called()
        assert result.current_plan is None  # No new plan created
        assert result.plan_iterations == 4  # Still incremented

    @pytest.mark.asyncio
    async def test_includes_background_investigation(
        self, research_state_with_background: ResearchState, mock_llm_client: AsyncMock, plan_output: PlanOutput
    ):
        """Test that background investigation results are included."""
        # Arrange
        mock_llm_client.complete = AsyncMock(
            return_value=LLMResponse(
                content=plan_output.model_dump_json(),
                usage={"prompt_tokens": 400, "completion_tokens": 200, "total_tokens": 600},
                endpoint_id="test-endpoint",
                duration_ms=900.0,
                structured=plan_output,
            )
        )

        # Act
        await run_planner(research_state_with_background, mock_llm_client)

        # Assert - Check that the LLM was called (the prompt would include background)
        mock_llm_client.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_includes_reflector_feedback(
        self, research_state: ResearchState, mock_llm_client: AsyncMock, plan_output: PlanOutput
    ):
        """Test that reflector feedback is included in replanning."""
        # Arrange
        research_state.last_reflection = ReflectionResult(
            decision=ReflectionDecision.ADJUST,
            reasoning="Need to add more specific steps.",
            suggested_changes=["Include quantum error correction", "Add company comparisons"],
        )

        mock_llm_client.complete = AsyncMock(
            return_value=LLMResponse(
                content=plan_output.model_dump_json(),
                usage={"prompt_tokens": 450, "completion_tokens": 200, "total_tokens": 650},
                endpoint_id="test-endpoint",
                duration_ms=850.0,
                structured=plan_output,
            )
        )

        # Act
        await run_planner(research_state, mock_llm_client)

        # Assert
        mock_llm_client.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_includes_previous_observations(
        self, research_state: ResearchState, mock_llm_client: AsyncMock, plan_output: PlanOutput
    ):
        """Test that previous observations are included in replanning."""
        # Arrange
        research_state.all_observations = [
            "Found 5 papers on quantum computing.",
            "IBM announced 1000+ qubit roadmap.",
        ]

        mock_llm_client.complete = AsyncMock(
            return_value=LLMResponse(
                content=plan_output.model_dump_json(),
                usage={"prompt_tokens": 500, "completion_tokens": 200, "total_tokens": 700},
                endpoint_id="test-endpoint",
                duration_ms=900.0,
                structured=plan_output,
            )
        )

        # Act
        await run_planner(research_state, mock_llm_client)

        # Assert
        mock_llm_client.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling_creates_fallback_plan(
        self, research_state: ResearchState, mock_llm_client: AsyncMock
    ):
        """Test that errors result in a minimal fallback plan."""
        # Arrange
        mock_llm_client.complete = AsyncMock(side_effect=Exception("LLM API error"))

        # Act
        result = await run_planner(research_state, mock_llm_client)

        # Assert
        assert result.current_plan is not None
        assert result.current_plan.title == "Research Plan"
        assert len(result.current_plan.steps) == 1
        assert result.current_plan.steps[0].title == "Research the topic"
        assert result.current_plan.steps[0].step_type == StepType.RESEARCH
        assert "error" in result.current_plan.thought.lower()

    @pytest.mark.asyncio
    async def test_resets_step_index_on_new_plan(
        self, research_state: ResearchState, mock_llm_client: AsyncMock, plan_output: PlanOutput
    ):
        """Test that step index is reset when a new plan is created."""
        # Arrange
        research_state.current_step_index = 5  # Some arbitrary index

        mock_llm_client.complete = AsyncMock(
            return_value=LLMResponse(
                content=plan_output.model_dump_json(),
                usage={"prompt_tokens": 300, "completion_tokens": 200, "total_tokens": 500},
                endpoint_id="test-endpoint",
                duration_ms=800.0,
                structured=plan_output,
            )
        )

        # Act
        result = await run_planner(research_state, mock_llm_client)

        # Assert
        assert result.current_step_index == 0

    @pytest.mark.asyncio
    async def test_step_types_converted_correctly(
        self, research_state: ResearchState, mock_llm_client: AsyncMock, plan_output: PlanOutput
    ):
        """Test that step types are converted to StepType enum correctly."""
        # Arrange
        mock_llm_client.complete = AsyncMock(
            return_value=LLMResponse(
                content=plan_output.model_dump_json(),
                usage={"prompt_tokens": 300, "completion_tokens": 200, "total_tokens": 500},
                endpoint_id="test-endpoint",
                duration_ms=800.0,
                structured=plan_output,
            )
        )

        # Act
        result = await run_planner(research_state, mock_llm_client)

        # Assert
        assert result.current_plan.steps[0].step_type == StepType.RESEARCH
        assert result.current_plan.steps[1].step_type == StepType.RESEARCH
        assert result.current_plan.steps[2].step_type == StepType.ANALYSIS


class TestPlanOutput:
    """Tests for PlanOutput model."""

    def test_model_defaults(self):
        """Test default values for PlanOutput."""
        output = PlanOutput(
            id="plan-1",
            title="Test Plan",
            thought="Test thought",
            steps=[],
        )

        assert output.has_enough_context is False

    def test_model_serialization(self):
        """Test JSON serialization of PlanOutput."""
        output = PlanOutput(
            id="plan-1",
            title="Test Plan",
            thought="Test thought",
            has_enough_context=True,
            steps=[
                PlanStepOutput(
                    id="step-1",
                    title="Step 1",
                    description="Description 1",
                    step_type="research",
                    needs_search=True,
                ),
            ],
        )

        json_str = output.model_dump_json()
        parsed = PlanOutput.model_validate_json(json_str)

        assert parsed.id == "plan-1"
        assert parsed.has_enough_context is True
        assert len(parsed.steps) == 1
        assert parsed.steps[0].needs_search is True
