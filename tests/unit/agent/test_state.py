"""Unit tests for ResearchState."""

import pytest
from datetime import UTC, datetime

from deep_research.agent.state import (
    ResearchState,
    StepStatus,
    Plan,
    PlanStep,
    StepType,
)


class TestResearchStateComplete:
    """Tests for ResearchState.complete() method."""

    def test_complete_with_valid_report(self) -> None:
        """Test that complete() sets final_report and completed_at."""
        state = ResearchState(query="test query")

        state.complete("This is a valid report.")

        assert state.final_report == "This is a valid report."
        assert state.completed_at is not None
        assert isinstance(state.completed_at, datetime)

    def test_complete_with_empty_string_raises(self) -> None:
        """Test that complete() raises ValueError for empty string."""
        state = ResearchState(query="test query")

        with pytest.raises(ValueError, match="Cannot complete research with empty report"):
            state.complete("")

    def test_complete_with_whitespace_only_raises(self) -> None:
        """Test that complete() raises ValueError for whitespace-only content."""
        state = ResearchState(query="test query")

        with pytest.raises(ValueError, match="Cannot complete research with empty report"):
            state.complete("   ")

        with pytest.raises(ValueError, match="Cannot complete research with empty report"):
            state.complete("\n\t\n")

    def test_complete_with_none_raises(self) -> None:
        """Test that complete() raises for None (if somehow passed)."""
        state = ResearchState(query="test query")

        # Type checker would catch this, but runtime should handle it too
        with pytest.raises(ValueError):
            state.complete(None)  # type: ignore[arg-type]

    def test_complete_with_minimal_content(self) -> None:
        """Test that complete() accepts minimal non-whitespace content."""
        state = ResearchState(query="test query")

        state.complete("X")

        assert state.final_report == "X"
        assert state.completed_at is not None

    def test_complete_preserves_leading_trailing_whitespace(self) -> None:
        """Test that complete() preserves whitespace in valid content."""
        state = ResearchState(query="test query")

        state.complete("  Report with spaces  ")

        assert state.final_report == "  Report with spaces  "


class TestResearchStateCancel:
    """Tests for ResearchState.cancel() method."""

    def test_cancel_sets_cancelled_flag(self) -> None:
        """Test that cancel() sets is_cancelled to True."""
        state = ResearchState(query="test query")

        state.cancel()

        assert state.is_cancelled is True
        assert state.completed_at is not None


class TestResearchStatePlanSteps:
    """Tests for plan step management methods."""

    def test_get_current_step_with_no_plan(self) -> None:
        """Test get_current_step returns None when no plan exists."""
        state = ResearchState(query="test query")

        assert state.get_current_step() is None

    def test_get_current_step_with_plan(self) -> None:
        """Test get_current_step returns correct step."""
        state = ResearchState(query="test query")
        state.current_plan = Plan(
            id="plan-1",
            title="Test Plan",
            thought="Planning",
            steps=[
                PlanStep(
                    id="step-1",
                    title="Step 1",
                    description="First step",
                    step_type=StepType.RESEARCH,
                    needs_search=True,
                ),
                PlanStep(
                    id="step-2",
                    title="Step 2",
                    description="Second step",
                    step_type=StepType.ANALYSIS,
                    needs_search=False,
                ),
            ],
        )

        current = state.get_current_step()

        assert current is not None
        assert current.id == "step-1"

    def test_advance_step(self) -> None:
        """Test advance_step increments step index."""
        state = ResearchState(query="test query")

        assert state.current_step_index == 0
        state.advance_step()
        assert state.current_step_index == 1

    def test_mark_step_complete(self) -> None:
        """Test mark_step_complete updates step status and observation."""
        state = ResearchState(query="test query")
        state.current_plan = Plan(
            id="plan-1",
            title="Test Plan",
            thought="Planning",
            steps=[
                PlanStep(
                    id="step-1",
                    title="Step 1",
                    description="First step",
                    step_type=StepType.RESEARCH,
                    needs_search=True,
                ),
            ],
        )

        state.mark_step_complete("Found relevant information")

        step = state.get_current_step()
        assert step is not None
        assert step.status == StepStatus.COMPLETED
        assert step.observation == "Found relevant information"
        assert state.last_observation == "Found relevant information"
        assert len(state.all_observations) == 1

    def test_get_completed_steps(self) -> None:
        """Test get_completed_steps returns only completed steps."""
        state = ResearchState(query="test query")
        state.current_plan = Plan(
            id="plan-1",
            title="Test Plan",
            thought="Planning",
            steps=[
                PlanStep(
                    id="step-1",
                    title="Step 1",
                    description="First step",
                    step_type=StepType.RESEARCH,
                    needs_search=True,
                    status=StepStatus.COMPLETED,
                ),
                PlanStep(
                    id="step-2",
                    title="Step 2",
                    description="Second step",
                    step_type=StepType.ANALYSIS,
                    needs_search=False,
                    status=StepStatus.PENDING,
                ),
            ],
        )

        completed = state.get_completed_steps()

        assert len(completed) == 1
        assert completed[0].id == "step-1"


class TestResearchStateDepth:
    """Tests for research depth resolution."""

    def test_resolve_depth_explicit(self) -> None:
        """Test resolve_depth returns explicit depth when set."""
        state = ResearchState(query="test query", research_depth="extended")

        depth = state.resolve_depth()

        assert depth == "extended"
        assert state.effective_depth == "extended"

    def test_resolve_depth_auto_without_classification(self) -> None:
        """Test resolve_depth defaults to medium when auto and no classification."""
        state = ResearchState(query="test query", research_depth="auto")

        depth = state.resolve_depth()

        assert depth == "medium"

    def test_resolve_depth_caches_result(self) -> None:
        """Test resolve_depth caches effective_depth."""
        state = ResearchState(query="test query", research_depth="light")

        state.resolve_depth()
        state.research_depth = "extended"  # Change after resolution

        # Should still return cached value
        assert state.resolve_depth() == "light"
