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

    def test_complete_same_content_is_idempotent(self) -> None:
        """Test that complete() with same content is a no-op (not an error)."""
        state = ResearchState(query="test query")
        state.complete("The report")
        first_timestamp = state.completed_at

        # Same content — should NOT raise
        state.complete("The report")

        # State unchanged
        assert state.final_report == "The report"
        assert state.completed_at == first_timestamp

    def test_complete_different_content_raises(self) -> None:
        """Test that complete() with different content still raises RuntimeError."""
        state = ResearchState(query="test query")
        state.complete("First report")

        with pytest.raises(RuntimeError, match="already completed"):
            state.complete("Different report")

        # Original state preserved
        assert state.final_report == "First report"

    def test_complete_explicit_overwrite(self) -> None:
        """Test that allow_overwrite=True permits overwriting."""
        state = ResearchState(query="test query")

        state.complete("First report")
        original_timestamp = state.completed_at

        # Explicit overwrite succeeds
        state.complete("Second report", allow_overwrite=True)
        assert state.final_report == "Second report"
        assert state.completed_at >= original_timestamp  # Should be same or later


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


class TestFileContentHelpers:
    """Tests for file content prompt injection helpers."""

    def test_get_file_context_empty_when_no_files(self) -> None:
        """Returns empty string when no file_contents."""
        state = ResearchState(query="test")
        assert state.get_file_context_for_prompt() == ""

    def test_get_file_context_inline_file(self) -> None:
        """Inline files include full content."""
        state = ResearchState(query="test")
        state.file_contents = [
            {
                "file_id": "f1",
                "filename": "notes.txt",
                "strategy": "inline",
                "content": "Hello world",
            }
        ]
        result = state.get_file_context_for_prompt()
        assert "## Uploaded File Contents" in result
        assert '<uploaded_file name="notes.txt">' in result
        assert "Hello world" in result
        assert "</uploaded_file>" in result

    def test_get_file_context_hybrid_file(self) -> None:
        """Hybrid files show preview hint."""
        state = ResearchState(query="test")
        state.file_contents = [
            {
                "file_id": "f2",
                "filename": "report.pdf",
                "strategy": "hybrid",
                "content": "Preview text...",
            }
        ]
        result = state.get_file_context_for_prompt()
        assert '<uploaded_file name="report.pdf" mode="preview">' in result
        assert "Preview text..." in result

    def test_get_file_context_retrieval_file(self) -> None:
        """Retrieval files show search instruction."""
        state = ResearchState(query="test")
        state.file_contents = [
            {
                "file_id": "f3",
                "filename": "big.pdf",
                "strategy": "retrieval",
                "content": "[Large file]",
            }
        ]
        result = state.get_file_context_for_prompt()
        assert '<uploaded_file name="big.pdf" mode="retrieval">' in result
        assert "file_search" in result

    def test_get_file_context_max_chars_truncation(self) -> None:
        """Truncation when content exceeds max_chars."""
        state = ResearchState(query="test")
        state.file_contents = [
            {
                "file_id": "f1",
                "filename": "long.txt",
                "strategy": "inline",
                "content": "x" * 5000,
            }
        ]
        result = state.get_file_context_for_prompt(max_chars=500)
        assert "truncated due to size limit" in result
        assert len(result) < 600  # Should be under budget + overhead

    def test_get_file_context_max_chars_skips_when_budget_exhausted(self) -> None:
        """Second file skipped entirely when budget is exhausted."""
        state = ResearchState(query="test")
        state.file_contents = [
            {
                "file_id": "f1",
                "filename": "first.txt",
                "strategy": "inline",
                "content": "x" * 400,
            },
            {
                "file_id": "f2",
                "filename": "second.txt",
                "strategy": "inline",
                "content": "y" * 400,
            },
        ]
        result = state.get_file_context_for_prompt(max_chars=500)
        assert "first.txt" in result
        assert "second.txt" not in result

    def test_get_file_context_no_limit(self) -> None:
        """All files included when max_chars=0 (unlimited)."""
        state = ResearchState(query="test")
        state.file_contents = [
            {"file_id": "f1", "filename": "a.txt", "strategy": "inline", "content": "aaa"},
            {"file_id": "f2", "filename": "b.txt", "strategy": "inline", "content": "bbb"},
        ]
        result = state.get_file_context_for_prompt(max_chars=0)
        assert "a.txt" in result
        assert "b.txt" in result

    def test_has_inline_file_content_true(self) -> None:
        """Returns True when inline files present."""
        state = ResearchState(query="test")
        state.file_contents = [
            {"file_id": "f1", "strategy": "inline", "content": "data"}
        ]
        assert state.has_inline_file_content() is True

    def test_has_inline_file_content_true_for_hybrid(self) -> None:
        """Returns True when hybrid files present."""
        state = ResearchState(query="test")
        state.file_contents = [
            {"file_id": "f1", "strategy": "hybrid", "content": "data"}
        ]
        assert state.has_inline_file_content() is True

    def test_has_inline_file_content_false_for_retrieval_only(self) -> None:
        """Returns False when only retrieval files present."""
        state = ResearchState(query="test")
        state.file_contents = [
            {"file_id": "f1", "strategy": "retrieval", "content": ""}
        ]
        assert state.has_inline_file_content() is False

    def test_has_inline_file_content_false_when_empty(self) -> None:
        """Returns False when no files."""
        state = ResearchState(query="test")
        assert state.has_inline_file_content() is False
