"""Tests for WorkflowState.extract_output() method."""

from __future__ import annotations

from databricks_deep_research.workflow.state import WorkflowState


def _make_state(key: str, value: object) -> WorkflowState:
    state = WorkflowState(query="test")
    state.append("node", key, value)
    return state


class TestExtractOutputString:
    def test_returns_string_as_is(self) -> None:
        state = _make_state("report", "Hello, world!")
        assert state.extract_output("report") == "Hello, world!"


class TestExtractOutputPydanticModel:
    def test_extracts_report_field(self) -> None:
        class FakeModel:
            report = "The full report text."

        state = _make_state("report", FakeModel())
        assert state.extract_output("report") == "The full report text."

    def test_extracts_direct_response_field(self) -> None:
        class FakeCoordinator:
            direct_response = "Quick answer."

        state = _make_state("coordination", FakeCoordinator())
        assert state.extract_output("coordination") == "Quick answer."

    def test_extracts_summary_field(self) -> None:
        class FakeBackground:
            summary = "Background summary."

        state = _make_state("background", FakeBackground())
        assert state.extract_output("background") == "Background summary."

    def test_priority_order_report_before_summary(self) -> None:
        class FakeModel:
            report = "The report."
            summary = "The summary."

        state = _make_state("out", FakeModel())
        assert state.extract_output("out") == "The report."


class TestExtractOutputDict:
    def test_extracts_from_dict(self) -> None:
        state = _make_state("report", {"report": "Dict report text."})
        assert state.extract_output("report") == "Dict report text."

    def test_skips_empty_string_values(self) -> None:
        state = _make_state("out", {"report": "", "summary": "Fallback."})
        assert state.extract_output("out") == "Fallback."


class TestExtractOutputMissing:
    def test_returns_none_for_missing_key(self) -> None:
        state = WorkflowState(query="test")
        assert state.extract_output("nonexistent") is None


class TestExtractOutputFallback:
    def test_falls_back_to_str(self) -> None:
        state = _make_state("out", 42)
        assert state.extract_output("out") == "42"

    def test_unknown_object_str_repr(self) -> None:
        class Custom:
            x = 1

            def __str__(self) -> str:
                return "Custom(x=1)"

        state = _make_state("out", Custom())
        assert state.extract_output("out") == "Custom(x=1)"
