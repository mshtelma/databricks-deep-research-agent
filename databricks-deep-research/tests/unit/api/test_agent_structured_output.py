"""Structured output (``output_type``) tests for ``Agent``."""

from __future__ import annotations

from pydantic import BaseModel

from databricks_deep_research.api import Agent
from databricks_deep_research.workflow.state import StateEntry, WorkflowState


class Report(BaseModel):
    title: str
    bullets: list[str]


def test_output_type_round_trip_via_build_result() -> None:
    agent = Agent(name="syn", output_type=Report)
    state = WorkflowState(query="x")
    state.append(
        node_id="syn",
        key="syn_output",
        value='{"title": "T", "bullets": ["b1", "b2"]}',
    )
    result = agent._build_result(state, events=[])
    assert isinstance(result.output, Report)
    assert result.output.title == "T"
    assert result.output.bullets == ["b1", "b2"]
    assert result.ok is True


def test_output_type_validation_failure_marks_not_ok() -> None:
    agent = Agent(name="syn", output_type=Report)
    state = WorkflowState(query="x")
    state.append(node_id="syn", key="syn_output", value="not json at all")
    result = agent._build_result(state, events=[])
    assert result.ok is False
    assert isinstance(result.output, str)


def test_no_output_type_returns_string_content() -> None:
    agent = Agent(name="agent")
    state = WorkflowState(query="x")
    state.append(node_id="agent", key="agent_output", value="plain text")
    result = agent._build_result(state, events=[])
    assert result.output == "plain text"
    assert result.ok is True


def test_synthesizer_subtype_attempts_verification_extraction() -> None:
    agent = Agent(name="syn", subtype="synthesizer")
    state = WorkflowState(query="x")
    state.append(node_id="syn", key="syn_output", value="report text [1]")
    state.enterprise_tools = [
        type("S", (), {"url": "http://a", "snippet": "s"})(),
    ]
    result = agent._build_result(state, events=[])
    # extract_verification_from_report fallback path runs since
    # state has no native ``claims`` artifact.
    assert result.verification is not None


def test_non_synthesizer_skips_verification() -> None:
    agent = Agent(name="r", subtype="custom")
    state = WorkflowState(query="x")
    state.append(node_id="r", key="r_output", value="text")
    result = agent._build_result(state, events=[])
    assert result.verification is None
