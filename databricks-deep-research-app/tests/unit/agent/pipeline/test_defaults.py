"""Unit tests for default pipeline configurations."""

import pytest

from deep_research.agent.pipeline.config import AgentType
from deep_research.agent.pipeline.defaults import (
    DEFAULT_DEEP_RESEARCH_PIPELINE,
    REACT_LOOP_PIPELINE,
    SIMPLE_RESEARCH_PIPELINE,
    get_default_pipeline,
    list_pipelines,
)


class TestDefaultDeepResearchPipeline:
    """Tests for DEFAULT_DEEP_RESEARCH_PIPELINE."""

    def test_valid_configuration(self) -> None:
        """Pipeline should be valid."""
        assert DEFAULT_DEEP_RESEARCH_PIPELINE.is_valid()

    def test_has_all_five_agents(self) -> None:
        """Should have all 5 agents."""
        names = DEFAULT_DEEP_RESEARCH_PIPELINE.get_agent_names()
        assert "coordinator" in names
        assert "background" in names
        assert "planner" in names
        assert "researcher" in names
        assert "reflector" in names
        assert "synthesizer" in names

    def test_starts_with_coordinator(self) -> None:
        """Should start with coordinator."""
        assert DEFAULT_DEEP_RESEARCH_PIPELINE.start_agent == "coordinator"

    def test_coordinator_transitions_to_background(self) -> None:
        """Coordinator should transition to background."""
        coordinator = DEFAULT_DEEP_RESEARCH_PIPELINE.get_agent_config("coordinator")
        assert coordinator is not None
        assert coordinator.next_on_success == "background"

    def test_reflector_has_loop_back(self) -> None:
        """Reflector should loop back to researcher."""
        reflector = DEFAULT_DEEP_RESEARCH_PIPELINE.get_agent_config("reflector")
        assert reflector is not None
        assert reflector.loop_condition == "CONTINUE"
        assert reflector.loop_back_to == "researcher"

    def test_synthesizer_ends_pipeline(self) -> None:
        """Synthesizer should end the pipeline."""
        synthesizer = DEFAULT_DEEP_RESEARCH_PIPELINE.get_agent_config("synthesizer")
        assert synthesizer is not None
        assert synthesizer.next_on_success is None


class TestSimpleResearchPipeline:
    """Tests for SIMPLE_RESEARCH_PIPELINE."""

    def test_valid_configuration(self) -> None:
        """Pipeline should be valid."""
        assert SIMPLE_RESEARCH_PIPELINE.is_valid()

    def test_has_three_agents(self) -> None:
        """Should have 3 agents."""
        names = SIMPLE_RESEARCH_PIPELINE.get_agent_names()
        assert len(names) == 3
        assert "planner" in names
        assert "researcher" in names
        assert "synthesizer" in names

    def test_starts_with_planner(self) -> None:
        """Should start with planner."""
        assert SIMPLE_RESEARCH_PIPELINE.start_agent == "planner"

    def test_researcher_has_step_loop(self) -> None:
        """Researcher should have step completion loop."""
        researcher = SIMPLE_RESEARCH_PIPELINE.get_agent_config("researcher")
        assert researcher is not None
        assert researcher.loop_condition == "step_incomplete"
        assert researcher.loop_back_to == "researcher"


class TestReactLoopPipeline:
    """Tests for REACT_LOOP_PIPELINE."""

    def test_valid_configuration(self) -> None:
        """Pipeline should be valid."""
        assert REACT_LOOP_PIPELINE.is_valid()

    def test_has_two_agents(self) -> None:
        """Should have 2 agents."""
        names = REACT_LOOP_PIPELINE.get_agent_names()
        assert len(names) == 2
        assert "researcher" in names
        assert "synthesizer" in names

    def test_starts_with_researcher(self) -> None:
        """Should start with researcher."""
        assert REACT_LOOP_PIPELINE.start_agent == "researcher"

    def test_researcher_has_react_config(self) -> None:
        """Researcher should be in ReAct mode."""
        researcher = REACT_LOOP_PIPELINE.get_agent_config("researcher")
        assert researcher is not None
        assert researcher.config.get("mode") == "react"

    def test_researcher_has_higher_limits(self) -> None:
        """ReAct researcher should have higher limits."""
        researcher = REACT_LOOP_PIPELINE.get_agent_config("researcher")
        assert researcher is not None
        assert researcher.max_iterations == 20
        assert researcher.config.get("max_tool_calls") == 15


class TestGetDefaultPipeline:
    """Tests for get_default_pipeline function."""

    def test_get_deep_research(self) -> None:
        """Should return deep-research pipeline."""
        pipeline = get_default_pipeline("deep-research")
        assert pipeline.name == "deep-research"

    def test_get_simple_research(self) -> None:
        """Should return simple-research pipeline."""
        pipeline = get_default_pipeline("simple-research")
        assert pipeline.name == "simple-research"

    def test_get_react_loop(self) -> None:
        """Should return react-loop pipeline."""
        pipeline = get_default_pipeline("react-loop")
        assert pipeline.name == "react-loop"

    def test_default_is_deep_research(self) -> None:
        """Default should be deep-research."""
        pipeline = get_default_pipeline()
        assert pipeline.name == "deep-research"

    def test_unknown_pipeline_raises(self) -> None:
        """Should raise for unknown pipeline name."""
        with pytest.raises(ValueError) as exc_info:
            get_default_pipeline("nonexistent")
        assert "Unknown pipeline" in str(exc_info.value)
        assert "Available pipelines" in str(exc_info.value)


class TestListPipelines:
    """Tests for list_pipelines function."""

    def test_returns_all_pipelines(self) -> None:
        """Should return all available pipeline names."""
        pipelines = list_pipelines()
        assert "deep-research" in pipelines
        assert "simple-research" in pipelines
        assert "react-loop" in pipelines

    def test_returns_list(self) -> None:
        """Should return a list."""
        pipelines = list_pipelines()
        assert isinstance(pipelines, list)
        assert len(pipelines) >= 3
