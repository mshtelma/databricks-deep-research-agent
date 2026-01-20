"""Unit tests for pipeline configuration classes."""

import pytest

from deep_research.agent.pipeline.config import (
    AgentConfig,
    AgentType,
    PipelineConfig,
)


class TestAgentType:
    """Tests for AgentType enum."""

    def test_agent_type_values(self) -> None:
        """AgentType should have expected values."""
        assert AgentType.COORDINATOR.value == "coordinator"
        assert AgentType.BACKGROUND.value == "background"
        assert AgentType.PLANNER.value == "planner"
        assert AgentType.RESEARCHER.value == "researcher"
        assert AgentType.REFLECTOR.value == "reflector"
        assert AgentType.SYNTHESIZER.value == "synthesizer"
        assert AgentType.CUSTOM.value == "custom"

    def test_agent_type_str(self) -> None:
        """AgentType should stringify to value."""
        assert str(AgentType.COORDINATOR) == "coordinator"
        assert str(AgentType.PLANNER) == "planner"


class TestAgentConfig:
    """Tests for AgentConfig dataclass."""

    def test_minimal_config(self) -> None:
        """Should create config with just agent_type."""
        config = AgentConfig(agent_type=AgentType.PLANNER)
        assert config.agent_type == "planner"
        assert config.name == "planner"  # Default from agent_type
        assert config.enabled is True
        assert config.model_tier == "analytical"
        assert config.config == {}

    def test_name_defaults_to_agent_type(self) -> None:
        """Name should default to agent_type value."""
        config = AgentConfig(agent_type=AgentType.RESEARCHER)
        assert config.name == "researcher"

    def test_custom_name(self) -> None:
        """Should allow custom name."""
        config = AgentConfig(
            agent_type=AgentType.RESEARCHER,
            name="custom_researcher",
        )
        assert config.name == "custom_researcher"

    def test_string_agent_type(self) -> None:
        """Should accept string agent type."""
        config = AgentConfig(agent_type="my_custom_agent")
        assert config.agent_type == "my_custom_agent"
        assert config.name == "my_custom_agent"

    def test_transition_config(self) -> None:
        """Should store transition configuration."""
        config = AgentConfig(
            agent_type=AgentType.RESEARCHER,
            next_on_success="reflector",
            next_on_failure="planner",
        )
        assert config.next_on_success == "reflector"
        assert config.next_on_failure == "planner"

    def test_loop_config(self) -> None:
        """Should store loop configuration."""
        config = AgentConfig(
            agent_type=AgentType.REFLECTOR,
            loop_condition="CONTINUE",
            loop_back_to="researcher",
        )
        assert config.loop_condition == "CONTINUE"
        assert config.loop_back_to == "researcher"

    def test_get_agent_type_enum_builtin(self) -> None:
        """Should return enum for built-in types."""
        config = AgentConfig(agent_type=AgentType.PLANNER)
        assert config.get_agent_type_enum() == AgentType.PLANNER

    def test_get_agent_type_enum_custom(self) -> None:
        """Should return None for custom types."""
        config = AgentConfig(agent_type="my_custom_agent")
        assert config.get_agent_type_enum() is None

    def test_config_dict(self) -> None:
        """Should store agent-specific config."""
        config = AgentConfig(
            agent_type=AgentType.RESEARCHER,
            config={"max_search_queries": 5, "timeout": 30},
        )
        assert config.config["max_search_queries"] == 5
        assert config.config["timeout"] == 30


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_minimal_config(self) -> None:
        """Should create config with required fields."""
        config = PipelineConfig(
            name="test-pipeline",
            description="Test pipeline",
            agents=[
                AgentConfig(agent_type=AgentType.PLANNER),
            ],
            start_agent="planner",
        )
        assert config.name == "test-pipeline"
        assert config.description == "Test pipeline"
        assert len(config.agents) == 1
        assert config.start_agent == "planner"

    def test_defaults(self) -> None:
        """Should have sensible defaults."""
        config = PipelineConfig(
            name="test",
            description="Test",
            agents=[AgentConfig(agent_type=AgentType.PLANNER)],
        )
        assert config.start_agent == "coordinator"
        assert config.max_iterations == 15
        assert config.timeout_seconds == 300

    def test_get_agent_config_found(self) -> None:
        """Should find agent config by name."""
        planner = AgentConfig(agent_type=AgentType.PLANNER)
        researcher = AgentConfig(agent_type=AgentType.RESEARCHER)
        config = PipelineConfig(
            name="test",
            description="Test",
            agents=[planner, researcher],
        )
        found = config.get_agent_config("planner")
        assert found is planner

    def test_get_agent_config_not_found(self) -> None:
        """Should return None for missing agent."""
        config = PipelineConfig(
            name="test",
            description="Test",
            agents=[AgentConfig(agent_type=AgentType.PLANNER)],
        )
        assert config.get_agent_config("missing") is None

    def test_get_agent_names(self) -> None:
        """Should return list of agent names."""
        config = PipelineConfig(
            name="test",
            description="Test",
            agents=[
                AgentConfig(agent_type=AgentType.PLANNER),
                AgentConfig(agent_type=AgentType.RESEARCHER),
                AgentConfig(agent_type=AgentType.SYNTHESIZER),
            ],
        )
        names = config.get_agent_names()
        assert names == ["planner", "researcher", "synthesizer"]

    def test_get_enabled_agents(self) -> None:
        """Should filter to enabled agents."""
        config = PipelineConfig(
            name="test",
            description="Test",
            agents=[
                AgentConfig(agent_type=AgentType.PLANNER, enabled=True),
                AgentConfig(agent_type=AgentType.RESEARCHER, enabled=False),
                AgentConfig(agent_type=AgentType.SYNTHESIZER, enabled=True),
            ],
        )
        enabled = config.get_enabled_agents()
        assert len(enabled) == 2
        assert enabled[0].name == "planner"
        assert enabled[1].name == "synthesizer"

    def test_to_dict(self) -> None:
        """Should serialize to dictionary."""
        config = PipelineConfig(
            name="test-pipeline",
            description="Test",
            agents=[
                AgentConfig(
                    agent_type=AgentType.PLANNER,
                    next_on_success="researcher",
                ),
            ],
            start_agent="planner",
            max_iterations=10,
        )
        result = config.to_dict()
        assert result["name"] == "test-pipeline"
        assert result["max_iterations"] == 10
        assert len(result["agents"]) == 1
        assert result["agents"][0]["agent_type"] == "planner"
        assert result["agents"][0]["next_on_success"] == "researcher"


class TestPipelineConfigValidation:
    """Tests for PipelineConfig validation."""

    def test_valid_simple_pipeline(self) -> None:
        """Should validate simple linear pipeline."""
        config = PipelineConfig(
            name="simple",
            description="Simple pipeline",
            agents=[
                AgentConfig(
                    agent_type=AgentType.PLANNER,
                    next_on_success="researcher",
                ),
                AgentConfig(
                    agent_type=AgentType.RESEARCHER,
                    next_on_success="synthesizer",
                ),
                AgentConfig(
                    agent_type=AgentType.SYNTHESIZER,
                    next_on_success=None,
                ),
            ],
            start_agent="planner",
        )
        errors = config.validate()
        assert errors == []
        assert config.is_valid() is True

    def test_valid_loop_pipeline(self) -> None:
        """Should validate pipeline with loops."""
        config = PipelineConfig(
            name="loop",
            description="Loop pipeline",
            agents=[
                AgentConfig(
                    agent_type=AgentType.RESEARCHER,
                    loop_condition="CONTINUE",
                    loop_back_to="researcher",
                    next_on_success="synthesizer",
                ),
                AgentConfig(
                    agent_type=AgentType.SYNTHESIZER,
                    next_on_success=None,
                ),
            ],
            start_agent="researcher",
        )
        errors = config.validate()
        assert errors == []

    def test_invalid_empty_agents(self) -> None:
        """Should reject empty agents list."""
        config = PipelineConfig(
            name="empty",
            description="Empty",
            agents=[],
        )
        errors = config.validate()
        assert any("at least one agent" in e for e in errors)
        assert config.is_valid() is False

    def test_invalid_duplicate_names(self) -> None:
        """Should reject duplicate agent names."""
        config = PipelineConfig(
            name="duplicate",
            description="Duplicate",
            agents=[
                AgentConfig(agent_type=AgentType.PLANNER, name="agent1"),
                AgentConfig(agent_type=AgentType.RESEARCHER, name="agent1"),
            ],
        )
        errors = config.validate()
        assert any("Duplicate agent name" in e for e in errors)

    def test_invalid_start_agent(self) -> None:
        """Should reject missing start_agent."""
        config = PipelineConfig(
            name="bad-start",
            description="Bad start",
            agents=[
                AgentConfig(agent_type=AgentType.PLANNER),
            ],
            start_agent="missing",
        )
        errors = config.validate()
        assert any("start_agent" in e and "not found" in e for e in errors)

    def test_invalid_next_on_success(self) -> None:
        """Should reject invalid next_on_success reference."""
        config = PipelineConfig(
            name="bad-transition",
            description="Bad transition",
            agents=[
                AgentConfig(
                    agent_type=AgentType.PLANNER,
                    next_on_success="nonexistent",
                ),
            ],
            start_agent="planner",
        )
        errors = config.validate()
        assert any("invalid next_on_success" in e for e in errors)

    def test_invalid_next_on_failure(self) -> None:
        """Should reject invalid next_on_failure reference."""
        config = PipelineConfig(
            name="bad-failure",
            description="Bad failure",
            agents=[
                AgentConfig(
                    agent_type=AgentType.PLANNER,
                    next_on_failure="nonexistent",
                ),
            ],
            start_agent="planner",
        )
        errors = config.validate()
        assert any("invalid next_on_failure" in e for e in errors)

    def test_invalid_loop_back_without_condition(self) -> None:
        """Should reject loop_back_to without loop_condition."""
        config = PipelineConfig(
            name="bad-loop",
            description="Bad loop",
            agents=[
                AgentConfig(
                    agent_type=AgentType.RESEARCHER,
                    loop_back_to="researcher",
                    # Missing loop_condition
                ),
            ],
            start_agent="researcher",
        )
        errors = config.validate()
        assert any("loop_back_to without loop_condition" in e for e in errors)

    def test_invalid_loop_back_reference(self) -> None:
        """Should reject invalid loop_back_to reference."""
        config = PipelineConfig(
            name="bad-loop-ref",
            description="Bad loop reference",
            agents=[
                AgentConfig(
                    agent_type=AgentType.RESEARCHER,
                    loop_condition="CONTINUE",
                    loop_back_to="nonexistent",
                ),
            ],
            start_agent="researcher",
        )
        errors = config.validate()
        assert any("invalid loop_back_to" in e for e in errors)

    def test_invalid_max_iterations(self) -> None:
        """Should reject invalid max_iterations."""
        config = PipelineConfig(
            name="bad-limits",
            description="Bad limits",
            agents=[AgentConfig(agent_type=AgentType.PLANNER)],
            start_agent="planner",
            max_iterations=0,
        )
        errors = config.validate()
        assert any("max_iterations" in e for e in errors)

    def test_invalid_timeout(self) -> None:
        """Should reject invalid timeout_seconds."""
        config = PipelineConfig(
            name="bad-timeout",
            description="Bad timeout",
            agents=[AgentConfig(agent_type=AgentType.PLANNER)],
            start_agent="planner",
            timeout_seconds=0,
        )
        errors = config.validate()
        assert any("timeout_seconds" in e for e in errors)
