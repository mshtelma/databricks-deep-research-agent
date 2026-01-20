"""Unit tests for pipeline customization protocols."""

from dataclasses import dataclass
from typing import Any

import pytest

from deep_research.agent.pipeline.config import AgentConfig, AgentType
from deep_research.agent.pipeline.protocols import (
    CustomPhase,
    PhaseInsertion,
    PhaseProvider,
    PipelineCustomization,
    PipelineCustomizer,
)


class TestPhaseInsertion:
    """Tests for PhaseInsertion dataclass."""

    def test_create_with_insert_after(self) -> None:
        """Should create with insert_after."""
        insertion = PhaseInsertion(
            phase_name="enrichment",
            insert_after="researcher",
        )
        assert insertion.phase_name == "enrichment"
        assert insertion.insert_after == "researcher"
        assert insertion.insert_before is None
        assert insertion.enabled is True
        assert insertion.config == {}

    def test_create_with_insert_before(self) -> None:
        """Should create with insert_before."""
        insertion = PhaseInsertion(
            phase_name="validation",
            insert_before="synthesizer",
        )
        assert insertion.phase_name == "validation"
        assert insertion.insert_before == "synthesizer"
        assert insertion.insert_after is None

    def test_create_with_config(self) -> None:
        """Should store phase config."""
        insertion = PhaseInsertion(
            phase_name="custom",
            insert_after="researcher",
            config={"max_retries": 3, "timeout": 30},
        )
        assert insertion.config["max_retries"] == 3
        assert insertion.config["timeout"] == 30

    def test_invalid_both_before_and_after(self) -> None:
        """Should reject both insert_before and insert_after."""
        with pytest.raises(ValueError) as exc_info:
            PhaseInsertion(
                phase_name="invalid",
                insert_before="synthesizer",
                insert_after="researcher",
            )
        assert "cannot specify both" in str(exc_info.value)

    def test_invalid_neither_before_nor_after(self) -> None:
        """Should reject missing insert_before and insert_after."""
        with pytest.raises(ValueError) as exc_info:
            PhaseInsertion(
                phase_name="invalid",
            )
        assert "must specify either" in str(exc_info.value)

    def test_disabled_insertion(self) -> None:
        """Should support disabled insertions."""
        insertion = PhaseInsertion(
            phase_name="optional",
            insert_after="researcher",
            enabled=False,
        )
        assert insertion.enabled is False


class TestPipelineCustomization:
    """Tests for PipelineCustomization dataclass."""

    def test_empty_customization(self) -> None:
        """Should create empty customization."""
        customization = PipelineCustomization()
        assert customization.agent_overrides == {}
        assert customization.disabled_agents == set()
        assert customization.phase_insertions == []
        assert customization.pipeline_name is None
        assert customization.config_overrides == {}

    def test_with_agent_overrides(self) -> None:
        """Should store agent overrides."""
        customization = PipelineCustomization(
            agent_overrides={
                "researcher": {"model_tier": "complex", "max_iterations": 5},
                "planner": {"enabled": False},
            },
        )
        assert "researcher" in customization.agent_overrides
        assert customization.agent_overrides["researcher"]["model_tier"] == "complex"

    def test_with_disabled_agents(self) -> None:
        """Should store disabled agents."""
        customization = PipelineCustomization(
            disabled_agents={"background", "reflector"},
        )
        assert "background" in customization.disabled_agents
        assert "reflector" in customization.disabled_agents

    def test_with_phase_insertions(self) -> None:
        """Should store phase insertions."""
        customization = PipelineCustomization(
            phase_insertions=[
                PhaseInsertion(phase_name="phase1", insert_after="researcher"),
                PhaseInsertion(phase_name="phase2", insert_before="synthesizer"),
            ],
        )
        assert len(customization.phase_insertions) == 2

    def test_is_agent_disabled(self) -> None:
        """Should check if agent is disabled."""
        customization = PipelineCustomization(
            disabled_agents={"background"},
        )
        assert customization.is_agent_disabled("background") is True
        assert customization.is_agent_disabled("researcher") is False

    def test_apply_to_agent_with_overrides(self) -> None:
        """Should apply overrides to agent config."""
        customization = PipelineCustomization(
            agent_overrides={
                "researcher": {"model_tier": "complex"},
            },
        )
        original = AgentConfig(
            agent_type=AgentType.RESEARCHER,
            model_tier="analytical",
        )
        modified = customization.apply_to_agent(original)

        # Original unchanged
        assert original.model_tier == "analytical"
        # Modified has override
        assert modified.model_tier == "complex"
        # Other fields preserved
        assert modified.agent_type == "researcher"

    def test_apply_to_agent_without_overrides(self) -> None:
        """Should return same config if no overrides."""
        customization = PipelineCustomization()
        original = AgentConfig(agent_type=AgentType.PLANNER)
        result = customization.apply_to_agent(original)
        assert result is original

    def test_get_phase_after(self) -> None:
        """Should get phases to insert after agent."""
        customization = PipelineCustomization(
            phase_insertions=[
                PhaseInsertion(phase_name="phase1", insert_after="researcher"),
                PhaseInsertion(phase_name="phase2", insert_after="researcher"),
                PhaseInsertion(phase_name="phase3", insert_after="synthesizer"),
                PhaseInsertion(
                    phase_name="phase4", insert_after="researcher", enabled=False
                ),
            ],
        )
        phases = customization.get_phase_after("researcher")
        assert len(phases) == 2
        assert phases[0].phase_name == "phase1"
        assert phases[1].phase_name == "phase2"

    def test_get_phase_before(self) -> None:
        """Should get phases to insert before agent."""
        customization = PipelineCustomization(
            phase_insertions=[
                PhaseInsertion(phase_name="phase1", insert_before="synthesizer"),
                PhaseInsertion(phase_name="phase2", insert_before="synthesizer"),
                PhaseInsertion(phase_name="phase3", insert_before="planner"),
            ],
        )
        phases = customization.get_phase_before("synthesizer")
        assert len(phases) == 2
        assert phases[0].phase_name == "phase1"
        assert phases[1].phase_name == "phase2"


class TestCustomPhaseProtocol:
    """Tests for CustomPhase protocol."""

    def test_implements_protocol(self) -> None:
        """Class implementing protocol should be recognized."""

        @dataclass
        class TestPhase:
            @property
            def name(self) -> str:
                return "test_phase"

            @property
            def description(self) -> str:
                return "A test phase"

            async def execute(
                self,
                context: Any,
                state: Any,
                config: dict[str, Any],
            ) -> Any:
                return state

        phase = TestPhase()
        assert isinstance(phase, CustomPhase)

    def test_not_implements_protocol(self) -> None:
        """Class missing methods should not be recognized."""

        class IncompletePhase:
            @property
            def name(self) -> str:
                return "incomplete"

            # Missing description and execute

        phase = IncompletePhase()
        assert not isinstance(phase, CustomPhase)


class TestPipelineCustomizerProtocol:
    """Tests for PipelineCustomizer protocol."""

    def test_implements_protocol(self) -> None:
        """Class implementing protocol should be recognized."""

        class TestCustomizer:
            def get_pipeline_customization(self) -> PipelineCustomization | None:
                return PipelineCustomization(
                    disabled_agents={"background"},
                )

        customizer = TestCustomizer()
        assert isinstance(customizer, PipelineCustomizer)

    def test_returns_none(self) -> None:
        """Protocol allows returning None."""

        class NullCustomizer:
            def get_pipeline_customization(self) -> PipelineCustomization | None:
                return None

        customizer = NullCustomizer()
        assert isinstance(customizer, PipelineCustomizer)
        assert customizer.get_pipeline_customization() is None


class TestPhaseProviderProtocol:
    """Tests for PhaseProvider protocol."""

    def test_implements_protocol(self) -> None:
        """Class implementing protocol should be recognized."""

        @dataclass
        class DummyPhase:
            @property
            def name(self) -> str:
                return "dummy"

            @property
            def description(self) -> str:
                return "Dummy phase"

            async def execute(
                self,
                context: Any,
                state: Any,
                config: dict[str, Any],
            ) -> Any:
                return state

        class TestProvider:
            def get_custom_phases(self) -> list[CustomPhase]:
                return [DummyPhase()]

        provider = TestProvider()
        assert isinstance(provider, PhaseProvider)

    def test_returns_empty_list(self) -> None:
        """Protocol allows returning empty list."""

        class EmptyProvider:
            def get_custom_phases(self) -> list[CustomPhase]:
                return []

        provider = EmptyProvider()
        assert isinstance(provider, PhaseProvider)
        assert provider.get_custom_phases() == []
