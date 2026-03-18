"""Unit tests for PipelineExecutor."""

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import pytest

from deep_research.agent.pipeline.config import AgentConfig, AgentType, PipelineConfig
from deep_research.agent.pipeline.executor import (
    ExecutionResult,
    PipelineExecutor,
    create_executor_for_pipeline,
)
from deep_research.agent.pipeline.protocols import (
    CustomPhase,
    PhaseInsertion,
    PipelineCustomization,
)
from deep_research.agent.tools.base import ResearchContext


@dataclass
class MockState:
    """Mock research state for testing."""

    decision: str | None = None
    status: str = "running"
    current_step_index: int = 0
    plan: Any = None
    agents_called: list[str] = field(default_factory=list)
    custom_data: dict[str, Any] = field(default_factory=dict)


def create_mock_context() -> ResearchContext:
    """Create mock research context."""
    return ResearchContext(
        chat_id=uuid4(),
        user_id="test-user",
        research_type="medium",
    )


def create_simple_pipeline() -> PipelineConfig:
    """Create a simple 2-agent pipeline for testing."""
    return PipelineConfig(
        name="test-simple",
        description="Simple test pipeline",
        agents=[
            AgentConfig(
                agent_type=AgentType.PLANNER,
                name="planner",
                next_on_success="synthesizer",
            ),
            AgentConfig(
                agent_type=AgentType.SYNTHESIZER,
                name="synthesizer",
                next_on_success=None,
            ),
        ],
        start_agent="planner",
        max_iterations=10,
        timeout_seconds=30,
    )


def create_loop_pipeline() -> PipelineConfig:
    """Create a pipeline with loop for testing."""
    return PipelineConfig(
        name="test-loop",
        description="Loop test pipeline",
        agents=[
            AgentConfig(
                agent_type=AgentType.RESEARCHER,
                name="researcher",
                loop_condition="CONTINUE",
                loop_back_to="researcher",
                next_on_success="synthesizer",
                max_iterations=3,
            ),
            AgentConfig(
                agent_type=AgentType.SYNTHESIZER,
                name="synthesizer",
                next_on_success=None,
            ),
        ],
        start_agent="researcher",
        max_iterations=10,
    )


class TestPipelineExecutorCreation:
    """Tests for PipelineExecutor creation."""

    def test_create_with_valid_config(self) -> None:
        """Should create executor with valid config."""
        config = create_simple_pipeline()
        executor = PipelineExecutor(config)
        assert executor.config == config

    def test_create_with_invalid_config_raises(self) -> None:
        """Should raise for invalid config."""
        config = PipelineConfig(
            name="invalid",
            description="Invalid",
            agents=[],
        )
        with pytest.raises(ValueError) as exc_info:
            PipelineExecutor(config)
        assert "Invalid pipeline" in str(exc_info.value)

    def test_create_with_customization(self) -> None:
        """Should apply customization."""
        config = create_simple_pipeline()
        customization = PipelineCustomization(
            disabled_agents={"planner"},
        )
        executor = PipelineExecutor(config, customization)
        assert "planner" not in executor.effective_agents
        assert "synthesizer" in executor.effective_agents


class TestPipelineExecutorRegistration:
    """Tests for agent and phase registration."""

    def test_register_single_agent(self) -> None:
        """Should register agent function."""
        config = create_simple_pipeline()
        executor = PipelineExecutor(config)

        async def mock_agent(state: MockState, config: dict) -> MockState:
            return state

        executor.register_agent("planner", mock_agent)
        assert "planner" in executor._agents

    def test_register_multiple_agents(self) -> None:
        """Should register multiple agent functions."""
        config = create_simple_pipeline()
        executor = PipelineExecutor(config)

        async def planner_fn(state: MockState, config: dict) -> MockState:
            return state

        async def synthesizer_fn(state: MockState, config: dict) -> MockState:
            return state

        executor.register_agents({
            "planner": planner_fn,
            "synthesizer": synthesizer_fn,
        })
        assert len(executor._agents) == 2

    def test_register_custom_phase(self) -> None:
        """Should register custom phase."""
        config = create_simple_pipeline()
        executor = PipelineExecutor(config)

        @dataclass
        class TestPhase:
            @property
            def name(self) -> str:
                return "test_phase"

            @property
            def description(self) -> str:
                return "Test phase"

            async def execute(
                self,
                context: Any,
                state: Any,
                config: dict[str, Any],
            ) -> Any:
                return state

        phase = TestPhase()
        executor.register_custom_phase(phase)
        assert "test_phase" in executor._custom_phases


class TestPipelineExecution:
    """Tests for pipeline execution."""

    @pytest.mark.asyncio
    async def test_simple_pipeline_execution(self) -> None:
        """Should execute simple linear pipeline."""
        config = create_simple_pipeline()
        executor = PipelineExecutor(config)

        async def planner_fn(state: MockState, config: dict) -> MockState:
            state.agents_called.append("planner")
            return state

        async def synthesizer_fn(state: MockState, config: dict) -> MockState:
            state.agents_called.append("synthesizer")
            return state

        executor.register_agents({
            "planner": planner_fn,
            "synthesizer": synthesizer_fn,
        })

        initial_state = MockState()
        context = create_mock_context()

        result = await executor.execute(initial_state, context)

        assert result.success
        assert result.iterations == 2
        assert "planner" in result.agents_executed
        assert "synthesizer" in result.agents_executed
        assert initial_state.agents_called == ["planner", "synthesizer"]

    @pytest.mark.asyncio
    async def test_loop_execution(self) -> None:
        """Should execute loop based on condition."""
        config = create_loop_pipeline()
        executor = PipelineExecutor(config)

        call_count = 0

        async def researcher_fn(state: MockState, config: dict) -> MockState:
            nonlocal call_count
            call_count += 1
            state.agents_called.append(f"researcher_{call_count}")
            # Set decision to CONTINUE for first 2 calls
            if call_count < 3:
                state.decision = "CONTINUE"
            else:
                state.decision = "COMPLETE"
            return state

        async def synthesizer_fn(state: MockState, config: dict) -> MockState:
            state.agents_called.append("synthesizer")
            return state

        executor.register_agents({
            "researcher": researcher_fn,
            "synthesizer": synthesizer_fn,
        })

        initial_state = MockState()
        context = create_mock_context()

        result = await executor.execute(initial_state, context)

        assert result.success
        # Should execute researcher 3 times (max_iterations per agent)
        # then synthesizer once
        assert call_count == 3
        assert result.agents_executed.count("researcher") == 3

    @pytest.mark.asyncio
    async def test_agent_failure_with_fallback(self) -> None:
        """Should transition to failure handler on error."""
        config = PipelineConfig(
            name="test-failure",
            description="Failure test",
            agents=[
                AgentConfig(
                    agent_type=AgentType.PLANNER,
                    next_on_success="synthesizer",
                    next_on_failure="fallback",
                ),
                AgentConfig(
                    agent_type="fallback",
                    name="fallback",
                    next_on_success=None,
                ),
                AgentConfig(
                    agent_type=AgentType.SYNTHESIZER,
                    next_on_success=None,
                ),
            ],
            start_agent="planner",
        )
        executor = PipelineExecutor(config)

        async def planner_fn(state: MockState, config: dict) -> MockState:
            raise RuntimeError("Planner failed!")

        async def fallback_fn(state: MockState, config: dict) -> MockState:
            state.agents_called.append("fallback")
            return state

        async def synthesizer_fn(state: MockState, config: dict) -> MockState:
            state.agents_called.append("synthesizer")
            return state

        executor.register_agents({
            "planner": planner_fn,
            "fallback": fallback_fn,
            "synthesizer": synthesizer_fn,
        })

        initial_state = MockState()
        context = create_mock_context()

        result = await executor.execute(initial_state, context)

        assert result.success
        assert "fallback" in result.agents_executed

    @pytest.mark.asyncio
    async def test_agent_failure_without_fallback(self) -> None:
        """Should return error if no fallback configured."""
        config = create_simple_pipeline()
        executor = PipelineExecutor(config)

        async def planner_fn(state: MockState, config: dict) -> MockState:
            raise RuntimeError("Planner failed!")

        executor.register_agent("planner", planner_fn)

        initial_state = MockState()
        context = create_mock_context()

        result = await executor.execute(initial_state, context)

        assert not result.success
        assert "failed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_iteration_limit(self) -> None:
        """Should stop at max_iterations."""
        config = PipelineConfig(
            name="test-limit",
            description="Limit test",
            agents=[
                AgentConfig(
                    agent_type=AgentType.RESEARCHER,
                    loop_condition="CONTINUE",
                    loop_back_to="researcher",
                    next_on_success=None,
                ),
            ],
            start_agent="researcher",
            max_iterations=5,
        )
        executor = PipelineExecutor(config)

        async def researcher_fn(state: MockState, config: dict) -> MockState:
            state.decision = "CONTINUE"  # Always continue
            state.agents_called.append("researcher")
            return state

        executor.register_agent("researcher", researcher_fn)

        initial_state = MockState()
        context = create_mock_context()

        result = await executor.execute(initial_state, context)

        assert result.success  # Completed but hit limit
        assert result.iterations == 5

    @pytest.mark.asyncio
    async def test_missing_agent_function(self) -> None:
        """Should return error for unregistered agent."""
        config = create_simple_pipeline()
        executor = PipelineExecutor(config)
        # Don't register any agents

        initial_state = MockState()
        context = create_mock_context()

        result = await executor.execute(initial_state, context)

        assert not result.success
        assert "not registered" in result.error.lower()

    @pytest.mark.asyncio
    async def test_stream_callback(self) -> None:
        """Should call stream callback during execution."""
        config = create_simple_pipeline()
        executor = PipelineExecutor(config)

        events: list[tuple[str, Any]] = []

        def callback(event_type: str, data: Any) -> None:
            events.append((event_type, data))

        async def planner_fn(state: MockState, config: dict) -> MockState:
            return state

        async def synthesizer_fn(state: MockState, config: dict) -> MockState:
            return state

        executor.register_agents({
            "planner": planner_fn,
            "synthesizer": synthesizer_fn,
        })

        initial_state = MockState()
        context = create_mock_context()

        await executor.execute(initial_state, context, stream_callback=callback)

        event_types = [e[0] for e in events]
        assert "agent_start" in event_types
        assert "agent_complete" in event_types


class TestCustomPhaseExecution:
    """Tests for custom phase execution."""

    @pytest.mark.asyncio
    async def test_phase_after_agent(self) -> None:
        """Should execute phase after agent."""
        config = create_simple_pipeline()
        customization = PipelineCustomization(
            phase_insertions=[
                PhaseInsertion(
                    phase_name="enrichment",
                    insert_after="planner",
                    config={"key": "value"},
                ),
            ],
        )
        executor = PipelineExecutor(config, customization)

        phase_called = False
        phase_config_received = {}

        @dataclass
        class EnrichmentPhase:
            @property
            def name(self) -> str:
                return "enrichment"

            @property
            def description(self) -> str:
                return "Enrichment phase"

            async def execute(
                self,
                context: Any,
                state: Any,
                config: dict[str, Any],
            ) -> Any:
                nonlocal phase_called, phase_config_received
                phase_called = True
                phase_config_received = config
                state.custom_data["enriched"] = True
                return state

        executor.register_custom_phase(EnrichmentPhase())

        async def planner_fn(state: MockState, config: dict) -> MockState:
            state.agents_called.append("planner")
            return state

        async def synthesizer_fn(state: MockState, config: dict) -> MockState:
            state.agents_called.append("synthesizer")
            return state

        executor.register_agents({
            "planner": planner_fn,
            "synthesizer": synthesizer_fn,
        })

        initial_state = MockState()
        context = create_mock_context()

        result = await executor.execute(initial_state, context)

        assert result.success
        assert phase_called
        assert phase_config_received == {"key": "value"}
        assert initial_state.custom_data.get("enriched") is True

    @pytest.mark.asyncio
    async def test_phase_before_agent(self) -> None:
        """Should execute phase before agent."""
        config = create_simple_pipeline()
        customization = PipelineCustomization(
            phase_insertions=[
                PhaseInsertion(
                    phase_name="preparation",
                    insert_before="synthesizer",
                ),
            ],
        )
        executor = PipelineExecutor(config, customization)

        execution_order: list[str] = []

        @dataclass
        class PrepPhase:
            @property
            def name(self) -> str:
                return "preparation"

            @property
            def description(self) -> str:
                return "Prep phase"

            async def execute(
                self,
                context: Any,
                state: Any,
                config: dict[str, Any],
            ) -> Any:
                execution_order.append("prep_phase")
                return state

        executor.register_custom_phase(PrepPhase())

        async def planner_fn(state: MockState, config: dict) -> MockState:
            execution_order.append("planner")
            return state

        async def synthesizer_fn(state: MockState, config: dict) -> MockState:
            execution_order.append("synthesizer")
            return state

        executor.register_agents({
            "planner": planner_fn,
            "synthesizer": synthesizer_fn,
        })

        initial_state = MockState()
        context = create_mock_context()

        await executor.execute(initial_state, context)

        assert execution_order == ["planner", "prep_phase", "synthesizer"]


class TestDisabledAgents:
    """Tests for disabled agent handling."""

    @pytest.mark.asyncio
    async def test_skips_disabled_agents(self) -> None:
        """Should skip disabled agents."""
        config = PipelineConfig(
            name="test-disable",
            description="Disable test",
            agents=[
                AgentConfig(
                    agent_type=AgentType.COORDINATOR,
                    next_on_success="planner",
                ),
                AgentConfig(
                    agent_type=AgentType.PLANNER,
                    next_on_success="synthesizer",
                ),
                AgentConfig(
                    agent_type=AgentType.SYNTHESIZER,
                    next_on_success=None,
                ),
            ],
            start_agent="coordinator",
        )
        customization = PipelineCustomization(
            disabled_agents={"coordinator"},
        )
        executor = PipelineExecutor(config, customization)

        # Coordinator is disabled, so planner should be first
        assert "coordinator" not in executor.effective_agents
        assert "planner" in executor.effective_agents

    @pytest.mark.asyncio
    async def test_all_disabled_returns_error(self) -> None:
        """Should error if all agents disabled."""
        config = create_simple_pipeline()
        customization = PipelineCustomization(
            disabled_agents={"planner", "synthesizer"},
        )
        executor = PipelineExecutor(config, customization)

        initial_state = MockState()
        context = create_mock_context()

        result = await executor.execute(initial_state, context)

        assert not result.success
        assert "No enabled agents" in result.error


class TestCreateExecutorForPipeline:
    """Tests for create_executor_for_pipeline function."""

    def test_create_deep_research(self) -> None:
        """Should create executor for deep-research pipeline."""
        executor = create_executor_for_pipeline("deep-research")
        assert executor.config.name == "deep-research"

    def test_create_with_customization(self) -> None:
        """Should apply customization."""
        customization = PipelineCustomization(
            disabled_agents={"background"},
        )
        executor = create_executor_for_pipeline("deep-research", customization)
        assert "background" not in executor.effective_agents

    def test_unknown_pipeline_raises(self) -> None:
        """Should raise for unknown pipeline."""
        with pytest.raises(ValueError):
            create_executor_for_pipeline("nonexistent")


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_successful_result(self) -> None:
        """Should create successful result."""
        state = MockState()
        result = ExecutionResult(
            state=state,
            success=True,
            iterations=5,
            execution_time=2.5,
            agents_executed=["planner", "researcher", "synthesizer"],
        )
        assert result.success
        assert result.error is None
        assert result.iterations == 5
        assert result.execution_time == 2.5
        assert len(result.agents_executed) == 3

    def test_failed_result(self) -> None:
        """Should create failed result."""
        state = MockState()
        result = ExecutionResult(
            state=state,
            success=False,
            error="Agent failed: timeout",
            iterations=2,
        )
        assert not result.success
        assert "timeout" in result.error
