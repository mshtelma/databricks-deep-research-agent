"""
Pipeline Executor
=================

Executes research pipelines based on declarative configuration.

The PipelineExecutor handles:
- Agent execution sequencing based on transitions
- Loop condition evaluation
- Custom phase insertion and execution
- Iteration and timeout limits
- Error handling and state management
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from deep_research.agent.pipeline.config import AgentConfig, AgentType, PipelineConfig
from deep_research.agent.pipeline.protocols import (
    CustomPhase,
    PhaseInsertion,
    PipelineCustomization,
)
from deep_research.core.logging_utils import get_logger

if TYPE_CHECKING:
    from deep_research.agent.state import ResearchState
    from deep_research.agent.tools.base import ResearchContext

logger = get_logger(__name__)


# Type alias for agent functions
AgentFunction = Callable[["ResearchState", dict[str, Any]], "ResearchState"]
AsyncAgentFunction = Callable[
    ["ResearchState", dict[str, Any]], "ResearchState"
]


@dataclass
class ExecutionResult:
    """Result of pipeline execution.

    Attributes:
        state: Final research state after execution
        success: Whether execution completed successfully
        error: Error message if execution failed
        iterations: Number of iterations executed
        execution_time: Total execution time in seconds
        agents_executed: List of agent names that were executed
    """

    state: "ResearchState"
    success: bool = True
    error: str | None = None
    iterations: int = 0
    execution_time: float = 0.0
    agents_executed: list[str] = field(default_factory=list)


class PipelineExecutor:
    """Executes research pipelines based on declarative configuration.

    The executor handles:
    - Agent execution based on PipelineConfig transitions
    - Loop condition evaluation (CONTINUE, ADJUST, step_incomplete, etc.)
    - Custom phase insertion before/after agents
    - Global iteration and timeout limits
    - Error isolation and recovery

    Example:
        >>> from deep_research.agent.pipeline import (
        ...     PipelineConfig,
        ...     AgentConfig,
        ...     AgentType,
        ... )
        >>>
        >>> config = PipelineConfig(
        ...     name="simple-pipeline",
        ...     description="Simple 2-agent pipeline",
        ...     agents=[
        ...         AgentConfig(
        ...             agent_type=AgentType.PLANNER,
        ...             next_on_success="researcher",
        ...         ),
        ...         AgentConfig(
        ...             agent_type=AgentType.RESEARCHER,
        ...             next_on_success=None,
        ...         ),
        ...     ],
        ...     start_agent="planner",
        ... )
        >>>
        >>> executor = PipelineExecutor(config)
        >>> executor.register_agent("planner", planner_function)
        >>> executor.register_agent("researcher", researcher_function)
        >>> result = await executor.execute(initial_state, context)
    """

    def __init__(
        self,
        config: PipelineConfig,
        customization: PipelineCustomization | None = None,
    ) -> None:
        """Initialize the pipeline executor.

        Args:
            config: Pipeline configuration
            customization: Optional customization to apply

        Raises:
            ValueError: If pipeline configuration is invalid
        """
        # Validate configuration
        errors = config.validate()
        if errors:
            raise ValueError(
                f"Invalid pipeline configuration: {'; '.join(errors)}"
            )

        self._config = config
        self._customization = customization or PipelineCustomization()
        self._agents: dict[str, AsyncAgentFunction] = {}
        self._custom_phases: dict[str, CustomPhase] = {}

        # Apply customization to build effective config
        self._effective_agents = self._build_effective_agents()

    def _build_effective_agents(self) -> dict[str, AgentConfig]:
        """Build effective agent configs after applying customization.

        Returns:
            Dictionary of agent name -> effective config
        """
        effective = {}
        for agent in self._config.agents:
            if agent.name is None:
                continue

            # Check if disabled
            if self._customization.is_agent_disabled(agent.name):
                logger.info("Agent disabled by customization", agent_name=agent.name)
                continue

            # Apply overrides
            effective_config = self._customization.apply_to_agent(agent)
            effective[agent.name] = effective_config

        return effective

    @property
    def config(self) -> PipelineConfig:
        """Get the pipeline configuration."""
        return self._config

    @property
    def effective_agents(self) -> dict[str, AgentConfig]:
        """Get effective agent configurations after customization."""
        return self._effective_agents

    def register_agent(
        self,
        name: str,
        function: AsyncAgentFunction,
    ) -> None:
        """Register an agent function.

        Args:
            name: Agent name (must match name in PipelineConfig)
            function: Async function to execute for this agent
        """
        if name not in self._effective_agents:
            available = list(self._effective_agents.keys())
            logger.warning(
                "Registering agent not in effective config",
                agent_name=name,
                available_agents=available,
            )
        self._agents[name] = function

    def register_agents(
        self,
        agents: dict[str, AsyncAgentFunction],
    ) -> None:
        """Register multiple agent functions.

        Args:
            agents: Dictionary of agent name -> function
        """
        for name, function in agents.items():
            self.register_agent(name, function)

    def register_custom_phase(self, phase: CustomPhase) -> None:
        """Register a custom phase.

        Args:
            phase: Custom phase implementation
        """
        self._custom_phases[phase.name] = phase

    def register_custom_phases(self, phases: list[CustomPhase]) -> None:
        """Register multiple custom phases.

        Args:
            phases: List of custom phase implementations
        """
        for phase in phases:
            self.register_custom_phase(phase)

    async def execute(
        self,
        state: "ResearchState",
        context: "ResearchContext",
        stream_callback: Callable[[str, Any], None] | None = None,
    ) -> ExecutionResult:
        """Execute the pipeline.

        Args:
            state: Initial research state
            context: Research context with tools and metadata
            stream_callback: Optional callback for streaming updates

        Returns:
            ExecutionResult with final state and execution metadata
        """
        start_time = time.monotonic()
        iterations = 0
        agents_executed: list[str] = []

        # Find starting agent
        current_agent = self._config.start_agent
        if current_agent not in self._effective_agents:
            # Find first enabled agent if start_agent is disabled
            for agent_name in self._effective_agents:
                current_agent = agent_name
                break
            else:
                return ExecutionResult(
                    state=state,
                    success=False,
                    error="No enabled agents in pipeline",
                )

        # Track per-agent iteration counts for loop limits
        agent_iterations: dict[str, int] = {}

        try:
            while current_agent and iterations < self._config.max_iterations:
                # Check timeout
                elapsed = time.monotonic() - start_time
                if elapsed > self._config.timeout_seconds:
                    logger.warning(
                        "Pipeline timeout reached",
                        timeout=self._config.timeout_seconds,
                        elapsed=elapsed,
                    )
                    return ExecutionResult(
                        state=state,
                        success=False,
                        error=f"Pipeline timeout after {elapsed:.1f}s",
                        iterations=iterations,
                        execution_time=elapsed,
                        agents_executed=agents_executed,
                    )

                # Get agent config
                agent_config = self._effective_agents.get(current_agent)
                if not agent_config:
                    logger.error("Agent not found", agent_name=current_agent)
                    return ExecutionResult(
                        state=state,
                        success=False,
                        error=f"Agent '{current_agent}' not found in config",
                        iterations=iterations,
                        execution_time=time.monotonic() - start_time,
                        agents_executed=agents_executed,
                    )

                # Check per-agent iteration limit
                agent_iterations[current_agent] = (
                    agent_iterations.get(current_agent, 0) + 1
                )
                if (
                    agent_config.max_iterations
                    and agent_iterations[current_agent] > agent_config.max_iterations
                ):
                    logger.warning(
                        "Agent iteration limit reached",
                        agent_name=current_agent,
                        max_iterations=agent_config.max_iterations,
                    )
                    # Move to next_on_success to exit the loop
                    current_agent = agent_config.next_on_success
                    continue

                # Execute custom phases BEFORE this agent
                for phase_config in self._customization.get_phase_before(
                    current_agent
                ):
                    state = await self._execute_phase(
                        phase_config, context, state, stream_callback
                    )

                # Get and execute agent function
                agent_fn = self._agents.get(current_agent)
                if not agent_fn:
                    logger.error(
                        "Agent function not registered",
                        agent_name=current_agent,
                    )
                    return ExecutionResult(
                        state=state,
                        success=False,
                        error=f"Agent '{current_agent}' function not registered",
                        iterations=iterations,
                        execution_time=time.monotonic() - start_time,
                        agents_executed=agents_executed,
                    )

                # Execute agent
                logger.info(
                    "Executing agent",
                    agent_name=current_agent,
                    iteration=iterations + 1,
                )
                if stream_callback:
                    stream_callback("agent_start", {"agent": current_agent})

                try:
                    state = await agent_fn(state, agent_config.config)
                    agents_executed.append(current_agent)
                    iterations += 1

                    if stream_callback:
                        stream_callback(
                            "agent_complete",
                            {"agent": current_agent, "status": "success"},
                        )
                except Exception as e:
                    logger.error(
                        "Agent execution failed",
                        agent_name=current_agent,
                        error=str(e),
                    )
                    if stream_callback:
                        stream_callback(
                            "agent_complete",
                            {"agent": current_agent, "status": "error", "error": str(e)},
                        )

                    # Transition to failure handler if configured
                    if agent_config.next_on_failure:
                        current_agent = agent_config.next_on_failure
                        continue
                    else:
                        return ExecutionResult(
                            state=state,
                            success=False,
                            error=f"Agent '{current_agent}' failed: {str(e)}",
                            iterations=iterations,
                            execution_time=time.monotonic() - start_time,
                            agents_executed=agents_executed,
                        )

                # Execute custom phases AFTER this agent
                for phase_config in self._customization.get_phase_after(
                    current_agent
                ):
                    state = await self._execute_phase(
                        phase_config, context, state, stream_callback
                    )

                # Determine next agent
                current_agent = self._determine_next_agent(
                    agent_config, state
                )

            # Check if we hit iteration limit
            if iterations >= self._config.max_iterations:
                logger.warning(
                    "Pipeline iteration limit reached",
                    max_iterations=self._config.max_iterations,
                )
                return ExecutionResult(
                    state=state,
                    success=True,  # Completed but hit limit
                    iterations=iterations,
                    execution_time=time.monotonic() - start_time,
                    agents_executed=agents_executed,
                )

            return ExecutionResult(
                state=state,
                success=True,
                iterations=iterations,
                execution_time=time.monotonic() - start_time,
                agents_executed=agents_executed,
            )

        except asyncio.CancelledError:
            logger.info("Pipeline execution cancelled")
            return ExecutionResult(
                state=state,
                success=False,
                error="Pipeline cancelled",
                iterations=iterations,
                execution_time=time.monotonic() - start_time,
                agents_executed=agents_executed,
            )
        except Exception as e:
            logger.exception("Pipeline execution error", error=str(e))
            return ExecutionResult(
                state=state,
                success=False,
                error=f"Pipeline error: {str(e)}",
                iterations=iterations,
                execution_time=time.monotonic() - start_time,
                agents_executed=agents_executed,
            )

    async def _execute_phase(
        self,
        phase_config: PhaseInsertion,
        context: "ResearchContext",
        state: "ResearchState",
        stream_callback: Callable[[str, Any], None] | None,
    ) -> "ResearchState":
        """Execute a custom phase.

        Args:
            phase_config: Phase insertion configuration
            context: Research context
            state: Current research state
            stream_callback: Optional streaming callback

        Returns:
            Updated research state
        """
        phase = self._custom_phases.get(phase_config.phase_name)
        if not phase:
            logger.warning(
                "Custom phase not found, skipping",
                phase_name=phase_config.phase_name,
            )
            return state

        logger.info("Executing custom phase", phase_name=phase_config.phase_name)
        if stream_callback:
            stream_callback("phase_start", {"phase": phase_config.phase_name})

        try:
            state = await phase.execute(context, state, phase_config.config)
            if stream_callback:
                stream_callback(
                    "phase_complete",
                    {"phase": phase_config.phase_name, "status": "success"},
                )
        except Exception as e:
            logger.error(
                "Custom phase failed",
                phase_name=phase_config.phase_name,
                error=str(e),
            )
            if stream_callback:
                stream_callback(
                    "phase_complete",
                    {
                        "phase": phase_config.phase_name,
                        "status": "error",
                        "error": str(e),
                    },
                )
            # Phases failing don't stop the pipeline, but we log it

        return state

    def _determine_next_agent(
        self,
        agent_config: AgentConfig,
        state: "ResearchState",
    ) -> str | None:
        """Determine the next agent based on config and state.

        Logic:
        1. If loop_condition is set and evaluates True, go to loop_back_to
        2. Else go to next_on_success
        3. If neither, pipeline ends

        Args:
            agent_config: Current agent configuration
            state: Current research state

        Returns:
            Name of next agent, or None if pipeline should end
        """
        # Check loop condition
        if agent_config.loop_condition and agent_config.loop_back_to:
            should_loop = self._evaluate_loop_condition(
                agent_config.loop_condition, state
            )
            if should_loop:
                logger.debug(
                    "Loop condition true, looping back",
                    condition=agent_config.loop_condition,
                    loop_back_to=agent_config.loop_back_to,
                )
                return agent_config.loop_back_to

        # Normal transition
        return agent_config.next_on_success

    def _evaluate_loop_condition(
        self,
        condition: str,
        state: "ResearchState",
    ) -> bool:
        """Evaluate a loop condition against the current state.

        Supported conditions:
        - "CONTINUE": State decision == "CONTINUE"
        - "ADJUST": State decision == "ADJUST"
        - "step_incomplete": Current step not completed
        - "not_complete": Research not complete
        - Custom conditions can be added via subclassing

        Args:
            condition: Condition string to evaluate
            state: Current research state

        Returns:
            True if condition is met (should loop)
        """
        # Check state decision (from reflector)
        if hasattr(state, "decision"):
            if condition == "CONTINUE" and state.decision == "CONTINUE":
                return True
            if condition == "ADJUST" and state.decision == "ADJUST":
                return True

        # Check step completion
        if condition == "step_incomplete":
            if hasattr(state, "current_step_index") and hasattr(state, "plan"):
                if state.plan and state.plan.steps:
                    current_idx = state.current_step_index
                    if current_idx < len(state.plan.steps):
                        current_step = state.plan.steps[current_idx]
                        return current_step.status != "completed"
            return False

        # Check overall completion
        if condition == "not_complete":
            if hasattr(state, "status"):
                return state.status != "complete"
            return False

        # Default: condition not recognized, don't loop
        logger.warning("Unknown loop condition", condition=condition)
        return False


def create_executor_for_pipeline(
    pipeline_name: str,
    customization: PipelineCustomization | None = None,
) -> PipelineExecutor:
    """Create a pipeline executor for a named default pipeline.

    Args:
        pipeline_name: Name of default pipeline (e.g., "deep-research")
        customization: Optional customization to apply

    Returns:
        Configured PipelineExecutor

    Raises:
        ValueError: If pipeline name is not recognized
    """
    from deep_research.agent.pipeline.defaults import get_default_pipeline

    config = get_default_pipeline(pipeline_name)
    return PipelineExecutor(config, customization)
