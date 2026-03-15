"""
Pipeline Configuration
======================

Declarative configuration for research pipelines.

This module provides:
- AgentType enum for built-in agent types
- AgentConfig for individual agent configuration
- PipelineConfig for complete pipeline specification
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentType(str, Enum):
    """Types of agents available in the pipeline."""

    COORDINATOR = "coordinator"
    BACKGROUND = "background"
    PLANNER = "planner"
    RESEARCHER = "researcher"
    REFLECTOR = "reflector"
    SYNTHESIZER = "synthesizer"
    CUSTOM = "custom"

    def __str__(self) -> str:
        return self.value


@dataclass
class AgentConfig:
    """Configuration for a single agent in the pipeline.

    Defines how an agent behaves and transitions to other agents.

    Attributes:
        agent_type: Type of agent (from AgentType enum or custom string)
        name: Unique identifier for this agent instance (defaults to agent_type)
        enabled: Whether this agent is enabled
        model_tier: LLM tier to use (simple, analytical, complex)
        next_on_success: Agent to transition to on success
        next_on_failure: Agent to transition to on failure
        loop_condition: Condition to evaluate for looping
        loop_back_to: Agent to loop back to if condition is true
        max_iterations: Maximum iterations for this agent
        config: Additional agent-specific configuration

    Transition Logic:
        1. If loop_condition is set and evaluates to True, go to loop_back_to
        2. Else if next_on_success is set, go to that agent
        3. Else pipeline ends

    Loop Conditions:
        - "CONTINUE": State indicates more research needed
        - "ADJUST": State indicates plan adjustment needed
        - "step_incomplete": Current step not finished
        - Custom conditions via callable
    """

    agent_type: str | AgentType
    name: str | None = None
    enabled: bool = True
    model_tier: str = "analytical"
    next_on_success: str | None = None
    next_on_failure: str | None = None
    loop_condition: str | None = None
    loop_back_to: str | None = None
    max_iterations: int | None = None
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize agent_type to string and set default name."""
        if isinstance(self.agent_type, AgentType):
            self.agent_type = self.agent_type.value
        if self.name is None:
            self.name = self.agent_type

    def get_agent_type_enum(self) -> AgentType | None:
        """Get AgentType enum if this is a built-in type."""
        try:
            return AgentType(self.agent_type)
        except ValueError:
            return None


@dataclass
class PipelineConfig:
    """Declarative configuration for a research pipeline.

    Defines the complete agent sequence with transitions and limits.

    Attributes:
        name: Pipeline identifier
        description: Human-readable description
        agents: List of agent configurations in order
        start_agent: Name of the first agent to execute
        max_iterations: Global maximum loop iterations
        timeout_seconds: Maximum pipeline execution time

    Example:
        config = PipelineConfig(
            name="simple-research",
            description="Streamlined 3-agent pipeline",
            agents=[
                AgentConfig(
                    agent_type=AgentType.PLANNER,
                    next_on_success="researcher",
                ),
                AgentConfig(
                    agent_type=AgentType.RESEARCHER,
                    loop_condition="CONTINUE",
                    loop_back_to="researcher",
                    next_on_success="synthesizer",
                ),
                AgentConfig(
                    agent_type=AgentType.SYNTHESIZER,
                ),
            ],
            start_agent="planner",
        )
    """

    name: str
    description: str
    agents: list[AgentConfig]
    start_agent: str = "coordinator"
    max_iterations: int = 15
    timeout_seconds: int = 300

    def get_agent_config(self, name: str) -> AgentConfig | None:
        """Get configuration for a specific agent by name.

        Args:
            name: Agent name to look up

        Returns:
            AgentConfig if found, None otherwise
        """
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None

    def get_agent_names(self) -> list[str]:
        """Get list of all agent names in the pipeline."""
        return [agent.name for agent in self.agents if agent.name]

    def get_enabled_agents(self) -> list[AgentConfig]:
        """Get list of enabled agent configurations."""
        return [agent for agent in self.agents if agent.enabled]

    def validate(self) -> list[str]:
        """Validate pipeline configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors: list[str] = []

        # Check agents list
        if not self.agents:
            errors.append("Pipeline must have at least one agent")
            return errors

        # Collect agent names
        agent_names = set()
        for agent in self.agents:
            if not agent.name:
                errors.append("All agents must have a name")
                continue
            if agent.name in agent_names:
                errors.append(f"Duplicate agent name: {agent.name}")
            agent_names.add(agent.name)

        # Validate start_agent
        if self.start_agent not in agent_names:
            errors.append(
                f"start_agent '{self.start_agent}' not found in agents list"
            )

        # Validate transitions
        for agent in self.agents:
            if agent.next_on_success and agent.next_on_success not in agent_names:
                errors.append(
                    f"Agent '{agent.name}' has invalid next_on_success: "
                    f"'{agent.next_on_success}'"
                )
            if agent.next_on_failure and agent.next_on_failure not in agent_names:
                errors.append(
                    f"Agent '{agent.name}' has invalid next_on_failure: "
                    f"'{agent.next_on_failure}'"
                )
            if agent.loop_back_to:
                if not agent.loop_condition:
                    errors.append(
                        f"Agent '{agent.name}' has loop_back_to without loop_condition"
                    )
                if agent.loop_back_to not in agent_names:
                    errors.append(
                        f"Agent '{agent.name}' has invalid loop_back_to: "
                        f"'{agent.loop_back_to}'"
                    )

        # Validate limits
        if self.max_iterations < 1:
            errors.append("max_iterations must be at least 1")
        if self.timeout_seconds < 1:
            errors.append("timeout_seconds must be at least 1")

        return errors

    def is_valid(self) -> bool:
        """Check if pipeline configuration is valid."""
        return len(self.validate()) == 0

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "agents": [
                {
                    "agent_type": a.agent_type,
                    "name": a.name,
                    "enabled": a.enabled,
                    "model_tier": a.model_tier,
                    "next_on_success": a.next_on_success,
                    "next_on_failure": a.next_on_failure,
                    "loop_condition": a.loop_condition,
                    "loop_back_to": a.loop_back_to,
                    "max_iterations": a.max_iterations,
                    "config": a.config,
                }
                for a in self.agents
            ],
            "start_agent": self.start_agent,
            "max_iterations": self.max_iterations,
            "timeout_seconds": self.timeout_seconds,
        }
