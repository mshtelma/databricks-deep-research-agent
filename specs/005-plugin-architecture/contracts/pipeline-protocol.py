"""
Pipeline Protocol Definitions
=============================

This file defines the pipeline customization protocols for extending the
Deep Research Agent's agent architecture.

Plugins can:
- Replace the entire pipeline with a custom configuration
- Insert custom phases at specified points
- Disable specific agents
- Override agent configurations

Location: src/deep_research/plugins/pipeline.py
"""

from typing import Protocol, Any, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Agent Configuration
# ---------------------------------------------------------------------------

class AgentType(str, Enum):
    """Built-in agent types."""
    COORDINATOR = "coordinator"
    BACKGROUND = "background"
    PLANNER = "planner"
    RESEARCHER = "researcher"
    REFLECTOR = "reflector"
    SYNTHESIZER = "synthesizer"
    CUSTOM = "custom"


@dataclass
class AgentConfig:
    """
    Configuration for a single agent in the pipeline.

    Defines agent behavior, model tier, and transition logic.
    """

    agent_type: str
    """Agent type (from AgentType enum or custom string)."""

    enabled: bool = True
    """Whether this agent is active. Disabled agents are skipped."""

    model_tier: str = "analytical"
    """Model tier to use: 'simple', 'analytical', or 'complex'."""

    next_on_success: str | None = None
    """Next agent to execute on successful completion. None = end pipeline."""

    next_on_failure: str | None = None
    """Next agent to execute on failure. None = use next_on_success."""

    loop_condition: str | None = None
    """
    Condition expression for looping back. Examples:
    - "decision == CONTINUE" (for reflector)
    - "needs_more_research" (for researcher)
    - "iteration < 5"
    """

    loop_back_to: str | None = None
    """Agent to loop back to when loop_condition is true."""

    config: dict[str, Any] = field(default_factory=dict)
    """
    Agent-specific configuration. Examples:
    - researcher: {"max_tool_calls": 15, "mode": "react"}
    - reflector: {"adjust_goes_to": "planner"}
    - synthesizer: {"output_type": "meeting_prep"}
    """


# ---------------------------------------------------------------------------
# Pipeline Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    Declarative pipeline configuration.

    Defines the complete agent execution flow, including transitions,
    loop conditions, and global settings.
    """

    name: str
    """Unique pipeline identifier."""

    description: str
    """Human-readable description of the pipeline behavior."""

    agents: list[AgentConfig]
    """Ordered list of agent configurations."""

    start_agent: str = "coordinator"
    """Agent to execute first."""

    max_iterations: int = 15
    """Maximum total iterations across all agents."""

    timeout_seconds: int = 300
    """Maximum total execution time."""

    def get_agent_config(self, agent_type: str) -> AgentConfig | None:
        """Get configuration for a specific agent type."""
        for agent in self.agents:
            if agent.agent_type == agent_type:
                return agent
        return None

    def validate(self) -> list[str]:
        """
        Validate pipeline configuration.

        Returns list of error messages (empty if valid).
        """
        errors: list[str] = []

        # Check start_agent exists
        agent_types = {a.agent_type for a in self.agents}
        if self.start_agent not in agent_types:
            errors.append(f"start_agent '{self.start_agent}' not in agents")

        # Check all transitions reference valid agents
        for agent in self.agents:
            if agent.next_on_success and agent.next_on_success not in agent_types:
                errors.append(
                    f"Agent '{agent.agent_type}' next_on_success "
                    f"'{agent.next_on_success}' not in agents"
                )
            if agent.loop_back_to and agent.loop_back_to not in agent_types:
                errors.append(
                    f"Agent '{agent.agent_type}' loop_back_to "
                    f"'{agent.loop_back_to}' not in agents"
                )

        return errors


# ---------------------------------------------------------------------------
# Phase Insertion
# ---------------------------------------------------------------------------

@dataclass
class PhaseInsertion:
    """
    Specification for inserting a custom phase into the pipeline.

    A phase is like an agent but defined by a plugin. Phases can be
    inserted before or after specific agents.
    """

    phase: "CustomPhase"
    """The custom phase to insert."""

    after: str | None = None
    """Insert after this agent. Mutually exclusive with 'before'."""

    before: str | None = None
    """Insert before this agent. Mutually exclusive with 'after'."""

    def validate(self) -> list[str]:
        """Validate phase insertion specification."""
        errors: list[str] = []
        if self.after and self.before:
            errors.append("Cannot specify both 'after' and 'before'")
        if not self.after and not self.before:
            errors.append("Must specify either 'after' or 'before'")
        return errors


# ---------------------------------------------------------------------------
# Pipeline Customization
# ---------------------------------------------------------------------------

@dataclass
class PipelineCustomization:
    """
    Customizations to apply to a pipeline configuration.

    Used to modify the default pipeline without replacing it entirely.
    """

    insert_phases: list[PhaseInsertion] = field(default_factory=list)
    """Custom phases to insert into the pipeline."""

    agent_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    """
    Per-agent configuration overrides. Example:
    {
        "researcher": {"config": {"max_tool_calls": 20}},
        "synthesizer": {"model_tier": "complex"}
    }
    """

    disabled_agents: list[str] = field(default_factory=list)
    """Agent types to disable (skip during execution)."""


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

class CustomPhase(Protocol):
    """
    Protocol for custom pipeline phases.

    Implement this to add domain-specific processing steps to the pipeline.
    Phases receive the same context as built-in agents.
    """

    @property
    def name(self) -> str:
        """Unique phase identifier."""
        ...

    async def execute(
        self,
        state: "ResearchState",
        context: "ResearchContext",
    ) -> AsyncGenerator["StreamEvent", None]:
        """
        Execute the custom phase.

        Args:
            state: Current research state (mutable)
            context: Research context with identity and registries

        Yields:
            StreamEvent instances for frontend updates
        """
        ...


class PipelineCustomizer(Protocol):
    """
    Protocol for plugins that customize the agent pipeline.

    Implement this to replace or modify the default pipeline.
    """

    def get_pipeline_config(
        self,
        ctx: "ResearchContext",
    ) -> PipelineConfig | None:
        """
        Return a complete custom pipeline configuration.

        If None, the default pipeline is used (possibly with customizations).
        If a PipelineConfig is returned, it replaces the default entirely.

        Args:
            ctx: Research context for conditional configuration

        Returns:
            Custom pipeline config, or None to use default
        """
        ...

    def get_pipeline_customizations(
        self,
        ctx: "ResearchContext",
    ) -> PipelineCustomization | None:
        """
        Return customizations to apply to the default pipeline.

        Called only if get_pipeline_config() returns None.
        Customizations are merged: phases inserted, agents overridden,
        disabled agents skipped.

        Args:
            ctx: Research context for conditional customization

        Returns:
            Pipeline customizations, or None for no changes
        """
        ...


class PhaseProvider(Protocol):
    """
    Protocol for plugins that provide custom phases.

    Simpler than PipelineCustomizer - only adds phases without
    replacing or extensively modifying the pipeline.
    """

    def get_phases(
        self,
        ctx: "ResearchContext",
    ) -> list[PhaseInsertion]:
        """
        Return custom phases to insert into the pipeline.

        Args:
            ctx: Research context for conditional phase inclusion

        Returns:
            List of phase insertions
        """
        ...


# ---------------------------------------------------------------------------
# Default Pipelines (Reference)
# ---------------------------------------------------------------------------

# These are defined in src/deep_research/agent/pipeline/defaults.py
# Shown here for documentation purposes

"""
DEFAULT_DEEP_RESEARCH_PIPELINE = PipelineConfig(
    name="deep_research",
    description="Multi-agent deep research with reflection",
    agents=[
        AgentConfig(agent_type="coordinator", next_on_success="background"),
        AgentConfig(agent_type="background", next_on_success="planner"),
        AgentConfig(agent_type="planner", next_on_success="researcher"),
        AgentConfig(agent_type="researcher", next_on_success="reflector"),
        AgentConfig(
            agent_type="reflector",
            next_on_success="synthesizer",
            loop_condition="decision == CONTINUE",
            loop_back_to="researcher",
            config={"adjust_goes_to": "planner"}
        ),
        AgentConfig(agent_type="synthesizer", model_tier="complex"),
    ],
    start_agent="coordinator",
)

SIMPLE_RESEARCH_PIPELINE = PipelineConfig(
    name="simple_research",
    description="Single-pass research without reflection",
    agents=[
        AgentConfig(agent_type="coordinator", next_on_success="researcher"),
        AgentConfig(agent_type="researcher", next_on_success="synthesizer"),
        AgentConfig(agent_type="synthesizer"),
    ],
    start_agent="coordinator",
)

REACT_LOOP_PIPELINE = PipelineConfig(
    name="react_loop",
    description="Single ReAct agent loop (sapresalesbot pattern)",
    agents=[
        AgentConfig(agent_type="coordinator", next_on_success="researcher"),
        AgentConfig(
            agent_type="researcher",
            next_on_success="synthesizer",
            loop_condition="needs_more_research",
            loop_back_to="researcher",
            config={"max_iterations": 10, "mode": "react"}
        ),
        AgentConfig(agent_type="synthesizer"),
    ],
    start_agent="coordinator",
)
"""
