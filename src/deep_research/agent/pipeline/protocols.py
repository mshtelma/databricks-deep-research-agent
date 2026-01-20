"""
Pipeline Customization Protocols
================================

Protocols for extending and customizing agent pipelines via plugins.

This module provides:
- PhaseInsertion: Configuration for inserting custom phases
- PipelineCustomization: Complete customization specification
- CustomPhase: Protocol for custom phase implementations
- PipelineCustomizer: Protocol for plugins providing customization
- PhaseProvider: Protocol for plugins providing custom phases
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from deep_research.agent.pipeline.config import AgentConfig
    from deep_research.agent.state import ResearchState
    from deep_research.agent.tools.base import ResearchContext


@dataclass
class PhaseInsertion:
    """Configuration for inserting a custom phase into the pipeline.

    Defines where and how a custom phase should be inserted relative
    to existing agents in the pipeline.

    Attributes:
        phase_name: Unique name for this phase
        insert_before: Name of agent to insert before (exclusive with insert_after)
        insert_after: Name of agent to insert after (exclusive with insert_before)
        enabled: Whether this phase is enabled
        config: Phase-specific configuration

    Example:
        >>> phase = PhaseInsertion(
        ...     phase_name="data_enrichment",
        ...     insert_after="researcher",
        ...     config={"max_enrichment_calls": 3},
        ... )
    """

    phase_name: str
    insert_before: str | None = None
    insert_after: str | None = None
    enabled: bool = True
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate phase insertion configuration."""
        if self.insert_before and self.insert_after:
            raise ValueError(
                f"Phase '{self.phase_name}' cannot specify both "
                f"insert_before and insert_after"
            )
        if not self.insert_before and not self.insert_after:
            raise ValueError(
                f"Phase '{self.phase_name}' must specify either "
                f"insert_before or insert_after"
            )


@dataclass
class PipelineCustomization:
    """Complete customization specification for a pipeline.

    Defines all modifications to apply to the default pipeline,
    including agent overrides, disabled agents, and custom phases.

    Attributes:
        agent_overrides: Dict of agent name -> config overrides
        disabled_agents: Set of agent names to disable
        phase_insertions: List of custom phases to insert
        pipeline_name: Name of base pipeline to customize (optional)
        config_overrides: Global pipeline config overrides

    Example:
        >>> customization = PipelineCustomization(
        ...     agent_overrides={
        ...         "researcher": {"model_tier": "complex"},
        ...     },
        ...     disabled_agents={"background"},
        ...     phase_insertions=[
        ...         PhaseInsertion(phase_name="validation", insert_after="synthesizer"),
        ...     ],
        ... )
    """

    agent_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    disabled_agents: set[str] = field(default_factory=set)
    phase_insertions: list[PhaseInsertion] = field(default_factory=list)
    pipeline_name: str | None = None
    config_overrides: dict[str, Any] = field(default_factory=dict)

    def apply_to_agent(self, config: "AgentConfig") -> "AgentConfig":
        """Apply overrides to an agent configuration.

        Args:
            config: Original agent configuration

        Returns:
            New AgentConfig with overrides applied
        """
        from dataclasses import replace

        if config.name not in self.agent_overrides:
            return config

        overrides = self.agent_overrides[config.name]
        return replace(config, **overrides)

    def is_agent_disabled(self, agent_name: str) -> bool:
        """Check if an agent is disabled.

        Args:
            agent_name: Name of agent to check

        Returns:
            True if agent is disabled
        """
        return agent_name in self.disabled_agents

    def get_phase_after(self, agent_name: str) -> list[PhaseInsertion]:
        """Get phases to insert after the given agent.

        Args:
            agent_name: Name of agent

        Returns:
            List of phase insertions after this agent
        """
        return [
            phase
            for phase in self.phase_insertions
            if phase.insert_after == agent_name and phase.enabled
        ]

    def get_phase_before(self, agent_name: str) -> list[PhaseInsertion]:
        """Get phases to insert before the given agent.

        Args:
            agent_name: Name of agent

        Returns:
            List of phase insertions before this agent
        """
        return [
            phase
            for phase in self.phase_insertions
            if phase.insert_before == agent_name and phase.enabled
        ]


@runtime_checkable
class CustomPhase(Protocol):
    """Protocol for custom phase implementations.

    Custom phases are callable objects that execute between standard
    pipeline agents. They receive the research context and state,
    perform their work, and optionally return a modified state.

    The phase can:
    - Read from ResearchState to access research data
    - Modify ResearchState to add/update data
    - Access tools via ResearchContext
    - Perform async operations (e.g., external API calls)

    Example:
        >>> class DataEnrichmentPhase:
        ...     @property
        ...     def name(self) -> str:
        ...         return "data_enrichment"
        ...
        ...     @property
        ...     def description(self) -> str:
        ...         return "Enriches research data with external sources"
        ...
        ...     async def execute(
        ...         self,
        ...         context: ResearchContext,
        ...         state: ResearchState,
        ...         config: dict[str, Any],
        ...     ) -> ResearchState:
        ...         # Perform enrichment...
        ...         return state
    """

    @property
    def name(self) -> str:
        """Unique name identifying this phase."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description of what this phase does."""
        ...

    async def execute(
        self,
        context: "ResearchContext",
        state: "ResearchState",
        config: dict[str, Any],
    ) -> "ResearchState":
        """Execute the custom phase.

        Args:
            context: Research context with tools and metadata
            state: Current research state
            config: Phase-specific configuration from PhaseInsertion

        Returns:
            Updated research state (can return same instance if unchanged)
        """
        ...


@runtime_checkable
class PipelineCustomizer(Protocol):
    """Protocol for plugins that customize the agent pipeline.

    Implement this protocol to provide pipeline customization from a plugin.
    The customization is applied when the pipeline is constructed.

    Example:
        >>> class MyPlugin:
        ...     def get_pipeline_customization(self) -> PipelineCustomization | None:
        ...         return PipelineCustomization(
        ...             disabled_agents={"background"},
        ...             agent_overrides={
        ...                 "researcher": {"max_iterations": 5},
        ...             },
        ...         )
    """

    def get_pipeline_customization(self) -> PipelineCustomization | None:
        """Get pipeline customization from this plugin.

        Returns:
            PipelineCustomization if this plugin wants to customize the pipeline,
            None otherwise.
        """
        ...


@runtime_checkable
class PhaseProvider(Protocol):
    """Protocol for plugins that provide custom phases.

    Implement this protocol to provide custom phases from a plugin.
    Phases are executed at specific points in the pipeline, as defined
    by the PhaseInsertion configuration.

    Example:
        >>> class MyPlugin:
        ...     def get_custom_phases(self) -> list[CustomPhase]:
        ...         return [DataEnrichmentPhase(), ValidationPhase()]
    """

    def get_custom_phases(self) -> list[CustomPhase]:
        """Get custom phases provided by this plugin.

        Returns:
            List of CustomPhase implementations.
        """
        ...
