"""
Base class for prompt-driven research phases.

Provides default execution behavior for custom phases defined by plugins.
Plugins extend this class and only need to define:
- name: Unique phase identifier
- description: Human-readable description
- prompt: The focused research prompt for this phase
- tools: List of tool names this phase can use

The execute() method is provided by the framework and uses
the researcher agent pattern internally.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from deep_research.agent.state import ResearchState
    from deep_research.agent.tools.base import ResearchContext


class BaseResearchPhase(ABC):
    """Base class for research phases with default execution.

    Plugins extend this class and only need to define:
    - name: Unique phase identifier
    - description: Human-readable description
    - prompt: The focused research prompt for this phase
    - tools: List of tool names this phase can use

    Optionally override:
    - should_run(): For conditional execution based on context
    - get_formatted_prompt(): For dynamic prompt templating

    The execute() method is provided by the framework and uses
    the researcher agent internally. Override only for truly
    custom execution needs.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique phase identifier used for registration and results."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description for logging."""
        ...

    @property
    @abstractmethod
    def prompt(self) -> str:
        """The research prompt for this phase."""
        ...

    @property
    def tools(self) -> list[str]:
        """Tool names this phase can use. Override to customize."""
        return ["web_search", "web_crawl"]

    def should_run(self, context: "ResearchContext") -> bool:
        """Check if phase should execute based on context.

        Override for conditional phases that depend on plugin_data.
        Default: always run.

        Args:
            context: Research context with plugin_data and session info

        Returns:
            True if phase should execute, False to skip
        """
        return True

    def get_formatted_prompt(self, context: "ResearchContext") -> str:
        """Format prompt with context variables.

        Override for dynamic prompt templating.
        Default: return prompt as-is.

        Args:
            context: Research context with plugin_data

        Returns:
            Formatted prompt string
        """
        return self.prompt

    async def execute(
        self,
        context: "ResearchContext",
        state: "ResearchState",
        config: dict[str, Any],
    ) -> "ResearchState":
        """Execute this phase using the researcher agent pattern.

        Default implementation:
        1. Check should_run() - skip if False
        2. Get formatted prompt
        3. Execute research using LLM with tools
        4. Store result in state.phase_results[self.name]

        Override only for custom execution logic.

        Args:
            context: Research context with tools and metadata
            state: Current research state
            config: Phase-specific configuration from PhaseInsertion

        Returns:
            Updated research state (can return same instance if unchanged)
        """
        # Skip if should_run returns False
        if not self.should_run(context):
            return state

        # Get formatted prompt with context variables
        prompt = self.get_formatted_prompt(context)

        # Execute using framework's researcher pattern
        # Import here to avoid circular imports
        from deep_research.agent.nodes.custom_phase_executor import execute_custom_phase

        result = await execute_custom_phase(
            phase_name=self.name,
            prompt=prompt,
            tools=self.tools,
            context=context,
            state=state,
            config=config,
        )

        # Store result in state
        state.add_phase_result(self.name, result)
        return state


__all__ = ["BaseResearchPhase"]
