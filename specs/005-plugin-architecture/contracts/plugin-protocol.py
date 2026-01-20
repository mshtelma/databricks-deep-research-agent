"""
Plugin Protocol Definitions
===========================

This file defines the plugin protocols for extending the Deep Research Agent.

Plugins can implement one or more protocols:
- ResearchPlugin: Base lifecycle (required)
- ToolProvider: Provide custom tools
- PromptProvider: Customize agent prompts

Location: src/deep_research/plugins/base.py
"""

from typing import Protocol, Any, runtime_checkable

# Import from tool protocol
from .tool_protocol import ResearchTool, ResearchContext


@runtime_checkable
class ResearchPlugin(Protocol):
    """
    Base protocol for all plugins.

    Every plugin must implement this protocol to participate in the
    plugin lifecycle (discovery, initialization, shutdown).
    """

    @property
    def name(self) -> str:
        """
        Unique plugin identifier.

        Used for:
        - Configuration key in app.yaml (plugins.<name>)
        - Logging and error messages
        - Tool name prefixing on conflicts
        """
        ...

    @property
    def version(self) -> str:
        """
        Plugin version string (semver recommended).

        Used for:
        - Logging on startup
        - Compatibility tracking
        """
        ...

    def initialize(self, app_config: Any) -> None:
        """
        Initialize plugin with application configuration.

        Called once on application startup after discovery.
        Plugin-specific configuration is available at:
        app_config.plugins.get(self.name, {})

        Args:
            app_config: Full application configuration (AppConfig instance)

        Raises:
            Exception: If initialization fails (logged, app continues)
        """
        ...

    def shutdown(self) -> None:
        """
        Clean up resources on application shutdown.

        Called on graceful shutdown. Should release connections,
        close files, etc.
        """
        ...


@runtime_checkable
class ToolProvider(Protocol):
    """
    Protocol for plugins that provide research tools.

    Implement this alongside ResearchPlugin to add custom tools
    to the researcher agent's toolkit.
    """

    def get_tools(self, context: ResearchContext) -> list[ResearchTool]:
        """
        Return list of tools provided by this plugin.

        Called when building the tool registry. Tools may be filtered
        or customized based on context (research type, user, etc.).

        Args:
            context: Research context for tool filtering

        Returns:
            List of ResearchTool implementations
        """
        ...


@runtime_checkable
class PromptProvider(Protocol):
    """
    Protocol for plugins that customize agent prompts.

    Implement this alongside ResearchPlugin to inject domain-specific
    instructions into agent prompts.
    """

    def get_prompt_overrides(
        self,
        context: ResearchContext,
    ) -> dict[str, str]:
        """
        Return prompt customizations for agents.

        Args:
            context: Research context for prompt customization

        Returns:
            Dict mapping agent names to prompt additions/overrides.

            Keys are agent names:
            - "coordinator": Query classification agent
            - "planner": Research planning agent
            - "researcher": Step execution agent
            - "reflector": Step-by-step reflection agent
            - "synthesizer": Report generation agent

            Values are prompt strings to append to the agent's
            system prompt. Use this to inject domain-specific
            instructions, constraints, or context.

        Example:
            {
                "researcher": '''
                    When researching companies, always check:
                    - Recent news and press releases
                    - Financial reports if available
                    - Key executive changes
                ''',
                "synthesizer": '''
                    Format the final report using MEDDPICC framework:
                    - Metrics
                    - Economic Buyer
                    - Decision Criteria
                    ...
                '''
            }
        """
        ...


# Combined plugin type for plugins implementing multiple protocols
class FullPlugin(ResearchPlugin, ToolProvider, PromptProvider, Protocol):
    """
    Combined protocol for plugins implementing all capabilities.

    This is a convenience type for type hints. Most plugins will
    implement ResearchPlugin plus one or both of ToolProvider
    and PromptProvider.
    """

    pass


# Example implementation skeleton
class ExamplePlugin:
    """
    Example plugin implementation showing all protocols.

    This is NOT part of the contract - just documentation.
    """

    name = "example"
    version = "0.1.0"

    def __init__(self) -> None:
        self._tools: list[ResearchTool] = []
        self._config: dict[str, Any] = {}

    def initialize(self, app_config: Any) -> None:
        """Initialize with config."""
        self._config = getattr(app_config, 'plugins', {}).get(self.name, {})
        # Initialize tools based on config
        # self._tools = [MyTool(self._config)]

    def shutdown(self) -> None:
        """Clean up."""
        self._tools.clear()

    def get_tools(self, context: ResearchContext) -> list[ResearchTool]:
        """Return tools."""
        return self._tools

    def get_prompt_overrides(
        self,
        context: ResearchContext,
    ) -> dict[str, str]:
        """Return prompt additions."""
        return {
            "researcher": self._config.get("researcher_prompt", ""),
        }
