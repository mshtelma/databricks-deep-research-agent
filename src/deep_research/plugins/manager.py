"""
Plugin Manager
==============

Manages plugin discovery, initialization, and lifecycle.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from deep_research.agent.tools.base import ResearchContext, ResearchTool
from deep_research.agent.tools.registry import ToolRegistry
from deep_research.plugins.base import (
    PromptProvider,
    ResearchPlugin,
    ToolProvider,
)
from deep_research.plugins.discovery import discover_plugins

logger = logging.getLogger(__name__)


class PluginManagerError(Exception):
    """Exception raised for plugin manager errors."""

    pass


@dataclass
class PluginManager:
    """
    Manages plugin discovery, initialization, and lifecycle.

    Responsibilities:
    - Discover plugins via entry points
    - Initialize plugins with configuration
    - Collect tools from ToolProvider plugins
    - Collect prompt overrides from PromptProvider plugins
    - Graceful shutdown of all plugins
    """

    _plugins: list[ResearchPlugin] = field(default_factory=list)
    _tool_registry: ToolRegistry = field(default_factory=ToolRegistry)
    _initialized: bool = False

    def discover_and_load(self, app_config: Any) -> None:
        """
        Discover and initialize all plugins.

        Args:
            app_config: Application configuration (AppConfig instance)

        Note:
            Plugin initialization failures are logged but don't prevent
            other plugins from loading or the app from starting.

            Plugins can be disabled via app_config.plugins configuration:
                plugins:
                  my_plugin:
                    enabled: false
        """
        if self._initialized:
            logger.warning("PluginManager already initialized, skipping")
            return

        # Discover plugin classes
        plugin_classes = discover_plugins()
        logger.info("Discovered %d plugin(s)", len(plugin_classes))

        # Get plugin configuration if available
        plugins_config = getattr(app_config, "plugins", None)

        # Instantiate and initialize each plugin
        for plugin_cls in plugin_classes:
            try:
                # Create instance first to get its name
                plugin = plugin_cls()

                # Check if plugin is disabled via configuration
                if plugins_config is not None:
                    if not plugins_config.is_enabled(plugin.name):
                        logger.info(
                            "Skipping disabled plugin: %s",
                            plugin.name,
                        )
                        continue

                plugin.initialize(app_config)
                self._plugins.append(plugin)
                logger.info(
                    "Loaded plugin: %s v%s",
                    plugin.name,
                    plugin.version,
                )
            except AttributeError as e:
                # Plugin doesn't implement required protocol
                logger.warning(
                    "Plugin class %s doesn't implement ResearchPlugin: %s",
                    plugin_cls.__name__,
                    e,
                )
            except Exception as e:
                # Initialization failed - log and continue
                logger.warning(
                    "Failed to initialize plugin %s: %s",
                    getattr(plugin_cls, "__name__", str(plugin_cls)),
                    e,
                )

        # Register tools from all ToolProvider plugins
        self._register_tools()

        self._initialized = True
        logger.info(
            "PluginManager initialized: %d plugins, %d tools",
            len(self._plugins),
            len(self._tool_registry),
        )

    def _register_tools(self) -> None:
        """Collect and register tools from all ToolProvider plugins."""
        # Create a minimal context for tool collection
        from uuid import uuid4

        context = ResearchContext(
            chat_id=uuid4(),
            user_id="system",
            research_type="medium",
        )

        for plugin in self._plugins:
            if isinstance(plugin, ToolProvider):
                try:
                    tools = plugin.get_tools(context)
                    for tool in tools:
                        try:
                            self._tool_registry.register(tool)
                        except Exception:
                            # Conflict - register with prefix
                            self._tool_registry.register_with_prefix(
                                tool,
                                prefix=plugin.name,
                            )
                    logger.debug(
                        "Registered %d tools from plugin '%s'",
                        len(tools),
                        plugin.name,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to get tools from plugin '%s': %s",
                        plugin.name,
                        e,
                    )

    def get_tools(self, context: ResearchContext) -> list[ResearchTool]:
        """
        Get all registered tools.

        Args:
            context: Research context (for future context-aware filtering)

        Returns:
            List of all registered tools
        """
        return list(self._tool_registry)

    def get_tool_registry(self) -> ToolRegistry:
        """Get the underlying tool registry."""
        return self._tool_registry

    def get_prompt_overrides(self, context: ResearchContext) -> dict[str, str]:
        """
        Collect prompt overrides from all PromptProvider plugins.

        Args:
            context: Research context for prompt customization

        Returns:
            Dict mapping agent names to prompt additions.
            Later plugins override earlier ones for the same key.
        """
        overrides: dict[str, str] = {}

        for plugin in self._plugins:
            if isinstance(plugin, PromptProvider):
                try:
                    plugin_overrides = plugin.get_prompt_overrides(context)
                    overrides.update(plugin_overrides)
                except Exception as e:
                    logger.warning(
                        "Failed to get prompt overrides from plugin '%s': %s",
                        plugin.name,
                        e,
                    )

        return overrides

    def get_plugins(self) -> list[ResearchPlugin]:
        """Get all loaded plugins."""
        return list(self._plugins)

    def get_plugin(self, name: str) -> ResearchPlugin | None:
        """
        Get a plugin by name.

        Args:
            name: Plugin name to look up

        Returns:
            The plugin, or None if not found
        """
        for plugin in self._plugins:
            if plugin.name == name:
                return plugin
        return None

    def shutdown(self) -> None:
        """
        Shutdown all plugins.

        Calls shutdown() on each plugin, logging any errors.
        Clears internal state after shutdown.
        """
        logger.info("Shutting down %d plugin(s)", len(self._plugins))

        for plugin in self._plugins:
            try:
                plugin.shutdown()
                logger.debug("Shutdown plugin: %s", plugin.name)
            except Exception as e:
                logger.warning(
                    "Error shutting down plugin '%s': %s",
                    plugin.name,
                    e,
                )

        self._plugins.clear()
        self._tool_registry.clear()
        self._initialized = False
        logger.info("PluginManager shutdown complete")

    @property
    def initialized(self) -> bool:
        """Check if the manager has been initialized."""
        return self._initialized

    def __len__(self) -> int:
        """Return number of loaded plugins."""
        return len(self._plugins)
