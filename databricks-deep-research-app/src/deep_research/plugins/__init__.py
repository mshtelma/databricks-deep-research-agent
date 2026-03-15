"""
Plugin System
=============

Provides the plugin infrastructure for extending Deep Research Agent.

Protocols:
- ResearchPlugin: Base lifecycle for all plugins
- ToolProvider: Plugins that provide tools
- PromptProvider: Plugins that customize prompts

Classes:
- PluginManager: Manages plugin discovery and lifecycle
- ToolRegistry: Central registry for tools

Discovery:
- Plugins are discovered via entry points (deep_research.plugins group)
- Register in pyproject.toml:
    [project.entry-points."deep_research.plugins"]
    my_plugin = "mypackage.plugin:MyPlugin"
"""

from deep_research.agent.tools.base import ResearchContext
from deep_research.plugins.base import (
    FullPlugin,
    PluginList,
    PromptProvider,
    ResearchPlugin,
    ToolProvider,
)
from deep_research.plugins.discovery import discover_plugins, get_plugin_count
from deep_research.plugins.manager import PluginManager, PluginManagerError

__all__ = [
    # Protocols
    "ResearchPlugin",
    "ToolProvider",
    "PromptProvider",
    "FullPlugin",
    # Types
    "PluginList",
    # Manager
    "PluginManager",
    "PluginManagerError",
    # Context (for plugin use)
    "ResearchContext",
    # Discovery
    "discover_plugins",
    "get_plugin_count",
]
