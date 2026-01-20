"""
Plugin Discovery
================

Discovers plugins via Python entry points mechanism.
Plugins register themselves in their pyproject.toml under:

[project.entry-points."deep_research.plugins"]
my_plugin = "mypackage.plugin:MyPlugin"
"""

import importlib.metadata
import logging
from typing import Any

logger = logging.getLogger(__name__)

PLUGIN_ENTRY_POINT_GROUP = "deep_research.plugins"


def discover_plugins() -> list[type[Any]]:
    """
    Discover all plugins registered via entry points.

    Looks for entry points in the 'deep_research.plugins' group.
    Each entry point should point to a class implementing ResearchPlugin.

    Returns:
        List of plugin classes (not instances).
        Failed loads are logged and skipped.

    Example pyproject.toml for a plugin:
        [project.entry-points."deep_research.plugins"]
        my_plugin = "mypackage.plugin:MyPlugin"
    """
    plugins: list[type[Any]] = []

    try:
        entry_points = importlib.metadata.entry_points(group=PLUGIN_ENTRY_POINT_GROUP)
    except Exception as e:
        logger.warning("Failed to get entry points: %s", e)
        return plugins

    for ep in entry_points:
        try:
            logger.debug("Loading plugin from entry point: %s", ep.name)
            plugin_cls = ep.load()
            plugins.append(plugin_cls)
            logger.info("Discovered plugin: %s from %s", ep.name, ep.value)
        except ImportError as e:
            logger.warning(
                "Failed to import plugin '%s' from '%s': %s",
                ep.name,
                ep.value,
                e,
            )
        except Exception as e:
            logger.error(
                "Error loading plugin '%s': %s",
                ep.name,
                e,
            )

    return plugins


def discover_tools() -> list[type[Any]]:
    """
    Discover all tools registered via entry points.

    Looks for entry points in the 'deep_research.tools' group.
    This is separate from plugins - core tools register here.

    Returns:
        List of tool classes (not instances).
    """
    tools: list[type[Any]] = []

    try:
        entry_points = importlib.metadata.entry_points(group="deep_research.tools")
    except Exception as e:
        logger.warning("Failed to get tool entry points: %s", e)
        return tools

    for ep in entry_points:
        try:
            logger.debug("Loading tool from entry point: %s", ep.name)
            tool_cls = ep.load()
            tools.append(tool_cls)
            logger.info("Discovered tool: %s from %s", ep.name, ep.value)
        except ImportError as e:
            logger.warning(
                "Failed to import tool '%s' from '%s': %s",
                ep.name,
                ep.value,
                e,
            )
        except Exception as e:
            logger.error(
                "Error loading tool '%s': %s",
                ep.name,
                e,
            )

    return tools


def get_plugin_count() -> int:
    """Get count of discovered plugins without loading them."""
    try:
        entry_points = importlib.metadata.entry_points(group=PLUGIN_ENTRY_POINT_GROUP)
        return len(list(entry_points))
    except Exception:
        return 0
