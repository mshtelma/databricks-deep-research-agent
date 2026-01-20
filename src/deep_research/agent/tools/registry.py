"""
Tool Registry
=============

Central registry for all research tools. Provides registration, lookup,
and conversion to OpenAI function calling format.
"""

import logging
from typing import Any

from deep_research.agent.tools.base import ResearchTool, ToolDefinition, ToolMap

logger = logging.getLogger(__name__)


class ToolRegistryError(Exception):
    """Exception raised for tool registry errors."""

    pass


class ToolRegistry:
    """
    Registry for research tools.

    Provides centralized management of tools with:
    - Registration with conflict detection
    - Lookup by name
    - Conversion to OpenAI function calling format
    - Support for tool prefixing to avoid conflicts
    """

    def __init__(self) -> None:
        """Initialize an empty tool registry."""
        self._tools: ToolMap = {}

    def register(self, tool: ResearchTool) -> None:
        """
        Register a tool.

        Args:
            tool: Tool implementing ResearchTool protocol

        Raises:
            ToolRegistryError: If a tool with the same name is already registered
        """
        name = tool.definition.name
        if name in self._tools:
            raise ToolRegistryError(
                f"Tool '{name}' already registered. "
                f"Use register_with_prefix() to avoid conflicts."
            )
        self._tools[name] = tool
        logger.debug("Registered tool: %s", name)

    def register_with_prefix(self, tool: ResearchTool, prefix: str) -> str:
        """
        Register tool with a prefix to avoid name conflicts.

        The tool will be registered with name: {prefix}_{original_name}

        Args:
            tool: Tool implementing ResearchTool protocol
            prefix: Prefix to add to tool name (e.g., plugin name)

        Returns:
            The actual registered name (with prefix)
        """
        original_name = tool.definition.name
        prefixed_name = f"{prefix}_{original_name}"

        # Create a wrapper that returns the prefixed definition
        class PrefixedTool:
            def __init__(self, wrapped: ResearchTool, new_name: str) -> None:
                self._wrapped = wrapped
                self._prefixed_def = ToolDefinition(
                    name=new_name,
                    description=wrapped.definition.description,
                    parameters=wrapped.definition.parameters,
                )

            @property
            def definition(self) -> ToolDefinition:
                return self._prefixed_def

            async def execute(
                self,
                arguments: dict[str, Any],
                context: Any,
            ) -> Any:
                return await self._wrapped.execute(arguments, context)

            def validate_arguments(self, arguments: dict[str, Any]) -> list[str]:
                return self._wrapped.validate_arguments(arguments)

        prefixed_tool = PrefixedTool(tool, prefixed_name)
        self._tools[prefixed_name] = prefixed_tool  # type: ignore[assignment]
        logger.debug(
            "Registered tool with prefix: %s (original: %s)",
            prefixed_name,
            original_name,
        )
        return prefixed_name

    def get(self, name: str) -> ResearchTool | None:
        """
        Get a tool by name.

        Args:
            name: Tool name to look up

        Returns:
            The tool, or None if not found
        """
        return self._tools.get(name)

    def unregister(self, name: str) -> bool:
        """
        Unregister a tool by name.

        Args:
            name: Tool name to remove

        Returns:
            True if tool was removed, False if not found
        """
        if name in self._tools:
            del self._tools[name]
            logger.debug("Unregistered tool: %s", name)
            return True
        return False

    def list_tools(self) -> list[ToolDefinition]:
        """
        List all registered tool definitions.

        Returns:
            List of ToolDefinition objects
        """
        return [t.definition for t in self._tools.values()]

    def get_tool_names(self) -> list[str]:
        """
        Get names of all registered tools.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def get_openai_tools(self) -> list[dict[str, Any]]:
        """
        Get tools in OpenAI function calling format.

        Returns:
            List of tool definitions in OpenAI format:
            [{"type": "function", "function": {...}}, ...]
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": t.definition.name,
                    "description": t.definition.description,
                    "parameters": t.definition.parameters,
                },
            }
            for t in self._tools.values()
        ]

    def get_tools(self) -> ToolMap:
        """
        Get a copy of the internal tools dictionary.

        Returns:
            Dictionary mapping tool names to tools
        """
        return dict(self._tools)

    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()
        logger.debug("Cleared tool registry")

    def __len__(self) -> int:
        """Return number of registered tools."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def __iter__(self):
        """Iterate over registered tools."""
        return iter(self._tools.values())
