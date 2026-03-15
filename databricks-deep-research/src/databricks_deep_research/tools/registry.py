"""Tool registry -- resolves ToolRef to concrete ResearchTool instances.

Builtin tools are registered by name at application startup.  External tools
(UC functions, enterprise connectors) are registered per-execution from the
orchestration context.

Resolved instances are cached for the lifetime of the registry (typically one
workflow execution) so that shared state such as ``UrlRegistry`` remains
consistent across calls.
"""

from __future__ import annotations

import logging
from typing import Any

from databricks_deep_research.tools.protocol import ResearchTool, ToolRef

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Resolves ``ToolRef`` to concrete ``ResearchTool`` instances.

    Builtin tools are registered by name.  External tools (``uc_function``,
    ``enterprise``) are passed in from the execution context.

    Tools are cached for the lifetime of a workflow execution to ensure
    consistent state (e.g., ``UrlRegistry`` shared across calls).
    """

    def __init__(self) -> None:
        self._builtins: dict[str, ResearchTool] = {}
        self._external: dict[str, ResearchTool] = {}
        self._cache: dict[str, ResearchTool] = {}

    # -- registration --------------------------------------------------------

    def register_builtin(self, name: str, tool: ResearchTool) -> None:
        """Register a builtin tool by name."""
        self._builtins[name] = tool
        logger.debug("Registered builtin tool: %s", name)

    def register_external(self, name: str, tool: ResearchTool) -> None:
        """Register an external (enterprise / UC) tool."""
        self._external[name] = tool
        logger.debug("Registered external tool: %s", name)

    # -- resolution ----------------------------------------------------------

    def resolve(self, ref: ToolRef) -> ResearchTool:
        """Resolve a ``ToolRef`` to a concrete tool instance.

        Resolution order:
        1. Return from cache if previously resolved.
        2. Look up in the appropriate store (builtins or external).
        3. Cache the result for future calls.

        Raises:
            ValueError: If the tool cannot be found.
        """
        cache_key = f"{ref.type}:{ref.name}"

        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        tool: ResearchTool | None = None

        if ref.type == "builtin":
            tool = self._builtins.get(ref.name)
        elif ref.type in ("uc_function", "uc_tool", "enterprise"):
            tool = self._external.get(ref.name)
        else:
            raise ValueError(f"Unknown tool type: {ref.type!r} for tool {ref.name!r}")

        if tool is None:
            available = self._available_names_for_type(ref.type)
            raise ValueError(
                f"Tool not found: {ref.type}:{ref.name}. "
                f"Available: {available}"
            )

        self._cache[cache_key] = tool
        return tool

    def resolve_many(self, refs: list[dict[str, Any]]) -> list[ResearchTool]:
        """Resolve a list of tool-ref dicts (from YAML config) to tools.

        Each dict must have ``type`` and ``name`` keys, matching the
        ``ToolRef`` dataclass fields.
        """
        tools: list[ResearchTool] = []
        for raw in refs:
            ref = ToolRef(type=raw["type"], name=raw["name"])
            tools.append(self.resolve(ref))
        return tools

    # -- queries -------------------------------------------------------------

    def get_all_builtins(self) -> dict[str, ResearchTool]:
        """Return all registered builtin tools (name -> instance)."""
        return dict(self._builtins)

    def has(self, name: str) -> bool:
        """Check if a tool is registered (builtin or external)."""
        return name in self._builtins or name in self._external

    # -- internals -----------------------------------------------------------

    def _available_names_for_type(self, tool_type: str) -> list[str]:
        """Return registered names for a given tool type (for error messages)."""
        if tool_type == "builtin":
            return sorted(self._builtins)
        return sorted(self._external)
