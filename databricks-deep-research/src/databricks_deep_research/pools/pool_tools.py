"""Auto-generated pool tools that expose PoolState via the ResearchTool protocol.

Each pool gets five tools (prefixed with the pool name):
  {pool}_search, {pool}_get_recent, {pool}_count, {pool}_topics, {pool}_get_by_index
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from databricks_deep_research.pools.pool_state import PoolState
from databricks_deep_research.tools.protocol import (
    ResearchTool,
    ToolContext,
    ToolDefinition,
    ToolResult,
)

if TYPE_CHECKING:
    from databricks_deep_research.pools.pool_registry import PoolRegistry

# ---------------------------------------------------------------------------
# PoolSearchTool
# ---------------------------------------------------------------------------


class PoolSearchTool:
    """Search a pool by keyword query (BM25+vector hybrid / overlap fallback)."""

    def __init__(
        self, pool_name: str, pool: PoolState,
        *, registry: PoolRegistry | None = None,
    ) -> None:
        self._pool_name = pool_name
        self._pool = pool
        self._registry = registry

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=f"{self._pool_name}_search",
            description=f"Search the '{self._pool_name}' pool by keyword query.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return.",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
            source_type="pool",
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if "query" not in arguments or not str(arguments["query"]).strip():
            raise ValueError("'query' is required and must be non-empty.")
        return {
            "query": str(arguments["query"]).strip(),
            "limit": int(arguments.get("limit", 10)),
        }

    async def execute(
        self, arguments: dict[str, Any], _context: ToolContext
    ) -> ToolResult:
        query = arguments["query"]
        limit = arguments["limit"]
        if self._registry is not None:
            results = await self._registry.search(self._pool_name, query, top_k=limit)
        else:
            results = self._pool.search(query, limit=limit)
        return ToolResult(
            content=json.dumps(results, default=str),
            data={"count": len(results)},
        )


# ---------------------------------------------------------------------------
# PoolGetRecentTool
# ---------------------------------------------------------------------------


class PoolGetRecentTool:
    """Get the N most recent items from a pool."""

    def __init__(self, pool_name: str, pool: PoolState) -> None:
        self._pool_name = pool_name
        self._pool = pool

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=f"{self._pool_name}_get_recent",
            description=f"Get the most recent items from the '{self._pool_name}' pool.",
            parameters={
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Number of recent items to return.",
                        "default": 10,
                    },
                },
                "required": [],
            },
            source_type="pool",
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return {"n": int(arguments.get("n", 10))}

    async def execute(
        self, arguments: dict[str, Any], _context: ToolContext
    ) -> ToolResult:
        items = self._pool.get_recent(arguments["n"])
        return ToolResult(
            content=json.dumps(items, default=str),
            data={"count": len(items)},
        )


# ---------------------------------------------------------------------------
# PoolCountTool
# ---------------------------------------------------------------------------


class PoolCountTool:
    """Get the number of items in a pool."""

    def __init__(self, pool_name: str, pool: PoolState) -> None:
        self._pool_name = pool_name
        self._pool = pool

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=f"{self._pool_name}_count",
            description=f"Get the number of items currently in the '{self._pool_name}' pool.",
            parameters={"type": "object", "properties": {}, "required": []},
            source_type="pool",
        )

    def validate_arguments(self, _arguments: dict[str, Any]) -> dict[str, Any]:
        return {}

    async def execute(
        self, _arguments: dict[str, Any], _context: ToolContext
    ) -> ToolResult:
        count = self._pool.count()
        return ToolResult(
            content=json.dumps({"count": count}),
            data={"count": count},
        )


# ---------------------------------------------------------------------------
# PoolTopicsTool
# ---------------------------------------------------------------------------


class PoolTopicsTool:
    """Get unique topics from a pool."""

    def __init__(self, pool_name: str, pool: PoolState) -> None:
        self._pool_name = pool_name
        self._pool = pool

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=f"{self._pool_name}_topics",
            description=f"Get unique topic labels from the '{self._pool_name}' pool.",
            parameters={"type": "object", "properties": {}, "required": []},
            source_type="pool",
        )

    def validate_arguments(self, _arguments: dict[str, Any]) -> dict[str, Any]:
        return {}

    async def execute(
        self, _arguments: dict[str, Any], _context: ToolContext
    ) -> ToolResult:
        topics = self._pool.topics()
        return ToolResult(
            content=json.dumps(topics),
            data={"topics": topics},
        )


# ---------------------------------------------------------------------------
# PoolGetByIndexTool
# ---------------------------------------------------------------------------


class PoolGetByIndexTool:
    """Get a specific item from a pool by its index."""

    def __init__(self, pool_name: str, pool: PoolState) -> None:
        self._pool_name = pool_name
        self._pool = pool

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=f"{self._pool_name}_get_by_index",
            description=f"Get a specific item from the '{self._pool_name}' pool by index.",
            parameters={
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "Zero-based index of the item to retrieve.",
                    },
                },
                "required": ["index"],
            },
            source_type="pool",
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if "index" not in arguments:
            raise ValueError("'index' is required.")
        return {"index": int(arguments["index"])}

    async def execute(
        self, arguments: dict[str, Any], _context: ToolContext
    ) -> ToolResult:
        item = self._pool.get_by_index(arguments["index"])
        if item is None:
            return ToolResult(
                content=f"No item at index {arguments['index']}.",
                success=False,
                error=f"Index {arguments['index']} out of range.",
            )
        return ToolResult(content=json.dumps(item, default=str))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_pool_tools(
    pool_name: str, pool: PoolState,
    *, registry: PoolRegistry | None = None,
) -> list[ResearchTool]:
    """Create all pool tools for a named pool.

    Tool names are prefixed with *pool_name*, e.g. ``sources_search``,
    ``observations_get_recent``.

    Args:
        pool_name: Name of the pool.
        pool: The PoolState instance.
        registry: Optional PoolRegistry for hybrid BM25+vector search.
    """
    return [
        PoolSearchTool(pool_name, pool, registry=registry),
        PoolGetRecentTool(pool_name, pool),
        PoolCountTool(pool_name, pool),
        PoolTopicsTool(pool_name, pool),
        PoolGetByIndexTool(pool_name, pool),
    ]
