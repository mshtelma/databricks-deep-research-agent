"""
Contract: Tool Protocol.

All tools (builtin, UC functions, enterprise, custom) implement this protocol.
The framework resolves tool references at workflow startup and caches them.

Changes from original:
- Single `definition` property (replaces separate name/description/parameters)
- `validate_arguments()` method added
- ToolResult: `success` field added, `metadata` → `data`
- Builtin tools use constructor DI (not extra dict)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ToolDefinition:
    """Tool definition — combines identity + schema for LLM function calling."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    source_type: str = "builtin"  # builtin, uc_function, uc_tool, enterprise


@dataclass(frozen=True)
class ToolResult:
    """Result returned by a tool execution."""

    content: str
    success: bool = True
    sources: list[SourceInfo] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class SourceInfo:
    """A source reference from a tool result."""

    url: str
    title: str = ""
    snippet: str = ""
    source_type: str = "web"  # web, enterprise, file, etc.


@dataclass(frozen=True)
class ToolRef:
    """Reference to a tool by type and name. Used in YAML configs."""

    type: str  # "builtin", "uc_function", "uc_tool", "enterprise"
    name: str  # Tool identifier


class UrlRegistry:
    """Maps integer indices to URLs. LLM sees indices only (security).

    Created per workflow execution, shared across all tool calls within
    a single workflow run. web_search registers discovered URLs and returns
    indices; web_crawl resolves indices back to URLs for fetching.

    Ported from existing app's tools/url_registry.py.
    """

    def register(self, url: str) -> int:
        """Register a URL and return its integer index."""
        ...

    def resolve(self, index: int) -> str | None:
        """Resolve an index back to its URL. Returns None if not found."""
        ...

    def get_all(self) -> list[tuple[int, str]]:
        """Return all (index, url) pairs."""
        ...


@dataclass(frozen=True)
class ToolContext:
    """Per-call context passed to tools at execution time.

    Tool dependencies (search clients, domain filters, user tokens) are
    constructor-injected at tool creation time, not passed per-call.
    Only per-call values that change between invocations belong here.
    """

    query: str = ""
    url_registry: UrlRegistry | None = None


@runtime_checkable
class ResearchTool(Protocol):
    """Protocol that all tools must implement.

    Builtin tools use constructor DI for dependencies:
        class WebSearchTool:
            def __init__(self, search_client: SearchClient) -> None:
                self._client = search_client

    Usage in YAML:
        tools:
          - type: builtin
            name: web_search
          - type: uc_function
            name: catalog.schema.my_function
          - type: enterprise
            name: my_vector_search
    """

    @property
    def definition(self) -> ToolDefinition:
        """Tool definition combining name, description, and parameter schema."""
        ...

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Validate and potentially transform arguments before execution.

        The returned dict is the canonical input to execute() — combining
        validation and transformation prevents bugs where uncleaned args
        are passed to execute().

        Args:
            arguments: Raw arguments from LLM tool call.

        Returns:
            Validated/transformed arguments dict.

        Raises:
            ValueError: If arguments are invalid.
        """
        ...

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the tool with given arguments.

        Args:
            arguments: Validated arguments matching self.definition.parameters.
            context: Execution context (query, domain filter, auth).

        Returns:
            ToolResult with content, success status, optional sources/data.
        """
        ...
