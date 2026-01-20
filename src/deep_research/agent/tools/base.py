"""
Tool Protocol Definition
========================

This module defines the ResearchTool protocol that all tools must implement.
Both core tools (web_search, web_crawl, vector_search, knowledge_assistant)
and plugin-provided tools follow this protocol.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
from uuid import UUID


@dataclass(frozen=True)
class ToolDefinition:
    """
    JSON Schema-compatible tool definition for LLM function calling.

    This definition is passed to the LLM to describe what the tool does
    and what arguments it accepts.
    """

    name: str
    """Unique tool identifier (e.g., 'web_search', 'search_product_docs')."""

    description: str
    """Human-readable description of what the tool does. Used by LLM to decide when to use it."""

    parameters: dict[str, Any]
    """
    JSON Schema for tool parameters. Example:
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "default": 10}
        },
        "required": ["query"]
    }
    """


@dataclass
class ToolResult:
    """
    Result from tool execution.

    Contains the content to return to the LLM, success status,
    optional source tracking for citations, and optional raw data.
    """

    content: str
    """Text content to return to the LLM."""

    success: bool
    """Whether the tool execution succeeded."""

    sources: list[dict[str, Any]] | None = None
    """
    Source tracking for citation pipeline. Each source dict should contain:
    - type: Source type ('web', 'vector_search', 'knowledge_assistant', 'custom')
    - url: Source URL (if applicable)
    - title: Source title
    - Additional type-specific metadata
    """

    data: dict[str, Any] | None = None
    """Raw data from tool execution (for debugging or downstream processing)."""

    error: str | None = None
    """Error message if success=False."""


@dataclass
class ResearchContext:
    """
    Contextual information passed to tool execution.

    Provides identity, configuration, and shared registries for tools.
    """

    # Identity
    chat_id: UUID
    """Current chat session ID."""

    user_id: str
    """Current user identifier."""

    research_session_id: UUID | None = None
    """Research session ID (if research is in progress)."""

    # Research configuration
    research_type: str = "medium"
    """Research depth: 'light', 'medium', or 'extended'."""

    # Shared registries
    url_registry: dict[str, Any] = field(default_factory=dict)
    """Registry of URLs already visited/fetched."""

    evidence_registry: dict[str, Any] = field(default_factory=dict)
    """Registry of collected evidence for citation tracking."""

    # Plugin-provided data
    plugin_data: dict[str, Any] = field(default_factory=dict)
    """Data injected by plugins for cross-tool communication."""

    # Execution limits
    tool_call_count: int = 0
    """Number of tool calls made in current step."""

    max_tool_calls: int = 20
    """Maximum tool calls allowed per step."""


@runtime_checkable
class ResearchTool(Protocol):
    """
    Protocol for all research tools.

    Implementations must provide:
    - definition: Tool schema for LLM function calling
    - execute(): Async execution with context
    - validate_arguments(): Argument validation before execution
    """

    @property
    def definition(self) -> ToolDefinition:
        """
        Return tool definition for LLM function calling.

        This is called once at registration time to build the tool catalog.
        """
        ...

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ResearchContext,
    ) -> ToolResult:
        """
        Execute the tool with validated arguments.

        Args:
            arguments: Tool arguments from LLM (already validated)
            context: Research context with identity and registries

        Returns:
            ToolResult with content, success status, and optional sources
        """
        ...

    def validate_arguments(self, arguments: dict[str, Any]) -> list[str]:
        """
        Validate arguments before execution.

        Args:
            arguments: Raw arguments from LLM

        Returns:
            List of error messages (empty list if valid)
        """
        ...


# Type aliases for tool collections
ToolList = list[ResearchTool]
ToolMap = dict[str, ResearchTool]
