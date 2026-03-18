"""Vector Search Tool for Databricks Vector Search.

Provides semantic search over Vector Search indexes configured via app.yaml.
Each configured endpoint creates a separate tool instance with a unique name.

Uses WorkspaceClient.vector_search_indexes.query_index() for consistent
authentication across all environments (profiles, SP, OBO).

Example configuration (config/app.yaml):
    vector_search:
      enabled: true
      endpoints:
        product_docs:
          endpoint_name: vs-endpoint-prod
          index_name: catalog.schema.product_docs_index
          columns: ["title", "content", "url"]
          description: Search product documentation
          num_results: 5
"""

import json
from typing import Any

from deep_research.agent.tools.base import (
    ResearchContext,
    ToolDefinition,
    ToolResult,
)
from deep_research.core.auth import get_workspace_client
from deep_research.core.logging_utils import get_logger
from deep_research.services.vector_search_query import VectorSearchQueryService

logger = get_logger(__name__)


class VectorSearchTool:
    """
    Vector Search tool implementing the ResearchTool protocol.

    Queries a Databricks Vector Search index for semantically similar documents.
    Tool name is generated as 'search_{endpoint_name}' to allow multiple indexes.

    Uses WorkspaceClient SDK for authentication (profile-aware, OAuth-native).
    """

    def __init__(
        self,
        *,
        endpoint_name: str,
        index_name: str,
        columns: list[str] | None = None,
        tool_name: str | None = None,
        description: str | None = None,
        num_results: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the Vector Search tool.

        Args:
            endpoint_name: Databricks Vector Search endpoint name.
            index_name: Fully qualified index name (catalog.schema.index).
            columns: Columns to return from search results.
                     Defaults to ["title", "content", "url"].
            tool_name: Custom tool name. Defaults to 'search_{endpoint_name}'.
            description: Custom description for LLM. Defaults to generic.
            num_results: Default number of results to return.
            filters: Optional filters to apply to all searches.
        """
        self._endpoint_name = endpoint_name
        self._index_name = index_name
        self._columns = columns or ["title", "content", "url"]
        self._num_results = num_results
        self._filters = filters or {}

        # Generate tool name
        self._tool_name = tool_name or f"search_{endpoint_name.replace('-', '_')}"

        # Generate description
        self._description = description or (
            f"Search the '{index_name}' vector index for semantically similar documents. "
            "Returns relevant passages ranked by similarity score."
        )

        self._definition = ToolDefinition(
            name=self._tool_name,
            description=self._description,
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search query. Use natural language describing what you're looking for."
                        ),
                    },
                    "num_results": {
                        "type": "integer",
                        "description": f"Number of results to return (default: {self._num_results})",
                        "default": self._num_results,
                    },
                },
                "required": ["query"],
            },
            source_type="vector_search",
        )

        self._query_service = VectorSearchQueryService()

    @property
    def definition(self) -> ToolDefinition:
        """Return tool definition for LLM function calling."""
        return self._definition

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ResearchContext,  # noqa: ARG002  # Protocol requires context
    ) -> ToolResult:
        """Execute vector search and return results.

        Args:
            arguments: Tool arguments containing 'query' and optional 'num_results'
            context: Research context with identity and registries

        Returns:
            ToolResult with formatted search results and source tracking
        """
        query = arguments.get("query", "")
        num_results = arguments.get("num_results", self._num_results)

        try:
            client = get_workspace_client()

            # Build filters_json from configured filters
            filters_json: str | None = None
            if self._filters:
                filters_json = json.dumps(self._filters)

            results = await self._query_service.query(
                client=client,
                index_name=self._index_name,
                query_text=query,
                columns=self._columns,
                num_results=num_results,
                filters_json=filters_json,
            )

            if not results:
                return ToolResult(
                    content="No results found matching your query.",
                    success=True,
                    sources=[],
                    data={"query": query, "num_results": 0},
                )

            # Build sources list for citation tracking
            sources: list[dict[str, Any]] = []
            formatted_results: list[str] = []

            for idx, result in enumerate(results):
                sources.append({
                    "type": "vector_search",
                    "index_name": self._index_name,
                    "endpoint_name": self._endpoint_name,
                    "url": result.url,
                    "title": result.title,
                    "content": result.content[:500] if result.content else "",
                    "relevance_score": result.score,
                    "search_index": idx,
                    "metadata": result.metadata,
                })

                # Format result for LLM
                url_display = f"\nURL: {result.url}" if result.url else ""
                formatted_results.append(
                    f"[{idx}] **{result.title}** (score: {result.score:.2f}){url_display}\n"
                    f"    {result.content[:300]}..."
                )

            content = "\n\n".join(formatted_results)

            return ToolResult(
                content=content,
                success=True,
                sources=sources,
                data={
                    "query": query,
                    "num_results": len(results),
                    "index_name": self._index_name,
                },
            )

        except Exception as e:
            logger.error(
                "Vector Search error",
                error=str(e),
                endpoint=self._endpoint_name,
                index=self._index_name,
            )
            return ToolResult(
                content=f"Search failed: {e}",
                success=False,
                error=str(e),
            )

    def validate_arguments(self, arguments: dict[str, Any]) -> list[str]:
        """Validate search arguments.

        Args:
            arguments: Raw arguments from LLM

        Returns:
            List of error messages (empty if valid)
        """
        errors: list[str] = []

        # Required: query
        query = arguments.get("query")
        if not query:
            errors.append("'query' is required")
        elif not isinstance(query, str):
            errors.append("'query' must be a string")
        elif len(query) > 1000:
            errors.append("'query' must be 1000 characters or less")

        # Optional: num_results
        num_results = arguments.get("num_results")
        if num_results is not None:
            if not isinstance(num_results, int):
                errors.append("'num_results' must be an integer")
            elif num_results < 1 or num_results > 100:
                errors.append("'num_results' must be between 1 and 100")

        return errors


def create_vector_search_tools_from_config(config: Any) -> list[VectorSearchTool]:
    """Create VectorSearchTool instances from app configuration.

    Args:
        config: VectorSearchConfig from app_config

    Returns:
        List of VectorSearchTool instances, one per enabled endpoint
    """
    tools: list[VectorSearchTool] = []

    if not config or not getattr(config, "enabled", False):
        return tools

    endpoints = getattr(config, "endpoints", {})

    for name, endpoint_config in endpoints.items():
        if not getattr(endpoint_config, "enabled", True):
            logger.debug("Skipping disabled Vector Search endpoint", endpoint=name)
            continue

        try:
            tool = VectorSearchTool(
                endpoint_name=endpoint_config.endpoint_name,
                index_name=endpoint_config.index_name,
                columns=getattr(endpoint_config, "columns", None),
                tool_name=getattr(endpoint_config, "tool_name", None),
                description=getattr(endpoint_config, "description", None),
                num_results=getattr(endpoint_config, "num_results", 5),
                filters=getattr(endpoint_config, "filters", None),
            )
            tools.append(tool)
            logger.info(
                "Created Vector Search tool",
                tool_name=tool.definition.name,
            )
        except Exception as e:
            logger.warning(
                "Failed to create Vector Search tool",
                endpoint=name,
                error=str(e),
            )

    return tools
