"""Vector Search Tool for Databricks Vector Search.

Provides semantic search over Vector Search indexes configured via app.yaml.
Each configured endpoint creates a separate tool instance with a unique name.

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

from dataclasses import dataclass
from typing import Any

from deep_research.agent.tools.base import (
    ResearchContext,
    ResearchTool,
    ToolDefinition,
    ToolResult,
)
from deep_research.core.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class VectorSearchResult:
    """A single result from Vector Search."""

    title: str
    content: str
    url: str | None
    score: float
    metadata: dict[str, Any]


class VectorSearchTool:
    """
    Vector Search tool implementing the ResearchTool protocol.

    Queries a Databricks Vector Search index for semantically similar documents.
    Tool name is generated as 'search_{endpoint_name}' to allow multiple indexes.

    Requires Databricks SDK with proper authentication (WorkspaceClient).
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

        # Lazy-loaded client
        self._client: Any = None
        self._index: Any = None

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
        )

    def _get_index(self) -> Any:
        """Get or create Vector Search index reference."""
        if self._index is None:
            try:
                from databricks.vector_search.client import VectorSearchClient

                self._client = VectorSearchClient()
                self._index = self._client.get_index(
                    endpoint_name=self._endpoint_name,
                    index_name=self._index_name,
                )
                logger.info(
                    "Vector Search index initialized",
                    endpoint=self._endpoint_name,
                    index=self._index_name,
                )
            except ImportError:
                raise ImportError(
                    "databricks-vector-search package not installed. "
                    "Install with: pip install databricks-vector-search"
                )
        return self._index

    @property
    def definition(self) -> ToolDefinition:
        """Return tool definition for LLM function calling."""
        return self._definition

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ResearchContext,
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
            index = self._get_index()

            # Execute similarity search
            response = index.similarity_search(
                query_text=query,
                columns=self._columns,
                num_results=num_results,
                filters=self._filters if self._filters else None,
            )

            # Parse response
            results = self._parse_response(response)

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

        except ImportError as e:
            logger.error("Vector Search SDK not available", error=str(e))
            return ToolResult(
                content="Vector Search is not available. SDK not installed.",
                success=False,
                error=str(e),
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

    def _parse_response(self, response: dict[str, Any]) -> list[VectorSearchResult]:
        """Parse Vector Search response into structured results.

        Args:
            response: Raw response from similarity_search()

        Returns:
            List of VectorSearchResult objects
        """
        results: list[VectorSearchResult] = []

        # Get column mapping from manifest
        manifest = response.get("manifest", {})
        columns = [col["name"] for col in manifest.get("columns", [])]

        # Get column indices
        col_indices: dict[str, int] = {name: idx for idx, name in enumerate(columns)}

        # Parse data rows
        data = response.get("result", {}).get("data_array", [])

        for row in data:
            # Extract known columns with defaults
            title = self._get_column_value(row, col_indices, "title", "Untitled")
            content = self._get_column_value(row, col_indices, "content", "")
            url = self._get_column_value(row, col_indices, "url", None)
            score = self._get_column_value(row, col_indices, "score", 0.0)

            # Collect remaining columns as metadata
            metadata: dict[str, Any] = {}
            for col_name, col_idx in col_indices.items():
                if col_name not in ("title", "content", "url", "score"):
                    if col_idx < len(row):
                        metadata[col_name] = row[col_idx]

            results.append(VectorSearchResult(
                title=title,
                content=content,
                url=url,
                score=score,
                metadata=metadata,
            ))

        return results

    def _get_column_value(
        self,
        row: list[Any],
        col_indices: dict[str, int],
        column: str,
        default: Any,
    ) -> Any:
        """Safely get column value from row."""
        idx = col_indices.get(column)
        if idx is not None and idx < len(row):
            return row[idx] if row[idx] is not None else default
        return default

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
