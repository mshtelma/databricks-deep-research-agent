"""Web search tool wrapper.

Provides both:
1. Legacy functional interface: `web_search()` async function
2. ResearchTool protocol: `WebSearchTool` class for plugin system
"""

from dataclasses import dataclass
from typing import Any

import mlflow
from mlflow.entities import SpanType

from deep_research.agent.tools.base import (
    ResearchContext,
    ResearchTool,
    ToolDefinition,
    ToolResult,
)
from deep_research.agent.tools.url_registry import UrlRegistry
from deep_research.core.logging_utils import get_logger, truncate
from deep_research.core.tracing_constants import (
    ATTR_SEARCH_COUNT,
    ATTR_SEARCH_QUERY,
    ATTR_SEARCH_RESULTS_COUNT,
    ATTR_SEARCH_TOP_URLS,
    list_to_attr,
    tool_span_name,
    truncate_for_attr,
)
from deep_research.services.search.brave import BraveSearchClient

logger = get_logger(__name__)


@dataclass
class WebSearchResult:
    """Result from web search tool."""

    url: str
    title: str
    snippet: str
    relevance_score: float


@dataclass
class WebSearchOutput:
    """Output from web search tool."""

    results: list[WebSearchResult]
    query: str
    total_results: int


async def web_search(
    query: str,
    count: int = 10,
    freshness: str | None = None,
    *,
    client: BraveSearchClient,
    context: str | None = None,
) -> WebSearchOutput:
    """Execute a web search using Brave Search API.

    Args:
        query: Search query string.
        count: Number of results to return (max 20).
        freshness: Time filter ("pd" = past day, "pw" = past week, etc.)
        client: BraveSearchClient instance (injected via DI).
        context: Optional context for span naming (e.g., "step_1", "background").

    Returns:
        WebSearchOutput with search results.
    """
    span_name = tool_span_name("web_search", context)

    with mlflow.start_span(name=span_name, span_type=SpanType.TOOL) as span:
        span.set_attributes({
            ATTR_SEARCH_QUERY: truncate_for_attr(query, 150),
            ATTR_SEARCH_COUNT: count,
            "search.freshness": freshness or "any",
        })

        search_client = client

        logger.info(
            "WEB_SEARCH_START",
            query=truncate(query, 80),
            count=count,
            freshness=freshness,
        )

        response = await search_client.search(
            query=query,
            count=count,
            freshness=freshness,
        )

        results = [
            WebSearchResult(
                url=r.url,
                title=r.title,
                snippet=r.snippet,
                relevance_score=r.relevance_score or 0.5,
            )
            for r in response.results
        ]

        # Log results summary
        urls = [r.url for r in results[:5]]  # First 5 URLs
        logger.info(
            "WEB_SEARCH_COMPLETE",
            query=truncate(query, 60),
            results=len(results),
            top_urls=urls,
        )

        # Set output attributes
        span.set_attributes({
            ATTR_SEARCH_RESULTS_COUNT: len(results),
            ATTR_SEARCH_TOP_URLS: list_to_attr(urls, max_items=5),
        })

        return WebSearchOutput(
            results=results,
            query=query,
            total_results=len(results),
        )


def format_search_results_indexed(
    output: WebSearchOutput,
    registry: UrlRegistry,
) -> str:
    """Format search results for LLM with indices (no URLs).

    This is a security feature - URLs are hidden from the LLM.
    The LLM sees only indices which are resolved internally.

    Args:
        output: Search output with results
        registry: URL registry to register results

    Returns:
        Formatted string with indexed results (no URLs)
    """
    if not output.results:
        return "No search results found. Try a different query."

    formatted = []
    for result in output.results:
        # Register URL and get index
        index = registry.register(
            url=result.url,
            title=result.title,
            snippet=result.snippet,
            relevance_score=result.relevance_score,
        )

        # Format WITHOUT URL - only index, title, snippet
        formatted.append(f"[{index}] **{result.title}**\n    {result.snippet}")

    return "\n\n".join(formatted)


# =============================================================================
# ResearchTool Protocol Implementation
# =============================================================================


class WebSearchTool:
    """
    Web search tool implementing the ResearchTool protocol.

    This class wraps the functional `web_search()` for use with the plugin system
    and ToolRegistry. It requires a BraveSearchClient to be injected at construction.

    Example:
        client = BraveSearchClient(api_key="...")
        tool = WebSearchTool(client)
        registry.register(tool)
    """

    def __init__(self, client: BraveSearchClient) -> None:
        """Initialize the web search tool.

        Args:
            client: BraveSearchClient instance for executing searches.
        """
        self._client = client
        self._definition = ToolDefinition(
            name="web_search",
            description=(
                "Search the web for information. Returns numbered results with titles and snippets. "
                "Use the result INDEX numbers (0, 1, 2, etc.) to select sources for crawling."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A specific, focused search query. "
                            "Include entities, dates, or metrics for best results. "
                            "Example: 'Apple Q4 2024 revenue earnings report'"
                        ),
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5, max: 10)",
                        "default": 5,
                    },
                    "freshness": {
                        "type": "string",
                        "description": "Time filter: 'pd' (past day), 'pw' (past week), 'pm' (past month), or None for any time",
                        "enum": ["pd", "pw", "pm"],
                    },
                },
                "required": ["query"],
            },
        )

    @property
    def definition(self) -> ToolDefinition:
        """Return tool definition for LLM function calling."""
        return self._definition

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ResearchContext,
    ) -> ToolResult:
        """Execute web search and return formatted results.

        Args:
            arguments: Tool arguments containing 'query', optional 'count', 'freshness'
            context: Research context with registries and identity

        Returns:
            ToolResult with formatted search results and source tracking
        """
        query = arguments.get("query", "")
        count = arguments.get("count", 5)
        freshness = arguments.get("freshness")

        try:
            # Execute search using the functional interface
            output = await web_search(
                query=query,
                count=count,
                freshness=freshness,
                client=self._client,
                context=f"chat_{context.chat_id}",
            )

            # Build sources list for citation tracking
            sources: list[dict[str, Any]] = []
            formatted_results: list[str] = []

            for idx, result in enumerate(output.results):
                sources.append({
                    "type": "web",
                    "url": result.url,
                    "title": result.title,
                    "snippet": result.snippet,
                    "relevance_score": result.relevance_score,
                    "search_index": idx,
                })
                # Format WITHOUT URL for security - LLM sees only indices
                formatted_results.append(
                    f"[{idx}] **{result.title}**\n    {result.snippet}"
                )

            if not formatted_results:
                content = "No search results found. Try a different query."
            else:
                content = "\n\n".join(formatted_results)

            return ToolResult(
                content=content,
                success=True,
                sources=sources,
                data={
                    "query": query,
                    "total_results": output.total_results,
                    "count": count,
                },
            )

        except Exception as e:
            logger.error("WEB_SEARCH_ERROR", error=str(e), query=truncate(query, 60))
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
        elif len(query) > 500:
            errors.append("'query' must be 500 characters or less")

        # Optional: count
        count = arguments.get("count")
        if count is not None:
            if not isinstance(count, int):
                errors.append("'count' must be an integer")
            elif count < 1 or count > 20:
                errors.append("'count' must be between 1 and 20")

        # Optional: freshness
        freshness = arguments.get("freshness")
        if freshness is not None:
            valid_values = {"pd", "pw", "pm"}
            if freshness not in valid_values:
                errors.append(f"'freshness' must be one of: {valid_values}")

        return errors
