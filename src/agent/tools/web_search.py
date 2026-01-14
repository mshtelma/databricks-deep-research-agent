"""Web search tool wrapper."""

from dataclasses import dataclass

import mlflow
from mlflow.entities import SpanType

from src.agent.tools.url_registry import UrlRegistry
from src.core.logging_utils import get_logger, truncate
from src.core.tracing_constants import (
    ATTR_SEARCH_COUNT,
    ATTR_SEARCH_QUERY,
    ATTR_SEARCH_RESULTS_COUNT,
    ATTR_SEARCH_TOP_URLS,
    list_to_attr,
    tool_span_name,
    truncate_for_attr,
)
from src.services.search.brave import BraveSearchClient

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
