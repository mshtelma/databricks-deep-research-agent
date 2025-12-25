"""Web search tool wrapper."""

from dataclasses import dataclass

import mlflow

from src.core.logging_utils import get_logger, truncate
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


@mlflow.trace(name="web_search", span_type="TOOL")
async def web_search(
    query: str,
    count: int = 10,
    freshness: str | None = None,
    *,
    client: BraveSearchClient,
) -> WebSearchOutput:
    """Execute a web search using Brave Search API.

    Args:
        query: Search query string.
        count: Number of results to return (max 20).
        freshness: Time filter ("pd" = past day, "pw" = past week, etc.)
        client: BraveSearchClient instance (injected via DI).

    Returns:
        WebSearchOutput with search results.
    """
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

    return WebSearchOutput(
        results=results,
        query=query,
        total_results=len(results),
    )
