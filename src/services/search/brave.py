"""Brave Search API client with rate limiting."""

import asyncio
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

from src.core.app_config import get_app_config
from src.core.config import get_settings
from src.core.exceptions import ExternalServiceError, RateLimitError
from src.core.logging_utils import (
    get_logger,
    log_search_error,
    log_search_request,
    log_search_response,
    truncate,
)
from src.services.search.domain_filter import DomainFilter

logger = get_logger(__name__)

BRAVE_API_URL = "https://api.search.brave.com/res/v1/web/search"


@dataclass
class SearchResult:
    """A single search result."""

    url: str
    title: str
    snippet: str
    relevance_score: float | None = None


@dataclass
class SearchResponse:
    """Response from Brave Search API."""

    results: list[SearchResult]
    query: str
    total_results: int


class BraveSearchClient:
    """Brave Search API client with rate limiting."""

    def __init__(
        self,
        api_key: str | None = None,
        requests_per_second: float | None = None,
        default_result_count: int | None = None,
        default_freshness: str | None = None,
    ):
        """Initialize Brave Search client.

        Args:
            api_key: Brave Search API key. If None, uses settings.
            requests_per_second: Rate limit for API calls. If None, uses central config.
            default_result_count: Default number of results. If None, uses central config.
            default_freshness: Default freshness filter. If None, uses central config.
        """
        settings = get_settings()
        self._api_key = api_key or settings.brave_api_key

        if not self._api_key:
            logger.warning("Brave API key not configured")

        # Load defaults from central config
        search_config = get_app_config().search
        brave_config = search_config.brave
        self._requests_per_second = requests_per_second or brave_config.requests_per_second
        self._default_result_count = default_result_count or brave_config.default_result_count
        self._default_freshness = default_freshness or brave_config.freshness

        # Domain filtering
        self._domain_filter = DomainFilter(search_config.domain_filter)

        # Rate limiting
        self._last_request: datetime | None = None
        self._lock = asyncio.Lock()

        # HTTP client
        self._client = httpx.AsyncClient(
            timeout=30.0,
            headers={"Accept": "application/json"},
        )

    async def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        async with self._lock:
            if self._last_request:
                elapsed = (datetime.now(UTC) - self._last_request).total_seconds()
                min_interval = 1.0 / self._requests_per_second
                if elapsed < min_interval:
                    await asyncio.sleep(min_interval - elapsed)
            self._last_request = datetime.now(UTC)

    async def search(
        self,
        query: str,
        count: int | None = None,
        freshness: str | None = None,
    ) -> SearchResponse:
        """Execute a web search.

        Args:
            query: Search query string.
            count: Number of results to return (max 20). Uses config default if None.
            freshness: Time filter ("pd" = past day, "pw" = past week, etc.).
                      Uses config default if None.

        Returns:
            SearchResponse with results.

        Raises:
            ExternalServiceError: If search fails.
            RateLimitError: If rate limited by Brave.
        """
        if not self._api_key:
            raise ExternalServiceError("Brave Search", "API key not configured")

        # Use defaults from config if not specified
        effective_count = count if count is not None else self._default_result_count
        effective_freshness = freshness if freshness is not None else self._default_freshness

        # Log the search request
        log_search_request(logger, query=query, count=effective_count)

        await self._rate_limit()

        params: dict[str, str | int] = {
            "q": query,
            "count": min(effective_count, 20),
        }
        if effective_freshness:
            params["freshness"] = effective_freshness

        start_time = time.perf_counter()

        try:
            response = await self._client.get(
                BRAVE_API_URL,
                params=params,
                headers={"X-Subscription-Token": self._api_key},
            )

            duration_ms = (time.perf_counter() - start_time) * 1000

            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                logger.warning(
                    "SEARCH_RATE_LIMITED",
                    query=truncate(query, 50),
                    retry_after=retry_after,
                )
                raise RateLimitError(retry_after=retry_after)

            response.raise_for_status()

            try:
                data = response.json()
            except ValueError as e:
                log_search_error(logger, query=query, error=e, status_code=response.status_code)
                raise ExternalServiceError("Brave Search", f"Invalid JSON response: {e}") from e

            # Parse results
            results = []
            web_results = data.get("web", {}).get("results", [])

            for i, item in enumerate(web_results):
                results.append(
                    SearchResult(
                        url=item.get("url", ""),
                        title=item.get("title", ""),
                        snippet=item.get("description", ""),
                        relevance_score=1.0 - (i * 0.1) if i < 10 else 0.1,
                    )
                )

            # Apply domain filtering
            filtered_count = len(results)
            if self._domain_filter.is_active:
                results = [
                    r for r in results if self._domain_filter.is_allowed(r.url).allowed
                ]
                filtered_count = filtered_count - len(results)
                if filtered_count > 0:
                    logger.debug(
                        "SEARCH_RESULTS_FILTERED",
                        filtered_count=filtered_count,
                        remaining_count=len(results),
                    )

            # Log the response
            urls = [r.url for r in results]
            log_search_response(
                logger,
                query=query,
                result_count=len(results),
                urls=urls,
                duration_ms=duration_ms,
            )

            return SearchResponse(
                results=results,
                query=query,
                total_results=len(results),
            )

        except httpx.HTTPStatusError as e:
            log_search_error(logger, query=query, error=e, status_code=e.response.status_code)
            raise ExternalServiceError("Brave Search", f"HTTP {e.response.status_code}") from e
        except httpx.RequestError as e:
            log_search_error(logger, query=query, error=e)
            raise ExternalServiceError("Brave Search", str(e)) from e

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
