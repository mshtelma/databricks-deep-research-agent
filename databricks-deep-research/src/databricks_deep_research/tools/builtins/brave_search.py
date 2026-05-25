"""Brave Search adapter — satisfies the ``SearchClient`` protocol from ``web_search.py``.

A lightweight wrapper around the `Brave Web Search API
<https://brave.com/search/api/>`_ using ``httpx``.

The adapter includes three runtime concerns that the framework must enforce
process-wide so that parallel lane researchers do not blow through the Brave
API's per-key rate limit:

* **Process-wide semaphore** — caps in-flight Brave requests across the whole
  process. Initialized lazily from ``BRAVE_MAX_CONCURRENCY`` env var (default 4).
  Configured app-side via ``search.brave.max_concurrency`` in ``app.yaml``,
  which exports the env var at startup.
* **429 retry with exponential backoff** — a single rate-limit response no
  longer kills the calling lane. Honors ``Retry-After`` header when present.
  Configured via ``BRAVE_MAX_RETRIES`` (default 3).
* **Reusable httpx client** — per-adapter ``AsyncClient`` reused across calls
  for connection pooling, instead of spinning up a fresh client per query.

Usage::

    from databricks_deep_research.tools.builtins.brave_search import BraveSearchAdapter

    adapter = BraveSearchAdapter(api_key="your-brave-key")
    results = await adapter.search("NVIDIA revenue 2025", count=5)
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from typing import TYPE_CHECKING, Any

from databricks_deep_research.tools.builtins.web_search import SearchResult

if TYPE_CHECKING:
    import httpx

__all__ = ["BraveSearchAdapter"]

logger = logging.getLogger(__name__)

_BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"

# Process-wide concurrency cap. Lazily initialized on first use so the env
# var can be set after import (e.g. by the app bootstrap). The default
# targets a paid Brave plan (~10 QPS). For the free tier (1 QPS), set
# ``BRAVE_MAX_CONCURRENCY=1`` in the environment.
_BRAVE_SEMAPHORE: asyncio.Semaphore | None = None
_BRAVE_SEMAPHORE_LIMIT_DEFAULT = 10


def _get_semaphore() -> asyncio.Semaphore:
    global _BRAVE_SEMAPHORE
    if _BRAVE_SEMAPHORE is None:
        try:
            limit = int(os.environ.get("BRAVE_MAX_CONCURRENCY", _BRAVE_SEMAPHORE_LIMIT_DEFAULT))
        except ValueError:
            limit = _BRAVE_SEMAPHORE_LIMIT_DEFAULT
        if limit < 1:
            limit = 1
        _BRAVE_SEMAPHORE = asyncio.Semaphore(limit)
        logger.info("BRAVE_SEMAPHORE_INIT limit=%d", limit)
    return _BRAVE_SEMAPHORE


def _reset_semaphore_for_tests() -> None:
    """Test hook — drop the module-level semaphore so it re-initializes."""
    global _BRAVE_SEMAPHORE
    _BRAVE_SEMAPHORE = None


def _inter_call_jitter() -> float:
    """Return the configured intra-permit jitter (seconds, ≥0).

    Each Brave call sleeps a uniform random in
    ``[0, BRAVE_INTER_CALL_JITTER_SECONDS)`` after acquiring the semaphore
    and before firing the HTTP request. Smooths bursts INSIDE the permit
    window so consecutive callers don't pile onto Brave's API-side rate
    limiter back-to-back. Default 0.15s; set to 0 to disable.
    """
    try:
        value = float(os.environ.get("BRAVE_INTER_CALL_JITTER_SECONDS", "0.15"))
    except (TypeError, ValueError):
        return 0.15
    if value < 0:
        return 0.0
    return value


def _max_retries() -> int:
    try:
        value = int(os.environ.get("BRAVE_MAX_RETRIES", "3"))
    except ValueError:
        value = 3
    return max(1, value)


class BraveSearchAdapter:
    """Brave Search client implementing the framework ``SearchClient`` protocol.

    Parameters:
        api_key: Brave Search API subscription token.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._client: object | None = None  # httpx.AsyncClient, imported lazily

    def _get_client(self) -> "httpx.AsyncClient":
        import httpx

        client = self._client
        if client is None or getattr(client, "is_closed", False):
            client = httpx.AsyncClient(timeout=30.0)
            self._client = client
        return client  # type: ignore[return-value]

    async def aclose(self) -> None:
        client = self._client
        if client is not None and not getattr(client, "is_closed", True):
            await client.aclose()  # type: ignore[attr-defined]
        self._client = None

    async def search(
        self,
        query: str,
        *,
        count: int = 10,
        freshness: str | None = None,
    ) -> list[SearchResult]:
        """Execute a Brave web search and return :class:`SearchResult` objects.

        Retries up to ``BRAVE_MAX_RETRIES`` times on 429 responses with
        exponential backoff (honoring ``Retry-After`` when present). All calls
        are gated by a process-wide semaphore so parallel lanes cannot burst.
        """
        import httpx

        params: dict[str, str | int] = {"q": query, "count": count}
        if freshness:
            params["freshness"] = freshness
        headers = {
            "X-Subscription-Token": self._api_key,
            "Accept": "application/json",
        }

        async with _get_semaphore():
            client = self._get_client()
            max_attempts = _max_retries()
            jitter_max = _inter_call_jitter()
            if jitter_max > 0:
                await asyncio.sleep(random.uniform(0.0, jitter_max))
            data: dict[str, Any] | None = None
            for attempt in range(max_attempts):
                try:
                    resp = await client.get(_BRAVE_URL, params=params, headers=headers)
                except httpx.RequestError as exc:
                    if attempt == max_attempts - 1:
                        raise
                    delay = (2 ** attempt) + random.uniform(0, 0.5)
                    logger.warning(
                        "BRAVE_REQUEST_ERROR attempt=%d/%d err=%s sleep=%.2fs",
                        attempt + 1, max_attempts, exc, delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                if resp.status_code != 429:
                    resp.raise_for_status()
                    data = resp.json()
                    break

                # 429: respect Retry-After if present, else exponential backoff.
                if attempt == max_attempts - 1:
                    logger.warning(
                        "BRAVE_429_EXHAUSTED query=%r attempts=%d", query, max_attempts,
                    )
                    resp.raise_for_status()
                try:
                    retry_after = float(resp.headers.get("Retry-After", "0"))
                except (TypeError, ValueError):
                    retry_after = 0.0
                delay = retry_after if retry_after > 0 else (2 ** attempt)
                delay += random.uniform(0, 0.5)  # jitter
                logger.warning(
                    "BRAVE_429 attempt=%d/%d retry_after=%.2fs query=%r",
                    attempt + 1, max_attempts, delay, query,
                )
                logger.info(
                    "DR_LEAK_TRACE phase=brave_throttled "
                    "attempt=%d/%d retry_after=%.2fs query_head=%r",
                    attempt + 1, max_attempts, delay, query[:120],
                )
                await asyncio.sleep(delay)

        if data is None:
            return []
        results: list[SearchResult] = []
        for item in data.get("web", {}).get("results", [])[:count]:
            results.append(
                SearchResult(
                    url=item.get("url", ""),
                    title=item.get("title", ""),
                    snippet=item.get("description", ""),
                )
            )
        return results
