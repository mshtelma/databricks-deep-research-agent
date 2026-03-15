"""Brave Search adapter — satisfies the ``SearchClient`` protocol from ``web_search.py``.

A lightweight wrapper around the `Brave Web Search API
<https://brave.com/search/api/>`_ using ``httpx``.  The adapter is intentionally
minimal (~35 lines of logic) so it can live in the framework without pulling in
heavy dependencies.

Usage::

    from databricks_deep_research.tools.builtins.brave_search import BraveSearchAdapter

    adapter = BraveSearchAdapter(api_key="your-brave-key")
    results = await adapter.search("NVIDIA revenue 2025", count=5)
"""

from __future__ import annotations

from databricks_deep_research.tools.builtins.web_search import SearchResult

__all__ = ["BraveSearchAdapter"]


class BraveSearchAdapter:
    """Brave Search client implementing the framework ``SearchClient`` protocol.

    Parameters:
        api_key: Brave Search API subscription token.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def search(
        self,
        query: str,
        *,
        count: int = 10,
        freshness: str | None = None,
    ) -> list[SearchResult]:
        """Execute a Brave web search and return :class:`SearchResult` objects."""
        import httpx

        params: dict[str, str | int] = {"q": query, "count": count}
        if freshness:
            params["freshness"] = freshness

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params=params,
                headers={
                    "X-Subscription-Token": self._api_key,
                    "Accept": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()

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
