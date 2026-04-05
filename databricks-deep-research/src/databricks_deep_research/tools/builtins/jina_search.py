"""Jina Search adapter — satisfies the ``SearchClient`` protocol from ``web_search.py``.

Calls the `Jina Search API <https://s.jina.ai/>`_ which returns search results
with full extracted page content per result.  Behind the scenes the API runs
the Jina Reader on each result URL, so a single search call yields both
snippets *and* full text.

No API key is required for basic usage (IP-based rate limiting).  A free key
(no credit card) unlocks higher rate limits and per-key tracking.

Usage::

    from databricks_deep_research.tools.builtins.jina_search import JinaSearchAdapter

    adapter = JinaSearchAdapter(api_key="jina_xxx")  # key is optional
    results = await adapter.search("NVIDIA revenue 2025", count=5)
"""

from __future__ import annotations

from databricks_deep_research.tools.builtins.web_search import SearchResult

__all__ = ["JinaSearchAdapter"]


class JinaSearchAdapter:
    """Jina Search client implementing the framework ``SearchClient`` protocol.

    Parameters:
        api_key: Optional Jina API key.  Works without one but rate limits
            are per-IP rather than per-key.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key

    async def search(
        self,
        query: str,
        *,
        count: int = 5,
        freshness: str | None = None,  # noqa: ARG002  — protocol conformance
    ) -> list[SearchResult]:
        """Execute a Jina web search and return :class:`SearchResult` objects.

        The *freshness* parameter is accepted for protocol conformance but
        ignored — the Jina Search API does not support time-based filtering.
        """
        import httpx

        headers: dict[str, str] = {"Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(
                "https://s.jina.ai/",
                params={"q": query, "num": count},
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

        results: list[SearchResult] = []
        for item in data.get("data", [])[:count]:
            content = item.get("content", "")
            description = item.get("description", "")
            results.append(
                SearchResult(
                    url=item.get("url", ""),
                    title=item.get("title", ""),
                    snippet=description or content[:200],
                    content=content or None,
                )
            )
        return results
