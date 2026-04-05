"""Jina Reader adapter — satisfies the ``ContentCrawler`` protocol from ``web_crawl.py``.

Calls the `Jina Reader API <https://r.jina.ai/>`_ to extract clean,
LLM-friendly text from any URL.  Useful as an alternative to the default
httpx + trafilatura pipeline, or for mixing providers (e.g. Brave search
with Jina crawl).

Usage::

    from databricks_deep_research.tools.builtins.jina_crawl import JinaCrawlAdapter

    crawler = JinaCrawlAdapter(api_key="jina_xxx")  # key is optional
    text, title = await crawler("https://example.com/article")
"""

from __future__ import annotations

__all__ = ["JinaCrawlAdapter"]


class JinaCrawlAdapter:
    """Jina Reader client implementing the framework ``ContentCrawler`` protocol.

    Parameters:
        api_key: Optional Jina API key.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key

    async def __call__(self, url: str) -> tuple[str, str | None]:
        """Fetch *url* via Jina Reader and return ``(extracted_text, title)``."""
        import httpx

        headers: dict[str, str] = {"Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(
                f"https://r.jina.ai/{url}",
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

        # Handle both wrapped {data: {…}} and direct {url, title, content}.
        item = data.get("data", data) if isinstance(data.get("data"), dict) else data
        content = item.get("content", "")
        title = item.get("title") or None

        if not content:
            raise ValueError(f"Jina Reader returned empty content for {url}")

        return content, title
