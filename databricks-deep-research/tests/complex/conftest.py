"""Shared fixtures for complex, long-running framework tests.

Complex tests run the FULL research pipeline with higher iteration
counts, verifying deep research quality end-to-end.

Requirements:
- DATABRICKS_HOST + DATABRICKS_TOKEN (or DATABRICKS_CONFIG_PROFILE)
- BRAVE_API_KEY
- Significant time (3-10 minutes per test)

Run with:
    cd databricks-deep-research
    uv run pytest tests/complex -v -s --timeout=600
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from databricks_deep_research.llm.client import FrameworkLLMClient
from databricks_deep_research.tools.builtins.web_crawl import WebCrawlTool
from databricks_deep_research.tools.builtins.web_search import SearchResult, WebSearchTool
from databricks_deep_research.tools.registry import ToolRegistry
from tests._databricks_auth import create_async_openai_client, has_databricks_credential_hint

# ---------------------------------------------------------------------------
# Credential checks
# ---------------------------------------------------------------------------


def _has_databricks_creds() -> bool:
    return has_databricks_credential_hint()


def _has_brave_key() -> bool:
    import os

    return bool(os.getenv("BRAVE_API_KEY"))


requires_all_credentials = pytest.mark.skipif(
    not (_has_databricks_creds() and _has_brave_key()),
    reason="Both Databricks and Brave credentials required",
)


# ---------------------------------------------------------------------------
# Examples directory
# ---------------------------------------------------------------------------


EXAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "examples"


@pytest.fixture
def examples_dir() -> Path:
    return EXAMPLES_DIR


# ---------------------------------------------------------------------------
# Databricks client creation
# ---------------------------------------------------------------------------


def _create_openai_client():
    """Create AsyncOpenAI pointing to Databricks serving endpoints."""
    return create_async_openai_client()


# ---------------------------------------------------------------------------
# LLM client fixture — uses better model for complex tests
# ---------------------------------------------------------------------------

_COMPLEX_MODEL = os.getenv("FRAMEWORK_COMPLEX_MODEL", "databricks-claude-haiku-4-5")
_ANALYTICAL_MODEL = os.getenv("FRAMEWORK_ANALYTICAL_MODEL", "databricks-claude-haiku-4-5")


@pytest.fixture
async def llm_client() -> FrameworkLLMClient:
    """FrameworkLLMClient with tiered model mapping for complex tests."""
    try:
        openai_client = _create_openai_client()
    except Exception as exc:
        pytest.skip(f"Databricks auth unavailable: {exc}")

    client = FrameworkLLMClient(
        openai_client=openai_client,
        model_mapping={
            "simple": "databricks-claude-haiku-4-5",
            "analytical": _ANALYTICAL_MODEL,
            "complex": _COMPLEX_MODEL,
        },
        client_provider=_create_openai_client,
    )
    try:
        yield client
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# Brave search adapter
# ---------------------------------------------------------------------------


class BraveSearchAdapter:
    """Brave Search → framework SearchClient adapter."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def search(
        self,
        query: str,
        *,
        count: int = 10,
        freshness: str | None = None,
    ) -> list[SearchResult]:
        import httpx

        params: dict[str, Any] = {"q": query, "count": count}
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


# ---------------------------------------------------------------------------
# Tool fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tool_registry() -> ToolRegistry:
    """Full tool registry with web_search + web_crawl."""
    api_key = os.getenv("BRAVE_API_KEY", "")
    if not api_key:
        pytest.skip("BRAVE_API_KEY not set")

    search_tool = WebSearchTool(BraveSearchAdapter(api_key), max_results=5)
    crawl_tool = WebCrawlTool(timeout=15.0, max_content_length=30_000)

    registry = ToolRegistry()
    registry.register_builtin("web_search", search_tool)
    registry.register_builtin("web_crawl", crawl_tool)
    return registry
