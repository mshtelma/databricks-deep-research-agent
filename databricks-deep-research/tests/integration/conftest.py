"""Shared fixtures for framework integration tests.

Integration tests run against REAL Databricks LLM endpoints and Brave Search API.
They verify the framework works end-to-end without mocks.

Requirements:
- DATABRICKS_HOST + DATABRICKS_TOKEN (or DATABRICKS_CONFIG_PROFILE)
- BRAVE_API_KEY for web search tests

Run with:
    cd databricks-deep-research
    uv run pytest tests/integration -v -s
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import httpx
from openai import APITimeoutError
from typing import Any

import pytest
from openai import AsyncOpenAI

from databricks_deep_research.llm.client import FrameworkLLMClient
from databricks_deep_research.tools.builtins.genie import DatabricksGenieTool
from databricks_deep_research.tools.builtins.web_crawl import WebCrawlTool
from databricks_deep_research.tools.builtins.web_search import (
    SearchResult,
    WebSearchTool,
)
from databricks_deep_research.tools.protocol import (
    SourceInfo,
    ToolContext,
    ToolDefinition,
    ToolResult,
)
from databricks_deep_research.tools.registry import ToolRegistry
from tests._databricks_auth import (
    create_async_openai_client,
    has_databricks_credential_hint,
    resolve_databricks_auth,
)

# ---------------------------------------------------------------------------
# Credential detection
# ---------------------------------------------------------------------------


def _has_databricks_creds() -> bool:
    return has_databricks_credential_hint()


def _has_brave_key() -> bool:
    return bool(os.getenv("BRAVE_API_KEY"))


requires_databricks = pytest.mark.skipif(
    not _has_databricks_creds(),
    reason="Databricks credentials not configured",
)
requires_brave = pytest.mark.skipif(
    not _has_brave_key(),
    reason="Brave API key not configured",
)
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
# Databricks AsyncOpenAI client
# ---------------------------------------------------------------------------


def _create_openai_client() -> AsyncOpenAI:
    """Create an AsyncOpenAI client authenticated against Databricks."""
    return create_async_openai_client()


# ---------------------------------------------------------------------------
# FrameworkLLMClient fixture
# ---------------------------------------------------------------------------

# Use the same model for all tiers in integration tests (fast + cheap).
_TEST_MODEL = os.getenv(
    "FRAMEWORK_TEST_MODEL", "databricks-claude-haiku-4-5"
)


@pytest.fixture
async def llm_client() -> FrameworkLLMClient:
    """Create a FrameworkLLMClient backed by Databricks endpoints."""
    try:
        openai_client = _create_openai_client()
    except Exception as exc:
        pytest.skip(f"Databricks auth unavailable: {exc}")

    client = FrameworkLLMClient(
        openai_client=openai_client,
        model_mapping={
            "simple": _TEST_MODEL,
            "analytical": _TEST_MODEL,
            "complex": _TEST_MODEL,
        },
        client_provider=_create_openai_client,
    )
    try:
        yield client
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# Brave Search adapter (satisfies framework SearchClient protocol)
# ---------------------------------------------------------------------------


from databricks_deep_research.tools.builtins.brave_search import BraveSearchAdapter


@pytest.fixture
def brave_adapter() -> BraveSearchAdapter:
    """Create a Brave search adapter for the framework."""
    api_key = os.getenv("BRAVE_API_KEY", "")
    if not api_key:
        pytest.skip("BRAVE_API_KEY not set")
    return BraveSearchAdapter(api_key)


# ---------------------------------------------------------------------------
# Tool fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def web_search_tool(brave_adapter: BraveSearchAdapter) -> WebSearchTool:
    """WebSearchTool wired to real Brave Search."""
    return WebSearchTool(brave_adapter, max_results=5)


@pytest.fixture
def web_crawl_tool() -> WebCrawlTool:
    """WebCrawlTool using default httpx + trafilatura pipeline."""
    return WebCrawlTool(timeout=15.0, max_content_length=20_000)


@pytest.fixture
def tool_registry(web_search_tool: WebSearchTool, web_crawl_tool: WebCrawlTool) -> ToolRegistry:
    """ToolRegistry with web_search and web_crawl registered."""
    registry = ToolRegistry()
    registry.register_builtin("web_search", web_search_tool)
    registry.register_builtin("web_crawl", web_crawl_tool)
    return registry


# ---------------------------------------------------------------------------
# Mixed tool registry (web + enterprise) for integration tests
# ---------------------------------------------------------------------------


@pytest.fixture
def mixed_tool_registry(
    web_search_tool: WebSearchTool,
    web_crawl_tool: WebCrawlTool,
    genie_tool: Any,
    vector_search_tool_mock: Any,
    knowledge_assistant_tool: Any,
) -> ToolRegistry:
    """ToolRegistry with both web (builtin) and enterprise (external) tools."""
    registry = ToolRegistry()
    registry.register_builtin("web_search", web_search_tool)
    registry.register_builtin("web_crawl", web_crawl_tool)
    registry.register_external("genie", genie_tool)
    registry.register_external("vector_search", vector_search_tool_mock)
    registry.register_external("knowledge_assistant", knowledge_assistant_tool)
    return registry


# ---------------------------------------------------------------------------
# Databricks WorkspaceClient helper
# ---------------------------------------------------------------------------


def _create_workspace_client() -> Any:
    """Create a Databricks WorkspaceClient.

    Supports two auth modes (mirrors _create_openai_client):
    1. Direct token: DATABRICKS_TOKEN + DATABRICKS_HOST
    2. Profile-based OAuth: DATABRICKS_CONFIG_PROFILE
    """
    from databricks.sdk import WorkspaceClient

    auth = resolve_databricks_auth()
    return WorkspaceClient(host=auth.host, token=auth.token)


# ---------------------------------------------------------------------------
# Real Vector Search tool (uses Databricks SDK against a live index)
# ---------------------------------------------------------------------------

_VS_INDEX_NAME = os.getenv(
    "FRAMEWORK_TEST_VS_INDEX", "main.dbdemos_ai_agent.earnings_vs_index"
)
_TRANSCRIPT_VS_INDEX_NAME = os.getenv(
    "FRAMEWORK_TEST_TRANSCRIPT_VS_INDEX",
    "main.dbdemos_ai_agent.transcript_vs_index",
)
_KB_VS_INDEX_NAME = os.getenv(
    "FRAMEWORK_TEST_KB_VS_INDEX",
    "",
).strip()
_GENIE_SPACE_ID = os.getenv(
    "FRAMEWORK_TEST_GENIE_SPACE_ID",
    "01f10143a8721de393e79d4396185977",
)

_VS_PARAMS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Natural language search query",
        },
        "num_results": {
            "type": "integer",
            "description": "Number of results to return (default 5)",
        },
    },
    "required": ["query"],
}


class RealVectorSearchTool:
    """Framework ResearchTool backed by a real Databricks Vector Search index.

    Follows the same manifest→columns→data_array parsing pattern used in the
    app's VectorSearchQueryService._parse_response.
    """

    def __init__(
        self,
        workspace_client: Any,
        index_name: str,
        columns: list[str] | None = None,
        num_results: int = 5,
        name: str = "vector_search",
    ) -> None:
        self._client = workspace_client
        self._index_name = index_name
        self._columns: list[str] | None = columns
        self._num_results = num_results
        self._name = name

    # -- ResearchTool protocol --

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description=(
                f"Semantic search over the '{self._index_name}' vector index. "
                "Returns relevant document chunks ranked by similarity."
            ),
            parameters=_VS_PARAMS_SCHEMA,
            source_type="enterprise",
            source_kind="vector_index",
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = arguments.get("query")
        if not query or not isinstance(query, str):
            raise ValueError("'query' must be a non-empty string")
        num = arguments.get("num_results", self._num_results)
        return {"query": query, "num_results": int(num)}

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        query = arguments["query"]
        num_results = arguments.get("num_results", self._num_results)

        # Discover columns on first call if not provided
        if self._columns is None:
            self._columns = await self._discover_columns()

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.vector_search_indexes.query_index(
                    index_name=self._index_name,
                    query_text=query,
                    columns=self._columns,
                    num_results=num_results,
                ),
            )
        except Exception as exc:
            return ToolResult(
                content=f"Vector search failed: {exc}",
                success=False,
                error=str(exc),
            )

        return self._parse_response(response, query)

    # -- Internals --

    async def _discover_columns(self) -> list[str]:
        """Discover queryable columns from the index metadata."""
        try:
            index_info = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.vector_search_indexes.get_index(
                    self._index_name,
                ),
            )
        except Exception:
            # Fallback: let the API decide
            return []

        columns: list[str] = []
        primary_key = getattr(index_info, "primary_key", None)
        if primary_key:
            columns.append(str(primary_key))

        delta_sync = getattr(index_info, "delta_sync_index_spec", None)
        if delta_sync:
            pk = getattr(delta_sync, "primary_key_columns", None)
            if pk:
                for col in (pk if isinstance(pk, list) else [pk]):
                    if col and col not in columns:
                        columns.append(str(col))

            src_cols = getattr(delta_sync, "embedding_source_columns", None) or []
            for sc in src_cols:
                col_name = (
                    sc.get("name") if isinstance(sc, dict)
                    else getattr(sc, "name", None)
                )
                if col_name and col_name not in columns:
                    columns.append(col_name)

        return columns if columns else []

    def _parse_response(self, response: Any, query: str) -> ToolResult:
        """Parse the VS query response using manifest→columns→data_array."""
        # Extract manifest and data_array
        manifest = (
            response.manifest if hasattr(response, "manifest")
            else response.get("manifest") if isinstance(response, dict)
            else None
        )
        result_data = (
            response.result if hasattr(response, "result")
            else response.get("result") if isinstance(response, dict)
            else None
        )

        if manifest is None or result_data is None:
            return ToolResult(
                content="No results returned from vector search.",
                success=True,
                sources=[],
            )

        # Build column index mapping
        raw_columns = (
            manifest.columns if hasattr(manifest, "columns")
            else manifest.get("columns", [])
        )
        col_names: list[str] = []
        for c in raw_columns:
            name = c.name if hasattr(c, "name") else c.get("name", "")
            col_names.append(name)
        col_indices = {name: i for i, name in enumerate(col_names)}

        data_array = (
            result_data.data_array if hasattr(result_data, "data_array")
            else result_data.get("data_array", [])
        )

        if not data_array:
            return ToolResult(
                content="Vector search returned no matching documents.",
                success=True,
                sources=[],
            )

        # Parse rows
        lines: list[str] = []
        sources: list[SourceInfo] = []
        for idx, row in enumerate(data_array):
            def _col(
                name: str,
                default: Any = "",
                row_values: list[Any] = row,
            ) -> Any:
                i = col_indices.get(name)
                if i is not None and i < len(row_values):
                    return (
                        row_values[i]
                        if row_values[i] is not None
                        else default
                    )
                return default

            score = _col("score", 0.0)
            # Try common content column names
            content = ""
            for cname in ("content", "text", "chunk_text", "page_content"):
                content = str(_col(cname, ""))
                if content:
                    break
            # If no known content column, use the first non-score, non-id column
            if not content:
                for cname in col_names:
                    if cname not in ("score", "id", "pk"):
                        val = str(_col(cname, ""))
                        if val and len(val) > 20:
                            content = val
                            break

            title = str(_col("title", ""))
            if not title:
                title = content[:80] + "..." if len(content) > 80 else content
            url = str(_col("url", ""))
            doc_id = str(_col("id", _col("pk", str(idx))))

            lines.append(
                f"[{idx + 1}] **{title}** (score: {score})\n"
                f"    {content[:400]}"
            )

            source_url = (
                url if url
                else f"enterprise://{self._name}/{self._index_name}/{doc_id}"
            )
            sources.append(
                SourceInfo(
                    url=source_url,
                    title=title,
                    snippet=content[:300],
                    content=content,
                    source_type="enterprise",
                    source_kind="vector_index",
                    relevance_score=float(score or 0.0),
                )
            )

        formatted = (
            f"Found {len(data_array)} results from {self._index_name}:\n\n"
            + "\n\n".join(lines)
        )

        return ToolResult(
            content=formatted,
            success=True,
            sources=sources,
            data={
                "index_name": self._index_name,
                "query": query,
                "count": len(data_array),
                "source_kind": "vector_index",
                "empty_result": len(data_array) == 0,
            },
        )


# ---------------------------------------------------------------------------
# Real Vector Search fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace_client() -> Any:
    """Create a Databricks WorkspaceClient for real API calls."""
    try:
        return _create_workspace_client()
    except Exception as exc:
        pytest.skip(f"WorkspaceClient unavailable: {exc}")


@pytest.fixture
def real_vector_search_tool(workspace_client: Any) -> RealVectorSearchTool:
    """RealVectorSearchTool wired to the configured VS index."""
    return RealVectorSearchTool(
        workspace_client=workspace_client,
        index_name=_VS_INDEX_NAME,
        num_results=5,
    )


@pytest.fixture
def make_real_vector_search_tool(
    workspace_client: Any,
) -> Any:
    """Factory for real vector search tools against arbitrary live indexes."""

    def _factory(
        index_name: str,
        *,
        columns: list[str] | None = None,
        num_results: int = 5,
        name: str = "vector_search",
    ) -> RealVectorSearchTool:
        return RealVectorSearchTool(
            workspace_client=workspace_client,
            index_name=index_name,
            columns=columns,
            num_results=num_results,
            name=name,
        )

    return _factory


@pytest.fixture
def real_transcript_vector_search_tool(
    make_real_vector_search_tool: Any,
) -> RealVectorSearchTool:
    """RealVectorSearchTool wired to the transcript VS index."""
    return make_real_vector_search_tool(
        _TRANSCRIPT_VS_INDEX_NAME,
        num_results=5,
    )


@pytest.fixture
def real_knowledge_base_vector_search_tool(
    make_real_vector_search_tool: Any,
) -> RealVectorSearchTool:
    """Optional real VS tool for the knowledge-base index."""
    if not _KB_VS_INDEX_NAME:
        pytest.skip("FRAMEWORK_TEST_KB_VS_INDEX not configured")
    return make_real_vector_search_tool(
        _KB_VS_INDEX_NAME,
        num_results=5,
    )


@pytest.fixture
def real_genie_tool(workspace_client: Any) -> DatabricksGenieTool:
    """Real Genie tool wired to the configured Genie space."""
    return DatabricksGenieTool(
        workspace_client=workspace_client,
        name="genie",
        space_id=_GENIE_SPACE_ID,
        description="Natural language SQL over the FSI Portfolio Assistant Genie space.",
    )


def is_transient_provider_error(exc: BaseException) -> bool:
    return isinstance(exc, (APITimeoutError, httpx.TimeoutException, httpx.ConnectTimeout))


def skip_if_transient_provider_failure(exc: BaseException) -> None:
    if is_transient_provider_error(exc):
        pytest.skip(f"Transient provider failure: {exc}")
    raise exc
