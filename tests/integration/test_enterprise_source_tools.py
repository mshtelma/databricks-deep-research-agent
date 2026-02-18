"""Integration tests for enterprise source tools (Knowledge Assistant + Genie + Vector Search).

Tests the end-to-end path: source ID → factory → tool construction → real API execution.

Requirements:
- .env file with DATABRICKS_TOKEN or DATABRICKS_CONFIG_PROFILE
- Access to KA endpoint 'ka-99a12b9d-endpoint'
- Access to Genie space '01f0b5ab5b841281858ae25da3f58125' (FSI Portfolio Explorer)
- Access to VS index 'anthony_ivan.demo-toolsapp.pdf_chunks_index'

Run with:
    uv run pytest tests/integration/test_enterprise_source_tools.py -v -s
"""

import asyncio
from uuid import uuid4

import pytest
from tests.integration.conftest import requires_databricks

from deep_research.agent.tools.base import ResearchContext, ToolResult
from deep_research.agent.tools.factory import create_tools_from_source_ids
from deep_research.agent.tools.genie import GenieTool
from deep_research.agent.tools.knowledge_assistant import KnowledgeAssistantTool
from deep_research.agent.tools.user_vector_search import UserVectorSearchTool

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KA_ENDPOINT_NAME = "ka-99a12b9d-endpoint"
KA_SOURCE_ID = f"assistant:{KA_ENDPOINT_NAME}"

GENIE_SPACE_ID = "01f0b5ab5b841281858ae25da3f58125"
GENIE_SOURCE_ID = f"genie:{GENIE_SPACE_ID}"

VS_INDEX_NAME = "anthony_ivan.demo-toolsapp.pdf_chunks_index"
VS_SOURCE_ID = f"vs:{VS_INDEX_NAME}"
VS_TIMEOUT = 60  # VS queries are typically fast

# Timeouts (seconds) — KA is fast, Genie needs SQL generation + polling
KA_TIMEOUT = 60
GENIE_TIMEOUT = 120
ALL_TIMEOUT = 240


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context() -> ResearchContext:
    """Create a minimal ResearchContext for service-principal auth."""
    return ResearchContext(
        chat_id=uuid4(),
        user_id="integration-test-user",
        research_type="light",
        user_token=None,  # Service principal auth
    )


async def _execute_ka(tool: object, question: str) -> ToolResult:
    """Execute KA tool with timeout."""
    ctx = _make_context()
    return await asyncio.wait_for(
        tool.execute({"question": question}, ctx),  # type: ignore[union-attr]
        timeout=KA_TIMEOUT,
    )


async def _execute_genie(tool: object, question: str) -> ToolResult:
    """Execute Genie tool with timeout."""
    ctx = _make_context()
    return await asyncio.wait_for(
        tool.execute({"question": question}, ctx),  # type: ignore[union-attr]
        timeout=GENIE_TIMEOUT,
    )


async def _execute_vs(tool: object, question: str) -> ToolResult:
    """Execute Vector Search tool with timeout."""
    ctx = _make_context()
    return await asyncio.wait_for(
        tool.execute({"query": question}, ctx),  # type: ignore[union-attr]
        timeout=VS_TIMEOUT,
    )


# ---------------------------------------------------------------------------
# Class 1: Factory source-ID creation
# ---------------------------------------------------------------------------


@requires_databricks
class TestFactorySourceIdCreation:
    """Verify create_tools_from_source_ids() produces correctly typed tools."""

    def test_factory_creates_ka_tool_from_source_id(self) -> None:
        tools = create_tools_from_source_ids([KA_SOURCE_ID])

        assert len(tools) == 1
        tool = tools[0]
        assert isinstance(tool, KnowledgeAssistantTool)
        assert tool.definition.source_type == "knowledge_assistant"
        assert tool.definition.name == "ask_ka_99a12b9d_endpoint"

    def test_factory_creates_genie_tool_from_source_id(self) -> None:
        tools = create_tools_from_source_ids([GENIE_SOURCE_ID])

        assert len(tools) == 1
        tool = tools[0]
        assert isinstance(tool, GenieTool)
        assert tool.definition.source_type == "genie"

    def test_factory_creates_vs_tool_from_source_id(self) -> None:
        tools = create_tools_from_source_ids([VS_SOURCE_ID])

        assert len(tools) == 1
        tool = tools[0]
        assert isinstance(tool, UserVectorSearchTool)
        assert tool.definition.source_type == "vector_search"

    def test_factory_creates_all_tools_together(self) -> None:
        tools = create_tools_from_source_ids([KA_SOURCE_ID, GENIE_SOURCE_ID, VS_SOURCE_ID])

        assert len(tools) == 3
        source_types = {t.definition.source_type for t in tools}
        assert source_types == {"knowledge_assistant", "genie", "vector_search"}

    def test_tool_definitions_have_required_parameters(self) -> None:
        tools = create_tools_from_source_ids([KA_SOURCE_ID, GENIE_SOURCE_ID, VS_SOURCE_ID])

        for tool in tools:
            defn = tool.definition
            required = defn.parameters.get("required", [])
            # VS uses "query"; KA and Genie use "question"
            assert "question" in required or "query" in required, (
                f"Tool '{defn.name}' missing 'question' or 'query' in required parameters"
            )


# ---------------------------------------------------------------------------
# Class 2: Knowledge Assistant execution
# ---------------------------------------------------------------------------


@requires_databricks
class TestKnowledgeAssistantExecution:
    """Test real KA endpoint queries with service-principal auth."""

    @pytest.mark.asyncio
    async def test_ka_execute_returns_answer_about_nvda(self) -> None:
        tools = create_tools_from_source_ids([KA_SOURCE_ID])
        tool = tools[0]

        result = await _execute_ka(tool, "What is the general sentiment for NVDA stock?")

        print(f"\nKA Response ({len(result.content)} chars):")
        print(result.content[:500])
        print(f"\nSuccess: {result.success}")
        print(f"Data: {result.data}")

        assert result.success is True
        assert result.content
        assert len(result.content) > 50

        if result.data:
            # has_answer may be True or False depending on endpoint state
            assert "has_answer" in result.data

    @pytest.mark.asyncio
    async def test_ka_sources_contain_citations(self) -> None:
        tools = create_tools_from_source_ids([KA_SOURCE_ID])
        tool = tools[0]

        result = await _execute_ka(tool, "What is the general sentiment for NVDA stock?")

        print(f"\nSources: {result.sources}")

        # KA may or may not return citations — but if it does, validate shape
        if result.sources:
            for source in result.sources:
                assert source.get("type") == "knowledge_assistant"
                assert "endpoint_name" in source

    @pytest.mark.asyncio
    async def test_ka_result_data_has_expected_fields(self) -> None:
        tools = create_tools_from_source_ids([KA_SOURCE_ID])
        tool = tools[0]

        result = await _execute_ka(tool, "What is the general sentiment for NVDA stock?")

        print(f"\nData keys: {list(result.data.keys()) if result.data else 'None'}")

        assert result.data is not None
        assert "question" in result.data
        assert "has_answer" in result.data
        assert "endpoint_name" in result.data


# ---------------------------------------------------------------------------
# Class 3: Genie execution
# ---------------------------------------------------------------------------


@requires_databricks
class TestGenieExecution:
    """Test real Genie space queries with service-principal auth."""

    @pytest.mark.asyncio
    async def test_genie_execute_returns_data_about_nvda(self) -> None:
        tools = create_tools_from_source_ids([GENIE_SOURCE_ID])
        tool = tools[0]

        result = await _execute_genie(tool, "What are the latest news about NVDA stock?")

        print(f"\nGenie Response ({len(result.content)} chars):")
        print(result.content[:500])
        print(f"\nSuccess: {result.success}")
        print(f"Data: {result.data}")

        assert result.success is True
        assert result.content
        assert len(result.content) > 20

    @pytest.mark.asyncio
    async def test_genie_sources_have_genie_metadata(self) -> None:
        tools = create_tools_from_source_ids([GENIE_SOURCE_ID])
        tool = tools[0]

        result = await _execute_genie(tool, "What are the latest news about NVDA stock?")

        print(f"\nGenie sources: {result.sources}")

        assert result.sources is not None
        assert len(result.sources) >= 1

        source = result.sources[0]
        assert source.get("type") == "genie"
        assert source.get("space_id") == GENIE_SPACE_ID
        assert str(source.get("url", "")).startswith("genie://")

    @pytest.mark.asyncio
    async def test_genie_result_data_has_expected_fields(self) -> None:
        tools = create_tools_from_source_ids([GENIE_SOURCE_ID])
        tool = tools[0]

        result = await _execute_genie(tool, "What are the latest news about NVDA stock?")

        print(f"\nGenie data keys: {list(result.data.keys()) if result.data else 'None'}")

        assert result.data is not None
        assert "question" in result.data
        assert "space_id" in result.data
        # At least one of sql/row_count should be present
        assert "sql" in result.data or "row_count" in result.data, (
            f"Expected 'sql' or 'row_count' in data, got keys: {list(result.data.keys())}"
        )


# ---------------------------------------------------------------------------
# Class 4: Vector Search execution
# ---------------------------------------------------------------------------


@requires_databricks
class TestVectorSearchExecution:
    """Test real Vector Search queries with service-principal auth."""

    @pytest.mark.asyncio
    async def test_vs_execute_returns_results_about_data_protection(self) -> None:
        tools = create_tools_from_source_ids([VS_SOURCE_ID])
        tool = tools[0]

        result = await _execute_vs(tool, "data protection regulations in Singapore")

        print(f"\nVS Response ({len(result.content)} chars):")
        print(result.content[:500])
        print(f"\nSuccess: {result.success}")
        print(f"Data: {result.data}")

        assert result.success is True
        assert result.content
        assert len(result.content) > 50

    @pytest.mark.asyncio
    async def test_vs_sources_contain_vector_search_metadata(self) -> None:
        tools = create_tools_from_source_ids([VS_SOURCE_ID])
        tool = tools[0]

        result = await _execute_vs(tool, "data protection regulations in Singapore")

        print(f"\nVS sources: {result.sources}")

        assert result.sources is not None
        assert len(result.sources) >= 1

        source = result.sources[0]
        assert source.get("type") == "vector_search"
        assert source.get("index_name") == VS_INDEX_NAME

    @pytest.mark.asyncio
    async def test_vs_result_data_has_expected_fields(self) -> None:
        tools = create_tools_from_source_ids([VS_SOURCE_ID])
        tool = tools[0]

        result = await _execute_vs(tool, "data protection regulations in Singapore")

        print(f"\nVS data keys: {list(result.data.keys()) if result.data else 'None'}")

        assert result.data is not None
        assert "query" in result.data
        assert "num_results" in result.data
        assert result.data["num_results"] > 0


# ---------------------------------------------------------------------------
# Class 5: End-to-end source-ID pipeline
# ---------------------------------------------------------------------------


@requires_databricks
class TestEndToEndSourceIdPipeline:
    """Full flow: source ID → factory → tool → execute → verify."""

    @pytest.mark.asyncio
    async def test_ka_end_to_end_from_source_id(self) -> None:
        """Factory → KA tool → execute → verify result."""
        tools = create_tools_from_source_ids([KA_SOURCE_ID])
        assert len(tools) == 1
        tool = tools[0]
        assert tool.definition.source_type == "knowledge_assistant"

        result = await _execute_ka(tool, "What is the general sentiment for NVDA stock?")

        print(f"\n[E2E KA] Success={result.success}, Content length={len(result.content)}")
        print(f"[E2E KA] Content preview: {result.content[:300]}")

        assert result.success is True
        assert len(result.content) > 50

        if result.data:
            assert result.data.get("endpoint_name") == KA_ENDPOINT_NAME

    @pytest.mark.asyncio
    async def test_genie_end_to_end_from_source_id(self) -> None:
        """Factory → Genie tool → execute → verify result."""
        tools = create_tools_from_source_ids([GENIE_SOURCE_ID])
        assert len(tools) == 1
        tool = tools[0]
        assert tool.definition.source_type == "genie"

        result = await _execute_genie(tool, "What are the latest news about NVDA stock?")

        print(f"\n[E2E Genie] Success={result.success}, Content length={len(result.content)}")
        print(f"[E2E Genie] Content preview: {result.content[:300]}")

        assert result.success is True
        assert len(result.content) > 20

        if result.data:
            assert result.data.get("space_id") == GENIE_SPACE_ID

    @pytest.mark.asyncio
    async def test_vs_end_to_end_from_source_id(self) -> None:
        """Factory → VS tool → execute → verify result."""
        tools = create_tools_from_source_ids([VS_SOURCE_ID])
        assert len(tools) == 1
        tool = tools[0]
        assert tool.definition.source_type == "vector_search"

        result = await _execute_vs(tool, "What are Singapore's data protection laws?")

        print(f"\n[E2E VS] Success={result.success}, Content length={len(result.content)}")
        print(f"[E2E VS] Content preview: {result.content[:300]}")

        assert result.success is True
        assert len(result.content) > 50

        if result.data:
            assert result.data.get("index_name") == VS_INDEX_NAME

    @pytest.mark.asyncio
    async def test_all_tools_execute_independently(self) -> None:
        """All tools created from source IDs execute and succeed independently."""
        tools = create_tools_from_source_ids([KA_SOURCE_ID, GENIE_SOURCE_ID, VS_SOURCE_ID])
        assert len(tools) == 3

        ka_tool = next(t for t in tools if t.definition.source_type == "knowledge_assistant")
        genie_tool = next(t for t in tools if t.definition.source_type == "genie")
        vs_tool = next(t for t in tools if t.definition.source_type == "vector_search")

        ka_result, genie_result, vs_result = await asyncio.wait_for(
            _run_all(ka_tool, genie_tool, vs_tool),
            timeout=ALL_TIMEOUT,
        )

        print(f"\n[All] KA: success={ka_result.success}, len={len(ka_result.content)}")
        print(f"[All] Genie: success={genie_result.success}, len={len(genie_result.content)}")
        print(f"[All] VS: success={vs_result.success}, len={len(vs_result.content)}")

        assert ka_result.success is True
        assert genie_result.success is True
        assert vs_result.success is True
        assert len(ka_result.content) > 50
        assert len(genie_result.content) > 20
        assert len(vs_result.content) > 50


async def _run_all(
    ka_tool: object, genie_tool: object, vs_tool: object,
) -> tuple[ToolResult, ToolResult, ToolResult]:
    """Execute KA, Genie, and VS tools sequentially."""
    ctx = _make_context()
    ka_result = await ka_tool.execute(  # type: ignore[union-attr]
        {"question": "What is the general sentiment for NVDA stock?"},
        ctx,
    )
    genie_result = await genie_tool.execute(  # type: ignore[union-attr]
        {"question": "What are the latest news about NVDA stock?"},
        ctx,
    )
    vs_result = await vs_tool.execute(  # type: ignore[union-attr]
        {"query": "data protection regulations in Singapore"},
        ctx,
    )
    return ka_result, genie_result, vs_result
