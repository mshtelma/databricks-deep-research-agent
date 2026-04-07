from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock

from databricks_deep_research.agents.source_aware import (
    admit_tool_result,
    plan_tool_arguments,
    select_step_tools,
    tool_source_kind,
)
from databricks_deep_research.tools.protocol import SourceKind, ToolDefinition, ToolResult


def _make_tool(
    name: str,
    description: str,
    *,
    source_type: str = "enterprise",
    metadata: dict[str, object] | None = None,
) -> MagicMock:
    tool = MagicMock()
    type(tool).definition = PropertyMock(
        return_value=ToolDefinition(
            name=name,
            description=description,
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            source_type=source_type,
            metadata=metadata or {},
        )
    )
    return tool


def test_select_step_tools_prefers_hinted_sources() -> None:
    step = {
        "title": "Retrieve Kroger earnings materials",
        "description": "Use earnings and transcript indexes first.",
        "source_hints": [
            {"source_name": "search_main_dbdemos_ai_agent_earnings_vs_index", "source_type": "vector_search", "priority": 1},
            {"source_name": "search_main_dbdemos_ai_agent_transcript_vs_index", "source_type": "vector_search", "priority": 2},
            {"source_name": "search_main_msh_dbdemos_ai_agent_knowledge_base_vs_index", "source_type": "vector_search", "priority": 3},
        ],
    }
    earnings = _make_tool("search_main_dbdemos_ai_agent_earnings_vs_index", "Quarterly earnings index")
    transcript = _make_tool("search_main_dbdemos_ai_agent_transcript_vs_index", "Earnings transcript index")
    kb = _make_tool("search_main_msh_dbdemos_ai_agent_knowledge_base_vs_index", "Knowledge base index")

    selection = select_step_tools([earnings, transcript, kb], step)

    assert [tool.definition.name for tool in selection.active_tools] == [
        "search_main_dbdemos_ai_agent_earnings_vs_index",
        "search_main_dbdemos_ai_agent_transcript_vs_index",
    ]
    assert [tool.definition.name for tool in selection.fallback_tools] == [
        "search_main_msh_dbdemos_ai_agent_knowledge_base_vs_index",
    ]


def test_select_step_tools_prefers_vector_search_for_infrastructure_docs() -> None:
    step = {
        "title": "Review deployment pipeline and infrastructure runbooks",
        "description": "Use internal architecture docs before general QA.",
    }
    vector = _make_tool(
        "vector_search",
        "Internal architecture, deployment, and runbook documents.",
        source_type="vector_search",
    )
    qa = _make_tool(
        "knowledge_assistant",
        "General enterprise Q&A assistant.",
        source_type="knowledge_assistant",
        metadata={"source_type": "knowledge_assistant"},
    )

    selection = select_step_tools([qa, vector], step)

    assert selection.active_tools[0].definition.name == "vector_search"


def test_select_step_tools_prefers_genie_for_growth_metrics() -> None:
    step = {
        "title": "Analyze cloud growth by product line",
        "description": "Use internal revenue and KPI analytics.",
    }
    genie = _make_tool("genie", "Enterprise metrics and KPI analytics.", source_type="genie")
    vector = _make_tool("vector_search", "Internal architecture documents.", source_type="vector_search")

    selection = select_step_tools([vector, genie], step)

    assert selection.active_tools[0].definition.name == "genie"


def test_plan_tool_arguments_builds_vector_search_alternates() -> None:
    step = {
        "title": "Search for Kroger's most recent earnings release",
        "description": "Focus on revenue, net income, EPS, and guidance.",
        "source_hints": [
            {
                "source_name": "search_main_dbdemos_ai_agent_earnings_vs_index",
                "source_type": "vector_search",
                "priority": 1,
                "query_hint": "Kroger quarterly earnings release with revenue, net income, EPS, and guidance",
            }
        ],
    }
    tool = _make_tool(
        "search_main_dbdemos_ai_agent_earnings_vs_index",
        "Search quarterly earnings release documents.",
        metadata={"index_name": "main.dbdemos_ai_agent.earnings_vs_index"},
    )

    planned = plan_tool_arguments(
        tool.definition,
        {"query": "Kroger Q3 2025 earnings report revenue net income EPS"},
        current_step=step,
        root_query="Why did Kroger miss earnings expectations?",
        background_summary="The topic centers on Kroger earnings and guidance.",
        recent_observations=[],
    )

    # Passthrough mode preserves the LLM's raw query
    assert planned.rewritten_query == "Kroger Q3 2025 earnings report revenue net income EPS"
    assert planned.strategy == "vector_passthrough"
    assert planned.arguments["_alternate_queries"] == planned.alternate_queries


def test_plan_tool_arguments_strips_question_leadins_from_subject() -> None:
    tool = _make_tool(
        "search_main_dbdemos_ai_agent_transcript_vs_index",
        "Search earnings call transcripts.",
        metadata={"index_name": "main.dbdemos_ai_agent.transcript_vs_index"},
    )

    planned = plan_tool_arguments(
        tool.definition,
        {"query": "What did Kroger management say about guidance?"},
        current_step={
            "title": "Analyze earnings call commentary",
            "description": "Use the transcript index.",
        },
        root_query="How did Kroger describe full-year guidance?",
        background_summary="Focus on earnings call commentary and management guidance.",
        recent_observations=[],
    )

    # Passthrough mode preserves the LLM's raw query (including question form)
    assert planned.rewritten_query == "What did Kroger management say about guidance?"
    assert planned.strategy == "vector_passthrough"


def test_admit_tool_result_rejects_irrelevant_vector_hits() -> None:
    step = {
        "title": "Research Kroger quarterly results",
        "description": "Look for revenue, EPS, and impairment charge details.",
    }
    definition = ToolDefinition(
        name="search_main_dbdemos_ai_agent_earnings_vs_index",
        description="Search earnings releases.",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        source_type="vector_search",
    )
    result = ToolResult(
        content="raw results",
        sources=[
            {
                "url": "https://example.com/1",
                "title": "Business Internet Setup Guide",
                "snippet": "Telecommunications installation manual for internet setup.",
                "source_type": "vector_search",
            },
            {
                "url": "https://example.com/2",
                "title": "Kroger Reports Fourth Quarter and Full-Year 2024 Results",
                "snippet": "Revenue, adjusted EPS, and impairment charge details.",
                "source_type": "vector_search",
            },
        ],
    )

    admitted = admit_tool_result(
        definition,
        result,
        current_step=step,
        root_query="Why did Kroger report an impairment charge?",
    )

    assert admitted.accepted_count == 1
    assert admitted.accepted_sources[0]["title"].startswith("Kroger Reports")
    assert admitted.rejected_count == 1
    assert admitted.rejected_sources[0]["title"] == "Business Internet Setup Guide"


def test_admit_enterprise_source_with_high_relevance_score() -> None:
    """Enterprise source with relevance_score > 0 should be admitted
    even without keyword overlap — trusts upstream semantic ranking."""
    step = {"title": "Analyze quarterly financial performance", "description": ""}
    definition = ToolDefinition(
        name="search_earnings_vs_index",
        description="Search earnings releases.",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        source_type="vector_search",
    )
    result = ToolResult(
        content="raw results",
        sources=[{
            "url": "https://example.com/1",
            "title": "Q3 FY2025 Financial Results Summary",
            "snippet": "Identical sales growth of 1.2% excluding fuel...",
            "source_type": "vector_search",
            "relevance_score": 0.85,
        }],
    )
    admitted = admit_tool_result(
        definition,
        result,
        current_step=step,
        root_query="how are the Kroger earnings?",
    )
    assert admitted.accepted_count == 1
    assert admitted.rejected_count == 0


def test_admit_enterprise_source_zero_relevance_rejected() -> None:
    """Enterprise source with relevance_score=0.0 and no keyword match
    should still be rejected (no upstream confidence)."""
    step = {"title": "Research quarterly results", "description": ""}
    definition = ToolDefinition(
        name="search_earnings_vs_index",
        description="Search earnings releases.",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        source_type="vector_search",
    )
    result = ToolResult(
        content="raw results",
        sources=[{
            "url": "https://example.com/1",
            "title": "Office Supply Catalog 2024",
            "snippet": "Paper clips and stapler accessories.",
            "source_type": "vector_search",
            "relevance_score": 0.0,
        }],
    )
    admitted = admit_tool_result(
        definition,
        result,
        current_step=step,
        root_query="how are the Kroger earnings?",
    )
    assert admitted.accepted_count == 0
    assert admitted.rejected_count == 1


def test_admit_genie_source_by_modality_when_successful() -> None:
    step = {"title": "Analyze FSI portfolio composition", "description": ""}
    definition = ToolDefinition(
        name="genie",
        description="Query enterprise analytics.",
        parameters={"type": "object", "properties": {"question": {"type": "string"}}},
        source_type="enterprise",
        source_kind=SourceKind.sql_analytics,
    )
    result = ToolResult(
        content="Portfolio companies grouped by industry and market cap.",
        sources=[{
            "url": "enterprise://genie/fsi",
            "title": "FSI Portfolio Assistant",
            "snippet": "Industry and market capitalization breakdown for portfolio companies.",
            "source_type": "enterprise",
            "source_kind": SourceKind.sql_analytics,
        }],
    )

    admitted = admit_tool_result(
        definition,
        result,
        current_step=step,
        root_query="Which companies are in the FSI portfolio by industry and market cap?",
    )

    assert admitted.accepted_count == 1
    assert admitted.rejected_count == 0


def test_admit_failed_enterprise_result_does_not_create_synthetic_source() -> None:
    definition = ToolDefinition(
        name="genie",
        description="Query enterprise analytics.",
        parameters={"type": "object", "properties": {"question": {"type": "string"}}},
        source_type="enterprise",
        source_kind=SourceKind.sql_analytics,
    )
    result = ToolResult(
        content="Failed: query execution timed out.",
        success=False,
        sources=[],
    )

    admitted = admit_tool_result(
        definition,
        result,
        current_step={"title": "Analyze portfolio composition", "description": ""},
        root_query="Which companies are in the FSI portfolio?",
    )

    assert admitted.accepted_count == 0
    assert admitted.rejected_count == 0
    assert admitted.raw_sources == []


def test_web_source_no_enterprise_boost() -> None:
    """Web search results should NOT get enterprise boost, even with
    relevance_score — web results use artificial scoring, not semantic."""
    step = {"title": "Research quarterly results", "description": ""}
    definition = ToolDefinition(
        name="web_search",
        description="Search the web.",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        source_type="web",
    )
    result = ToolResult(
        content="raw results",
        sources=[{
            "url": "https://example.com/random",
            "title": "Unrelated Web Page About Gardening",
            "snippet": "How to plant tomatoes in spring...",
            "source_type": "web",
            "relevance_score": 0.9,
        }],
    )
    admitted = admit_tool_result(
        definition,
        result,
        current_step=step,
        root_query="how are the Kroger earnings?",
    )
    assert admitted.accepted_count == 0
    assert admitted.rejected_count == 1


def test_enterprise_source_moderate_relevance_plus_keyword() -> None:
    """Enterprise source with relevance_score > 0 AND keyword match
    should be comfortably admitted."""
    step = {"title": "Research Kroger quarterly results", "description": ""}
    definition = ToolDefinition(
        name="search_earnings_vs_index",
        description="Search earnings releases.",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        source_type="vector_search",
    )
    result = ToolResult(
        content="raw results",
        sources=[{
            "url": "https://example.com/1",
            "title": "Kroger Reports Third Quarter 2025 Results",
            "snippet": "Revenue grew 2% to $35.1 billion...",
            "source_type": "vector_search",
            "relevance_score": 0.72,
        }],
    )
    admitted = admit_tool_result(
        definition,
        result,
        current_step=step,
        root_query="how are the Kroger earnings?",
    )
    assert admitted.accepted_count == 1
    # Score should be high: keyword matches + enterprise boost
    assert admitted.accepted_sources[0]["admission_score"] >= 4


# ---------------------------------------------------------------------------
# Step 5b: SourceKind preference tests
# ---------------------------------------------------------------------------


def test_tool_source_kind_prefers_explicit_source_kind() -> None:
    """When ToolDefinition has source_kind set, use it over heuristic."""
    definition = ToolDefinition(
        name="custom_tool",
        description="A custom enterprise tool",
        parameters={"type": "object", "properties": {}},
        source_type="enterprise",
        source_kind="vector_index",
    )
    assert tool_source_kind(definition) == "vector_index"


def test_tool_source_kind_falls_back_to_heuristic() -> None:
    """When source_kind is 'builtin' (default), use heuristic."""
    definition = ToolDefinition(
        name="search_earnings_vs_index",
        description="Search earnings releases",
        parameters={"type": "object", "properties": {}},
        source_type="vector_search",
        # source_kind defaults to "builtin"
    )
    assert tool_source_kind(definition) == "vector_search"


def test_tool_source_kind_sql_analytics() -> None:
    """Explicit sql_analytics source_kind is respected."""
    definition = ToolDefinition(
        name="query_genie_financials",
        description="Query financial data via Genie",
        parameters={"type": "object", "properties": {}},
        source_type="genie",
        source_kind=SourceKind.sql_analytics,
    )
    assert tool_source_kind(definition) == "sql_analytics"


# ---------------------------------------------------------------------------
# Step 5c: ToolDefinition.source_kind field tests
# ---------------------------------------------------------------------------


def test_tool_definition_default_source_kind() -> None:
    td = ToolDefinition(name="x", description="y", parameters={})
    assert td.source_kind == "builtin"


def test_tool_definition_explicit_source_kind() -> None:
    td = ToolDefinition(name="x", description="y", parameters={}, source_kind="vector_index")
    assert td.source_kind == "vector_index"


def test_source_kind_enum_values() -> None:
    """SourceKind enum has all expected values."""
    assert SourceKind.web == "web"
    assert SourceKind.vector_index == "vector_index"
    assert SourceKind.sql_analytics == "sql_analytics"
    assert SourceKind.qa_assistant == "qa_assistant"
    assert SourceKind.file == "file"
    assert SourceKind.builtin == "builtin"


def test_source_kind_is_str_compatible() -> None:
    """SourceKind values work as plain strings (e.g., in frozenset lookups)."""
    kinds = frozenset({"vector_index", "sql_analytics"})
    assert SourceKind.vector_index in kinds
    assert SourceKind.web not in kinds


# ---------------------------------------------------------------------------
# VS admission threshold + tiered enterprise boost tests
# ---------------------------------------------------------------------------


def test_vs_source_low_relevance_rejected() -> None:
    """VS source with relevance_score=0.15 and no keyword overlap is rejected."""
    step = {"title": "Analyze cloud architecture", "description": "Review deployment docs."}
    definition = ToolDefinition(
        name="search_docs_vs_index",
        description="Search internal documents.",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        source_type="vector_search",
    )
    result = ToolResult(
        content="raw results",
        sources=[{
            "url": "https://example.com/low",
            "title": "Office Cafeteria Menu Spring 2025",
            "snippet": "Weekly lunch specials and catering options.",
            "source_type": "vector_search",
            "relevance_score": 0.15,
        }],
    )
    admitted = admit_tool_result(
        definition,
        result,
        current_step=step,
        root_query="How is the cloud deployment pipeline configured?",
    )
    assert admitted.accepted_count == 0
    assert admitted.rejected_count == 1


def test_vs_source_moderate_relevance_with_keyword_accepted() -> None:
    """VS source with relevance_score=0.35 + 1 keyword match is accepted.

    Tiered boost: 0.35 >= 0.3 gives enterprise_boost=1.
    1 keyword match gives +1.  Total score = 2 which meets threshold.
    """
    step = {"title": "Review deployment pipeline", "description": "Check infrastructure docs."}
    definition = ToolDefinition(
        name="search_docs_vs_index",
        description="Search internal documents.",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        source_type="vector_search",
    )
    result = ToolResult(
        content="raw results",
        sources=[{
            "url": "https://example.com/moderate",
            "title": "Infrastructure Deployment Guide",
            "snippet": "Pipeline configuration for deployment automation.",
            "source_type": "vector_search",
            "relevance_score": 0.35,
        }],
    )
    admitted = admit_tool_result(
        definition,
        result,
        current_step=step,
        root_query="How is the deployment pipeline configured?",
    )
    assert admitted.accepted_count == 1
    assert admitted.rejected_count == 0


def test_vs_source_moderate_relevance_no_keyword_fallback() -> None:
    """VS source with relevance_score=0.35 and zero keyword overlap is accepted
    via the fallback threshold (0.35 >= 0.3)."""
    step = {"title": "Analyze quarterly results", "description": "Review earnings data."}
    definition = ToolDefinition(
        name="search_docs_vs_index",
        description="Search internal documents.",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        source_type="vector_search",
    )
    result = ToolResult(
        content="raw results",
        sources=[{
            "url": "https://example.com/moderate-nokey",
            "title": "Unrelated Topic With No Keyword Overlap",
            "snippet": "Content that shares no terms with the query profile.",
            "source_type": "vector_search",
            "relevance_score": 0.35,
        }],
    )
    admitted = admit_tool_result(
        definition,
        result,
        current_step=step,
        root_query="How is the deployment pipeline configured?",
    )
    assert admitted.accepted_count == 1
    assert admitted.rejected_count == 0


def test_vs_source_strong_relevance_always_accepted() -> None:
    """VS source with relevance_score=0.7 is accepted regardless of keywords.

    Tiered boost: 0.7 >= 0.5 gives enterprise_boost=2.
    Score = 0 (no keywords) + 2 (boost) = 2, which meets threshold.
    """
    step = {"title": "Analyze quarterly results", "description": "Check financial data."}
    definition = ToolDefinition(
        name="search_docs_vs_index",
        description="Search internal documents.",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        source_type="vector_search",
    )
    result = ToolResult(
        content="raw results",
        sources=[{
            "url": "https://example.com/strong",
            "title": "Completely Unrelated Title No Keywords",
            "snippet": "Zero overlap with any profile terms whatsoever.",
            "source_type": "vector_search",
            "relevance_score": 0.7,
        }],
    )
    admitted = admit_tool_result(
        definition,
        result,
        current_step=step,
        root_query="How is the deployment pipeline configured?",
    )
    assert admitted.accepted_count == 1
    assert admitted.rejected_count == 0
    # Score should be exactly 2 from enterprise boost alone
    assert admitted.accepted_sources[0]["admission_score"] == 2


def test_empty_profile_enterprise_falls_through_to_scoring() -> None:
    """Empty profile + VS source with relevance_score=0.1 is rejected,
    not blindly accepted as before the fix."""
    # Use a step/query that produces an empty profile (no extractable terms)
    step = {"title": "", "description": ""}
    definition = ToolDefinition(
        name="search_docs_vs_index",
        description="Search internal documents.",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        source_type="vector_search",
    )
    result = ToolResult(
        content="raw results",
        sources=[{
            "url": "https://example.com/low-enterprise",
            "title": "Random Noise Document",
            "snippet": "Completely irrelevant content.",
            "source_type": "vector_search",
            "relevance_score": 0.1,
        }],
    )
    admitted = admit_tool_result(
        definition,
        result,
        current_step=step,
        root_query="",
    )
    assert admitted.accepted_count == 0
    assert admitted.rejected_count == 1


def test_empty_profile_web_source_still_accepted() -> None:
    """Empty profile + web source is still accepted (backward compat)."""
    step = {"title": "", "description": ""}
    definition = ToolDefinition(
        name="web_search",
        description="Search the web.",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        source_type="web",
    )
    result = ToolResult(
        content="raw results",
        sources=[{
            "url": "https://example.com/web-page",
            "title": "Some Web Page",
            "snippet": "Some web content.",
            "source_type": "web",
        }],
    )
    admitted = admit_tool_result(
        definition,
        result,
        current_step=step,
        root_query="",
    )
    assert admitted.accepted_count == 1
    assert admitted.rejected_count == 0
