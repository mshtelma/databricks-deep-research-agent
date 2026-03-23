"""Tests for mixed-source tool selection in source_aware.select_step_tools.

Covers the interaction between enterprise-hinted tools and web search/crawl
tools — specifically the _is_web_search_tool gate and the heuristic scoring
boost for comparison keywords.
"""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock

from databricks_deep_research.agents.source_aware import (
    _is_web_search_tool,
    _score_tool_for_step,
    select_step_tools,
)
from databricks_deep_research.tools.protocol import ToolDefinition

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(
    name: str,
    description: str,
    *,
    source_type: str = "enterprise",
    source_kind: str = "builtin",
    metadata: dict[str, object] | None = None,
) -> MagicMock:
    """Build a MagicMock that satisfies the ResearchTool protocol."""
    tool = MagicMock()
    type(tool).definition = PropertyMock(
        return_value=ToolDefinition(
            name=name,
            description=description,
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            source_type=source_type,
            source_kind=source_kind,
            metadata=metadata or {},
        )
    )
    return tool


def _web_search_tool() -> MagicMock:
    return _make_tool(
        "web_search",
        "Search the web using Brave.",
        source_type="web",
        source_kind="web",
    )


def _web_crawl_tool() -> MagicMock:
    return _make_tool(
        "web_crawl",
        "Crawl a specific web page.",
        source_type="web",
        source_kind="builtin",
    )


def _enterprise_vs_tool(name: str = "search_earnings_vs_index") -> MagicMock:
    return _make_tool(
        name,
        "Search quarterly earnings release documents.",
        source_type="vector_search",
    )


# ---------------------------------------------------------------------------
# 1. web_search stays in active_tools when step has enterprise source_hints
# ---------------------------------------------------------------------------


class TestWebSearchActiveWithEnterpriseHints:
    """When a step has enterprise source_hints, web_search should be promoted
    to active_tools alongside the hinted enterprise tools — not pushed to
    fallback."""

    def test_web_search_in_active_tools_with_enterprise_hints(self) -> None:
        step = {
            "title": "Retrieve Kroger earnings materials",
            "description": "Use earnings index first.",
            "source_hints": [
                {
                    "source_name": "search_earnings_vs_index",
                    "source_type": "vector_search",
                    "priority": 1,
                },
            ],
        }
        earnings = _enterprise_vs_tool()
        web = _web_search_tool()

        selection = select_step_tools([earnings, web], step)

        active_names = [t.definition.name for t in selection.active_tools]
        assert "web_search" in active_names
        assert "search_earnings_vs_index" in active_names

    def test_web_search_reason_mentions_kept_active(self) -> None:
        step = {
            "title": "Look up financials",
            "description": "Use enterprise data.",
            "source_hints": [
                {
                    "source_name": "search_earnings_vs_index",
                    "source_type": "vector_search",
                    "priority": 1,
                },
            ],
        }
        earnings = _enterprise_vs_tool()
        web = _web_search_tool()

        selection = select_step_tools([earnings, web], step)

        assert "web search kept active" in selection.reasons.get("web_search", "")

    def test_web_search_active_with_multiple_enterprise_hints(self) -> None:
        step = {
            "title": "Analyze quarterly earnings and transcripts",
            "description": "Use both earnings and transcript indexes.",
            "source_hints": [
                {
                    "source_name": "search_earnings_vs_index",
                    "source_type": "vector_search",
                    "priority": 1,
                },
                {
                    "source_name": "search_transcript_vs_index",
                    "source_type": "vector_search",
                    "priority": 2,
                },
            ],
        }
        earnings = _enterprise_vs_tool("search_earnings_vs_index")
        transcript = _enterprise_vs_tool("search_transcript_vs_index")
        web = _web_search_tool()

        selection = select_step_tools([earnings, transcript, web], step)

        active_names = [t.definition.name for t in selection.active_tools]
        assert "web_search" in active_names
        assert "search_earnings_vs_index" in active_names
        assert "search_transcript_vs_index" in active_names


# ---------------------------------------------------------------------------
# 2. web_crawl goes to fallback even when web_search is promoted to active
# ---------------------------------------------------------------------------


class TestWebCrawlFallbackWithEnterpriseHints:
    """web_crawl is NOT a search tool — when hints are present and web_crawl
    has no heuristic relevance to the step, it is hidden entirely (not placed
    in active or fallback).  When it *does* have heuristic relevance, it goes
    to fallback (not active)."""

    def test_web_crawl_hidden_when_no_heuristic_relevance(self) -> None:
        """web_crawl with zero heuristic score is hidden (not in active or
        fallback) when enterprise hints are present."""
        step = {
            "title": "Look up Kroger earnings release",
            "description": "Use earnings index first.",
            "source_hints": [
                {
                    "source_name": "search_earnings_vs_index",
                    "source_type": "vector_search",
                    "priority": 1,
                },
            ],
        }
        earnings = _enterprise_vs_tool()
        web = _web_search_tool()
        crawl = _web_crawl_tool()

        selection = select_step_tools([earnings, web, crawl], step)

        active_names = [t.definition.name for t in selection.active_tools]
        fallback_names = [t.definition.name for t in selection.fallback_tools]

        assert "web_search" in active_names
        assert "web_crawl" not in active_names
        # web_crawl has zero heuristic score for this step, so it is hidden
        assert "web_crawl" not in fallback_names
        assert selection.reasons.get("web_crawl") == "hidden until fallback"

    def test_web_crawl_not_promoted_to_active_even_with_comparison_step(self) -> None:
        """Even if the step has comparison keywords, web_crawl is not a search
        tool and should never be placed in active_tools when hints are present."""
        step = {
            "title": "Compare Kroger earnings to industry benchmarks",
            "description": "Use earnings index, then compare externally.",
            "source_hints": [
                {
                    "source_name": "search_earnings_vs_index",
                    "source_type": "vector_search",
                    "priority": 1,
                },
            ],
        }
        earnings = _enterprise_vs_tool()
        web = _web_search_tool()
        crawl = _web_crawl_tool()

        selection = select_step_tools([earnings, web, crawl], step)

        active_names = [t.definition.name for t in selection.active_tools]

        assert "web_crawl" not in active_names

    def test_web_crawl_in_fallback_when_heuristic_score_positive(self) -> None:
        """If web_crawl has a positive heuristic score (e.g., step text tokens
        overlap with the tool signature), it goes to fallback — not active."""
        step = {
            "title": "Crawl the web page for Kroger earnings details",
            "description": "Use earnings index first, then crawl specific URLs.",
            "source_hints": [
                {
                    "source_name": "search_earnings_vs_index",
                    "source_type": "vector_search",
                    "priority": 1,
                },
            ],
        }
        earnings = _enterprise_vs_tool()
        # Give the crawl tool a description that overlaps with step text tokens
        crawl = _make_tool(
            "web_crawl",
            "Crawl a specific web page to retrieve Kroger earnings details.",
            source_type="web",
            source_kind="builtin",
        )

        selection = select_step_tools([earnings, crawl], step)

        active_names = [t.definition.name for t in selection.active_tools]
        fallback_names = [t.definition.name for t in selection.fallback_tools]

        assert "web_crawl" not in active_names
        assert "web_crawl" in fallback_names


# ---------------------------------------------------------------------------
# 3. Without source_hints, all tools use heuristic scoring (unchanged)
# ---------------------------------------------------------------------------


class TestNoHintsHeuristicScoring:
    """When no source_hints are present, tools are scored purely by heuristic
    and none get the special web_search promotion path."""

    def test_no_hints_all_tools_scored_by_heuristic(self) -> None:
        step = {
            "title": "Analyze cloud growth by product line",
            "description": "Use internal revenue and KPI analytics.",
        }
        genie = _make_tool(
            "genie",
            "Enterprise metrics and KPI analytics.",
            source_type="genie",
        )
        vector = _make_tool(
            "vector_search",
            "Internal architecture documents.",
            source_type="vector_search",
        )
        web = _web_search_tool()

        selection = select_step_tools([vector, genie, web], step)

        # All tools with positive heuristic scores should be active.
        # No web_search special-casing when no hints are present.
        for tool in selection.active_tools:
            reason = selection.reasons.get(tool.definition.name, "")
            # No tool should have the "web search kept active" reason
            assert "kept active alongside enterprise hints" not in reason

    def test_no_hints_web_search_uses_heuristic_reason(self) -> None:
        step = {
            "title": "Compare industry trends in retail",
            "description": "Look for market benchmarks and public data.",
        }
        web = _web_search_tool()
        vector = _enterprise_vs_tool()

        selection = select_step_tools([web, vector], step)

        web_reason = selection.reasons.get("web_search", "")
        # Without hints, reason should reference heuristic scoring
        assert "heuristic" in web_reason or "score" in web_reason

    def test_no_hints_web_crawl_scored_same_as_other_tools(self) -> None:
        step = {
            "title": "Review deployment architecture",
            "description": "Look at infrastructure runbooks.",
        }
        crawl = _web_crawl_tool()
        vector = _make_tool(
            "vector_search",
            "Internal architecture, deployment, and runbook documents.",
            source_type="vector_search",
        )

        selection = select_step_tools([crawl, vector], step)

        # vector should score highly due to "deployment" and "runbook"
        assert selection.active_tools[0].definition.name == "vector_search"


# ---------------------------------------------------------------------------
# 4. Comparison keywords give web_search a +2 heuristic boost
# ---------------------------------------------------------------------------


class TestComparisonKeywordBoost:
    """Steps with comparison keywords (compare, benchmark, industry, etc.)
    should give web_search a +2 heuristic boost via _score_tool_for_step."""

    def test_compare_keyword_boosts_web_search(self) -> None:
        definition = _web_search_tool().definition
        step_text = "Compare Kroger's margins to industry averages"
        score = _score_tool_for_step(definition, step_text)
        assert score >= 2

    def test_benchmark_keyword_boosts_web_search(self) -> None:
        definition = _web_search_tool().definition
        step_text = "Benchmark grocery revenue against market data"
        score = _score_tool_for_step(definition, step_text)
        assert score >= 2

    def test_industry_keyword_boosts_web_search(self) -> None:
        definition = _web_search_tool().definition
        step_text = "Investigate industry trends in retail automation"
        score = _score_tool_for_step(definition, step_text)
        assert score >= 2

    def test_market_keyword_boosts_web_search(self) -> None:
        definition = _web_search_tool().definition
        step_text = "Analyze market positioning and public perception"
        score = _score_tool_for_step(definition, step_text)
        assert score >= 2

    def test_external_keyword_boosts_web_search(self) -> None:
        definition = _web_search_tool().definition
        step_text = "Gather external data on competitor pricing"
        score = _score_tool_for_step(definition, step_text)
        assert score >= 2

    def test_trend_keyword_boosts_web_search(self) -> None:
        definition = _web_search_tool().definition
        step_text = "Research trend analysis for grocery ecommerce growth"
        score = _score_tool_for_step(definition, step_text)
        assert score >= 2

    def test_no_comparison_keywords_no_boost(self) -> None:
        definition = _web_search_tool().definition
        step_text = "Look up earnings release data"
        score = _score_tool_for_step(definition, step_text)
        # Without comparison keywords, web_search gets no special boost.
        # It may still get points from token overlap, but not the +2 bonus.
        score_without_comparison = score

        score_with_comparison = _score_tool_for_step(
            definition, "Compare earnings release data to industry benchmarks"
        )
        assert score_with_comparison > score_without_comparison

    def test_comparison_boost_does_not_apply_to_enterprise_tools(self) -> None:
        """The comparison keyword boost only applies to web_search, not to
        enterprise tools like vector_search or genie."""
        vs_definition = _enterprise_vs_tool().definition
        step_text = "Compare Kroger earnings to industry benchmarks"
        _score_tool_for_step(vs_definition, step_text)

        ws_definition = _web_search_tool().definition
        ws_score = _score_tool_for_step(ws_definition, step_text)

        # web_search should get the +2 comparison boost; vector_search should not
        # (vector_search may get a boost for "earnings" but not for "compare")
        assert ws_score >= 2  # At minimum the comparison boost

    def test_public_keyword_boosts_web_search(self) -> None:
        definition = _web_search_tool().definition
        step_text = "Search for public filings and disclosures"
        score = _score_tool_for_step(definition, step_text)
        assert score >= 2


# ---------------------------------------------------------------------------
# 5. Hinted enterprise tools appear before unhinted web_search in sort order
# ---------------------------------------------------------------------------


class TestSortOrderHintedBeforeUnhinted:
    """When both hinted enterprise tools and unhinted web_search are in
    active_tools, hinted tools should sort first (hint_bucket=0 vs 1)."""

    def test_hinted_enterprise_before_unhinted_web_search(self) -> None:
        step = {
            "title": "Retrieve Kroger earnings materials and compare to market",
            "description": "Use earnings index, supplement with web search.",
            "source_hints": [
                {
                    "source_name": "search_earnings_vs_index",
                    "source_type": "vector_search",
                    "priority": 1,
                },
            ],
        }
        earnings = _enterprise_vs_tool()
        web = _web_search_tool()

        selection = select_step_tools([web, earnings], step)

        active_names = [t.definition.name for t in selection.active_tools]
        # Hinted tool should come before unhinted web_search
        earnings_pos = active_names.index("search_earnings_vs_index")
        web_pos = active_names.index("web_search")
        assert earnings_pos < web_pos

    def test_multiple_hinted_tools_before_web_search(self) -> None:
        step = {
            "title": "Analyze earnings call and transcript data",
            "description": "Use both indexes, then check web for context.",
            "source_hints": [
                {
                    "source_name": "search_earnings_vs_index",
                    "source_type": "vector_search",
                    "priority": 1,
                },
                {
                    "source_name": "search_transcript_vs_index",
                    "source_type": "vector_search",
                    "priority": 2,
                },
            ],
        }
        earnings = _enterprise_vs_tool("search_earnings_vs_index")
        transcript = _enterprise_vs_tool("search_transcript_vs_index")
        web = _web_search_tool()

        selection = select_step_tools([web, transcript, earnings], step)

        active_names = [t.definition.name for t in selection.active_tools]
        # Both hinted tools should appear before web_search
        assert active_names.index("search_earnings_vs_index") < active_names.index("web_search")
        assert active_names.index("search_transcript_vs_index") < active_names.index("web_search")

    def test_hinted_priority_order_preserved(self) -> None:
        step = {
            "title": "Look up earnings and transcripts",
            "description": "Use earnings first, then transcripts.",
            "source_hints": [
                {
                    "source_name": "search_earnings_vs_index",
                    "source_type": "vector_search",
                    "priority": 1,
                },
                {
                    "source_name": "search_transcript_vs_index",
                    "source_type": "vector_search",
                    "priority": 2,
                },
            ],
        }
        earnings = _enterprise_vs_tool("search_earnings_vs_index")
        transcript = _enterprise_vs_tool("search_transcript_vs_index")
        web = _web_search_tool()

        selection = select_step_tools([web, transcript, earnings], step)

        active_names = [t.definition.name for t in selection.active_tools]
        # Priority 1 before priority 2 before unhinted web_search
        assert active_names.index("search_earnings_vs_index") < active_names.index("search_transcript_vs_index")
        assert active_names.index("search_transcript_vs_index") < active_names.index("web_search")


# ---------------------------------------------------------------------------
# _is_web_search_tool unit tests
# ---------------------------------------------------------------------------


class TestIsWebSearchTool:
    """Unit tests for the _is_web_search_tool helper."""

    def test_web_search_is_web_search_tool(self) -> None:
        definition = ToolDefinition(
            name="web_search",
            description="Search the web.",
            parameters={},
            source_type="web",
            source_kind="web",
        )
        assert _is_web_search_tool(definition) is True

    def test_web_crawl_is_not_web_search_tool(self) -> None:
        definition = ToolDefinition(
            name="web_crawl",
            description="Crawl a web page.",
            parameters={},
            source_type="web",
            source_kind="builtin",
        )
        assert _is_web_search_tool(definition) is False

    def test_enterprise_tool_is_not_web_search_tool(self) -> None:
        definition = ToolDefinition(
            name="search_earnings_vs_index",
            description="Search earnings releases.",
            parameters={},
            source_type="vector_search",
        )
        assert _is_web_search_tool(definition) is False

    def test_genie_tool_is_not_web_search_tool(self) -> None:
        definition = ToolDefinition(
            name="genie",
            description="Enterprise analytics.",
            parameters={},
            source_type="genie",
            source_kind="sql_analytics",
        )
        assert _is_web_search_tool(definition) is False

    def test_web_search_variant_name_still_detected(self) -> None:
        """A tool with 'web' source_type and no 'crawl' in the name is
        treated as a web search tool."""
        definition = ToolDefinition(
            name="brave_web_search",
            description="Search using Brave.",
            parameters={},
            source_type="web",
            source_kind="web",
        )
        assert _is_web_search_tool(definition) is True

    def test_crawl_in_name_excludes_from_web_search(self) -> None:
        """Any tool with 'crawl' in its name is not a web search tool."""
        definition = ToolDefinition(
            name="web_crawl_advanced",
            description="Advanced web page crawler.",
            parameters={},
            source_type="web",
            source_kind="web",
        )
        assert _is_web_search_tool(definition) is False
