"""Tests for tracing constants and span name builders."""

import pytest

from deep_research.core.tracing_constants import (
    DOMAIN_CITATION,
    DOMAIN_RESEARCH,
    DOMAIN_TOOL,
    PHASE_CLASSIFY,
    PHASE_EXECUTE,
    PHASE_PLAN,
    PHASE_REFLECT,
    STAGE_7_ARE,
    citation_span_name,
    list_to_attr,
    research_span_name,
    safe_attr_value,
    tool_span_name,
    truncate_for_attr,
)


class TestResearchSpanName:
    """Tests for research_span_name function."""

    def test_basic_span_name(self):
        """Test basic span name without step or iteration."""
        result = research_span_name(PHASE_CLASSIFY, "coordinator")
        assert result == "research.classify.coordinator"

    def test_span_name_with_step(self):
        """Test span name with step index."""
        result = research_span_name(PHASE_EXECUTE, "researcher", step=2)
        assert result == "research.execute.researcher.step_2"

    def test_span_name_with_iteration(self):
        """Test span name with iteration number."""
        result = research_span_name(PHASE_PLAN, "planner", iteration=1)
        assert result == "research.plan.planner.iteration_1"

    def test_step_takes_precedence_over_iteration(self):
        """Test that step takes precedence when both provided."""
        result = research_span_name(PHASE_EXECUTE, "researcher", step=3, iteration=2)
        assert result == "research.execute.researcher.step_3"

    def test_different_phases(self):
        """Test span names for different research phases."""
        assert research_span_name(PHASE_CLASSIFY, "coordinator") == "research.classify.coordinator"
        assert research_span_name(PHASE_PLAN, "planner") == "research.plan.planner"
        assert research_span_name(PHASE_EXECUTE, "researcher") == "research.execute.researcher"
        assert research_span_name(PHASE_REFLECT, "reflector") == "research.reflect.reflector"


class TestCitationSpanName:
    """Tests for citation_span_name function."""

    def test_basic_citation_span(self):
        """Test basic citation span without indices."""
        result = citation_span_name(STAGE_7_ARE, "retrieve_and_revise")
        assert result == "citation.stage_7.retrieve_and_revise"

    def test_citation_span_with_claim_index(self):
        """Test citation span with claim index."""
        result = citation_span_name(STAGE_7_ARE, "decompose", 2)
        assert result == "citation.stage_7.claim[2].decompose"

    def test_citation_span_with_claim_and_fact_index(self):
        """Test citation span with both claim and fact indices."""
        result = citation_span_name(STAGE_7_ARE, "verify", 2, 0)
        assert result == "citation.stage_7.claim[2].fact[0].verify"

    def test_citation_span_operations(self):
        """Test various Stage 7 operations."""
        assert citation_span_name(STAGE_7_ARE, "internal_search", 1, 2).endswith("internal_search")
        assert citation_span_name(STAGE_7_ARE, "external_search", 0, 1).endswith("external_search")
        assert citation_span_name(STAGE_7_ARE, "entailment_check", 3, 0).endswith("entailment_check")


class TestToolSpanName:
    """Tests for tool_span_name function."""

    def test_basic_tool_span(self):
        """Test basic tool span without context."""
        result = tool_span_name("web_search")
        assert result == "tool.web_search"

    def test_tool_span_with_context(self):
        """Test tool span with context."""
        result = tool_span_name("web_search", "step_1")
        assert result == "tool.web_search.step_1"

    def test_different_tools(self):
        """Test span names for different tools."""
        assert tool_span_name("web_crawl") == "tool.web_crawl"
        assert tool_span_name("web_crawl", "background") == "tool.web_crawl.background"


class TestTruncateForAttr:
    """Tests for truncate_for_attr function."""

    def test_short_text_unchanged(self):
        """Test that short text is unchanged."""
        result = truncate_for_attr("short text", max_length=100)
        assert result == "short text"

    def test_long_text_truncated(self):
        """Test that long text is truncated with ellipsis."""
        long_text = "a" * 300
        result = truncate_for_attr(long_text, max_length=100)
        assert len(result) == 100
        assert result.endswith("...")

    def test_none_returns_empty_string(self):
        """Test that None returns empty string."""
        result = truncate_for_attr(None)
        assert result == ""

    def test_default_max_length(self):
        """Test default max_length of 200."""
        long_text = "a" * 300
        result = truncate_for_attr(long_text)
        assert len(result) == 200


class TestListToAttr:
    """Tests for list_to_attr function."""

    def test_short_list_json(self):
        """Test that short list is serialized to JSON."""
        result = list_to_attr(["a", "b", "c"])
        assert result == '["a", "b", "c"]'

    def test_list_truncated_to_max_items(self):
        """Test that list is truncated to max_items."""
        result = list_to_attr(["a", "b", "c", "d", "e"], max_items=2)
        assert result == '["a", "b"]'

    def test_empty_list(self):
        """Test empty list."""
        result = list_to_attr([])
        assert result == "[]"


class TestSafeAttrValue:
    """Tests for safe_attr_value function."""

    def test_string_unchanged(self):
        """Test that string is unchanged."""
        result = safe_attr_value("hello")
        assert result == "hello"

    def test_int_unchanged(self):
        """Test that int is unchanged."""
        result = safe_attr_value(42)
        assert result == 42

    def test_float_unchanged(self):
        """Test that float is unchanged."""
        result = safe_attr_value(3.14)
        assert result == 3.14

    def test_bool_unchanged(self):
        """Test that bool is unchanged."""
        result = safe_attr_value(True)
        assert result is True

    def test_list_converted_to_json(self):
        """Test that list is converted to JSON string."""
        result = safe_attr_value(["a", "b", "c"])
        assert result == '["a", "b", "c"]'

    def test_dict_converted_to_json(self):
        """Test that dict is converted to JSON string."""
        result = safe_attr_value({"key": "value"})
        assert result == '{"key": "value"}'

    def test_long_list_truncated(self):
        """Test that long list is truncated to 10 items."""
        long_list = list(range(20))
        result = safe_attr_value(long_list)
        import json
        parsed = json.loads(result)
        assert len(parsed) == 10


class TestDomainConstants:
    """Tests for domain constants."""

    def test_domain_values(self):
        """Test domain constant values."""
        assert DOMAIN_RESEARCH == "research"
        assert DOMAIN_CITATION == "citation"
        assert DOMAIN_TOOL == "tool"

    def test_stage_7_constant(self):
        """Test Stage 7 constant."""
        assert STAGE_7_ARE == "stage_7"
