"""Unit tests for ReAct synthesis components.

Tests for:
- EvidenceRegistry: Index-based evidence access and retrieval tracking
- Synthesis tools: Tool definitions and formatters
- Claim extraction: Parsing claims from ReAct-generated content
"""

from uuid import uuid4

import pytest

from src.agent.tools.evidence_registry import (
    EvidenceRegistry,
    IndexedEvidence,
    RetrievalContext,
)
from src.agent.tools.synthesis_tools import (
    SYNTHESIS_TOOLS,
    format_search_results,
    format_snippet,
    get_synthesis_tool_by_name,
    get_synthesis_tool_names,
)
from src.services.citation.evidence_selector import RankedEvidence


# ============================================================================
# RetrievalContext Tests
# ============================================================================


class TestRetrievalContext:
    """Tests for the RetrievalContext grounding window."""

    def test_add_retrieval_basic(self) -> None:
        """Test adding retrievals to context."""
        context = RetrievalContext(window_size=3)

        context.add_retrieval(0, "First quote")
        context.add_retrieval(1, "Second quote")

        active = context.get_active_evidence()
        assert len(active) == 2
        assert active[0] == (0, "First quote")
        assert active[1] == (1, "Second quote")

    def test_window_size_limit(self) -> None:
        """Test that window size is enforced."""
        context = RetrievalContext(window_size=2)

        context.add_retrieval(0, "First")
        context.add_retrieval(1, "Second")
        context.add_retrieval(2, "Third")

        active = context.get_active_evidence()
        assert len(active) == 2
        # First should be evicted
        assert active[0] == (1, "Second")
        assert active[1] == (2, "Third")

    def test_clear_context(self) -> None:
        """Test clearing the context."""
        context = RetrievalContext(window_size=3)
        context.add_retrieval(0, "Quote")

        context.clear()
        assert len(context.get_active_evidence()) == 0


# ============================================================================
# EvidenceRegistry Tests
# ============================================================================


def create_evidence_pool() -> list[RankedEvidence]:
    """Create a test evidence pool."""
    return [
        RankedEvidence(
            source_id=uuid4(),
            source_url="https://arxiv.org/paper1",
            source_title="ArXiv Paper on ML",
            quote_text="Machine learning has achieved 95% accuracy on benchmark tasks.",
            start_offset=0,
            end_offset=100,
            section_heading="Results",
            relevance_score=0.9,
            has_numeric_content=True,
        ),
        RankedEvidence(
            source_id=uuid4(),
            source_url="https://github.com/project",
            source_title="GitHub Repository",
            quote_text="The implementation uses PyTorch and supports distributed training.",
            start_offset=0,
            end_offset=80,
            section_heading="README",
            relevance_score=0.7,
            has_numeric_content=False,
        ),
        RankedEvidence(
            source_id=uuid4(),
            source_url="https://wikipedia.org/article",
            source_title="Wikipedia Article",
            quote_text="Deep learning is a subset of machine learning.",
            start_offset=0,
            end_offset=50,
            section_heading=None,
            relevance_score=0.5,
            has_numeric_content=False,
        ),
    ]


class TestEvidenceRegistry:
    """Tests for the EvidenceRegistry."""

    def test_initialization(self) -> None:
        """Test registry initialization from evidence pool."""
        pool = create_evidence_pool()
        registry = EvidenceRegistry(pool)

        assert len(registry) == 3
        assert 0 in registry
        assert 1 in registry
        assert 2 in registry
        assert 99 not in registry

    def test_get_evidence(self) -> None:
        """Test getting evidence by index."""
        pool = create_evidence_pool()
        registry = EvidenceRegistry(pool)

        evidence = registry.get(0)
        assert evidence is not None
        assert evidence.source_url == "https://arxiv.org/paper1"
        assert evidence.has_numeric_content is True
        assert evidence.relevance_score == 0.9

    def test_get_records_access(self) -> None:
        """Test that get() records read access."""
        pool = create_evidence_pool()
        registry = EvidenceRegistry(pool)

        registry.get(0)
        registry.get(1)

        read_indices = registry.get_read_indices()
        assert 0 in read_indices
        assert 1 in read_indices
        assert 2 not in read_indices

    def test_get_entry_no_access_record(self) -> None:
        """Test that get_entry() does not record access."""
        pool = create_evidence_pool()
        registry = EvidenceRegistry(pool)

        registry.get_entry(0)

        read_indices = registry.get_read_indices()
        assert 0 not in read_indices

    def test_search_basic(self) -> None:
        """Test basic search functionality."""
        pool = create_evidence_pool()
        registry = EvidenceRegistry(pool)

        results = registry.search("machine learning accuracy")

        assert len(results) > 0
        # ArXiv paper should be top result (has ML and accuracy)
        assert results[0]["index"] == 0

    def test_search_numeric_type(self) -> None:
        """Test search with numeric claim type."""
        pool = create_evidence_pool()
        registry = EvidenceRegistry(pool)

        results = registry.search("performance", claim_type="numeric")

        # Should prefer evidence with has_numeric_content=True
        has_numeric = [r for r in results if r.get("has_numeric")]
        assert len(has_numeric) > 0

    def test_search_records_access(self) -> None:
        """Test that search records access for returned results."""
        pool = create_evidence_pool()
        registry = EvidenceRegistry(pool)

        results = registry.search("machine learning")

        audit = registry.get_access_audit()
        assert len(audit) > 0
        assert all(a["access_type"] == "search" for a in audit)

    def test_retrieval_context_updated(self) -> None:
        """Test that get() updates retrieval context."""
        pool = create_evidence_pool()
        registry = EvidenceRegistry(pool, retrieval_window_size=2)

        registry.get(0)
        registry.get(1)

        active = registry.get_active_retrievals()
        assert len(active) == 2

    def test_build_citation_key(self) -> None:
        """Test citation key generation."""
        pool = create_evidence_pool()
        registry = EvidenceRegistry(pool)

        key0 = registry.build_citation_key(0)
        key1 = registry.build_citation_key(1)
        key2 = registry.build_citation_key(2)

        assert key0 == "Arxiv"
        assert key1 == "Github"
        assert key2 == "Wikipedia"

    def test_list_all(self) -> None:
        """Test listing all evidence."""
        pool = create_evidence_pool()
        registry = EvidenceRegistry(pool)

        all_evidence = registry.list_all()
        assert len(all_evidence) == 3
        assert all_evidence[0].index == 0
        assert all_evidence[1].index == 1
        assert all_evidence[2].index == 2


# ============================================================================
# Synthesis Tools Tests
# ============================================================================


class TestSynthesisTools:
    """Tests for synthesis tool definitions."""

    def test_tool_definitions_exist(self) -> None:
        """Test that synthesis tools are defined."""
        assert len(SYNTHESIS_TOOLS) == 2
        names = get_synthesis_tool_names()
        assert "search_evidence" in names
        assert "read_snippet" in names

    def test_get_tool_by_name(self) -> None:
        """Test getting tool by name."""
        search_tool = get_synthesis_tool_by_name("search_evidence")
        assert search_tool is not None
        assert search_tool["function"]["name"] == "search_evidence"

        read_tool = get_synthesis_tool_by_name("read_snippet")
        assert read_tool is not None
        assert read_tool["function"]["name"] == "read_snippet"

        unknown = get_synthesis_tool_by_name("unknown_tool")
        assert unknown is None

    def test_search_evidence_schema(self) -> None:
        """Test search_evidence tool schema."""
        tool = get_synthesis_tool_by_name("search_evidence")
        assert tool is not None

        params = tool["function"]["parameters"]
        assert "query" in params["properties"]
        assert "claim_type" in params["properties"]
        assert "query" in params["required"]

    def test_read_snippet_schema(self) -> None:
        """Test read_snippet tool schema."""
        tool = get_synthesis_tool_by_name("read_snippet")
        assert tool is not None

        params = tool["function"]["parameters"]
        assert "index" in params["properties"]
        assert "index" in params["required"]


class TestToolFormatters:
    """Tests for tool output formatters."""

    def test_format_search_results_empty(self) -> None:
        """Test formatting empty search results."""
        result = format_search_results([])
        assert "No relevant evidence found" in result

    def test_format_search_results_with_items(self) -> None:
        """Test formatting search results with items."""
        results = [
            {
                "index": 0,
                "title": "Test Paper",
                "snippet_preview": "This is a test...",
                "relevance_score": 0.9,
                "has_numeric": True,
            },
            {
                "index": 1,
                "title": "Another Paper",
                "snippet_preview": "Another test...",
                "relevance_score": 0.7,
                "has_numeric": False,
            },
        ]

        formatted = format_search_results(results)
        assert "[0]" in formatted
        assert "[1]" in formatted
        assert "Test Paper" in formatted
        assert "[NUMERIC]" in formatted
        assert "read_snippet" in formatted

    def test_format_snippet(self) -> None:
        """Test formatting a snippet."""
        result = format_snippet(
            index=0,
            title="ArXiv Paper",
            quote_text="Machine learning achieves 95% accuracy.",
            section_heading="Results",
            citation_key="Arxiv",
        )

        assert "Evidence [0]" in result
        assert "ArXiv Paper" in result
        assert "(Section: Results)" in result
        assert "Machine learning achieves 95% accuracy" in result
        assert "[Arxiv]" in result

    def test_format_snippet_no_section(self) -> None:
        """Test formatting snippet without section."""
        result = format_snippet(
            index=1,
            title="GitHub Repo",
            quote_text="Implementation details.",
            section_heading=None,
            citation_key="Github",
        )

        assert "(Section:" not in result
        assert "GitHub Repo" in result


# ============================================================================
# Run tests with pytest
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
