"""Unit tests for ReAct synthesis components.

Tests for:
- EvidenceRegistry: Index-based evidence access and retrieval tracking
- Synthesis tools: Tool definitions and formatters
- Claim extraction: Parsing claims from ReAct-generated content
- Thinking text strip: Removes planning text from LLM output
- Citation validation: Ensures citations preserved after post-processing
- Block assembly: Source-based paragraph grouping
"""

from uuid import uuid4

import pytest

from deep_research.agent.nodes.react_synthesizer import (
    deduplicate_report,
    parse_tagged_content,
    strip_thinking_text,
    validate_citations_preserved,
)
from deep_research.agent.tools.evidence_registry import (
    EvidenceRegistry,
    IndexedEvidence,
    RetrievalContext,
)
from deep_research.agent.tools.synthesis_tools import (
    SYNTHESIS_TOOLS,
    format_search_results,
    format_snippet,
    get_synthesis_tool_by_name,
    get_synthesis_tool_names,
)
from deep_research.services.citation.evidence_selector import RankedEvidence


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
# Thinking Text Strip Tests
# ============================================================================


class TestStripThinkingText:
    """Tests for strip_thinking_text function."""

    def test_strips_ill_search_pattern(self) -> None:
        """Test stripping 'I'll search for...' patterns."""
        content = """I'll search for evidence about Basel III.
The Basel III framework was established in 2010 [Arxiv].
Implementation varies across jurisdictions [Wiki]."""
        result = strip_thinking_text(content)
        assert "I'll search for" not in result
        assert "Basel III framework" in result

    def test_strips_let_me_pattern(self) -> None:
        """Test stripping 'Let me search...' patterns."""
        content = """Let me find information about capital requirements.
Capital requirements are set at 10.5% [Source-1].
This includes the capital conservation buffer [Source-2]."""
        result = strip_thinking_text(content)
        assert "Let me find" not in result
        assert "Capital requirements" in result

    def test_strips_now_ill_pattern(self) -> None:
        """Test stripping 'Now I'll...' patterns."""
        content = """Now I'll look for more details.
First, I'll retrieve the implementation timeline.
The CRR came into effect in 2014 [Official]."""
        result = strip_thinking_text(content)
        assert "Now I'll" not in result
        assert "First, I'll retrieve" not in result
        assert "CRR came into effect" in result

    def test_strips_searching_pattern(self) -> None:
        """Test stripping 'Searching for...' patterns."""
        content = """Searching for regulatory information...
The framework includes capital requirements [Source]."""
        result = strip_thinking_text(content)
        assert "Searching for" not in result
        assert "framework includes" in result

    def test_preserves_legitimate_content(self) -> None:
        """Test that legitimate content is not stripped."""
        content = """The CEO stated "I'll search for a solution to this problem."
This quote shows the company's commitment [Press]."""
        result = strip_thinking_text(content)
        # This should be preserved because it's within a quote
        assert "CEO stated" in result
        assert "commitment" in result

    def test_handles_empty_content(self) -> None:
        """Test handling empty content."""
        assert strip_thinking_text("") == ""
        assert strip_thinking_text("   ") == ""

    def test_handles_only_thinking_text(self) -> None:
        """Test content that is only thinking text."""
        content = """I'll search for information.
Let me find the relevant data.
Now I'll look for more context."""
        result = strip_thinking_text(content)
        assert result == ""

    def test_preserves_markdown_headers(self) -> None:
        """Test that markdown headers are preserved."""
        content = """## Key Findings
I'll search for implementation details.
The framework was introduced in 2010 [Source]."""
        result = strip_thinking_text(content)
        assert "## Key Findings" in result
        assert "I'll search" not in result

    def test_strips_i_need_to_pattern(self) -> None:
        """Test stripping 'I need to search...' patterns."""
        content = """I need to find more information about this.
I should search for regulatory details.
The regulation is complex [Source]."""
        result = strip_thinking_text(content)
        assert "I need to find" not in result
        assert "I should search" not in result
        assert "regulation is complex" in result


# ============================================================================
# Report Deduplication Tests
# ============================================================================


class TestDeduplicateReport:
    """Tests for deduplicate_report function that removes duplicated report sections."""

    def test_no_duplication_returns_unchanged(self) -> None:
        """Test that content with single Introduction is unchanged."""
        content = """## Introduction

This is a research report about Basel III.

## Key Findings

Capital requirements are set at 10.5% [Source-1].

## Conclusion

The regulatory framework is comprehensive."""
        result = deduplicate_report(content)
        assert result == content

    def test_duplicate_introduction_keeps_longer_with_conclusion(self) -> None:
        """Test that when both have conclusions, longer one is kept."""
        content = """## Introduction

First report content about Basel III.

## Conclusion

First conclusion.

## Introduction

Second report content (duplicate) with more detailed information.
This one has significantly more content than the first.

## Key Findings

Detailed duplicate findings with extra content.

## Conclusion

Longer duplicate conclusion with more analysis."""
        result = deduplicate_report(content)
        # Both have conclusions, longer one (second) is kept
        assert "Second report content" in result
        assert "Detailed duplicate findings" in result
        assert "Longer duplicate conclusion" in result
        # First report should be removed (shorter)
        assert "First report content about Basel III" not in result
        assert "First conclusion" not in result

    def test_handles_empty_content(self) -> None:
        """Test handling empty content."""
        assert deduplicate_report("") == ""

    def test_handles_no_introduction(self) -> None:
        """Test content without ## Introduction is unchanged."""
        content = """Some content without a header.

More content here."""
        result = deduplicate_report(content)
        assert result == content

    def test_handles_h3_introduction(self) -> None:
        """Test that ### Introduction (h3) is NOT treated as duplication marker."""
        content = """## Introduction

Main intro.

### Introduction Details

More details here."""
        result = deduplicate_report(content)
        # h3 should not trigger deduplication
        assert result == content

    def test_strips_trailing_whitespace(self) -> None:
        """Test that trailing whitespace is stripped after selection."""
        content = """## Introduction

First content (shorter, with conclusion).

## Conclusion

First.


## Introduction

Second content is longer than first.
More lines here.

## Conclusion

Second conclusion."""
        result = deduplicate_report(content)
        assert not result.endswith(" ")
        assert not result.endswith("\n")
        # Second is longer with conclusion, so it's kept
        assert "Second content is longer" in result
        assert "First content (shorter" not in result

    def test_real_world_duplication_pattern(self) -> None:
        """Test pattern observed in Basel IV trace: both have conclusions, pick longest."""
        content = """## Introduction

Basel III Endgame: Core capital changes.

## Key Regulatory Changes

CRR III entered into force on 1 January 2025 [ECB].

## Conclusion

Banks' capital stacking strategies will need adjustment.

## Introduction

Basel III Endgame ("Basel IV"): Core framework revisions.

## Key Regulatory Updates

CRR 3 (EU): Implementation timeline details. More detailed content here.

## Conclusion

EU Capital Stack Complexity increases. This conclusion is longer."""
        result = deduplicate_report(content)
        # Both have conclusions, so longer one is kept (second one)
        assert 'Basel III Endgame ("Basel IV")' in result
        assert "CRR 3 (EU)" in result
        assert "This conclusion is longer" in result
        # First version removed
        assert "Basel III Endgame: Core capital changes" not in result

    def test_multiline_introduction(self) -> None:
        """Test ## Introduction at various line positions - keeps longer when no conclusion."""
        content = """## Introduction
First paragraph.

More content.

## Introduction
Second intro with more text here."""
        result = deduplicate_report(content)
        assert result.count("## Introduction") == 1
        # Neither has conclusion, so longer one is kept (second)
        assert "Second intro" in result
        assert "First paragraph" not in result

    def test_keeps_report_with_conclusion(self) -> None:
        """Test that report with conclusion is preferred over incomplete one."""
        content = """## Introduction

First report content that is incomplete.

## Key Findings

Some findings here.

## Introduction

Second report content.

## Key Findings

Different findings.

## Conclusion

This is the conclusion of the second report."""
        result = deduplicate_report(content)
        # Second report has conclusion, first doesn't - keep second
        assert "Second report content" in result
        assert "This is the conclusion" in result
        assert "First report content that is incomplete" not in result

    def test_keeps_longer_when_neither_has_conclusion(self) -> None:
        """When neither report has a conclusion, keep the longer one."""
        content = """## Introduction

Short first report.

## Introduction

This is a much longer second report with lots of content.
It has multiple paragraphs and detailed information.
This makes it the better choice when neither has a conclusion."""
        result = deduplicate_report(content)
        # Neither has conclusion, keep longer (second)
        assert "much longer second report" in result
        assert "Short first report" not in result


# ============================================================================
# Citation Validation Tests
# ============================================================================


class TestValidateCitationsPreserved:
    """Tests for validate_citations_preserved function."""

    def test_identical_citations_returns_true(self) -> None:
        """Test that identical citations return True."""
        original = "The rate is 10% [Source-1]. The limit is 8% [Source-2]."
        polished = "At 10%, the rate is significant [Source-1]. Meanwhile, the limit stands at 8% [Source-2]."
        assert validate_citations_preserved(original, polished) is True

    def test_missing_citation_returns_false(self) -> None:
        """Test that missing citation returns False."""
        original = "Rate is 10% [Source-1]. Limit is 8% [Source-2]."
        polished = "Rate is 10% [Source-1]. Limit is 8%."
        assert validate_citations_preserved(original, polished) is False

    def test_extra_citation_returns_false(self) -> None:
        """Test that extra citation returns False."""
        original = "Rate is 10% [Source-1]."
        polished = "Rate is 10% [Source-1] [Source-2]."
        assert validate_citations_preserved(original, polished) is False

    def test_empty_content_returns_true(self) -> None:
        """Test empty content returns True."""
        assert validate_citations_preserved("", "") is True

    def test_no_citations_returns_true(self) -> None:
        """Test content without citations returns True."""
        original = "Some text without citations."
        polished = "Polished text without citations."
        assert validate_citations_preserved(original, polished) is True

    def test_complex_citation_keys(self) -> None:
        """Test various citation key formats."""
        original = "[Arxiv] paper and [Github-2] repo and [Wikipedia-10] article."
        polished = "Article from [Wikipedia-10], repo [Github-2], and paper [Arxiv]."
        assert validate_citations_preserved(original, polished) is True

    def test_citation_order_doesnt_matter(self) -> None:
        """Test that citation order doesn't affect validation."""
        original = "[A] first [B] second [C] third."
        polished = "[C] first [A] second [B] third."
        assert validate_citations_preserved(original, polished) is True

    def test_duplicate_citations_counted(self) -> None:
        """Test that duplicate citations are handled correctly."""
        original = "[Source-1] once [Source-1] twice."
        polished = "[Source-1] appears twice."
        # Both have the same set of unique citations
        assert validate_citations_preserved(original, polished) is True


# ============================================================================
# Block Assembly Tests (parse_tagged_content)
# ============================================================================


class TestParseTaggedContent:
    """Tests for parse_tagged_content function."""

    def test_parses_cite_tags(self) -> None:
        """Test parsing <cite> tags."""
        content = '<cite key="Source-1">This is a cited claim.</cite>'
        assembled, blocks = parse_tagged_content(content)
        assert len(blocks) == 1
        assert blocks[0].tag_type == "cite"
        assert blocks[0].citation_key == "Source-1"
        assert "Source-1" in assembled

    def test_parses_free_tags(self) -> None:
        """Test parsing <free> tags."""
        content = "<free>## Introduction</free>"
        assembled, blocks = parse_tagged_content(content)
        assert len(blocks) == 1
        assert blocks[0].tag_type == "free"
        assert "## Introduction" in assembled

    def test_parses_mixed_content(self) -> None:
        """Test parsing mixed cite and free tags."""
        content = """<free>## Key Findings</free>
<free>The regulatory landscape has evolved.</free>
<cite key="Arxiv">The framework was established in 2010.</cite>
<free>This laid the groundwork.</free>
<cite key="Wiki">Implementation varied across jurisdictions.</cite>"""
        assembled, blocks = parse_tagged_content(content)
        assert len(blocks) == 5
        assert blocks[0].tag_type == "free"
        assert blocks[2].tag_type == "cite"
        assert "[Arxiv]" in assembled
        assert "[Wiki]" in assembled

    def test_headers_get_own_line(self) -> None:
        """Test that markdown headers get their own paragraph."""
        content = """<free>## Section One</free>
<cite key="Source-1">First claim.</cite>
<free>## Section Two</free>
<cite key="Source-2">Second claim.</cite>"""
        assembled, blocks = parse_tagged_content(content)
        lines = [line.strip() for line in assembled.split("\n\n") if line.strip()]
        # Headers should be separate
        assert any("## Section One" in line for line in lines)
        assert any("## Section Two" in line for line in lines)

    def test_multiple_sources_same_paragraph(self) -> None:
        """Test that multiple sources can be in the same paragraph (flowing prose)."""
        content = """<cite key="Arxiv-1">First Arxiv claim.</cite>
<cite key="Arxiv-2">Second Arxiv claim.</cite>
<cite key="Wiki">Wiki claim.</cite>"""
        assembled, blocks = parse_tagged_content(content)
        # All claims should be in same paragraph (no source-based breaks)
        paragraphs = [p.strip() for p in assembled.split("\n\n") if p.strip()]
        assert len(paragraphs) == 1
        # All citations should be in the same paragraph
        assert "[Arxiv-1]" in paragraphs[0]
        assert "[Arxiv-2]" in paragraphs[0]
        assert "[Wiki]" in paragraphs[0]

    def test_different_sources_flow_together(self) -> None:
        """Test that different sources flow together in prose (no forced breaks)."""
        content = """<cite key="Source-A">Claim from A.</cite>
<cite key="Source-B">Claim from B.</cite>"""
        assembled, blocks = parse_tagged_content(content)
        paragraphs = [p.strip() for p in assembled.split("\n\n") if p.strip()]
        # Should be 1 paragraph - no source-based breaks for flowing prose
        assert len(paragraphs) == 1
        assert "[Source-A]" in paragraphs[0]
        assert "[Source-B]" in paragraphs[0]

    def test_untagged_content_is_discarded(self) -> None:
        """Test that untagged content is discarded (scratchpad behavior)."""
        content = "Plain text without any tags."
        assembled, blocks = parse_tagged_content(content)
        # Untagged content is treated as scratchpad and discarded
        assert assembled == ""
        assert len(blocks) == 0

    def test_handles_empty_content(self) -> None:
        """Test handling empty content."""
        assembled, blocks = parse_tagged_content("")
        assert assembled == ""
        assert len(blocks) == 0

    def test_parses_unverified_tags(self) -> None:
        """Test parsing <unverified> tags."""
        content = "<unverified>This claim is unverified.</unverified>"
        assembled, blocks = parse_tagged_content(content)
        assert len(blocks) == 1
        assert blocks[0].tag_type == "unverified"
        # Unverified content is included in paragraph but without special marker
        assert "This claim is unverified." in assembled

    def test_parses_analysis_tags(self) -> None:
        """Test parsing <analysis> tags for author's synthesis/conclusions."""
        content = """<cite key="Arxiv">GPT-4 achieves 86.4% accuracy on MMLU.</cite>
<analysis>This demonstrates significant progress in language model capabilities.</analysis>"""
        assembled, blocks = parse_tagged_content(content)
        assert len(blocks) == 2
        assert blocks[0].tag_type == "cite"
        assert blocks[0].citation_key == "Arxiv"
        assert blocks[1].tag_type == "analysis"
        assert blocks[1].citation_key is None  # Analysis has no citation
        # Analysis content flows into the same paragraph
        assert "significant progress" in assembled
        # Analysis is NOT marked with a citation
        assert "[Arxiv]" in assembled
        assert "[analysis]" not in assembled.lower()

    def test_analysis_mixed_with_cite_and_free(self) -> None:
        """Test <analysis> blocks mix correctly with <cite> and <free>."""
        content = """<free>## Results</free>
<cite key="Paper-1">The model achieved state-of-the-art results.</cite>
<analysis>These findings suggest a paradigm shift in the field.</analysis>
<free>## Conclusion</free>
<analysis>Overall, the research represents a major advancement.</analysis>"""
        assembled, blocks = parse_tagged_content(content)
        assert len(blocks) == 5
        # Verify tag types
        assert blocks[0].tag_type == "free"
        assert blocks[1].tag_type == "cite"
        assert blocks[2].tag_type == "analysis"
        assert blocks[3].tag_type == "free"
        assert blocks[4].tag_type == "analysis"
        # Headers should be on separate lines
        assert "## Results" in assembled
        assert "## Conclusion" in assembled
        # Analysis content included
        assert "paradigm shift" in assembled
        assert "major advancement" in assembled

    def test_free_text_joins_paragraph(self) -> None:
        """Test that free text stays with current paragraph."""
        content = """<cite key="Source-1">Main claim here.</cite>
<free>Furthermore, this is important context.</free>
<cite key="Source-1">Another related claim.</cite>"""
        assembled, blocks = parse_tagged_content(content)
        # All three should be in same paragraph (same source + connector)
        paragraphs = [p.strip() for p in assembled.split("\n\n") if p.strip()]
        # Should be a single paragraph since same source
        assert len(paragraphs) == 1
        assert "Furthermore" in paragraphs[0]


# ============================================================================
# Run tests with pytest
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
