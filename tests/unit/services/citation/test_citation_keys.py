"""Unit tests for citation key generation utilities.

Tests the human-readable citation key generation that replaces
numeric markers [0], [1], [2] with [Arxiv], [Zhipu], [Github-2].
"""

import pytest

from src.services.citation.citation_keys import (
    extract_domain_key,
    abbreviate_title,
    build_citation_key_map,
    replace_numeric_markers,
    parse_citation_key,
)
from src.services.citation.evidence_selector import RankedEvidence


class TestExtractDomainKey:
    """Tests for extract_domain_key function."""

    def test_simple_domain(self) -> None:
        """Test simple domain extraction."""
        assert extract_domain_key("https://arxiv.org/abs/123") == "Arxiv"

    def test_www_prefix_removed(self) -> None:
        """Test that www. prefix is removed."""
        assert extract_domain_key("https://www.github.com/repo") == "Github"

    def test_subdomain(self) -> None:
        """Test subdomain extraction (takes first part)."""
        assert extract_domain_key("https://docs.databricks.com/page") == "Docs"

    def test_invalid_url_fallback(self) -> None:
        """Test fallback for invalid URLs."""
        assert extract_domain_key("not-a-url") == "Web"
        assert extract_domain_key("") == "Web"


class TestAbbreviateTitle:
    """Tests for abbreviate_title function."""

    def test_model_with_version(self) -> None:
        """Test title with version number."""
        assert abbreviate_title("GLM-4.7 Technical Report") == "GLM47"

    def test_model_name_only(self) -> None:
        """Test model name without version."""
        assert abbreviate_title("Qwen2 Model Card") == "Qwen2"

    def test_long_first_word(self) -> None:
        """Test truncation of long first word."""
        assert abbreviate_title("Transformers are All You Need") == "Transf"

    def test_empty_title(self) -> None:
        """Test fallback for empty title."""
        assert abbreviate_title("") == "Doc"
        assert abbreviate_title("   ") == "Doc"

    def test_numbers_only_title(self) -> None:
        """Test title with only numbers."""
        # Only numbers, no letters - should extract numbers
        result = abbreviate_title("12345")
        # First "word" is 12345, so key should be 12345[:6] = "12345"
        assert result == "12345"


class TestBuildCitationKeyMap:
    """Tests for build_citation_key_map function."""

    def test_unique_domains(self) -> None:
        """Test unique keys for different domains."""
        evidence_pool = [
            RankedEvidence(
                source_url="https://arxiv.org/abs/123",
                source_title="Paper One",
                quote_text="Quote 1",
                relevance_score=0.9,
                source_id=None,
                start_offset=0,
                end_offset=10,
                section_heading=None,
                has_numeric_content=False,
            ),
            RankedEvidence(
                source_url="https://github.com/repo",
                source_title="Repo",
                quote_text="Quote 2",
                relevance_score=0.8,
                source_id=None,
                start_offset=0,
                end_offset=10,
                section_heading=None,
                has_numeric_content=False,
            ),
        ]

        key_map = build_citation_key_map(evidence_pool)

        assert key_map[0] == "Arxiv"
        assert key_map[1] == "Github"

    def test_collision_handling(self) -> None:
        """Test collision handling with discriminator suffix."""
        evidence_pool = [
            RankedEvidence(
                source_url="https://arxiv.org/abs/123",
                source_title="Paper One",
                quote_text="Quote 1",
                relevance_score=0.9,
                source_id=None,
                start_offset=0,
                end_offset=10,
                section_heading=None,
                has_numeric_content=False,
            ),
            RankedEvidence(
                source_url="https://arxiv.org/abs/456",
                source_title="Paper Two",
                quote_text="Quote 2",
                relevance_score=0.8,
                source_id=None,
                start_offset=0,
                end_offset=10,
                section_heading=None,
                has_numeric_content=False,
            ),
            RankedEvidence(
                source_url="https://arxiv.org/abs/789",
                source_title="Paper Three",
                quote_text="Quote 3",
                relevance_score=0.7,
                source_id=None,
                start_offset=0,
                end_offset=10,
                section_heading=None,
                has_numeric_content=False,
            ),
        ]

        key_map = build_citation_key_map(evidence_pool)

        assert key_map[0] == "Arxiv"
        assert key_map[1] == "Arxiv-2"
        assert key_map[2] == "Arxiv-3"

    def test_fallback_to_title(self) -> None:
        """Test fallback to title when URL not available."""
        evidence_pool = [
            RankedEvidence(
                source_url=None,
                source_title="GLM-4.7 Technical Report",
                quote_text="Quote",
                relevance_score=0.9,
                source_id=None,
                start_offset=0,
                end_offset=10,
                section_heading=None,
                has_numeric_content=False,
            ),
        ]

        key_map = build_citation_key_map(evidence_pool)

        assert key_map[0] == "GLM47"

    def test_fallback_to_source(self) -> None:
        """Test fallback to 'Source' when nothing available."""
        evidence_pool = [
            RankedEvidence(
                source_url=None,
                source_title=None,
                quote_text="Quote",
                relevance_score=0.9,
                source_id=None,
                start_offset=0,
                end_offset=10,
                section_heading=None,
                has_numeric_content=False,
            ),
        ]

        key_map = build_citation_key_map(evidence_pool)

        assert key_map[0] == "Source"

    def test_empty_pool(self) -> None:
        """Test empty evidence pool returns empty map."""
        key_map = build_citation_key_map([])
        assert key_map == {}


class TestReplaceNumericMarkers:
    """Tests for replace_numeric_markers function."""

    def test_simple_replacement(self) -> None:
        """Test simple marker replacement."""
        content = "Claim one [0]. Claim two [1]."
        key_map = {0: "Arxiv", 1: "Zhipu"}

        result = replace_numeric_markers(content, key_map)

        assert result == "Claim one [Arxiv]. Claim two [Zhipu]."

    def test_multiple_same_marker(self) -> None:
        """Test multiple occurrences of same marker."""
        content = "First [0]. Second [0] reference."
        key_map = {0: "Arxiv"}

        result = replace_numeric_markers(content, key_map)

        assert result == "First [Arxiv]. Second [Arxiv] reference."

    def test_missing_key_preserved(self) -> None:
        """Test that markers without mapping are preserved."""
        content = "Known [0]. Unknown [5]."
        key_map = {0: "Arxiv"}

        result = replace_numeric_markers(content, key_map)

        assert result == "Known [Arxiv]. Unknown [5]."

    def test_no_markers(self) -> None:
        """Test content without markers unchanged."""
        content = "No citations here."
        key_map = {0: "Arxiv"}

        result = replace_numeric_markers(content, key_map)

        assert result == "No citations here."

    def test_empty_content(self) -> None:
        """Test empty content returns empty."""
        result = replace_numeric_markers("", {0: "Arxiv"})
        assert result == ""


class TestParseCitationKey:
    """Tests for parse_citation_key function."""

    def test_simple_key(self) -> None:
        """Test parsing simple key."""
        assert parse_citation_key("[Arxiv]") == "Arxiv"
        assert parse_citation_key("[Zhipu]") == "Zhipu"

    def test_key_with_discriminator(self) -> None:
        """Test parsing key with discriminator."""
        assert parse_citation_key("[Arxiv-2]") == "Arxiv-2"
        assert parse_citation_key("[Github-10]") == "Github-10"

    def test_alphanumeric_key(self) -> None:
        """Test parsing alphanumeric key."""
        assert parse_citation_key("[GLM47]") == "GLM47"
        assert parse_citation_key("[Qwen2]") == "Qwen2"

    def test_numeric_marker_rejected(self) -> None:
        """Test that numeric markers return None."""
        assert parse_citation_key("[0]") is None
        assert parse_citation_key("[123]") is None

    def test_invalid_format_rejected(self) -> None:
        """Test that invalid formats return None."""
        assert parse_citation_key("Arxiv") is None  # No brackets
        assert parse_citation_key("[Arxiv") is None  # Missing close
        assert parse_citation_key("Arxiv]") is None  # Missing open
        assert parse_citation_key("[]") is None  # Empty
        assert parse_citation_key("[-Arxiv]") is None  # Starts with hyphen
