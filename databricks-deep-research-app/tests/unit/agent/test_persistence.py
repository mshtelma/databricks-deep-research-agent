"""Unit tests for agent persistence layer.

Tests the citation key extraction fallback for the grey references bug fix.
"""

import pytest

from deep_research.agent.persistence import (
    CITATION_KEY_PATTERN,
    _ensure_citation_key,
)
from deep_research.agent.state import ClaimInfo, EvidenceInfo


class TestCitationKeyPattern:
    """Tests for the CITATION_KEY_PATTERN regex."""

    def test_matches_simple_key(self):
        """Test matching simple alphabetic keys like [Arxiv]."""
        matches = CITATION_KEY_PATTERN.findall("[Arxiv]")
        assert matches == ["Arxiv"]

    def test_matches_key_with_suffix(self):
        """Test matching keys with numeric suffix like [Arxiv-1]."""
        matches = CITATION_KEY_PATTERN.findall("[Arxiv-1]")
        assert matches == ["Arxiv-1"]

    def test_matches_alphanumeric_key(self):
        """Test matching alphanumeric keys like [Source2]."""
        matches = CITATION_KEY_PATTERN.findall("[Source2]")
        assert matches == ["Source2"]

    def test_matches_hyphenated_key(self):
        """Test matching hyphenated keys like [News-Site]."""
        matches = CITATION_KEY_PATTERN.findall("[News-Site]")
        assert matches == ["News-Site"]

    def test_matches_multiple_keys(self):
        """Test extracting multiple keys from text."""
        text = "This claim [Arxiv] is supported by [Wikipedia-1] and [Nature]."
        matches = CITATION_KEY_PATTERN.findall(text)
        assert matches == ["Arxiv", "Wikipedia-1", "Nature"]

    def test_does_not_match_numeric_only(self):
        """Test that pure numeric citations like [0] are not matched."""
        matches = CITATION_KEY_PATTERN.findall("[0]")
        assert matches == []

    def test_does_not_match_invalid_patterns(self):
        """Test that invalid patterns are not matched."""
        # Must start with letter
        assert CITATION_KEY_PATTERN.findall("[123]") == []
        assert CITATION_KEY_PATTERN.findall("[-Test]") == []
        # Must have closing bracket
        assert CITATION_KEY_PATTERN.findall("[Test") == []


class TestEnsureCitationKey:
    """Tests for _ensure_citation_key function."""

    def test_preserves_existing_citation_key(self):
        """Test that existing citation_key is preserved."""
        claim = ClaimInfo(
            claim_text="AI is transforming healthcare. [Arxiv]",
            claim_type="general",
            position_start=0,
            position_end=40,
            citation_key="Arxiv",
        )

        result = _ensure_citation_key(claim)

        assert result["citation_key"] == "Arxiv"
        assert result["claim_text"] == claim.claim_text

    def test_extracts_missing_citation_key_from_text(self):
        """Test extraction of citation_key when missing."""
        claim = ClaimInfo(
            claim_text="AI is transforming healthcare. [Arxiv]",
            claim_type="general",
            position_start=0,
            position_end=40,
            citation_key=None,  # Missing!
        )

        result = _ensure_citation_key(claim)

        assert result["citation_key"] == "Arxiv"

    def test_extracts_multiple_keys(self):
        """Test extraction when claim has multiple citation markers."""
        claim = ClaimInfo(
            claim_text="This fact [Arxiv] [Wikipedia] is well documented.",
            claim_type="general",
            position_start=0,
            position_end=50,
            citation_key=None,
        )

        result = _ensure_citation_key(claim)

        # First key becomes citation_key
        assert result["citation_key"] == "Arxiv"
        # All keys are stored in citation_keys
        assert result["citation_keys"] == ["Arxiv", "Wikipedia"]

    def test_no_extraction_for_uncited_claim(self):
        """Test that uncited claims (no markers) don't get a key."""
        claim = ClaimInfo(
            claim_text="This is an uncited statement.",
            claim_type="general",
            position_start=0,
            position_end=30,
            citation_key=None,
        )

        result = _ensure_citation_key(claim)

        # No key should be extracted
        assert result["citation_key"] is None

    def test_preserves_evidence_data(self):
        """Test that evidence data is preserved in output."""
        evidence = EvidenceInfo(
            source_url="https://arxiv.org/paper123",
            quote_text="AI research shows...",
        )
        claim = ClaimInfo(
            claim_text="AI research is advancing. [Arxiv]",
            claim_type="general",
            position_start=0,
            position_end=35,
            citation_key="Arxiv",
            evidence=evidence,
        )

        result = _ensure_citation_key(claim)

        assert result["evidence"] is not None
        assert result["evidence"]["source_url"] == "https://arxiv.org/paper123"

    def test_extracts_key_with_numeric_suffix(self):
        """Test extraction of keys with numeric suffix like [Source-1]."""
        claim = ClaimInfo(
            claim_text="Multiple sources confirm this. [Wikipedia-3]",
            claim_type="general",
            position_start=0,
            position_end=45,
            citation_key=None,
        )

        result = _ensure_citation_key(claim)

        assert result["citation_key"] == "Wikipedia-3"

    def test_preserves_other_claim_fields(self):
        """Test that all other claim fields are preserved."""
        claim = ClaimInfo(
            claim_text="Test claim [Source]",
            claim_type="numeric",
            position_start=100,
            position_end=120,
            citation_key=None,
            confidence_level="high",
            verification_verdict="supported",
            verification_reasoning="Evidence found",
            abstained=False,
            from_free_block=True,
        )

        result = _ensure_citation_key(claim)

        assert result["claim_type"] == "numeric"
        assert result["position_start"] == 100
        assert result["position_end"] == 120
        assert result["confidence_level"] == "high"
        assert result["verification_verdict"] == "supported"
        assert result["verification_reasoning"] == "Evidence found"
        assert result["abstained"] is False
        assert result["from_free_block"] is True
