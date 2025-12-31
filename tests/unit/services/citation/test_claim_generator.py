"""Unit tests for InterleavedGenerator service."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from src.services.citation.claim_generator import InterleavedClaim, InterleavedGenerator
from src.services.citation.evidence_selector import RankedEvidence

from .conftest import MockLLMResponse


class TestSelectBestEvidence:
    """Tests for select_best_evidence method."""

    def test_returns_highest_relevance(
        self, mock_llm_client, sample_evidence_list, patch_app_config
    ):
        """Evidence ranking by relevance."""
        generator = InterleavedGenerator(mock_llm_client)

        # Query that matches first evidence best (contains "revenue", "billion")
        best_evidence, index = generator.select_best_evidence(
            _query="company revenue financial results",
            claim_context="revenue increased by billions",
            evidence_pool=sample_evidence_list,
        )

        assert best_evidence is not None
        assert best_evidence.relevance_score == 0.9
        assert index == 0

    def test_returns_none_below_threshold(
        self, mock_llm_client, patch_app_config
    ):
        """Threshold enforcement - returns None when no match."""
        generator = InterleavedGenerator(mock_llm_client)

        # Evidence that doesn't match the claim context
        evidence_pool = [
            RankedEvidence(
                source_id=None,
                source_url="https://example.com",
                source_title="Test",
                quote_text="The weather is nice today.",
                start_offset=0,
                end_offset=30,
                section_heading=None,
                relevance_score=0.2,
                has_numeric_content=False,
            )
        ]

        best_evidence, index = generator.select_best_evidence(
            _query="artificial intelligence machine learning",
            claim_context="AI and ML technologies",
            evidence_pool=evidence_pool,
        )

        # Should return None when no good match
        assert best_evidence is None
        assert index is None


class TestParseInterleavedContent:
    """Tests for _parse_interleaved_content method."""

    def test_extracts_citations(self, mock_llm_client, sample_evidence_list, patch_app_config):
        """Citation marker parsing."""
        generator = InterleavedGenerator(mock_llm_client)

        content = "Revenue increased significantly [0]. The company expanded globally [1]."

        claims = generator._parse_interleaved_content(content, sample_evidence_list)

        assert len(claims) == 2
        assert claims[0].claim_text == "Revenue increased significantly."
        assert claims[0].evidence_index == 0
        assert claims[1].claim_text == "The company expanded globally."
        assert claims[1].evidence_index == 1

    def test_handles_multiple_citations(
        self, mock_llm_client, sample_evidence_list, patch_app_config
    ):
        """Multiple citation handling in single sentence."""
        generator = InterleavedGenerator(mock_llm_client)

        content = "The data shows growth [0] and expansion [1] trends."

        claims = generator._parse_interleaved_content(content, sample_evidence_list)

        assert len(claims) == 1
        # Should use first citation as primary
        assert claims[0].evidence_index == 0

    def test_handles_no_citations(self, mock_llm_client, sample_evidence_list, patch_app_config):
        """Content without citations."""
        generator = InterleavedGenerator(mock_llm_client)

        content = "This is a statement without any citations."

        claims = generator._parse_interleaved_content(content, sample_evidence_list)

        assert len(claims) == 1
        assert claims[0].evidence is None
        assert claims[0].evidence_index is None


class TestHasNumericContent:
    """Tests for _has_numeric_content method."""

    def test_detects_currency(self, mock_llm_client, patch_app_config):
        """Currency detection."""
        generator = InterleavedGenerator(mock_llm_client)

        assert generator._has_numeric_content("Revenue reached $5.2B")
        assert generator._has_numeric_content("Costs were $1,234M")

    def test_detects_percentages(self, mock_llm_client, patch_app_config):
        """Percentage detection."""
        generator = InterleavedGenerator(mock_llm_client)

        assert generator._has_numeric_content("Growth was 25%")
        assert generator._has_numeric_content("Increased by 3.5%")

    def test_returns_false_for_no_numbers(self, mock_llm_client, patch_app_config):
        """Returns False for text without numbers."""
        generator = InterleavedGenerator(mock_llm_client)

        assert not generator._has_numeric_content("The company grew significantly")
        assert not generator._has_numeric_content("Markets expanded globally")


class TestSynthesizeWithInterleaving:
    """Tests for synthesize_with_interleaving async generator."""

    @pytest.mark.asyncio
    async def test_yields_claims(
        self, mock_llm_client, sample_evidence_list, patch_app_config
    ):
        """Async generator flow with mocked LLM."""
        # Setup mock LLM response with citations
        mock_response = "Revenue increased by 30% [0]. The company also expanded into new markets [1]. Customer satisfaction improved significantly [2]."
        mock_llm_client.complete.return_value = MockLLMResponse(content=mock_response)

        generator = InterleavedGenerator(mock_llm_client)

        claims = []
        async for claim in generator.synthesize_with_interleaving(
            query="company performance",
            evidence_pool=sample_evidence_list,
            previous_content="",
        ):
            claims.append(claim)

        assert len(claims) >= 1
        assert all(isinstance(c, InterleavedClaim) for c in claims)

    @pytest.mark.asyncio
    async def test_yields_nothing_for_empty_pool(self, mock_llm_client, patch_app_config):
        """Returns nothing when evidence pool is empty."""
        generator = InterleavedGenerator(mock_llm_client)

        claims = []
        async for claim in generator.synthesize_with_interleaving(
            query="company performance",
            evidence_pool=[],
            previous_content="",
        ):
            claims.append(claim)

        assert len(claims) == 0


class TestMatchClaimToEvidence:
    """Tests for match_claim_to_evidence method."""

    @pytest.mark.asyncio
    async def test_returns_index(
        self, mock_llm_client, sample_evidence_list, patch_app_config
    ):
        """Claim-evidence matching returns correct index."""
        from src.services.citation.claim_generator import ClaimEvidenceMatchOutput

        # Mock LLM to return a structured response
        mock_structured = ClaimEvidenceMatchOutput(
            evidence_index=0,
            entailment="full",
            reasoning="Direct match"
        )
        mock_llm_client.complete.return_value = MockLLMResponse(
            content="",
            structured=mock_structured,
        )

        generator = InterleavedGenerator(mock_llm_client)

        index, entailment, reasoning = await generator.match_claim_to_evidence(
            claim_text="Revenue increased significantly",
            evidence_pool=sample_evidence_list,
        )

        assert index == 0
        assert entailment == "full"
        assert "match" in reasoning.lower()

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_pool(self, mock_llm_client, patch_app_config):
        """Returns None for empty evidence pool."""
        generator = InterleavedGenerator(mock_llm_client)

        index, entailment, reasoning = await generator.match_claim_to_evidence(
            claim_text="Some claim",
            evidence_pool=[],
        )

        assert index is None
        assert entailment == "none"
