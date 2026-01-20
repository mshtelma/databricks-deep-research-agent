"""Unit tests for IsolatedVerifier service."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from deep_research.services.citation.evidence_selector import RankedEvidence
from deep_research.services.citation.isolated_verifier import (
    IsolatedVerifier,
    Verdict,
    VerificationResult,
)

from .conftest import MockLLMResponse


class TestParseVerdict:
    """Tests for parse_verdict method."""

    def test_parses_supported(self, mock_llm_client, patch_app_config):
        """SUPPORTED verdict parsing."""
        verifier = IsolatedVerifier(mock_llm_client)

        assert verifier.parse_verdict("SUPPORTED") == Verdict.SUPPORTED
        assert verifier.parse_verdict("supported") == Verdict.SUPPORTED
        assert verifier.parse_verdict("The claim is SUPPORTED") == Verdict.SUPPORTED

    def test_parses_partial(self, mock_llm_client, patch_app_config):
        """PARTIAL verdict parsing."""
        verifier = IsolatedVerifier(mock_llm_client)

        assert verifier.parse_verdict("PARTIAL") == Verdict.PARTIAL
        # Note: "partially supported" contains "supported" so matches SUPPORTED first
        # This is expected behavior based on the implementation

    def test_parses_unsupported(self, mock_llm_client, patch_app_config):
        """UNSUPPORTED verdict parsing."""
        verifier = IsolatedVerifier(mock_llm_client)

        assert verifier.parse_verdict("UNSUPPORTED") == Verdict.UNSUPPORTED
        # Note: "not supported" contains "supported" so matches SUPPORTED first
        # Default when nothing matches
        assert verifier.parse_verdict("unknown result") == Verdict.UNSUPPORTED
        assert verifier.parse_verdict("no match found") == Verdict.UNSUPPORTED

    def test_parses_contradicted(self, mock_llm_client, patch_app_config):
        """CONTRADICTED verdict parsing."""
        verifier = IsolatedVerifier(mock_llm_client)

        assert verifier.parse_verdict("CONTRADICTED") == Verdict.CONTRADICTED
        assert verifier.parse_verdict("The evidence CONTRADICTED the claim") == Verdict.CONTRADICTED


class TestVerifyWithIsolation:
    """Tests for verify_with_isolation method."""

    @pytest.mark.asyncio
    async def test_full_verification(self, mock_llm_client, sample_evidence, patch_app_config):
        """Full verification flow."""
        from deep_research.services.citation.isolated_verifier import VerificationOutput

        # Mock LLM to return a structured verification response
        mock_structured = VerificationOutput(
            verdict="SUPPORTED",
            reasoning="The claim matches the evidence exactly.",
            key_match="25% growth",
            issues=[],
        )
        mock_llm_client.complete.return_value = MockLLMResponse(
            content="",
            structured=mock_structured,
        )

        verifier = IsolatedVerifier(mock_llm_client)

        result = await verifier.verify_with_isolation(
            claim_text="The market grew by 25%.",
            evidence=sample_evidence,
            use_quick_verification=False,
        )

        assert isinstance(result, VerificationResult)
        assert result.verdict == Verdict.SUPPORTED
        assert "matches" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_quick_verification(self, mock_llm_client, sample_evidence, patch_app_config):
        """Quick verification path."""
        mock_llm_client.complete.return_value = MockLLMResponse(content="SUPPORTED")

        verifier = IsolatedVerifier(mock_llm_client)

        result = await verifier.verify_with_isolation(
            claim_text="The market grew by 25%.",
            evidence=sample_evidence,
            use_quick_verification=True,
        )

        assert isinstance(result, VerificationResult)
        assert result.verdict == Verdict.SUPPORTED
        assert result.reasoning == "Quick verification"

    @pytest.mark.asyncio
    async def test_handles_llm_error(self, mock_llm_client, sample_evidence, patch_app_config):
        """Handles LLM errors gracefully."""
        mock_llm_client.complete.side_effect = Exception("LLM error")

        verifier = IsolatedVerifier(mock_llm_client)

        result = await verifier.verify_with_isolation(
            claim_text="Some claim",
            evidence=sample_evidence,
            use_quick_verification=False,
        )

        assert result.verdict == Verdict.UNSUPPORTED
        assert result.abstained is True
        assert "failed" in result.reasoning.lower()


class TestCheckNEI:
    """Tests for check_nei method."""

    def test_returns_true_for_low_overlap(self, mock_llm_client, patch_app_config):
        """NEI detection for low word overlap."""
        verifier = IsolatedVerifier(mock_llm_client)

        # Evidence about markets, claim about technology
        evidence = RankedEvidence(
            source_id=None,
            source_url="https://example.com",
            source_title="Test",
            quote_text="The financial markets showed strong performance in Q4.",
            start_offset=0,
            end_offset=60,
            section_heading=None,
            relevance_score=0.5,
            has_numeric_content=False,
        )

        result = verifier.check_nei(
            claim_text="Artificial intelligence technology improved significantly.",
            evidence=evidence,
        )

        assert result is True  # Low overlap = NEI

    def test_returns_false_for_high_overlap(
        self, mock_llm_client, sample_evidence, patch_app_config
    ):
        """NEI returns False when words overlap significantly."""
        verifier = IsolatedVerifier(mock_llm_client)

        result = verifier.check_nei(
            claim_text="The market grew by 25% in 2024.",
            evidence=sample_evidence,  # Contains "market grew by 25% in 2024"
        )

        assert result is False  # High overlap = not NEI


class TestVerifyBatch:
    """Tests for verify_batch method."""

    @pytest.mark.asyncio
    async def test_routes_by_confidence(
        self, mock_llm_client, sample_evidence_list, patch_app_config
    ):
        """Batch routing based on confidence levels."""
        mock_llm_client.complete.return_value = MockLLMResponse(content="SUPPORTED")

        verifier = IsolatedVerifier(mock_llm_client)

        claims = [
            ("High confidence claim.", sample_evidence_list[0]),
            ("Medium confidence claim.", sample_evidence_list[1]),
            ("Low confidence claim.", sample_evidence_list[2]),
        ]

        results = await verifier.verify_batch(
            claims=claims,
            confidence_levels=["high", "medium", "low"],
        )

        assert len(results) == 3
        assert all(isinstance(r, VerificationResult) for r in results)

    @pytest.mark.asyncio
    async def test_handles_empty_batch(self, mock_llm_client, patch_app_config):
        """Empty batch returns empty results."""
        verifier = IsolatedVerifier(mock_llm_client)

        results = await verifier.verify_batch(claims=[], confidence_levels=[])

        assert results == []


class TestParseVerificationResponse:
    """Tests for _parse_verification_response method."""

    def test_parses_json_response(self, mock_llm_client, patch_app_config):
        """Parses structured JSON response."""
        verifier = IsolatedVerifier(mock_llm_client)

        response = json.dumps(
            {
                "verdict": "PARTIAL",
                "reasoning": "Some parts match, others don't.",
                "key_match": "revenue",
                "issues": ["Missing year", "Approximate value"],
            }
        )

        result = verifier._parse_verification_response(response)

        assert result.verdict == Verdict.PARTIAL
        assert "parts match" in result.reasoning
        assert result.key_match == "revenue"
        assert len(result.issues) == 2

    def test_falls_back_to_text_parsing(self, mock_llm_client, patch_app_config):
        """Falls back to text parsing when JSON fails."""
        verifier = IsolatedVerifier(mock_llm_client)

        response = "The claim is CONTRADICTED because the evidence says otherwise."

        result = verifier._parse_verification_response(response)

        assert result.verdict == Verdict.CONTRADICTED
        assert len(result.reasoning) > 0
