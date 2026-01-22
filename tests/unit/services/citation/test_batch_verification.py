"""Unit tests for batch verification in IsolatedVerifier.

Tests the TOKEN OPTIMIZATION batch verification functionality.
"""

import pytest

from deep_research.services.citation.isolated_verifier import (
    BatchVerificationItem,
    BatchVerificationOutput,
    IsolatedVerifier,
    Verdict,
    VerificationResult,
)


class TestClaimFingerprinting:
    """Tests for claim fingerprinting used in caching."""

    def test_fingerprint_basic(self) -> None:
        """Test basic fingerprinting produces consistent hashes."""
        claim = "Tesla sold 500,000 vehicles in Q3 2024."
        fp1 = IsolatedVerifier.fingerprint_claim(claim)
        fp2 = IsolatedVerifier.fingerprint_claim(claim)
        assert fp1 == fp2
        assert len(fp1) == 16  # 16-char MD5 hash

    def test_fingerprint_case_insensitive(self) -> None:
        """Test fingerprinting is case-insensitive."""
        fp1 = IsolatedVerifier.fingerprint_claim("Tesla sold cars")
        fp2 = IsolatedVerifier.fingerprint_claim("TESLA SOLD CARS")
        assert fp1 == fp2

    def test_fingerprint_punctuation_insensitive(self) -> None:
        """Test fingerprinting ignores punctuation."""
        fp1 = IsolatedVerifier.fingerprint_claim("Tesla sold cars!")
        fp2 = IsolatedVerifier.fingerprint_claim("Tesla sold cars")
        assert fp1 == fp2

    def test_fingerprint_word_order_insensitive(self) -> None:
        """Test fingerprinting sorts words (near-duplicate detection)."""
        fp1 = IsolatedVerifier.fingerprint_claim("Tesla sold cars")
        fp2 = IsolatedVerifier.fingerprint_claim("cars sold Tesla")
        assert fp1 == fp2

    def test_fingerprint_different_claims_different_hashes(self) -> None:
        """Test different claims produce different hashes."""
        fp1 = IsolatedVerifier.fingerprint_claim("Tesla sold cars")
        fp2 = IsolatedVerifier.fingerprint_claim("Ford sold trucks")
        assert fp1 != fp2


class TestBatchVerificationModels:
    """Tests for batch verification Pydantic models."""

    def test_batch_verification_item(self) -> None:
        """Test BatchVerificationItem model."""
        item = BatchVerificationItem(
            claim_index=0,
            verdict="SUPPORTED",
            reasoning="Evidence matches claim",
            key_match="Tesla sold 500,000",
        )
        assert item.claim_index == 0
        assert item.verdict == "SUPPORTED"

    def test_batch_verification_output(self) -> None:
        """Test BatchVerificationOutput model."""
        output = BatchVerificationOutput(
            results=[
                BatchVerificationItem(
                    claim_index=0, verdict="SUPPORTED", reasoning="Match"
                ),
                BatchVerificationItem(
                    claim_index=1, verdict="PARTIAL", reasoning="Partial match"
                ),
            ]
        )
        assert len(output.results) == 2
        assert output.results[0].verdict == "SUPPORTED"
        assert output.results[1].verdict == "PARTIAL"


class TestBatchResultParsing:
    """Tests for parsing batch verification results."""

    @pytest.fixture
    def verifier(self) -> IsolatedVerifier:
        """Create a verifier instance with mock LLM."""
        # Create mock LLM client
        class MockLLM:
            pass

        return IsolatedVerifier(llm_client=MockLLM())  # type: ignore

    def test_parse_batch_results_ordered(self, verifier: IsolatedVerifier) -> None:
        """Test parsing batch results maintains order."""
        output = BatchVerificationOutput(
            results=[
                BatchVerificationItem(
                    claim_index=0, verdict="SUPPORTED", reasoning="Match"
                ),
                BatchVerificationItem(
                    claim_index=1, verdict="UNSUPPORTED", reasoning="No match"
                ),
                BatchVerificationItem(
                    claim_index=2, verdict="PARTIAL", reasoning="Partial"
                ),
            ]
        )
        results = verifier._parse_batch_results(output, expected_count=3)

        assert len(results) == 3
        assert results[0].verdict == Verdict.SUPPORTED
        assert results[1].verdict == Verdict.UNSUPPORTED
        assert results[2].verdict == Verdict.PARTIAL

    def test_parse_batch_results_reordered(self, verifier: IsolatedVerifier) -> None:
        """Test parsing handles LLM reordering results."""
        # LLM returns results out of order
        output = BatchVerificationOutput(
            results=[
                BatchVerificationItem(
                    claim_index=2, verdict="PARTIAL", reasoning="Partial"
                ),
                BatchVerificationItem(
                    claim_index=0, verdict="SUPPORTED", reasoning="Match"
                ),
                BatchVerificationItem(
                    claim_index=1, verdict="UNSUPPORTED", reasoning="No match"
                ),
            ]
        )
        results = verifier._parse_batch_results(output, expected_count=3)

        assert len(results) == 3
        assert results[0].verdict == Verdict.SUPPORTED
        assert results[1].verdict == Verdict.UNSUPPORTED
        assert results[2].verdict == Verdict.PARTIAL

    def test_parse_batch_results_missing_indices(
        self, verifier: IsolatedVerifier
    ) -> None:
        """Test parsing fills missing indices with abstained."""
        output = BatchVerificationOutput(
            results=[
                BatchVerificationItem(
                    claim_index=0, verdict="SUPPORTED", reasoning="Match"
                ),
                # Missing index 1
                BatchVerificationItem(
                    claim_index=2, verdict="PARTIAL", reasoning="Partial"
                ),
            ]
        )
        results = verifier._parse_batch_results(output, expected_count=3)

        assert len(results) == 3
        assert results[0].verdict == Verdict.SUPPORTED
        assert results[1].abstained is True  # Missing -> abstained
        assert results[2].verdict == Verdict.PARTIAL

    def test_parse_batch_content_fallback(self, verifier: IsolatedVerifier) -> None:
        """Test fallback parsing from raw JSON content."""
        content = """Here are the verification results:
        ```json
        {
            "results": [
                {"claim_index": 0, "verdict": "SUPPORTED", "reasoning": "Match"},
                {"claim_index": 1, "verdict": "PARTIAL", "reasoning": "Partial"}
            ]
        }
        ```
        """
        results = verifier._parse_batch_response_content(content, expected_count=2)

        assert len(results) == 2
        assert results[0].verdict == Verdict.SUPPORTED
        assert results[1].verdict == Verdict.PARTIAL
