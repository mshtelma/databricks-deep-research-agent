"""Unit tests for NumericVerifier service."""

import json
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from src.services.citation.evidence_selector import RankedEvidence
from src.services.citation.numeric_verifier import (
    NumericValue,
    NumericVerificationResult,
    NumericVerifier,
    QAVerificationResult,
)

from .conftest import MockLLMResponse


class TestParseNumericValue:
    """Tests for parse_numeric_value method."""

    def test_parses_currency_billions(self, mock_llm_client, patch_app_config):
        """Parses currency with billions."""
        verifier = NumericVerifier(mock_llm_client)

        result = verifier.parse_numeric_value("Revenue reached $3.2B.")

        assert result is not None
        assert result.normalized_value == Decimal("3200000000")
        assert result.unit == "USD"

    def test_parses_currency_millions(self, mock_llm_client, patch_app_config):
        """Parses currency with millions."""
        verifier = NumericVerifier(mock_llm_client)

        result = verifier.parse_numeric_value("Costs were $5.5M.")

        assert result is not None
        assert result.normalized_value == Decimal("5500000")
        assert result.unit == "USD"

    def test_parses_percentage(self, mock_llm_client, patch_app_config):
        """Parses percentage values."""
        verifier = NumericVerifier(mock_llm_client)

        result = verifier.parse_numeric_value("Growth was 25%.")

        assert result is not None
        assert result.normalized_value == Decimal("25")
        assert result.unit == "percent"

    def test_parses_decimal_percentage(self, mock_llm_client, patch_app_config):
        """Parses decimal percentage values."""
        verifier = NumericVerifier(mock_llm_client)

        result = verifier.parse_numeric_value("Increased by 3.5%.")

        assert result is not None
        assert result.normalized_value == Decimal("3.5")
        assert result.unit == "percent"

    def test_parses_comma_separated_numbers(self, mock_llm_client, patch_app_config):
        """Parses numbers with comma separators."""
        verifier = NumericVerifier(mock_llm_client)

        result = verifier.parse_numeric_value("1,234,567 users joined.")

        assert result is not None
        assert result.normalized_value == Decimal("1234567")

    def test_parses_written_billion(self, mock_llm_client, patch_app_config):
        """Parses written-out 'billion'."""
        verifier = NumericVerifier(mock_llm_client)

        result = verifier.parse_numeric_value("$5.2 billion in revenue.")

        assert result is not None
        assert result.normalized_value == Decimal("5200000000")
        assert result.unit == "USD"

    def test_parses_written_million(self, mock_llm_client, patch_app_config):
        """Parses written-out 'million'."""
        verifier = NumericVerifier(mock_llm_client)

        result = verifier.parse_numeric_value("200 million customers.")

        assert result is not None
        assert result.normalized_value == Decimal("200000000")

    def test_returns_none_for_no_number(self, mock_llm_client, patch_app_config):
        """Returns None when no number is found."""
        verifier = NumericVerifier(mock_llm_client)

        result = verifier.parse_numeric_value("The company grew significantly.")

        assert result is None

    def test_parses_euro_currency(self, mock_llm_client, patch_app_config):
        """Parses Euro currency."""
        verifier = NumericVerifier(mock_llm_client)

        result = verifier.parse_numeric_value("Revenue was €5.5B.")

        assert result is not None
        assert result.unit == "EUR"


class TestVerifyNumericClaim:
    """Tests for verify_numeric_claim async method."""

    @pytest.mark.asyncio
    async def test_returns_result(self, mock_llm_client, patch_app_config):
        """Returns NumericVerificationResult."""
        # Mock LLM response for QA
        mock_response = json.dumps([
            {
                "question": "What is the revenue figure?",
                "claim_answer": "$3.2 billion",
                "evidence_answer": "$3.2B",
            }
        ])
        mock_llm_client.complete.return_value = MockLLMResponse(content=mock_response)

        verifier = NumericVerifier(mock_llm_client)

        evidence = RankedEvidence(
            source_id=None,
            source_url="https://example.com",
            source_title="Report",
            quote_text="Company revenue reached $3.2B in Q4.",
            start_offset=0,
            end_offset=40,
            section_heading=None,
            relevance_score=0.9,
            has_numeric_content=True,
        )

        result = await verifier.verify_numeric_claim(
            claim_text="Revenue was $3.2 billion.",
            evidence=evidence,
        )

        assert isinstance(result, NumericVerificationResult)
        assert result.parsed_value is not None
        assert result.derivation_type in ["direct", "computed"]

    @pytest.mark.asyncio
    async def test_handles_no_numeric_value(self, mock_llm_client, patch_app_config):
        """Handles claims without numeric values."""
        verifier = NumericVerifier(mock_llm_client)

        evidence = RankedEvidence(
            source_id=None,
            source_url="https://example.com",
            source_title="Report",
            quote_text="The company grew significantly.",
            start_offset=0,
            end_offset=35,
            section_heading=None,
            relevance_score=0.8,
            has_numeric_content=False,
        )

        result = await verifier.verify_numeric_claim(
            claim_text="The company grew significantly.",
            evidence=evidence,
        )

        assert isinstance(result, NumericVerificationResult)
        assert result.overall_match is False
        assert result.qa_results == []

    @pytest.mark.asyncio
    async def test_detects_computed_derivation(self, mock_llm_client, patch_app_config):
        """Detects computed derivation type."""
        mock_llm_client.complete.return_value = MockLLMResponse(content="[]")

        verifier = NumericVerifier(mock_llm_client)

        evidence = RankedEvidence(
            source_id=None,
            source_url="https://example.com",
            source_title="Report",
            quote_text="Revenue was $5B and costs were $3B.",
            start_offset=0,
            end_offset=40,
            section_heading=None,
            relevance_score=0.9,
            has_numeric_content=True,
        )

        result = await verifier.verify_numeric_claim(
            claim_text="Profit was calculated as $2B.",
            evidence=evidence,
        )

        assert result.derivation_type == "computed"


class TestCompareAnswers:
    """Tests for _compare_answers method."""

    def test_exact_match(self, mock_llm_client, patch_app_config):
        """Detects exact match."""
        verifier = NumericVerifier(mock_llm_client)

        result = verifier._compare_answers("$3.2 billion", "$3.2 billion")

        assert result is True

    def test_case_insensitive_match(self, mock_llm_client, patch_app_config):
        """Matches are case insensitive."""
        verifier = NumericVerifier(mock_llm_client)

        result = verifier._compare_answers("$3.2 BILLION", "$3.2 billion")

        assert result is True

    def test_partial_match_f1(self, mock_llm_client, patch_app_config):
        """F1 scoring for partial matches."""
        verifier = NumericVerifier(mock_llm_client)

        # Very high token overlap - use similar wording
        result = verifier._compare_answers(
            "revenue was 3.2 billion in 2024",
            "revenue reached 3.2 billion in 2024",
        )

        # Should match due to high F1 overlap (same key tokens)
        assert result is True

    def test_no_match(self, mock_llm_client, patch_app_config):
        """Detects no match."""
        verifier = NumericVerifier(mock_llm_client)

        result = verifier._compare_answers(
            "revenue was $3.2 billion",
            "costs reached $5.5 million",
        )

        # Low overlap - should not match
        assert result is False

    def test_empty_answers(self, mock_llm_client, patch_app_config):
        """Handles empty answers."""
        verifier = NumericVerifier(mock_llm_client)

        result = verifier._compare_answers("", "something")

        assert result is False


class TestValuesMatch:
    """Tests for _values_match method."""

    def test_exact_match(self, mock_llm_client, patch_app_config):
        """Exact values match."""
        verifier = NumericVerifier(mock_llm_client)

        result = verifier._values_match(Decimal("3200000000"), Decimal("3200000000"))

        assert result is True

    def test_within_tolerance(self, mock_llm_client, patch_app_config):
        """Values within tolerance match."""
        verifier = NumericVerifier(mock_llm_client)

        # 1% difference should be within 5% tolerance
        result = verifier._values_match(Decimal("100"), Decimal("101"))

        assert result is True

    def test_outside_tolerance(self, mock_llm_client, patch_app_config):
        """Values outside tolerance don't match."""
        verifier = NumericVerifier(mock_llm_client)

        # 10% difference should exceed 5% tolerance
        result = verifier._values_match(Decimal("100"), Decimal("110"))

        assert result is False

    def test_zero_values(self, mock_llm_client, patch_app_config):
        """Zero values match each other."""
        verifier = NumericVerifier(mock_llm_client)

        result = verifier._values_match(Decimal("0"), Decimal("0"))

        assert result is True


class TestDetectNumericClaims:
    """Tests for detect_numeric_claims method."""

    def test_detects_currency_claims(self, mock_llm_client, patch_app_config):
        """Detects sentences with currency values."""
        verifier = NumericVerifier(mock_llm_client)

        text = "The company reported $3.2B revenue. Markets were volatile. Profits reached $500M."

        claims = verifier.detect_numeric_claims(text)

        assert len(claims) >= 2
        assert any("$3.2B" in c for c in claims)
        assert any("$500M" in c for c in claims)

    def test_detects_percentage_claims(self, mock_llm_client, patch_app_config):
        """Detects sentences with percentages."""
        verifier = NumericVerifier(mock_llm_client)

        text = "Growth was 25%. The weather was nice. User retention improved to 85%."

        claims = verifier.detect_numeric_claims(text)

        assert len(claims) >= 2
        assert any("25%" in c for c in claims)
        assert any("85%" in c for c in claims)

    def test_returns_empty_for_no_numbers(self, mock_llm_client, patch_app_config):
        """Returns empty list when no numbers."""
        verifier = NumericVerifier(mock_llm_client)

        text = "The company grew. Markets expanded. Users loved the product."

        claims = verifier.detect_numeric_claims(text)

        assert claims == []


class TestRunQAVerification:
    """Tests for _run_qa_verification method.

    Note: _parse_qa_response was replaced by Pydantic structured output.
    These tests verify the higher-level QA verification flow.
    """

    @pytest.mark.asyncio
    async def test_returns_empty_on_no_structured_output(
        self, mock_llm_client, patch_app_config
    ):
        """Returns empty results when LLM provides no structured output."""
        mock_llm_client.complete.return_value = MockLLMResponse(
            content="not structured",
            structured=None,
        )

        verifier = NumericVerifier(mock_llm_client)

        evidence = RankedEvidence(
            source_id=None,
            source_url="https://example.com",
            source_title="Report",
            quote_text="Revenue was $3.2B.",
            start_offset=0,
            end_offset=20,
            section_heading=None,
            relevance_score=0.9,
            has_numeric_content=True,
        )

        parsed = verifier.parse_numeric_value("$3.2 billion")
        result = await verifier._run_qa_verification(
            claim_text="Revenue was $3.2 billion.",
            evidence=evidence,
            parsed=parsed,
        )

        # Empty results when no structured output
        assert result == []

    @pytest.mark.asyncio
    async def test_handles_llm_exception(self, mock_llm_client, patch_app_config):
        """Handles LLM exceptions gracefully."""
        mock_llm_client.complete.side_effect = Exception("LLM error")

        verifier = NumericVerifier(mock_llm_client)

        evidence = RankedEvidence(
            source_id=None,
            source_url="https://example.com",
            source_title="Report",
            quote_text="Revenue was $3.2B.",
            start_offset=0,
            end_offset=20,
            section_heading=None,
            relevance_score=0.9,
            has_numeric_content=True,
        )

        parsed = verifier.parse_numeric_value("$3.2 billion")
        result = await verifier._run_qa_verification(
            claim_text="Revenue was $3.2 billion.",
            evidence=evidence,
            parsed=parsed,
        )

        # Returns empty on exception
        assert result == []


class TestDetectUnit:
    """Tests for _detect_unit method."""

    def test_detects_usd(self, mock_llm_client, patch_app_config):
        """Detects USD from $."""
        verifier = NumericVerifier(mock_llm_client)

        unit = verifier._detect_unit("$5.2B in revenue")

        assert unit == "USD"

    def test_detects_eur(self, mock_llm_client, patch_app_config):
        """Detects EUR from €."""
        verifier = NumericVerifier(mock_llm_client)

        unit = verifier._detect_unit("€500M invested")

        assert unit == "EUR"

    def test_detects_percent(self, mock_llm_client, patch_app_config):
        """Detects percent."""
        verifier = NumericVerifier(mock_llm_client)

        unit = verifier._detect_unit("Growth was 25%")

        assert unit == "percent"

    def test_detects_years(self, mock_llm_client, patch_app_config):
        """Detects years unit."""
        verifier = NumericVerifier(mock_llm_client)

        unit = verifier._detect_unit("Founded 10 years ago")

        assert unit == "years"

    def test_returns_none_for_unknown(self, mock_llm_client, patch_app_config):
        """Returns None for unknown units."""
        verifier = NumericVerifier(mock_llm_client)

        unit = verifier._detect_unit("Some random text")

        assert unit is None
