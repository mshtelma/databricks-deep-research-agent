"""Unit tests for numeric QA heuristic in NumericVerifier.

Tests the TOKEN OPTIMIZATION exact match heuristic.
"""

import pytest

from deep_research.services.citation.numeric_verifier import is_exact_numeric_match


class TestExactNumericMatch:
    """Tests for the exact numeric match heuristic."""

    def test_exact_match_simple_number(self) -> None:
        """Test exact match with simple number."""
        claim = "The company has 500 employees."
        evidence = "According to reports, the company has 500 employees worldwide."
        assert is_exact_numeric_match(claim, evidence) is True

    def test_exact_match_decimal(self) -> None:
        """Test exact match with decimal number."""
        claim = "Revenue was $3.2 billion."
        evidence = "The company reported revenue of 3.2 billion dollars."
        assert is_exact_numeric_match(claim, evidence) is True

    def test_exact_match_percentage(self) -> None:
        """Test exact match with percentage."""
        claim = "Growth increased by 25%."
        evidence = "The annual growth rate was 25% year over year."
        assert is_exact_numeric_match(claim, evidence) is True

    def test_exact_match_comma_formatted(self) -> None:
        """Test exact match with comma-formatted numbers."""
        claim = "Tesla sold 1,234,567 vehicles."
        evidence = "Tesla's sales reached 1234567 units globally."
        assert is_exact_numeric_match(claim, evidence) is True

    def test_exact_match_multiple_numbers(self) -> None:
        """Test exact match requires ALL numbers to match."""
        claim = "Revenue was $5 billion with 1000 employees."
        evidence = "The company generated $5 billion in revenue."
        # Missing "1000" in evidence
        assert is_exact_numeric_match(claim, evidence) is False

    def test_exact_match_multiple_numbers_all_present(self) -> None:
        """Test exact match with multiple numbers all present."""
        claim = "Revenue was $5 billion with 1000 employees."
        evidence = "The company has 1000 employees and generated $5 billion."
        assert is_exact_numeric_match(claim, evidence) is True

    def test_no_match_different_number(self) -> None:
        """Test no match when numbers differ."""
        claim = "The company has 500 employees."
        evidence = "According to reports, the company has 600 employees worldwide."
        assert is_exact_numeric_match(claim, evidence) is False

    def test_no_match_no_numbers_in_claim(self) -> None:
        """Test returns False when claim has no numbers."""
        claim = "The company is growing rapidly."
        evidence = "The company grew by 25% last year."
        assert is_exact_numeric_match(claim, evidence) is False

    def test_match_with_percentage_symbol(self) -> None:
        """Test match with percentage symbol."""
        claim = "Accuracy improved to 95%."
        evidence = "The model achieved 95% accuracy on the benchmark."
        assert is_exact_numeric_match(claim, evidence) is True

    def test_match_ignores_surrounding_context(self) -> None:
        """Test match works regardless of surrounding text."""
        claim = "OpenAI released GPT-4 with 100 trillion parameters (estimated)."
        evidence = "GPT-4 is rumored to have approximately 100 trillion parameters."
        assert is_exact_numeric_match(claim, evidence) is True

    def test_match_year_as_number(self) -> None:
        """Test that years are treated as numbers."""
        claim = "The product launched in 2024."
        evidence = "In 2024, the company launched its flagship product."
        assert is_exact_numeric_match(claim, evidence) is True
