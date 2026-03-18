"""Tests for numeric claim parsing and verification helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

from databricks_deep_research.citation.numeric_verifier import NumericVerifier


def _make_verifier() -> NumericVerifier:
    return NumericVerifier(MagicMock())


def test_parse_numeric_value_prefers_eps_over_quarter_marker() -> None:
    verifier = _make_verifier()

    parsed = verifier.parse_numeric_value(
        "The Q4 2024 adjusted EPS of $1.14 provides a baseline for comparison."
    )

    assert parsed is not None
    assert parsed.raw_text == "$1.14"
    assert float(parsed.normalized_value) == 1.14
    assert parsed.unit == "USD"


def test_extract_numeric_values_keeps_multiple_finance_metrics() -> None:
    verifier = _make_verifier()

    values = verifier.extract_numeric_values(
        "In Q1 2025, adjusted EPS reached $1.49, compared to Q2 2025's adjusted EPS of $1.04."
    )

    assert len(values) >= 2
    assert values[0].raw_text == "$1.49"
    assert values[1].raw_text == "$1.04"


def test_parse_numeric_value_prefers_percentage_over_year() -> None:
    verifier = _make_verifier()

    parsed = verifier.parse_numeric_value(
        "In Q4 2024, Kroger reported identical sales without fuel increased 2.4%."
    )

    assert parsed is not None
    assert parsed.raw_text == "2.4%"
    assert float(parsed.normalized_value) == 2.4
    assert parsed.unit == "percent"


def test_parse_numeric_value_handles_parenthesized_loss() -> None:
    verifier = _make_verifier()

    parsed = verifier.parse_numeric_value(
        "Kroger reported an operating loss of $(1,541) million in the quarter."
    )

    assert parsed is not None
    assert parsed.raw_text.strip() == "$(1,541) million"
    assert float(parsed.normalized_value) == -1541000000.0


def test_parse_numeric_value_does_not_consume_to_as_multiplier() -> None:
    verifier = _make_verifier()

    values = verifier.extract_numeric_values(
        "The company is narrowing guidance to a new range of $4.75 to $4.80."
    )

    assert len(values) >= 2
    assert values[0].raw_text == "$4.75"
    assert values[1].raw_text == "$4.80"
