"""Unit tests for ``SourceReputationScorer`` and ``_score_source_relevance``
reputation integration.

These tests pin the soft-ranking behaviour without making any assumption
about which specific domains are good or bad — the scorer is data-driven
end-to-end, and the tests treat the domain values as opaque illustrations.
"""
from __future__ import annotations

import pytest

from databricks_deep_research.agents.source_aware import _score_source_relevance
from databricks_deep_research.agents.source_reputation import (
    DEPRECATED_DELTA,
    IR_SUBDOMAIN_DELTA,
    PREFERRED_DELTA,
    ReputationAdjustment,
    SourceReputationScorer,
)


# ---------------------------------------------------------------------------
# SourceReputationScorer — direct tests
# ---------------------------------------------------------------------------


def test_scorer_zero_when_both_lists_empty_and_no_ir_boost() -> None:
    """All-empty config + IR boost disabled → scorer is inactive."""
    scorer = SourceReputationScorer(
        preferred_patterns=[],
        deprecated_patterns=[],
        ir_subdomain_boost=False,
    )
    assert scorer.is_active is False
    adj = scorer.score("https://example.com/page")
    assert adj == ReputationAdjustment(0, None, "")


def test_scorer_active_when_ir_boost_enabled_even_with_empty_lists() -> None:
    """Default ``ir_subdomain_boost=True`` keeps the scorer live so it can
    still boost IR pages without any explicit pattern config."""
    scorer = SourceReputationScorer(preferred_patterns=[], deprecated_patterns=[])
    assert scorer.is_active is True


def test_scorer_preferred_match_returns_positive_delta() -> None:
    """A preferred suffix-wildcard match yields ``+PREFERRED_DELTA``."""
    scorer = SourceReputationScorer(
        preferred_patterns=["*.gov"],
        deprecated_patterns=[],
        ir_subdomain_boost=False,
    )
    adj = scorer.score("https://www.cdc.gov/some/page")
    assert adj.delta == PREFERRED_DELTA
    assert adj.matched_pattern == "preferred:*.gov"
    assert "*.gov" in adj.reason


def test_scorer_deprecated_match_returns_negative_delta() -> None:
    """A deprecated exact-match yields ``-|DEPRECATED_DELTA|``."""
    scorer = SourceReputationScorer(
        preferred_patterns=[],
        deprecated_patterns=["example-spam.com"],
        ir_subdomain_boost=False,
    )
    adj = scorer.score("https://example-spam.com/x")
    assert adj.delta == DEPRECATED_DELTA
    assert adj.matched_pattern == "deprecated:example-spam.com"


def test_scorer_url_matching_both_lists_sums_deltas() -> None:
    """A pattern in BOTH lists produces a net delta of the sum.

    With the default magnitudes (+2 / -2), this nets to 0 — the scorer
    effectively no-ops the source. The reason field surfaces both matches
    so an operator can spot the contradictory config.
    """
    scorer = SourceReputationScorer(
        preferred_patterns=["conflict.example"],
        deprecated_patterns=["conflict.example"],
        ir_subdomain_boost=False,
    )
    adj = scorer.score("https://conflict.example/x")
    assert adj.delta == PREFERRED_DELTA + DEPRECATED_DELTA
    assert "preferred:conflict.example" in adj.reason
    assert "deprecated:conflict.example" in adj.reason


def test_scorer_ir_subdomain_boost_applies() -> None:
    """``investors.<any>`` and ``ir.<any>`` hosts get ``+IR_SUBDOMAIN_DELTA``."""
    scorer = SourceReputationScorer(
        preferred_patterns=[],
        deprecated_patterns=[],
        ir_subdomain_boost=True,
    )
    adj = scorer.score("https://investors.anything.example/q")
    assert adj.delta == IR_SUBDOMAIN_DELTA
    assert "ir_subdomain:" in (adj.reason or "")


def test_scorer_ir_subdomain_disabled_does_not_boost() -> None:
    """``ir_subdomain_boost=False`` disables the bonus completely."""
    scorer = SourceReputationScorer(
        preferred_patterns=[],
        deprecated_patterns=[],
        ir_subdomain_boost=False,
    )
    adj = scorer.score("https://investors.anything.example/q")
    assert adj.delta == 0


def test_scorer_strips_www_prefix_before_matching() -> None:
    """``www.`` host prefix is stripped so patterns match the canonical host."""
    scorer = SourceReputationScorer(
        preferred_patterns=["news.example"],
        deprecated_patterns=[],
        ir_subdomain_boost=False,
    )
    adj = scorer.score("https://www.news.example/article")
    # 'www.news.example' → 'news.example' should match the exact-match pattern
    assert adj.delta == PREFERRED_DELTA


def test_scorer_empty_url_returns_zero() -> None:
    scorer = SourceReputationScorer(
        preferred_patterns=["*.gov"], deprecated_patterns=[]
    )
    assert scorer.score("").delta == 0


def test_scorer_malformed_url_returns_zero_safely() -> None:
    """Garbage input does not raise."""
    scorer = SourceReputationScorer(
        preferred_patterns=["*.gov"], deprecated_patterns=[]
    )
    adj = scorer.score("not a url at all")
    assert adj.delta == 0


def test_scorer_prefix_wildcard_news_dot_star() -> None:
    """``news.*`` matches ``news.com`` / ``news.org`` but not ``other.com``."""
    scorer = SourceReputationScorer(
        preferred_patterns=["news.*"],
        deprecated_patterns=[],
        ir_subdomain_boost=False,
    )
    assert scorer.score("https://news.example/x").delta == PREFERRED_DELTA
    assert scorer.score("https://news.org/x").delta == PREFERRED_DELTA
    assert scorer.score("https://other.com/x").delta == 0


def test_scorer_normalises_patterns_lower_strip() -> None:
    """Patterns with whitespace / mixed case still match."""
    scorer = SourceReputationScorer(
        preferred_patterns=["  *.GOV  "],
        deprecated_patterns=[],
        ir_subdomain_boost=False,
    )
    assert scorer.score("https://www.example.gov/x").delta == PREFERRED_DELTA


# ---------------------------------------------------------------------------
# Integration with _score_source_relevance
# ---------------------------------------------------------------------------


def _make_profile(terms: list[str], phrases: list[str] | None = None) -> dict:
    """Minimal profile shape understood by ``_score_source_relevance``."""
    return {"terms": terms, "phrases": phrases or []}


def test_score_source_relevance_without_scorer_unchanged() -> None:
    """Without a scorer the function behaves exactly as before PR 3."""
    src = {
        "title": "Example",
        "snippet": "snowflake snow",
        "content": "",
        "url": "https://anything.example",
    }
    score, reason = _score_source_relevance(src, _make_profile(["snowflake", "snow"]))
    # Two terms matched → score ≥ 2 → admission would pass.
    assert score >= 2
    assert "reputation=" not in reason  # no reputation line when scorer absent


def test_score_source_relevance_with_preferred_url_boosts() -> None:
    """A URL on the preferred list gets a positive reputation delta added to score."""
    src = {
        "title": "Some primary source",
        "snippet": "snowflake",
        "content": "",
        "url": "https://docs.example.gov/page",
    }
    profile = _make_profile(["snowflake"])
    scorer = SourceReputationScorer(
        preferred_patterns=["*.gov"],
        deprecated_patterns=[],
        ir_subdomain_boost=False,
    )
    score_with, reason_with = _score_source_relevance(src, profile, reputation_scorer=scorer)
    score_without, _ = _score_source_relevance(src, profile, reputation_scorer=None)
    assert score_with - score_without == PREFERRED_DELTA
    assert "reputation=+" in reason_with


def test_score_source_relevance_with_deprecated_url_penalises() -> None:
    """A URL on the deprecated list gets a negative reputation delta."""
    src = {
        "title": "Stub",
        "snippet": "snowflake",
        "content": "",
        "url": "https://example-content-farm.com/page",
    }
    profile = _make_profile(["snowflake"])
    scorer = SourceReputationScorer(
        preferred_patterns=[],
        deprecated_patterns=["example-content-farm.com"],
        ir_subdomain_boost=False,
    )
    score_with, reason_with = _score_source_relevance(src, profile, reputation_scorer=scorer)
    score_without, _ = _score_source_relevance(src, profile, reputation_scorer=None)
    assert score_with - score_without == DEPRECATED_DELTA
    assert "reputation=-" in reason_with


def test_score_source_relevance_with_empty_url_unchanged() -> None:
    """A source with no URL gets no reputation adjustment (scorer skips)."""
    src = {
        "title": "noURL",
        "snippet": "snowflake",
        "content": "",
        "url": "",
    }
    profile = _make_profile(["snowflake"])
    scorer = SourceReputationScorer(
        preferred_patterns=["*.gov"],
        deprecated_patterns=[],
        ir_subdomain_boost=False,
    )
    score_with, _ = _score_source_relevance(src, profile, reputation_scorer=scorer)
    score_without, _ = _score_source_relevance(src, profile, reputation_scorer=None)
    assert score_with == score_without
