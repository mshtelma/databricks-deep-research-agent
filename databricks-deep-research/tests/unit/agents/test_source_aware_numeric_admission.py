"""Tests for numeric-content admission signal in source_aware (feature 4.1).

Covers:
  * ``_score_source_relevance`` adds ``NUMERIC_ADMISSION_BOOST`` for numeric
    text and stamps ``source["has_numeric"]``, but does NOT boost prose.
  * The relaxed web/default acceptance gate in ``_should_accept_source`` admits a
    borderline numeric source when the flag is on, and is byte-identical (same
    boolean) when off (the default).
"""

from __future__ import annotations

from unittest.mock import patch

from databricks_deep_research.agents import source_aware
from databricks_deep_research.agents.source_aware import (
    NUMERIC_ADMISSION_BOOST,
    _score_source_relevance,
    _should_accept_source,
)
from databricks_deep_research.tools.protocol import (
    SourceKind,
    ToolDefinition,
    ToolResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _profile(terms: list[str] | None = None, phrases: list[str] | None = None) -> dict:
    """Minimal step profile understood by ``_score_source_relevance``."""
    return {"terms": terms or [], "phrases": phrases or []}


def _web_definition() -> ToolDefinition:
    return ToolDefinition(
        name="web_search",
        description="Search the web.",
        parameters={"type": "object", "properties": {}},
        source_type="web",
        source_kind="web",
    )


def _ok_result(content: str = "real content from the web") -> ToolResult:
    return ToolResult(success=True, content=content)


# ---------------------------------------------------------------------------
# _score_source_relevance — numeric boost
# ---------------------------------------------------------------------------


def test_numeric_text_gets_boost_and_stamp() -> None:
    """A source whose text carries figures gets +NUMERIC_ADMISSION_BOOST and is
    stamped ``has_numeric=True``."""
    source = {
        "title": "Quarterly results",
        "snippet": "Revenue rose to $4.2 billion, up 12% year over year.",
        "content": "",
        "url": "https://example.com/a",
    }
    score, reason = _score_source_relevance(source, _profile())
    assert source["has_numeric"] is True
    assert score == NUMERIC_ADMISSION_BOOST  # no keyword overlap -> boost only
    assert "numeric_boost" in reason


def test_prose_without_numbers_is_not_boosted_or_stamped() -> None:
    """Pure prose (no figures) gets neither the boost nor the stamp."""
    source = {
        "title": "Company overview",
        "snippet": "The firm operates across many regions and serves customers.",
        "content": "",
        "url": "https://example.com/b",
    }
    score, reason = _score_source_relevance(source, _profile())
    assert "has_numeric" not in source
    assert score == 0
    assert "numeric_boost" not in reason


def test_numeric_boost_is_additive_to_keyword_score() -> None:
    """The numeric boost stacks on top of keyword/phrase overlap."""
    source = {
        "title": "Acme revenue report",
        "snippet": "Acme revenue grew to $3.1 billion.",
        "content": "",
        "url": "https://example.com/c",
    }
    # 'revenue' is both a profile term and a domain term -> base 1 (term) + 1
    # (domain-term bonus) = 2, plus numeric boost.
    score, _ = _score_source_relevance(source, _profile(terms=["revenue"]))
    assert source["has_numeric"] is True
    assert score == 2 + NUMERIC_ADMISSION_BOOST


# ---------------------------------------------------------------------------
# _should_accept_source — relaxed gate (flag-gated)
# ---------------------------------------------------------------------------


def test_default_gate_rejects_borderline_numeric_source() -> None:
    """DEFAULT behaviour (flag off): a numeric source at score==1 is rejected by
    the historical ``score >= 2`` gate — byte-identical to pre-feature."""
    source = {"source_kind": "web", "has_numeric": True, "url": "https://example.com/d"}
    # Flag default is False; assert that explicitly to lock the default.
    assert source_aware._RELAX_GATE_FOR_NUMERIC is False
    accepted = _should_accept_source(_web_definition(), _ok_result(), source, score=1)
    assert accepted is False


def test_relaxed_gate_admits_borderline_numeric_source_when_on() -> None:
    """Flag on: a numeric source at score==1 is admitted."""
    source = {"source_kind": "web", "has_numeric": True, "url": "https://example.com/e"}
    with patch.object(source_aware, "_RELAX_GATE_FOR_NUMERIC", True):
        accepted = _should_accept_source(_web_definition(), _ok_result(), source, score=1)
    assert accepted is True


def test_relaxed_gate_still_rejects_non_numeric_below_threshold() -> None:
    """Flag on but source is NOT numeric: still rejected at score==1."""
    source = {"source_kind": "web", "url": "https://example.com/f"}
    with patch.object(source_aware, "_RELAX_GATE_FOR_NUMERIC", True):
        accepted = _should_accept_source(_web_definition(), _ok_result(), source, score=1)
    assert accepted is False


def test_gate_accepts_score_two_regardless_of_flag() -> None:
    """A score>=2 web source is accepted both with the flag off and on."""
    source = {"source_kind": "web", "url": "https://example.com/g"}
    off = _should_accept_source(_web_definition(), _ok_result(), source, score=2)
    with patch.object(source_aware, "_RELAX_GATE_FOR_NUMERIC", True):
        on = _should_accept_source(_web_definition(), _ok_result(), source, score=2)
    assert off is True
    assert on is True


def test_gate_byte_identical_off_for_full_score_range() -> None:
    """For non-numeric web sources the relaxed code path collapses to the exact
    same decisions as the historical gate across a representative score range."""
    for score in range(0, 5):
        source = {"source_kind": "web", "url": f"https://example.com/h{score}"}
        result = _should_accept_source(_web_definition(), _ok_result(), source, score)
        assert result is (score >= 2)


def test_vector_search_kind_unaffected_by_numeric_changes() -> None:
    """VS sources keep their own fallback gate (unchanged path)."""
    source = {
        "source_kind": SourceKind.vector_index,
        "relevance_score": 0.4,
        "url": "vs://index/1",
    }
    # score below 2 but relevance >= fallback threshold -> accepted via VS branch
    accepted = _should_accept_source(_web_definition(), _ok_result(), source, score=0)
    assert accepted is True
