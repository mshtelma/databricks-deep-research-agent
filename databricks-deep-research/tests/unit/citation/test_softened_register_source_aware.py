"""Defect E / Part 2: corpus-grounded claims softened in Stage 8 must use the
neutral [unverified] marker, not a multi-source hedge."""
from __future__ import annotations

from databricks_deep_research.citation.pipeline import _build_softened_fact_text

_HEDGES = ("Reportedly", "Some sources", "It has been", "According to")


def test_corpus_grounded_uses_unverified_not_hedge() -> None:
    out = _build_softened_fact_text("Outlays were $543 million.", None, corpus_grounded=True)
    assert out == "Outlays were $543 million. [unverified]"
    assert not out.startswith(_HEDGES)


def test_web_claim_keeps_multisource_hedge() -> None:
    out = _build_softened_fact_text("Outlays were $543 million.", None, corpus_grounded=False)
    assert out.startswith(_HEDGES)


def test_default_kwarg_is_backcompat() -> None:
    # Existing callers omit corpus_grounded -> behavior unchanged.
    out = _build_softened_fact_text("Outlays were $543 million.", None)
    assert out.startswith(_HEDGES)
