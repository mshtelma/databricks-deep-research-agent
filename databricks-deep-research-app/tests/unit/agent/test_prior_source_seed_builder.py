"""Phase 2b-4: the app builds a bounded, canonicalized, citable, query-ranked
seed payload (``prior_sources_for_seed``) distinct from ``existing_sources``
(durable history). Folds in the 2b-0 gate requirement: every seed record carries
``evidence_quality="cached"`` so it survives the synthesizer's substantive
filter and becomes citable.
"""

from __future__ import annotations

import pytest

from deep_research.agent.framework_orchestrator import _build_prior_source_seed

pytestmark = pytest.mark.unit


def test_seed_canonicalizes_dedups_filters_and_adapts() -> None:
    sources = [
        {"url": "https://x.com/acme?utm_source=g", "title": "Acme revenue", "snippet": "rev up"},
        {"url": "https://x.com/acme", "title": "Acme revenue", "snippet": "rev up"},  # dup canonical
        {"url": "https://x.com/bare", "title": "Acme"},                                # no evidence -> drop
        {"url": "https://x.com/off", "title": "weather", "snippet": "rain"},           # 0 overlap -> drop
    ]
    seed = _build_prior_source_seed(sources, query="acme revenue", top_k=20)
    assert [s["url"] for s in seed] == ["https://x.com/acme"]  # canonical, deduped, overlap-only
    # 2b-0 gate requirement: every seeded record must be citable (substantive).
    assert all(s["evidence_quality"] == "cached" for s in seed)


def test_seed_is_bounded_to_top_k() -> None:
    sources = [
        {"url": f"https://x.com/{i}", "title": f"acme {i}", "snippet": "acme report"}
        for i in range(30)
    ]
    seed = _build_prior_source_seed(sources, query="acme", top_k=20)
    assert len(seed) == 20


def test_seed_falls_back_to_all_citable_when_no_overlap() -> None:
    # No record overlaps the query lexically → seed all citable (the pool's
    # hybrid index + synthesizer admission refine downstream; better than
    # starving the follow-up). Still bounded + citable.
    sources = [
        {"url": "https://x.com/1", "snippet": "alpha"},
        {"url": "https://x.com/2", "snippet": "beta"},
    ]
    seed = _build_prior_source_seed(sources, query="zzz nomatch here", top_k=20)
    assert len(seed) == 2
    assert all(s["evidence_quality"] == "cached" for s in seed)


def test_seed_empty_input_is_empty() -> None:
    assert _build_prior_source_seed([], query="anything", top_k=20) == []
