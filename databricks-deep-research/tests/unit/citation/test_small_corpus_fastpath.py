"""Small-corpus fast-path skip (Wave 1.5 WIN A).

When the corpus is already both few-sourced (``<= small_corpus_skip_threshold``)
and small (combined content under the module cap), the citation pipeline skips
per-source LLM span extraction and routes ALL sources through the EXISTING
corpus passthrough branch (full content kept verbatim -> numeric facts
retained). The default threshold of ``0`` keeps the fast-path OFF, so behavior
is byte-identical to before the knob existed.

Two layers are exercised:
  * ``EvidenceSelector`` honors the per-source ``_force_passthrough`` flag
    (reuses the corpus passthrough; no LLM call; provenance preserved).
  * ``CitationVerificationPipeline.preselect_evidence`` only sets that flag
    when the threshold is set AND the corpus is small.
"""
from __future__ import annotations

import asyncio
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from databricks_deep_research.citation.config import CitationConfig
from databricks_deep_research.citation.evidence_selector import (
    EvidenceSelectionConfig,
    EvidenceSelector,
    _is_corpus_source,
)
from databricks_deep_research.citation.pipeline import CitationVerificationPipeline
from databricks_deep_research.citation.types import RankedEvidence
from databricks_deep_research.llm.client import FrameworkLLMClient

# ---------------------------------------------------------------------------
# Layer 1 — EvidenceSelector honors the _force_passthrough override
# ---------------------------------------------------------------------------


class _ExplodingLLM:
    """The passthrough path must NOT call the LLM. If it does, fail loudly."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        self.calls += 1
        raise AssertionError("LLM extraction invoked for a fast-path source")


def _make_selector(threshold: int = 0) -> EvidenceSelector:
    cfg = EvidenceSelectionConfig(
        max_span_length=4000,
        max_sources=10,
        small_corpus_skip_threshold=threshold,
    )
    return EvidenceSelector(cast(FrameworkLLMClient, _ExplodingLLM()), cfg)


def test_force_passthrough_flag_recognized_by_is_corpus_source() -> None:
    assert _is_corpus_source({"_force_passthrough": True})
    # A web source is normally NOT corpus, but the flag flips it.
    assert _is_corpus_source({"source_kind": "web", "_force_passthrough": True})
    # Flag absent / falsey -> unchanged classification.
    assert not _is_corpus_source({"source_kind": "web"})
    assert not _is_corpus_source({"source_kind": "web", "_force_passthrough": False})


def test_flagged_web_source_uses_passthrough_no_llm_and_keeps_numbers() -> None:
    """A flagged web source flows through the corpus passthrough: no LLM call,
    full content kept verbatim (numbers retained), provenance unchanged."""
    selector = _make_selector()
    src = {
        "url": "https://example.com/report",
        "title": "Annual report",
        "content": (
            "Fiscal year 2024 total revenue was $42.5 billion, up 12% from "
            "the prior year; operating margin reached 18.3%."
        ),
        "source_kind": "web",
        "source_pool_index": 3,
        "_force_passthrough": True,
    }
    result = asyncio.run(
        selector.select_evidence("revenue 2024", [src], filter_quality=False)
    )
    assert len(result.evidence) == 1
    ev = result.evidence[0]
    assert "$42.5 billion" in ev.quote_text  # numeric fact retained verbatim
    assert ev.has_numeric_content is True
    assert ev.source_pool_index == 3
    # Provenance is NOT corrupted by the flag -> still "web".
    assert ev.source_kind == "web"


def test_small_corpus_skip_threshold_property_reads_config() -> None:
    assert _make_selector(threshold=0).small_corpus_skip_threshold == 0
    assert _make_selector(threshold=4).small_corpus_skip_threshold == 4


# ---------------------------------------------------------------------------
# Layer 2 — pipeline.preselect_evidence decides whether to flag
# ---------------------------------------------------------------------------


class _SpySelector:
    """Records the sources it receives so the test can assert whether the
    pipeline flagged them for passthrough. Returns a trivial span list."""

    def __init__(self, threshold: int) -> None:
        self.small_corpus_skip_threshold = threshold
        self.seen_sources: list[dict[str, Any]] = []
        self.call_count = 0

    async def select_evidence_spans(
        self,
        query: str,
        sources: list[dict[str, Any]],
        max_spans_per_source: int,
    ) -> list[RankedEvidence]:
        self.call_count += 1
        self.seen_sources = sources
        return [
            RankedEvidence(
                source_url=str(s.get("url", "")),
                quote_text=str(s.get("content", "")),
                relevance_score=0.9,
                source_pool_index=s.get("source_pool_index"),
            )
            for s in sources
        ]


def _make_pipeline(selector: _SpySelector) -> CitationVerificationPipeline:
    """Minimal pipeline: only preselect_evidence is exercised, so the other
    stage components are inert mocks."""
    return CitationVerificationPipeline(
        cast(FrameworkLLMClient, MagicMock()),
        evidence_selector=cast(Any, selector),
        claim_generator=MagicMock(),
        confidence_classifier=MagicMock(),
        isolated_verifier=MagicMock(),
        citation_corrector=MagicMock(),
        numeric_verifier=MagicMock(),
        config=CitationConfig(),
    )


_WEB_SOURCES = [
    {
        "url": "https://example.com/a",
        "title": "Doc A",
        "content": "Revenue rose to $10 million in 2024, a record high for the firm.",
        "source_type": "web",
    },
    {
        "url": "https://example.com/b",
        "title": "Doc B",
        "content": "Headcount grew by 5% while costs fell 3% over the same period.",
        "source_type": "web",
    },
]


def test_threshold_zero_is_a_noop_sources_not_flagged() -> None:
    """Default threshold 0 -> fast-path OFF -> sources reach the selector
    unflagged (byte-identical to pre-feature behavior)."""
    spy = _SpySelector(threshold=0)
    pipeline = _make_pipeline(spy)
    asyncio.run(pipeline.preselect_evidence(_WEB_SOURCES, "company performance"))
    assert spy.call_count == 1
    assert spy.seen_sources, "selector must still be called"
    assert all("_force_passthrough" not in s for s in spy.seen_sources)


def test_threshold_set_small_corpus_flags_all_sources() -> None:
    """Threshold >= source count AND small total content -> every source is
    flagged ``_force_passthrough`` before reaching the selector."""
    spy = _SpySelector(threshold=4)
    pipeline = _make_pipeline(spy)
    asyncio.run(pipeline.preselect_evidence(_WEB_SOURCES, "company performance"))
    assert spy.call_count == 1
    assert len(spy.seen_sources) == 2
    assert all(s.get("_force_passthrough") is True for s in spy.seen_sources)


def test_threshold_set_but_corpus_too_large_not_flagged() -> None:
    """Few sources but huge combined content -> fast-path does NOT trigger
    (LLM extraction still wanted for long pages)."""
    spy = _SpySelector(threshold=4)
    pipeline = _make_pipeline(spy)
    big_sources = [
        {
            "url": "https://example.com/big",
            "title": "Big doc",
            # > _SMALL_CORPUS_TOTAL_CONTENT_CHARS (8_000)
            "content": "Quarterly revenue figures and analysis. " * 400,
            "source_type": "web",
        }
    ]
    asyncio.run(pipeline.preselect_evidence(big_sources, "revenue"))
    assert spy.call_count == 1
    assert all("_force_passthrough" not in s for s in spy.seen_sources)


def test_threshold_set_too_many_sources_not_flagged() -> None:
    """More sources than the threshold -> not a 'small' corpus -> no flag."""
    spy = _SpySelector(threshold=1)
    pipeline = _make_pipeline(spy)
    asyncio.run(pipeline.preselect_evidence(_WEB_SOURCES, "company performance"))
    assert spy.call_count == 1
    assert all("_force_passthrough" not in s for s in spy.seen_sources)


def test_fastpath_does_not_mutate_caller_source_dicts() -> None:
    """The pipeline copies sources before flagging, so the caller's dicts are
    never mutated with ``_force_passthrough``."""
    spy = _SpySelector(threshold=4)
    pipeline = _make_pipeline(spy)
    caller_sources = [dict(s) for s in _WEB_SOURCES]
    asyncio.run(pipeline.preselect_evidence(caller_sources, "company performance"))
    assert all("_force_passthrough" not in s for s in caller_sources)


@pytest.mark.asyncio
async def test_threshold_set_small_corpus_skips_llm_extraction_end_to_end() -> None:
    """End-to-end through the REAL EvidenceSelector: with the threshold set and
    a small web corpus, the per-source LLM span extractor is never invoked
    (call counter == 0) yet evidence still flows and numbers are retained."""
    exploding = _ExplodingLLM()
    selector = EvidenceSelector(
        cast(FrameworkLLMClient, exploding),
        EvidenceSelectionConfig(max_sources=10, small_corpus_skip_threshold=4),
    )

    # Adapter mirrors synthesizer._EvidenceSelectorAdapter: surfaces the
    # threshold + bridges to select_evidence(filter_quality=False).
    class _Adapter:
        def __init__(self, inner: EvidenceSelector) -> None:
            self._inner = inner

        @property
        def small_corpus_skip_threshold(self) -> int:
            return self._inner.small_corpus_skip_threshold

        async def select_evidence_spans(
            self,
            query: str,
            sources: list[dict[str, Any]],
            max_spans_per_source: int,
        ) -> list[RankedEvidence]:
            res = await self._inner.select_evidence(
                query, sources, max_spans_per_source=max_spans_per_source,
                filter_quality=False,
            )
            return res.evidence

    pipeline = _make_pipeline(cast(Any, _Adapter(selector)))
    evidence = await pipeline.preselect_evidence(_WEB_SOURCES, "company performance")

    assert exploding.calls == 0  # NO per-source LLM span extraction
    assert evidence, "evidence must still flow via passthrough"
    joined = " ".join(e.quote_text for e in evidence)
    assert "$10 million" in joined  # numeric fact retained verbatim
