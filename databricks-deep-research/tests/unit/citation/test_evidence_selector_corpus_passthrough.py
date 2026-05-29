"""Plan v2.3 — corpus-grounded sources bypass LLM span extraction.

When ``source_kind`` is a corpus-grounded value (``vector_index``,
``sql_analytics``, ``qa_assistant``, ``file``) the evidence selector
keeps the source content verbatim as the ``RankedEvidence.quote_text``.
Web sources still go through the LLM extraction path. This fixes the
mode where the synthesizer cited specific numbers from vector_search
chunks but the verifier saw only metadata, leading to 100% unsupported
verdicts and an empty final report.
"""
from __future__ import annotations

import asyncio
from typing import Any, cast

from databricks_deep_research.citation.evidence_selector import (
    EvidenceSelectionConfig,
    EvidenceSelector,
    _is_corpus_source,
)
from databricks_deep_research.llm.client import FrameworkLLMClient


class _DummyLLM:
    """Stub stand-in — the corpus path must NOT call this. If it does the
    assertion in the test fails."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        self.calls += 1
        raise AssertionError(
            "LLM extraction was invoked for a corpus-grounded source"
        )


def _make_selector() -> EvidenceSelector:
    cfg = EvidenceSelectionConfig(
        max_spans_per_source=10,
        min_span_length=50,
        max_span_length=4000,
        relevance_threshold=0.3,
        numeric_content_boost=0.2,
        max_sources=10,
    )
    return EvidenceSelector(cast(FrameworkLLMClient, _DummyLLM()), cfg)


def test_is_corpus_source_recognizes_kinds() -> None:
    assert _is_corpus_source({"source_kind": "vector_index"})
    assert _is_corpus_source({"source_kind": "sql_analytics"})
    assert _is_corpus_source({"source_kind": "qa_assistant"})
    assert _is_corpus_source({"source_kind": "file"})


def test_is_corpus_source_recognizes_legacy_source_types() -> None:
    assert _is_corpus_source({"source_type": "vector_search"})
    assert _is_corpus_source({"source_type": "genie"})
    assert _is_corpus_source({"source_type": "knowledge_assistant"})


def test_is_corpus_source_rejects_web() -> None:
    assert not _is_corpus_source({"source_kind": "web"})
    assert not _is_corpus_source({"source_type": "web"})
    assert not _is_corpus_source({})  # missing → not corpus


def test_corpus_source_keeps_full_content_verbatim() -> None:
    """The treasury_chunks_vs_index regression: the cited $-amount must
    end up in the resulting RankedEvidence.quote_text. Previously the
    LLM-extracted spans grabbed table headers and dropped the numbers."""
    selector = _make_selector()
    src = {
        "url": "vs://main.officeqa_benchmark.treasury_chunks_vs_index/row42",
        "title": "treasury_bulletin_1945_10.txt",
        "content": (
            "Treasury bulletin 1945-10 | Category: war_activities | "
            "Fiscal year 1945 total expenditures: $90.5 billion. "
            "Munitions: $58.5 billion. Pay and subsistence: $21.6 billion."
        ),
        "source_kind": "vector_index",
        "source_pool_index": 0,
    }
    result = asyncio.run(
        selector.select_evidence("US war expenditures 1945", [src], filter_quality=False)
    )
    assert len(result.evidence) == 1
    quote = result.evidence[0].quote_text
    assert "$90.5 billion" in quote
    assert "$58.5 billion" in quote
    assert result.evidence[0].has_numeric_content is True


def test_corpus_source_caps_at_max_span_length() -> None:
    """A pathologically large corpus chunk gets capped at
    ``max_span_length`` so we don't blow the token budget."""
    selector = _make_selector()
    huge = "Treasury data row. " * 1000  # ~20 KB
    src = {
        "url": "vs://x/y",
        "content": huge,
        "source_kind": "vector_index",
        "source_pool_index": 0,
    }
    result = asyncio.run(
        selector.select_evidence("treasury", [src], filter_quality=False)
    )
    assert len(result.evidence) == 1
    assert len(result.evidence[0].quote_text) <= 4000


def test_corpus_source_too_short_returns_no_evidence() -> None:
    """Below min_span_length the corpus chunk is treated as noise (no
    RankedEvidence emitted) — same shape as the snippet-only branch above."""
    selector = _make_selector()
    src = {
        "url": "vs://x/y",
        "content": "tiny",
        "source_kind": "vector_index",
    }
    result = asyncio.run(
        selector.select_evidence("treasury", [src], filter_quality=False)
    )
    assert result.evidence == []


def test_corpus_source_preserves_source_pool_index() -> None:
    """source_pool_index must round-trip through the corpus path so the
    citation pipeline can later resolve the evidence back to its source."""
    selector = _make_selector()
    src = {
        "url": "vs://x/y",
        "content": "Treasury bulletin 1945-10 reported $90.5 billion in war expenditures across all federal agencies during fiscal year 1945.",
        "source_kind": "vector_index",
        "source_pool_index": 7,
    }
    result = asyncio.run(
        selector.select_evidence("treasury", [src], filter_quality=False)
    )
    assert result.evidence[0].source_pool_index == 7


def test_corpus_source_populates_source_kind() -> None:
    """source_kind must round-trip onto RankedEvidence so Stage 8 can pick a
    source-appropriate softening register (corpus -> neutral [unverified])."""
    from databricks_deep_research.citation.types import is_corpus_source_value

    selector = _make_selector()
    src = {
        "url": "vs://x/y",
        "content": "Treasury bulletin 1945-10 reported $90.5 billion in war expenditures across all federal agencies during fiscal year 1945.",
        "source_kind": "vector_index",
        "source_pool_index": 0,
    }
    result = asyncio.run(
        selector.select_evidence("treasury", [src], filter_quality=False)
    )
    assert result.evidence[0].source_kind == "vector_index"
    assert is_corpus_source_value(result.evidence[0].source_kind) is True


def test_corpus_kind_set_is_generic_across_kinds() -> None:
    """Every corpus-grounded kind in the framework's SourceKind enum has
    the same path — no kind-specific carve-outs that violate
    'no hardcoded domains or topologies'."""
    selector = _make_selector()
    base_content = (
        "Bulletin reported $90.5 billion in fiscal year 1945 war activities; "
        "this draft is long enough to clear the min_span_length floor."
    )
    for kind in ("vector_index", "sql_analytics", "qa_assistant", "file"):
        result = asyncio.run(
            selector.select_evidence(
                "treasury",
                [
                    {
                        "url": f"x://{kind}",
                        "content": base_content,
                        "source_kind": kind,
                    }
                ],
                filter_quality=False,
            )
        )
        assert len(result.evidence) == 1, kind
        assert "$90.5 billion" in result.evidence[0].quote_text
