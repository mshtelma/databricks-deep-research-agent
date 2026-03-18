"""T108: Tests for AtomicDecomposer and VerificationRetriever (Stage 7: ARE).

Verifies:
- AtomicDecomposer single and batch decomposition
- Fallback behavior on LLM failure
- Short claim passthrough
- VerificationRetriever claim filtering
- VerificationRetriever retrieve_and_revise flow
- InternalPoolSearcher BM25 search
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from databricks_deep_research.citation.atomic_decomposer import (
    AtomicDecomposer,
    AtomicDecompositionOutput,
    AtomicFact,
    BatchDecompositionItem,
    BatchDecompositionOutput,
    ClaimDecomposition,
    EvidenceSource,
)
from databricks_deep_research.citation.types import (
    ClaimInfo,
    EvidenceInfo,
    RankedEvidence,
)
from databricks_deep_research.citation.verification_retriever import (
    InternalPoolSearcher,
    VerificationEvent,
    VerificationRetriever,
)
from databricks_deep_research.llm.client import LLMResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ranked_evidence(**overrides: Any) -> RankedEvidence:
    defaults: dict[str, Any] = {
        "source_id": None,
        "source_url": "https://example.com/article",
        "source_title": "Example Article",
        "quote_text": "Tesla sold 500,000 vehicles in Q3 2024.",
        "start_offset": 0,
        "end_offset": 50,
        "section_heading": None,
        "relevance_score": 0.9,
        "has_numeric_content": True,
        "is_snippet_based": False,
    }
    defaults.update(overrides)
    return RankedEvidence(**defaults)


def _make_claim_info(**overrides: Any) -> ClaimInfo:
    defaults: dict[str, Any] = {
        "claim_text": "Tesla sold 500,000 vehicles in Q3 2024 and became the most valuable automaker.",
        "claim_type": "general",
        "position_start": 0,
        "position_end": 80,
        "evidence": EvidenceInfo(
            source_url="https://example.com",
            quote_text="Tesla sold 500,000 vehicles in Q3 2024.",
        ),
        "verification_verdict": "unsupported",
    }
    defaults.update(overrides)
    return ClaimInfo(**defaults)


def _mock_llm_structured(
    structured: Any = None, content: str = ""
) -> MagicMock:
    llm = MagicMock()
    resp = LLMResponse(content=content, structured=structured)
    llm.complete = AsyncMock(return_value=resp)
    return llm


# ===========================================================================
# AtomicDecomposer Tests
# ===========================================================================


# ---------------------------------------------------------------------------
# T108-1: Single claim decomposition
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decompose_single_claim_structured() -> None:
    """Decompose a claim into atomic facts via structured output."""
    structured = AtomicDecompositionOutput(
        atomic_facts=[
            "Tesla sold 500,000 vehicles in Q3 2024.",
            "Tesla became the most valuable automaker.",
        ],
        reasoning="Separated sales and valuation claims.",
    )
    llm = _mock_llm_structured(structured=structured)
    decomposer = AtomicDecomposer(llm)

    claim = _make_claim_info()
    result = await decomposer.decompose(claim, claim_index=0)

    assert isinstance(result, ClaimDecomposition)
    assert len(result.atomic_facts) == 2
    assert result.atomic_facts[0].fact_text == "Tesla sold 500,000 vehicles in Q3 2024."
    assert result.atomic_facts[1].fact_text == "Tesla became the most valuable automaker."
    assert result.decomposition_reasoning == "Separated sales and valuation claims."
    llm.complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_decompose_short_claim_passthrough() -> None:
    """Short claims (<= 8 words) are returned as-is without LLM call."""
    llm = _mock_llm_structured()
    decomposer = AtomicDecomposer(llm)

    claim = _make_claim_info(claim_text="Tesla is profitable.")
    result = await decomposer.decompose(claim, claim_index=0)

    assert len(result.atomic_facts) == 1
    assert result.atomic_facts[0].fact_text == "Tesla is profitable."
    assert "atomic" in result.decomposition_reasoning.lower()
    llm.complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_decompose_fallback_on_no_structured_output() -> None:
    """Fallback to single-fact decomposition when structured output is None."""
    llm = _mock_llm_structured(structured=None, content="no structured output")
    decomposer = AtomicDecomposer(llm)

    claim = _make_claim_info()
    result = await decomposer.decompose(claim, claim_index=0)

    assert len(result.atomic_facts) == 1
    assert result.atomic_facts[0].fact_text == claim.claim_text
    assert "fallback" in result.decomposition_reasoning.lower()


@pytest.mark.asyncio
async def test_decompose_fallback_on_exception() -> None:
    """Fallback on LLM exception."""
    llm = MagicMock()
    llm.complete = AsyncMock(side_effect=RuntimeError("LLM down"))
    decomposer = AtomicDecomposer(llm)

    claim = _make_claim_info()
    result = await decomposer.decompose(claim, claim_index=0)

    assert len(result.atomic_facts) == 1
    assert "fallback" in result.decomposition_reasoning.lower()


@pytest.mark.asyncio
async def test_decompose_deduplicates_facts() -> None:
    """Duplicate atomic facts should be removed."""
    structured = AtomicDecompositionOutput(
        atomic_facts=[
            "Tesla sold 500,000 vehicles.",
            "Tesla sold 500,000 vehicles.",  # duplicate
            "Tesla became the most valuable automaker.",
        ],
        reasoning="Test dedup.",
    )
    llm = _mock_llm_structured(structured=structured)
    decomposer = AtomicDecomposer(llm)

    claim = _make_claim_info()
    result = await decomposer.decompose(claim, claim_index=0)

    assert len(result.atomic_facts) == 2  # duplicate removed


@pytest.mark.asyncio
async def test_decompose_caps_at_max_facts() -> None:
    """Atomic facts should be capped at max_atomic_facts_per_claim."""
    structured = AtomicDecompositionOutput(
        atomic_facts=[f"Fact {i}" for i in range(10)],
        reasoning="Many facts.",
    )
    llm = _mock_llm_structured(structured=structured)
    decomposer = AtomicDecomposer(llm, max_atomic_facts_per_claim=3)

    claim = _make_claim_info()
    result = await decomposer.decompose(claim, claim_index=0)

    assert len(result.atomic_facts) <= 3


# ---------------------------------------------------------------------------
# T108-2: Multi-claim decomposition (decompose_claims)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decompose_claims_filters_by_verdict() -> None:
    """decompose_claims should only process claims with matching verdicts."""
    structured = AtomicDecompositionOutput(
        atomic_facts=["Fact 1."],
        reasoning="OK",
    )
    llm = _mock_llm_structured(structured=structured)
    decomposer = AtomicDecomposer(llm)

    claims = [
        _make_claim_info(verification_verdict="supported"),  # skip
        _make_claim_info(verification_verdict="unsupported"),  # process
        _make_claim_info(verification_verdict="partial"),  # process
    ]

    decompositions, metrics = await decomposer.decompose_claims(claims)

    assert metrics.total_claims_processed == 2
    assert len(decompositions) == 2


@pytest.mark.asyncio
async def test_decompose_claims_metrics() -> None:
    """decompose_claims should return correct metrics."""
    structured = AtomicDecompositionOutput(
        atomic_facts=["Fact A.", "Fact B."],
        reasoning="Split.",
    )
    llm = _mock_llm_structured(structured=structured)
    decomposer = AtomicDecomposer(llm)

    claims = [
        _make_claim_info(verification_verdict="unsupported"),
    ]

    decompositions, metrics = await decomposer.decompose_claims(claims)

    assert metrics.total_claims_processed == 1
    assert metrics.total_atomic_facts == 2
    assert metrics.multi_fact_claims == 1
    assert metrics.single_fact_claims == 0
    assert metrics.avg_facts_per_claim == 2.0


# ---------------------------------------------------------------------------
# T108-3: Batch decomposition
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_decompose_structured_output() -> None:
    """batch_decompose should process multiple claims in one LLM call."""
    batch_output = BatchDecompositionOutput(
        decompositions=[
            BatchDecompositionItem(
                claim_index=0,
                atomic_facts=["Tesla sold vehicles.", "Q3 2024 sales."],
                reasoning="Split sales claim.",
            ),
        ]
    )
    llm = _mock_llm_structured(structured=batch_output)
    decomposer = AtomicDecomposer(llm)

    claims = [(0, _make_claim_info())]
    results = await decomposer.batch_decompose(claims)

    assert len(results) == 1
    assert len(results[0].atomic_facts) == 2


@pytest.mark.asyncio
async def test_batch_decompose_empty_input() -> None:
    """batch_decompose returns empty list for empty input."""
    llm = _mock_llm_structured()
    decomposer = AtomicDecomposer(llm)

    results = await decomposer.batch_decompose([])
    assert results == []


@pytest.mark.asyncio
async def test_batch_decompose_short_claims_skip_llm() -> None:
    """Short claims in batch should be passed through without LLM calls."""
    llm = _mock_llm_structured()
    decomposer = AtomicDecomposer(llm)

    claims = [(0, _make_claim_info(claim_text="Short claim."))]
    results = await decomposer.batch_decompose(claims)

    assert len(results) == 1
    assert results[0].atomic_facts[0].fact_text == "Short claim."
    llm.complete.assert_not_awaited()


# ---------------------------------------------------------------------------
# T108-4: ClaimDecomposition.update_verification_status
# ---------------------------------------------------------------------------


def test_update_verification_status_all_verified() -> None:
    claim = _make_claim_info()
    facts = [
        AtomicFact(fact_text="Fact 1", fact_index=0, parent_claim_id=0, is_verified=True),
        AtomicFact(fact_text="Fact 2", fact_index=1, parent_claim_id=0, is_verified=True),
    ]
    decomp = ClaimDecomposition(original_claim=claim, atomic_facts=facts)
    decomp.update_verification_status()

    assert decomp.all_verified is True
    assert decomp.partial_verified is False
    assert decomp.verified_count == 2
    assert decomp.total_count == 2


def test_update_verification_status_partial() -> None:
    claim = _make_claim_info()
    facts = [
        AtomicFact(fact_text="Fact 1", fact_index=0, parent_claim_id=0, is_verified=True),
        AtomicFact(fact_text="Fact 2", fact_index=1, parent_claim_id=0, is_verified=False),
    ]
    decomp = ClaimDecomposition(original_claim=claim, atomic_facts=facts)
    decomp.update_verification_status()

    assert decomp.all_verified is False
    assert decomp.partial_verified is True
    assert decomp.verified_count == 1


def test_update_verification_status_none_verified() -> None:
    claim = _make_claim_info()
    facts = [
        AtomicFact(fact_text="Fact 1", fact_index=0, parent_claim_id=0, is_verified=False),
    ]
    decomp = ClaimDecomposition(original_claim=claim, atomic_facts=facts)
    decomp.update_verification_status()

    assert decomp.all_verified is False
    assert decomp.partial_verified is False
    assert decomp.verified_count == 0


# ===========================================================================
# InternalPoolSearcher Tests
# ===========================================================================


# ---------------------------------------------------------------------------
# T108-5: BM25-based internal pool search
# ---------------------------------------------------------------------------


def test_internal_pool_searcher_finds_relevant_evidence() -> None:
    """InternalPoolSearcher should find evidence matching query terms."""
    pool = [
        _make_ranked_evidence(quote_text="Tesla sold five hundred thousand vehicles in the third quarter of 2024"),
        _make_ranked_evidence(quote_text="Apple released the iPhone 16 with improved camera features"),
        _make_ranked_evidence(quote_text="Tesla electric vehicle sales reached record numbers in 2024"),
    ]
    searcher = InternalPoolSearcher(pool)

    results = searcher.search("Tesla vehicle sales 2024", threshold=0.0)

    assert len(results) >= 1
    # Top result should be Tesla-related
    top_evidence, top_score = results[0]
    assert "tesla" in top_evidence.quote_text.lower()


def test_internal_pool_searcher_empty_pool() -> None:
    """Search on empty pool should return empty list."""
    searcher = InternalPoolSearcher([])
    results = searcher.search("anything")
    assert results == []


def test_internal_pool_searcher_threshold_filtering() -> None:
    """Results below threshold should be filtered out."""
    pool = [
        _make_ranked_evidence(quote_text="Completely unrelated content about gardening"),
    ]
    searcher = InternalPoolSearcher(pool)

    results = searcher.search("quantum computing superconductors", threshold=100.0)
    assert len(results) == 0


def test_internal_pool_searcher_top_k_limit() -> None:
    """Search should respect top_k limit."""
    pool = [
        _make_ranked_evidence(quote_text=f"Tesla related content {i}")
        for i in range(10)
    ]
    searcher = InternalPoolSearcher(pool)

    results = searcher.search("Tesla content", threshold=0.0, top_k=3)
    assert len(results) <= 3


# ===========================================================================
# VerificationRetriever Tests
# ===========================================================================


# ---------------------------------------------------------------------------
# T108-6: Claim filtering
# ---------------------------------------------------------------------------


def test_filter_claims_by_verdict() -> None:
    """VerificationRetriever should filter claims matching trigger_on_verdicts."""
    llm = _mock_llm_structured()
    retriever = VerificationRetriever(llm)

    claims = [
        _make_claim_info(verification_verdict="supported"),
        _make_claim_info(verification_verdict="unsupported"),
        _make_claim_info(verification_verdict="partial"),
        _make_claim_info(verification_verdict="contradicted"),
    ]

    filtered = retriever._filter_claims(claims)

    assert len(filtered) == 2
    verdicts = {c.verification_verdict for _, c in filtered}
    assert verdicts == {"unsupported", "partial"}


def test_filter_claims_custom_verdicts() -> None:
    """Custom trigger_on_verdicts should be respected."""
    llm = _mock_llm_structured()
    retriever = VerificationRetriever(
        llm, trigger_on_verdicts=["contradicted"]
    )

    claims = [
        _make_claim_info(verification_verdict="unsupported"),
        _make_claim_info(verification_verdict="contradicted"),
    ]

    filtered = retriever._filter_claims(claims)
    assert len(filtered) == 1
    assert filtered[0][1].verification_verdict == "contradicted"


# ---------------------------------------------------------------------------
# T108-7: retrieve_and_revise flow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_and_revise_skips_when_no_claims() -> None:
    """retrieve_and_revise should emit stage_7_skipped if no claims match."""
    llm = _mock_llm_structured()
    retriever = VerificationRetriever(llm)

    claims = [
        _make_claim_info(verification_verdict="supported"),
    ]
    evidence_pool = [_make_ranked_evidence()]

    events: list[Any] = []
    async for item in retriever.retrieve_and_revise(
        claims, evidence_pool, "report content", "query"
    ):
        events.append(item)

    assert len(events) == 1
    assert isinstance(events[0], VerificationEvent)
    assert events[0].event_type == "stage_7_skipped"


@pytest.mark.asyncio
async def test_retrieve_and_revise_processes_unsupported() -> None:
    """retrieve_and_revise should process unsupported claims and emit events."""
    # Mock decomposition
    decompose_output = AtomicDecompositionOutput(
        atomic_facts=["Tesla sold vehicles."],
        reasoning="Single fact.",
    )
    # Mock entailment check
    from databricks_deep_research.citation.verification_retriever import (
        EntailmentCheckOutput,
    )

    entailment_output = EntailmentCheckOutput(
        entails=False,
        score=0.3,
        reasoning="No match",
        key_match="",
    )

    # We need the LLM to return different structured outputs for different calls.
    # First call: decomposition, subsequent calls: entailment + reconstruction + softening.
    call_count = 0

    async def _mock_complete(*args: Any, **kwargs: Any) -> LLMResponse:
        nonlocal call_count
        call_count += 1
        so = kwargs.get("structured_output")
        if so is not None and so.__name__ == "AtomicDecompositionOutput":
            return LLMResponse(content="", structured=decompose_output)
        if so is not None and so.__name__ == "EntailmentCheckOutput":
            return LLMResponse(content="", structured=entailment_output)
        # Default: return plain text for reconstruction/softening
        return LLMResponse(content="Reportedly, Tesla sold vehicles.", structured=None)

    llm = MagicMock()
    llm.complete = AsyncMock(side_effect=_mock_complete)
    retriever = VerificationRetriever(llm)

    claims = [
        _make_claim_info(verification_verdict="unsupported"),
    ]
    evidence_pool = [_make_ranked_evidence()]

    events: list[Any] = []
    async for item in retriever.retrieve_and_revise(
        claims, evidence_pool, "Tesla sold vehicles.", "Tesla vehicles"
    ):
        events.append(item)

    # Should have: stage_7_started, claim_verification_started, ClaimRevision, stage_7_complete
    event_types = [
        e.event_type if isinstance(e, VerificationEvent) else type(e).__name__
        for e in events
    ]
    assert "stage_7_started" in event_types
    assert "stage_7_complete" in event_types


# ---------------------------------------------------------------------------
# T108-8: VerificationRetriever metrics reset
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_and_revise_resets_metrics() -> None:
    """Metrics should be reset at the start of each retrieve_and_revise call."""
    llm = _mock_llm_structured()
    retriever = VerificationRetriever(llm)

    # Pollute metrics
    retriever.metrics.total_claims_processed = 999

    claims = [_make_claim_info(verification_verdict="supported")]
    evidence_pool = [_make_ranked_evidence()]

    async for _ in retriever.retrieve_and_revise(
        claims, evidence_pool, "content", "query"
    ):
        pass

    # After running (with no claims to process), metrics should be reset
    assert retriever.metrics.total_claims_processed == 0


# ---------------------------------------------------------------------------
# T108-9: AtomicFact data model
# ---------------------------------------------------------------------------


def test_atomic_fact_to_dict() -> None:
    """AtomicFact.to_dict should produce a complete dictionary."""
    fact = AtomicFact(
        fact_text="Tesla sold 500,000 vehicles.",
        fact_index=0,
        parent_claim_id=0,
        is_verified=True,
        evidence_source=EvidenceSource.INTERNAL,
        entailment_score=0.85,
    )
    d = fact.to_dict()

    assert d["fact_text"] == "Tesla sold 500,000 vehicles."
    assert d["is_verified"] is True
    assert d["evidence_source"] == "internal"
    assert d["entailment_score"] == 0.85
    assert d["evidence"] is None  # No evidence attached


def test_atomic_fact_to_dict_with_evidence() -> None:
    """AtomicFact.to_dict should include evidence when attached."""
    evidence = _make_ranked_evidence()
    fact = AtomicFact(
        fact_text="Tesla sold vehicles.",
        fact_index=0,
        parent_claim_id=0,
        is_verified=True,
        evidence=evidence,
        evidence_source=EvidenceSource.EXTERNAL,
    )
    d = fact.to_dict()

    assert d["evidence"] is not None
    assert d["evidence"]["source_url"] == evidence.source_url


# ---------------------------------------------------------------------------
# T108-10: ClaimDecomposition.to_dict
# ---------------------------------------------------------------------------


def test_claim_decomposition_to_dict() -> None:
    """ClaimDecomposition.to_dict should capture all fields."""
    claim = _make_claim_info()
    facts = [
        AtomicFact(fact_text="Fact 1", fact_index=0, parent_claim_id=0),
    ]
    decomp = ClaimDecomposition(
        original_claim=claim,
        atomic_facts=facts,
        decomposition_reasoning="Test decomposition",
    )
    decomp.update_verification_status()

    d = decomp.to_dict()
    assert d["original_claim"] == claim.claim_text
    assert len(d["atomic_facts"]) == 1
    assert d["all_verified"] is False
    assert d["total_count"] == 1


# ---------------------------------------------------------------------------
# T108-11: EvidenceSource enum
# ---------------------------------------------------------------------------


def test_evidence_source_values() -> None:
    assert EvidenceSource.INTERNAL.value == "internal"
    assert EvidenceSource.EXTERNAL.value == "external"
    assert EvidenceSource.NONE.value == "none"
