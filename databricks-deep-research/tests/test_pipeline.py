"""T105: Tests for CitationVerificationPipeline (7-stage orchestration).

Verifies:
- 7-stage execution flow
- Event emission types
- Content streaming
- Stage skip behavior when disabled
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from databricks_deep_research.citation.config import CitationConfig
from databricks_deep_research.citation.pipeline import (
    CitationVerificationPipeline,
    VerificationEvent,
    _build_verification_summary,
)
from databricks_deep_research.citation.types import (
    ClaimInfo,
    ClaimRole,
    ConfidenceLevel,
    ConfidenceResult,
    CorrectionAction,
    CorrectionMetrics,
    CorrectionResult,
    EvidenceInfo,
    InterleavedClaim,
    NumericValue,
    NumericVerificationResult,
    RankedEvidence,
    VerificationResult,
    VerificationVerdict,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ranked_evidence(**overrides: Any) -> RankedEvidence:
    defaults: dict[str, Any] = {
        "source_id": None,
        "source_url": "https://example.com/article",
        "source_title": "Example Article",
        "quote_text": "Example evidence quote text for testing.",
        "start_offset": 0,
        "end_offset": 50,
        "section_heading": None,
        "relevance_score": 0.9,
        "has_numeric_content": False,
        "is_snippet_based": False,
    }
    defaults.update(overrides)
    return RankedEvidence(**defaults)


def _make_evidence_info(**overrides: Any) -> EvidenceInfo:
    defaults: dict[str, Any] = {
        "source_url": "https://example.com/article",
        "quote_text": "Example evidence quote text for testing.",
        "start_offset": 0,
        "end_offset": 50,
        "relevance_score": 0.9,
        "has_numeric_content": False,
    }
    defaults.update(overrides)
    return EvidenceInfo(**defaults)


def _make_interleaved_claim(**overrides: Any) -> InterleavedClaim:
    defaults: dict[str, Any] = {
        "claim_text": "The company grew by 20% last year.",
        "claim_type": "general",
        "position_start": 0,
        "position_end": 35,
        "evidence": _make_ranked_evidence(),
        "evidence_index": 0,
        "confidence_score": 0.85,
        "citation_key": "Example",
        "citation_keys": ["Example"],
    }
    defaults.update(overrides)
    return InterleavedClaim(**defaults)


def _make_claim_info(**overrides: Any) -> ClaimInfo:
    defaults: dict[str, Any] = {
        "claim_text": "The company grew by 20% last year.",
        "claim_type": "general",
        "position_start": 0,
        "position_end": 35,
        "evidence": _make_evidence_info(),
        "confidence_level": "medium",
    }
    defaults.update(overrides)
    return ClaimInfo(**defaults)


def _mock_evidence_selector(
    evidence: list[RankedEvidence] | None = None,
) -> MagicMock:
    """Mock Stage 1: EvidenceSelector protocol."""
    selector = MagicMock()
    selector.select_evidence_spans = AsyncMock(
        return_value=evidence if evidence is not None else [_make_ranked_evidence()]
    )
    return selector


def _mock_claim_generator(
    content: str = "Generated content [0].",
    claims: list[InterleavedClaim] | None = None,
) -> MagicMock:
    """Mock Stage 2: ClaimGenerator protocol."""
    generator = MagicMock()
    claim_list = claims or [_make_interleaved_claim()]

    async def _stream(*args: Any, **kwargs: Any) -> AsyncGenerator[
        tuple[str, InterleavedClaim | None], None
    ]:
        yield content, None
        for c in claim_list:
            yield "", c

    generator.synthesize_with_streaming = _stream
    return generator


def _mock_confidence_classifier() -> MagicMock:
    """Mock Stage 3: ConfidenceClassifierProtocol."""
    classifier = MagicMock()
    classifier.classify.return_value = ConfidenceResult(
        level=ConfidenceLevel.MEDIUM,
        score=0.65,
        indicators=["neutral"],
        reasoning="Medium confidence",
    )
    classifier.should_use_quick_verification.return_value = False
    return classifier


def _mock_isolated_verifier() -> MagicMock:
    """Mock Stage 4: IsolatedVerifierProtocol."""
    verifier = MagicMock()
    verifier.verify_with_isolation = AsyncMock(
        return_value=VerificationResult(
            verdict=VerificationVerdict.SUPPORTED,
            reasoning="Evidence directly supports claim.",
        )
    )
    return verifier


def _mock_analysis_grounding_verifier() -> MagicMock:
    verifier = MagicMock()
    verifier.verify_analysis_claim = AsyncMock(
        return_value=VerificationResult(
            verdict=VerificationVerdict.PARTIAL,
            reasoning="Analysis is directionally supported but somewhat overstated.",
            confidence=0.72,
        )
    )
    return verifier


def _mock_citation_corrector() -> MagicMock:
    """Mock Stage 5: CitationCorrectorProtocol."""
    corrector = MagicMock()
    corrector.correct_citations = AsyncMock(
        return_value=([], CorrectionMetrics())
    )
    return corrector


def _mock_numeric_verifier() -> MagicMock:
    """Mock Stage 6: NumericVerifierProtocol."""
    verifier = MagicMock()
    verifier.verify_numeric_claim = AsyncMock(
        return_value=NumericVerificationResult(
            claim_text="Revenue was $3.2B",
            parsed_value=NumericValue(
                raw_text="$3.2B", normalized_value=None, unit="USD", entity="revenue"
            ),
            qa_results=[],
            overall_match=True,
            derivation_type="direct",
            confidence=0.9,
        )
    )
    return verifier


def _mock_llm() -> MagicMock:
    llm = MagicMock()
    llm.complete = AsyncMock()
    return llm


def _build_pipeline(
    config: CitationConfig | None = None,
    **overrides: Any,
) -> CitationVerificationPipeline:
    """Build a pipeline with all mocked stages."""
    kwargs: dict[str, Any] = {
        "llm": _mock_llm(),
        "evidence_selector": _mock_evidence_selector(),
        "claim_generator": _mock_claim_generator(),
        "confidence_classifier": _mock_confidence_classifier(),
        "isolated_verifier": _mock_isolated_verifier(),
        "citation_corrector": _mock_citation_corrector(),
        "numeric_verifier": _mock_numeric_verifier(),
        "analysis_grounding_verifier": _mock_analysis_grounding_verifier(),
        "config": config,
    }
    kwargs.update(overrides)
    return CitationVerificationPipeline(**kwargs)


# ---------------------------------------------------------------------------
# T105-1: Full pipeline yields content then events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_pipeline_yields_content_and_events() -> None:
    """Run the full pipeline and verify it yields content and events."""
    sources = [
        {"url": "https://example.com/1", "title": "Article 1", "content": "Long " * 200}
    ]
    pipeline = _build_pipeline()

    collected: list[str | VerificationEvent] = []
    async for item in pipeline.run_full_pipeline(
        sources=sources,
        observations=["Some observation"],
        query="test query",
    ):
        collected.append(item)

    # Must have at least content and a verification_summary event
    content_items = [i for i in collected if isinstance(i, str)]
    event_items = [i for i in collected if isinstance(i, VerificationEvent)]

    assert len(content_items) >= 1, "Pipeline should yield content"
    assert len(event_items) >= 1, "Pipeline should yield events"

    # Verify a verification_summary event is emitted
    summary_events = [
        e for e in event_items if e.event_type == "verification_summary"
    ]
    assert len(summary_events) == 1, "Exactly one verification_summary expected"
    summary_data = summary_events[0].data
    assert "total_claims" in summary_data
    assert "supported" in summary_data
    assert "supported_rate" in summary_data

    claim_generated_events = [
        e for e in event_items if e.event_type == "claim_generated"
    ]
    assert len(claim_generated_events) == 1
    assert claim_generated_events[0].data["claim_index"] == 0

    assert len(pipeline.last_generated_claims) == 1
    assert pipeline.last_verification_summary is not None
    assert pipeline.last_final_content == "Generated content [0]."


@pytest.mark.asyncio
async def test_full_pipeline_strips_internal_reclaim_tags_from_final_content() -> None:
    """Internal reclaim tags should not remain in the final report."""
    sources = [
        {"url": "https://example.com/1", "title": "Article 1", "content": "Long " * 200}
    ]
    pipeline = _build_pipeline(
        claim_generator=_mock_claim_generator(
            "<free>Overview section.</free>\n\nRevenue increased 12% [0].",
            [
                _make_interleaved_claim(
                    claim_text="Revenue increased 12%.",
                    claim_type="numeric",
                )
            ],
        )
    )

    async for _ in pipeline.run_full_pipeline(
        sources=sources,
        observations=["Some observation"],
        query="test query",
    ):
        pass

    assert "<free>" not in pipeline.last_final_content
    assert "</free>" not in pipeline.last_final_content


# ---------------------------------------------------------------------------
# T105-2: Stage 1 pre-selects evidence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stage1_preselect_evidence() -> None:
    """Stage 1 should call the evidence selector and return ranked evidence."""
    evidence = [_make_ranked_evidence(relevance_score=0.95)]
    selector = _mock_evidence_selector(evidence)
    pipeline = _build_pipeline(evidence_selector=selector)

    sources = [{"url": "https://a.com", "title": "A", "content": "content " * 100}]
    result = await pipeline.preselect_evidence(sources, "test query")

    assert len(result) == 1
    assert result[0].relevance_score == 0.95
    selector.select_evidence_spans.assert_awaited_once()


# ---------------------------------------------------------------------------
# T105-3: Stage 1 skips when disabled
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stage1_skips_when_disabled() -> None:
    """Stage 1 returns empty list when evidence preselection is disabled."""
    config = CitationConfig(enable_evidence_preselection=False)
    pipeline = _build_pipeline(config=config)

    result = await pipeline.preselect_evidence(
        [{"url": "https://a.com", "content": "c"}], "q"
    )
    assert result == []


# ---------------------------------------------------------------------------
# T105-4: Stage 2 generates claims via streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stage2_interleaved_generation_yields_claims() -> None:
    """Stage 2 should yield content and claims from the generator."""
    evidence = [_make_ranked_evidence()]
    claim = _make_interleaved_claim()
    generator = _mock_claim_generator("Test content [0].", [claim])
    pipeline = _build_pipeline(claim_generator=generator)

    results: list[tuple[str, InterleavedClaim | None]] = []
    async for content, interleaved_claim in pipeline.generate_with_interleaving(
        evidence_pool=evidence,
        observations=[],
        query="test query",
    ):
        results.append((content, interleaved_claim))

    # First yield has content, second has claim
    assert any(content for content, _ in results)
    assert any(c for _, c in results if c is not None)


@pytest.mark.asyncio
async def test_stage2_forwards_generation_instructions() -> None:
    """Workflow-specific report contracts must reach Stage 2 generation."""
    evidence = [_make_ranked_evidence()]
    captured: dict[str, Any] = {}

    async def _stream(*args: Any, **kwargs: Any) -> AsyncGenerator[
        tuple[str, InterleavedClaim | None], None
    ]:
        captured.update(kwargs)
        yield "Contract-aware content [0].", None
        yield "", _make_interleaved_claim()

    generator = MagicMock()
    generator.synthesize_with_streaming = _stream
    pipeline = _build_pipeline(claim_generator=generator)

    async for _content, _claim in pipeline.generate_with_interleaving(
        evidence_pool=evidence,
        observations=[],
        query="test query",
        generation_instructions="Use sections A and B.",
    ):
        pass

    assert captured["generation_instructions"] == "Use sections A and B."


# ---------------------------------------------------------------------------
# T105-5: Stage 3 confidence classification
# ---------------------------------------------------------------------------


def test_stage3_confidence_classification() -> None:
    """Stage 3 classifies claim confidence level."""
    pipeline = _build_pipeline()
    claim = _make_claim_info()

    level = pipeline.classify_confidence(claim)
    assert level in ("high", "medium", "low")


def test_stage3_returns_medium_when_disabled() -> None:
    """Stage 3 returns 'medium' when confidence classification is disabled."""
    config = CitationConfig(enable_confidence_classification=False)
    pipeline = _build_pipeline(config=config)
    claim = _make_claim_info()

    level = pipeline.classify_confidence(claim)
    assert level == "medium"


def test_stage3_routes_exact_numeric_claims_high() -> None:
    """Exact numeric fact matches should use the high-confidence quick path."""
    pipeline = _build_pipeline()
    claim = _make_claim_info(
        claim_text="Operating profit reached $912 million with EPS of $0.90.",
        claim_type="numeric",
        evidence=_make_evidence_info(
            quote_text="Operating Profit of $912 million; EPS of $0.90",
        ),
    )
    pipeline._classify_and_link_claims([claim])

    result = pipeline.classify_confidence_result(claim)

    assert result.level == ConfidenceLevel.HIGH
    assert result.score >= 0.9


def test_stage3_routes_analysis_claims_low() -> None:
    """Analysis claims should always avoid the quick factual verifier."""
    pipeline = _build_pipeline()
    claim = _make_claim_info(
        claim_text="This may indicate stronger enterprise demand.",
        claim_role=ClaimRole.ANALYSIS.value,
        evidence=None,
    )
    pipeline._classify_and_link_claims([claim])

    result = pipeline.classify_confidence_result(claim)

    assert result.level == ConfidenceLevel.LOW
    assert result.score <= 0.3


# ---------------------------------------------------------------------------
# T105-6: Stage 4 verify_claims yields events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stage4_verify_claims_emits_events() -> None:
    """Stage 4 should emit claim_verified events."""
    pipeline = _build_pipeline()
    claims = [_make_claim_info()]

    events: list[VerificationEvent] = []
    async for event in pipeline.verify_claims(claims):
        events.append(event)

    assert len(events) >= 1
    assert events[0].event_type == "claim_verified"
    assert events[0].data["verdict"] == "supported"


@pytest.mark.asyncio
async def test_stage4_abstains_when_no_evidence() -> None:
    """Stage 4 should mark claims without evidence as abstained."""
    pipeline = _build_pipeline()
    claim = _make_claim_info(evidence=None)

    events: list[VerificationEvent] = []
    async for event in pipeline.verify_claims([claim]):
        events.append(event)

    assert claim.abstained is True
    assert len(events) == 1
    assert events[0].event_type == "claim_verified"
    assert events[0].data["verdict"] == "abstained"
    assert events[0].data["confidence"] == 0.0
    assert events[0].data["verification_confidence"] == 0.0


@pytest.mark.asyncio
async def test_stage4_verifies_multi_citation_claim_with_combined_evidence() -> None:
    """Multi-citation claims should be verified against the union of cited evidence."""
    verifier = _mock_isolated_verifier()
    verifier.verify_with_isolation = AsyncMock(
        return_value=VerificationResult(
            verdict=VerificationVerdict.SUPPORTED,
            reasoning="Combined evidence supports the comparison.",
            confidence=0.88,
        )
    )
    pipeline = _build_pipeline(isolated_verifier=verifier)
    claim = _make_claim_info(
        claim_text="Q2 identical sales were 3.4%, compared with 3.2% in Q1.",
        evidence=_make_evidence_info(
            source_url="https://example.com/q2",
            quote_text="Second Quarter Highlights - Identical Sales without fuel increased 3.4%",
        ),
        evidences=[
            _make_evidence_info(
                source_url="https://example.com/q2",
                quote_text="Second Quarter Highlights - Identical Sales without fuel increased 3.4%",
            ),
            _make_evidence_info(
                source_url="https://example.com/q1",
                quote_text="First Quarter Highlights - Identical Sales without fuel increased 3.2%",
            ),
        ],
        confidence_level="medium",
        routing_confidence_score=0.65,
    )

    events: list[VerificationEvent] = []
    async for event in pipeline.verify_claims([claim]):
        events.append(event)

    verifier.verify_with_isolation.assert_awaited_once()
    evidence_arg = verifier.verify_with_isolation.await_args.kwargs["evidence"]
    assert "Source 1:" in evidence_arg.quote_text
    assert "3.4%" in evidence_arg.quote_text
    assert "3.2%" in evidence_arg.quote_text
    assert events[0].data["confidence"] == pytest.approx(0.88)
    assert events[0].data["routing_confidence_level"] == "medium"


def test_stage21_classifies_fact_analysis_and_free_roles() -> None:
    """Stage 2.1 should enforce fact, analysis, and free boundaries."""
    pipeline = _build_pipeline()
    claims = [
        _make_claim_info(
            claim_text="Revenue increased 12%, indicating strong demand.",
            claim_type="numeric",
        ),
        _make_claim_info(
            claim_text="This may indicate stronger enterprise demand.",
            claim_type="general",
            claim_role=ClaimRole.ANALYSIS.value,
        ),
        _make_claim_info(
            claim_text="## Overview",
            claim_type="general",
            claim_role=ClaimRole.FREE.value,
            evidence=None,
        ),
    ]

    pipeline._classify_and_link_claims(claims)

    assert claims[0].claim_role == ClaimRole.FACT.value
    assert claims[0].verification_text == "Revenue increased 12%"
    assert claims[1].claim_role == ClaimRole.ANALYSIS.value
    assert claims[1].verification_method == "grounding"
    assert claims[2].claim_role == ClaimRole.FREE.value
    assert claims[2].abstained is True


def test_stage21_trims_concessive_analysis_tail_from_fact_core() -> None:
    """Fact-core extraction must not leave dangling connectors behind."""
    pipeline = _build_pipeline()
    claim = _make_claim_info(
        claim_text=(
            "Net revenue retention rate stood at 125% as of the most recent "
            "fiscal year-end, down from a peak of 158% reported in an earlier "
            "period, but still indicating meaningful expansion."
        ),
        claim_type="numeric",
    )

    pipeline._classify_and_link_claims([claim])

    assert claim.claim_role == ClaimRole.FACT.value
    assert claim.verification_text == (
        "Net revenue retention rate stood at 125% as of the most recent "
        "fiscal year-end, down from a peak of 158% reported in an earlier period"
    )


def test_stage21_keeps_structural_free_blocks_out_of_analysis_lane() -> None:
    """Purely structural free text should not become an abstained analysis claim."""
    pipeline = _build_pipeline()
    claim = _make_claim_info(
        claim_text="Kroger's 2024 financial performance provides essential context.",
        claim_type="general",
        claim_role=ClaimRole.FREE.value,
        evidence=None,
    )

    pipeline._classify_and_link_claims([claim])

    assert claim.claim_role == ClaimRole.FREE.value
    assert claim.abstained is True


def test_stage21_routes_editorial_quarter_summary_to_analysis() -> None:
    """Quarter/date mentions should not keep editorial summary lines in the fact lane."""
    pipeline = _build_pipeline()
    claim = _make_claim_info(
        claim_text="The second quarter of 2025 continued the positive momentum established in Q1.",
        claim_type="general",
    )

    pipeline._classify_and_link_claims([claim])

    assert claim.claim_role == ClaimRole.ANALYSIS.value
    assert claim.verification_method == "grounding"


def test_stage21_keeps_structural_numeric_lines_out_of_fact_lane() -> None:
    """Markdown-heavy structural lines with numbers should not become numeric fact claims."""
    pipeline = _build_pipeline()
    claim = _make_claim_info(
        claim_text="**Design validation**: Scaling to 7,000 GPUs required extensive testing.",
        claim_type="numeric",
        claim_role=ClaimRole.FREE.value,
        evidence=None,
    )

    pipeline._classify_and_link_claims([claim])

    assert claim.claim_role == ClaimRole.FREE.value
    assert claim.abstained is True


@pytest.mark.asyncio
async def test_stage4_routes_analysis_claims_to_grounding_verifier() -> None:
    """Analysis claims should use the analysis-grounding verifier, not entailment."""
    isolated_verifier = _mock_isolated_verifier()
    analysis_verifier = _mock_analysis_grounding_verifier()
    pipeline = _build_pipeline(
        isolated_verifier=isolated_verifier,
        analysis_grounding_verifier=analysis_verifier,
    )
    fact_claim = _make_claim_info(
        claim_text="Revenue increased 12%.",
        claim_type="numeric",
        claim_role=ClaimRole.FACT.value,
    )
    analysis_claim = _make_claim_info(
        claim_text="This may indicate stronger demand.",
        claim_type="general",
        claim_role=ClaimRole.ANALYSIS.value,
        evidence=None,
    )
    pipeline._classify_and_link_claims([fact_claim, analysis_claim])

    events: list[VerificationEvent] = []
    async for event in pipeline.verify_claims([fact_claim, analysis_claim]):
        events.append(event)

    analysis_event = next(
        event
        for event in events
        if event.event_type == "claim_verified"
        and event.data["claim_role"] == ClaimRole.ANALYSIS.value
    )
    analysis_verifier.verify_analysis_claim.assert_awaited_once()
    isolated_verifier.verify_with_isolation.assert_awaited_once()
    assert analysis_event.data["verification_method"] == "grounding"


@pytest.mark.asyncio
async def test_analysis_grounding_supports_bounded_interpretation() -> None:
    """Hedged analysis over verified facts should be treated as grounded."""
    from databricks_deep_research.citation.analysis_grounding import (
        AnalysisGroundingVerifier,
    )

    verifier = AnalysisGroundingVerifier(_mock_llm())
    result = await verifier.verify_analysis_claim(
        claim_text=(
            "This may indicate that comparable sales remained resilient despite the "
            "reported loss."
        ),
        supporting_claims=[
            "Identical sales without fuel increased 2.6%.",
            "Operating loss was $(1,541) million.",
        ],
        evidences=[_make_ranked_evidence(quote_text="Identical sales without fuel increased 2.6%")],
        supporting_fact_contexts=[
            {
                "claim_text": "Identical sales without fuel increased 2.6%.",
                "verification_text": "Identical sales without fuel increased 2.6%.",
                "verdict": "supported",
            },
            {
                "claim_text": "Operating loss was $(1,541) million.",
                "verification_text": "Operating loss was $(1,541) million.",
                "verdict": "supported",
            },
        ],
    )

    assert result.verdict == VerificationVerdict.SUPPORTED


@pytest.mark.asyncio
async def test_analysis_grounding_rejects_new_numeric_payload() -> None:
    """Analysis should not be allowed to introduce fresh numeric payload."""
    from databricks_deep_research.citation.analysis_grounding import (
        AnalysisGroundingVerifier,
    )

    verifier = AnalysisGroundingVerifier(_mock_llm())
    result = await verifier.verify_analysis_claim(
        claim_text="This may indicate margins improved to 4.5% in Q4 2025.",
        supporting_claims=["Operating profit was $863 million."],
        evidences=[_make_ranked_evidence(quote_text="Operating profit was $863 million.")],
        supporting_fact_contexts=[
            {
                "claim_text": "Operating profit was $863 million.",
                "verification_text": "Operating profit was $863 million.",
                "verdict": "supported",
            }
        ],
    )

    assert result.verdict == VerificationVerdict.UNSUPPORTED


@pytest.mark.asyncio
async def test_stage5_skips_analysis_claims() -> None:
    """Citation correction should only run on fact claims."""
    corrector = _mock_citation_corrector()
    pipeline = _build_pipeline(citation_corrector=corrector)
    claim = _make_claim_info(
        claim_text="This may indicate stronger demand.",
        claim_role=ClaimRole.ANALYSIS.value,
        verification_verdict="unsupported",
        evidence=None,
    )

    events: list[VerificationEvent] = []
    async for event in pipeline.correct_citations([claim], [_make_ranked_evidence()]):
        events.append(event)

    corrector.correct_citations.assert_not_awaited()
    assert events == []


# ---------------------------------------------------------------------------
# T105-7: Stage 5 citation correction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stage5_correction_skips_when_disabled() -> None:
    """Stage 5 should skip when citation correction is disabled."""
    config = CitationConfig(enable_citation_correction=False)
    pipeline = _build_pipeline(config=config)
    claims = [_make_claim_info(verification_verdict="partial")]
    evidence = [_make_ranked_evidence()]

    events: list[VerificationEvent] = []
    async for event in pipeline.correct_citations(claims, evidence):
        events.append(event)

    assert len(events) == 0


@pytest.mark.asyncio
async def test_stage5_correction_emits_metrics() -> None:
    """Stage 5 should emit correction_metrics event."""
    corrector = _mock_citation_corrector()
    corrector.correct_citations = AsyncMock(
        return_value=(
            [CorrectionResult(
                claim_text="test claim",
                correction_type=CorrectionAction.KEEP,
                original_evidence=None,
                corrected_evidence=None,
            )],
            CorrectionMetrics(total_claims=1, kept=1),
        )
    )
    pipeline = _build_pipeline(citation_corrector=corrector)
    claims = [_make_claim_info(verification_verdict="unsupported")]
    evidence = [_make_ranked_evidence()]

    events: list[VerificationEvent] = []
    async for event in pipeline.correct_citations(claims, evidence):
        events.append(event)

    metric_events = [e for e in events if e.event_type == "correction_metrics"]
    assert len(metric_events) == 1
    assert metric_events[0].data["kept"] == 1


@pytest.mark.asyncio
async def test_stage5_replace_reverifies_and_refreshes_claim_state() -> None:
    """Replacing evidence should re-run verification and update citation keys."""
    verifier = _mock_isolated_verifier()
    verifier.verify_with_isolation = AsyncMock(
        return_value=VerificationResult(
            verdict=VerificationVerdict.SUPPORTED,
            reasoning="Replacement evidence directly supports the claim.",
            confidence=0.82,
        )
    )
    corrected_evidence = _make_ranked_evidence(
        source_url="https://replacement.com/article",
        quote_text="Replacement source says revenue increased 20%.",
        source_pool_index=1,
        evidence_pool_index=1,
    )
    corrector = _mock_citation_corrector()
    corrector.correct_citations = AsyncMock(
        return_value=(
            [
                CorrectionResult(
                    claim_text="Revenue increased 20%.",
                    correction_type=CorrectionAction.REPLACE,
                    original_evidence=_make_ranked_evidence(),
                    corrected_evidence=corrected_evidence,
                    reasoning="Found stronger source.",
                )
            ],
            CorrectionMetrics(total_claims=1, replaced=1),
        )
    )
    pipeline = _build_pipeline(
        isolated_verifier=verifier,
        citation_corrector=corrector,
    )
    claim = _make_claim_info(
        claim_text="Revenue increased 20%.",
        verification_verdict="unsupported",
        evidence=_make_evidence_info(),
        evidences=[_make_evidence_info()],
        citation_key="Example",
        citation_keys=["Example"],
    )
    evidence_pool = [
        _make_ranked_evidence(source_url="https://example.com/article"),
        corrected_evidence,
    ]

    events: list[VerificationEvent] = []
    async for event in pipeline.correct_citations([claim], evidence_pool):
        events.append(event)

    correction_event = next(e for e in events if e.event_type == "citation_corrected")
    reverify_event = next(e for e in events if e.event_type == "claim_verified")
    assert correction_event.data["original_key"] == "Example"
    assert correction_event.data["corrected_key"] == "Replacement"
    assert correction_event.data["corrected_source_pool_index"] == 1
    assert reverify_event.data["confidence"] == pytest.approx(0.82)
    assert claim.verification_verdict == "supported"
    assert claim.verification_confidence == pytest.approx(0.82)
    assert claim.citation_key == "Replacement"


# ---------------------------------------------------------------------------
# T105-8: Disabled pipeline yields nothing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disabled_pipeline_yields_nothing() -> None:
    """A globally disabled pipeline should yield nothing."""
    config = CitationConfig(enabled=False)
    pipeline = _build_pipeline(config=config)
    sources = [{"url": "https://a.com", "content": "stuff " * 100}]

    collected: list[Any] = []
    async for item in pipeline.run_full_pipeline(sources, [], "query"):
        collected.append(item)

    assert len(collected) == 0


@pytest.mark.asyncio
async def test_stage7_retrieval_revises_report_content() -> None:
    """Stage 7 should apply claim revisions before Stage 8 finalization."""
    config = CitationConfig(enable_verification_retrieval=True)
    claim = _make_interleaved_claim(
        claim_text="Revenue increased sharply.",
        position_start=0,
        position_end=25,
    )
    generator = _mock_claim_generator("Revenue increased sharply. [0]", [claim])
    verifier = _mock_isolated_verifier()
    verifier.verify_with_isolation = AsyncMock(
        return_value=VerificationResult(
            verdict=VerificationVerdict.UNSUPPORTED,
            reasoning="Needs decomposition.",
            confidence=0.55,
        )
    )

    class _MockVerificationRetriever:
        def __init__(self) -> None:
            self.metrics = SimpleNamespace(
                to_dict=lambda: {
                    "total_atomic_facts": 1,
                    "facts_verified": 1,
                    "facts_softened": 0,
                    "claims_fully_verified": 1,
                    "claims_partially_softened": 0,
                    "claims_fully_softened": 0,
                    "external_searches": 0,
                    "new_sources_added": 0,
                }
            )

        async def retrieve_and_revise(
            self,
            claims: list[ClaimInfo],
            evidence_pool: list[RankedEvidence],
            report_content: str,
            research_query: str,
        ) -> AsyncGenerator[Any, None]:
            yield SimpleNamespace(
                revision_type="fully_verified",
                original_position_start=claims[0].position_start,
                original_position_end=claims[0].position_end,
                original_claim=claims[0].claim_text,
                revised_claim="Revenue increased 20%. [Example]",
            )

        def apply_all_revisions(self, report: str, revisions: list[Any]) -> str:
            return report.replace(
                "Revenue increased sharply.",
                revisions[0].revised_claim,
            )

    pipeline = _build_pipeline(
        config=config,
        claim_generator=generator,
        isolated_verifier=verifier,
        verification_retriever=_MockVerificationRetriever(),
    )
    sources = [{"url": "https://example.com/1", "title": "Article 1", "content": "Long " * 200}]

    async for _ in pipeline.run_full_pipeline(sources=sources, observations=[], query="q"):
        pass

    assert "Revenue increased 20%. [Example]" in pipeline.last_final_content
    assert pipeline.last_verification_summary is not None
    assert pipeline.last_verification_summary.claims_fully_verified == 1


@pytest.mark.asyncio
async def test_stage7_retrieval_skips_analysis_claims() -> None:
    """Stage 7 retrieval should only consider fact claims."""
    config = CitationConfig(enable_verification_retrieval=True)
    claim = _make_interleaved_claim(
        claim_text="This may indicate stronger demand.",
        claim_type="general",
        claim_role=ClaimRole.ANALYSIS.value,
        evidence=None,
        position_start=0,
        position_end=33,
    )
    generator = _mock_claim_generator("This may indicate stronger demand.", [claim])

    retriever = MagicMock()
    retriever.retrieve_and_revise = AsyncMock()
    retriever.apply_all_revisions = MagicMock(return_value="This may indicate stronger demand.")

    pipeline = _build_pipeline(
        config=config,
        claim_generator=generator,
        verification_retriever=retriever,
    )
    sources = [{"url": "https://example.com/1", "title": "Article 1", "content": "Long " * 200}]

    async for _ in pipeline.run_full_pipeline(sources=sources, observations=[], query="q"):
        pass

    retriever.retrieve_and_revise.assert_not_awaited()


def test_build_verification_summary_warns_on_low_supported_rate() -> None:
    """Low supported-rate runs should raise a summary warning."""
    claims = [
        _make_claim_info(verification_verdict="supported"),
        _make_claim_info(verification_verdict="partial"),
        _make_claim_info(verification_verdict="partial"),
        _make_claim_info(verification_verdict="partial"),
        _make_claim_info(verification_verdict="partial"),
        _make_claim_info(verification_verdict="partial"),
    ]

    summary = _build_verification_summary(claims)

    assert summary.supported_rate == pytest.approx(1 / 6)
    assert summary.warning is True


def test_build_verification_summary_excludes_abstained_from_supported_rate() -> None:
    """Fact supported-rate should ignore abstained structural items."""
    claims = [
        _make_claim_info(
            verification_verdict="supported",
            claim_role=ClaimRole.FACT.value,
        ),
        _make_claim_info(
            verification_verdict="partial",
            claim_role=ClaimRole.FACT.value,
        ),
        _make_claim_info(
            verification_verdict="unsupported",
            claim_role=ClaimRole.FACT.value,
        ),
        _make_claim_info(
            verification_verdict=None,
            claim_role=ClaimRole.FACT.value,
            abstained=True,
        ),
    ]

    summary = _build_verification_summary(claims)

    assert summary.supported_rate == pytest.approx(1 / 3)
    assert summary.unsupported_rate == pytest.approx(1 / 3)


def test_build_verification_summary_splits_fact_and_analysis_metrics() -> None:
    """Top-level summary should remain fact-only and expose analysis_summary separately."""
    claims = [
        _make_claim_info(
            verification_verdict="supported",
            claim_role=ClaimRole.FACT.value,
        ),
        _make_claim_info(
            verification_verdict="partial",
            claim_role=ClaimRole.FACT.value,
        ),
        _make_claim_info(
            claim_text="This may indicate stronger demand.",
            verification_verdict="partial",
            claim_role=ClaimRole.ANALYSIS.value,
        ),
        _make_claim_info(
            claim_text="This may indicate stronger demand.",
            verification_verdict="unsupported",
            claim_role=ClaimRole.ANALYSIS.value,
        ),
    ]

    summary = _build_verification_summary(claims)

    assert summary.total_claims == 2
    assert summary.supported_count == 1
    assert summary.partial_count == 1
    assert summary.analysis_summary.total_claims == 2
    assert summary.analysis_summary.partial_count == 1
    assert summary.analysis_summary.unsupported_count == 1


@pytest.mark.asyncio
async def test_stage8_softens_partial_analysis_claims_by_default() -> None:
    """With default disposition, partial analysis claims are softened (not kept).

    The framework's defaults were flipped to favour SOFTEN over KEEP/REMOVE
    for non-contradicted verdicts so unverified content is preserved with
    hedge language instead of being deleted (REMOVE) or presented as fact
    (KEEP). See ``ClaimDispositionConfig`` docstring.
    """
    pipeline = _build_pipeline()
    claim = _make_claim_info(
        claim_text="This indicates strong demand.",
        claim_role=ClaimRole.ANALYSIS.value,
        verification_verdict="partial",
        position_start=0,
        position_end=30,
    )

    content, removed, softened, rewritten = await pipeline.process_unverified_claims(
        "This indicates strong demand.",
        [claim],
    )

    # New default analysis_partial = SOFTEN.
    assert removed == 0
    assert softened == 1
    assert rewritten == 0


@pytest.mark.asyncio
async def test_stage8_disposition_softens_unsupported_fact_by_default() -> None:
    """Unsupported fact claims are softened (not removed) with default disposition.

    Flipped from REMOVE → SOFTEN to preserve report shape; see
    ``ClaimDispositionConfig`` docstring for the rationale. Callers wanting
    the old REMOVE behaviour pass an explicit config (see
    ``test_stage8_disposition_keep`` for the symmetric escape hatch).
    """
    pipeline = _build_pipeline()
    claim = _make_claim_info(
        claim_text="Revenue was $5B.",
        claim_role=ClaimRole.FACT.value,
        verification_verdict="unsupported",
        position_start=0,
        position_end=16,
    )

    content, removed, softened, rewritten = await pipeline.process_unverified_claims(
        "Revenue was $5B.",
        [claim],
    )

    # New default unsupported = SOFTEN (was REMOVE pre-flip).
    assert removed == 0
    assert softened == 1


@pytest.mark.asyncio
async def test_stage8_disposition_remove_when_explicitly_configured() -> None:
    """Callers opting into the legacy REMOVE behaviour still get it.

    Verifies the back-out path documented in ``ClaimDispositionConfig`` —
    constructing the config with ``unsupported=REMOVE`` restores pre-flip
    semantics for compliance pipelines that need hard-removal.
    """
    from databricks_deep_research.citation.config import ClaimDisposition, ClaimDispositionConfig

    cfg = CitationConfig(
        claim_disposition=ClaimDispositionConfig(unsupported=ClaimDisposition.REMOVE),
    )
    pipeline = _build_pipeline(config=cfg)
    claim = _make_claim_info(
        claim_text="Revenue was $5B.",
        claim_role=ClaimRole.FACT.value,
        verification_verdict="unsupported",
        position_start=0,
        position_end=16,
    )

    _, removed, softened, _ = await pipeline.process_unverified_claims(
        "Revenue was $5B.",
        [claim],
    )

    assert removed == 1
    assert softened == 0


@pytest.mark.asyncio
async def test_stage8_disposition_keep() -> None:
    """Unsupported fact claims are kept when disposition says keep."""
    from databricks_deep_research.citation.config import ClaimDisposition, ClaimDispositionConfig

    cfg = CitationConfig(
        claim_disposition=ClaimDispositionConfig(unsupported=ClaimDisposition.KEEP),
    )
    pipeline = _build_pipeline(config=cfg)
    claim = _make_claim_info(
        claim_text="Revenue was $5B.",
        claim_role=ClaimRole.FACT.value,
        verification_verdict="unsupported",
        position_start=0,
        position_end=16,
    )

    content, removed, softened, rewritten = await pipeline.process_unverified_claims(
        "Revenue was $5B.",
        [claim],
    )

    assert removed == 0
    assert softened == 0
    assert "Revenue was $5B." in content


@pytest.mark.asyncio
async def test_stage8_disposition_soften() -> None:
    """Partial fact claims are softened when disposition says soften."""
    from databricks_deep_research.citation.config import ClaimDisposition, ClaimDispositionConfig

    cfg = CitationConfig(
        claim_disposition=ClaimDispositionConfig(partial=ClaimDisposition.SOFTEN),
    )
    pipeline = _build_pipeline(config=cfg)
    claim = _make_claim_info(
        claim_text="Revenue grew by 20%.",
        claim_role=ClaimRole.FACT.value,
        verification_verdict="partial",
        position_start=0,
        position_end=20,
    )

    content, removed, softened, rewritten = await pipeline.process_unverified_claims(
        "Revenue grew by 20%.",
        [claim],
    )

    assert softened == 1
    assert removed == 0


@pytest.mark.asyncio
async def test_stage8_rewrite_exclusive_with_soften() -> None:
    """A claim set to SOFTEN must NOT also get a rewrite modification."""
    from databricks_deep_research.citation.config import ClaimDisposition, ClaimDispositionConfig

    cfg = CitationConfig(
        claim_disposition=ClaimDispositionConfig(partial=ClaimDisposition.SOFTEN),
    )
    pipeline = _build_pipeline(config=cfg)
    claim = _make_claim_info(
        claim_text="Revenue grew by 20%.",
        claim_role=ClaimRole.FACT.value,
        verification_verdict="partial",
        verification_text="Revenue increased approximately 20%.",
        position_start=0,
        position_end=20,
    )

    content, removed, softened, rewritten = await pipeline.process_unverified_claims(
        "Revenue grew by 20%.",
        [claim],
    )

    # Soften takes precedence — no rewrite should happen
    assert softened == 1
    assert rewritten == 0


@pytest.mark.asyncio
async def test_stage8_rewrite_keeps_claim_cited_and_sentence_safe() -> None:
    """Fact-core rewrites should preserve citations and sentence boundaries."""
    pipeline = _build_pipeline()
    original_sentence = (
        "Gross profit margin reached 67.16% on a trailing twelve-month basis "
        "as of January 31, 2026, reflecting the asset-light delivery model "
        "[Example]."
    )
    content = f"{original_sentence}\n\nNext sentence."
    claim = _make_claim_info(
        claim_text=(
            "Gross profit margin reached 67.16% on a trailing twelve-month "
            "basis as of January 31, 2026, reflecting the asset-light "
            "delivery model."
        ),
        claim_role=ClaimRole.FACT.value,
        claim_type="numeric",
        citation_key="Example",
        citation_keys=["Example"],
        verification_verdict="supported",
        verification_text=(
            "Gross profit margin reached 67.16% on a trailing twelve-month "
            "basis as of January 31, 2026"
        ),
        position_start=0,
        position_end=len(original_sentence),
    )

    content, removed, softened, rewritten = await pipeline.process_unverified_claims(
        content,
        [claim],
    )

    assert removed == 0
    assert softened == 0
    assert rewritten == 1
    assert (
        "Gross profit margin reached 67.16% on a trailing twelve-month basis "
        "as of January 31, 2026 [Example].\n\nNext sentence."
    ) in content
    assert "delivery model" not in content


@pytest.mark.asyncio
async def test_stage8_skips_malformed_dangling_fact_rewrite() -> None:
    """A malformed factual core should not be spliced before the next sentence."""
    pipeline = _build_pipeline()
    original_sentence = (
        "Net revenue retention rate stood at 125%, but still indicating "
        "meaningful expansion [Example]."
    )
    content = f"{original_sentence} By April 30, customers grew."
    claim = _make_claim_info(
        claim_text=(
            "Net revenue retention rate stood at 125%, but still indicating "
            "meaningful expansion."
        ),
        claim_role=ClaimRole.FACT.value,
        claim_type="numeric",
        citation_key="Example",
        citation_keys=["Example"],
        verification_verdict="supported",
        verification_text="Net revenue retention rate stood at 125%, but still",
        position_start=0,
        position_end=len(original_sentence),
    )

    content, removed, softened, rewritten = await pipeline.process_unverified_claims(
        content,
        [claim],
    )

    assert removed == 0
    assert softened == 0
    assert rewritten == 0
    assert "but still By April" not in content
    assert original_sentence in content


@pytest.mark.asyncio
async def test_stage8_analysis_disposition() -> None:
    """analysis_partial and analysis_unsupported overrides work correctly."""
    from databricks_deep_research.citation.config import ClaimDisposition, ClaimDispositionConfig

    cfg = CitationConfig(
        claim_disposition=ClaimDispositionConfig(
            analysis_partial=ClaimDisposition.SOFTEN,
            analysis_unsupported=ClaimDisposition.REMOVE,
        ),
    )
    pipeline = _build_pipeline(config=cfg)
    partial_claim = _make_claim_info(
        claim_text="This suggests growth.",
        claim_role=ClaimRole.ANALYSIS.value,
        verification_verdict="partial",
        position_start=0,
        position_end=20,
    )
    unsupported_claim = _make_claim_info(
        claim_text=" This implies decline.",
        claim_role=ClaimRole.ANALYSIS.value,
        verification_verdict="unsupported",
        position_start=20,
        position_end=42,
    )

    content, removed, softened, rewritten = await pipeline.process_unverified_claims(
        "This suggests growth. This implies decline.",
        [partial_claim, unsupported_claim],
    )

    assert softened == 1
    assert removed == 1


# ---------------------------------------------------------------------------
# T105-9: build_citation_key utility
# ---------------------------------------------------------------------------


def test_build_citation_key_from_url() -> None:
    """build_citation_key should extract domain-based key."""
    pool = [_make_ranked_evidence(source_url="https://arxiv.org/abs/123")]
    key = CitationVerificationPipeline.build_citation_key(
        0, "https://arxiv.org/abs/123", pool
    )
    assert key == "Arxiv"


def test_build_citation_key_duplicate() -> None:
    """build_citation_key should handle duplicates with suffix."""
    pool = [
        _make_ranked_evidence(source_url="https://arxiv.org/abs/1"),
        _make_ranked_evidence(source_url="https://arxiv.org/abs/2"),
    ]
    key = CitationVerificationPipeline.build_citation_key(
        1, "https://arxiv.org/abs/2", pool
    )
    assert key == "Arxiv-2"


# ---------------------------------------------------------------------------
# T105-10: Empty evidence pool emits summary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_evidence_pool_emits_summary() -> None:
    """When no evidence is found, pipeline should emit a skipped summary."""
    selector = _mock_evidence_selector(evidence=[])
    pipeline = _build_pipeline(evidence_selector=selector)
    sources = [{"url": "https://a.com", "content": "stuff " * 100}]

    collected: list[Any] = []
    async for item in pipeline.run_full_pipeline(sources, [], "query"):
        collected.append(item)

    summary_events = [
        e for e in collected
        if isinstance(e, VerificationEvent) and e.event_type == "verification_summary"
    ]
    assert len(summary_events) == 1
    assert summary_events[0].data.get("verification_skipped") is True


# ---------------------------------------------------------------------------
# T105-11: Stage 5 skips PARTIAL claims (Bug A)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stage5_skips_partial_claims() -> None:
    """PARTIAL claims should not be sent to the corrector."""
    corrector = _mock_citation_corrector()
    pipeline = _build_pipeline(citation_corrector=corrector)
    claims = [_make_claim_info(verification_verdict="partial")]
    evidence = [_make_ranked_evidence()]

    events: list[VerificationEvent] = []
    async for event in pipeline.correct_citations(claims, evidence):
        events.append(event)

    # Corrector should not be called since the only claim is PARTIAL
    corrector.correct_citations.assert_not_awaited()
    # Verdict should remain partial (not flipped to unsupported)
    assert claims[0].verification_verdict == "partial"


# ---------------------------------------------------------------------------
# T105-12: Fallback evidence matching (Bug B)
# ---------------------------------------------------------------------------


def test_fallback_evidence_matching() -> None:
    """Uncited claims should get fallback evidence via keyword matching."""
    from databricks_deep_research.citation.pipeline import _assign_fallback_evidence

    claims = [
        _make_claim_info(
            claim_text="Revenue grew by 20% in Q4 2024.",
            evidence=None,
        ),
    ]
    evidence_pool = [
        _make_ranked_evidence(
            quote_text="The company revenue grew by 20% during the fourth quarter of 2024.",
            relevance_score=0.8,
        ),
    ]

    _assign_fallback_evidence(claims, evidence_pool)

    assert claims[0].evidence is not None
    assert claims[0].has_fallback_evidence is True


def test_fallback_uses_context_aware_matching() -> None:
    """Fallback matching should prefer quarter-compatible evidence over lexical noise."""
    from databricks_deep_research.citation.citation_corrector import CitationCorrector
    from databricks_deep_research.citation.pipeline import _assign_fallback_evidence

    claims = [
        _make_claim_info(
            claim_text=(
                "The company generated approximately $1.1 billion in adjusted "
                "operating profit and $1.05 in adjusted EPS in Q3 2025."
            ),
            claim_type="numeric",
            evidence=None,
        ),
    ]
    evidence_pool = [
        _make_ranked_evidence(
            source_url="enterprise://vector_search/main/0",
            quote_text="Adjusted FIFO Operating Profit of $4.7 billion and Adjusted EPS of $4.47",
            source_pool_index=0,
            evidence_pool_index=0,
        ),
        _make_ranked_evidence(
            source_url="enterprise://vector_search/main/3",
            quote_text="Adjusted FIFO Operating Profit of $1,089 million and Adjusted EPS of $1.05 for Q3 2025",
            source_pool_index=3,
            evidence_pool_index=1,
        ),
    ]
    corrector = CitationCorrector(_mock_llm())

    _assign_fallback_evidence(
        claims,
        evidence_pool,
        scorer=corrector.score_claim_evidence,
    )

    assert claims[0].evidence is not None
    assert claims[0].evidence.source_pool_index == 3


@pytest.mark.asyncio
async def test_same_source_replacement_becomes_alternate_citation() -> None:
    """A better span from the same source should not surface as a replacement."""
    from databricks_deep_research.citation.citation_corrector import CitationCorrector

    current_evidence = _make_ranked_evidence(
        source_url="enterprise://vector_search/main/0",
        quote_text="Adjusted EPS was $0.95 in Q2 2025.",
        source_pool_index=0,
        evidence_pool_index=0,
    )
    better_same_source = _make_ranked_evidence(
        source_url="enterprise://vector_search/main/0",
        quote_text="Adjusted EPS was $1.05 in Q3 2025.",
        source_pool_index=0,
        evidence_pool_index=1,
    )
    corrector = CitationCorrector(_mock_llm())

    result = await corrector.correct_single_citation(
        claim="Adjusted EPS was $1.05 in Q3 2025.",
        current_evidence=current_evidence,
        evidence_pool=[current_evidence, better_same_source],
        current_verdict="unsupported",
    )

    assert result.correction_type == CorrectionAction.ADD_ALTERNATE
    assert result.corrected_evidence is current_evidence
    assert result.alternate_evidence == [better_same_source]


def test_fallback_respects_threshold() -> None:
    """Claims with zero keyword overlap should stay without evidence."""
    from databricks_deep_research.citation.pipeline import _assign_fallback_evidence

    claims = [
        _make_claim_info(
            claim_text="Quantum entanglement enables faster communication.",
            evidence=None,
        ),
    ]
    evidence_pool = [
        _make_ranked_evidence(
            quote_text="The price of bananas increased in January.",
            relevance_score=0.3,
        ),
    ]

    _assign_fallback_evidence(claims, evidence_pool)

    assert claims[0].evidence is None


def test_softened_fact_text_lowercases_leading_article() -> None:
    """Fact softening should avoid grammatically awkward hedge prefixes."""
    from databricks_deep_research.citation.pipeline import _build_softened_fact_text

    softened = _build_softened_fact_text(
        "The first quarter of 2025 marked a strong start.",
        None,
    )

    assert " that The first quarter" not in softened
    assert " that the first quarter" in softened or " information, the first quarter" in softened or softened.startswith("Reportedly, the")


# ---------------------------------------------------------------------------
# T105-13: Snippet relevance differentiation (Bug C)
# ---------------------------------------------------------------------------


def test_snippet_relevance_differentiation() -> None:
    """Snippet relevance scores should vary, not all be 0.5."""
    from databricks_deep_research.citation.evidence_selector import _keyword_relevance

    high_overlap = _keyword_relevance(
        "company revenue growth 2024", "The company revenue growth exceeded 2024 targets"
    )
    low_overlap = _keyword_relevance(
        "company revenue growth 2024", "Bananas are a popular fruit."
    )

    # High overlap should be significantly higher than low overlap
    assert high_overlap > low_overlap
    # Floor at 0.3 means snippet-only scores won't all collapse to 0.5
    assert low_overlap < 0.5


# ---------------------------------------------------------------------------
# T105-14: Position tracking keeps original on miss (Bug D)
# ---------------------------------------------------------------------------


def test_position_tracking_keeps_original_on_miss() -> None:
    """Positions should be preserved when exact match fails."""
    from databricks_deep_research.citation.pipeline import _recalculate_claim_positions

    claim = _make_claim_info(
        claim_text="This exact text is not in the content.",
        position_start=100,
        position_end=140,
    )
    content = "Some completely different content that has nothing to do with the claim."

    _recalculate_claim_positions(content, [claim])

    # Original positions preserved — no regex fallback corruption
    assert claim.position_start == 100
    assert claim.position_end == 140


# ---------------------------------------------------------------------------
# T105-15: Cross-source evidence deduplication
# ---------------------------------------------------------------------------


def test_dedup_evidence_cross_source() -> None:
    """Near-duplicate evidence with same prefix should be deduped; different content preserved."""
    from databricks_deep_research.citation.evidence_selector import _dedup_evidence_cross_source

    shared_prefix = "Databricks provides a unified analytics platform " * 5  # ~250 chars
    ev1 = _make_ranked_evidence(
        quote_text=shared_prefix,
        source_url="https://a.com",
        relevance_score=0.8,
    )
    ev2 = _make_ranked_evidence(
        quote_text=shared_prefix,
        source_url="https://b.com",
        relevance_score=0.9,
    )
    ev3 = _make_ranked_evidence(
        quote_text="Completely different evidence about machine learning.",
        source_url="https://c.com",
        relevance_score=0.7,
    )

    result = _dedup_evidence_cross_source([ev1, ev2, ev3])

    # Two duplicates → one kept (higher score), plus the unique one
    assert len(result) == 2
    # Higher-scoring duplicate kept
    urls = {e.source_url for e in result}
    assert "https://b.com" in urls
    assert "https://c.com" in urls
