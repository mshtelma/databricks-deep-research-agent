"""Citation Verification Pipeline -- Orchestrates the 7-stage citation verification.

This is the **framework** port of the app's ``CitationVerificationPipeline``.
It orchestrates the same 7 stages but depends only on framework types -- no
``deep_research.*`` (app) imports.

Stages:
    1. Evidence Pre-Selection   -- rank evidence spans from sources
    2. Interleaved Generation   -- generate claims constrained by evidence
    3. Confidence Classification -- route claims by confidence level
    4. Isolated Verification    -- produce verdicts per claim
    5. Citation Correction      -- swap citations from evidence pool
    6. Numeric QA Verification  -- deep verification for numeric claims
    7. ARE Verification Retrieval -- atomic fact decomposition + external search

The pipeline powers both the strict interleaved reclaim lane and the
post-synthesis `classical_lite` grounding lane. In reclaim mode it generates
claims interleaved with evidence; in classical_lite mode it parses an existing
draft into claims and then reuses the same downstream verification stages.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import Counter
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlparse

from databricks_deep_research.citation.citation_keys import build_citation_key_map
from databricks_deep_research.citation.claim_classifier import (
    classify_claim_role,
    contains_material_analysis,
    extract_factual_core,
)
from databricks_deep_research.citation.confidence_classifier import (
    confidence_score_from_level as _confidence_score_from_level,
)
from databricks_deep_research.citation.confidence_classifier import (
    extract_numeric_tokens,
    extract_temporal_tokens,
    quote_overlap_score,
)
from databricks_deep_research.citation.config import (
    CitationConfig,
    ClaimDisposition,
    SynthesisMode,
)
from databricks_deep_research.citation.types import (
    AnalysisSummaryInfo,
    ClaimInfo,
    ClaimRole,
    ConfidenceLevel,
    ConfidenceResult,
    ContentQuality,
    CorrectionAction,
    CorrectionMetrics,
    CorrectionResult,
    EvidenceInfo,
    InterleavedClaim,
    NumericVerificationResult,
    RankedEvidence,
    VerificationMethod,
    VerificationResult,
    VerificationSummaryInfo,
    VerificationVerdict,
)
from databricks_deep_research.citation.utils import truncate as _truncate
from databricks_deep_research.llm.client import FrameworkLLMClient
from databricks_deep_research.tracing import trace_span

logger = logging.getLogger(__name__)


# Module-level monotonic counter so multiple invocations of
# ``process_unverified_claims`` within a single run can be distinguished in
# the trace logs (Stage 8 of the citation pipeline). PR3-0 instrumentation.
_STAGE8_PASS_COUNTER = 0


def _next_stage8_pass_id() -> int:
    global _STAGE8_PASS_COUNTER
    _STAGE8_PASS_COUNTER += 1
    return _STAGE8_PASS_COUNTER


# PR3-E R2.2: feature flag controlling the is_negative_existence classifier
# + force-REMOVE rule. Default off; flip to "true" only after the
# investment_research regression bar passes (per plan AC7).
def _synth_pipeline_v2_enabled() -> bool:
    import os

    return os.environ.get("SYNTH_PIPELINE_V2", "").lower() in ("true", "1", "yes")


async def _classify_negative_existence_batch(
    claims: list["ClaimInfo"],
    llm_client: Any,
) -> None:
    """Set ``claim.is_negative_existence`` on eligible claims via the
    Haiku-tier classifier.

    PR3-E R2.2 wiring. Iterates claims with non-fully-supported verdicts
    (abstained/unsupported/contradicted/partial) and consults
    ``classify_negative_existence``. Mutates each claim in place;
    failures and supported-verdict claims are no-ops. Latency budget
    per plan: ≤30s sequential on ~25 claims.
    """
    from databricks_deep_research.citation.claim_classifier import (
        classify_negative_existence,
    )

    eligible = [
        c
        for c in claims
        if c.abstained
        or (c.verification_verdict or "").lower()
        in ("abstained", "unsupported", "contradicted", "partial")
    ]
    logger.info(
        "DR_NEGATIVE_EXISTENCE_CLASSIFIER_RUN eligible=%d total=%d",
        len(eligible),
        len(claims),
    )
    for claim in eligible:
        try:
            flag, reasoning = await classify_negative_existence(
                claim, llm_client
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.info(
                "DR_NEGATIVE_EXISTENCE_CLASSIFIER_FAIL claim_head=%r error=%s",
                claim.claim_text[:60],
                exc,
            )
            continue
        if flag:
            claim.is_negative_existence = True
            claim.is_negative_existence_reasoning = reasoning


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _counter_dict(values: list[str]) -> dict[str, int]:
    """Return a stable string counter for debug logging."""
    return dict(sorted(Counter(value for value in values if value).items()))


def _evidence_info_from_ranked(ranked: RankedEvidence) -> EvidenceInfo:
    """Convert RankedEvidence to EvidenceInfo (lossless field copy)."""
    return EvidenceInfo(
        source_url=ranked.source_url or "",
        canonical_source_url=ranked.canonical_source_url,
        source_title=ranked.source_title,
        quote_text=ranked.quote_text,
        start_offset=ranked.start_offset,
        end_offset=ranked.end_offset,
        section_heading=ranked.section_heading,
        relevance_score=ranked.relevance_score,
        has_numeric_content=ranked.has_numeric_content,
        source_pool_index=ranked.source_pool_index,
        evidence_pool_index=ranked.evidence_pool_index,
    )


def _claim_info_from_interleaved(claim: InterleavedClaim) -> ClaimInfo:
    """Convert InterleavedClaim to ClaimInfo with evidence conversion."""
    return ClaimInfo(
        claim_text=claim.claim_text,
        claim_type=claim.claim_type,
        position_start=claim.position_start,
        position_end=claim.position_end,
        evidence=(
            _evidence_info_from_ranked(claim.evidence) if claim.evidence else None
        ),
        evidences=[_evidence_info_from_ranked(e) for e in claim.evidences],
        citation_key=claim.citation_key,
        citation_keys=claim.citation_keys,
        claim_role=claim.claim_role,
        verification_text=claim.verification_text,
        analysis_parent_claim_indices=claim.analysis_parent_claim_indices,
        from_free_block=claim.from_free_block,
    )


# ---------------------------------------------------------------------------
# Pipeline events
# ---------------------------------------------------------------------------


@dataclass
class VerificationEvent:
    """Event emitted during citation verification."""

    event_type: str  # claim_generated, claim_verified, citation_corrected, etc.
    data: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------
# The pipeline uses ``CitationConfig`` from ``citation.config``.  All stage
# toggles, sub-configs and thresholds live there.  The pipeline constructor
# accepts an optional ``CitationConfig``; when ``None`` the default is used.


# ---------------------------------------------------------------------------
# Stage protocols -- thin interfaces so stages can be swapped / mocked.
# ---------------------------------------------------------------------------


@runtime_checkable
class EvidenceSelector(Protocol):
    """Stage 1: selects evidence spans from source documents."""

    async def select_evidence_spans(
        self,
        query: str,
        sources: list[dict[str, Any]],
        max_spans_per_source: int,
    ) -> list[RankedEvidence]: ...


@runtime_checkable
class ClaimGenerator(Protocol):
    """Stage 2: generates claims interleaved with evidence."""

    def synthesize_with_streaming(
        self,
        query: str,
        evidence_pool: list[RankedEvidence],
        previous_content: str = ...,
        target_word_count: int = ...,
        max_tokens: int = ...,
        generation_instructions: str = ...,
    ) -> AsyncGenerator[tuple[str, InterleavedClaim | None], None]: ...


@runtime_checkable
class ConfidenceClassifierProtocol(Protocol):
    """Stage 3: classifies claim confidence level."""

    def classify(
        self, claim_text: str, evidence_quote: str | None
    ) -> ConfidenceResult: ...

    def should_use_quick_verification(
        self, claim_text: str, evidence_quote: str | None
    ) -> bool: ...


@runtime_checkable
class IsolatedVerifierProtocol(Protocol):
    """Stage 4: verifies a claim against evidence in isolation."""

    async def verify_with_isolation(
        self,
        claim_text: str,
        evidence: RankedEvidence,
        *,
        use_quick_verification: bool = ...,
    ) -> VerificationResult: ...


@runtime_checkable
class CitationCorrectorProtocol(Protocol):
    """Stage 5: corrects citations from the evidence pool."""

    async def correct_citations(
        self,
        claims_with_evidence: list[
            tuple[str, RankedEvidence | None, str | None]
        ],
        evidence_pool: list[RankedEvidence],
    ) -> tuple[list[CorrectionResult], CorrectionMetrics]: ...


@runtime_checkable
class NumericVerifierProtocol(Protocol):
    """Stage 6: verifies numeric claims using QA approach."""

    async def verify_numeric_claim(
        self, claim_text: str, evidence: RankedEvidence
    ) -> NumericVerificationResult: ...


@runtime_checkable
class AnalysisGroundingVerifierProtocol(Protocol):
    """Analysis-grounding verifier used for reclaim analysis blocks."""

    async def verify_analysis_claim(
        self,
        claim_text: str,
        supporting_claims: list[str],
        evidences: list[RankedEvidence],
        supporting_fact_contexts: list[dict[str, str]] | None = None,
    ) -> VerificationResult: ...


@runtime_checkable
class VerificationRetrieverProtocol(Protocol):
    """Stage 7: revises claims after atomic verification retrieval."""

    def retrieve_and_revise(
        self,
        claims: list[ClaimInfo],
        evidence_pool: list[RankedEvidence],
        report_content: str,
        research_query: str,
    ) -> AsyncGenerator[Any, None]: ...

    def apply_all_revisions(
        self,
        report: str,
        revisions: list[Any],
    ) -> str: ...


@runtime_checkable
class ContentQualityEvaluator(Protocol):
    """Evaluates crawled-content quality for pre-Stage-1 filtering."""

    def __call__(self, content: str, query: str) -> ContentQuality: ...


# ---------------------------------------------------------------------------
# Search / crawl tool protocols for Stage 7
# ---------------------------------------------------------------------------


@runtime_checkable
class SearchClient(Protocol):
    """Minimal web-search protocol used by Stage 7."""

    async def search(self, query: str, count: int = 5) -> list[dict[str, Any]]: ...


@runtime_checkable
class WebCrawlerProtocol(Protocol):
    """Minimal web-crawler protocol used by Stage 7."""

    async def crawl(self, url: str) -> str: ...


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


class CitationVerificationPipeline:
    """Orchestrates the 7-stage citation verification pipeline.

    Integrates with a synthesizer to provide claim-level attribution:
    - Pre-selects evidence from sources before generation
    - Generates claims constrained by available evidence
    - Verifies each claim in isolation
    - Corrects citations if needed
    - Verifies numeric claims with QA approach
    - Revises unsupported claims using ARE pattern (Stage 7)
    """

    def __init__(
        self,
        llm: FrameworkLLMClient,
        *,
        evidence_selector: EvidenceSelector,
        claim_generator: ClaimGenerator,
        confidence_classifier: ConfidenceClassifierProtocol,
        isolated_verifier: IsolatedVerifierProtocol,
        citation_corrector: CitationCorrectorProtocol,
        numeric_verifier: NumericVerifierProtocol,
        analysis_grounding_verifier: AnalysisGroundingVerifierProtocol | None = None,
        verification_retriever: VerificationRetrieverProtocol | None = None,
        content_quality_evaluator: ContentQualityEvaluator | None = None,
        search_client: SearchClient | None = None,
        web_crawler: WebCrawlerProtocol | None = None,
        config: CitationConfig | None = None,
    ) -> None:
        """Initialise the pipeline.

        Args:
            llm: Framework LLM client for model calls.
            evidence_selector: Stage 1 component.
            claim_generator: Stage 2 component.
            confidence_classifier: Stage 3 component.
            isolated_verifier: Stage 4 component.
            citation_corrector: Stage 5 component.
            numeric_verifier: Stage 6 component.
            content_quality_evaluator: Optional pre-Stage-1 quality filter.
            search_client: Optional search client for Stage 7.
            web_crawler: Optional crawler for Stage 7.
            config: Pipeline configuration.  Uses defaults when ``None``.
        """
        self.llm = llm
        self.config = config or CitationConfig()

        # Stage components
        self.evidence_selector = evidence_selector
        self.claim_generator = claim_generator
        self.confidence_classifier = confidence_classifier
        self.verifier = isolated_verifier
        self.corrector = citation_corrector
        self.numeric_verifier = numeric_verifier
        self.analysis_grounding_verifier = analysis_grounding_verifier
        self.verification_retriever = verification_retriever
        self.content_quality_evaluator = content_quality_evaluator

        # Stage 7 dependencies
        self.search_client = search_client
        self.web_crawler = web_crawler

        # Last-run artifacts for framework executors and tests.
        self.last_evidence_pool: list[RankedEvidence] = []
        self.last_generated_claims: list[ClaimInfo] = []
        self.last_verification_summary: VerificationSummaryInfo | None = None
        self.last_final_content: str = ""
        self.last_routing_summary: dict[str, Any] = {}
        self._active_claims_context: list[ClaimInfo] = []
        self._verification_route_stats: list[dict[str, Any]] = []

    # ===================================================================
    # Stage 1: Evidence Pre-Selection
    # ===================================================================

    async def preselect_evidence(
        self,
        sources: list[dict[str, Any]],
        query: str,
    ) -> list[RankedEvidence]:
        """Stage 1: Pre-select evidence spans from sources.

        Args:
            sources: List of source dicts with keys ``url``, ``title``,
                     ``content``, ``snippet``, ``source_type``.
            query: Research query for relevance scoring.

        Returns:
            Ranked evidence spans, sorted by relevance.
        """
        if not self.config.enable_evidence_preselection:
            logger.info("CITATION_PIPELINE stage=1 action=skipped reason=disabled")
            return []

        logger.info(
            "CITATION_PIPELINE_STAGE1 sources_count=%d query=%s",
            len(sources),
            _truncate(query, 50),
        )

        indexed_sources: list[dict[str, Any]] = []
        for source_index, source in enumerate(sources):
            normalized_source = dict(source)
            normalized_source.setdefault("source_pool_index", source_index)
            normalized_source.setdefault(
                "canonical_url",
                str(source.get("canonical_url") or source.get("url") or ""),
            )
            indexed_sources.append(normalized_source)

        # Filter by content / snippet availability
        usable_sources = [
            s for s in indexed_sources if s.get("content") or s.get("snippet")
        ]
        if not usable_sources:
            logger.warning("CITATION_PIPELINE_STAGE1 no_sources_with_content_or_snippet")
            return []

        # Content quality filtering
        high_quality_sources = self._filter_source_quality(usable_sources, query)
        if not high_quality_sources:
            logger.warning(
                "CITATION_PIPELINE_STAGE1 no_high_quality_sources total=%d",
                len(usable_sources),
            )
            high_quality_sources = usable_sources  # fall back to all

        logger.info(
            "CITATION_PIPELINE_STAGE1_QUALITY_FILTER total=%d kept=%d",
            len(usable_sources),
            len(high_quality_sources),
        )

        max_spans = self.config.evidence_preselection.max_spans_per_source

        try:
            all_evidence = await self.evidence_selector.select_evidence_spans(
                query=query,
                sources=high_quality_sources,
                max_spans_per_source=max_spans,
            )
        except Exception:
            logger.warning("CITATION_PIPELINE_STAGE1_ERROR", exc_info=True)
            return []

        # Sort by relevance and cap to configured maximum
        all_evidence.sort(key=lambda e: e.relevance_score, reverse=True)
        pre_cap_count = len(all_evidence)
        max_total = self.config.interleaved_generation.max_evidence_spans
        all_evidence = all_evidence[:max_total]
        if len(all_evidence) < pre_cap_count:
            logger.info(
                "EVIDENCE_POOL_CAPPED pre=%d post=%d cap=%d",
                pre_cap_count, len(all_evidence), max_total,
            )

        for evidence_index, evidence in enumerate(all_evidence):
            if getattr(evidence, "canonical_source_url", None) is None:
                evidence.canonical_source_url = evidence.source_url
            if getattr(evidence, "source_pool_index", None) is None:
                evidence.source_pool_index = self._resolve_source_pool_index(
                    evidence.source_url,
                    indexed_sources,
                    getattr(evidence, "canonical_source_url", None),
                )
            evidence.evidence_pool_index = evidence_index

        logger.info("CITATION_PIPELINE_STAGE1_COMPLETE total_evidence=%d", len(all_evidence))
        return all_evidence

    def _filter_source_quality(
        self,
        sources: list[dict[str, Any]],
        query: str,
    ) -> list[dict[str, Any]]:
        """Apply content quality filtering to sources.

        Enterprise sources and snippet-only sources bypass quality filtering.
        """
        evaluator = self.content_quality_evaluator
        threshold = self.config.evidence_preselection.relevance_threshold
        kept: list[dict[str, Any]] = []

        for source in sources:
            content = source.get("content") or ""
            snippet = source.get("snippet") or ""
            source_type = source.get("source_type", "web")

            # Enterprise sources are authoritative -- bypass quality filter
            if source_type in ("genie", "vector_search", "knowledge_assistant"):
                if content or snippet:
                    kept.append(source)
                continue

            # Snippet-only sources: accept without quality evaluation
            if not content and snippet:
                kept.append(source)
                continue

            # Full content: evaluate quality if evaluator is available
            if evaluator is not None:
                quality = evaluator(content, query)
                if quality.score >= threshold and not quality.is_abstract_only:
                    kept.append(source)
                else:
                    logger.debug(
                        "SOURCE_QUALITY_REJECTED url=%s score=%.2f reason=%s",
                        _truncate(source.get("url", ""), 50),
                        quality.score,
                        quality.reason,
                    )
            else:
                # No evaluator -- accept all sources with content
                kept.append(source)

        return kept

    # ===================================================================
    # Stage 2: Interleaved Generation
    # ===================================================================

    async def generate_with_interleaving(
        self,
        evidence_pool: list[RankedEvidence],
        observations: list[str],
        query: str,
        target_word_count: int = 600,
        max_tokens: int = 2000,
        generation_instructions: str = "",
    ) -> AsyncGenerator[tuple[str, InterleavedClaim | None], None]:
        """Stage 2: Generate synthesis with interleaved claim/evidence pairs.

        Yields:
            ``(content_chunk, claim_or_none)`` tuples.
        """
        if not self.config.enable_interleaved_generation:
            logger.info("CITATION_PIPELINE stage=2 action=skipped reason=disabled")
            return

        logger.info(
            "CITATION_PIPELINE_STAGE2_START evidence=%d observations=%d",
            len(evidence_pool),
            len(observations),
        )

        previous_content = "\n\n".join(observations) if observations else ""
        claim_index = 0

        async for content, claim in self.claim_generator.synthesize_with_streaming(
            query=query,
            evidence_pool=evidence_pool,
            previous_content=previous_content,
            target_word_count=target_word_count,
            max_tokens=max_tokens,
            generation_instructions=generation_instructions,
        ):
            if content:
                yield content, None
            if claim:
                claim_index += 1
                logger.debug(
                    "CLAIM_GENERATED index=%d text=%s",
                    claim_index,
                    _truncate(claim.claim_text, 50),
                )
                yield "", claim

    # ===================================================================
    # Stage 3: Confidence Classification
    # ===================================================================

    def _score_claim_evidence_text(
        self,
        claim_text: str,
        evidence_quote: str | None,
    ) -> float:
        """Compute a deterministic claim/evidence match score when available."""
        if not evidence_quote:
            return 0.0

        score_method = getattr(self.corrector, "score_claim_evidence", None)
        if callable(score_method):
            try:
                return float(score_method(claim_text, evidence_quote))
            except Exception:
                logger.debug("CLAIM_EVIDENCE_SCORE_FAILED", exc_info=True)

        overlap = self.confidence_classifier.classify(
            claim_text,
            evidence_quote,
        )
        return overlap.score

    def _has_exact_numeric_support(
        self,
        claim: ClaimInfo,
        evidence_quote: str | None,
    ) -> bool:
        """Return ``True`` when a numeric claim directly restates the evidence."""
        if claim.claim_type != "numeric" or not evidence_quote:
            return False

        claim_numbers = extract_numeric_tokens(claim.claim_text)
        evidence_numbers = extract_numeric_tokens(evidence_quote)
        if not claim_numbers or not claim_numbers.issubset(evidence_numbers):
            return False

        claim_temporal = extract_temporal_tokens(claim.claim_text)
        evidence_temporal = extract_temporal_tokens(evidence_quote)
        if claim_temporal and evidence_temporal:
            return not claim_temporal.isdisjoint(evidence_temporal)
        return True

    def _deterministic_confidence_result(
        self,
        claim: ClaimInfo,
        evidence_quote: str | None,
    ) -> ConfidenceResult | None:
        """Apply the fast deterministic routing policy before heuristics."""
        claim_text = claim.verification_text or claim.claim_text
        lowered = claim_text.lower()

        if claim.claim_role == ClaimRole.ANALYSIS.value:
            return ConfidenceResult(
                level=ConfidenceLevel.LOW,
                score=0.2,
                indicators=["analysis_lane"],
                reasoning="Analysis claims always use the full grounding verifier.",
            )

        if claim.claim_role == ClaimRole.FREE.value:
            return ConfidenceResult(
                level=ConfidenceLevel.LOW,
                score=0.0,
                indicators=["free_block"],
                reasoning="Structural free blocks do not use factual quick verification.",
            )

        if not evidence_quote:
            return ConfidenceResult(
                level=ConfidenceLevel.LOW,
                score=0.3,
                indicators=["missing_evidence"],
                reasoning="Claim has no evidence attached, so it must use full verification.",
            )

        evidence_match_score = self._score_claim_evidence_text(claim_text, evidence_quote)
        quote_overlap = quote_overlap_score(claim_text, evidence_quote)

        if self._has_exact_numeric_support(claim, evidence_quote):
            return ConfidenceResult(
                level=ConfidenceLevel.HIGH,
                score=0.95,
                indicators=["exact_numeric_match"],
                reasoning="Numeric claim exactly matches the cited evidence.",
            )

        if contains_material_analysis(lowered):
            level = ConfidenceLevel.LOW if evidence_match_score < 0.75 else ConfidenceLevel.MEDIUM
            score = 0.35 if level == ConfidenceLevel.LOW else 0.55
            return ConfidenceResult(
                level=level,
                score=score,
                indicators=["material_analysis_language"],
                reasoning="Claim includes interpretive or comparative language that requires full verification.",
            )

        if evidence_match_score >= 0.82 and quote_overlap >= 0.45:
            return ConfidenceResult(
                level=ConfidenceLevel.HIGH,
                score=max(0.75, min(0.92, evidence_match_score)),
                indicators=["strong_evidence_match", f"quote_overlap:{quote_overlap:.2f}"],
                reasoning="Claim strongly overlaps with cited evidence and can use quick verification.",
            )

        if any(marker in lowered for marker in ("reportedly", "according to", "some sources indicate")):
            return ConfidenceResult(
                level=ConfidenceLevel.LOW,
                score=0.35,
                indicators=["hedged_fact"],
                reasoning="Hedged factual claim should use full verification.",
            )

        return None

    def classify_confidence_result(self, claim: ClaimInfo) -> ConfidenceResult:
        """Stage 3: Classify confidence level using heuristics."""
        if not self.config.enable_confidence_classification:
            return ConfidenceResult(
                level=ConfidenceLevel.MEDIUM,
                score=0.6,
                indicators=[],
                reasoning="Confidence classification disabled; defaulted to medium.",
            )

        evidence_quote = claim.evidence.quote_text if claim.evidence else None
        deterministic = self._deterministic_confidence_result(claim, evidence_quote)
        if deterministic is not None:
            return deterministic
        return self.confidence_classifier.classify(
            claim.verification_text or claim.claim_text,
            evidence_quote,
        )

    def classify_confidence(self, claim: ClaimInfo) -> str:
        """Return the qualitative routing confidence for a claim."""
        if not self.config.enable_confidence_classification:
            return "medium"
        return self.classify_confidence_result(claim).level.value

    # ===================================================================
    # Stage 4: Isolated Verification
    # ===================================================================

    @staticmethod
    def _claim_evidences(claim: ClaimInfo) -> list[EvidenceInfo]:
        """Return all evidence references attached to a claim."""
        evidences = list(claim.evidences)
        if claim.evidence and not evidences:
            evidences.append(claim.evidence)
        if claim.evidence and all(
            evidence.quote_text != claim.evidence.quote_text
            or evidence.source_url != claim.evidence.source_url
            for evidence in evidences
        ):
            evidences.insert(0, claim.evidence)
        return evidences

    @staticmethod
    def _ranked_evidences_for_claim(claim: ClaimInfo) -> list[RankedEvidence]:
        """Convert a claim's evidence references into ranked evidence spans."""
        ranked: list[RankedEvidence] = []
        for evidence in CitationVerificationPipeline._claim_evidences(claim):
            ranked.append(
                RankedEvidence(
                    source_id=None,
                    source_url=evidence.source_url,
                    canonical_source_url=evidence.canonical_source_url,
                    source_title=evidence.source_title,
                    quote_text=evidence.quote_text,
                    start_offset=evidence.start_offset,
                    end_offset=evidence.end_offset,
                    section_heading=evidence.section_heading,
                    relevance_score=evidence.relevance_score or 0.0,
                    has_numeric_content=evidence.has_numeric_content,
                    source_pool_index=evidence.source_pool_index,
                    evidence_pool_index=evidence.evidence_pool_index,
                )
            )
        return ranked

    @staticmethod
    def _merge_evidence_for_verification(
        evidences: list[RankedEvidence],
    ) -> RankedEvidence | None:
        """Merge multiple cited evidence spans into one verifier input."""
        if not evidences:
            return None
        if len(evidences) == 1:
            return evidences[0]

        merged_quote = "\n\n".join(
            f"Source {index + 1}: {evidence.quote_text}"
            for index, evidence in enumerate(evidences)
            if evidence.quote_text
        )
        return RankedEvidence(
            source_id=None,
            source_url=evidences[0].source_url,
            canonical_source_url=evidences[0].canonical_source_url,
            source_title="Multiple cited sources",
            quote_text=merged_quote,
            start_offset=evidences[0].start_offset,
            end_offset=evidences[-1].end_offset,
            section_heading=None,
            relevance_score=(
                sum(evidence.relevance_score for evidence in evidences) / len(evidences)
            ),
            has_numeric_content=any(
                evidence.has_numeric_content for evidence in evidences
            ),
            source_pool_index=evidences[0].source_pool_index,
        )

    def _link_analysis_claims(self, claims: list[ClaimInfo]) -> None:
        """Link analysis claims to nearby fact claims they interpret."""
        fact_indices = [
            index for index, claim in enumerate(claims)
            if claim.claim_role == ClaimRole.FACT.value
        ]
        max_preceding = self.config.grounding_validation.max_preceding_citations

        for index, claim in enumerate(claims):
            if claim.claim_role != ClaimRole.ANALYSIS.value:
                claim.analysis_parent_claim_indices = []
                continue

            claim_keys = set(claim.citation_keys or ([] if claim.citation_key is None else [claim.citation_key]))
            parents: list[int] = []
            for fact_index in reversed(fact_indices):
                if fact_index >= index:
                    continue
                fact_claim = claims[fact_index]
                fact_keys = set(
                    fact_claim.citation_keys
                    or ([] if fact_claim.citation_key is None else [fact_claim.citation_key])
                )
                if claim_keys and fact_keys and claim_keys.isdisjoint(fact_keys):
                    continue
                parents.append(fact_index)
                if len(parents) >= max_preceding:
                    break

            if not parents:
                parents = [fact_index for fact_index in fact_indices if fact_index < index][-3:]

            claim.analysis_parent_claim_indices = sorted(parents)

    def _classify_and_link_claims(self, claims: list[ClaimInfo]) -> None:
        """Stage 2.1: enforce fact/analysis/free boundaries before verification."""
        for claim in claims:
            claim.claim_role = classify_claim_role(claim)
            claim.verification_text = None
            claim.analysis_parent_claim_indices = []
            claim.abstained = False
            if claim.claim_role == ClaimRole.FREE.value:
                claim.verification_method = VerificationMethod.STRUCTURAL.value
                claim.abstained = True
                continue

            if claim.claim_role == ClaimRole.ANALYSIS.value:
                claim.verification_method = VerificationMethod.GROUNDING.value
                continue

            factual_core = extract_factual_core(claim.claim_text)
            if factual_core and factual_core != claim.claim_text:
                claim.verification_text = factual_core
            claim.verification_method = (
                VerificationMethod.NUMERIC_QA.value
                if claim.claim_type == "numeric"
                else VerificationMethod.ENTAILMENT.value
            )

        self._link_analysis_claims(claims)
        logger.info(
            "CLAIM_BOUNDARY_ENFORCED total=%d roles=%s methods=%s linked_analysis=%d "
            "with_verification_text=%d",
            len(claims),
            _counter_dict([claim.claim_role for claim in claims]),
            _counter_dict(
                [claim.verification_method or "" for claim in claims if claim.verification_method]
            ),
            sum(
                1 for claim in claims
                if claim.claim_role == ClaimRole.ANALYSIS.value
                and claim.analysis_parent_claim_indices
            ),
            sum(1 for claim in claims if claim.verification_text),
        )
        for claim_index, claim in enumerate(claims):
            logger.debug(
                "CLAIM_BOUNDARY claim_index=%d role=%s type=%s method=%s citations=%s "
                "parents=%s verification_text=%s",
                claim_index,
                claim.claim_role,
                claim.claim_type,
                claim.verification_method,
                claim.citation_keys or ([claim.citation_key] if claim.citation_key else []),
                claim.analysis_parent_claim_indices,
                _truncate(claim.verification_text or "", 80),
            )

    def _analysis_support_for_claim(
        self,
        claim_index: int,
        claim: ClaimInfo,
    ) -> tuple[list[ClaimInfo], list[str], list[RankedEvidence]]:
        """Collect grounded fact claims and evidence for an analysis claim."""
        supporting_facts: list[ClaimInfo] = []
        supporting_claims: list[str] = []
        evidences: list[RankedEvidence] = []
        seen_evidence: set[tuple[str, str]] = set()

        for parent_index in claim.analysis_parent_claim_indices:
            if not 0 <= parent_index < len(self._active_claims_context):
                continue
            parent = self._active_claims_context[parent_index]
            if parent.claim_role != ClaimRole.FACT.value:
                continue
            if parent.verification_verdict not in ("supported", "partial"):
                continue
            supporting_facts.append(parent)
            supporting_claims.append(parent.verification_text or parent.claim_text)
            for evidence in self._ranked_evidences_for_claim(parent):
                fingerprint = (evidence.source_url, evidence.quote_text)
                if fingerprint in seen_evidence:
                    continue
                seen_evidence.add(fingerprint)
                evidences.append(evidence)

        for evidence in self._ranked_evidences_for_claim(claim):
            fingerprint = (evidence.source_url, evidence.quote_text)
            if fingerprint in seen_evidence:
                continue
            seen_evidence.add(fingerprint)
            evidences.append(evidence)

        # Fall back to nearby facts if the analysis block was not explicitly linked.
        if not supporting_claims:
            for previous in range(claim_index - 1, -1, -1):
                candidate = self._active_claims_context[previous]
                if candidate.claim_role != ClaimRole.FACT.value:
                    continue
                if candidate.verification_verdict not in ("supported", "partial"):
                    continue
                supporting_facts.append(candidate)
                supporting_claims.append(candidate.verification_text or candidate.claim_text)
                for evidence in self._ranked_evidences_for_claim(candidate):
                    fingerprint = (evidence.source_url, evidence.quote_text)
                    if fingerprint in seen_evidence:
                        continue
                    seen_evidence.add(fingerprint)
                    evidences.append(evidence)
                if len(supporting_claims) >= 3:
                    break

        logger.debug(
            "ANALYSIS_SUPPORT claim_index=%d parents=%s supporting_facts=%d evidences=%d",
            claim_index,
            claim.analysis_parent_claim_indices,
            len(supporting_claims),
            len(evidences),
        )
        return supporting_facts, supporting_claims, evidences

    async def _verify_claim_once(
        self,
        claim_index: int,
        claim: ClaimInfo,
        *,
        include_numeric_verification: bool = True,
    ) -> list[VerificationEvent]:
        """Verify one claim and return the events emitted for it."""
        if claim.claim_role == ClaimRole.FREE.value:
            claim.verification_method = VerificationMethod.STRUCTURAL.value
            claim.abstained = True
            return []

        verification_text = claim.verification_text or claim.claim_text
        ranked_evidences = self._ranked_evidences_for_claim(claim)
        verification_evidence = self._merge_evidence_for_verification(ranked_evidences)
        analysis_evidences: list[RankedEvidence] = []
        numeric_result: dict[str, Any] = {}
        used_quick = False
        evidence_match_score = 0.0
        verification_started = time.monotonic()
        logger.debug(
            "CLAIM_VERIFY_START index=%d role=%s type=%s route=%s method=%s "
            "evidence_count=%d fallback=%s citations=%s text=%s",
            claim_index,
            claim.claim_role,
            claim.claim_type,
            claim.confidence_level,
            claim.verification_method,
            len(ranked_evidences),
            claim.has_fallback_evidence,
            claim.citation_keys or ([claim.citation_key] if claim.citation_key else []),
            _truncate(verification_text, 120),
        )

        if claim.claim_role == ClaimRole.ANALYSIS.value:
            supporting_facts, supporting_claims, analysis_evidences = self._analysis_support_for_claim(
                claim_index,
                claim,
            )
            verification_evidence = self._merge_evidence_for_verification(analysis_evidences)
            if analysis_evidences:
                evidence_match_score = sum(
                    self._score_claim_evidence_text(claim.claim_text, evidence.quote_text)
                    for evidence in analysis_evidences
                ) / len(analysis_evidences)
            logger.debug(
                "CLAIM_VERIFY_ANALYSIS index=%d supporting_facts=%d evidences=%d evidence_match=%.2f",
                claim_index,
                len(supporting_claims),
                len(analysis_evidences),
                evidence_match_score,
            )
            if self.analysis_grounding_verifier is not None:
                result = await self.analysis_grounding_verifier.verify_analysis_claim(
                    claim_text=claim.claim_text,
                    supporting_claims=supporting_claims,
                    evidences=analysis_evidences,
                    supporting_fact_contexts=[
                        {
                            "claim_text": fact.claim_text,
                            "verification_text": fact.verification_text or fact.claim_text,
                            "verdict": fact.verification_verdict or "",
                        }
                        for fact in supporting_facts
                    ],
                )
            else:
                result = VerificationResult(
                    verdict=(
                        VerificationVerdict.PARTIAL
                        if supporting_claims
                        else VerificationVerdict.UNSUPPORTED
                    ),
                    reasoning=(
                        "Analysis references grounded facts but no dedicated grounding "
                        "verifier was configured."
                        if supporting_claims
                        else "Analysis could not be grounded because no supporting facts "
                        "were linked."
                    ),
                    confidence=0.4 if supporting_claims else 0.0,
                )
        else:
            if verification_evidence is None:
                claim.abstained = True
                logger.info(
                    "CITATION_ABSTAIN_SILENT claim_role=%s citation_keys=%s "
                    "evidences_count=%d claim_head=%r",
                    getattr(claim, "claim_role", "?"),
                    [getattr(e, "citation_key", None) for e in (claim.evidences or [])],
                    len(claim.evidences or []),
                    _truncate(claim.claim_text, 80),
                )
                return []

            evidence_match_score = self._score_claim_evidence_text(
                verification_text,
                verification_evidence.quote_text,
            )
            used_quick = claim.confidence_level == ConfidenceLevel.HIGH.value
            result = await self.verifier.verify_with_isolation(
                claim_text=verification_text,
                evidence=verification_evidence,
                use_quick_verification=used_quick,
            )

            if (
                include_numeric_verification
                and claim.claim_type == "numeric"
                and self.config.enable_numeric_qa_verification
            ):
                numeric_result = await self._verify_numeric_claim(
                    claim,
                    verification_evidence,
                )
                if numeric_result:
                    claim.verification_method = VerificationMethod.NUMERIC_QA.value

        verification_latency_ms = (time.monotonic() - verification_started) * 1000.0
        claim.verification_verdict = result.verdict.value
        claim.verification_reasoning = result.reasoning
        claim.verification_confidence = result.confidence
        claim.evidence_match_score = evidence_match_score
        claim.used_quick_verification = used_quick
        claim.verification_latency_ms = verification_latency_ms
        if claim.claim_role == ClaimRole.ANALYSIS.value:
            claim.verification_method = VerificationMethod.GROUNDING.value
        elif not claim.verification_method:
            claim.verification_method = VerificationMethod.ENTAILMENT.value

        routing_score = (
            claim.routing_confidence_score
            if claim.routing_confidence_score is not None
            else _confidence_score_from_level(claim.confidence_level)
        )
        events = [
            VerificationEvent(
                event_type="claim_verified",
                data={
                    "claim_index": claim_index,
                    "claim_text": claim.claim_text,
                    "position_start": claim.position_start,
                    "position_end": claim.position_end,
                    "verdict": result.verdict.value,
                    "confidence_level": claim.confidence_level,
                    "routing_confidence_level": claim.confidence_level,
                    "routing_confidence_score": routing_score,
                    "confidence": result.confidence,
                    "verification_confidence": result.confidence,
                    "evidence_match_score": evidence_match_score,
                    "used_quick_verification": used_quick,
                    "verification_latency_ms": verification_latency_ms,
                    "evidence_preview": _truncate(
                        verification_evidence.quote_text if verification_evidence else "",
                        100,
                    ),
                    "reasoning": result.reasoning,
                    "citation_key": claim.citation_key,
                    "citation_keys": claim.citation_keys,
                    "claim_role": claim.claim_role,
                    "verification_method": claim.verification_method or "",
                },
            )
        ]

        self._verification_route_stats.append(
            {
                "route": claim.confidence_level or ConfidenceLevel.MEDIUM.value,
                "used_quick_verification": used_quick,
                "verification_latency_ms": verification_latency_ms,
            }
        )

        if numeric_result:
            parsed = numeric_result.get("parsed_value", {})
            raw_value = str(parsed.get("raw_text", "") or "").strip()
            if (
                raw_value
                and len(raw_value) <= 80
                and "\n" not in raw_value
                and not raw_value.startswith("#")
                and "**" not in raw_value
            ):
                events.append(
                    VerificationEvent(
                        event_type="numeric_claim_detected",
                        data={
                            "claim_index": claim_index,
                            "claim_text": claim.claim_text,
                            "raw_value": raw_value,
                            "normalized_value": parsed.get("normalized_value"),
                            "unit": parsed.get("unit"),
                            "derivation_type": numeric_result.get(
                                "derivation_type", "direct"
                            ),
                            "qa_verified": numeric_result.get(
                                "overall_match", False
                            ),
                        },
                    )
                )

        evidence_count = (
            len(analysis_evidences)
            if claim.claim_role == ClaimRole.ANALYSIS.value
            else len(ranked_evidences)
        )
        logger.debug(
            "CLAIM_VERIFIED index=%d verdict=%s role=%s method=%s route=%s numeric=%s "
            "evidence_count=%d evidence_match=%.2f quick=%s latency_ms=%.1f claim=%s",
            claim_index,
            result.verdict.value,
            claim.claim_role,
            claim.verification_method,
            claim.confidence_level,
            claim.claim_type == "numeric",
            evidence_count,
            evidence_match_score,
            used_quick,
            verification_latency_ms,
            _truncate(claim.claim_text, 50),
        )
        return events

    async def verify_claims(
        self,
        claims: list[ClaimInfo],
        *,
        target_roles: set[str] | None = None,
    ) -> AsyncGenerator[VerificationEvent, None]:
        """Stage 4: Verify claims in isolation.

        Yields:
            ``VerificationEvent`` for each claim verified.
        """
        # Stage 4 always runs when the pipeline is used.
        # (Master toggle is checked in run_full_pipeline.)
        if not self.config.enabled:
            logger.info("CITATION_PIPELINE stage=4 action=skipped reason=disabled")
            return

        logger.info(
            "CITATION_PIPELINE_STAGE4 claims_count=%d target_roles=%s",
            len(claims),
            sorted(target_roles) if target_roles is not None else ["all"],
        )

        self._active_claims_context = claims
        try:
            # Phase 1: Pre-filter — handle FREE and no-evidence claims immediately
            verify_tasks: list[tuple[int, ClaimInfo]] = []
            for claim_index, claim in enumerate(claims):
                if target_roles is not None and claim.claim_role not in target_roles:
                    continue
                if claim.claim_role == ClaimRole.FREE.value:
                    claim.abstained = True
                    continue
                if (
                    claim.claim_role == ClaimRole.FACT.value
                    and not self._claim_evidences(claim)
                ):
                    claim.abstained = True
                    claim.verification_verdict = "abstained"
                    claim.verification_reasoning = "No supporting evidence available for isolated verification."
                    claim.verification_confidence = 0.0
                    claim.evidence_match_score = 0.0
                    claim.used_quick_verification = False
                    claim.verification_latency_ms = 0.0
                    if not claim.verification_method:
                        claim.verification_method = VerificationMethod.ENTAILMENT.value
                    yield VerificationEvent(
                        event_type="claim_verified",
                        data={
                            "claim_index": claim_index,
                            "claim_text": claim.claim_text,
                            "position_start": claim.position_start,
                            "position_end": claim.position_end,
                            "verdict": "abstained",
                            "confidence_level": claim.confidence_level,
                            "routing_confidence_level": claim.confidence_level,
                            "routing_confidence_score": (
                                claim.routing_confidence_score
                                if claim.routing_confidence_score is not None
                                else _confidence_score_from_level(claim.confidence_level)
                            ),
                            "confidence": 0.0,
                            "verification_confidence": 0.0,
                            "evidence_match_score": 0.0,
                            "used_quick_verification": False,
                            "verification_latency_ms": 0.0,
                            "evidence_preview": "",
                            "reasoning": claim.verification_reasoning,
                            "citation_key": claim.citation_key,
                            "citation_keys": claim.citation_keys,
                            "claim_role": claim.claim_role,
                            "verification_method": claim.verification_method or "",
                        },
                    )
                    continue
                verify_tasks.append((claim_index, claim))

            # Phase 2: Parallel verification with bounded concurrency
            concurrency = getattr(
                self.config.isolated_verification,
                "max_concurrent_verifications", 5,
            )
            sem = asyncio.Semaphore(concurrency)

            async def _bounded_verify(
                claim_index: int, claim: ClaimInfo,
            ) -> list[VerificationEvent]:
                async with sem:
                    try:
                        return await self._verify_claim_once(claim_index, claim)
                    except Exception:
                        logger.warning(
                            "CLAIM_VERIFICATION_ERROR claim=%s",
                            _truncate(claim.claim_text, 50),
                            exc_info=True,
                        )
                        claim.abstained = True
                        return []

            all_event_lists = await asyncio.gather(
                *[_bounded_verify(i, c) for i, c in verify_tasks],
            )

            # Phase 3: Yield events in original claim order (gather preserves order)
            for events in all_event_lists:
                for event in events:
                    yield event

            # Observability: warn on high abstain rate
            abstained = sum(1 for c in claims if c.abstained)
            if claims and abstained / len(claims) > 0.10:
                logger.warning(
                    "HIGH_ABSTAIN_RATE rate=%.1f%% abstained=%d total=%d",
                    abstained / len(claims) * 100, abstained, len(claims),
                )
        finally:
            self._active_claims_context = []

    def _refresh_claim_citation_keys(
        self,
        claim: ClaimInfo,
        evidence_pool: list[RankedEvidence],
    ) -> None:
        """Recompute citation keys from the claim's current evidence references."""
        key_map = build_citation_key_map(evidence_pool)
        refreshed_keys: list[str] = []
        claim_evidences = self._claim_evidences(claim)
        if not claim_evidences:
            claim.citation_key = None
            claim.citation_keys = None
            return

        for claim_evidence in claim_evidences:
            evidence_index = self._resolve_evidence_pool_index(
                claim_evidence,
                evidence_pool,
            )
            if evidence_index is None:
                continue
            citation_key = key_map.get(evidence_index)
            if citation_key is not None:
                refreshed_keys.append(citation_key)

        claim.citation_keys = refreshed_keys or None
        claim.citation_key = refreshed_keys[0] if refreshed_keys else None

    @staticmethod
    def _resolve_source_pool_index(
        evidence_url: str,
        sources: list[dict[str, Any]],
        canonical_source_url: str | None = None,
    ) -> int | None:
        """Resolve a source-pool index from source metadata."""
        target_urls = {
            url
            for url in (canonical_source_url, evidence_url)
            if url
        }
        for source in sources:
            source_url = str(source.get("url") or "")
            canonical_url = str(source.get("canonical_url") or source_url)
            if source_url in target_urls or canonical_url in target_urls:
                source_index = source.get("source_pool_index")
                if isinstance(source_index, int):
                    return source_index
        return None

    @staticmethod
    def _resolve_evidence_pool_index(
        claim_evidence: EvidenceInfo,
        evidence_pool: list[RankedEvidence],
    ) -> int | None:
        """Resolve a claim evidence reference back to an evidence-pool index.

        Diagnostic log ``CITATION_RESOLVE_FAILED`` fires when all three
        fallback paths miss; the ``reason`` field tells which one (out-of-pool
        url, url/quote mismatch, source_pool_index miss) so we can attribute
        Stage 4 silent-abstains to a specific upstream gap.
        """
        if claim_evidence.evidence_pool_index is not None:
            return claim_evidence.evidence_pool_index

        for index, evidence in enumerate(evidence_pool):
            if (
                evidence.source_url == claim_evidence.source_url
                and evidence.quote_text == claim_evidence.quote_text
            ):
                return evidence.evidence_pool_index if evidence.evidence_pool_index is not None else index

        if claim_evidence.source_pool_index is None:
            logger.info(
                "CITATION_RESOLVE_FAILED reason=no_source_pool_index "
                "claim_url=%s pool_size=%d quote_head=%r",
                (claim_evidence.source_url or "")[:80],
                len(evidence_pool),
                (claim_evidence.quote_text or "")[:60],
            )
            return None

        for index, evidence in enumerate(evidence_pool):
            if evidence.source_pool_index == claim_evidence.source_pool_index:
                return evidence.evidence_pool_index if evidence.evidence_pool_index is not None else index

        logger.info(
            "CITATION_RESOLVE_FAILED reason=source_pool_index_miss "
            "claim_source_pool_index=%s pool_size=%d claim_url=%s",
            claim_evidence.source_pool_index,
            len(evidence_pool),
            (claim_evidence.source_url or "")[:80],
        )
        return None

    # ===================================================================
    # Stage 5: Citation Correction
    # ===================================================================

    async def correct_citations(
        self,
        claims: list[ClaimInfo],
        evidence_pool: list[RankedEvidence],
    ) -> AsyncGenerator[VerificationEvent, None]:
        """Stage 5: Correct citations for non-supported claims.

        Yields:
            ``VerificationEvent`` for each correction and correction metrics.
        """
        if not self.config.enable_citation_correction:
            logger.info("CITATION_PIPELINE stage=5 action=skipped reason=disabled")
            return

        claims_to_correct: list[
            tuple[str, RankedEvidence | None, str | None]
        ] = [
            (
                c.verification_text or c.claim_text,
                self._merge_evidence_for_verification(
                    self._ranked_evidences_for_claim(c)
                ),
                c.verification_verdict,
            )
            for c in claims
            if (
                c.claim_role == ClaimRole.FACT.value
                and c.verification_verdict not in ("supported", "partial")
                and not c.abstained
            )
        ]

        if not claims_to_correct:
            logger.info("CITATION_PIPELINE_STAGE5 no_claims_to_correct")
            return

        logger.info(
            "CITATION_PIPELINE_STAGE5 claims_to_correct=%d",
            len(claims_to_correct),
        )

        results, metrics = await self.corrector.correct_citations(
            claims_with_evidence=claims_to_correct,
            evidence_pool=evidence_pool,
        )

        unsupported_claims = [
            (idx, c)
            for idx, c in enumerate(claims)
            if (
                c.claim_role == ClaimRole.FACT.value
                and c.verification_verdict not in ("supported", "partial")
                and not c.abstained
            )
        ]
        for (claim_index, claim), result in zip(unsupported_claims, results, strict=False):
            original_key = claim.citation_key or ""
            logger.debug(
                "CITATION_CORRECTION_RESULT claim_index=%d action=%s original_key=%s "
                "original_source_pool_index=%s corrected_evidence_index=%s match=%.2f claim=%s",
                claim_index,
                result.correction_type.value,
                original_key,
                claim.evidence.source_pool_index if claim.evidence else None,
                result.corrected_evidence_index,
                result.evidence_match_score,
                _truncate(claim.claim_text, 100),
            )
            if (
                result.correction_type == CorrectionAction.REPLACE
                and result.corrected_evidence
            ):
                claim.evidence = _evidence_info_from_ranked(result.corrected_evidence)
                claim.evidences = [claim.evidence]
                self._refresh_claim_citation_keys(claim, evidence_pool)
                logger.info(
                    "CITATION_CORRECTION_APPLIED claim_index=%d action=replace original_key=%s "
                    "corrected_key=%s corrected_source_pool_index=%s corrected_evidence_pool_index=%s",
                    claim_index,
                    original_key,
                    claim.citation_key or "",
                    result.corrected_evidence.source_pool_index,
                    result.corrected_evidence_index,
                )
                yield VerificationEvent(
                    event_type="citation_corrected",
                    data={
                        "claim_index": claim_index,
                        "claim_text": claim.claim_text,
                        "correction_type": result.correction_type.value,
                        "action": result.correction_type.value,
                        "original_evidence": (
                            result.original_evidence.quote_text
                            if result.original_evidence
                            else ""
                        ),
                        "corrected_evidence": result.corrected_evidence.quote_text,
                        "reasoning": result.reasoning,
                        "original_key": original_key,
                        "corrected_key": claim.citation_key or "",
                        "original_source_pool_index": (
                            result.original_evidence.source_pool_index
                            if result.original_evidence
                            else None
                        ),
                        "corrected_source_pool_index": result.corrected_evidence.source_pool_index,
                        "original_evidence_pool_index": result.original_evidence_index,
                        "corrected_evidence_pool_index": result.corrected_evidence_index,
                        "evidence_match_score": result.evidence_match_score,
                    },
                )
                for verification_event in await self._verify_claim_once(
                    claim_index,
                    claim,
                ):
                    yield verification_event

            elif result.correction_type == CorrectionAction.REMOVE:
                # Demote to unsupported so post-processing can soften
                claim.verification_verdict = "unsupported"
                claim.verification_confidence = 0.0
                logger.info(
                    "CITATION_CORRECTION_APPLIED claim_index=%d action=remove original_key=%s",
                    claim_index,
                    claim.citation_key or "",
                )
                yield VerificationEvent(
                    event_type="citation_corrected",
                    data={
                        "claim_index": claim_index,
                        "claim_text": claim.claim_text,
                        "correction_type": result.correction_type.value,
                        "action": result.correction_type.value,
                        "reasoning": result.reasoning,
                        "original_key": claim.citation_key or "",
                        "corrected_key": "",
                        "original_source_pool_index": (
                            claim.evidence.source_pool_index if claim.evidence else None
                        ),
                        "corrected_source_pool_index": None,
                        "original_evidence_pool_index": (
                            claim.evidence.evidence_pool_index if claim.evidence else None
                        ),
                        "corrected_evidence_pool_index": None,
                        "evidence_match_score": result.evidence_match_score,
                    },
                )

            elif result.correction_type == CorrectionAction.ADD_ALTERNATE:
                if claim.evidence and not claim.evidences:
                    claim.evidences = [claim.evidence]
                for alternate in result.alternate_evidence:
                    alternate_info = _evidence_info_from_ranked(alternate)
                    if all(
                        existing.source_url != alternate_info.source_url
                        or existing.quote_text != alternate_info.quote_text
                        for existing in claim.evidences
                    ):
                        claim.evidences.append(alternate_info)
                self._refresh_claim_citation_keys(claim, evidence_pool)
                logger.info(
                    "CITATION_CORRECTION_APPLIED claim_index=%d action=add_alternate "
                    "original_key=%s corrected_key=%s alternate_count=%d",
                    claim_index,
                    original_key,
                    claim.citation_key or "",
                    len(result.alternate_evidence),
                )
                yield VerificationEvent(
                    event_type="citation_corrected",
                    data={
                        "claim_index": claim_index,
                        "claim_text": claim.claim_text,
                        "correction_type": result.correction_type.value,
                        "action": result.correction_type.value,
                        "alternate_count": len(result.alternate_evidence),
                        "reasoning": result.reasoning,
                        "original_key": original_key,
                        "corrected_key": claim.citation_key or "",
                        "original_source_pool_index": (
                            claim.evidence.source_pool_index if claim.evidence else None
                        ),
                        "corrected_source_pool_index": (
                            claim.evidence.source_pool_index if claim.evidence else None
                        ),
                        "original_evidence_pool_index": (
                            claim.evidence.evidence_pool_index if claim.evidence else None
                        ),
                        "corrected_evidence_pool_index": result.corrected_evidence_index,
                        "evidence_match_score": result.evidence_match_score,
                    },
                )
                for verification_event in await self._verify_claim_once(
                    claim_index,
                    claim,
                ):
                    yield verification_event

        yield VerificationEvent(
            event_type="correction_metrics",
            data={
                "total_corrected": (
                    metrics.replaced + metrics.removed + metrics.added_alternate
                ),
                "kept": metrics.kept,
                "replaced": metrics.replaced,
                "removed": metrics.removed,
                "added_alternate": metrics.added_alternate,
                "correction_rate": metrics.correction_rate,
            },
        )

    # ===================================================================
    # Stage 6: Numeric QA Verification
    # ===================================================================

    async def _verify_numeric_claim(
        self,
        claim: ClaimInfo,
        evidence: RankedEvidence,
    ) -> dict[str, Any]:
        """Verify a numeric claim using QA-based verification.

        Returns:
            Verification result as dict, or empty dict on skip/error.
        """
        if not self.config.enable_numeric_qa_verification:
            return {}
        if claim.claim_role != ClaimRole.FACT.value or claim.claim_type != "numeric":
            return {}

        result = await self.numeric_verifier.verify_numeric_claim(
            claim_text=claim.claim_text,
            evidence=evidence,
        )

        return {
            "overall_match": result.overall_match,
            "derivation_type": result.derivation_type,
            "confidence": result.confidence,
            "parsed_value": {
                "raw_text": result.parsed_value.raw_text,
                "normalized_value": (
                    float(result.parsed_value.normalized_value)
                    if result.parsed_value.normalized_value
                    else None
                ),
                "unit": result.parsed_value.unit,
                "entity": result.parsed_value.entity,
            },
            "qa_results": [
                {
                    "question": r.question,
                    "claim_answer": r.claim_answer,
                    "evidence_answer": r.evidence_answer,
                    "match": r.match,
                }
                for r in result.qa_results
            ],
        }

    # ===================================================================
    # Stage 8: Post-verification claim modification
    # ===================================================================

    async def process_unverified_claims(
        self,
        content: str,
        claims: list[ClaimInfo],
    ) -> tuple[str, int, int, int]:
        """Rewrite, remove, or hedge claims after verification.

        All modifications are done in position-descending order to preserve
        correct positions for subsequent modifications.

        Args:
            content: Report content.
            claims: Claims to process.

        Returns:
            ``(modified_content, removed_count, softened_count, rewritten_count)``.
        """
        modifications: list[tuple[str, ClaimInfo]] = []

        # Confidence threshold for promoting abstained-but-confident NLI
        # rejections from KEEP → REMOVE. See GroundingValidationConfig.
        _abstained_threshold = getattr(
            self.config.grounding_validation,
            "abstained_unsupported_remove_threshold",
            0.5,
        )

        # PR3-E R2.2 wiring: when SYNTH_PIPELINE_V2 is enabled and an LLM
        # client is available, classify negative-existence claims BEFORE
        # the disposition loop runs so the force-REMOVE rule has the flag
        # to read. Skipped on the disposition_applier path (which bypasses
        # __init__ so self.llm is unset) and when the flag is off.
        if _synth_pipeline_v2_enabled() and getattr(self, "llm", None) is not None:
            await _classify_negative_existence_batch(claims, self.llm)

        # PR3-0 instrumentation: trace what actually reaches Stage 8. The
        # failing officeqa run had 7 contradicted + 39 abstained verdicts in
        # events.jsonl but ``removed_claims=0`` in verification_summary; this
        # trace identifies whether the claims list arriving here matches what
        # the reflector emitted.
        _pass_id = _next_stage8_pass_id()
        _count_by_verdict = Counter(
            (c.verification_verdict or "<none>") for c in claims
        )
        _count_by_role = Counter(c.claim_role for c in claims)
        _abstained_count = sum(1 for c in claims if c.abstained)
        _with_verification_text = sum(1 for c in claims if c.verification_text)
        logger.info(
            "STAGE8_INPUT_TRACE pass_id=%d input_claims=%d "
            "by_verdict=%s by_role=%s abstained=%d with_verification_text=%d "
            "abstained_threshold=%.2f",
            _pass_id,
            len(claims),
            dict(_count_by_verdict),
            dict(_count_by_role),
            _abstained_count,
            _with_verification_text,
            _abstained_threshold,
        )

        for claim in claims:
            # Determine verdict key for disposition lookup
            _claim_confidence = getattr(claim, "verification_confidence", None) or 0.0
            if (
                claim.abstained
                and claim.verification_verdict in ("unsupported", "contradicted")
                and _claim_confidence < _abstained_threshold
            ):
                # NLI judged unsupported and abstention reflects insufficient
                # evidence rather than principled uncertainty — promote to
                # REMOVE so it doesn't leak through the report uncited.
                verdict_key = claim.verification_verdict
                logger.info(
                    "DR_LEAK_TRACE phase=abstained_promoted_to_remove "
                    "verdict=%s confidence=%.2f threshold=%.2f claim_head=%r",
                    claim.verification_verdict,
                    _claim_confidence,
                    _abstained_threshold,
                    _truncate(claim.claim_text, 80),
                )
            elif claim.abstained:
                verdict_key = "abstained"
            elif claim.claim_role == ClaimRole.ANALYSIS.value:
                if claim.verification_verdict == "partial":
                    verdict_key = "analysis_partial"
                elif claim.verification_verdict == "unsupported":
                    verdict_key = "analysis_unsupported"
                elif claim.verification_verdict == "contradicted":
                    verdict_key = "contradicted"
                else:
                    continue
            elif claim.claim_role == ClaimRole.FACT.value:
                verdict_key = claim.verification_verdict or "abstained"
            else:
                continue

            disposition = getattr(
                self.config.claim_disposition, verdict_key, ClaimDisposition.KEEP,
            )

            # PR3-E R2.2: force REMOVE when is_negative_existence=True AND
            # verdict is not fully supported. This wins over the per-verdict
            # policy (e.g. abstained→SOFTEN default). Gated by
            # SYNTH_PIPELINE_V2 so the legacy path is preserved unchanged.
            forced_remove = False
            if (
                _synth_pipeline_v2_enabled()
                and getattr(claim, "is_negative_existence", False)
                and (
                    claim.abstained
                    or (claim.verification_verdict or "").lower()
                    in ("abstained", "unsupported", "contradicted", "partial")
                )
            ):
                disposition = ClaimDisposition.REMOVE
                forced_remove = True
                logger.info(
                    "DR_NEGATIVE_EXISTENCE_FORCE_REMOVE pass_id=%d "
                    "verdict=%s claim_head=%r",
                    _pass_id,
                    claim.verification_verdict,
                    _truncate(claim.claim_text, 60),
                )

            # PR3-0 instrumentation: emit a per-claim disposition decision for
            # every non-supported verdict so the events.jsonl-vs-summary
            # discrepancy can be traced claim-by-claim. Tight conditional to
            # avoid log spam on supported claims.
            if (
                claim.verification_verdict in ("contradicted", "unsupported")
                or claim.abstained
            ):
                logger.info(
                    "DR_DISPOSITION_DECISION pass_id=%d verdict_key=%s "
                    "disposition=%s claim_role=%s confidence=%.2f "
                    "forced_negative_existence=%s claim_head=%r",
                    _pass_id,
                    verdict_key,
                    disposition.value if hasattr(disposition, "value") else str(disposition),
                    claim.claim_role,
                    _claim_confidence,
                    forced_remove,
                    _truncate(claim.claim_text, 60),
                )

            if disposition == ClaimDisposition.REMOVE:
                modifications.append(("remove", claim))
            elif disposition == ClaimDisposition.SOFTEN:
                action = (
                    "soften_analysis"
                    if claim.claim_role == ClaimRole.ANALYSIS.value
                    else "soften"
                )
                modifications.append((action, claim))
            elif (  # "keep" — check for rewrite (EXCLUSIVE with remove/soften)
                claim.claim_role == ClaimRole.FACT.value
                and claim.verification_text
                and claim.verification_text != claim.claim_text
                and claim.verification_verdict in ("supported", "partial")
            ):
                modifications.append(("rewrite", claim))

        if not modifications:
            return content, 0, 0, 0

        # Sort descending by position -- process from end to start
        modifications.sort(key=lambda x: x[1].position_start, reverse=True)
        modifications = self._merge_overlapping_modifications(modifications)

        removed_count = 0
        softened_count = 0
        rewritten_count = 0

        for action, claim in modifications:
            context = _detect_special_context(content, claim.position_start)
            if context == "code":
                continue

            if action == "remove":
                content = _remove_claim(content, claim, context)
                if claim.claim_role == ClaimRole.ANALYSIS.value:
                    claim.abstained = True
                removed_count += 1
                logger.info(
                    "STAGE8_CLAIM_REMOVED verdict=%s confidence=%.2f "
                    "citation_keys=%s evidences_count=%d claim=%s",
                    claim.verification_verdict,
                    float(getattr(claim, "verification_confidence", 0.0) or 0.0),
                    [getattr(e, "citation_key", None) for e in (claim.evidences or [])],
                    len(claim.evidences or []),
                    _truncate(claim.claim_text, 50),
                )
            elif action == "soften":
                softened_text = _build_softened_fact_text(claim.claim_text, context)
                content = _replace_claim_span(content, claim, softened_text)
                claim.claim_text = softened_text
                softened_count += 1
                logger.info(
                    "CLAIM_SOFTENED claim=%s",
                    _truncate(claim.claim_text, 50),
                )
            elif action == "soften_analysis":
                softened_text = _build_softened_analysis_text(
                    claim.claim_text,
                    self.config.grounding_validation.hedging_prefix,
                )
                content = _replace_claim_span(content, claim, softened_text)
                claim.claim_text = softened_text
                claim.verification_verdict = "partial"
                softened_count += 1
                logger.info(
                    "ANALYSIS_SOFTENED claim=%s",
                    _truncate(claim.claim_text, 50),
                )
            else:
                rewritten_text = claim.verification_text or claim.claim_text
                if not _is_well_formed_claim_rewrite(rewritten_text):
                    logger.info(
                        "CLAIM_REWRITE_SKIPPED_MALFORMED claim=%s rewrite=%s",
                        _truncate(claim.claim_text, 50),
                        _truncate(rewritten_text, 50),
                    )
                    continue
                rewritten_text = _format_claim_rewrite_text(
                    claim,
                    rewritten_text,
                    context,
                )
                content = _replace_claim_span(content, claim, rewritten_text)
                claim.claim_text = rewritten_text
                rewritten_count += 1
                logger.info(
                    "CLAIM_REWRITTEN claim=%s",
                    _truncate(claim.claim_text, 50),
                )

        if removed_count > 0:
            content = _clean_empty_sections(content)

        _recalculate_claim_positions(content, claims)

        # PR3-0 instrumentation: per-verdict breakdown of how the disposition
        # policy resolved each modification. Pair this with STAGE8_INPUT_TRACE
        # via ``pass_id`` to see input → output for a given invocation.
        _removed_by_verdict: Counter[str] = Counter()
        _softened_by_verdict: Counter[str] = Counter()
        for action, claim in modifications:
            verdict_key = claim.verification_verdict or "<none>"
            if action == "remove":
                _removed_by_verdict[verdict_key] += 1
            elif action in ("soften", "soften_analysis"):
                _softened_by_verdict[verdict_key] += 1
        _kept_count = len(claims) - len(modifications)
        logger.info(
            "STAGE8_COMPLETE pass_id=%d removed=%d softened=%d rewritten=%d "
            "total_modifications=%d kept=%d input_claims=%d "
            "removed_by_verdict=%s softened_by_verdict=%s",
            _pass_id,
            removed_count,
            softened_count,
            rewritten_count,
            len(modifications),
            _kept_count,
            len(claims),
            dict(_removed_by_verdict),
            dict(_softened_by_verdict),
        )
        return content, removed_count, softened_count, rewritten_count

    # ===================================================================
    # Full pipeline orchestration
    # ===================================================================

    async def run_full_pipeline(
        self,
        sources: list[dict[str, Any]],
        observations: list[str],
        query: str,
        *,
        target_word_count: int = 600,
        max_tokens: int = 2000,
        draft_content: str | None = None,
        generation_instructions: str = "",
    ) -> AsyncGenerator[VerificationEvent | str, None]:
        """Run the complete 7-stage citation verification pipeline.

        Stages executed:
            1. Evidence pre-selection
            2. Interleaved generation (streams content + extracts claims)
            3. Confidence classification (per-claim)
            4. Isolated verification (per-claim)
            5. Citation correction (for non-supported claims)
            6. Numeric QA verification (inline with Stage 4)
            7. ARE verification retrieval (placeholder -- delegated to caller)

        After stages 1-7 the caller may invoke ``process_unverified_claims``
        for Stage 8 post-processing.

        Args:
            sources: Source dicts with ``url``, ``title``, ``content``,
                     ``snippet``, ``source_type`` keys.
            observations: Research observations (context for generation).
            query: Original research query.
            target_word_count: Target word count for the report.
            max_tokens: Maximum tokens to generate.
            draft_content: Existing report content to verify instead of running
                interleaved generation. Used by classical-lite grounding.
            generation_instructions: Workflow-specific report shape and quality
                contract for Stage 2 generation. This does not affect evidence
                pre-selection, which continues to use ``query`` only.

        Yields:
            ``str`` content chunks and ``VerificationEvent`` objects.
        """
        if not self.config.enabled:
            logger.info("CITATION_PIPELINE action=disabled_globally")
            return

        logger.info(
            "CITATION_PIPELINE_START sources=%d observations=%d "
            "target_words=%d max_tokens=%d",
            len(sources),
            len(observations),
            target_word_count,
            max_tokens,
        )

        self.last_evidence_pool = []
        self.last_generated_claims = []
        self.last_verification_summary = None
        self.last_final_content = ""
        self.last_routing_summary = {}
        self._verification_route_stats = []

        async with trace_span("citation.pipeline", attributes={"sources": len(sources), "target_words": target_word_count, "max_tokens": max_tokens}):

            # ------------------------------------------------------------------
            # Stage 1: Pre-select evidence
            # ------------------------------------------------------------------
            async with trace_span("citation.stage1.evidence_selection", attributes={"sources": len(sources)}) as span:
                evidence_pool = await self.preselect_evidence(sources, query)
                self.last_evidence_pool = list(evidence_pool)

                sources_with_content = sum(1 for s in sources if s.get("content"))
                logger.info(
                    "CITATION_PIPELINE_STAGE1_RESULT evidence=%d sources=%d "
                    "with_content=%d",
                    len(evidence_pool),
                    len(sources),
                    sources_with_content,
                )
                if span:
                    span.set_attributes({
                        "evidence_count": len(evidence_pool),
                        "sources_with_content": sources_with_content,
                        "sources_input": len(sources),
                    })

            if not evidence_pool:
                skipped_summary = VerificationSummaryInfo(
                    total_claims=0,
                    supported_count=0,
                    partial_count=0,
                    unsupported_count=0,
                    contradicted_count=0,
                    abstained_count=0,
                    unsupported_rate=0.0,
                    contradicted_rate=0.0,
                    warning=False,
                    citation_corrections=0,
                )
                self.last_verification_summary = skipped_summary
                logger.warning(
                    "CITATION_PIPELINE_EMPTY_EVIDENCE sources=%d with_content=%d",
                    len(sources),
                    sources_with_content,
                )
                yield VerificationEvent(
                    event_type="verification_summary",
                    data={
                        "verification_skipped": True,
                        "reason": "empty_evidence_pool",
                        "total_claims": 0,
                        "supported": 0,
                        "partial": 0,
                        "unsupported": 0,
                        "contradicted": 0,
                        "abstained_count": 0,
                        "supported_rate": 0.0,
                        "warning": False,
                        "analysis_summary": skipped_summary.analysis_summary.to_dict(),
                        "routing_summary": {},
                    },
                )
                return

            generated_claims: list[ClaimInfo] = []
            full_content = draft_content or ""

            # ------------------------------------------------------------------
            # Stage 2: Generate with interleaved claims or parse an existing draft
            # ------------------------------------------------------------------
            async with trace_span("citation.stage2.interleaved_generation", attributes={"target_words": target_word_count, "max_tokens": max_tokens}) as span:
                if draft_content is not None:
                    from databricks_deep_research.citation.claim_generator import (
                        _parse_interleaved_content,
                    )

                    parsed_claims = _parse_interleaved_content(
                        draft_content,
                        evidence_pool,
                    )
                    for parsed_claim in parsed_claims:
                        generated_claims.append(_claim_info_from_interleaved(parsed_claim))
                elif self.config.synthesis_mode == SynthesisMode.REACT:
                    from databricks_deep_research.citation.react_generator import (
                        ReactGenerator,
                    )

                    react_gen = ReactGenerator(self.llm)
                    async for react_content, react_claim in react_gen.synthesize(
                        query=query,
                        evidence_pool=evidence_pool,
                        target_word_count=target_word_count,
                        max_tokens=max_tokens,
                        max_tool_calls=self.config.react_synthesis.max_tool_calls,
                        section_descriptions="\n".join(
                            part
                            for part in [
                                generation_instructions,
                                "\n".join(observations) if observations else "",
                            ]
                            if part
                        ),
                    ):
                        if react_content:
                            full_content = react_content
                            yield react_content

                        if react_claim:
                            generated_claims.append(_claim_info_from_interleaved(react_claim))

                else:
                    async for interleaved_content, interleaved_claim in self.generate_with_interleaving(
                        evidence_pool=evidence_pool,
                        observations=observations,
                        query=query,
                        target_word_count=target_word_count,
                        max_tokens=max_tokens,
                        generation_instructions=generation_instructions,
                    ):
                        if interleaved_content:
                            full_content = interleaved_content
                            yield interleaved_content

                        if interleaved_claim:
                            generated_claims.append(_claim_info_from_interleaved(interleaved_claim))

                logger.info(
                    "CITATION_PIPELINE_STAGE2_RESULT content_chars=%d generated_claims=%d types=%s",
                    len(full_content),
                    len(generated_claims),
                    _counter_dict([claim.claim_type for claim in generated_claims]),
                )

                # ------------------------------------------------------------------
                # Stage 2.1: Fact vs analysis boundary enforcement
                # ------------------------------------------------------------------
                self._classify_and_link_claims(generated_claims)

                # ------------------------------------------------------------------
                # Stage 2.5: Fallback evidence matching for uncited fact claims
                # ------------------------------------------------------------------
                if evidence_pool:
                    _assign_fallback_evidence(
                        generated_claims,
                        evidence_pool,
                        scorer=getattr(self.corrector, "score_claim_evidence", None),
                    )
                logger.info(
                    "CITATION_PIPELINE_STAGE25_RESULT fallback_evidence=%d",
                    sum(1 for claim in generated_claims if claim.has_fallback_evidence),
                )

                if span:
                    span.set_attributes({"claims_generated": len(generated_claims), "content_length": len(full_content)})

            # ------------------------------------------------------------------
            # Stage 3: Classify confidence after role enforcement and fallback evidence
            # ------------------------------------------------------------------
            async with trace_span("citation.stage3.confidence", attributes={"claims": len(generated_claims)}) as span:
                for claim_info in generated_claims:
                    confidence = self.classify_confidence_result(claim_info)
                    claim_info.confidence_level = confidence.level.value
                    claim_info.routing_confidence_score = confidence.score
                    logger.debug(
                        "CLAIM_ROUTED role=%s type=%s route=%s score=%.2f indicators=%s "
                        "fallback=%s evidence_match=%.2f claim=%s",
                        claim_info.claim_role,
                        claim_info.claim_type,
                        claim_info.confidence_level,
                        confidence.score,
                        confidence.indicators,
                        claim_info.has_fallback_evidence,
                        claim_info.evidence_match_score or 0.0,
                        _truncate(claim_info.claim_text, 100),
                    )

                routes = _counter_dict([claim.confidence_level or "" for claim in generated_claims])
                logger.info(
                    "CITATION_PIPELINE_STAGE3_RESULT routes=%s",
                    routes,
                )

                if span:
                    span.set_attributes({"routes": str(routes)})

                for claim_index, claim_info in enumerate(generated_claims):
                    yield VerificationEvent(
                        event_type="claim_generated",
                        data={
                            "claim_index": claim_index,
                            "claim_text": claim_info.claim_text,
                            "claim_type": claim_info.claim_type,
                            "claim_role": claim_info.claim_role,
                            "position_start": claim_info.position_start,
                            "position_end": claim_info.position_end,
                            "citation_key": claim_info.citation_key,
                            "citation_keys": claim_info.citation_keys or [],
                            "evidence": (
                                claim_info.evidence.to_dict()
                                if claim_info.evidence
                                else None
                            ),
                            "confidence_level": claim_info.confidence_level,
                            "verification_method": claim_info.verification_method,
                        },
                    )

            # ------------------------------------------------------------------
            # Stage 4: Verify all claims (includes inline Stage 6 numeric QA)
            # ------------------------------------------------------------------
            async with trace_span("citation.stage4.verification") as span:
                async for event in self.verify_claims(
                    generated_claims,
                    target_roles={ClaimRole.FACT.value},
                ):
                    yield event
                logger.info(
                    "CITATION_PIPELINE_STAGE4_FACT_RESULT verdicts=%s",
                    _counter_dict(
                        [
                            claim.verification_verdict or "abstained"
                            for claim in generated_claims
                            if claim.claim_role == ClaimRole.FACT.value
                        ]
                    ),
                )
                if span:
                    verdicts = _counter_dict([c.verification_verdict or "abstained" for c in generated_claims if c.claim_role == ClaimRole.FACT.value])
                    span.set_attributes({"verdicts": str(verdicts), "claims_verified": sum(1 for c in generated_claims if c.claim_role == ClaimRole.FACT.value)})

            # ------------------------------------------------------------------
            # Stage 5: Correct citations for non-supported claims
            # ------------------------------------------------------------------
            async with trace_span("citation.stage5.correction") as span:
                correction_count = 0
                async for event in self.correct_citations(generated_claims, evidence_pool):
                    if event.event_type == "correction_metrics":
                        correction_count = event.data.get("total_corrected", 0)
                    yield event
                if span:
                    span.set_attributes({"corrections_made": correction_count})

            # ------------------------------------------------------------------
            # Stage 7: ARE-style Verification Retrieval (placeholder)
            # ------------------------------------------------------------------
            # Stage 7 is complex (atomic decomposition, external search,
            # revision) and tightly coupled to specific tool implementations.
            # The framework exposes a hook point: callers that have a
            # VerificationRetriever can run it on the claims that still
            # need revision.
            stage_7_claims = [
                c
                for c in generated_claims
                if (
                    c.claim_role == ClaimRole.FACT.value
                    and c.verification_verdict
                    in self.config.verification_retrieval.trigger_on_verdicts
                    and not c.abstained
                )
            ]
            stage_7_metrics: dict[str, Any] | None = None
            if (
                stage_7_claims
                and self.config.enable_verification_retrieval
                and self.verification_retriever is not None
            ):
                async with trace_span("citation.stage7.retrieval", attributes={"claims": len(stage_7_claims)}) as span:
                    logger.info(
                        "CITATION_PIPELINE_STAGE7_START claims=%d trigger_verdicts=%s",
                        len(stage_7_claims),
                        [c.verification_verdict for c in stage_7_claims],
                    )
                    yield VerificationEvent(
                        event_type="stage_7_ready",
                        data={
                            "claims_count": len(stage_7_claims),
                            "verdicts": [
                                c.verification_verdict for c in stage_7_claims
                            ],
                        },
                    )
                    revisions: list[Any] = []
                    async for stage_7_item in self.verification_retriever.retrieve_and_revise(
                        claims=generated_claims,
                        evidence_pool=evidence_pool,
                        report_content=full_content,
                        research_query=query,
                    ):
                        if hasattr(stage_7_item, "revision_type"):
                            revisions.append(stage_7_item)
                            continue
                        if hasattr(stage_7_item, "event_type") and hasattr(stage_7_item, "data"):
                            yield VerificationEvent(
                                event_type=str(stage_7_item.event_type),
                                data=dict(stage_7_item.data),
                            )

                    if revisions:
                        full_content = self.verification_retriever.apply_all_revisions(
                            full_content,
                            revisions,
                        )
                        for revision in revisions:
                            for claim in generated_claims:
                                if (
                                    claim.position_start == revision.original_position_start
                                    and claim.position_end == revision.original_position_end
                                    and claim.claim_text == revision.original_claim
                                ):
                                    claim.claim_text = revision.revised_claim
                                    if revision.revision_type == "fully_verified":
                                        claim.verification_verdict = "supported"
                                        claim.verification_reasoning = (
                                            "Stage 7 verified the claim via atomic fact retrieval."
                                        )
                                        claim.verification_confidence = max(
                                            claim.verification_confidence or 0.0,
                                            0.85,
                                        )
                                    elif revision.revision_type == "partially_softened":
                                        claim.verification_verdict = "partial"
                                        claim.verification_reasoning = (
                                            "Stage 7 softened unverified atomic facts."
                                        )
                                        claim.verification_confidence = max(
                                            claim.verification_confidence or 0.0,
                                            0.7,
                                        )
                                    else:
                                        claim.verification_verdict = "unsupported"
                                        claim.verification_reasoning = (
                                            "Stage 7 could not verify the claim and softened it."
                                        )
                                        claim.verification_confidence = (
                                            claim.verification_confidence or 0.5
                                        )
                                    break
                        _recalculate_claim_positions(full_content, generated_claims)

                    metrics = getattr(self.verification_retriever, "metrics", None)
                    if metrics is not None and hasattr(metrics, "to_dict"):
                        stage_7_metrics = metrics.to_dict()
                        logger.info(
                            "CITATION_PIPELINE_STAGE7_RESULT metrics=%s",
                            stage_7_metrics,
                        )
                    if span and stage_7_metrics:
                        span.set_attributes({"claims_revised": stage_7_metrics.get("claims_fully_verified", 0) + stage_7_metrics.get("claims_partially_softened", 0)})

            # ------------------------------------------------------------------
            # Stage 4b: Verify analysis after facts are finalized
            # ------------------------------------------------------------------
            async with trace_span("citation.stage4b.verification_analysis") as span:
                async for event in self.verify_claims(
                    generated_claims,
                    target_roles={ClaimRole.ANALYSIS.value},
                ):
                    yield event
                logger.info(
                    "CITATION_PIPELINE_STAGE4_ANALYSIS_RESULT verdicts=%s",
                    _counter_dict(
                        [
                            claim.verification_verdict or "abstained"
                            for claim in generated_claims
                            if claim.claim_role == ClaimRole.ANALYSIS.value
                        ]
                    ),
                )
                if span:
                    verdicts = _counter_dict([c.verification_verdict or "abstained" for c in generated_claims if c.claim_role == ClaimRole.ANALYSIS.value])
                    span.set_attributes({"verdicts": str(verdicts)})

            # ------------------------------------------------------------------
            # Stage 8: Post-verification claim modification
            # ------------------------------------------------------------------
            async with trace_span("citation.stage8.post_processing") as span:
                (
                    full_content,
                    stage_8_removed,
                    stage_8_softened,
                    stage_8_rewritten,
                ) = await self.process_unverified_claims(full_content, generated_claims)
                if span:
                    span.set_attributes({"removed": stage_8_removed, "softened": stage_8_softened, "rewritten": stage_8_rewritten})

                if stage_8_removed > 0 or stage_8_softened > 0 or stage_8_rewritten > 0:
                    yield VerificationEvent(
                        event_type="claims_processed",
                        data={
                            "removed_count": stage_8_removed,
                            "softened_count": stage_8_softened,
                            "rewritten_count": stage_8_rewritten,
                        },
                    )
                    yield VerificationEvent(
                        event_type="content_revised",
                        data={
                            "content": full_content,
                            "stage": "claim_modification",
                            "removed": stage_8_removed,
                            "softened": stage_8_softened,
                            "rewritten": stage_8_rewritten,
                        },
                    )

            full_content = _strip_reclaim_block_tags(full_content)
            # Normalize markdown header whitespace BEFORE recalc so subsequent
            # claim-position consumers see positions valid against the final
            # text the user will see.
            full_content = _ensure_header_breaks(full_content)
            _recalculate_claim_positions(full_content, generated_claims)

            # ------------------------------------------------------------------
            # Summary
            # ------------------------------------------------------------------
            summary = _build_verification_summary(generated_claims, correction_count)
            summary.routing_summary = self._build_routing_summary()
            if stage_7_metrics:
                summary.claim_revisions = (
                    stage_7_metrics.get("claims_fully_verified", 0)
                    + stage_7_metrics.get("claims_partially_softened", 0)
                    + stage_7_metrics.get("claims_fully_softened", 0)
                )
                summary.atomic_facts_total = stage_7_metrics.get("total_atomic_facts", 0)
                summary.atomic_facts_verified = stage_7_metrics.get("facts_verified", 0)
                summary.atomic_facts_softened = stage_7_metrics.get("facts_softened", 0)
                summary.claims_fully_verified = stage_7_metrics.get("claims_fully_verified", 0)
                summary.claims_partially_softened = stage_7_metrics.get(
                    "claims_partially_softened", 0
                )
                summary.claims_fully_softened = stage_7_metrics.get(
                    "claims_fully_softened", 0
                )
                summary.external_searches = stage_7_metrics.get("external_searches", 0)
                summary.new_sources_added = stage_7_metrics.get("new_sources_added", 0)
            self.last_generated_claims = list(generated_claims)
            self.last_verification_summary = summary
            self.last_final_content = full_content
            self.last_routing_summary = dict(summary.routing_summary)
            # DR_LEAK_TRACE citation_final: capture the final processed report
            # content. Compare to state_write (synthesizer) — if planning text
            # is present here that wasn't present in the synthesizer's output,
            # citation pipeline introduced it.
            try:
                _head = (full_content or "")[:300].replace("\n", "\\n")
                _tail = (full_content or "")[-300:].replace("\n", "\\n")
                logger.info(
                    "DR_LEAK_TRACE phase=citation_final "
                    "content_len=%d claims=%d head=%r tail=%r",
                    len(full_content or ""),
                    len(generated_claims),
                    _head,
                    _tail,
                )
            except Exception as _exc:  # pragma: no cover — diagnostic only
                logger.debug("DR_LEAK_TRACE citation_final skipped: %s", _exc)

            yield VerificationEvent(
                event_type="verification_summary",
                data={
                    "total_claims": summary.total_claims,
                    "supported": summary.supported_count,
                    "partial": summary.partial_count,
                    "unsupported": summary.unsupported_count,
                    "contradicted": summary.contradicted_count,
                    "abstained_count": summary.abstained_count,
                    "supported_rate": summary.supported_rate,
                    "citation_corrections": correction_count,
                    "warning": summary.warning,
                    "unsupported_rate": summary.unsupported_rate,
                    "contradicted_rate": summary.contradicted_rate,
                    "analysis_summary": summary.analysis_summary.to_dict(),
                    "routing_summary": summary.routing_summary,
                },
            )

            logger.info(
                "CITATION_PIPELINE_COMPLETE claims=%d verified=%d supported=%d partial=%d "
                "unsupported=%d contradicted=%d analysis_supported=%d routing=%s",
                len(generated_claims),
                sum(1 for c in generated_claims if c.verification_verdict),
                sum(
                    1
                    for c in generated_claims
                    if c.verification_verdict == "supported"
                ),
                sum(
                    1
                    for c in generated_claims
                    if c.claim_role == ClaimRole.FACT.value and c.verification_verdict == "partial"
                ),
                sum(
                    1
                    for c in generated_claims
                    if c.claim_role == ClaimRole.FACT.value and c.verification_verdict == "unsupported"
                ),
                sum(
                    1
                    for c in generated_claims
                    if c.claim_role == ClaimRole.FACT.value and c.verification_verdict == "contradicted"
                ),
                sum(
                    1
                    for c in generated_claims
                    if c.claim_role == ClaimRole.ANALYSIS.value and c.verification_verdict == "supported"
                ),
                summary.routing_summary,
            )

    # ===================================================================
    # Utilities
    # ===================================================================

    def _build_routing_summary(self) -> dict[str, Any]:
        """Aggregate Stage 3 routing and verification-path telemetry."""
        route_counts = {
            ConfidenceLevel.HIGH.value: 0,
            ConfidenceLevel.MEDIUM.value: 0,
            ConfidenceLevel.LOW.value: 0,
        }
        route_latencies: dict[str, list[float]] = {
            ConfidenceLevel.HIGH.value: [],
            ConfidenceLevel.MEDIUM.value: [],
            ConfidenceLevel.LOW.value: [],
        }
        quick_count = 0

        for item in self._verification_route_stats:
            route = str(item.get("route") or ConfidenceLevel.MEDIUM.value)
            if route not in route_counts:
                route_counts[route] = 0
                route_latencies[route] = []
            route_counts[route] += 1
            route_latencies[route].append(float(item.get("verification_latency_ms", 0.0)))
            if item.get("used_quick_verification"):
                quick_count += 1

        total = sum(route_counts.values())
        average_latency_ms = {
            route: (
                sum(latencies) / len(latencies)
                if latencies else 0.0
            )
            for route, latencies in route_latencies.items()
        }
        return {
            "high_count": route_counts.get(ConfidenceLevel.HIGH.value, 0),
            "medium_count": route_counts.get(ConfidenceLevel.MEDIUM.value, 0),
            "low_count": route_counts.get(ConfidenceLevel.LOW.value, 0),
            "quick_path_count": quick_count,
            "full_path_count": max(total - quick_count, 0),
            "quick_path_hit_rate": (quick_count / total if total > 0 else 0.0),
            "average_latency_ms": average_latency_ms,
        }

    @staticmethod
    def build_citation_key(
        index: int, url: str, evidence_pool: list[RankedEvidence]
    ) -> str:
        """Build a citation key matching EvidenceRegistry logic.

        Duplicate-detection: suffix is based on how many same-domain URLs
        appear BEFORE *index*, not on the index itself.

        Args:
            index: Evidence index in the pool.
            url: Source URL for this evidence.
            evidence_pool: Full evidence pool for duplicate detection.

        Returns:
            Citation key like ``"Arxiv"``, ``"Github"``, ``"Arxiv-2"``.
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.replace("www.", "")
            key = domain.split(".")[0].capitalize()
        except Exception:
            key = "Source"

        count = 0
        for ev in evidence_pool[:index]:
            try:
                other_parsed = urlparse(ev.source_url or "")
                other_domain = other_parsed.netloc.replace("www.", "")
                other_key = other_domain.split(".")[0].capitalize()
                if other_key == key:
                    count += 1
            except Exception:
                pass

        return f"{key}-{count + 1}" if count > 0 else key

    @staticmethod
    def _merge_overlapping_modifications(
        modifications: list[tuple[str, ClaimInfo]],
    ) -> list[tuple[str, ClaimInfo]]:
        """Merge overlapping modifications to avoid position conflicts."""
        if len(modifications) <= 1:
            return modifications

        merged: list[tuple[str, ClaimInfo]] = []
        for action, claim in modifications:
            if not merged:
                merged.append((action, claim))
                continue

            prev_action, prev_claim = merged[-1]
            if claim.position_end > prev_claim.position_start:
                merged_action = (
                    "remove"
                    if action == "remove" or prev_action == "remove"
                    else "soften"
                )
                prev_claim.position_start = min(
                    claim.position_start, prev_claim.position_start
                )
                prev_claim.position_end = max(
                    claim.position_end, prev_claim.position_end
                )
                merged[-1] = (merged_action, prev_claim)
            else:
                merged.append((action, claim))

        return merged


# ---------------------------------------------------------------------------
# Free functions -- content modification helpers
# ---------------------------------------------------------------------------


def _detect_special_context(content: str, position: int) -> str | None:
    """Detect if *position* is inside a table, list, or code block."""
    start = max(0, position - 200)
    before = content[start:position]
    lines_before = before.split("\n")
    if not lines_before:
        return None

    last_line = lines_before[-1]
    if "|" in last_line:
        return "table"

    stripped = last_line.strip()
    if stripped.startswith(("-", "*", "+")):
        return "list"
    if stripped and stripped[0].isdigit() and "." in stripped[:3]:
        return "list"

    if before.count("```") % 2 == 1:
        return "code"

    return None


def _remove_claim(content: str, claim: ClaimInfo, context: str | None) -> str:
    """Remove a contradicted claim from *content*."""
    if context == "table":
        before = content[: claim.position_start]
        after = content[claim.position_end :]
        return before + "[removed for factual inaccuracy]" + after

    if context == "list":
        start = content.rfind("\n", 0, claim.position_start) + 1
        end = content.find("\n", claim.position_end)
        if end == -1:
            end = len(content)
        return content[:start] + content[end:]

    before = content[: claim.position_start].rstrip()
    after = content[claim.position_end :].lstrip()
    if (
        before
        and after
        and not before.endswith("\n")
        and not after.startswith("\n")
    ):
        return before + " " + after
    return before + after


def _replace_claim_span(content: str, claim: ClaimInfo, replacement: str) -> str:
    """Replace a claim span with *replacement* using stored offsets."""
    before = content[: claim.position_start]
    after = content[claim.position_end :]
    return before + replacement + after


_TERMINAL_CITATION_RE = re.compile(
    r"(?:\s*\[[A-Za-z0-9-]+\])+\s*[.!?]?\s*$"
)

_CLAIM_TERMINAL_RE = re.compile(r"[.!?]\s*$")

_DANGLING_REWRITE_TAIL_RE = re.compile(
    r"(?:,|;|:|\b(?:and|or|but|still|with|including|while|although|though|as|"
    r"at|to|from|of|for|by|in|on|than|versus|against|between))\s*$",
    re.IGNORECASE,
)


def _is_well_formed_claim_rewrite(text: str) -> bool:
    """Return True when *text* is safe to splice in as a standalone claim."""
    clean_text = text.strip()
    if not clean_text:
        return False
    return _DANGLING_REWRITE_TAIL_RE.search(clean_text) is None


def _claim_citation_suffix(claim: ClaimInfo) -> str:
    """Return existing claim citation keys as adjacent markdown markers."""
    keys: list[str] = []
    claim_keys = (
        claim.citation_keys
        or ([] if claim.citation_key is None else [claim.citation_key])
    )
    for key in claim_keys:
        normalized = str(key).strip()
        if normalized and normalized not in keys:
            keys.append(normalized)
    return "".join(f"[{key}]" for key in keys)


def _ensure_terminal_punctuation(text: str) -> str:
    """End prose replacements with sentence punctuation."""
    clean_text = text.strip()
    if not clean_text or _CLAIM_TERMINAL_RE.search(clean_text):
        return clean_text
    return f"{clean_text}."


def _format_claim_rewrite_text(
    claim: ClaimInfo,
    replacement: str,
    context: str | None,
) -> str:
    """Format a fact rewrite so it remains a cited, sentence-safe replacement."""
    clean_text = replacement.strip()
    if context == "table":
        return clean_text

    citation_suffix = _claim_citation_suffix(claim)
    if citation_suffix and not _TERMINAL_CITATION_RE.search(clean_text):
        clean_text = clean_text.rstrip(".!?;:, ")
        clean_text = f"{clean_text} {citation_suffix}"
    return _ensure_terminal_punctuation(clean_text)


def _strip_citation_markers(text: str) -> str:
    return re.sub(
        r"\s*\[[A-Za-z0-9-]+\]\s*",
        " ",
        text,
    ).strip()


def _normalize_softened_lead(text: str) -> str:
    """Lowercase leading articles after a hedging prefix without harming proper nouns."""
    for prefix in ("The ", "A ", "An ", "This ", "These ", "Those ", "That "):
        if text.startswith(prefix):
            return prefix.lower() + text[len(prefix):]
    return text


def _build_softened_fact_text(claim_text: str, context: str | None) -> str:
    """Return a hedged version of a fact claim."""
    clean_text = _strip_citation_markers(claim_text)
    if not _needs_hedging(clean_text):
        return clean_text
    if context == "table":
        return f"{clean_text} [unverified]"
    hedging_phrases = [
        "It has been suggested that",
        "Some sources indicate that",
        "According to available information,",
        "Reportedly,",
    ]
    hedge_idx = sha256(clean_text.encode("utf-8")).digest()[0] % len(hedging_phrases)
    hedge = hedging_phrases[hedge_idx]
    return f"{hedge} {_normalize_softened_lead(clean_text)}"


def _build_softened_analysis_text(claim_text: str, hedging_prefix: str) -> str:
    """Return a hedged version of an analysis sentence."""
    clean_text = _strip_citation_markers(claim_text)
    if not clean_text:
        return clean_text
    if not _needs_hedging(clean_text):
        return clean_text
    prefix = hedging_prefix.strip() or "Based on the cited facts, "
    if not prefix.endswith(" "):
        prefix += " "
    return f"{prefix}{_normalize_softened_lead(clean_text)}" if clean_text else clean_text


def _needs_hedging(text: str) -> bool:
    """Return ``True`` if *text* does NOT already contain hedging language."""
    existing = [
        "it appears", "it seems", "may have", "might be",
        "reportedly", "allegedly", "according to", "suggests that",
        "it has been suggested", "some sources indicate",
        "available information", "unverified", "uncertain",
        "possibly", "potentially",
    ]
    lower = text.lower()
    return not any(h in lower for h in existing)


_HEADER_FIX_RE = re.compile(r"([^\s#])([ \t]*)(#{1,6}\s)")


def _ensure_header_breaks(text: str) -> str:
    """Insert '\\n\\n' before any '#' header that isn't already on its own line.

    Skips fenced code blocks so ``# inside`` is preserved verbatim. This
    addresses synthesizer output where ``... margin of 8% [0]. ## Market``
    leaves the header inline and the markdown renderer treats it as prose.
    """
    parts = re.split(r"(```[\s\S]*?```)", text)
    fixups = 0
    for i in range(0, len(parts), 2):  # Even indices are non-code
        before = parts[i]
        parts[i] = _HEADER_FIX_RE.sub(
            lambda m: f"{m.group(1)}\n\n{m.group(3)}", before
        )
        if parts[i] != before:
            fixups += 1
    if fixups:
        logger.info(
            "DR_LEAK_TRACE phase=header_fix segments_fixed=%d total_len=%d",
            fixups, len(text),
        )
    return "".join(parts)


def _clean_empty_sections(content: str) -> str:
    """Remove section headers that have no content beneath them."""
    lines = content.split("\n")
    cleaned: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("#"):
            j = i + 1
            has_content = False
            while j < len(lines):
                next_stripped = lines[j].strip()
                if next_stripped.startswith("#"):
                    break
                if next_stripped:
                    has_content = True
                    break
                j += 1
            if has_content:
                cleaned.append(line)
        else:
            cleaned.append(line)
        i += 1
    return "\n".join(cleaned)


def _strip_reclaim_block_tags(content: str) -> str:
    """Remove internal reclaim role tags before returning final content."""
    content = re.sub(r"</?(?:analysis|free)>", "", content, flags=re.IGNORECASE)
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip()


def _recalculate_claim_positions(content: str, claims: list[ClaimInfo]) -> None:
    """Update claim positions to match modified content."""
    for claim in claims:
        new_pos = content.find(claim.claim_text)
        if new_pos >= 0:
            claim.position_start = new_pos
            claim.position_end = new_pos + len(claim.claim_text)
        # Otherwise keep original positions from parsing — still approximately correct


def _fallback_keyword_match(
    claim_text: str,
    evidence_pool: list[RankedEvidence],
    min_score: float = 0.25,
    scorer: Any | None = None,
) -> tuple[RankedEvidence | None, float]:
    """Find the best keyword-overlap match for a claim in the evidence pool.

    Uses the same word-overlap algorithm as ``ClaimGenerator.select_best_evidence``
    but with a lower threshold (0.25 vs 0.5) to catch partial matches.

    Returns:
        ``(best_evidence, score)`` or ``(None, 0.0)``.
    """
    claim_lower = claim_text.lower()
    words = set(re.findall(r"\b\w{3,}\b", claim_lower))
    if not words:
        return None, 0.0

    best_score = 0.0
    best_evidence: RankedEvidence | None = None

    for evidence in evidence_pool:
        quote_lower = evidence.quote_text.lower()
        matches = sum(1 for w in words if w in quote_lower)
        score = matches / len(words)
        if evidence.relevance_score:
            score = (score + evidence.relevance_score) / 2
        if callable(scorer):
            try:
                score = float(scorer(claim_text, evidence.quote_text))
            except Exception:
                logger.debug("FALLBACK_SCORER_FAILED", exc_info=True)

        if score > best_score:
            best_score = score
            best_evidence = evidence

    if best_score >= min_score:
        return best_evidence, best_score
    return None, 0.0


def _assign_fallback_evidence(
    claims: list[ClaimInfo],
    evidence_pool: list[RankedEvidence],
    scorer: Any | None = None,
) -> None:
    """Stage 2.5: Assign evidence to uncited claims via keyword matching.

    Iterates claims that have no evidence and attempts to find a match
    from the evidence pool using word-overlap scoring. Matched claims
    get ``has_fallback_evidence = True`` so downstream stages can
    distinguish them from LLM-cited claims.
    """
    uncited = [
        c for c in claims
        if c.claim_role == ClaimRole.FACT.value and c.evidence is None
    ]
    if not uncited:
        return

    matched = 0
    for claim in uncited:
        best, score = _fallback_keyword_match(
            claim.claim_text,
            evidence_pool,
            scorer=scorer,
        )
        if best is not None:
            claim.evidence = EvidenceInfo(
                source_url=best.source_url,
                canonical_source_url=best.canonical_source_url,
                source_title=best.source_title,
                quote_text=best.quote_text,
                start_offset=best.start_offset,
                end_offset=best.end_offset,
                section_heading=best.section_heading,
                relevance_score=best.relevance_score,
                has_numeric_content=best.has_numeric_content,
                source_pool_index=best.source_pool_index,
                evidence_pool_index=best.evidence_pool_index,
            )
            claim.evidences = [claim.evidence]
            claim.has_fallback_evidence = True
            claim.evidence_match_score = score
            matched += 1
            logger.debug(
                "FALLBACK_EVIDENCE_ASSIGNED claim=%s source_pool_index=%s evidence_pool_index=%s "
                "score=%.2f",
                _truncate(claim.claim_text, 100),
                best.source_pool_index,
                best.evidence_pool_index,
                score,
            )
        else:
            logger.debug(
                "FALLBACK_EVIDENCE_UNMATCHED claim=%s",
                _truncate(claim.claim_text, 100),
            )

    logger.info(
        "FALLBACK_EVIDENCE_MATCHING uncited=%d matched=%d unmatched=%d",
        len(uncited),
        matched,
        len(uncited) - matched,
    )


def _build_verification_summary(
    claims: list[ClaimInfo],
    correction_count: int = 0,
) -> VerificationSummaryInfo:
    """Build fact-only verification summary plus analysis-grounding summary."""
    fact_claims = [claim for claim in claims if claim.claim_role == ClaimRole.FACT.value]
    analysis_claims = [
        claim
        for claim in claims
        if claim.claim_role == ClaimRole.ANALYSIS.value and not claim.abstained
    ]

    total = len(fact_claims)
    supported = sum(1 for c in fact_claims if c.verification_verdict == "supported")
    partial = sum(1 for c in fact_claims if c.verification_verdict == "partial")
    unsupported = sum(1 for c in fact_claims if c.verification_verdict == "unsupported")
    contradicted = sum(1 for c in fact_claims if c.verification_verdict == "contradicted")
    abstained = sum(1 for c in fact_claims if c.abstained)

    editorial_fallback = sum(
        1
        for c in fact_claims
        if c.has_fallback_evidence
        and c.verification_verdict in ("unsupported", "contradicted")
    )
    rate_denominator = total - abstained - editorial_fallback
    if rate_denominator <= 0:
        rate_denominator = total - abstained if total > abstained else total
    supported_rate = supported / rate_denominator if rate_denominator > 0 else 0.0
    unsupported_rate = unsupported / rate_denominator if rate_denominator > 0 else 0.0
    contradicted_rate = contradicted / rate_denominator if rate_denominator > 0 else 0.0
    warning = supported_rate < 0.2 or unsupported_rate > 0.3 or contradicted_rate > 0.1

    analysis_total = len(analysis_claims)
    analysis_supported = sum(
        1 for c in analysis_claims if c.verification_verdict == "supported"
    )
    analysis_partial = sum(
        1 for c in analysis_claims if c.verification_verdict == "partial"
    )
    analysis_unsupported = sum(
        1 for c in analysis_claims if c.verification_verdict == "unsupported"
    )
    analysis_contradicted = sum(
        1 for c in analysis_claims if c.verification_verdict == "contradicted"
    )
    analysis_denominator = analysis_total if analysis_total > 0 else 0
    analysis_summary = AnalysisSummaryInfo(
        total_claims=analysis_total,
        supported_count=analysis_supported,
        partial_count=analysis_partial,
        unsupported_count=analysis_unsupported,
        contradicted_count=analysis_contradicted,
        grounded_rate=(
            analysis_supported / analysis_denominator if analysis_denominator > 0 else 0.0
        ),
        unsupported_rate=(
            analysis_unsupported / analysis_denominator if analysis_denominator > 0 else 0.0
        ),
        warning=(analysis_contradicted > 0 or analysis_unsupported > 0),
    )

    return VerificationSummaryInfo(
        total_claims=total,
        supported_count=supported,
        partial_count=partial,
        unsupported_count=unsupported,
        contradicted_count=contradicted,
        abstained_count=abstained,
        fact_rate_denominator=rate_denominator,
        supported_rate=supported_rate,
        unsupported_rate=unsupported_rate,
        contradicted_rate=contradicted_rate,
        warning=warning,
        citation_corrections=correction_count,
        analysis_summary=analysis_summary,
    )
