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
from hashlib import md5
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlparse

from databricks_deep_research.citation.citation_keys import build_citation_key_map
from databricks_deep_research.citation.config import CitationConfig
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
from databricks_deep_research.llm.client import FrameworkLLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _truncate(text: str, max_len: int = 100) -> str:
    """Truncate *text* to *max_len* characters with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _confidence_score_from_level(level: str | None) -> float:
    """Map a qualitative confidence label to a coarse numeric score."""
    mapping = {
        "high": 0.9,
        "medium": 0.6,
        "low": 0.3,
    }
    return mapping.get((level or "").lower(), 0.0)


def _counter_dict(values: list[str]) -> dict[str, int]:
    """Return a stable string counter for debug logging."""
    return dict(sorted(Counter(value for value in values if value).items()))


_ANALYSIS_ROLE_CUES = (
    "suggests",
    "may indicate",
    "appears consistent with",
    "appears to",
    "indicates",
    "reflects",
    "demonstrates",
    "shows that",
    "implies",
    "signals",
    "points to",
    "distorted",
    "obscured",
    "momentum",
    "trajectory",
    "resilience",
    "headwind",
    "tailwind",
    "positioned",
    "strong foundation",
    "healthy performance",
    "strong performance",
    "positive momentum",
    "complex earnings picture",
    "earnings picture",
    "growth driver",
    "bright spot",
    "current earnings trajectory",
    "essential context",
    "comparable store sales momentum",
)
_STRUCTURAL_TEXT_PATTERNS = (
    "introduction",
    "conclusion",
    "overview",
    "summary",
    "in summary",
    "overall",
    "the following sections",
    "this report examines",
    "this analysis examines",
)
_FACTUAL_PAYLOAD_PATTERNS = re.compile(
    r"\b("
    r"reported|increased|decreased|declined|reached|totaled|includes|announced|"
    r"delivered|generated|recorded|operating profit|operating loss|eps|sales|guidance|"
    r"quarter|full-year|fiscal|digital growth|ecommerce"
    r")\b",
    re.IGNORECASE,
)
_ANALYSIS_SPLIT_MARKERS = (
    " indicating ",
    " suggesting ",
    " reflecting ",
    " demonstrating ",
    " showing ",
    " highlighting ",
    " marking ",
    " continuing ",
    " which suggests ",
    " which indicates ",
    " which reflects ",
    " which demonstrates ",
)
_ANALYSIS_TAIL_PATTERN = re.compile(
    r",\s*(?:"
    r"marking|continuing|demonstrating|highlighting|suggesting|indicating|"
    r"reflecting|showing|underscoring"
    r")\b.*$",
    re.IGNORECASE,
)
_LEADING_CONCESSIVE_PATTERN = re.compile(
    r"^(?:(?:some sources indicate that|according to available information,|reportedly,)\s+)?"
    r"(?:(?:despite|while|although|though|however)\b[^,]*,\s*)+",
    re.IGNORECASE,
)
_MATERIAL_ANALYSIS_MARKERS = (
    "because",
    "due to",
    "driven by",
    "reflects",
    "indicates",
    "suggests",
    "demonstrates",
    "strongest",
    "weakest",
    "accelerating",
    "momentum",
    "trajectory",
    "healthy",
    "robust",
    "resilience",
    "outperformed",
    "exceeded expectations",
    "non-recurring",
)
_QUARTER_OR_DATE_PATTERN = re.compile(
    r"\b(?:q[1-4]|20\d{2}|first quarter|second quarter|third quarter|fourth quarter|full-year)\b",
    re.IGNORECASE,
)
_NUMERIC_TEXT_PATTERN = re.compile(
    r"[$€£]?\(?\d[\d,]*(?:\.\d+)?(?:\s*(?:%|million|billion|m|b|k|x))?\)?",
    re.IGNORECASE,
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

    async def synthesize_with_streaming(
        self,
        query: str,
        evidence_pool: list[RankedEvidence],
        previous_content: str,
        target_word_count: int,
        max_tokens: int,
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
        use_quick_verification: bool,
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

    async def retrieve_and_revise(
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

        # Sort by relevance and limit
        all_evidence.sort(key=lambda e: e.relevance_score, reverse=True)
        max_total = max_spans * min(len(indexed_sources), 10)
        all_evidence = all_evidence[:max_total]

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

        async for content, claim in self.claim_generator.synthesize_with_streaming(  # type: ignore[attr-defined]
            query=query,
            evidence_pool=evidence_pool,
            previous_content=previous_content,
            target_word_count=target_word_count,
            max_tokens=max_tokens,
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

    @staticmethod
    def _extract_numeric_tokens(text: str) -> set[str]:
        """Extract normalized numeric/financial tokens from text."""
        tokens: set[str] = set()
        for match in _NUMERIC_TEXT_PATTERN.findall(text):
            normalized = re.sub(r"\s+", "", match).lower().replace(",", "")
            if normalized:
                tokens.add(normalized)
        return tokens

    @staticmethod
    def _extract_temporal_tokens(text: str) -> set[str]:
        """Extract coarse quarter/year scope tokens from text."""
        return {
            match.strip().lower()
            for match in _QUARTER_OR_DATE_PATTERN.findall(text)
        }

    @staticmethod
    def _quote_overlap_score(claim_text: str, evidence_quote: str | None) -> float:
        """Compute deterministic word overlap without depending on classifier internals."""
        if not evidence_quote:
            return 0.0
        claim_words = set(re.findall(r"\b\w{4,}\b", claim_text.lower()))
        evidence_words = set(re.findall(r"\b\w{4,}\b", evidence_quote.lower()))
        if not claim_words:
            return 0.0
        return len(claim_words & evidence_words) / len(claim_words)

    def _has_exact_numeric_support(
        self,
        claim: ClaimInfo,
        evidence_quote: str | None,
    ) -> bool:
        """Return ``True`` when a numeric claim directly restates the evidence."""
        if claim.claim_type != "numeric" or not evidence_quote:
            return False

        claim_numbers = self._extract_numeric_tokens(claim.claim_text)
        evidence_numbers = self._extract_numeric_tokens(evidence_quote)
        if not claim_numbers or not claim_numbers.issubset(evidence_numbers):
            return False

        claim_temporal_tokens = self._extract_temporal_tokens(claim.claim_text)
        evidence_temporal_tokens = self._extract_temporal_tokens(evidence_quote)
        if claim_temporal_tokens and evidence_temporal_tokens:
            return not claim_temporal_tokens.isdisjoint(evidence_temporal_tokens)
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
        quote_overlap = self._quote_overlap_score(claim_text, evidence_quote)

        if self._has_exact_numeric_support(claim, evidence_quote):
            return ConfidenceResult(
                level=ConfidenceLevel.HIGH,
                score=0.95,
                indicators=["exact_numeric_match"],
                reasoning="Numeric claim exactly matches the cited evidence.",
            )

        if self._contains_material_analysis(lowered):
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

    @staticmethod
    def _claim_has_citation_keys(claim: ClaimInfo) -> bool:
        return bool(claim.citation_keys or claim.citation_key)

    @staticmethod
    def _looks_structural(text: str) -> bool:
        stripped = text.strip().lower()
        if not stripped:
            return True
        if stripped.startswith("#"):
            return True
        if re.match(r"^\*\*[^*]+\*\*:\s*", text.strip()):
            return True
        if stripped.endswith(":") and not _FACTUAL_PAYLOAD_PATTERNS.search(stripped):
            return True
        return any(pattern in stripped for pattern in _STRUCTURAL_TEXT_PATTERNS)

    @staticmethod
    def _contains_metric_payload(text: str) -> bool:
        for match in _NUMERIC_TEXT_PATTERN.finditer(text):
            raw = match.group(0).strip().lower()
            if re.fullmatch(r"20\d{2}", raw):
                continue
            if re.fullmatch(r"\(?\d+\)?", raw):
                continue
            return True
        return False

    @staticmethod
    def _contains_numeric_or_date_payload(text: str) -> bool:
        if CitationVerificationPipeline._contains_metric_payload(text):
            return True
        return bool(
            _QUARTER_OR_DATE_PATTERN.search(text)
            and _FACTUAL_PAYLOAD_PATTERNS.search(text)
        )

    @staticmethod
    def _contains_analysis_cues(text: str) -> bool:
        lowered = text.lower()
        return any(cue in lowered for cue in _ANALYSIS_ROLE_CUES)

    @staticmethod
    def _contains_material_analysis(text: str) -> bool:
        lowered = text.lower()
        return any(marker in lowered for marker in _MATERIAL_ANALYSIS_MARKERS)

    @staticmethod
    def _contains_factual_payload(claim: ClaimInfo) -> bool:
        text = claim.claim_text.strip()
        return bool(
            (
                claim.claim_type == "numeric"
                and CitationVerificationPipeline._contains_metric_payload(text)
            )
            or CitationVerificationPipeline._claim_has_citation_keys(claim)
            or CitationVerificationPipeline._contains_numeric_or_date_payload(text)
            or _FACTUAL_PAYLOAD_PATTERNS.search(text)
        )

    @staticmethod
    def _extract_factual_core(text: str) -> str | None:
        stripped = text.strip()
        lowered = stripped.lower()
        stripped = re.sub(_LEADING_CONCESSIVE_PATTERN, "", stripped).strip()
        lowered = stripped.lower()

        trimmed = re.sub(_ANALYSIS_TAIL_PATTERN, "", stripped).rstrip(" ,;:")
        if trimmed and trimmed != stripped:
            return trimmed

        for marker in _ANALYSIS_SPLIT_MARKERS:
            index = lowered.find(marker)
            if index <= 0:
                continue
            core = stripped[:index].rstrip(" ,;:")
            if core:
                core = re.sub(r"^(?:while|although|though)\s+", "", core, flags=re.IGNORECASE)
                return core
        return None

    def _classify_claim_role(
        self,
        claim: ClaimInfo,
    ) -> str:
        """Classify a claim as fact, analysis, or free after generation."""
        text = claim.claim_text.strip()
        explicit_role = claim.claim_role or ClaimRole.FACT.value

        if explicit_role == ClaimRole.FREE.value:
            if self._looks_structural(text) and not _FACTUAL_PAYLOAD_PATTERNS.search(text):
                return ClaimRole.FREE.value
            if (
                not self._claim_has_citation_keys(claim)
                and not _FACTUAL_PAYLOAD_PATTERNS.search(text)
                and not self._contains_numeric_or_date_payload(text)
            ):
                return ClaimRole.FREE.value
            if claim.claim_type == "numeric" and self._contains_metric_payload(text):
                return ClaimRole.FACT.value
            if self._contains_analysis_cues(text):
                return ClaimRole.ANALYSIS.value
            return ClaimRole.FACT.value

        if explicit_role == ClaimRole.ANALYSIS.value:
            if not self._contains_analysis_cues(text) and self._contains_factual_payload(claim):
                return ClaimRole.FACT.value
            return ClaimRole.ANALYSIS.value

        if self._looks_structural(text) and not self._contains_factual_payload(claim):
            return ClaimRole.FREE.value

        if self._contains_analysis_cues(text):
            if claim.claim_type == "numeric" and self._extract_factual_core(text):
                return ClaimRole.FACT.value
            return ClaimRole.ANALYSIS.value

        if claim.claim_type == "numeric":
            if not self._contains_metric_payload(text) and self._looks_structural(text):
                return ClaimRole.FREE.value
            return ClaimRole.FACT.value

        return ClaimRole.FACT.value

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
            claim.claim_role = self._classify_claim_role(claim)
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

            factual_core = self._extract_factual_core(claim.claim_text)
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
        """Resolve a claim evidence reference back to an evidence-pool index."""
        if claim_evidence.evidence_pool_index is not None:
            return claim_evidence.evidence_pool_index

        for index, evidence in enumerate(evidence_pool):
            if (
                evidence.source_url == claim_evidence.source_url
                and evidence.quote_text == claim_evidence.quote_text
            ):
                return evidence.evidence_pool_index if evidence.evidence_pool_index is not None else index

        if claim_evidence.source_pool_index is None:
            return None

        for index, evidence in enumerate(evidence_pool):
            if evidence.source_pool_index == claim_evidence.source_pool_index:
                return evidence.evidence_pool_index if evidence.evidence_pool_index is not None else index

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
                claim.evidence = EvidenceInfo(
                    source_url=result.corrected_evidence.source_url or "",
                    canonical_source_url=result.corrected_evidence.canonical_source_url,
                    source_title=result.corrected_evidence.source_title,
                    quote_text=result.corrected_evidence.quote_text,
                    start_offset=result.corrected_evidence.start_offset,
                    end_offset=result.corrected_evidence.end_offset,
                    section_heading=result.corrected_evidence.section_heading,
                    relevance_score=result.corrected_evidence.relevance_score,
                    has_numeric_content=result.corrected_evidence.has_numeric_content,
                    source_pool_index=result.corrected_evidence.source_pool_index,
                    evidence_pool_index=result.corrected_evidence.evidence_pool_index,
                )
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
                    alternate_info = EvidenceInfo(
                        source_url=alternate.source_url or "",
                        canonical_source_url=alternate.canonical_source_url,
                        source_title=alternate.source_title,
                        quote_text=alternate.quote_text,
                        start_offset=alternate.start_offset,
                        end_offset=alternate.end_offset,
                        section_heading=alternate.section_heading,
                        relevance_score=alternate.relevance_score,
                        has_numeric_content=alternate.has_numeric_content,
                        source_pool_index=alternate.source_pool_index,
                        evidence_pool_index=alternate.evidence_pool_index,
                    )
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

        for claim in claims:
            if claim.abstained:
                continue

            if (
                claim.claim_role == ClaimRole.FACT.value
                and claim.verification_text
                and claim.verification_text != claim.claim_text
                and claim.verification_verdict in ("supported", "partial")
            ):
                modifications.append(("rewrite", claim))

            if claim.claim_role == ClaimRole.ANALYSIS.value:
                if claim.verification_verdict == "contradicted":
                    modifications.append(("remove", claim))
                elif claim.verification_verdict == "unsupported":
                    action = (
                        "remove"
                        if self._contains_numeric_or_date_payload(claim.claim_text)
                        else "soften_analysis"
                    )
                    modifications.append((action, claim))
                elif claim.verification_verdict == "partial":
                    modifications.append(("soften_analysis", claim))
                continue

            if claim.verification_verdict == "contradicted":
                modifications.append(("remove", claim))
            elif claim.verification_verdict == "unsupported":
                modifications.append(("soften", claim))

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
                    "CLAIM_REMOVED claim=%s verdict=%s",
                    _truncate(claim.claim_text, 50),
                    claim.verification_verdict,
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

        logger.info(
            "STAGE8_COMPLETE removed=%d softened=%d total=%d",
            removed_count,
            softened_count,
            len(modifications),
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

        # ------------------------------------------------------------------
        # Stage 1: Pre-select evidence
        # ------------------------------------------------------------------
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
        if draft_content is not None:
            from databricks_deep_research.citation.claim_generator import (
                _parse_interleaved_content,
            )

            parsed_claims = _parse_interleaved_content(
                draft_content,
                evidence_pool,
            )
            for claim in parsed_claims:
                generated_claims.append(
                    ClaimInfo(
                        claim_text=claim.claim_text,
                        claim_type=claim.claim_type,
                        position_start=claim.position_start,
                        position_end=claim.position_end,
                        evidence=(
                            EvidenceInfo(
                                source_url=claim.evidence.source_url or "",
                                canonical_source_url=claim.evidence.canonical_source_url,
                                source_title=claim.evidence.source_title,
                                quote_text=claim.evidence.quote_text,
                                start_offset=claim.evidence.start_offset,
                                end_offset=claim.evidence.end_offset,
                                section_heading=claim.evidence.section_heading,
                                relevance_score=claim.evidence.relevance_score,
                                has_numeric_content=claim.evidence.has_numeric_content,
                                source_pool_index=claim.evidence.source_pool_index,
                                evidence_pool_index=claim.evidence.evidence_pool_index,
                            )
                            if claim.evidence
                            else None
                        ),
                        evidences=[
                            EvidenceInfo(
                                source_url=evidence.source_url or "",
                                canonical_source_url=evidence.canonical_source_url,
                                source_title=evidence.source_title,
                                quote_text=evidence.quote_text,
                                start_offset=evidence.start_offset,
                                end_offset=evidence.end_offset,
                                section_heading=evidence.section_heading,
                                relevance_score=evidence.relevance_score,
                                has_numeric_content=evidence.has_numeric_content,
                                source_pool_index=evidence.source_pool_index,
                                evidence_pool_index=evidence.evidence_pool_index,
                            )
                            for evidence in claim.evidences
                        ],
                        citation_key=claim.citation_key,
                        citation_keys=claim.citation_keys,
                        claim_role=claim.claim_role,
                        verification_text=claim.verification_text,
                        analysis_parent_claim_indices=claim.analysis_parent_claim_indices,
                        from_free_block=claim.from_free_block,
                    )
                )
        else:
            async for content, claim in self.generate_with_interleaving(
                evidence_pool=evidence_pool,
                observations=observations,
                query=query,
                target_word_count=target_word_count,
                max_tokens=max_tokens,
            ):
                if content:
                    full_content = content
                    yield content

                if claim:
                    claim_info = ClaimInfo(
                        claim_text=claim.claim_text,
                        claim_type=claim.claim_type,
                        position_start=claim.position_start,
                        position_end=claim.position_end,
                        evidence=(
                            EvidenceInfo(
                                source_url=claim.evidence.source_url or "",
                                canonical_source_url=claim.evidence.canonical_source_url,
                                source_title=claim.evidence.source_title,
                                quote_text=claim.evidence.quote_text,
                                start_offset=claim.evidence.start_offset,
                                end_offset=claim.evidence.end_offset,
                                section_heading=claim.evidence.section_heading,
                                relevance_score=claim.evidence.relevance_score,
                                has_numeric_content=claim.evidence.has_numeric_content,
                                source_pool_index=claim.evidence.source_pool_index,
                                evidence_pool_index=claim.evidence.evidence_pool_index,
                            )
                            if claim.evidence
                            else None
                        ),
                        evidences=[
                            EvidenceInfo(
                                source_url=evidence.source_url or "",
                                canonical_source_url=evidence.canonical_source_url,
                                source_title=evidence.source_title,
                                quote_text=evidence.quote_text,
                                start_offset=evidence.start_offset,
                                end_offset=evidence.end_offset,
                                section_heading=evidence.section_heading,
                                relevance_score=evidence.relevance_score,
                                has_numeric_content=evidence.has_numeric_content,
                                source_pool_index=evidence.source_pool_index,
                                evidence_pool_index=evidence.evidence_pool_index,
                            )
                            for evidence in claim.evidences
                        ],
                        citation_key=claim.citation_key,
                        citation_keys=claim.citation_keys,
                        claim_role=claim.claim_role,
                        verification_text=claim.verification_text,
                        analysis_parent_claim_indices=claim.analysis_parent_claim_indices,
                        from_free_block=claim.from_free_block,
                    )

                    generated_claims.append(claim_info)

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

        # ------------------------------------------------------------------
        # Stage 3: Classify confidence after role enforcement and fallback evidence
        # ------------------------------------------------------------------
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

        logger.info(
            "CITATION_PIPELINE_STAGE3_RESULT routes=%s",
            _counter_dict([claim.confidence_level or "" for claim in generated_claims]),
        )

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

        # ------------------------------------------------------------------
        # Stage 5: Correct citations for non-supported claims
        # ------------------------------------------------------------------
        correction_count = 0
        async for event in self.correct_citations(generated_claims, evidence_pool):
            if event.event_type == "correction_metrics":
                correction_count = event.data.get("total_corrected", 0)
            yield event

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

        # ------------------------------------------------------------------
        # Stage 4b: Verify analysis after facts are finalized
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Stage 8: Post-verification claim modification
        # ------------------------------------------------------------------
        stage_8_removed = 0
        stage_8_softened = 0
        stage_8_rewritten = 0

        # Always run post-verification claim modification
        (
            full_content,
            stage_8_removed,
            stage_8_softened,
            stage_8_rewritten,
        ) = await self.process_unverified_claims(full_content, generated_claims)

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
    hedge_idx = md5(clean_text.encode("utf-8")).digest()[0] % len(hedging_phrases)
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
