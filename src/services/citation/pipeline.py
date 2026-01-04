"""Citation Verification Pipeline - Orchestrates the 7-stage citation verification.

This service coordinates the entire citation verification workflow:
1. Evidence Pre-Selection (before generation)
2. Interleaved Generation (during synthesis)
3. Confidence Classification (after claim extraction)
4. Isolated Verification (for each claim)
5. Citation Correction (post-hoc fixes)
6. Numeric QA Verification (for numeric claims)
7. ARE-style Verification Retrieval (atomic fact decomposition + external search)

Stage 7 implements the ARE (Atomic fact decomposition-based Retrieval and Editing)
pattern for verifying and revising unsupported/partial claims. It decomposes claims
into atomic facts, searches for evidence (internal pool first, then external Brave),
and reconstructs claims with verified facts or hedging language for unverified parts.

Scientific basis: ARE (arXiv:2410.16708), FActScore (EMNLP 2023), SAFE (DeepMind)
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mlflow

from src.agent.config import get_citation_config_for_depth
from src.agent.state import ClaimInfo, EvidenceInfo, ResearchState, SourceInfo

if TYPE_CHECKING:
    from src.agent.nodes.react_synthesizer import ParsedContent
    from src.agent.tools.web_crawler import WebCrawler
    from src.services.search.brave import BraveSearchClient
from src.core.app_config import (
    CitationVerificationConfig,
    GenerationMode,
    GroundingValidationConfig,
    SynthesisMode,
    get_app_config,
)
from src.core.logging_utils import get_logger, truncate
from src.services.citation.citation_corrector import CitationCorrector, CorrectionType
from src.services.citation.claim_generator import InterleavedClaim, InterleavedGenerator
from src.services.citation.confidence_classifier import ConfidenceClassifier
from src.services.citation.content_evaluator import evaluate_content_quality
from src.services.citation.evidence_selector import EvidencePreSelector, RankedEvidence
from src.services.citation.isolated_verifier import IsolatedVerifier
from src.services.citation.numeric_verifier import NumericVerifier
from src.services.citation.atomic_decomposer import ClaimRevision
from src.services.citation.verification_retriever import (
    NewExternalEvidence,
    VerificationEvent as Stage7Event,
    VerificationRetriever,
    VerificationRetrievalMetrics,
)
from src.services.llm.client import LLMClient

logger = get_logger(__name__)


@dataclass
class VerificationEvent:
    """Event emitted during citation verification."""

    event_type: str  # claim_generated, claim_verified, citation_corrected, etc.
    data: dict[str, Any]


@dataclass
class ValidationIssue:
    """Represents a grounding validation issue for <analysis> or <free> blocks.

    Based on SOTA research (FACTS Grounding, FActScore, SAFE):
    - Analysis blocks should be derivable from preceding citations
    - Free blocks should contain only structural content
    """

    block_index: int
    issue_type: str  # "ungrounded_analysis", "baseless_analysis", "hidden_claims_in_free"
    message: str
    content: str | None = None  # First 100 chars of the problematic content
    reason: str | None = None  # LLM judge's explanation (if applicable)


class CitationVerificationPipeline:
    """Orchestrates the 7-stage citation verification pipeline.

    Integrates with the Synthesizer to provide claim-level attribution:
    - Pre-selects evidence from sources before generation
    - Generates claims constrained by available evidence
    - Verifies each claim in isolation
    - Corrects citations if needed
    - Verifies and revises unsupported claims using ARE pattern (Stage 7)
    """

    def __init__(
        self,
        llm: LLMClient,
        depth: str | None = None,
        brave_client: BraveSearchClient | None = None,
        web_crawler: WebCrawler | None = None,
    ):
        """Initialize the pipeline with LLM client.

        Args:
            llm: LLM client for all stages.
            depth: Research depth (light/medium/extended) for per-depth config.
                   If None, uses global citation_verification config.
            brave_client: Optional Brave Search client for Stage 7 external search.
            web_crawler: Optional web crawler for Stage 7 external search.
        """
        self.llm = llm
        self.depth = depth
        self.brave_client = brave_client
        self.web_crawler = web_crawler

        # Get config: per-depth if specified, otherwise global
        if depth:
            self.config: CitationVerificationConfig = get_citation_config_for_depth(depth)
        else:
            self.config = get_app_config().citation_verification

        self.evidence_selector = EvidencePreSelector(llm)
        self.claim_generator = InterleavedGenerator(llm)
        self.confidence_classifier = ConfidenceClassifier()
        self.verifier = IsolatedVerifier(llm)
        self.numeric_verifier = NumericVerifier(llm)
        self.citation_corrector = CitationCorrector(llm)

        # Stage 7: Verification Retriever (initialized lazily when needed)
        self._verification_retriever: VerificationRetriever | None = None

    def _get_verification_retriever(self) -> VerificationRetriever | None:
        """Get or create the verification retriever for Stage 7.

        Returns None if Stage 7 is disabled or dependencies are missing.
        """
        if not self.config.enable_verification_retrieval:
            return None

        if self._verification_retriever is None:
            self._verification_retriever = VerificationRetriever(
                llm=self.llm,
                brave_client=self.brave_client,
                web_crawler=self.web_crawler,
                config=self.config.verification_retrieval,
            )

        return self._verification_retriever

    @mlflow.trace(name="citation.stage_1.preselect", span_type="CHAIN")
    async def preselect_evidence(
        self,
        sources: list[SourceInfo],
        query: str,
    ) -> list[RankedEvidence]:
        """Stage 1: Pre-select evidence spans from sources.

        Args:
            sources: List of sources with content.
            query: Research query for relevance scoring.

        Returns:
            List of ranked evidence spans.
        """
        if not self.config.enable_evidence_preselection:
            logger.info("CITATION_PIPELINE", stage=1, action="skipped", reason="disabled")
            return []

        logger.info(
            "CITATION_PIPELINE_STAGE1",
            sources_count=len(sources),
            query=truncate(query, 50),
        )

        # Convert SourceInfo objects to dicts for evidence selector
        source_dicts = [
            {
                "url": source.url,
                "title": source.title,
                "content": source.content,
            }
            for source in sources
            if source.content
        ]

        if not source_dicts:
            logger.warning("CITATION_PIPELINE_STAGE1", action="no_sources_with_content")
            return []

        # Filter sources by content quality - discard abstract-only, paywalled, low-quality
        high_quality_sources = []
        for source in source_dicts:
            content = source.get("content") or ""
            quality = evaluate_content_quality(content, query)
            # Accept sources with score >= 0.5 that aren't abstract-only
            if quality.score >= 0.5 and not quality.is_abstract_only:
                high_quality_sources.append(source)
                logger.debug(
                    "SOURCE_QUALITY_ACCEPTED",
                    url=truncate(source.get("url", ""), 50),
                    score=round(quality.score, 2),
                    reason=quality.reason,
                )
            else:
                logger.debug(
                    "SOURCE_QUALITY_REJECTED",
                    url=truncate(source.get("url", ""), 50),
                    score=round(quality.score, 2),
                    reason=quality.reason,
                )

        if not high_quality_sources:
            logger.warning(
                "CITATION_PIPELINE_STAGE1",
                action="no_high_quality_sources",
                total_sources=len(source_dicts),
            )
            # Fall back to all sources if none pass quality filter
            high_quality_sources = source_dicts

        logger.info(
            "CITATION_PIPELINE_STAGE1_QUALITY_FILTER",
            total_sources=len(source_dicts),
            high_quality_sources=len(high_quality_sources),
            filtered_out=len(source_dicts) - len(high_quality_sources),
        )

        # Use high-quality sources for evidence extraction
        source_dicts = high_quality_sources

        max_spans = self.config.evidence_preselection.max_spans_per_source

        try:
            all_evidence = await self.evidence_selector.select_evidence_spans(
                query=query,
                sources=source_dicts,
                max_spans_per_source=max_spans,
            )
        except Exception as e:
            logger.warning(
                "CITATION_PIPELINE_STAGE1_ERROR",
                error=str(e)[:100],
            )
            return []

        # Sort by relevance and limit total evidence
        all_evidence.sort(key=lambda e: e.relevance_score, reverse=True)
        max_total = max_spans * min(len(sources), 10)
        all_evidence = all_evidence[:max_total]

        logger.info(
            "CITATION_PIPELINE_STAGE1_COMPLETE",
            total_evidence=len(all_evidence),
        )

        return all_evidence

    @mlflow.trace(name="citation.stage_2.generate", span_type="CHAIN")
    async def generate_with_interleaving(
        self,
        evidence_pool: list[RankedEvidence],
        observations: list[str],
        query: str,
        target_word_count: int = 600,
        max_tokens: int = 2000,
    ) -> AsyncGenerator[tuple[str, InterleavedClaim | None], None]:
        """Stage 2: Generate synthesis with interleaved claim/evidence pairs.

        Args:
            evidence_pool: Pre-selected evidence from Stage 1.
            observations: Research observations (used as context).
            query: Original research query.
            target_word_count: Target word count for the report.
            max_tokens: Maximum tokens to generate.

        Yields:
            Tuples of (content_chunk, claim_if_generated).
            - First yields the full content (with markdown preserved)
            - Then yields claims for verification
        """
        if not self.config.enable_interleaved_generation:
            logger.info("CITATION_PIPELINE", stage=2, action="skipped", reason="disabled")
            # Fall back to standard synthesis
            return

        logger.info(
            "CITATION_PIPELINE_STAGE2_START",
            evidence_count=len(evidence_pool),
            observations_count=len(observations),
            target_word_count=target_word_count,
            max_tokens=max_tokens,
            generation_mode=self.config.generation_mode.value,
        )

        # Build context from observations for previous_content parameter
        previous_content = "\n\n".join(observations) if observations else ""

        claim_index = 0
        async for content, claim in self.claim_generator.synthesize_with_streaming(
            query=query,
            evidence_pool=evidence_pool,
            previous_content=previous_content,
            target_word_count=target_word_count,
            max_tokens=max_tokens,
        ):
            if content:
                # Yield the raw markdown content
                yield content, None
            if claim:
                claim_index += 1
                logger.debug(
                    "CLAIM_GENERATED",
                    claim_index=claim_index,
                    claim_text=truncate(claim.claim_text, 50),
                )
                # Yield the claim for verification
                yield "", claim

    @mlflow.trace(name="citation.stage_3.verify", span_type="CHAIN")
    async def verify_claims(
        self,
        claims: list[ClaimInfo],
    ) -> AsyncGenerator[VerificationEvent, None]:
        """Stage 4: Verify claims in isolation.

        Uses isolated verification to prevent generation context bias.

        Args:
            claims: Claims to verify.

        Yields:
            VerificationEvent for each claim verified.
        """
        if not self.config.isolated_verification:
            logger.info("CITATION_PIPELINE", stage=4, action="skipped", reason="disabled")
            return

        logger.info(
            "CITATION_PIPELINE_STAGE4",
            claims_count=len(claims),
        )

        for claim in claims:
            if not claim.evidence:
                # No evidence to verify against
                claim.abstained = True
                continue

            try:
                # Convert EvidenceInfo to RankedEvidence for verifier
                evidence = RankedEvidence(
                    source_id=None,
                    source_url=claim.evidence.source_url,
                    source_title=None,  # EvidenceInfo doesn't have title
                    quote_text=claim.evidence.quote_text,
                    start_offset=claim.evidence.start_offset,
                    end_offset=claim.evidence.end_offset,
                    section_heading=claim.evidence.section_heading,
                    relevance_score=claim.evidence.relevance_score or 0.0,
                    has_numeric_content=claim.evidence.has_numeric_content,
                )

                # Use quick verification for high-confidence claims
                use_quick = self.confidence_classifier.should_use_quick_verification(
                    claim.claim_text,
                    claim.evidence.quote_text if claim.evidence else None,
                )

                result = await self.verifier.verify_with_isolation(
                    claim_text=claim.claim_text,
                    evidence=evidence,
                    use_quick_verification=use_quick,
                )

                # Update claim with verification result
                claim.verification_verdict = result.verdict.value
                claim.verification_reasoning = result.reasoning

                yield VerificationEvent(
                    event_type="claim_verified",
                    data={
                        "claim_id": id(claim),  # Temporary ID
                        "claim_text": claim.claim_text,
                        "position_start": claim.position_start,
                        "position_end": claim.position_end,
                        "verdict": result.verdict.value,
                        "confidence_level": claim.confidence_level,
                        "evidence_preview": truncate(claim.evidence.quote_text, 100),
                        "reasoning": result.reasoning,
                    },
                )

                # Stage 6: Numeric QA verification for numeric claims
                if claim.claim_type == "numeric" and self.config.enable_numeric_qa_verification:
                    numeric_result = await self.verify_numeric_claim(claim, evidence)
                    if numeric_result:
                        parsed = numeric_result.get("parsed_value", {})
                        yield VerificationEvent(
                            event_type="numeric_claim_detected",
                            data={
                                "claim_id": id(claim),
                                "raw_value": parsed.get("raw_text", ""),
                                "normalized_value": parsed.get("normalized_value"),
                                "unit": parsed.get("unit"),
                                "derivation_type": numeric_result.get("derivation_type", "direct"),
                                "qa_verified": numeric_result.get("overall_match", False),
                            },
                        )

                logger.debug(
                    "CLAIM_VERIFIED",
                    verdict=result.verdict.value,
                    claim=truncate(claim.claim_text, 50),
                    is_numeric=claim.claim_type == "numeric",
                )

            except Exception as e:
                logger.warning(
                    "CLAIM_VERIFICATION_ERROR",
                    claim=truncate(claim.claim_text, 50),
                    error=str(e)[:100],
                )
                claim.abstained = True

    def classify_confidence(self, claim: ClaimInfo) -> str:
        """Stage 3: Classify confidence level using the ConfidenceClassifier.

        Uses linguistic heuristics (HaluGate-style) since Foundation Model
        APIs don't expose logprobs.

        Args:
            claim: Claim to classify.

        Returns:
            Confidence level: "high", "medium", or "low".
        """
        if not self.config.enable_confidence_classification:
            return "medium"

        evidence_quote = claim.evidence.quote_text if claim.evidence else None
        result = self.confidence_classifier.classify(claim.claim_text, evidence_quote)
        return result.level.value

    @mlflow.trace(name="citation.stage_6.verify_numeric", span_type="CHAIN")
    async def verify_numeric_claim(
        self,
        claim: ClaimInfo,
        evidence: RankedEvidence,
    ) -> dict[str, Any]:
        """Stage 6: Verify a numeric claim using QA-based verification.

        Args:
            claim: The numeric claim to verify.
            evidence: The evidence to verify against.

        Returns:
            Numeric verification result as dict.
        """
        if not self.config.enable_numeric_qa_verification:
            return {}

        if claim.claim_type != "numeric":
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
                "normalized_value": float(result.parsed_value.normalized_value)
                if result.parsed_value.normalized_value else None,
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

    @mlflow.trace(name="citation.stage_5.correct", span_type="CHAIN")
    async def correct_citations(
        self,
        claims: list[ClaimInfo],
        evidence_pool: list[RankedEvidence],
    ) -> AsyncGenerator[VerificationEvent, None]:
        """Stage 5: Correct citations for claims that are not fully supported.

        Uses CiteFix hybrid keyword+semantic matching to find better evidence.

        Args:
            claims: Claims to correct.
            evidence_pool: Pool of available evidence.

        Yields:
            VerificationEvent for each citation corrected.
        """
        if not self.config.enable_citation_correction:
            logger.info("CITATION_PIPELINE", stage=5, action="skipped", reason="disabled")
            return

        # Only correct claims that are not fully supported
        claims_to_correct = [
            (
                c.claim_text,
                RankedEvidence(
                    source_id=None,
                    source_url=c.evidence.source_url if c.evidence else "",
                    source_title=None,
                    quote_text=c.evidence.quote_text if c.evidence else "",
                    start_offset=c.evidence.start_offset if c.evidence else 0,
                    end_offset=c.evidence.end_offset if c.evidence else 0,
                    section_heading=c.evidence.section_heading if c.evidence else None,
                    relevance_score=c.evidence.relevance_score or 0.0 if c.evidence else 0.0,
                    has_numeric_content=c.evidence.has_numeric_content if c.evidence else False,
                ) if c.evidence else None,
                c.verification_verdict,
            )
            for c in claims
            if c.verification_verdict != "supported"
        ]

        if not claims_to_correct:
            logger.info("CITATION_PIPELINE_STAGE5", action="no_claims_to_correct")
            return

        logger.info(
            "CITATION_PIPELINE_STAGE5",
            claims_to_correct=len(claims_to_correct),
        )

        results, metrics = await self.citation_corrector.correct_citations(
            claims_with_evidence=claims_to_correct,
            evidence_pool=evidence_pool,
        )

        # Update claims with corrections and emit events
        for claim, result in zip(
            [c for c in claims if c.verification_verdict != "supported"],
            results,
            strict=False,
        ):
            if result.correction_type == CorrectionType.REPLACE and result.corrected_evidence:
                # Update claim's evidence with corrected citation
                claim.evidence = EvidenceInfo(
                    source_url=result.corrected_evidence.source_url or "",
                    quote_text=result.corrected_evidence.quote_text,
                    start_offset=result.corrected_evidence.start_offset,
                    end_offset=result.corrected_evidence.end_offset,
                    section_heading=result.corrected_evidence.section_heading,
                    relevance_score=result.corrected_evidence.relevance_score,
                    has_numeric_content=result.corrected_evidence.has_numeric_content,
                )

                yield VerificationEvent(
                    event_type="citation_corrected",
                    data={
                        "claim_id": id(claim),
                        "claim_text": claim.claim_text,
                        "correction_type": result.correction_type.value,
                        "original_evidence": result.original_evidence.quote_text if result.original_evidence else "",
                        "corrected_evidence": result.corrected_evidence.quote_text,
                        "reasoning": result.reasoning,
                    },
                )

            elif result.correction_type == CorrectionType.REMOVE:
                # No suitable evidence found - demote to unsupported for softening
                # This ensures the claim goes through _soften_claim() in post-processing
                # which removes the citation marker and adds hedging language
                logger.warning(
                    "CITATION_DEMOTED_TO_UNSUPPORTED",
                    claim_text=truncate(claim.claim_text, 100),
                    original_verdict=claim.verification_verdict,
                    reason="no_suitable_evidence",
                    citation_key=claim.citation_key,
                )
                claim.verification_verdict = "unsupported"
                # Note: NOT setting abstained, so post-processing will soften it

                yield VerificationEvent(
                    event_type="citation_corrected",
                    data={
                        "claim_id": id(claim),
                        "claim_text": claim.claim_text,
                        "correction_type": result.correction_type.value,
                        "reasoning": result.reasoning,
                        "action": "demoted_to_unsupported",
                    },
                )

            elif result.correction_type == CorrectionType.ADD_ALTERNATE:
                # Store alternate citations (in a real impl, would persist)
                yield VerificationEvent(
                    event_type="citation_corrected",
                    data={
                        "claim_id": id(claim),
                        "claim_text": claim.claim_text,
                        "correction_type": result.correction_type.value,
                        "alternate_count": len(result.alternate_evidence),
                        "reasoning": result.reasoning,
                    },
                )

        # Emit correction metrics
        yield VerificationEvent(
            event_type="correction_metrics",
            data={
                "total_corrected": metrics.replaced + metrics.removed + metrics.added_alternate,
                "kept": metrics.kept,
                "replaced": metrics.replaced,
                "removed": metrics.removed,
                "added_alternate": metrics.added_alternate,
                "correction_rate": metrics.correction_rate,
            },
        )

    async def _run_classical_synthesis(
        self,
        state: ResearchState,
    ) -> AsyncGenerator[str, None]:
        """Run classical free-form synthesis without citation markers.

        Uses the existing stream_synthesis approach with inline [Title](url) links.
        Best text quality, skips verification stages 3-6.

        Args:
            state: Current research state.

        Yields:
            Content chunks from stream_synthesis.
        """
        from src.agent.nodes.synthesizer import stream_synthesis

        logger.info(
            "CLASSICAL_SYNTHESIS_START",
            sources=len(state.sources),
            observations=len(state.all_observations),
        )

        full_content = ""
        async for chunk in stream_synthesis(state, self.llm):
            full_content += chunk
            yield chunk

        # Complete state with full content
        state.complete(full_content)

        logger.info(
            "CLASSICAL_SYNTHESIS_COMPLETE",
            content_length=len(full_content),
        )

    async def run_full_pipeline(
        self,
        state: ResearchState,
        target_word_count: int = 600,
        max_tokens: int = 2000,
    ) -> AsyncGenerator[VerificationEvent | str, None]:
        """Run the complete citation verification pipeline during synthesis.

        Supports three generation modes:
        - "classical": Free-form prose with [Title](url) links, skips verification
        - "natural": Light-touch [N] citations with verification stages 3-6
        - "strict": Heavy [N] constraints with verification stages 3-6

        Integrates all stages (for natural/strict modes):
        1. Pre-select evidence from sources
        2. Generate synthesis with interleaved claims
        3. Classify claim confidence
        4. Verify claims in isolation
        5. Correct citations
        6. Numeric QA verification

        Args:
            state: Current research state.
            target_word_count: Target word count for the report.
            max_tokens: Maximum tokens to generate.

        Yields:
            Content chunks (str) and VerificationEvents.
        """
        if not state.enable_citation_verification:
            logger.info("CITATION_PIPELINE", action="disabled_globally")
            return

        # Check generation mode and route accordingly
        generation_mode = self.config.generation_mode

        logger.info(
            "CITATION_PIPELINE_START",
            generation_mode=generation_mode.value,
            sources=len(state.sources),
            observations=len(state.all_observations),
            target_word_count=target_word_count,
            max_tokens=max_tokens,
        )

        # Classical mode: use free-form synthesis, skip verification stages
        if generation_mode == GenerationMode.CLASSICAL:
            logger.info(
                "CITATION_PIPELINE_MODE",
                mode="classical",
                action="using_stream_synthesis",
                skip_verification=True,
            )
            async for chunk in self._run_classical_synthesis(state):
                yield chunk

            # Emit summary event indicating verification was skipped
            yield VerificationEvent(
                event_type="verification_summary",
                data={
                    "message_id": str(state.session_id),
                    "generation_mode": "classical",
                    "verification_skipped": True,
                    "total_claims": 0,
                    "supported": 0,
                    "partial": 0,
                    "unsupported": 0,
                    "contradicted": 0,
                    "abstained_count": 0,
                    "citation_corrections": 0,
                    "warning": False,
                },
            )
            return

        # Check synthesis mode: ReAct or Interleaved
        synthesis_mode = self.config.synthesis_mode

        # Natural/Strict modes: run full verification pipeline
        logger.info(
            "CITATION_PIPELINE_MODE",
            mode=generation_mode.value,
            synthesis_mode=synthesis_mode.value,
            action="running_verification_pipeline",
        )

        # Stage 1: Pre-select evidence
        evidence_pool = await self.preselect_evidence(
            sources=state.sources,
            query=state.query,
        )

        # Log Stage 1 results
        sources_with_content = sum(1 for s in state.sources if s.content)
        logger.info(
            "CITATION_PIPELINE_STAGE1_RESULT",
            evidence_count=len(evidence_pool),
            sources_count=len(state.sources),
            sources_with_content=sources_with_content,
        )

        # Fallback to classical synthesis if no evidence found
        if not evidence_pool:
            logger.warning(
                "CITATION_PIPELINE_EMPTY_EVIDENCE",
                sources_count=len(state.sources),
                sources_with_content=sources_with_content,
                action="falling_back_to_classical",
            )
            # Fall back to classical synthesis which produces [Title](url) links
            async for chunk in self._run_classical_synthesis(state):
                yield chunk
            # Emit summary indicating verification was skipped due to empty evidence
            yield VerificationEvent(
                event_type="verification_summary",
                data={
                    "message_id": str(state.session_id),
                    "generation_mode": "classical_fallback",
                    "verification_skipped": True,
                    "reason": "empty_evidence_pool",
                    "total_claims": 0,
                    "supported": 0,
                    "partial": 0,
                    "unsupported": 0,
                    "contradicted": 0,
                    "abstained_count": 0,
                    "warning": False,
                },
            )
            return

        # Store evidence in state
        for evidence in evidence_pool:
            state.add_evidence(
                EvidenceInfo(
                    source_url=evidence.source_url or "",
                    quote_text=evidence.quote_text,
                    start_offset=evidence.start_offset,
                    end_offset=evidence.end_offset,
                    section_heading=evidence.section_heading,
                    relevance_score=evidence.relevance_score,
                    has_numeric_content=evidence.has_numeric_content,
                )
            )

        # Route based on synthesis mode
        if synthesis_mode == SynthesisMode.REACT:
            logger.info(
                "CITATION_PIPELINE_REACT_MODE",
                evidence_count=len(evidence_pool),
                action="using_react_synthesis",
            )
            async for event in self._run_react_synthesis(
                state=state,
                evidence_pool=evidence_pool,
                max_tokens=max_tokens,
            ):
                yield event
            return

        # Stage 2: Generate with interleaved claims (INTERLEAVED mode)
        # Stream raw LLM content first (preserving markdown), then collect claims
        generated_claims: list[ClaimInfo] = []
        full_content = ""

        async for content, claim in self.generate_with_interleaving(
            evidence_pool=evidence_pool,
            observations=state.all_observations,
            query=state.query,
            target_word_count=target_word_count,
            max_tokens=max_tokens,
        ):
            if content:
                # Stream the raw markdown content (preserves structure)
                full_content = content  # Store the full content
                yield content  # Stream to client

            if claim:
                # Convert to ClaimInfo and add to state
                claim_info = ClaimInfo(
                    claim_text=claim.claim_text,
                    claim_type="numeric" if claim.evidence and claim.evidence.has_numeric_content else "general",
                    position_start=claim.position_start,
                    position_end=claim.position_end,
                    evidence=EvidenceInfo(
                        source_url=claim.evidence.source_url or "",
                        quote_text=claim.evidence.quote_text,
                        start_offset=claim.evidence.start_offset,
                        end_offset=claim.evidence.end_offset,
                        section_heading=claim.evidence.section_heading,
                        relevance_score=claim.evidence.relevance_score,
                        has_numeric_content=claim.evidence.has_numeric_content,
                    ) if claim.evidence else None,
                    citation_key=claim.citation_key,
                    citation_keys=claim.citation_keys,
                )

                # Stage 3: Classify confidence
                claim_info.confidence_level = self.classify_confidence(claim_info)

                generated_claims.append(claim_info)
                state.add_claim(claim_info)

        # Stage 4: Verify all claims
        async for event in self.verify_claims(generated_claims):
            yield event

        # Stage 5: Correct citations for non-supported claims
        correction_count = 0
        async for event in self.correct_citations(generated_claims, evidence_pool):
            if event.event_type == "correction_metrics":
                correction_count = event.data.get("total_corrected", 0)
            yield event

        # Stage 7: ARE-style Verification Retrieval for unsupported/partial claims
        stage_7_metrics: VerificationRetrievalMetrics | None = None
        verification_retriever = self._get_verification_retriever()

        if verification_retriever:
            # Filter claims that still need verification retrieval
            # (unsupported/partial that weren't fixed by Stage 5)
            claims_for_stage_7 = [
                c for c in generated_claims
                if c.verification_verdict in self.config.verification_retrieval.trigger_on_verdicts
                and not c.abstained
            ]

            # Log claims that could have been processed but were abstained
            abstained_partials = [
                c for c in generated_claims
                if c.verification_verdict in self.config.verification_retrieval.trigger_on_verdicts
                and c.abstained
            ]
            if abstained_partials:
                logger.warning(
                    "STAGE7_SKIPPING_ABSTAINED_CLAIMS",
                    count=len(abstained_partials),
                    verdicts=[c.verification_verdict for c in abstained_partials],
                    claims=[truncate(c.claim_text, 50) for c in abstained_partials[:3]],
                )

            if claims_for_stage_7:
                logger.info(
                    "CITATION_PIPELINE_STAGE7_START",
                    claims_count=len(claims_for_stage_7),
                    verdicts=[c.verification_verdict for c in claims_for_stage_7],
                )

                revisions: list[ClaimRevision] = []
                async for stage7_event in verification_retriever.retrieve_and_revise(
                    claims=generated_claims,
                    evidence_pool=evidence_pool,
                    report_content=full_content,
                    research_query=state.query,
                    sources=state.sources,
                ):
                    if isinstance(stage7_event, ClaimRevision):
                        revisions.append(stage7_event)
                        yield VerificationEvent(
                            event_type="claim_revised",
                            data={
                                "original_text": truncate(stage7_event.original_claim, 100),
                                "revised_text": truncate(stage7_event.revised_claim, 100),
                                "revision_type": stage7_event.revision_type,
                                "verified_facts": len(stage7_event.verified_facts),
                                "softened_facts": len(stage7_event.softened_facts),
                            },
                        )
                    elif isinstance(stage7_event, Stage7Event):
                        yield VerificationEvent(
                            event_type=f"stage_7_{stage7_event.event_type}",
                            data=stage7_event.data,
                        )

                # Apply all revisions to content (reverse order handled internally)
                if revisions:
                    full_content = verification_retriever.apply_all_revisions(
                        full_content, revisions
                    )

                    # Build key_to_evidence map for existing evidence
                    key_to_evidence: dict[str, RankedEvidence] = {}
                    for idx, ev in enumerate(evidence_pool):
                        key = self._build_citation_key(idx, ev.source_url or "", evidence_pool)
                        key_to_evidence[key] = ev

                    # Register new external evidence from Stage 7 into state
                    new_evidence_items = verification_retriever.get_new_evidence_with_keys()

                    if new_evidence_items:
                        self._register_stage7_evidence(state, new_evidence_items, key_to_evidence)

                    # Always update claims with revised content and new citation keys
                    updated_count = self._update_claims_with_stage7_revisions(
                        revisions=revisions,
                        claims=generated_claims,
                        revised_content=full_content,
                        key_to_evidence=key_to_evidence,
                    )

                    logger.info(
                        "CITATION_PIPELINE_STAGE7_APPLIED",
                        revisions_count=len(revisions),
                        content_len=len(full_content),
                        updated_claims=updated_count,
                        new_evidence=len(new_evidence_items),
                    )

                    # Emit content_revised event so frontend can update displayed content
                    # This is necessary because content was streamed BEFORE Stage 7 revisions
                    yield VerificationEvent(
                        event_type="content_revised",
                        data={
                            "content": full_content,
                            "revision_count": len(revisions),
                        },
                    )

                stage_7_metrics = verification_retriever.metrics

        # Update verification summary after Stage 7
        state.update_verification_summary()

        if state.verification_summary:
            summary_data = {
                "message_id": str(state.session_id),
                "total_claims": state.verification_summary.total_claims,
                "supported": state.verification_summary.supported_count,
                "partial": state.verification_summary.partial_count,
                "unsupported": state.verification_summary.unsupported_count,
                "contradicted": state.verification_summary.contradicted_count,
                "abstained_count": state.verification_summary.abstained_count,
                "citation_corrections": correction_count,
                "warning": state.verification_summary.warning,
            }
            # Add Stage 7 metrics if available
            if stage_7_metrics:
                summary_data["stage_7"] = stage_7_metrics.to_dict()

            yield VerificationEvent(
                event_type="verification_summary",
                data=summary_data,
            )

        # Complete state with (possibly revised) content
        state.complete(full_content)

        logger.info(
            "CITATION_PIPELINE_COMPLETE",
            claims=len(generated_claims),
            verified=sum(1 for c in generated_claims if c.verification_verdict),
            supported=sum(1 for c in generated_claims if c.verification_verdict == "supported"),
            stage_7_revisions=stage_7_metrics.total_claims_processed if stage_7_metrics else 0,
        )

    async def _run_react_synthesis(
        self,
        state: ResearchState,
        evidence_pool: list[RankedEvidence],
        max_tokens: int = 2000,
    ) -> AsyncGenerator[VerificationEvent | str, None]:
        """Run ReAct-based synthesis with grounded generation.

        Uses tools to retrieve evidence before making claims, then runs
        verification on extracted claims.

        Args:
            state: Current research state.
            evidence_pool: Pre-selected evidence from Stage 1.
            max_tokens: Maximum tokens for generation.

        Yields:
            Content chunks (str) and VerificationEvents.
        """
        from src.agent.nodes.react_synthesizer import (
            ReactSynthesisEvent,
            run_react_synthesis,
            run_react_synthesis_sectioned,
        )

        react_config = self.config.react_synthesis

        logger.info(
            "REACT_SYNTHESIS_START",
            evidence_pool_size=len(evidence_pool),
            max_tool_calls=react_config.max_tool_calls,
            sectioned=react_config.use_sectioned_synthesis,
        )

        # Choose synthesis function based on config
        if react_config.use_sectioned_synthesis:
            synthesis_gen = run_react_synthesis_sectioned(
                state=state,
                llm=self.llm,
                evidence_pool=evidence_pool,
                tool_budget_per_section=react_config.tool_budget_per_section,
                retrieval_window_size=react_config.retrieval_window_size,
                enable_post_processing=react_config.enable_post_processing,
            )
        else:
            synthesis_gen = run_react_synthesis(
                state=state,
                llm=self.llm,
                evidence_pool=evidence_pool,
                max_tool_calls=react_config.max_tool_calls,
                retrieval_window_size=react_config.retrieval_window_size,
                enable_post_processing=react_config.enable_post_processing,
            )

        # Stream ReAct events and collect content
        full_content = ""
        tool_call_count = 0
        parsed_blocks: list[ParsedContent] | None = None

        async for event in synthesis_gen:
            if event.event_type == "content":
                chunk = event.data.get("chunk", "")
                full_content += chunk
                yield chunk  # Stream to client

            elif event.event_type == "tool_call":
                tool_call_count += 1
                yield VerificationEvent(
                    event_type="react_tool_call",
                    data={
                        "tool": event.data.get("tool"),
                        "args": event.data.get("args"),
                        "call_number": tool_call_count,
                    },
                )

            elif event.event_type == "section_start":
                yield VerificationEvent(
                    event_type="section_start",
                    data={"section": event.data.get("section")},
                )

            elif event.event_type == "section_complete":
                yield VerificationEvent(
                    event_type="section_complete",
                    data=event.data,
                )

            elif event.event_type == "synthesis_complete":
                # Capture parsed_blocks from Hybrid ReClaim for claim extraction
                parsed_blocks = event.data.get("parsed_blocks")
                logger.info(
                    "REACT_SYNTHESIS_DONE",
                    reason=event.data.get("reason"),
                    tool_calls=event.data.get("tool_calls", 0),
                    content_len=len(full_content),
                    parsed_blocks_count=len(parsed_blocks) if parsed_blocks else 0,
                )

            elif event.event_type == "grounding_warning":
                yield VerificationEvent(
                    event_type="grounding_warning",
                    data=event.data,
                )

            elif event.event_type == "error":
                logger.error(
                    "REACT_SYNTHESIS_ERROR",
                    error=event.data.get("error"),
                )

        # Grounding Validation: Run AFTER synthesis, BEFORE claim extraction
        # Validates that <analysis> blocks are grounded and <free> blocks are structural
        if parsed_blocks and self.config.grounding_validation.enabled:
            full_content, validation_issues = await self.validate_content_grounding(
                parsed_blocks, full_content
            )

            if validation_issues:
                yield VerificationEvent(
                    event_type="grounding_validation",
                    data={
                        "total_issues": len(validation_issues),
                        "ungrounded_analysis": sum(1 for i in validation_issues if i.issue_type == "ungrounded_analysis"),
                        "baseless_analysis": sum(1 for i in validation_issues if i.issue_type == "baseless_analysis"),
                        "hidden_claims_in_free": sum(1 for i in validation_issues if i.issue_type == "hidden_claims_in_free"),
                    },
                )

        # Extract claims from generated content
        # Uses parsed blocks from Hybrid ReClaim when available, falls back to regex
        generated_claims = self._extract_claims_from_react_content(
            full_content, evidence_pool, parsed_blocks
        )

        logger.info(
            "REACT_CLAIMS_EXTRACTED",
            claims_count=len(generated_claims),
            content_len=len(full_content),
        )

        # Add claims to state
        for claim_info in generated_claims:
            state.add_claim(claim_info)

            # Stage 3: Classify confidence
            claim_info.confidence_level = self.classify_confidence(claim_info)

        # Stage 4: Verify all claims
        async for event in self.verify_claims(generated_claims):
            yield event

        # Stage 5: Correct citations if needed
        correction_count = 0
        async for event in self.correct_citations(generated_claims, evidence_pool):
            if event.event_type == "correction_metrics":
                correction_count = event.data.get("total_corrected", 0)
            yield event

        # Stage 7: ARE-style Verification Retrieval for unsupported/partial claims
        stage_7_metrics: VerificationRetrievalMetrics | None = None
        verification_retriever = self._get_verification_retriever()

        if verification_retriever:
            claims_for_stage_7 = [
                c for c in generated_claims
                if c.verification_verdict in self.config.verification_retrieval.trigger_on_verdicts
                and not c.abstained
            ]

            # Log claims that could have been processed but were abstained
            abstained_partials = [
                c for c in generated_claims
                if c.verification_verdict in self.config.verification_retrieval.trigger_on_verdicts
                and c.abstained
            ]
            if abstained_partials:
                logger.warning(
                    "REACT_STAGE7_SKIPPING_ABSTAINED_CLAIMS",
                    count=len(abstained_partials),
                    verdicts=[c.verification_verdict for c in abstained_partials],
                    claims=[truncate(c.claim_text, 50) for c in abstained_partials[:3]],
                )

            if claims_for_stage_7:
                logger.info(
                    "REACT_STAGE7_START",
                    claims_count=len(claims_for_stage_7),
                )

                revisions: list[ClaimRevision] = []
                async for stage7_event in verification_retriever.retrieve_and_revise(
                    claims=generated_claims,
                    evidence_pool=evidence_pool,
                    report_content=full_content,
                    research_query=state.query,
                    sources=state.sources,
                ):
                    if isinstance(stage7_event, ClaimRevision):
                        revisions.append(stage7_event)
                        yield VerificationEvent(
                            event_type="claim_revised",
                            data={
                                "original_text": truncate(stage7_event.original_claim, 100),
                                "revised_text": truncate(stage7_event.revised_claim, 100),
                                "revision_type": stage7_event.revision_type,
                            },
                        )
                    elif isinstance(stage7_event, Stage7Event):
                        yield VerificationEvent(
                            event_type=f"stage_7_{stage7_event.event_type}",
                            data=stage7_event.data,
                        )

                if revisions:
                    full_content = verification_retriever.apply_all_revisions(
                        full_content, revisions
                    )

                    # Build key_to_evidence map for existing evidence
                    key_to_evidence: dict[str, RankedEvidence] = {}
                    for idx, ev in enumerate(evidence_pool):
                        key = self._build_citation_key(idx, ev.source_url or "", evidence_pool)
                        key_to_evidence[key] = ev

                    # Register new external evidence from Stage 7 into state
                    # This ensures proper database persistence and prevents orphaned markers
                    new_evidence_items = verification_retriever.get_new_evidence_with_keys()

                    if new_evidence_items:
                        # Register new evidence into state and update key map
                        self._register_stage7_evidence(state, new_evidence_items, key_to_evidence)

                    # Always update claims with revised content and new citation keys
                    updated_count = self._update_claims_with_stage7_revisions(
                        revisions=revisions,
                        claims=generated_claims,
                        revised_content=full_content,
                        key_to_evidence=key_to_evidence,
                    )

                    logger.info(
                        "STAGE7_CLAIMS_UPDATED",
                        updated_count=updated_count,
                        new_evidence_count=len(new_evidence_items),
                        revision_count=len(revisions),
                    )

                    # Emit content_revised event so frontend can update displayed content
                    # This is necessary because content was streamed BEFORE Stage 7 revisions
                    yield VerificationEvent(
                        event_type="content_revised",
                        data={
                            "content": full_content,
                            "revision_count": len(revisions),
                        },
                    )

                stage_7_metrics = verification_retriever.metrics

        # Stage 8: Post-verification claim modification (remove contradicted, soften unsupported)
        stage_8_removed = 0
        stage_8_softened = 0

        if self.config.enable_claim_removal or self.config.enable_claim_softening:
            full_content, stage_8_removed, stage_8_softened = await self._process_unverified_claims(
                full_content, generated_claims
            )

            if stage_8_removed > 0 or stage_8_softened > 0:
                yield VerificationEvent(
                    event_type="claims_processed",
                    data={
                        "removed_count": stage_8_removed,
                        "softened_count": stage_8_softened,
                    },
                )

                # Emit content_revised event so frontend can update
                yield VerificationEvent(
                    event_type="content_revised",
                    data={
                        "content": full_content,
                        "stage": "claim_modification",
                        "removed": stage_8_removed,
                        "softened": stage_8_softened,
                    },
                )

        # Update verification summary after Stage 7 and 8
        state.update_verification_summary()

        if state.verification_summary:
            summary_data = {
                "message_id": str(state.session_id),
                "synthesis_mode": "react",
                "tool_calls": tool_call_count,
                "total_claims": state.verification_summary.total_claims,
                "supported": state.verification_summary.supported_count,
                "partial": state.verification_summary.partial_count,
                "unsupported": state.verification_summary.unsupported_count,
                "contradicted": state.verification_summary.contradicted_count,
                "abstained_count": state.verification_summary.abstained_count,
                "citation_corrections": correction_count,
                "warning": state.verification_summary.warning,
            }
            if stage_7_metrics:
                summary_data["stage_7"] = stage_7_metrics.to_dict()

            yield VerificationEvent(
                event_type="verification_summary",
                data=summary_data,
            )

        # Complete state with (possibly revised) content
        state.complete(full_content)

        logger.info(
            "REACT_PIPELINE_COMPLETE",
            claims=len(generated_claims),
            verified=sum(1 for c in generated_claims if c.verification_verdict),
            supported=sum(1 for c in generated_claims if c.verification_verdict == "supported"),
            tool_calls=tool_call_count,
            stage_7_revisions=stage_7_metrics.total_claims_processed if stage_7_metrics else 0,
        )

        # Audit grey references: citation markers in report with abstained claims
        import re

        content_markers = set(
            re.findall(r"\[([A-Za-z][A-Za-z0-9-]*(?:-\d+)?)\]", full_content or "")
        )
        abstained_keys = {
            c.citation_key for c in generated_claims if c.abstained and c.citation_key
        }
        true_grey = content_markers & abstained_keys

        if true_grey:
            logger.warning(
                "GREY_REFERENCE_AUDIT",
                markers_in_report=len(content_markers),
                abstained_claims=len(abstained_keys),
                true_grey_count=len(true_grey),
                true_grey_keys=list(true_grey)[:5],
            )

    def _extract_claims_from_react_content(
        self,
        content: str,
        evidence_pool: list[RankedEvidence],
        parsed_blocks: list[ParsedContent] | None = None,
    ) -> list[ClaimInfo]:
        """Extract claims from ReAct-generated content.

        Uses parsed blocks from Hybrid ReClaim when available, otherwise falls
        back to regex-based extraction finding sentences with [Key] markers.

        Args:
            content: Generated content with [Key] citation markers.
            evidence_pool: Evidence pool for linking citations.
            parsed_blocks: Optional parsed blocks from Hybrid ReClaim (XML tags).

        Returns:
            List of extracted ClaimInfo objects.
        """
        import re

        claims: list[ClaimInfo] = []

        # Build citation key to evidence map
        # Keys must match what EvidenceRegistry.build_citation_key() produces
        key_to_evidence: dict[str, RankedEvidence] = {}
        for idx, ev in enumerate(evidence_pool):
            key = self._build_citation_key(idx, ev.source_url or "", evidence_pool)
            key_to_evidence[key] = ev

        logger.debug(
            "REACT_CLAIM_EXTRACTION_KEY_MAP",
            key_count=len(key_to_evidence),
            keys=list(key_to_evidence.keys()),
        )

        # Use parsed blocks from Hybrid ReClaim if available
        if parsed_blocks:
            # Track search position to handle duplicate text correctly
            search_start = 0

            for block in parsed_blocks:
                if block.tag_type == "cite" and block.citation_key:
                    # Grounded claim with citation
                    evidence = key_to_evidence.get(block.citation_key)
                    claim_text = f"{block.text} [{block.citation_key}]"

                    # Find actual position in assembled content
                    position = content.find(claim_text, search_start)
                    if position == -1:
                        # Fallback: try finding just the text without marker
                        position = content.find(block.text, search_start)

                    position_start = position if position >= 0 else 0
                    position_end = position_start + len(claim_text)
                    search_start = position_end if position >= 0 else search_start

                    claim_info = ClaimInfo(
                        claim_text=claim_text,
                        claim_type="numeric" if evidence and evidence.has_numeric_content else "general",
                        position_start=position_start,
                        position_end=position_end,
                        evidence=EvidenceInfo(
                            source_url=evidence.source_url or "",
                            quote_text=evidence.quote_text,
                            start_offset=evidence.start_offset,
                            end_offset=evidence.end_offset,
                            section_heading=evidence.section_heading,
                            relevance_score=evidence.relevance_score,
                            has_numeric_content=evidence.has_numeric_content,
                        ) if evidence else None,
                        citation_key=block.citation_key,
                    )
                    claims.append(claim_info)
                elif block.tag_type == "unverified":
                    # Uncertain claim - will be verified in Stage 4-5
                    # Find actual position in assembled content
                    position = content.find(block.text, search_start)
                    position_start = position if position >= 0 else 0
                    position_end = position_start + len(block.text)
                    search_start = position_end if position >= 0 else search_start

                    claim_info = ClaimInfo(
                        claim_text=block.text,
                        claim_type="general",
                        position_start=position_start,
                        position_end=position_end,
                        evidence=None,  # No evidence - needs verification
                    )
                    claims.append(claim_info)
                elif block.tag_type == "free":
                    # Check if we should extract claims from <free> blocks
                    if not self.config.enable_free_block_extraction:
                        continue

                    # Skip if below minimum length
                    text = block.text.strip()
                    if len(text) < self.config.free_block_min_length:
                        continue

                    # Skip structural content (headers, transitions)
                    if self._is_structural_content(text):
                        continue

                    # Check max limit
                    free_block_count = len([
                        c for c in claims
                        if not c.citation_key and not c.evidence
                    ])
                    if (
                        self.config.max_free_block_claims > 0
                        and free_block_count >= self.config.max_free_block_claims
                    ):
                        continue

                    # This looks like factual content - create uncited claim
                    position = content.find(text, search_start)
                    position_start = position if position >= 0 else 0
                    position_end = position_start + len(text)
                    search_start = position_end if position >= 0 else search_start

                    claim_info = ClaimInfo(
                        claim_text=text,
                        claim_type="general",
                        position_start=position_start,
                        position_end=position_end,
                        evidence=None,  # No evidence - needs Stage 5/7 verification
                        from_free_block=True,  # Mark as extracted from <free> block
                    )
                    claims.append(claim_info)

            # Log extraction results including free block claims
            free_block_claims = [c for c in claims if getattr(c, 'from_free_block', False)]
            logger.info(
                "REACT_CLAIM_EXTRACTION_FROM_BLOCKS",
                parsed_blocks_count=len(parsed_blocks),
                cite_blocks=len([b for b in parsed_blocks if b.tag_type == "cite"]),
                unverified_blocks=len([b for b in parsed_blocks if b.tag_type == "unverified"]),
                free_block_claims=len(free_block_claims),
                claims_count=len(claims),
            )
            return claims

        # Fallback: regex-based extraction for legacy content
        # Find sentences with citation markers
        # Pattern: sentence ending with [Key] or [Key-N]
        citation_pattern = r'\[([A-Za-z][A-Za-z0-9-]*(?:-\d+)?)\]'

        # Split into sentences (simple approach)
        sentences = re.split(r'(?<=[.!?])\s+', content)

        position = 0
        for sentence in sentences:
            # Find all citation markers in this sentence
            markers = re.findall(citation_pattern, sentence)

            if markers:
                # This sentence has citations - it's a claim
                claim_text = sentence.strip()
                position_start = content.find(claim_text, position)
                position_end = position_start + len(claim_text) if position_start != -1 else position

                # Get evidence for primary citation
                primary_key = markers[0]
                evidence = key_to_evidence.get(primary_key)

                claim_info = ClaimInfo(
                    claim_text=claim_text,
                    claim_type="numeric" if evidence and evidence.has_numeric_content else "general",
                    position_start=position_start if position_start != -1 else position,
                    position_end=position_end,
                    evidence=EvidenceInfo(
                        source_url=evidence.source_url or "",
                        quote_text=evidence.quote_text,
                        start_offset=evidence.start_offset,
                        end_offset=evidence.end_offset,
                        section_heading=evidence.section_heading,
                        relevance_score=evidence.relevance_score,
                        has_numeric_content=evidence.has_numeric_content,
                    ) if evidence else None,
                    citation_key=primary_key,
                    citation_keys=markers if len(markers) > 1 else None,
                )
                claims.append(claim_info)

            position = content.find(sentence, position)
            if position != -1:
                position += len(sentence)

        # Log extraction summary with matched/unmatched markers
        all_markers_in_content = set(re.findall(citation_pattern, content))
        matched_keys = {c.citation_key for c in claims if c.citation_key}
        unmatched = all_markers_in_content - matched_keys

        if unmatched:
            logger.warning(
                "REACT_CLAIM_EXTRACTION_UNMATCHED_MARKERS",
                unmatched_markers=list(unmatched),
                available_keys=list(key_to_evidence.keys()),
            )

        logger.info(
            "REACT_CLAIM_EXTRACTION_COMPLETE",
            claims_count=len(claims),
            content_len=len(content),
            key_map_size=len(key_to_evidence),
            matched_count=len(matched_keys),
            unmatched_count=len(unmatched),
        )

        return claims

    def _build_citation_key(
        self, index: int, url: str, evidence_pool: list[RankedEvidence]
    ) -> str:
        """Build citation key matching EvidenceRegistry.build_citation_key() logic.

        Must produce IDENTICAL keys to what LLM sees during synthesis.
        Uses duplicate-detection: suffix is based on how many same-domain URLs
        appear BEFORE this index, not on the index itself.

        Args:
            index: Evidence index in the pool.
            url: Source URL for this evidence.
            evidence_pool: Full evidence pool for duplicate detection.

        Returns:
            Citation key like "Arxiv", "Github", "Arxiv-2", etc.
        """
        from urllib.parse import urlparse

        # Extract domain key for this URL
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.replace("www.", "")
            key = domain.split(".")[0].capitalize()
        except Exception:
            key = "Source"

        # Count duplicates BEFORE this index (same as EvidenceRegistry)
        count = 0
        for idx, ev in enumerate(evidence_pool[:index]):
            try:
                other_parsed = urlparse(ev.source_url or "")
                other_domain = other_parsed.netloc.replace("www.", "")
                other_key = other_domain.split(".")[0].capitalize()
                if other_key == key:
                    count += 1
            except Exception:
                pass

        return f"{key}-{count + 1}" if count > 0 else key

    def _is_structural_content(self, text: str) -> bool:
        """Check if text is structural (headers, transitions) vs factual content.

        Used to filter out non-factual content from <free> blocks that shouldn't
        be treated as claims for verification.

        Args:
            text: Text to check.

        Returns:
            True if text is structural content that should NOT be a claim.
        """
        import re

        stripped = text.strip()
        if not stripped:
            return True

        # Markdown headers (# through ######)
        if stripped.startswith("#"):
            return True

        # Bullet lists (-, *, +)
        if stripped.startswith(("-", "*", "+")):
            return True

        # Numbered lists (1., 2., etc.)
        if len(stripped) > 2 and stripped[0].isdigit() and stripped[1] == ".":
            return True

        # Horizontal rules
        if stripped in ("---", "***", "___"):
            return True

        # Short transitional phrases (less than ~25 chars and looks like transition)
        transition_patterns = [
            r"^(In summary|In conclusion|Additionally|Furthermore|Moreover|"
            r"However|Therefore|Thus|As a result|Moving (on|forward)|"
            r"Next|Finally|To conclude|In closing|Overall|Notably|"
            r"For example|For instance|Specifically|In particular)[,:]?\s*$",
            r"^(The following|Below|Above|See also)[,:]?\s*$",
        ]
        for pattern in transition_patterns:
            if re.match(pattern, stripped, re.IGNORECASE):
                return True

        # Very short text is likely structural (headers, labels)
        if len(stripped) < 15:
            return True

        # Questions are often structural ("What are the key findings?")
        if stripped.endswith("?") and len(stripped) < 50:
            return True

        return False

    def _find_claim_by_position(
        self,
        claims: list[ClaimInfo],
        start: int,
        end: int,
    ) -> ClaimInfo | None:
        """Find claim by original position (before Stage 7 revisions).

        Uses position overlap matching to handle slight position variations.

        Args:
            claims: List of claims to search.
            start: Original position start.
            end: Original position end.

        Returns:
            Matching ClaimInfo or None if not found.
        """
        for claim in claims:
            # Exact match
            if claim.position_start == start and claim.position_end == end:
                return claim
            # Overlap check for fuzzy matching (>80% overlap)
            overlap = min(claim.position_end, end) - max(claim.position_start, start)
            span = max(claim.position_end - claim.position_start, end - start)
            if span > 0 and overlap / span > 0.8:
                return claim
        return None

    def _register_stage7_evidence(
        self,
        state: ResearchState,
        new_evidence_items: list[NewExternalEvidence],
        key_to_evidence: dict[str, RankedEvidence],
    ) -> None:
        """Register new external evidence from Stage 7 into state.

        Adds new sources and evidence to state for proper database persistence.
        Updates key_to_evidence map with pre-assigned citation keys.

        Args:
            state: Research state to update.
            new_evidence_items: New evidence with pre-assigned citation keys.
            key_to_evidence: Citation key to evidence map to update.
        """
        import re

        for item in new_evidence_items:
            # Add source to state (required for persistence FK constraint)
            state.add_source(SourceInfo(
                url=item.source_url,
                title=item.evidence.source_title or "",
                snippet=None,
                content=item.evidence.quote_text,  # Store quote as content
                relevance_score=item.evidence.relevance_score,
            ))

            # Add evidence to state evidence pool
            state.add_evidence(EvidenceInfo(
                source_url=item.source_url,
                quote_text=item.evidence.quote_text,
                start_offset=item.evidence.start_offset,
                end_offset=item.evidence.end_offset,
                section_heading=item.evidence.section_heading,
                relevance_score=item.evidence.relevance_score,
                has_numeric_content=item.evidence.has_numeric_content or False,
            ))

            # Add to key map with PRE-ASSIGNED key
            key_to_evidence[item.citation_key] = item.evidence

            logger.info(
                "STAGE7_EVIDENCE_REGISTERED",
                citation_key=item.citation_key,
                url=truncate(item.source_url, 50),
                fact=truncate(item.fact_text, 50),
            )

    def _update_claims_with_stage7_revisions(
        self,
        revisions: list[ClaimRevision],
        claims: list[ClaimInfo],
        revised_content: str,
        key_to_evidence: dict[str, RankedEvidence],
    ) -> int:
        """Update original ClaimInfo objects with Stage 7 revisions in place.

        This ensures that revised claims have their positions and citation keys
        updated to match the revised content, preventing orphaned markers.

        Args:
            revisions: List of claim revisions from Stage 7.
            claims: Original claims list to update.
            revised_content: Content after all revisions applied.
            key_to_evidence: Citation key to evidence map.

        Returns:
            Number of claims successfully updated.
        """
        import re

        updated_count = 0
        citation_pattern = re.compile(r'\[([A-Za-z][A-Za-z0-9-]*(?:-\d+)?)\]')

        for revision in revisions:
            # Find the original claim by position match
            original_claim = self._find_claim_by_position(
                claims,
                revision.original_position_start,
                revision.original_position_end,
            )

            if not original_claim:
                logger.warning(
                    "STAGE7_CLAIM_NOT_FOUND",
                    original_start=revision.original_position_start,
                    original_end=revision.original_position_end,
                    original_text=truncate(revision.original_claim, 50),
                )
                continue

            # Find new position in revised content
            new_position = revised_content.find(revision.revised_claim)
            if new_position == -1:
                # Try fuzzy search - look for partial match
                logger.warning(
                    "STAGE7_REVISED_CLAIM_NOT_FOUND",
                    revised_text=truncate(revision.revised_claim, 50),
                )
                # Keep original positions as fallback
                new_position = original_claim.position_start

            # Extract citation keys from revised claim
            markers = citation_pattern.findall(revision.revised_claim)

            # UPDATE the original claim IN PLACE
            original_claim.claim_text = revision.revised_claim
            original_claim.position_start = new_position
            original_claim.position_end = new_position + len(revision.revised_claim)

            # Update citation keys - only if evidence exists to prevent grey references
            if markers:
                primary_key = markers[0]

                # VALIDATE: Only update citation_key if evidence exists
                if primary_key in key_to_evidence:
                    original_claim.citation_key = primary_key
                    original_claim.citation_keys = markers if len(markers) > 1 else None
                    new_evidence = key_to_evidence[primary_key]
                    original_claim.evidence = EvidenceInfo(
                        source_url=new_evidence.source_url or "",
                        quote_text=new_evidence.quote_text,
                        start_offset=new_evidence.start_offset,
                        end_offset=new_evidence.end_offset,
                        section_heading=new_evidence.section_heading,
                        relevance_score=new_evidence.relevance_score,
                        has_numeric_content=new_evidence.has_numeric_content or False,
                    )
                else:
                    # Log warning and keep original citation_key to prevent grey reference
                    logger.warning(
                        "STAGE7_KEY_NOT_IN_EVIDENCE",
                        key=primary_key,
                        available_keys=list(key_to_evidence.keys())[:10],
                        original_key=original_claim.citation_key,
                    )

            # Update verdict based on revision type
            if revision.revision_type == "fully_verified":
                original_claim.verification_verdict = "supported"
            elif revision.revision_type == "partially_softened":
                original_claim.verification_verdict = "partial"
            # fully_softened keeps existing verdict or stays unsupported

            updated_count += 1
            logger.debug(
                "CLAIM_UPDATED_BY_STAGE7",
                original_len=revision.original_position_end - revision.original_position_start,
                revised_len=len(revision.revised_claim),
                new_citations=markers,
                verdict=original_claim.verification_verdict,
            )

        return updated_count

    # =========================================================================
    # Stage 8: Post-verification claim modification
    # =========================================================================

    async def _process_unverified_claims(
        self,
        content: str,
        claims: list[ClaimInfo],
    ) -> tuple[str, int, int]:
        """Process contradicted/unsupported claims in a single pass.

        Removes contradicted claims and softens unsupported claims with hedging.
        All modifications are done in position-descending order to maintain
        correct positions for subsequent modifications.

        Args:
            content: Report content to modify.
            claims: List of claims to process.

        Returns:
            Tuple of (modified_content, removed_count, softened_count).
        """
        import re

        # Collect all modifications
        modifications: list[tuple[str, ClaimInfo]] = []

        if self.config.enable_claim_removal:
            for claim in claims:
                if claim.verification_verdict == "contradicted" and not claim.abstained:
                    modifications.append(("remove", claim))

        if self.config.enable_claim_softening:
            for claim in claims:
                if claim.verification_verdict == "unsupported" and not claim.abstained:
                    modifications.append(("soften", claim))

        if not modifications:
            return content, 0, 0

        # Sort by position descending - process from end to start
        modifications.sort(key=lambda x: x[1].position_start, reverse=True)

        # Merge overlapping claims
        modifications = self._merge_overlapping_modifications(modifications)

        removed_count = 0
        softened_count = 0

        for action, claim in modifications:
            # Skip if claim is in special context (table, code block)
            context = self._is_in_special_context(content, claim.position_start)
            if context == "code":
                logger.debug(
                    "SKIP_CLAIM_IN_CODE_BLOCK",
                    claim_text=truncate(claim.claim_text, 50),
                )
                continue

            if action == "remove":
                # Remove contradicted claim
                content = self._remove_claim(content, claim, context)
                removed_count += 1
                logger.info(
                    "CLAIM_REMOVED",
                    claim_text=truncate(claim.claim_text, 50),
                    verdict=claim.verification_verdict,
                    position=(claim.position_start, claim.position_end),
                )
            else:  # soften
                # Soften unsupported claim with hedging
                content = self._soften_claim(content, claim, context)
                softened_count += 1
                logger.info(
                    "CLAIM_SOFTENED",
                    claim_text=truncate(claim.claim_text, 50),
                )

        # Clean up empty sections after removals
        if removed_count > 0:
            content = self._clean_empty_sections(content)

        # Recalculate all claim positions after modifications
        self._recalculate_claim_positions(content, claims)

        logger.info(
            "STAGE8_COMPLETE",
            removed=removed_count,
            softened=softened_count,
            modifications_total=len(modifications),
        )

        return content, removed_count, softened_count

    def _merge_overlapping_modifications(
        self,
        modifications: list[tuple[str, ClaimInfo]],
    ) -> list[tuple[str, ClaimInfo]]:
        """Merge overlapping modifications to avoid position conflicts.

        Args:
            modifications: List of (action, claim) tuples, sorted by position desc.

        Returns:
            Merged list with overlapping claims combined.
        """
        if len(modifications) <= 1:
            return modifications

        merged: list[tuple[str, ClaimInfo]] = []
        for action, claim in modifications:
            if not merged:
                merged.append((action, claim))
                continue

            prev_action, prev_claim = merged[-1]
            # Check for overlap (note: sorted descending, so current is BEFORE prev)
            if claim.position_end > prev_claim.position_start:
                # Overlap detected - keep the more severe action
                if action == "remove" or prev_action == "remove":
                    # If either is remove, remove the combined span
                    merged_action = "remove"
                else:
                    merged_action = "soften"

                # Merge positions (take widest span)
                prev_claim.position_start = min(claim.position_start, prev_claim.position_start)
                prev_claim.position_end = max(claim.position_end, prev_claim.position_end)
                merged[-1] = (merged_action, prev_claim)
            else:
                merged.append((action, claim))

        return merged

    def _remove_claim(self, content: str, claim: ClaimInfo, context: str | None) -> str:
        """Remove a contradicted claim from content.

        Args:
            content: Report content.
            claim: Claim to remove.
            context: Special context ("table", "list", None).

        Returns:
            Content with claim removed.
        """
        # For tables/lists, mark as removed but keep structure
        if context == "table":
            # In table: replace with [removed]
            before = content[:claim.position_start]
            after = content[claim.position_end:]
            return before + "[removed for factual inaccuracy]" + after

        if context == "list":
            # In list: remove entire list item (line)
            start = content.rfind("\n", 0, claim.position_start) + 1
            end = content.find("\n", claim.position_end)
            if end == -1:
                end = len(content)
            return content[:start] + content[end:]

        # Normal paragraph: remove with whitespace cleanup
        before = content[:claim.position_start].rstrip()
        after = content[claim.position_end:].lstrip()

        # Add space between remaining content
        if before and after and not before.endswith("\n") and not after.startswith("\n"):
            return before + " " + after
        return before + after

    def _soften_claim(self, content: str, claim: ClaimInfo, context: str | None) -> str:
        """Soften an unsupported claim with hedging language.

        Args:
            content: Report content.
            claim: Claim to soften.
            context: Special context ("table", "list", None).

        Returns:
            Content with softened claim.
        """
        import re

        claim_text = claim.claim_text

        # Remove citation markers if present
        clean_text = re.sub(r'\s*\[[A-Za-z][A-Za-z0-9-]*(?:-\d+)?\]\s*', '', claim_text).strip()

        # Check if already hedged
        if not self._needs_hedging(clean_text):
            logger.debug(
                "SKIP_ALREADY_HEDGED",
                claim_text=truncate(clean_text, 50),
            )
            return content

        # For tables, just add [unverified] marker
        if context == "table":
            before = content[:claim.position_start]
            after = content[claim.position_end:]
            return before + f"{clean_text} [unverified]" + after

        # Choose hedging phrase based on content
        hedging_phrases = [
            "It has been suggested that",
            "Some sources indicate that",
            "According to available information,",
            "Reportedly,",
        ]

        # Use a consistent hedge based on claim text hash for reproducibility
        hedge_idx = hash(clean_text) % len(hedging_phrases)
        hedge = hedging_phrases[hedge_idx]

        # Lowercase first letter after hedge
        if clean_text and clean_text[0].isupper():
            clean_text = clean_text[0].lower() + clean_text[1:]

        softened = f"{hedge} {clean_text}"

        before = content[:claim.position_start]
        after = content[claim.position_end:]

        return before + softened + after

    def _needs_hedging(self, text: str) -> bool:
        """Check if text already has hedging language.

        Args:
            text: Text to check.

        Returns:
            True if text needs hedging (doesn't already have it).
        """
        existing_hedges = [
            "it appears", "it seems", "may have", "might be",
            "reportedly", "allegedly", "according to", "suggests that",
            "it has been suggested", "some sources indicate", "available information",
            "unverified", "uncertain", "possibly", "potentially",
        ]
        lower = text.lower()
        return not any(hedge in lower for hedge in existing_hedges)

    def _is_in_special_context(self, content: str, position: int) -> str | None:
        """Check if position is inside special markdown context.

        Args:
            content: Full content.
            position: Position to check.

        Returns:
            "table", "list", "code", or None.
        """
        # Find surrounding context
        start = max(0, position - 200)
        before = content[start:position]
        lines_before = before.split('\n')

        if not lines_before:
            return None

        last_line = lines_before[-1]

        # In table? Look for | characters on same line
        if '|' in last_line:
            return "table"

        # In list? Look for leading - or * or numbered
        stripped = last_line.strip()
        if stripped.startswith(('-', '*', '+')):
            return "list"
        if stripped and stripped[0].isdigit() and '.' in stripped[:3]:
            return "list"

        # In code block? Count backticks
        if before.count('```') % 2 == 1:
            return "code"

        return None

    def _clean_empty_sections(self, content: str) -> str:
        """Remove empty section headers after claim removal.

        Args:
            content: Content that may have empty sections.

        Returns:
            Content with empty sections removed.
        """
        import re

        # Pattern: header followed by only whitespace until next header or EOF
        # Must be careful not to remove headers followed by content
        lines = content.split('\n')
        cleaned_lines: list[str] = []
        i = 0

        while i < len(lines):
            line = lines[i]

            # Check if this is a header
            if line.strip().startswith('#'):
                # Look ahead to see if there's content before next header
                j = i + 1
                has_content = False
                while j < len(lines):
                    next_line = lines[j].strip()
                    if next_line.startswith('#'):
                        break
                    if next_line and not next_line.isspace():
                        has_content = True
                        break
                    j += 1

                if has_content:
                    cleaned_lines.append(line)
                else:
                    # Skip empty section header
                    logger.debug(
                        "EMPTY_SECTION_REMOVED",
                        header=line.strip(),
                    )
            else:
                cleaned_lines.append(line)

            i += 1

        return '\n'.join(cleaned_lines)

    def _recalculate_claim_positions(
        self,
        content: str,
        claims: list[ClaimInfo],
    ) -> None:
        """Update claim positions to match modified content.

        Args:
            content: Modified content.
            claims: Claims to update.
        """
        for claim in claims:
            # Find the claim text in the new content
            new_pos = content.find(claim.claim_text)
            if new_pos >= 0:
                claim.position_start = new_pos
                claim.position_end = new_pos + len(claim.claim_text)
            else:
                # Claim was removed or significantly modified - mark as abstained
                if claim.verification_verdict == "contradicted":
                    claim.abstained = True

    # =========================================================================
    # Grounding Validation (Scientific Citation Style)
    # =========================================================================

    async def validate_content_grounding(
        self,
        parsed_blocks: list[ParsedContent],
        content: str,
    ) -> tuple[str, list[ValidationIssue]]:
        """Validate that <analysis> blocks are grounded and <free> blocks are structural.

        Based on SOTA approaches (FACTS Grounding, FActScore, SAFE):
        - Analysis blocks should be derivable from preceding citations
        - Free blocks should contain only structural content

        Args:
            parsed_blocks: Parsed content blocks from ReAct synthesis.
            content: Assembled report content.

        Returns:
            Tuple of (possibly modified content, list of validation issues).
        """
        config = self.config.grounding_validation

        if not config.enabled:
            logger.debug("GROUNDING_VALIDATION_DISABLED")
            return content, []

        issues: list[ValidationIssue] = []
        preceding_citations: list[str] = []
        validated_count = 0

        logger.info(
            "GROUNDING_VALIDATION_START",
            total_blocks=len(parsed_blocks),
            analysis_blocks=len([b for b in parsed_blocks if b.tag_type == "analysis"]),
            free_blocks=len([b for b in parsed_blocks if b.tag_type == "free"]),
        )

        for i, block in enumerate(parsed_blocks):
            # Stop if we've validated enough blocks
            if validated_count >= config.max_blocks_to_validate:
                break

            if block.tag_type == "cite":
                # Accumulate citations for grounding context
                preceding_citations.append(block.text)
                # Cap the size to avoid huge prompts
                if len(preceding_citations) > config.max_preceding_citations:
                    preceding_citations = preceding_citations[-config.max_preceding_citations:]

            elif block.tag_type == "analysis":
                # Skip short analysis blocks (topic sentences)
                if len(block.text) < config.min_analysis_length:
                    continue

                # Check if this is a topic sentence after a header
                if config.allow_topic_sentences:
                    if i > 0 and parsed_blocks[i - 1].tag_type == "free":
                        prev_content = parsed_blocks[i - 1].text.strip()
                        if prev_content.startswith("#"):
                            # Allow topic sentence after header
                            continue

                validated_count += 1

                # Verify this analysis is grounded in preceding citations
                if not preceding_citations:
                    # No citations before this analysis - check if it's common knowledge
                    is_common = await self._is_common_knowledge(block.text)
                    if not is_common:
                        issues.append(ValidationIssue(
                            block_index=i,
                            issue_type="ungrounded_analysis",
                            message="Analysis has no preceding citations to support it",
                            content=block.text[:100],
                        ))
                else:
                    # Use LLM judge to check grounding (FACTS Grounding style)
                    is_grounded, reason = await self._check_analysis_grounding(
                        analysis=block.text,
                        preceding_citations=preceding_citations,
                    )
                    if not is_grounded:
                        issues.append(ValidationIssue(
                            block_index=i,
                            issue_type="baseless_analysis",
                            message="Analysis makes claims not supported by cited evidence",
                            content=block.text[:100],
                            reason=reason,
                        ))

            elif block.tag_type == "free":
                # Skip structural content (headers, short transitions)
                text = block.text.strip()
                if len(text) < 20 or text.startswith("#"):
                    continue

                # Check for hidden claims in free blocks
                has_claims, claim_text = await self._check_for_hidden_claims(text)
                if has_claims:
                    issues.append(ValidationIssue(
                        block_index=i,
                        issue_type="hidden_claims_in_free",
                        message="Free block contains factual claims that should be cited",
                        content=claim_text,
                    ))
                    validated_count += 1

        # Log validation results
        ungrounded = sum(1 for i in issues if i.issue_type == "ungrounded_analysis")
        baseless = sum(1 for i in issues if i.issue_type == "baseless_analysis")
        hidden = sum(1 for i in issues if i.issue_type == "hidden_claims_in_free")

        logger.info(
            "GROUNDING_VALIDATION_COMPLETE",
            total_issues=len(issues),
            ungrounded_analysis=ungrounded,
            baseless_analysis=baseless,
            hidden_claims_in_free=hidden,
            validated_blocks=validated_count,
        )

        # Apply remediation if there are issues
        if issues:
            content = await self._remediate_validation_issues(
                content, parsed_blocks, issues
            )

        return content, issues

    async def _check_analysis_grounding(
        self,
        analysis: str,
        preceding_citations: list[str],
    ) -> tuple[bool, str]:
        """Use LLM judge to verify analysis is derived from cited facts.

        Based on SOTA research (FACTS Grounding, SAFE):
        - Analysis CAN draw conclusions, make predictions, synthesize
        - Analysis CANNOT introduce NEW factual claims not derivable from citations

        Args:
            analysis: The analysis text to verify.
            preceding_citations: List of preceding citation texts.

        Returns:
            Tuple of (is_grounded, reason).
        """
        from src.services.llm.types import ModelTier

        citations_context = "\n".join(f"- {c}" for c in preceding_citations)

        # Two-question approach (inspired by FACTS Grounding two-phase evaluation)
        prompt = f"""You are a grounding verification judge. Evaluate if an analysis statement is properly grounded.

## CITED FACTS (Premise):
{citations_context}

## ANALYSIS TO VERIFY (Hypothesis):
{analysis}

## EVALUATION CRITERIA:
Analysis IS GROUNDED if it:
- Draws conclusions from the cited facts
- Makes predictions based on the evidence
- Synthesizes information across citations
- Provides assessments/evaluations of cited content

Analysis is NOT GROUNDED if it:
- Introduces NEW factual claims not in the citations (dates, names, statistics, events)
- Contradicts the cited facts
- Makes claims requiring additional evidence not provided

## RESPOND IN THIS EXACT FORMAT:
VERDICT: [GROUNDED or UNGROUNDED]
REASON: [One sentence explanation]"""

        try:
            response = await self.llm.complete(
                messages=[{"role": "user", "content": prompt}],
                tier=ModelTier.SIMPLE,
                max_tokens=50,
            )

            content = response.content.upper()
            is_grounded = "GROUNDED" in content and "UNGROUNDED" not in content

            # Extract reason if present
            reason = ""
            if "REASON:" in response.content:
                reason = response.content.split("REASON:")[-1].strip()

            return is_grounded, reason

        except Exception as e:
            logger.warning(
                "GROUNDING_CHECK_ERROR",
                error=str(e)[:100],
            )
            # Default to grounded on error (conservative)
            return True, "Error during check"

    async def _check_for_hidden_claims(self, free_content: str) -> tuple[bool, str | None]:
        """Check if a <free> block contains hidden factual claims.

        Based on FACTS Grounding sentence-level classification approach.

        Args:
            free_content: Content from a <free> block.

        Returns:
            Tuple of (has_claims, detected_claim_text).
        """
        import re
        from src.services.llm.types import ModelTier

        # Quick heuristic checks (skip LLM call when possible)
        content = free_content.strip()

        if len(content) < 20:
            return False, None  # Too short

        if content.startswith('#'):
            return False, None  # Header

        # Known transition phrases
        transitions = [
            'in summary,', 'moving on,', 'next,', 'finally,',
            'in conclusion,', 'to summarize,', 'overall,',
            'additionally,', 'furthermore,', 'however,',
            'the following', 'below', 'above', 'see also',
        ]
        lower = content.lower().strip()
        if any(lower.startswith(t) for t in transitions):
            return False, None

        # Check if content is likely factual (has dates, numbers, specific names)
        # Heuristic: if it contains specific data patterns, it's likely factual
        has_date = bool(re.search(r'\d{4}|\d{1,2}/\d{1,2}|\b(January|February|March|April|May|June|July|August|September|October|November|December)\b', content))
        has_number = bool(re.search(r'\d+%|\$\d|€\d|\d+\s*(million|billion|trillion)', content, re.IGNORECASE))
        has_specific_names = bool(re.search(r'[A-Z][a-z]+\s+[A-Z][a-z]+', content))  # Proper noun pattern

        # If heuristics suggest factual content, do LLM check
        if has_date or has_number or has_specific_names:
            try:
                prompt = f"""Classify this text:

TEXT: {free_content}

CLASSIFICATION:
- STRUCTURAL: Section headers, transition phrases, questions to reader
- FACTUAL: Contains specific facts (dates, names, statistics, events, claims)

Examples of STRUCTURAL: "## Key Findings", "Moving to the next topic:", "In summary,"
Examples of FACTUAL: "The regulation was adopted in 2024", "Banks face three challenges"

Answer ONLY "STRUCTURAL" or "FACTUAL":"""

                response = await self.llm.complete(
                    messages=[{"role": "user", "content": prompt}],
                    tier=ModelTier.SIMPLE,
                    max_tokens=10,
                )

                has_claims = "FACTUAL" in response.content.upper()
                return has_claims, free_content[:100] if has_claims else None

            except Exception as e:
                logger.warning(
                    "HIDDEN_CLAIMS_CHECK_ERROR",
                    error=str(e)[:100],
                )
                # Default to no claims on error (conservative)
                return False, None

        return False, None

    async def _is_common_knowledge(self, text: str) -> bool:
        """Quick check if text is common knowledge (no citation needed).

        Args:
            text: Text to check.

        Returns:
            True if text appears to be common knowledge.
        """
        import re

        # Heuristic: no specific dates, numbers, names, or technical claims
        has_specifics = bool(re.search(
            r'\d{4}|\d+%|\$\d|€\d|[A-Z][a-z]+\s+[A-Z][a-z]+',
            text
        ))

        if not has_specifics and len(text) < 100:
            return True  # Likely common knowledge

        return False

    async def _remediate_validation_issues(
        self,
        content: str,
        parsed_blocks: list[ParsedContent],
        issues: list[ValidationIssue],
    ) -> str:
        """Apply remediation for validation issues.

        Based on plan:
        - ungrounded_analysis: Allow if short or after header, else add hedging
        - baseless_analysis: Add hedging (never remove)
        - hidden_claims_in_free: Log warning only (synthesis error, not content error)

        Args:
            content: Report content.
            parsed_blocks: Parsed blocks for position lookup.
            issues: Validation issues to remediate.

        Returns:
            Remediated content.
        """
        config = self.config.grounding_validation

        # Sort by block index descending to modify from end (preserve positions)
        sorted_issues = sorted(issues, key=lambda i: i.block_index, reverse=True)

        for issue in sorted_issues:
            block = parsed_blocks[issue.block_index]

            if issue.issue_type == "ungrounded_analysis":
                # Allow short topic sentences
                if len(block.text) <= 50:
                    logger.debug(
                        "ALLOW_SHORT_ANALYSIS",
                        content=truncate(block.text, 50),
                    )
                    continue

                # Check if immediately follows a header
                if issue.block_index > 0:
                    prev_block = parsed_blocks[issue.block_index - 1]
                    if prev_block.tag_type == "free" and prev_block.text.strip().startswith('#'):
                        logger.debug(
                            "ALLOW_TOPIC_SENTENCE",
                            content=truncate(block.text, 50),
                        )
                        continue

                # Otherwise add hedging
                content = self._add_hedging_to_block(content, block, "Generally speaking, ")
                logger.info(
                    "HEDGING_ADDED_UNGROUNDED",
                    content=truncate(block.text, 50),
                )

            elif issue.issue_type == "baseless_analysis":
                # Add hedging to signal this is interpretation, not fact
                content = self._add_hedging_to_block(content, block, config.hedging_prefix)
                logger.info(
                    "HEDGING_ADDED_BASELESS",
                    content=truncate(block.text, 50),
                    reason=issue.reason,
                )

            elif issue.issue_type == "hidden_claims_in_free":
                # Just log - don't modify (synthesis error, content may be fine)
                logger.warning(
                    "HIDDEN_CLAIMS_IN_FREE",
                    content=issue.content,
                )

        return content

    def _add_hedging_to_block(
        self,
        content: str,
        block: ParsedContent,
        hedge_prefix: str,
    ) -> str:
        """Add hedging language to a block in the content.

        Args:
            content: Full report content.
            block: Block to add hedging to.
            hedge_prefix: Hedging prefix to add.

        Returns:
            Modified content with hedging.
        """
        # Find the block text in content
        position = content.find(block.text)
        if position == -1:
            logger.warning(
                "BLOCK_NOT_FOUND_FOR_HEDGING",
                block_text=truncate(block.text, 50),
            )
            return content

        # Lowercase first letter after hedge
        modified_text = block.text
        if modified_text and modified_text[0].isupper():
            modified_text = modified_text[0].lower() + modified_text[1:]

        hedged_text = f"{hedge_prefix}{modified_text}"

        # Replace in content
        return content[:position] + hedged_text + content[position + len(block.text):]
