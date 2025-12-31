"""Citation Verification Pipeline - Orchestrates the 6-stage citation verification.

This service coordinates the entire citation verification workflow:
1. Evidence Pre-Selection (before generation)
2. Interleaved Generation (during synthesis)
3. Confidence Classification (after claim extraction)
4. Isolated Verification (for each claim)
5. Citation Correction (post-hoc fixes)
6. Numeric QA Verification (for numeric claims)
"""

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

import mlflow

from src.agent.config import get_citation_config_for_depth
from src.agent.state import ClaimInfo, EvidenceInfo, ResearchState, SourceInfo
from src.core.app_config import CitationVerificationConfig, GenerationMode, get_app_config
from src.core.logging_utils import get_logger, truncate
from src.services.citation.citation_corrector import CitationCorrector, CorrectionType
from src.services.citation.claim_generator import InterleavedClaim, InterleavedGenerator
from src.services.citation.confidence_classifier import ConfidenceClassifier
from src.services.citation.content_evaluator import evaluate_content_quality
from src.services.citation.evidence_selector import EvidencePreSelector, RankedEvidence
from src.services.citation.isolated_verifier import IsolatedVerifier
from src.services.citation.numeric_verifier import NumericVerifier
from src.services.llm.client import LLMClient

logger = get_logger(__name__)


@dataclass
class VerificationEvent:
    """Event emitted during citation verification."""

    event_type: str  # claim_generated, claim_verified, citation_corrected, etc.
    data: dict[str, Any]


class CitationVerificationPipeline:
    """Orchestrates the 6-stage citation verification pipeline.

    Integrates with the Synthesizer to provide claim-level attribution:
    - Pre-selects evidence from sources before generation
    - Generates claims constrained by available evidence
    - Verifies each claim in isolation
    - Corrects citations if needed
    """

    def __init__(self, llm: LLMClient, depth: str | None = None):
        """Initialize the pipeline with LLM client.

        Args:
            llm: LLM client for all stages.
            depth: Research depth (light/medium/extended) for per-depth config.
                   If None, uses global citation_verification config.
        """
        self.llm = llm
        self.depth = depth

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

    @mlflow.trace(name="citation_pipeline.preselect_evidence", span_type="CHAIN")
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
            quality = evaluate_content_quality(source.get("content", ""), query)
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

    @mlflow.trace(name="citation_pipeline.generate_with_citations", span_type="CHAIN")
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

    @mlflow.trace(name="citation_pipeline.verify_claims", span_type="CHAIN")
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

    @mlflow.trace(name="citation_pipeline.verify_numeric", span_type="CHAIN")
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

    @mlflow.trace(name="citation_pipeline.correct_citations", span_type="CHAIN")
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
                claim.abstained = True

                yield VerificationEvent(
                    event_type="citation_corrected",
                    data={
                        "claim_id": id(claim),
                        "claim_text": claim.claim_text,
                        "correction_type": result.correction_type.value,
                        "reasoning": result.reasoning,
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
                    "warning": None,
                },
            )
            return

        # Natural/Strict modes: run full verification pipeline
        logger.info(
            "CITATION_PIPELINE_MODE",
            mode=generation_mode.value,
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
                    "warning": None,
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

        # Stage 2: Generate with interleaved claims
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

        # Update verification summary after corrections
        state.update_verification_summary()

        if state.verification_summary:
            yield VerificationEvent(
                event_type="verification_summary",
                data={
                    "message_id": str(state.session_id),
                    "total_claims": state.verification_summary.total_claims,
                    "supported": state.verification_summary.supported_count,
                    "partial": state.verification_summary.partial_count,
                    "unsupported": state.verification_summary.unsupported_count,
                    "contradicted": state.verification_summary.contradicted_count,
                    "abstained_count": state.verification_summary.abstained_count,
                    "citation_corrections": correction_count,
                    "warning": state.verification_summary.warning,
                },
            )

        # Complete state with full content
        state.complete(full_content)

        logger.info(
            "CITATION_PIPELINE_COMPLETE",
            claims=len(generated_claims),
            verified=sum(1 for c in generated_claims if c.verification_verdict),
            supported=sum(1 for c in generated_claims if c.verification_verdict == "supported"),
        )
