"""Citation Corrector Service - Stage 5 of the Citation Verification Pipeline.

Implements CiteFix-style citation correction with hybrid keyword+semantic matching.
When a claim is not fully supported by its current citation, this service:
1. Checks entailment between claim and evidence
2. Searches for better matching evidence from the evidence pool
3. Applies corrections: keep, replace, remove, or add_alternate
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from deep_research.agent.prompts.citation.verification import CITATION_CORRECTION_PROMPT
from deep_research.core.app_config import get_app_config
from deep_research.core.logging_utils import get_logger, truncate
from deep_research.services.citation.evidence_selector import RankedEvidence
from deep_research.services.llm.client import LLMClient
from deep_research.services.llm.types import ModelTier

logger = get_logger(__name__)


# Pydantic models for structured LLM output
class CorrectionDecisionOutput(BaseModel):
    """Output from citation correction LLM call."""

    action: Literal["keep", "replace", "remove"] = Field(
        description="Correction action: keep, replace, or remove"
    )
    evidence_index: int | None = Field(
        default=None,
        description="1-indexed evidence option if replacing, null otherwise"
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of the decision"
    )


class CorrectionType(Enum):
    """Types of citation corrections that can be applied."""

    KEEP = "keep"  # Citation is correct as-is
    REPLACE = "replace"  # Replace with better evidence
    REMOVE = "remove"  # Remove citation (no evidence supports claim)
    ADD_ALTERNATE = "add_alternate"  # Add additional supporting evidence


@dataclass
class CorrectionResult:
    """Result of citation correction for a single claim."""

    claim_text: str
    correction_type: CorrectionType
    original_evidence: RankedEvidence | None
    corrected_evidence: RankedEvidence | None
    alternate_evidence: list[RankedEvidence] = field(default_factory=list)
    reasoning: str = ""
    confidence: float = 0.0


@dataclass
class CorrectionMetrics:
    """Aggregate metrics for citation corrections."""

    total_claims: int = 0
    kept: int = 0
    replaced: int = 0
    removed: int = 0
    added_alternate: int = 0

    @property
    def correction_rate(self) -> float:
        """Percentage of claims that needed correction."""
        if self.total_claims == 0:
            return 0.0
        corrected = self.replaced + self.removed + self.added_alternate
        return corrected / self.total_claims


class CitationCorrector:
    """Citation correction service using hybrid keyword+semantic matching.

    Applies CiteFix methodology:
    - Keyword entailment check for quick filtering
    - Semantic similarity for precise matching
    - Configurable lambda weight between keyword and semantic scores
    """

    # Keywords that must be present in evidence to support claim
    REQUIRED_ENTITY_PATTERNS = [
        r"\$[\d,.]+[BMKbmk]?\b",  # Currency amounts
        r"\d+(?:\.\d+)?%",  # Percentages
        r"\d{4}",  # Years
        r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*",  # Proper nouns
    ]

    def __init__(self, llm: LLMClient):
        """Initialize the corrector with LLM client.

        Args:
            llm: LLM client for semantic correction checks.
        """
        self.llm = llm
        config = get_app_config().citation_verification
        self.correction_config = config.citation_correction
        self.lambda_weight = self.correction_config.lambda_weight
        self.threshold = self.correction_config.correction_threshold
        self.allow_alternates = self.correction_config.allow_alternate_citations

    def _extract_key_entities(self, text: str) -> set[str]:
        """Extract key entities from text for keyword matching.

        Args:
            text: Text to extract entities from.

        Returns:
            Set of extracted entity strings.
        """
        entities: set[str] = set()

        for pattern in self.REQUIRED_ENTITY_PATTERNS:
            matches = re.findall(pattern, text)
            entities.update(matches)

        # Also extract significant words (longer than 4 chars, not stopwords)
        stopwords = {
            "about", "after", "before", "between", "could", "during",
            "every", "their", "there", "these", "those", "through",
            "under", "until", "where", "which", "while", "would",
        }
        words = re.findall(r"\b[a-zA-Z]{5,}\b", text.lower())
        significant = {w for w in words if w not in stopwords}
        entities.update(significant)

        return entities

    def _compute_keyword_overlap(self, claim: str, evidence: str) -> float:
        """Compute keyword overlap score between claim and evidence.

        Args:
            claim: Claim text.
            evidence: Evidence text.

        Returns:
            Overlap score between 0 and 1.
        """
        claim_entities = self._extract_key_entities(claim)
        evidence_entities = self._extract_key_entities(evidence)

        if not claim_entities:
            return 0.0

        overlap = claim_entities & evidence_entities
        return len(overlap) / len(claim_entities)

    def _compute_semantic_similarity(self, claim: str, evidence: str) -> float:
        """Compute semantic similarity between claim and evidence.

        Uses token overlap as a lightweight semantic proxy.
        For production, could integrate embedding-based similarity.

        Args:
            claim: Claim text.
            evidence: Evidence text.

        Returns:
            Similarity score between 0 and 1.
        """
        # Tokenize and normalize
        claim_tokens = set(claim.lower().split())
        evidence_tokens = set(evidence.lower().split())

        # Remove very short tokens
        claim_tokens = {t for t in claim_tokens if len(t) > 2}
        evidence_tokens = {t for t in evidence_tokens if len(t) > 2}

        if not claim_tokens or not evidence_tokens:
            return 0.0

        # Compute Jaccard-like similarity
        intersection = len(claim_tokens & evidence_tokens)
        union = len(claim_tokens | evidence_tokens)

        return intersection / union if union > 0 else 0.0

    def _compute_hybrid_score(self, claim: str, evidence: str) -> float:
        """Compute hybrid entailment score combining keyword and semantic.

        Args:
            claim: Claim text.
            evidence: Evidence text.

        Returns:
            Hybrid score between 0 and 1.
        """
        keyword_score = self._compute_keyword_overlap(claim, evidence)
        semantic_score = self._compute_semantic_similarity(claim, evidence)

        # Lambda-weighted combination (CiteFix approach)
        hybrid = self.lambda_weight * keyword_score + (1 - self.lambda_weight) * semantic_score

        return hybrid

    def citation_entails(self, claim: str, evidence: str) -> bool:
        """Quick entailment check using hybrid scoring.

        Args:
            claim: Claim text.
            evidence: Evidence text.

        Returns:
            True if evidence likely entails the claim.
        """
        score = self._compute_hybrid_score(claim, evidence)
        return score >= self.threshold

    def find_better_citation(
        self,
        claim: str,
        current_evidence: RankedEvidence | None,
        evidence_pool: list[RankedEvidence],
    ) -> tuple[RankedEvidence | None, float]:
        """Find better matching evidence from the pool.

        Args:
            claim: Claim text.
            current_evidence: Current citation evidence (may be None).
            evidence_pool: Pool of available evidence.

        Returns:
            Tuple of (best_evidence, best_score).
        """
        if not evidence_pool:
            return None, 0.0

        current_score = 0.0
        if current_evidence:
            current_score = self._compute_hybrid_score(claim, current_evidence.quote_text)

        best_evidence: RankedEvidence | None = None
        best_score = current_score

        for evidence in evidence_pool:
            if current_evidence and evidence.quote_text == current_evidence.quote_text:
                continue  # Skip current evidence

            score = self._compute_hybrid_score(claim, evidence.quote_text)

            if score > best_score:
                best_score = score
                best_evidence = evidence

        return best_evidence, best_score

    def find_alternate_citations(
        self,
        claim: str,
        primary_evidence: RankedEvidence | None,
        evidence_pool: list[RankedEvidence],
        max_alternates: int = 2,
    ) -> list[RankedEvidence]:
        """Find additional supporting evidence for a claim.

        Args:
            claim: Claim text.
            primary_evidence: Primary citation evidence.
            evidence_pool: Pool of available evidence.
            max_alternates: Maximum number of alternates to return.

        Returns:
            List of alternate evidence spans.
        """
        if not self.allow_alternates:
            return []

        alternates: list[tuple[RankedEvidence, float]] = []

        for evidence in evidence_pool:
            # Skip primary evidence
            if primary_evidence and evidence.quote_text == primary_evidence.quote_text:
                continue

            score = self._compute_hybrid_score(claim, evidence.quote_text)

            if score >= self.threshold * 0.8:  # Slightly lower threshold for alternates
                alternates.append((evidence, score))

        # Sort by score and return top N
        alternates.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in alternates[:max_alternates]]

    async def correct_single_citation(
        self,
        claim: str,
        current_evidence: RankedEvidence | None,
        evidence_pool: list[RankedEvidence],
        current_verdict: str | None = None,
    ) -> CorrectionResult:
        """Correct a single citation using hybrid matching.

        Args:
            claim: Claim text.
            current_evidence: Current citation evidence.
            evidence_pool: Pool of available evidence.
            current_verdict: Current verification verdict (if any).

        Returns:
            CorrectionResult with correction type and new evidence.
        """
        # If already supported and entails, keep it
        if current_evidence and current_verdict == "supported":
            if self.citation_entails(claim, current_evidence.quote_text):
                return CorrectionResult(
                    claim_text=claim,
                    correction_type=CorrectionType.KEEP,
                    original_evidence=current_evidence,
                    corrected_evidence=current_evidence,
                    reasoning="Citation is correct and fully supported.",
                    confidence=1.0,
                )

        # Try to find better evidence
        better_evidence, better_score = self.find_better_citation(
            claim, current_evidence, evidence_pool
        )

        if better_evidence and better_score >= self.threshold:
            # Found better evidence - replace
            correction_type = CorrectionType.REPLACE
            corrected_evidence = better_evidence
            reasoning = f"Found better matching evidence (score: {better_score:.2f})"
        elif current_evidence and self._compute_hybrid_score(claim, current_evidence.quote_text) >= self.threshold * 0.6:
            # Current evidence is acceptable, look for alternates
            alternates = self.find_alternate_citations(
                claim, current_evidence, evidence_pool
            )
            if alternates:
                return CorrectionResult(
                    claim_text=claim,
                    correction_type=CorrectionType.ADD_ALTERNATE,
                    original_evidence=current_evidence,
                    corrected_evidence=current_evidence,
                    alternate_evidence=alternates,
                    reasoning=f"Added {len(alternates)} alternate citation(s) for additional support.",
                    confidence=0.8,
                )
            # Keep original if no better options
            return CorrectionResult(
                claim_text=claim,
                correction_type=CorrectionType.KEEP,
                original_evidence=current_evidence,
                corrected_evidence=current_evidence,
                reasoning="Citation is acceptable, no better alternatives found.",
                confidence=0.7,
            )
        elif better_evidence:
            # Some evidence found but below threshold
            correction_type = CorrectionType.REPLACE
            corrected_evidence = better_evidence
            reasoning = f"Replaced with best available evidence (score: {better_score:.2f})"
        else:
            # No suitable evidence found
            correction_type = CorrectionType.REMOVE
            corrected_evidence = None
            reasoning = "No suitable evidence found to support this claim."

        return CorrectionResult(
            claim_text=claim,
            correction_type=correction_type,
            original_evidence=current_evidence,
            corrected_evidence=corrected_evidence,
            reasoning=reasoning,
            confidence=better_score if better_evidence else 0.0,
        )

    async def correct_citations(
        self,
        claims_with_evidence: list[tuple[str, RankedEvidence | None, str | None]],
        evidence_pool: list[RankedEvidence],
    ) -> tuple[list[CorrectionResult], CorrectionMetrics]:
        """Correct citations for multiple claims.

        Args:
            claims_with_evidence: List of (claim_text, current_evidence, verdict) tuples.
            evidence_pool: Pool of available evidence.

        Returns:
            Tuple of (correction_results, metrics).
        """
        logger.info(
            "CITATION_CORRECTION_START",
            claims_count=len(claims_with_evidence),
            evidence_pool_size=len(evidence_pool),
        )

        results: list[CorrectionResult] = []
        metrics = CorrectionMetrics(total_claims=len(claims_with_evidence))

        for claim, evidence, verdict in claims_with_evidence:
            result = await self.correct_single_citation(
                claim=claim,
                current_evidence=evidence,
                evidence_pool=evidence_pool,
                current_verdict=verdict,
            )

            results.append(result)

            # Update metrics
            if result.correction_type == CorrectionType.KEEP:
                metrics.kept += 1
            elif result.correction_type == CorrectionType.REPLACE:
                metrics.replaced += 1
            elif result.correction_type == CorrectionType.REMOVE:
                metrics.removed += 1
            elif result.correction_type == CorrectionType.ADD_ALTERNATE:
                metrics.added_alternate += 1

            logger.debug(
                "CITATION_CORRECTED",
                claim=truncate(claim, 50),
                correction_type=result.correction_type.value,
                confidence=result.confidence,
            )

        logger.info(
            "CITATION_CORRECTION_COMPLETE",
            kept=metrics.kept,
            replaced=metrics.replaced,
            removed=metrics.removed,
            added_alternate=metrics.added_alternate,
            correction_rate=f"{metrics.correction_rate:.1%}",
        )

        return results, metrics

    async def correct_citation_with_llm(
        self,
        claim: str,
        current_evidence: RankedEvidence | None,
        evidence_pool: list[RankedEvidence],
    ) -> CorrectionResult:
        """Use LLM for more sophisticated citation correction.

        Falls back to hybrid matching if LLM fails.

        Args:
            claim: Claim text.
            current_evidence: Current citation evidence.
            evidence_pool: Pool of available evidence.

        Returns:
            CorrectionResult with LLM-based correction.
        """
        # First try quick hybrid check
        quick_result = await self.correct_single_citation(
            claim, current_evidence, evidence_pool
        )

        # If clearly correct or clearly wrong, return quick result
        if quick_result.correction_type == CorrectionType.KEEP and quick_result.confidence > 0.9:
            return quick_result
        if quick_result.correction_type == CorrectionType.REMOVE:
            return quick_result

        # Use LLM for uncertain cases
        try:
            # Build evidence options for LLM
            evidence_options = []
            for i, e in enumerate(evidence_pool[:5]):  # Limit to top 5
                evidence_options.append(f"[{i+1}] {truncate(e.quote_text, 200)}")

            prompt = CITATION_CORRECTION_PROMPT.format(
                claim=claim,
                current_evidence=current_evidence.quote_text if current_evidence else "None",
                evidence_options="\n".join(evidence_options),
            )

            response = await self.llm.complete(
                messages=[{"role": "user", "content": prompt}],
                tier=ModelTier.SIMPLE,
                structured_output=CorrectionDecisionOutput,
            )

            if response.structured:
                output: CorrectionDecisionOutput = response.structured
                action = output.action

                if action == "keep":
                    correction_type = CorrectionType.KEEP
                    corrected_evidence = None
                elif action == "replace" and output.evidence_index is not None:
                    correction_type = CorrectionType.REPLACE
                    idx = output.evidence_index - 1  # 1-indexed in prompt
                    if 0 <= idx < len(evidence_pool):
                        corrected_evidence = evidence_pool[idx]
                    else:
                        return quick_result  # Invalid index, fall back
                elif action == "remove":
                    correction_type = CorrectionType.REMOVE
                    corrected_evidence = None
                else:
                    return quick_result  # Unknown action, fall back

                return CorrectionResult(
                    claim_text="",  # Will be set by caller
                    correction_type=correction_type,
                    original_evidence=None,  # Will be set by caller
                    corrected_evidence=corrected_evidence,
                    reasoning=output.reasoning,
                    confidence=0.85,  # LLM-based correction
                )

        except Exception as e:
            logger.warning(
                "LLM_CORRECTION_FAILED",
                error=str(e)[:100],
                falling_back_to="hybrid",
            )

        # Fall back to hybrid result
        return quick_result
