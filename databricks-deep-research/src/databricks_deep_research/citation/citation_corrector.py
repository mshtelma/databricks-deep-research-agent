"""Stage 5: Citation Correction -- CiteFix-style hybrid matching.

Implements post-hoc citation correction with hybrid keyword + semantic matching.
When a claim is not fully supported by its current citation, this service:
1. Checks entailment between claim and evidence
2. Searches for better matching evidence from the evidence pool
3. Applies corrections: keep, replace, remove, or add_alternate

Ported from the app's citation verification pipeline.
Uses ``FrameworkLLMClient`` instead of app LLMClient.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from databricks_deep_research.citation.types import (
    CorrectionAction,
    CorrectionMetrics,
    CorrectionResult,
    RankedEvidence,
)
from databricks_deep_research.llm.client import FrameworkLLMClient, ModelTier

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _MatchContext:
    quarters: frozenset[str]
    years: frozenset[str]
    scopes: frozenset[str]
    metrics: frozenset[str]
    qualifiers: frozenset[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truncate(text: str | None, max_length: int = 100) -> str:
    """Truncate text for logging, adding ellipsis if truncated."""
    if text is None:
        return "<none>"
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CitationCorrectorConfig:
    """Configuration knobs for citation correction.

    Replaces the app's ``CitationCorrectionConfig`` Pydantic model.
    """

    lambda_weight: float = 0.8
    correction_threshold: float = 0.6
    allow_alternate_citations: bool = True


_DEFAULT_CONFIG = CitationCorrectorConfig()

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

CITATION_CORRECTION_PROMPT = """You are a Citation Corrector. A claim has been flagged as potentially misattributed.
Your task is to decide whether to KEEP, REPLACE, or REMOVE the citation.

## Claim to Verify
"{claim}"

## Current Evidence
{current_evidence}

## Available Evidence Options
{evidence_options}

## Decision Criteria

### KEEP
- The current evidence adequately supports the claim
- Key entities (numbers, names, dates) match between claim and evidence
- The claim can be reasonably inferred from the evidence

### REPLACE
- One of the evidence options better supports the claim
- The current evidence is too tangential or weak
- Choose the option with highest keyword/semantic overlap

### REMOVE
- None of the evidence options support this claim
- The claim appears to be unsupported by any available source
- This is a last resort when no suitable evidence exists

## Response Format (JSON)
```json
{{
  "action": "keep" | "replace" | "remove",
  "evidence_index": <1-5 if replacing, null otherwise>,
  "reasoning": "Brief explanation of your decision"
}}
```

Make your correction decision:"""


# ---------------------------------------------------------------------------
# Pydantic output model for structured LLM call
# ---------------------------------------------------------------------------


class CorrectionDecisionOutput(BaseModel):
    """Output from citation correction LLM call."""

    action: Literal["keep", "replace", "remove"] = Field(
        description="Correction action: keep, replace, or remove"
    )
    evidence_index: int | None = Field(
        default=None,
        description="1-indexed evidence option if replacing, null otherwise",
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of the decision",
    )


# ---------------------------------------------------------------------------
# Corrector
# ---------------------------------------------------------------------------


class CitationCorrector:
    """Citation correction service using hybrid keyword + semantic matching.

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
    _QUARTER_PATTERNS = {
        "q1": re.compile(r"\b(?:q1|first quarter|1q(?:25|24|26)?)\b", re.IGNORECASE),
        "q2": re.compile(r"\b(?:q2|second quarter|2q(?:25|24|26)?)\b", re.IGNORECASE),
        "q3": re.compile(r"\b(?:q3|third quarter|3q(?:25|24|26)?)\b", re.IGNORECASE),
        "q4": re.compile(r"\b(?:q4|fourth quarter|4q(?:25|24|26)?)\b", re.IGNORECASE),
    }
    _SCOPE_PATTERNS = {
        "full_year": re.compile(r"\b(?:full[- ]year|fiscal\s+20\d{2})\b", re.IGNORECASE),
        "year_to_date": re.compile(r"\b(?:year[- ]to[- ]date|ytd)\b", re.IGNORECASE),
        "guidance": re.compile(r"\b(?:guidance|outlook|expected|expectations)\b", re.IGNORECASE),
    }
    _METRIC_PATTERNS = {
        "identical_sales": re.compile(r"\b(?:identical sales|same-store sales|comparable sales)\b", re.IGNORECASE),
        "sales": re.compile(r"\b(?:sales|revenue)\b", re.IGNORECASE),
        "operating_profit": re.compile(r"\boperating profit\b", re.IGNORECASE),
        "operating_loss": re.compile(r"\boperating loss\b", re.IGNORECASE),
        "eps": re.compile(r"\b(?:eps|earnings per share)\b", re.IGNORECASE),
        "net_earnings": re.compile(r"\bnet earnings\b", re.IGNORECASE),
        "ecommerce": re.compile(r"\b(?:ecommerce|e-commerce|digital growth|digital sales)\b", re.IGNORECASE),
        "guidance": re.compile(r"\bguidance\b", re.IGNORECASE),
    }
    _QUALIFIER_PATTERNS = {
        "adjusted": re.compile(r"\badjusted\b", re.IGNORECASE),
        "reported": re.compile(r"\breported\b", re.IGNORECASE),
        "loss": re.compile(r"\bloss\b", re.IGNORECASE),
        "profit": re.compile(r"\bprofit\b", re.IGNORECASE),
    }

    def __init__(
        self,
        llm: FrameworkLLMClient,
        config: CitationCorrectorConfig | None = None,
    ) -> None:
        """Initialize the corrector.

        Args:
            llm: Framework LLM client for semantic correction checks.
            config: Optional configuration. Uses defaults if *None*.
        """
        self.llm = llm
        cfg = config or _DEFAULT_CONFIG
        self.lambda_weight = cfg.lambda_weight
        self.threshold = cfg.correction_threshold
        self.allow_alternates = cfg.allow_alternate_citations

    # -- Entity / keyword extraction ----------------------------------------

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

    # -- Scoring ------------------------------------------------------------

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
        hybrid = (
            self.lambda_weight * keyword_score
            + (1 - self.lambda_weight) * semantic_score
        )
        return hybrid

    def _extract_match_context(self, text: str) -> _MatchContext:
        """Extract coarse temporal and financial context from text."""
        lowered = text.lower()
        quarters = frozenset(
            quarter
            for quarter, pattern in self._QUARTER_PATTERNS.items()
            if pattern.search(text)
        )
        years = frozenset(re.findall(r"\b20\d{2}\b", lowered))
        scopes = frozenset(
            scope
            for scope, pattern in self._SCOPE_PATTERNS.items()
            if pattern.search(text)
        )
        metrics = frozenset(
            metric
            for metric, pattern in self._METRIC_PATTERNS.items()
            if pattern.search(text)
        )
        qualifiers = frozenset(
            qualifier
            for qualifier, pattern in self._QUALIFIER_PATTERNS.items()
            if pattern.search(text)
        )
        return _MatchContext(
            quarters=quarters,
            years=years,
            scopes=scopes,
            metrics=metrics,
            qualifiers=qualifiers,
        )

    def _context_penalty(
        self,
        claim_context: _MatchContext,
        evidence_context: _MatchContext,
    ) -> tuple[bool, float]:
        """Return ``(compatible, penalty)`` for a claim/evidence context pair."""
        penalty = 0.0

        if (
            claim_context.quarters
            and evidence_context.quarters
            and claim_context.quarters.isdisjoint(evidence_context.quarters)
        ):
            return False, 1.0

        if "full_year" in claim_context.scopes and evidence_context.quarters:
            return False, 1.0
        if claim_context.quarters and "full_year" in evidence_context.scopes:
            return False, 1.0
        if "year_to_date" in claim_context.scopes and "year_to_date" not in evidence_context.scopes:
            return False, 1.0

        if "guidance" in claim_context.scopes and "guidance" not in evidence_context.scopes:
            return False, 1.0
        if "guidance" not in claim_context.scopes and "guidance" in evidence_context.scopes:
            penalty += 0.45

        if "adjusted" in claim_context.qualifiers and "adjusted" not in evidence_context.qualifiers:
            return False, 1.0
        if "adjusted" not in claim_context.qualifiers and "adjusted" in evidence_context.qualifiers:
            penalty += 0.25

        if "loss" in claim_context.qualifiers and "profit" in evidence_context.qualifiers:
            return False, 1.0
        if "profit" in claim_context.qualifiers and "loss" in evidence_context.qualifiers:
            return False, 1.0

        if (
            claim_context.metrics
            and evidence_context.metrics
            and claim_context.metrics.isdisjoint(evidence_context.metrics)
        ):
            return False, 1.0

        if (
            claim_context.years
            and evidence_context.years
            and claim_context.years.isdisjoint(evidence_context.years)
        ):
            penalty += 0.20

        return True, penalty

    def score_claim_evidence(self, claim: str, evidence: str) -> float:
        """Score claim/evidence compatibility for correction and fallback matching."""
        base_score = self._compute_hybrid_score(claim, evidence)
        claim_context = self._extract_match_context(claim)
        evidence_context = self._extract_match_context(evidence)
        compatible, penalty = self._context_penalty(claim_context, evidence_context)
        if not compatible:
            return 0.0
        return max(0.0, base_score * (1.0 - penalty))

    # -- Entailment ---------------------------------------------------------

    def citation_entails(self, claim: str, evidence: str) -> bool:
        """Quick entailment check using hybrid scoring.

        Args:
            claim: Claim text.
            evidence: Evidence text.

        Returns:
            True if evidence likely entails the claim.
        """
        score = self.score_claim_evidence(claim, evidence)
        return score >= self.threshold

    # -- Evidence search ----------------------------------------------------

    def find_better_citation(
        self,
        claim: str,
        current_evidence: RankedEvidence | None,
        evidence_pool: list[RankedEvidence],
    ) -> tuple[RankedEvidence | None, float, int | None]:
        """Find better matching evidence from the pool.

        Args:
            claim: Claim text.
            current_evidence: Current citation evidence (may be None).
            evidence_pool: Pool of available evidence.

        Returns:
            Tuple of (best_evidence, best_score, evidence_pool_index).
        """
        if not evidence_pool:
            return None, 0.0, None

        current_score = 0.0
        if current_evidence:
            current_score = self.score_claim_evidence(
                claim, current_evidence.quote_text
            )

        best_evidence: RankedEvidence | None = None
        best_score = current_score
        best_index: int | None = None

        for evidence_index, evidence in enumerate(evidence_pool):
            if current_evidence and evidence.quote_text == current_evidence.quote_text:
                continue  # Skip current evidence

            score = self.score_claim_evidence(claim, evidence.quote_text)
            if score > best_score:
                best_score = score
                best_evidence = evidence
                best_index = evidence.evidence_pool_index
                if best_index is None:
                    best_index = evidence_index

        return best_evidence, best_score, best_index

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

            score = self.score_claim_evidence(claim, evidence.quote_text)

            # Slightly lower threshold for alternates
            if score >= self.threshold * 0.8:
                alternates.append((evidence, score))

        # Sort by score and return top N
        alternates.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in alternates[:max_alternates]]

    @staticmethod
    def _preserves_visible_citation(
        current_evidence: RankedEvidence,
        candidate_evidence: RankedEvidence,
    ) -> bool:
        """Return True when swapping evidence would keep the same citation slot."""
        if (
            current_evidence.source_pool_index is not None
            and candidate_evidence.source_pool_index is not None
        ):
            return (
                current_evidence.source_pool_index
                == candidate_evidence.source_pool_index
            )

        current_url = (
            current_evidence.canonical_source_url or current_evidence.source_url
        )
        candidate_url = (
            candidate_evidence.canonical_source_url or candidate_evidence.source_url
        )
        return bool(current_url and current_url == candidate_url)

    # -- Single citation correction -----------------------------------------

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
        if (
            current_evidence
            and current_verdict == "supported"
            and self.citation_entails(claim, current_evidence.quote_text)
        ):
            return CorrectionResult(
                claim_text=claim,
                correction_type=CorrectionAction.KEEP,
                original_evidence=current_evidence,
                corrected_evidence=current_evidence,
                original_evidence_index=current_evidence.evidence_pool_index,
                corrected_evidence_index=current_evidence.evidence_pool_index,
                reasoning="Citation is correct and fully supported.",
                confidence=1.0,
                evidence_match_score=1.0,
            )

        # Try to find better evidence
        current_score = 0.0
        current_index = current_evidence.evidence_pool_index if current_evidence else None
        if current_evidence:
            current_score = self.score_claim_evidence(claim, current_evidence.quote_text)

        better_evidence, better_score, better_index = self.find_better_citation(
            claim, current_evidence, evidence_pool
        )

        if (
            better_evidence
            and better_score >= self.threshold
            and better_score >= current_score + 0.05
        ):
            if (
                current_evidence is not None
                and self.allow_alternates
                and self._preserves_visible_citation(
                    current_evidence,
                    better_evidence,
                )
            ):
                return CorrectionResult(
                    claim_text=claim,
                    correction_type=CorrectionAction.ADD_ALTERNATE,
                    original_evidence=current_evidence,
                    corrected_evidence=current_evidence,
                    original_evidence_index=current_index,
                    corrected_evidence_index=current_index,
                    alternate_evidence=[better_evidence],
                    alternate_evidence_indices=(
                        [better_index] if better_index is not None else []
                    ),
                    reasoning=(
                        "Found a better supporting span within the same cited "
                        "source, so the citation stays in place and gains an "
                        "alternate evidence span."
                    ),
                    confidence=better_score,
                    evidence_match_score=better_score,
                )
            # Found better evidence -- replace
            correction_type = CorrectionAction.REPLACE
            corrected_evidence = better_evidence
            reasoning = f"Found better matching evidence (score: {better_score:.2f})"
        elif (
            current_evidence
            and current_score >= self.threshold * 0.8
        ):
            # Current evidence is acceptable, look for alternates
            alternates = self.find_alternate_citations(
                claim, current_evidence, evidence_pool
            )
            alternate_indices = [
                alternate.evidence_pool_index
                if alternate.evidence_pool_index is not None
                else next(
                    (
                        evidence_index
                        for evidence_index, evidence in enumerate(evidence_pool)
                        if evidence.quote_text == alternate.quote_text
                        and evidence.source_url == alternate.source_url
                    ),
                    None,
                )
                for alternate in alternates
            ]
            if alternates:
                return CorrectionResult(
                    claim_text=claim,
                    correction_type=CorrectionAction.ADD_ALTERNATE,
                    original_evidence=current_evidence,
                    corrected_evidence=current_evidence,
                    original_evidence_index=current_index,
                    corrected_evidence_index=current_index,
                    alternate_evidence=alternates,
                    alternate_evidence_indices=[
                        index for index in alternate_indices if index is not None
                    ],
                    reasoning=(
                        f"Added {len(alternates)} alternate citation(s) "
                        f"for additional support."
                    ),
                    confidence=0.8,
                    evidence_match_score=current_score,
                )
            # Keep original if no better options
            return CorrectionResult(
                claim_text=claim,
                correction_type=CorrectionAction.KEEP,
                original_evidence=current_evidence,
                corrected_evidence=current_evidence,
                original_evidence_index=current_index,
                corrected_evidence_index=current_index,
                reasoning="Citation is acceptable, no better alternatives found.",
                confidence=0.7,
                evidence_match_score=current_score,
            )
        else:
            # No suitable evidence found
            correction_type = CorrectionAction.REMOVE
            corrected_evidence = None
            reasoning = "No suitable evidence found to support this claim."

        return CorrectionResult(
            claim_text=claim,
            correction_type=correction_type,
            original_evidence=current_evidence,
            corrected_evidence=corrected_evidence,
            original_evidence_index=current_index,
            corrected_evidence_index=better_index if correction_type == CorrectionAction.REPLACE else None,
            reasoning=reasoning,
            confidence=better_score if better_evidence else 0.0,
            evidence_match_score=max(current_score, better_score),
        )

    # -- Batch correction ---------------------------------------------------

    async def correct_citations(
        self,
        claims_with_evidence: list[tuple[str, RankedEvidence | None, str | None]],
        evidence_pool: list[RankedEvidence],
    ) -> tuple[list[CorrectionResult], CorrectionMetrics]:
        """Correct citations for multiple claims.

        Args:
            claims_with_evidence: List of (claim_text, current_evidence, verdict)
                tuples.
            evidence_pool: Pool of available evidence.

        Returns:
            Tuple of (correction_results, metrics).
        """
        logger.info(
            "CITATION_CORRECTION_START claims_count=%d evidence_pool_size=%d",
            len(claims_with_evidence),
            len(evidence_pool),
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
            if result.correction_type == CorrectionAction.KEEP:
                metrics.kept += 1
            elif result.correction_type == CorrectionAction.REPLACE:
                metrics.replaced += 1
            elif result.correction_type == CorrectionAction.REMOVE:
                metrics.removed += 1
            elif result.correction_type == CorrectionAction.ADD_ALTERNATE:
                metrics.added_alternate += 1

            logger.debug(
                "CITATION_CORRECTED claim=%s correction_type=%s confidence=%.2f",
                _truncate(claim, 50),
                result.correction_type.value,
                result.confidence,
            )

        logger.info(
            "CITATION_CORRECTION_COMPLETE kept=%d replaced=%d removed=%d "
            "added_alternate=%d correction_rate=%.1f%%",
            metrics.kept,
            metrics.replaced,
            metrics.removed,
            metrics.added_alternate,
            metrics.correction_rate * 100,
        )

        return results, metrics

    # -- LLM-assisted correction --------------------------------------------

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
        if (
            quick_result.correction_type == CorrectionAction.KEEP
            and quick_result.confidence > 0.9
        ):
            return quick_result
        if quick_result.correction_type == CorrectionAction.REMOVE:
            return quick_result

        # Use LLM for uncertain cases
        try:
            # Build evidence options for LLM
            evidence_options: list[str] = []
            for i, e in enumerate(evidence_pool[:5]):  # Limit to top 5
                evidence_options.append(
                    f"[{i + 1}] {_truncate(e.quote_text, 800)}"
                )

            prompt = CITATION_CORRECTION_PROMPT.format(
                claim=claim,
                current_evidence=(
                    current_evidence.quote_text if current_evidence else "None"
                ),
                evidence_options="\n".join(evidence_options),
            )

            response = await self.llm.complete(
                messages=[{"role": "user", "content": prompt}],
                tier=ModelTier.simple,
                structured_output=CorrectionDecisionOutput,
            )

            if response.structured:
                output: CorrectionDecisionOutput = response.structured
                action = output.action

                if action == "keep":
                    correction_type = CorrectionAction.KEEP
                    corrected_evidence = None
                elif action == "replace" and output.evidence_index is not None:
                    correction_type = CorrectionAction.REPLACE
                    idx = output.evidence_index - 1  # 1-indexed in prompt
                    if 0 <= idx < len(evidence_pool):
                        corrected_evidence = evidence_pool[idx]
                    else:
                        return quick_result  # Invalid index, fall back
                elif action == "remove":
                    correction_type = CorrectionAction.REMOVE
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
                "LLM_CORRECTION_FAILED error=%s falling_back_to=hybrid",
                str(e)[:100],
            )

        # Fall back to hybrid result
        return quick_result
