"""Stage 3: Confidence Classification.

Classifies claim confidence levels using the HaluGate-style approach
to route high-confidence claims to quick verification and low-confidence
claims to full analytical verification.

Ported from the app's citation verification pipeline. Uses rule-based
linguistic indicators -- no LLM calls required.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from databricks_deep_research.citation.types import ConfidenceLevel, ConfidenceResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfidenceClassifierConfig:
    """Configuration for confidence classification thresholds.

    Replaces the app's ``ConfidenceClassificationConfig`` Pydantic model
    with a plain frozen dataclass to avoid app dependencies.
    """

    high_threshold: float = 0.70
    low_threshold: float = 0.40
    quote_match_bonus: float = 0.4
    hedging_word_penalty: float = 0.2


_DEFAULT_CONFIG = ConfidenceClassifierConfig()

# ---------------------------------------------------------------------------
# Linguistic indicator lists
# ---------------------------------------------------------------------------

# High confidence indicators -- direct quotes, citations, attributions
HIGH_CONFIDENCE_PHRASES: list[str] = [
    "according to",
    "states that",
    "reports that",
    "shows that",
    "indicates that",
    "confirms that",
    "demonstrates that",
    "as stated in",
    "as reported by",
    "based on",
    "per the",
    "as per",
    "the report notes",
    "the study found",
    "data shows",
]

# Low confidence indicators -- hedging, uncertainty
LOW_CONFIDENCE_PHRASES: list[str] = [
    "may",
    "might",
    "could",
    "possibly",
    "perhaps",
    "likely",
    "probably",
    "appears to",
    "seems to",
    "suggests that",
    "approximately",
    "around",
    "about",
    "roughly",
    "estimated",
    "it is possible",
    "it is believed",
    "some say",
    "reportedly",
    "allegedly",
]


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class ConfidenceClassifier:
    """Stage 3: Confidence Classification.

    Classifies claims into high/medium/low confidence levels using
    linguistic indicators (HaluGate-style), without requiring logprobs.
    """

    def __init__(
        self,
        config: ConfidenceClassifierConfig | None = None,
    ) -> None:
        """Initialize the confidence classifier.

        Args:
            config: Optional classifier configuration. Uses defaults if *None*.
        """
        self._config = config or _DEFAULT_CONFIG

    def classify(
        self,
        claim_text: str,
        evidence_quote: str | None = None,
    ) -> ConfidenceResult:
        """Classify the confidence level of a claim.

        Uses linguistic indicators and optional evidence matching:
        - High: Direct quotes, strong attribution language
        - Medium: Neutral factual statements
        - Low: Hedging, uncertainty, comparative language

        Args:
            claim_text: The claim text to classify.
            evidence_quote: Optional evidence quote for matching.

        Returns:
            ConfidenceResult with level, score, and reasoning.
        """
        claim_lower = claim_text.lower()
        indicators: list[str] = []

        # Base score starts at medium (0.5)
        score = 0.5

        # Check for high confidence indicators
        high_matches = self._count_phrase_matches(claim_lower, HIGH_CONFIDENCE_PHRASES)
        for phrase in HIGH_CONFIDENCE_PHRASES:
            if phrase in claim_lower:
                indicators.append(f"high: '{phrase}'")

        # Check for low confidence indicators
        low_matches = self._count_phrase_matches(claim_lower, LOW_CONFIDENCE_PHRASES)
        for phrase in LOW_CONFIDENCE_PHRASES:
            if phrase in claim_lower:
                indicators.append(f"low: '{phrase}'")

        # Adjust score based on indicator counts
        score += high_matches * 0.15  # Each high indicator adds 0.15
        score -= low_matches * self._config.hedging_word_penalty

        # Check for quote match if evidence provided
        if evidence_quote:
            quote_overlap = self._compute_quote_overlap(claim_text, evidence_quote)
            if quote_overlap > 0.5:
                score += self._config.quote_match_bonus
                indicators.append(f"quote_match: {quote_overlap:.2f}")

        # Clamp score to [0, 1]
        score = max(0.0, min(1.0, score))

        # Determine level based on thresholds
        if score >= self._config.high_threshold:
            level = ConfidenceLevel.HIGH
        elif score < self._config.low_threshold:
            level = ConfidenceLevel.LOW
        else:
            level = ConfidenceLevel.MEDIUM

        reasoning = self._build_reasoning(level, score, indicators)

        logger.debug(
            "CONFIDENCE_CLASSIFIED claim_preview=%s level=%s score=%.3f",
            claim_text[:50],
            level.value,
            score,
        )

        return ConfidenceResult(
            level=level,
            score=score,
            indicators=indicators,
            reasoning=reasoning,
        )

    def classify_batch(
        self,
        claims: list[tuple[str, str | None]],
    ) -> list[ConfidenceResult]:
        """Classify multiple claims.

        Args:
            claims: List of (claim_text, evidence_quote) tuples.

        Returns:
            List of ConfidenceResult objects.
        """
        return [
            self.classify(claim_text, evidence_quote)
            for claim_text, evidence_quote in claims
        ]

    def is_high_confidence(
        self,
        claim_text: str,
        evidence_quote: str | None = None,
    ) -> bool:
        """Quick check if a claim is high confidence.

        Convenience method for routing decisions.
        """
        result = self.classify(claim_text, evidence_quote)
        return result.level == ConfidenceLevel.HIGH

    def should_use_quick_verification(
        self,
        claim_text: str,
        evidence_quote: str | None = None,
    ) -> bool:
        """Determine if quick verification is appropriate.

        High confidence claims can use quick (simple tier) verification.
        Low confidence claims require full analytical verification.
        """
        result = self.classify(claim_text, evidence_quote)
        return result.level == ConfidenceLevel.HIGH

    # -- Private helpers ----------------------------------------------------

    def _count_phrase_matches(self, text: str, phrases: list[str]) -> int:
        """Count how many phrases match in the text."""
        count = 0
        for phrase in phrases:
            if phrase in text:
                count += 1
        return count

    def _compute_quote_overlap(self, claim: str, evidence: str) -> float:
        """Compute word overlap between claim and evidence.

        Args:
            claim: Claim text.
            evidence: Evidence quote.

        Returns:
            Overlap ratio (0.0 - 1.0).
        """
        # Extract significant words (4+ chars)
        claim_words = set(re.findall(r"\b\w{4,}\b", claim.lower()))
        evidence_words = set(re.findall(r"\b\w{4,}\b", evidence.lower()))

        if not claim_words:
            return 0.0

        overlap = len(claim_words & evidence_words) / len(claim_words)
        return overlap

    def _build_reasoning(
        self,
        level: ConfidenceLevel,
        score: float,
        indicators: list[str],
    ) -> str:
        """Build human-readable reasoning for the classification."""
        if not indicators:
            return f"Neutral claim with no strong indicators (score: {score:.2f})"

        indicator_summary = ", ".join(indicators[:5])  # Limit to 5
        return (
            f"Classified as {level.value} (score: {score:.2f}) "
            f"due to: {indicator_summary}"
        )
