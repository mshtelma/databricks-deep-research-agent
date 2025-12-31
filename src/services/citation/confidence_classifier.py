"""Stage 3: Confidence Classification service.

Classifies claim confidence levels using the HaluGate-style approach
to route high-confidence claims to quick verification and low-confidence
claims to full analytical verification.
"""

import re
from dataclasses import dataclass
from enum import Enum

from src.core.app_config import get_app_config
from src.core.logging_utils import get_logger

logger = get_logger(__name__)


class ConfidenceLevel(str, Enum):
    """HaluGate-style confidence levels for routing."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ConfidenceResult:
    """Result of confidence classification."""

    level: ConfidenceLevel
    score: float  # 0.0 - 1.0
    indicators: list[str]  # Matched indicators
    reasoning: str


# High confidence indicators - direct quotes, citations, attributions
HIGH_CONFIDENCE_PHRASES = [
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

# Low confidence indicators - hedging, uncertainty
LOW_CONFIDENCE_PHRASES = [
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


class ConfidenceClassifier:
    """Stage 3: Confidence Classification.

    Classifies claims into high/medium/low confidence levels using
    linguistic indicators (HaluGate-style), without requiring logprobs.
    """

    def __init__(self) -> None:
        """Initialize the confidence classifier."""
        self._config = get_app_config().citation_verification.confidence_classification

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
        elif score <= self._config.low_threshold:
            level = ConfidenceLevel.LOW
        else:
            level = ConfidenceLevel.MEDIUM

        reasoning = self._build_reasoning(level, score, indicators)

        logger.debug(
            "CONFIDENCE_CLASSIFIED",
            claim_preview=claim_text[:50],
            level=level.value,
            score=round(score, 3),
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
        return f"Classified as {level.value} (score: {score:.2f}) due to: {indicator_summary}"

    def is_high_confidence(self, claim_text: str, evidence_quote: str | None = None) -> bool:
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
