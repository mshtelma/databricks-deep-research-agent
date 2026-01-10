"""Stage 6: Numeric QA Verification service.

Verifies numeric claims using a QA-based approach (QAFactEval pattern):
1. Generate questions about numeric values
2. Answer questions from both claim and evidence separately
3. Compare answers to detect mismatches
"""

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import BaseModel, Field

from src.agent.prompts.citation.verification import NUMERIC_QA_PROMPT
from src.core.app_config import get_app_config
from src.core.logging_utils import get_logger, truncate
from src.services.citation.evidence_selector import RankedEvidence
from src.services.llm.client import LLMClient
from src.services.llm.types import ModelTier

logger = get_logger(__name__)


# Pydantic models for structured LLM output
class QAPairOutput(BaseModel):
    """A single QA pair for numeric verification."""

    question: str = Field(description="Question about the numeric value")
    claim_answer: str = Field(description="Answer based on the claim only")
    evidence_answer: str = Field(description="Answer based on the evidence only")


class NumericQAOutput(BaseModel):
    """Output from numeric QA verification LLM call."""

    qa_pairs: list[QAPairOutput] = Field(
        default_factory=list,
        description="List of QA pairs for verification"
    )


@dataclass
class NumericValue:
    """Parsed numeric value with unit and context."""

    raw_text: str
    normalized_value: Decimal | None
    unit: str | None
    entity: str | None
    multiplier: int = 1


@dataclass
class QAVerificationResult:
    """Result of a single QA verification."""

    question: str
    claim_answer: str
    evidence_answer: str
    match: bool
    normalized_comparison: dict[str, Any] | None


@dataclass
class NumericVerificationResult:
    """Complete numeric verification result."""

    claim_text: str
    parsed_value: NumericValue
    qa_results: list[QAVerificationResult]
    overall_match: bool
    derivation_type: str  # "direct" or "computed"
    confidence: float


# Common multiplier patterns
MULTIPLIER_PATTERNS = {
    r"\btrillion\b": 1_000_000_000_000,
    r"\bbillion\b": 1_000_000_000,
    r"\bmillion\b": 1_000_000,
    r"\bthousand\b": 1_000,
    r"[Tt]": 1_000_000_000_000,  # $1T
    r"[Bb]": 1_000_000_000,  # $1B
    r"[Mm]": 1_000_000,  # $1M
    r"[Kk]": 1_000,  # $1K
}

# Unit patterns
UNIT_PATTERNS = [
    (r"\$", "USD"),
    (r"€", "EUR"),
    (r"£", "GBP"),
    (r"¥", "JPY"),
    (r"%", "percent"),
    (r"percent", "percent"),
    (r"percentage", "percent"),
    (r"years?", "years"),
    (r"months?", "months"),
    (r"days?", "days"),
    (r"hours?", "hours"),
    (r"users?", "users"),
    (r"customers?", "customers"),
    (r"employees?", "employees"),
]


class NumericVerifier:
    """Stage 6: Numeric QA Verification.

    Verifies numeric claims using QA-based comparison (QAFactEval pattern).
    This catches semantic errors that simple text matching would miss.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize the numeric verifier.

        Args:
            llm_client: LLM client for QA generation and answering.
        """
        self._llm = llm_client
        self._config = get_app_config().citation_verification.numeric_qa_verification

    def parse_numeric_value(self, text: str) -> NumericValue | None:
        """Parse a numeric value from text.

        Extracts value, unit, multiplier, and entity from text like:
        - "$3.2 billion"
        - "25% growth"
        - "1,234,567 users"
        - "revenue of $5.2B"

        Args:
            text: Text containing a numeric value.

        Returns:
            NumericValue or None if no number found.
        """
        # Find numeric patterns
        patterns = [
            # Currency with multiplier: $3.2B, $1.5 billion
            r"[\$€£¥]?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*([TBMKtbmk]|trillion|billion|million|thousand)?",
            # Percentage: 25%, 3.5 percent
            r"(\d+(?:\.\d+)?)\s*(%|percent|percentage)",
            # Plain number with unit: 1,234,567 users
            r"(\d+(?:,\d{3})*(?:\.\d+)?)\s+(\w+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                raw_num = match.group(1).replace(",", "")
                multiplier_str = match.group(2) if len(match.groups()) > 1 else None

                try:
                    value = Decimal(raw_num)
                except InvalidOperation:
                    continue

                # Determine multiplier
                multiplier = 1
                if multiplier_str:
                    for pattern_re, mult in MULTIPLIER_PATTERNS.items():
                        if re.match(pattern_re, multiplier_str, re.IGNORECASE):
                            multiplier = mult
                            break

                # Apply multiplier
                normalized = value * multiplier

                # Detect unit
                unit = self._detect_unit(text)

                # Extract entity context
                entity = self._extract_entity(text)

                return NumericValue(
                    raw_text=match.group(0),
                    normalized_value=normalized,
                    unit=unit,
                    entity=entity,
                    multiplier=multiplier,
                )

        return None

    def _detect_unit(self, text: str) -> str | None:
        """Detect unit from text."""
        for pattern, unit in UNIT_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return unit
        return None

    def _extract_entity(self, text: str) -> str | None:
        """Extract entity reference from text."""
        # Look for common entity patterns
        entity_patterns = [
            r"(?:revenue|income|profit|sales|earnings)\s+(?:of|for|from)\s+([^,.\n]+)",
            r"([^,.\n]+?)(?:'s|')\s+(?:revenue|income|profit|sales)",
            r"(?:for|of|by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",  # Company names
        ]

        for pattern in entity_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    async def verify_numeric_claim(
        self,
        claim_text: str,
        evidence: RankedEvidence,
    ) -> NumericVerificationResult:
        """Verify a numeric claim against evidence using QA approach.

        Args:
            claim_text: The claim containing a numeric value.
            evidence: The evidence to verify against.

        Returns:
            NumericVerificationResult with QA results.
        """
        logger.debug(
            "NUMERIC_VERIFY_START",
            claim=truncate(claim_text, 50),
            evidence=truncate(evidence.quote_text, 50),
        )

        # Parse numeric value from claim
        parsed = self.parse_numeric_value(claim_text)
        if not parsed:
            return NumericVerificationResult(
                claim_text=claim_text,
                parsed_value=NumericValue(
                    raw_text=claim_text,
                    normalized_value=None,
                    unit=None,
                    entity=None,
                ),
                qa_results=[],
                overall_match=False,
                derivation_type="direct",
                confidence=0.0,
            )

        # Generate QA pairs using LLM
        qa_results = await self._run_qa_verification(claim_text, evidence, parsed)

        # Determine overall match
        if qa_results:
            match_count = sum(1 for r in qa_results if r.match)
            overall_match = match_count >= len(qa_results) * 0.5  # At least 50% match
            confidence = match_count / len(qa_results)
        else:
            # Fallback to simple comparison
            overall_match = self._simple_numeric_match(parsed, evidence.quote_text)
            confidence = 0.8 if overall_match else 0.2

        # Determine derivation type
        derivation_type = "direct"
        if any(op in claim_text.lower() for op in ["calculated", "computed", "derived", "estimated at"]):
            derivation_type = "computed"

        logger.debug(
            "NUMERIC_VERIFY_COMPLETE",
            overall_match=overall_match,
            confidence=round(confidence, 2),
            qa_count=len(qa_results),
        )

        return NumericVerificationResult(
            claim_text=claim_text,
            parsed_value=parsed,
            qa_results=qa_results,
            overall_match=overall_match,
            derivation_type=derivation_type,
            confidence=confidence,
        )

    async def _run_qa_verification(
        self,
        claim_text: str,
        evidence: RankedEvidence,
        parsed: NumericValue,
    ) -> list[QAVerificationResult]:
        """Run QA-based verification.

        Args:
            claim_text: The claim text.
            evidence: The evidence to verify against.
            parsed: Parsed numeric value from claim.

        Returns:
            List of QA verification results.
        """
        results: list[QAVerificationResult] = []

        prompt = NUMERIC_QA_PROMPT.format(
            claim_text=claim_text,
            evidence_quote=evidence.quote_text,
            raw_value=parsed.raw_text,
            unit=parsed.unit or "unknown",
            entity=parsed.entity or "unknown",
        )

        try:
            response = await self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                tier=ModelTier.BULK_ANALYSIS,  # Use Gemini for QA/comparison analysis
                structured_output=NumericQAOutput,
            )

            if response.structured:
                output: NumericQAOutput = response.structured
                for qa in output.qa_pairs:
                    match = self._compare_answers(qa.claim_answer, qa.evidence_answer)

                    results.append(
                        QAVerificationResult(
                            question=qa.question,
                            claim_answer=qa.claim_answer,
                            evidence_answer=qa.evidence_answer,
                            match=match,
                            normalized_comparison=self._normalize_for_comparison(
                                qa.claim_answer, qa.evidence_answer
                            ),
                        )
                    )

        except Exception as e:
            logger.warning(
                "NUMERIC_QA_ERROR",
                error=str(e)[:100],
            )

        return results

    def _compare_answers(self, claim_answer: str, evidence_answer: str) -> bool:
        """Compare two answers for match.

        Uses the configured comparison method:
        - exact_match: Exact string comparison
        - f1: Token-level F1 score
        - lerc: Learned evaluation (fallback to f1)
        """
        method = self._config.answer_comparison_method

        if method.value == "exact_match":
            return claim_answer.strip().lower() == evidence_answer.strip().lower()

        # F1 or LERC (fallback to F1)
        claim_tokens = set(re.findall(r"\w+", claim_answer.lower()))
        evidence_tokens = set(re.findall(r"\w+", evidence_answer.lower()))

        if not claim_tokens or not evidence_tokens:
            return False

        overlap = len(claim_tokens & evidence_tokens)
        precision = overlap / len(claim_tokens) if claim_tokens else 0
        recall = overlap / len(evidence_tokens) if evidence_tokens else 0

        if precision + recall == 0:
            return False

        f1 = 2 * precision * recall / (precision + recall)
        return f1 >= 0.5  # Threshold for match

    def _normalize_for_comparison(
        self,
        claim_answer: str,
        evidence_answer: str,
    ) -> dict[str, Any] | None:
        """Normalize numeric values for comparison."""
        claim_parsed = self.parse_numeric_value(claim_answer)
        evidence_parsed = self.parse_numeric_value(evidence_answer)

        if claim_parsed and evidence_parsed:
            claim_val = claim_parsed.normalized_value
            evidence_val = evidence_parsed.normalized_value

            if claim_val is not None and evidence_val is not None:
                return {
                    "claim_value": float(claim_val),
                    "evidence_value": float(evidence_val),
                    "difference": float(abs(claim_val - evidence_val)),
                    "match": self._values_match(claim_val, evidence_val),
                }

        return None

    def _values_match(self, a: Decimal, b: Decimal) -> bool:
        """Check if two values match within tolerance."""
        if a == 0 and b == 0:
            return True
        if a == 0 or b == 0:
            return False

        # Calculate relative difference
        diff = abs(a - b) / max(abs(a), abs(b))
        return float(diff) <= self._config.rounding_tolerance

    def _simple_numeric_match(self, parsed: NumericValue, evidence_text: str) -> bool:
        """Simple fallback check for numeric match."""
        if parsed.normalized_value is None:
            return False

        # Parse numeric value from evidence
        evidence_parsed = self.parse_numeric_value(evidence_text)
        if not evidence_parsed or evidence_parsed.normalized_value is None:
            return False

        # Check if values match
        if not self._values_match(parsed.normalized_value, evidence_parsed.normalized_value):
            return False

        # Optionally check units
        if self._config.require_unit_match:
            if parsed.unit and evidence_parsed.unit:
                if parsed.unit.lower() != evidence_parsed.unit.lower():
                    return False

        return True

    def detect_numeric_claims(self, text: str) -> list[str]:
        """Detect sentences containing numeric claims.

        Args:
            text: Full text to scan.

        Returns:
            List of sentences containing numeric content.
        """
        sentences = re.split(r"(?<=[.!?])\s+", text)
        numeric_sentences = []

        for sentence in sentences:
            if self.parse_numeric_value(sentence):
                numeric_sentences.append(sentence.strip())

        return numeric_sentences
