"""Stage 6: Numeric QA Verification.

Verifies numeric claims using a QA-based approach (QAFactEval pattern):
1. Generate questions about numeric values
2. Answer questions from both claim and evidence separately
3. Compare answers to detect mismatches

Token Optimization Features:
- Exact match heuristic: Skip QA when numeric values appear exactly in evidence

Ported from the app's citation verification pipeline.
Uses ``FrameworkLLMClient`` instead of app LLMClient.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from databricks_deep_research.citation.types import (
    NumericValue,
    NumericVerificationResult,
    QAVerificationResult,
    RankedEvidence,
)
from databricks_deep_research.llm.client import FrameworkLLMClient, ModelTier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class AnswerComparisonMethod(StrEnum):
    """Method for comparing answers in numeric QA verification."""

    EXACT_MATCH = "exact_match"
    F1 = "f1"
    LERC = "lerc"


@dataclass(frozen=True)
class NumericVerifierConfig:
    """Configuration for numeric QA verification.

    Replaces the app's ``NumericQAVerificationConfig`` Pydantic model.
    """

    rounding_tolerance: float = 0.05
    answer_comparison_method: AnswerComparisonMethod = AnswerComparisonMethod.F1
    require_unit_match: bool = True
    require_entity_match: bool = True


_DEFAULT_CONFIG = NumericVerifierConfig()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truncate(text: str | None, max_length: int = 100) -> str:
    """Truncate text for logging."""
    if text is None:
        return "<none>"
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def is_exact_numeric_match(claim: str, evidence: str) -> bool:
    """Check if numeric values in claim appear exactly in evidence.

    This is a TOKEN OPTIMIZATION heuristic that skips full QA verification
    when simple numeric values can be trivially matched.

    Args:
        claim: The claim text containing numeric value(s).
        evidence: The evidence text to check against.

    Returns:
        True if ALL numeric values from claim appear in evidence.
    """
    # Extract numbers from claim (including percentages, decimals)
    number_pattern = r"\d+(?:,\d{3})*(?:\.\d+)?%?"
    claim_numbers = set(re.findall(number_pattern, claim))

    if not claim_numbers:
        return False

    # Check if ALL numbers appear in evidence
    matches = 0
    for num in claim_numbers:
        # Check direct match
        if num in evidence or num.replace(",", "") in evidence.replace(",", ""):
            matches += 1

    # All numbers must match
    return matches == len(claim_numbers)


# ---------------------------------------------------------------------------
# Pydantic output models for structured LLM call
# ---------------------------------------------------------------------------


class QAPairOutput(BaseModel):
    """A single QA pair for numeric verification."""

    question: str = Field(description="Question about the numeric value")
    claim_answer: str = Field(description="Answer based on the claim only")
    evidence_answer: str = Field(description="Answer based on the evidence only")


class NumericQAOutput(BaseModel):
    """Output from numeric QA verification LLM call."""

    qa_pairs: list[QAPairOutput] = Field(
        default_factory=list,
        description="List of QA pairs for verification",
    )


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

NUMERIC_QA_PROMPT = """Generate QA pairs to verify a numeric claim.

## Claim with Numeric Value
"{claim_text}"

## Evidence Quote
"{evidence_quote}"

## Numeric Value Details
- Raw value: {raw_value}
- Unit: {unit}
- Entity: {entity}

## Task
Generate 2-3 fact-checking questions about this numeric value.
For each question:
1. Ask a specific question about the numeric value
2. Answer from the CLAIM ONLY
3. Answer from the EVIDENCE ONLY
4. Compare answers to check if they match

## Response Format (JSON array)
```json
[
  {{
    "question": "What was the revenue figure mentioned?",
    "claim_answer": "The claim states $3.2 billion",
    "evidence_answer": "The evidence mentions $3.2B revenue"
  }},
  {{
    "question": "What entity does this value refer to?",
    "claim_answer": "Company X's Q4 2024 revenue",
    "evidence_answer": "Company X fourth quarter revenue"
  }}
]
```

Generate QA pairs for verification:"""


# ---------------------------------------------------------------------------
# Multiplier / unit tables
# ---------------------------------------------------------------------------

MULTIPLIER_PATTERNS: dict[str, int] = {
    r"\btrillion\b": 1_000_000_000_000,
    r"\bbillion\b": 1_000_000_000,
    r"\bmillion\b": 1_000_000,
    r"\bthousand\b": 1_000,
    r"[Tt]": 1_000_000_000_000,  # $1T
    r"[Bb]": 1_000_000_000,  # $1B
    r"[Mm]": 1_000_000,  # $1M
    r"[Kk]": 1_000,  # $1K
}

UNIT_PATTERNS: list[tuple[str, str]] = [
    (r"\$", "USD"),
    (r"\u20ac", "EUR"),
    (r"\u00a3", "GBP"),
    (r"\u00a5", "JPY"),
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


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class NumericVerifier:
    """Stage 6: Numeric QA Verification.

    Verifies numeric claims using QA-based comparison (QAFactEval pattern).
    This catches semantic errors that simple text matching would miss.
    """

    def __init__(
        self,
        llm_client: FrameworkLLMClient,
        config: NumericVerifierConfig | None = None,
    ) -> None:
        """Initialize the numeric verifier.

        Args:
            llm_client: Framework LLM client for QA generation and answering.
            config: Optional configuration. Uses defaults if *None*.
        """
        self._llm = llm_client
        self._config = config or _DEFAULT_CONFIG

    # -- Parsing ------------------------------------------------------------

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
        candidates = self.extract_numeric_values(text)
        return candidates[0] if candidates else None

    def extract_numeric_values(self, text: str) -> list[NumericValue]:
        """Extract and rank all salient numeric values from text."""
        candidates: list[tuple[int, int, NumericValue]] = []
        patterns = [
            (
                "currency",
                re.compile(
                    r"(?P<raw>(?P<currency>[\$\u20ac\u00a3\u00a5])\s*\(?\s*"
                    r"(?P<number>\d+(?:,\d{3})*(?:\.\d+)?)\s*\)?\s*"
                    r"(?P<multiplier>trillion|billion|million|thousand|[TBMKtbmk](?![a-zA-Z]))?)",
                    re.IGNORECASE,
                ),
            ),
            (
                "scaled",
                re.compile(
                    r"(?P<raw>\(?\s*(?P<number>\d+(?:,\d{3})*(?:\.\d+)?)\s*\)?\s*"
                    r"(?P<multiplier>trillion|billion|million|thousand)\b)",
                    re.IGNORECASE,
                ),
            ),
            (
                "percent",
                re.compile(
                    r"(?P<raw>(?P<number>\d+(?:\.\d+)?)\s*(?P<unit>%|percent|percentage))",
                    re.IGNORECASE,
                ),
            ),
            (
                "unit",
                re.compile(
                    r"(?P<raw>(?P<number>\d+(?:,\d{3})*(?:\.\d+)?)\s+"
                    r"(?P<unit>users?|customers?|employees?|years?|months?|days?|hours?))",
                    re.IGNORECASE,
                ),
            ),
        ]

        for kind, pattern in patterns:
            for match in pattern.finditer(text):
                parsed = self._build_numeric_value(text, kind, match)
                if parsed is None:
                    continue
                score = self._score_numeric_candidate(text, match, kind, parsed)
                candidates.append((score, match.start(), parsed))

        candidates.sort(key=lambda item: (-item[0], item[1]))
        return [candidate for _, _, candidate in candidates]

    def _build_numeric_value(
        self,
        text: str,
        kind: str,
        match: re.Match[str],
    ) -> NumericValue | None:
        """Convert a regex match into a normalized numeric value."""
        raw_text = match.group("raw").strip()
        raw_num = match.group("number").replace(",", "")
        unit_token = match.groupdict().get("unit")
        multiplier_str = match.groupdict().get("multiplier")
        currency_symbol = match.groupdict().get("currency")

        if kind == "unit" and unit_token:
            normalized_unit = unit_token.lower()
            if normalized_unit in {"q1", "q2", "q3", "q4", "fy"}:
                return None

        try:
            value = Decimal(raw_num)
        except InvalidOperation:
            return None

        multiplier = 1
        if multiplier_str:
            for pattern_re, mult in MULTIPLIER_PATTERNS.items():
                if re.fullmatch(pattern_re, multiplier_str, re.IGNORECASE):
                    multiplier = mult
                    break

        if "(" in raw_text or raw_text.strip().startswith("-"):
            value *= -1

        unit = self._detect_match_unit(currency_symbol, unit_token)
        normalized = value * multiplier
        entity = self._extract_entity(text)
        return NumericValue(
            raw_text=raw_text,
            normalized_value=normalized,
            unit=unit,
            entity=entity,
            multiplier=multiplier,
        )

    def _score_numeric_candidate(
        self,
        text: str,
        match: re.Match[str],
        kind: str,
        parsed: NumericValue,
    ) -> int:
        """Rank numeric candidates so salient finance metrics beat quarter/year noise."""
        score = 0
        if kind == "currency":
            score += 6
        elif kind == "percent":
            score += 5
        else:
            score += 2

        raw_lower = parsed.raw_text.lower()
        window_start = max(0, match.start() - 40)
        window_end = min(len(text), match.end() + 40)
        context = text[window_start:window_end].lower()

        metric_terms = (
            "eps",
            "earnings",
            "profit",
            "loss",
            "sales",
            "revenue",
            "growth",
            "margin",
            "operating",
            "adjusted",
            "digital",
            "ecommerce",
            "commerce",
        )
        if any(term in context for term in metric_terms):
            score += 4

        if parsed.multiplier > 1:
            score += 2

        normalized_value = parsed.normalized_value
        if (
            normalized_value is not None
            and parsed.unit is None
            and kind != "currency"
            and normalized_value == int(normalized_value)
            and 1900 <= int(normalized_value) <= 2100
        ):
            score -= 6

        prefix = text[max(0, match.start() - 2):match.start()].lower()
        if prefix.endswith("q") or "quarter" in context:
            score -= 7

        if raw_lower in {"1", "2", "3", "4"} and ("q1" in context or "q2" in context or "q3" in context or "q4" in context):
            score -= 8

        return score

    def _detect_match_unit(
        self,
        currency_symbol: str | None,
        unit_token: str | None,
    ) -> str | None:
        """Detect unit from the matched token instead of the whole sentence."""
        if currency_symbol:
            return {
                "$": "USD",
                "\u20ac": "EUR",
                "\u00a3": "GBP",
                "\u00a5": "JPY",
            }.get(currency_symbol)
        if unit_token:
            normalized = unit_token.lower()
            if normalized in {"%", "percent", "percentage"}:
                return "percent"
            for pattern, unit in UNIT_PATTERNS:
                if re.fullmatch(pattern, normalized, re.IGNORECASE):
                    return unit
        return None

    def _detect_unit(self, text: str) -> str | None:
        """Detect unit from text."""
        for pattern, unit in UNIT_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return unit
        return None

    def _extract_entity(self, text: str) -> str | None:
        """Extract entity reference from text."""
        entity_patterns = [
            r"(?:revenue|income|profit|sales|earnings)"
            r"\s+(?:of|for|from)\s+([^,.\n]+)",
            r"([^,.\n]+?)(?:'s|')\s+(?:revenue|income|profit|sales)",
            r"(?:for|of|by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        ]

        for pattern in entity_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    # -- Verification -------------------------------------------------------

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
            "NUMERIC_VERIFY_START claim=%s evidence=%s",
            _truncate(claim_text, 50),
            _truncate(evidence.quote_text, 50),
        )

        # TOKEN OPTIMIZATION: Fast path for exact numeric match
        if is_exact_numeric_match(claim_text, evidence.quote_text):
            logger.debug(
                "NUMERIC_VERIFY_EXACT_MATCH claim=%s",
                _truncate(claim_text, 50),
            )
            parsed = self.parse_numeric_value(claim_text)
            return NumericVerificationResult(
                claim_text=claim_text,
                parsed_value=parsed or NumericValue(
                    raw_text=claim_text,
                    normalized_value=None,
                    unit=None,
                    entity=None,
                ),
                qa_results=[],
                overall_match=True,
                derivation_type="direct",
                confidence=0.95,  # High confidence for exact match
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
            overall_match = match_count >= len(qa_results) * 0.5
            confidence = match_count / len(qa_results)
        else:
            # Fallback to simple comparison
            overall_match = self._simple_numeric_match(parsed, evidence.quote_text)
            confidence = 0.8 if overall_match else 0.2

        # Determine derivation type
        derivation_type = "direct"
        if any(
            op in claim_text.lower()
            for op in ["calculated", "computed", "derived", "estimated at"]
        ):
            derivation_type = "computed"

        logger.debug(
            "NUMERIC_VERIFY_COMPLETE overall_match=%s confidence=%.2f qa_count=%d",
            overall_match,
            confidence,
            len(qa_results),
        )

        return NumericVerificationResult(
            claim_text=claim_text,
            parsed_value=parsed,
            qa_results=qa_results,
            overall_match=overall_match,
            derivation_type=derivation_type,
            confidence=confidence,
        )

    # -- QA generation and comparison ---------------------------------------

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
                tier=ModelTier.simple,
                structured_output=NumericQAOutput,
            )

            if response.structured:
                output: NumericQAOutput = response.structured
                for qa in output.qa_pairs:
                    match = self._compare_answers(
                        qa.claim_answer, qa.evidence_answer
                    )

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
                "NUMERIC_QA_ERROR error=%s",
                str(e)[:100],
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

        if method == AnswerComparisonMethod.EXACT_MATCH:
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
        if not self._values_match(
            parsed.normalized_value, evidence_parsed.normalized_value
        ):
            return False

        # Optionally check units
        return not (
            self._config.require_unit_match
            and parsed.unit
            and evidence_parsed.unit
            and parsed.unit.lower() != evidence_parsed.unit.lower()
        )

    # -- Detection ----------------------------------------------------------

    def detect_numeric_claims(self, text: str) -> list[str]:
        """Detect sentences containing numeric claims.

        Args:
            text: Full text to scan.

        Returns:
            List of sentences containing numeric content.
        """
        sentences = re.split(r"(?<=[.!?])\s+", text)
        numeric_sentences: list[str] = []

        for sentence in sentences:
            if self.parse_numeric_value(sentence):
                numeric_sentences.append(sentence.strip())

        return numeric_sentences
