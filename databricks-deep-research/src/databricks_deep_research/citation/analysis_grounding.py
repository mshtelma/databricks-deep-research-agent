"""Analysis grounding verifier for reclaim-mode interpretation blocks."""

from __future__ import annotations

import logging
import re
from typing import Literal

from pydantic import BaseModel, Field

from databricks_deep_research.citation.config import GroundingValidationConfig
from databricks_deep_research.citation.types import (
    RankedEvidence,
    VerificationResult,
    VerificationVerdict,
)
from databricks_deep_research.llm.client import FrameworkLLMClient, ModelTier

logger = logging.getLogger(__name__)

_ANALYSIS_NUMERIC_RE = re.compile(
    r"[$€£¥]?\(?\d[\d,]*(?:\.\d+)?(?:\s*(?:%|million|billion|m|b|k|x))?\)?",
    re.IGNORECASE,
)
_ANALYSIS_TEMPORAL_RE = re.compile(
    r"\b(?:q[1-4]|20\d{2}|first quarter|second quarter|third quarter|fourth quarter|full[- ]year|year[- ]to[- ]date)\b",
    re.IGNORECASE,
)
_HEDGED_ANALYSIS_CUES = (
    "may indicate",
    "suggests",
    "appears consistent with",
    "appears to",
    "could reflect",
    "may reflect",
    "may suggest",
)
_STRONG_ANALYSIS_CUES = (
    "because",
    "due to",
    "driven by",
    "strongest",
    "weakest",
    "confirms",
    "proves",
    "clearly shows",
    "non-recurring",
    "should be viewed",
)


class AnalysisGroundingOutput(BaseModel):
    """Structured grounding verdict for an analysis sentence."""

    verdict: Literal["SUPPORTED", "PARTIAL", "UNSUPPORTED", "CONTRADICTED"]
    verification_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str = ""


_ANALYSIS_GROUNDING_PROMPT = """\
You are validating whether an analysis sentence is properly grounded in previously established facts.

## Analysis Sentence
{claim_text}

## Established Fact Claims
{supporting_claims}

## Supporting Evidence
{evidence_text}

## Task
Judge whether the analysis sentence is a reasonable interpretation of the established facts.

### SUPPORTED
- The sentence is a fair synthesis of the facts
- It does not introduce new factual payload
- Its scope and certainty match the evidence

### PARTIAL
- The sentence is directionally grounded
- But it overstates causality, certainty, ranking, or scope
- Or it adds light interpretation beyond what the facts strictly show

### UNSUPPORTED
- The sentence introduces new factual payload not present in the established facts
- Or it makes an inference that is not justified by the facts

### CONTRADICTED
- The sentence conflicts with the established facts

## Rules
- Analysis may interpret facts, but may not invent new metrics, dates, quarters, entities, rankings, or causes
- Prefer PARTIAL over UNSUPPORTED when the interpretation is plausible but overstated
- Base your judgment only on the supporting facts and evidence provided

## Response Format
```json
{{
  "verdict": "SUPPORTED" | "PARTIAL" | "UNSUPPORTED" | "CONTRADICTED",
  "verification_confidence": 0.0-1.0,
  "reasoning": "Short explanation"
}}
```
"""


class AnalysisGroundingVerifier:
    """Validate analysis sentences against already grounded facts."""

    def __init__(
        self,
        llm_client: FrameworkLLMClient,
        config: GroundingValidationConfig | None = None,
    ) -> None:
        self._llm = llm_client
        self._config = config or GroundingValidationConfig()

    async def verify_analysis_claim(
        self,
        claim_text: str,
        supporting_claims: list[str],
        evidences: list[RankedEvidence],
        supporting_fact_contexts: list[dict[str, str]] | None = None,
    ) -> VerificationResult:
        """Return a grounding verdict for an analysis sentence."""
        logger.debug(
            "ANALYSIS_GROUNDING_START claim=%s supporting_claims=%d evidences=%d fact_contexts=%d",
            claim_text[:120],
            len(supporting_claims),
            len(evidences),
            len(supporting_fact_contexts or []),
        )
        if not claim_text.strip():
            logger.debug("ANALYSIS_GROUNDING_EMPTY_CLAIM")
            return VerificationResult(
                verdict=VerificationVerdict.UNSUPPORTED,
                reasoning="Analysis sentence is empty.",
                confidence=0.0,
            )

        if not supporting_claims and not evidences:
            logger.debug("ANALYSIS_GROUNDING_NO_SUPPORT claim=%s", claim_text[:120])
            return VerificationResult(
                verdict=VerificationVerdict.UNSUPPORTED,
                reasoning="No grounded facts were available to support this analysis.",
                confidence=0.0,
            )

        supporting_fact_contexts = supporting_fact_contexts or []
        normalized_supporting_claims = supporting_claims[: self._config.max_preceding_citations]
        if supporting_fact_contexts:
            supporting_text = "\n".join(
                "- [{verdict}] {text}".format(
                    verdict=context.get("verdict", "supported") or "supported",
                    text=context.get("verification_text")
                    or context.get("claim_text")
                    or "",
                )
                for context in supporting_fact_contexts[: self._config.max_preceding_citations]
            ) or "- None provided"
            supporting_payloads = [
                context.get("verification_text") or context.get("claim_text") or ""
                for context in supporting_fact_contexts
            ]
        else:
            supporting_text = "\n".join(
                f"- {claim}" for claim in normalized_supporting_claims
            ) or "- None provided"
            supporting_payloads = normalized_supporting_claims

        if self._introduces_new_fact_payload(claim_text, supporting_payloads):
            logger.debug(
                "ANALYSIS_GROUNDING_NEW_PAYLOAD claim=%s supporting_payloads=%d",
                claim_text[:120],
                len(supporting_payloads),
            )
            return VerificationResult(
                verdict=VerificationVerdict.UNSUPPORTED,
                reasoning=(
                    "The analysis sentence introduces new numeric or temporal payload that "
                    "is not established by the verified fact claims."
                ),
                confidence=0.82,
            )

        lowered_claim = claim_text.lower()
        if self._contains_hedged_analysis(lowered_claim):
            logger.debug("ANALYSIS_GROUNDING_HEURISTIC_SUPPORTED claim=%s", claim_text[:120])
            return VerificationResult(
                verdict=VerificationVerdict.SUPPORTED,
                reasoning=(
                    "The sentence is a bounded interpretation of previously verified facts "
                    "and does not add fresh factual payload."
                ),
                confidence=0.72,
            )

        if self._contains_strong_analysis(lowered_claim):
            logger.debug("ANALYSIS_GROUNDING_HEURISTIC_PARTIAL claim=%s", claim_text[:120])
            return VerificationResult(
                verdict=VerificationVerdict.PARTIAL,
                reasoning=(
                    "The sentence is directionally grounded in verified facts but overstates "
                    "causality, certainty, or business interpretation."
                ),
                confidence=0.68,
            )

        evidence_text = "\n\n".join(
            f"[{index + 1}] {evidence.quote_text[:1200]}"
            for index, evidence in enumerate(evidences[: self._config.max_preceding_citations])
        ) or "[1] No supporting evidence provided"

        prompt = _ANALYSIS_GROUNDING_PROMPT.format(
            claim_text=claim_text,
            supporting_claims=supporting_text,
            evidence_text=evidence_text,
        )
        logger.debug(
            "ANALYSIS_GROUNDING_LLM claim=%s supporting_lines=%d evidence_chars=%d",
            claim_text[:120],
            supporting_text.count("\n") + 1,
            len(evidence_text),
        )

        try:
            response = await self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                tier=ModelTier.analytical,
                structured_output=AnalysisGroundingOutput,
            )
            output = response.structured
            if output is None:
                raise ValueError("analysis grounding response missing structured payload")
            logger.debug(
                "ANALYSIS_GROUNDING_RESULT verdict=%s confidence=%.2f claim=%s",
                output.verdict,
                output.verification_confidence,
                claim_text[:120],
            )
            return VerificationResult(
                verdict=VerificationVerdict(output.verdict.lower()),
                reasoning=output.reasoning,
                confidence=output.verification_confidence,
            )
        except Exception:
            logger.warning(
                "ANALYSIS_GROUNDING_FAILED claim=%s supporting_claims=%d evidences=%d",
                claim_text[:120],
                len(supporting_claims),
                len(evidences),
                exc_info=True,
            )
            fallback_verdict = (
                VerificationVerdict.PARTIAL if supporting_claims else VerificationVerdict.UNSUPPORTED
            )
            fallback_reasoning = (
                "Grounding model failed; treating analysis as partially grounded because it "
                "references previously established facts."
                if supporting_claims
                else "Grounding model failed and no supporting facts were available."
            )
            return VerificationResult(
                verdict=fallback_verdict,
                reasoning=fallback_reasoning,
                confidence=0.35 if supporting_claims else 0.0,
            )

    @staticmethod
    def _contains_hedged_analysis(claim_text: str) -> bool:
        return any(cue in claim_text for cue in _HEDGED_ANALYSIS_CUES)

    @staticmethod
    def _contains_strong_analysis(claim_text: str) -> bool:
        return any(cue in claim_text for cue in _STRONG_ANALYSIS_CUES)

    @staticmethod
    def _introduces_new_fact_payload(
        claim_text: str,
        supporting_claims: list[str],
    ) -> bool:
        supporting_text = " ".join(supporting_claims).lower()
        claim_numeric = {
            re.sub(r"\s+", "", token).lower().replace(",", "")
            for token in _ANALYSIS_NUMERIC_RE.findall(claim_text)
        }
        claim_temporal = {
            token.strip().lower()
            for token in _ANALYSIS_TEMPORAL_RE.findall(claim_text)
        }
        if claim_numeric:
            supporting_numeric = {
                re.sub(r"\s+", "", token).lower().replace(",", "")
                for token in _ANALYSIS_NUMERIC_RE.findall(supporting_text)
            }
            if not claim_numeric.issubset(supporting_numeric):
                return True
        if claim_temporal:
            supporting_temporal = {
                token.strip().lower()
                for token in _ANALYSIS_TEMPORAL_RE.findall(supporting_text)
            }
            if not claim_temporal.issubset(supporting_temporal):
                return True
        return False
