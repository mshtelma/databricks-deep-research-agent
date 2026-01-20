"""Stage 4: Isolated Verification service.

Verifies claims against evidence IN ISOLATION (no generation context)
to prevent bias propagation using the CoVe pattern.
"""

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from deep_research.agent.prompts.citation.verification import (
    ISOLATED_VERIFICATION_PROMPT,
    QUICK_VERIFICATION_PROMPT,
)
from deep_research.core.app_config import get_app_config
from deep_research.services.citation.evidence_selector import RankedEvidence
from deep_research.services.llm.client import LLMClient
from deep_research.services.llm.types import ModelTier

logger = logging.getLogger(__name__)


# Pydantic models for structured LLM output
class VerificationOutput(BaseModel):
    """Output from isolated verification LLM call."""

    verdict: Literal["SUPPORTED", "PARTIAL", "UNSUPPORTED", "CONTRADICTED"] = Field(
        description="Verification verdict"
    )
    reasoning: str = Field(
        default="",
        description="Detailed explanation of why this verdict was chosen"
    )
    key_match: str | None = Field(
        default=None,
        description="Specific part of evidence that supports/contradicts"
    )
    issues: list[str] | None = Field(
        default=None,
        description="List of specific issues found"
    )


class Verdict(str, Enum):
    """Four-tier verification verdict."""

    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"


@dataclass
class VerificationResult:
    """Result of claim verification."""

    verdict: Verdict
    reasoning: str
    key_match: str | None = None
    issues: list[str] | None = None
    abstained: bool = False


class IsolatedVerifier:
    """Stage 4: Isolated Verification.

    Verifies claims against evidence IN ISOLATION using the CoVe
    (Chain of Verification) pattern to prevent bias propagation.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize isolated verifier.

        Args:
            llm_client: LLM client for verification.
        """
        self._llm = llm_client
        self._config = get_app_config().citation_verification.isolated_verification

    async def verify_with_isolation(
        self,
        claim_text: str,
        evidence: RankedEvidence,
        use_quick_verification: bool = False,
    ) -> VerificationResult:
        """Verify a claim against evidence in isolation.

        CRITICAL: This method receives NO generation context to prevent
        the LLM from "remembering" what it generated and confirming bias.

        Args:
            claim_text: The claim to verify.
            evidence: The supporting evidence span.
            use_quick_verification: Use fast verification for high-confidence claims.

        Returns:
            VerificationResult with verdict and reasoning.
        """
        if use_quick_verification:
            return await self._quick_verify(claim_text, evidence)

        return await self._full_verify(claim_text, evidence)

    async def _full_verify(
        self,
        claim_text: str,
        evidence: RankedEvidence,
    ) -> VerificationResult:
        """Full verification with detailed reasoning.

        Args:
            claim_text: The claim to verify.
            evidence: The supporting evidence span.

        Returns:
            VerificationResult with detailed reasoning.
        """
        prompt = ISOLATED_VERIFICATION_PROMPT.format(
            claim_text=claim_text,
            source_title=evidence.source_title or "Unknown",
            source_url=evidence.source_url,
            evidence_quote=evidence.quote_text,
        )

        tier_str = self._config.verification_model_tier
        tier = ModelTier(tier_str) if tier_str else ModelTier.ANALYTICAL

        try:
            response = await self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                tier=tier,
                structured_output=VerificationOutput,
            )

            if response.structured:
                output: VerificationOutput = response.structured
                verdict = self.parse_verdict(output.verdict)
                return VerificationResult(
                    verdict=verdict,
                    reasoning=output.reasoning,
                    key_match=output.key_match,
                    issues=output.issues,
                )

            # Fallback if structured output unavailable
            return self._parse_verification_response(response.content)

        except Exception as e:
            logger.error(f"Full verification failed: {e}")
            return VerificationResult(
                verdict=Verdict.UNSUPPORTED,
                reasoning=f"Verification failed: {e}",
                abstained=True,
            )

    async def _quick_verify(
        self,
        claim_text: str,
        evidence: RankedEvidence,
    ) -> VerificationResult:
        """Quick verification for high-confidence claims.

        Args:
            claim_text: The claim to verify.
            evidence: The supporting evidence span.

        Returns:
            VerificationResult with basic verdict.
        """
        prompt = QUICK_VERIFICATION_PROMPT.format(
            claim_text=claim_text,
            evidence_quote=evidence.quote_text,
        )

        tier_str = self._config.quick_verification_tier
        tier = ModelTier(tier_str) if tier_str else ModelTier.SIMPLE

        try:
            response = await self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                tier=tier,
            )

            verdict = self.parse_verdict(response.content.strip().upper())

            return VerificationResult(
                verdict=verdict,
                reasoning="Quick verification",
            )

        except Exception as e:
            logger.error(f"Quick verification failed: {e}")
            return VerificationResult(
                verdict=Verdict.UNSUPPORTED,
                reasoning=f"Quick verification failed: {e}",
                abstained=True,
            )

    def parse_verdict(self, verdict_text: str) -> Verdict:
        """Parse verdict from LLM response.

        Args:
            verdict_text: Raw verdict text from LLM.

        Returns:
            Parsed Verdict enum value.
        """
        verdict_text = verdict_text.upper().strip()

        if "SUPPORTED" in verdict_text and "UNSUPPORTED" not in verdict_text:
            return Verdict.SUPPORTED
        elif "PARTIAL" in verdict_text:
            return Verdict.PARTIAL
        elif "CONTRADICTED" in verdict_text:
            return Verdict.CONTRADICTED
        else:
            return Verdict.UNSUPPORTED

    def _parse_verification_response(self, response: str) -> VerificationResult:
        """Parse full verification response from LLM.

        Args:
            response: LLM response text.

        Returns:
            Parsed VerificationResult.
        """
        try:
            # Try to extract JSON
            json_match = re.search(r"\{[\s\S]*?\}", response)
            if json_match:
                data = json.loads(json_match.group())
                verdict = self.parse_verdict(data.get("verdict", "UNSUPPORTED"))

                return VerificationResult(
                    verdict=verdict,
                    reasoning=data.get("reasoning", ""),
                    key_match=data.get("key_match"),
                    issues=data.get("issues"),
                )
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Failed to parse verification JSON: {e}")

        # Fallback: try to find verdict in text
        verdict = self.parse_verdict(response)
        return VerificationResult(
            verdict=verdict,
            reasoning=response[:500],  # Use first part of response as reasoning
        )

    async def verify_batch(
        self,
        claims: list[tuple[str, RankedEvidence]],
        confidence_levels: list[str] | None = None,
    ) -> list[VerificationResult]:
        """Verify multiple claims in batch.

        Args:
            claims: List of (claim_text, evidence) tuples.
            confidence_levels: Optional list of confidence levels for routing.

        Returns:
            List of VerificationResult objects.
        """
        results = []

        for i, (claim_text, evidence) in enumerate(claims):
            # Determine if quick verification is appropriate
            use_quick = False
            if confidence_levels and i < len(confidence_levels):
                use_quick = confidence_levels[i] == "high"

            result = await self.verify_with_isolation(
                claim_text=claim_text,
                evidence=evidence,
                use_quick_verification=use_quick,
            )
            results.append(result)

        return results

    def check_nei(
        self,
        claim_text: str,
        evidence: RankedEvidence,
    ) -> bool:
        """Check if Not Enough Information (NEI) verdict applies.

        Quick heuristic check before full verification.

        Args:
            claim_text: The claim to check.
            evidence: The evidence span.

        Returns:
            True if NEI likely applies.
        """
        if not self._config.enable_nei_verdict:
            return False

        # Check for minimal overlap between claim and evidence
        claim_words = set(re.findall(r"\b\w{4,}\b", claim_text.lower()))
        evidence_words = set(re.findall(r"\b\w{4,}\b", evidence.quote_text.lower()))

        if not claim_words:
            return True

        overlap = len(claim_words & evidence_words) / len(claim_words)

        # If less than 20% word overlap, likely NEI
        return overlap < 0.2
