"""Stage 4: Isolated Verification service.

Verifies claims against evidence IN ISOLATION (no generation context)
to prevent bias propagation using the CoVe pattern.

Token Optimization Features:
- Batch verification: Process multiple claims in single LLM call (5 claims per batch)
- Verification caching: Cache results by normalized claim fingerprint
- Model tier escalation: Start with simple tier, escalate only on uncertainty
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from deep_research.agent.prompts.citation.verification import (
    BATCH_QUICK_VERIFICATION_PROMPT,
    BATCH_VERIFICATION_PROMPT,
    ISOLATED_VERIFICATION_PROMPT,
    QUICK_VERIFICATION_PROMPT,
)
from deep_research.core.app_config import get_app_config
from deep_research.services.citation.evidence_selector import RankedEvidence
from deep_research.services.llm.client import LLMClient
from deep_research.services.llm.types import ModelTier

logger = logging.getLogger(__name__)


# Default batch size for verification (balances token efficiency vs. reliability)
DEFAULT_BATCH_SIZE = 5


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


# Batch verification models (Token Optimization)
class BatchVerificationItem(BaseModel):
    """Single claim verification result in a batch."""

    claim_index: int = Field(description="0-based index of claim in input batch")
    verdict: Literal["SUPPORTED", "PARTIAL", "UNSUPPORTED", "CONTRADICTED"] = Field(
        description="Verification verdict"
    )
    reasoning: str = Field(default="", max_length=500)
    key_match: str | None = Field(default=None, max_length=200)


class BatchVerificationOutput(BaseModel):
    """Output for batched verification."""

    results: list[BatchVerificationItem] = Field(
        description="Verification results in same order as input claims"
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

    # =========================================================================
    # Token Optimization: Batch Verification
    # =========================================================================

    @staticmethod
    def fingerprint_claim(claim_text: str) -> str:
        """Create normalized fingerprint for claim caching.

        Used to identify duplicate or near-duplicate claims to avoid
        redundant verification.

        Args:
            claim_text: The claim text to fingerprint.

        Returns:
            16-character MD5 hash of normalized claim.
        """
        # Normalize: lowercase, remove punctuation, sort words
        normalized = re.sub(r"[^\w\s]", "", claim_text.lower())
        words = sorted(normalized.split())
        return hashlib.md5(" ".join(words).encode()).hexdigest()[:16]

    def _format_claims_for_batch(
        self,
        claims: list[tuple[str, RankedEvidence]],
    ) -> str:
        """Format claims for batch verification prompt.

        Args:
            claims: List of (claim_text, evidence) tuples.

        Returns:
            Formatted string for the batch prompt.
        """
        sections = []
        for i, (claim_text, evidence) in enumerate(claims):
            section = f"""### Claim {i}
**Claim:** "{claim_text}"
**Source:** {evidence.source_title or 'Unknown'}
**Evidence:** "{evidence.quote_text[:500]}"
"""
            sections.append(section)
        return "\n".join(sections)

    async def verify_batch_grouped(
        self,
        claims: list[tuple[str, RankedEvidence]],
        batch_size: int = DEFAULT_BATCH_SIZE,
        use_quick_verification: bool = False,
        verification_cache: dict[str, "VerificationResult"] | None = None,
    ) -> list[VerificationResult]:
        """Verify multiple claims using batched LLM calls.

        This is a TOKEN OPTIMIZATION method that processes multiple claims
        in a single LLM call, reducing overhead significantly.

        Args:
            claims: List of (claim_text, evidence) tuples to verify.
            batch_size: Number of claims per batch (default: 5).
            use_quick_verification: Use faster, simpler verification.
            verification_cache: Optional cache dict for result reuse.

        Returns:
            List of VerificationResult objects in same order as input.
        """
        if not claims:
            return []

        results: list[VerificationResult | None] = [None] * len(claims)
        uncached_indices: list[int] = []

        # Phase 1: Check cache for existing results
        if verification_cache:
            for i, (claim_text, _) in enumerate(claims):
                fingerprint = self.fingerprint_claim(claim_text)
                if fingerprint in verification_cache:
                    results[i] = verification_cache[fingerprint]
                    logger.debug(
                        "VERIFICATION_CACHE_HIT",
                        claim_index=i,
                        fingerprint=fingerprint,
                    )
                else:
                    uncached_indices.append(i)
        else:
            uncached_indices = list(range(len(claims)))

        if not uncached_indices:
            # All results were cached
            return [r for r in results if r is not None]

        # Phase 2: Group uncached claims into batches
        batches: list[list[int]] = []
        for i in range(0, len(uncached_indices), batch_size):
            batches.append(uncached_indices[i : i + batch_size])

        logger.info(
            "BATCH_VERIFICATION_START",
            total_claims=len(claims),
            cached=len(claims) - len(uncached_indices),
            uncached=len(uncached_indices),
            batches=len(batches),
        )

        # Phase 3: Process each batch
        for batch_num, batch_indices in enumerate(batches):
            batch_claims = [claims[i] for i in batch_indices]

            try:
                batch_results = await self._process_batch(
                    batch_claims, use_quick_verification
                )

                # Map results back to original indices
                for j, idx in enumerate(batch_indices):
                    if j < len(batch_results):
                        results[idx] = batch_results[j]
                        # Cache the result
                        if verification_cache:
                            fingerprint = self.fingerprint_claim(claims[idx][0])
                            verification_cache[fingerprint] = batch_results[j]
                    else:
                        # Fallback if batch returned fewer results
                        results[idx] = VerificationResult(
                            verdict=Verdict.UNSUPPORTED,
                            reasoning="Batch verification returned no result",
                            abstained=True,
                        )

            except Exception as e:
                logger.warning(
                    "BATCH_VERIFICATION_ERROR",
                    batch_num=batch_num,
                    error=str(e)[:100],
                )
                # Fall back to sequential verification for this batch
                for idx in batch_indices:
                    claim_text, evidence = claims[idx]
                    results[idx] = await self.verify_with_isolation(
                        claim_text=claim_text,
                        evidence=evidence,
                        use_quick_verification=use_quick_verification,
                    )

        # Fill any remaining None values with abstained results
        for i, result in enumerate(results):
            if result is None:
                results[i] = VerificationResult(
                    verdict=Verdict.UNSUPPORTED,
                    reasoning="Verification incomplete",
                    abstained=True,
                )

        logger.info(
            "BATCH_VERIFICATION_COMPLETE",
            total_claims=len(claims),
            results=len([r for r in results if r is not None]),
        )

        return [r for r in results if r is not None]

    async def _process_batch(
        self,
        claims: list[tuple[str, RankedEvidence]],
        use_quick_verification: bool = False,
    ) -> list[VerificationResult]:
        """Process a single batch of claims.

        Args:
            claims: List of (claim_text, evidence) tuples in this batch.
            use_quick_verification: Use faster verification.

        Returns:
            List of VerificationResult objects.
        """
        if not claims:
            return []

        # Format claims for batch prompt
        claims_section = self._format_claims_for_batch(claims)

        # Select prompt and tier based on verification mode
        if use_quick_verification:
            prompt = BATCH_QUICK_VERIFICATION_PROMPT.format(
                claims_section=claims_section
            )
            tier_str = self._config.quick_verification_tier
            tier = ModelTier(tier_str) if tier_str else ModelTier.SIMPLE
        else:
            prompt = BATCH_VERIFICATION_PROMPT.format(claims_section=claims_section)
            tier_str = self._config.verification_model_tier
            tier = ModelTier(tier_str) if tier_str else ModelTier.ANALYTICAL

        try:
            response = await self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                tier=tier,
                structured_output=BatchVerificationOutput,
            )

            if response.structured:
                output: BatchVerificationOutput = response.structured
                return self._parse_batch_results(output, len(claims))

            # Fallback: try to parse from content
            return self._parse_batch_response_content(response.content, len(claims))

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise

    def _parse_batch_results(
        self,
        output: BatchVerificationOutput,
        expected_count: int,
    ) -> list[VerificationResult]:
        """Parse batch verification output into results list.

        Handles potential reordering by using claim_index from output.

        Args:
            output: Structured batch output from LLM.
            expected_count: Expected number of results.

        Returns:
            List of VerificationResult objects in original order.
        """
        # Initialize results with abstained defaults
        results: list[VerificationResult] = [
            VerificationResult(
                verdict=Verdict.UNSUPPORTED,
                reasoning="No result in batch output",
                abstained=True,
            )
            for _ in range(expected_count)
        ]

        # Map results by claim_index to handle any reordering
        for item in output.results:
            if 0 <= item.claim_index < expected_count:
                verdict = self.parse_verdict(item.verdict)
                results[item.claim_index] = VerificationResult(
                    verdict=verdict,
                    reasoning=item.reasoning,
                    key_match=item.key_match,
                )
            else:
                logger.warning(
                    "BATCH_RESULT_INDEX_OUT_OF_RANGE",
                    claim_index=item.claim_index,
                    expected_count=expected_count,
                )

        return results

    def _parse_batch_response_content(
        self,
        content: str,
        expected_count: int,
    ) -> list[VerificationResult]:
        """Fallback parser for batch response when structured output fails.

        Args:
            content: Raw LLM response content.
            expected_count: Expected number of results.

        Returns:
            List of VerificationResult objects.
        """
        results: list[VerificationResult] = []

        try:
            # Try to extract JSON from response
            json_match = re.search(r"\{[\s\S]*\}", content)
            if json_match:
                data = json.loads(json_match.group())
                if "results" in data and isinstance(data["results"], list):
                    # Initialize with defaults
                    results = [
                        VerificationResult(
                            verdict=Verdict.UNSUPPORTED,
                            reasoning="No result in batch output",
                            abstained=True,
                        )
                        for _ in range(expected_count)
                    ]

                    for item in data["results"]:
                        idx = item.get("claim_index", -1)
                        if 0 <= idx < expected_count:
                            verdict = self.parse_verdict(
                                item.get("verdict", "UNSUPPORTED")
                            )
                            results[idx] = VerificationResult(
                                verdict=verdict,
                                reasoning=item.get("reasoning", ""),
                                key_match=item.get("key_match"),
                            )
                    return results

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse batch response: {e}")

        # Last resort: return abstained results
        return [
            VerificationResult(
                verdict=Verdict.UNSUPPORTED,
                reasoning="Failed to parse batch response",
                abstained=True,
            )
            for _ in range(expected_count)
        ]
