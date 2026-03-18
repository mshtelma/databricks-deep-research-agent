"""Stage 4: Isolated Verification for the deep-research framework.

Verifies claims against evidence IN ISOLATION (no generation context)
to prevent bias propagation using the CoVe (Chain of Verification)
pattern.

Token-optimisation features:
- Batch verification: process up to 10 claim/evidence pairs per LLM call
- MD5-based verification cache to skip duplicate claim+evidence pairs
- Model tier escalation: quick path for high-confidence, full path otherwise
- Structured output with JSON fallback parsing

Ported from the app's ``services/citation/isolated_verifier.py``.
Prompts are inlined to keep the module self-contained.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re

from databricks_deep_research.citation.config import IsolatedVerificationConfig
from databricks_deep_research.citation.types import (
    BatchVerificationOutput,
    RankedEvidence,
    VerificationOutput,
    VerificationResult,
    VerificationVerdict,
)
from databricks_deep_research.llm.client import FrameworkLLMClient, ModelTier

logger = logging.getLogger(__name__)

# Default batch size for verification (spec mandates 10).
DEFAULT_BATCH_SIZE = 10


# ---------------------------------------------------------------------------
# Prompts (inlined from app's agent/prompts/citation/verification.py)
# ---------------------------------------------------------------------------

ISOLATED_VERIFICATION_PROMPT = """\
You are a Fact Checker verifying whether a claim is supported by evidence.

## CRITICAL: Isolated Verification
- You are checking this claim IN ISOLATION
- You have NO context about how this claim was generated
- Base your judgment ONLY on the evidence provided below

## Claim to Verify
"{claim_text}"

## Supporting Evidence
Source: {source_title}
URL: {source_url}
Quote: "{evidence_quote}"

## Verdict Categories

### SUPPORTED
The claim is FULLY entailed by the evidence:
- All facts in the claim are present in the evidence
- Numbers match exactly (or are correctly rounded)
- No extrapolation beyond what the evidence states

### PARTIAL
The claim is PARTIALLY supported:
- Some aspects are supported by the evidence
- Other aspects are not mentioned (neither confirmed nor denied)
- May involve reasonable inference from the evidence
- IMPORTANT: If the core factual claim (especially numeric values, dates, or named entities)
  is confirmed but the claim also contains editorial interpretation, analysis, or commentary
  that goes beyond the evidence, use PARTIAL — not UNSUPPORTED

### UNSUPPORTED
The claim has NO evidence basis:
- The evidence does not address the PRIMARY factual assertion in the claim
- The core factual content has no match in the evidence
- Cannot determine if the claim is true or false
- Different from CONTRADICTED -- this is "we don't know"
- IMPORTANT: Do NOT use UNSUPPORTED if the evidence confirms the main fact but
  doesn't confirm surrounding editorial commentary — that is PARTIAL

### CONTRADICTED
The evidence DIRECTLY opposes the claim:
- The evidence states the opposite
- Numbers are clearly different (not a rounding difference)
- Factual disagreement between claim and evidence

## Response Format
```json
{{
  "verdict": "SUPPORTED" | "PARTIAL" | "UNSUPPORTED" | "CONTRADICTED",
  "verification_confidence": 0.0-1.0,
  "reasoning": "Detailed explanation of why this verdict was chosen",
  "key_match": "Quote the specific part of evidence that supports/contradicts",
  "issues": ["List any specific issues found"]
}}
```

Verify the claim against the evidence:"""


QUICK_VERIFICATION_PROMPT = """\
Quickly verify if this claim matches the evidence.

## Claim
"{claim_text}"

## Evidence
"{evidence_quote}"

## Quick Check
1. Is the core fact in the claim present in the evidence? (Y/N)
2. Do any numbers match exactly? (Y/N/NA)
3. Is there any contradiction? (Y/N)

Based on these checks:
- If 1=Y and 3=N: SUPPORTED
- If 1=Partial and 3=N: PARTIAL
- If 1=N and 3=N: UNSUPPORTED
- If 3=Y: CONTRADICTED

Respond with just the verdict (SUPPORTED/PARTIAL/UNSUPPORTED/CONTRADICTED):"""


BATCH_VERIFICATION_PROMPT = """\
Verify each claim against its evidence INDEPENDENTLY.

## Claims to Verify
{claims_section}

## Instructions
For EACH claim above:
1. Verify ONLY against its provided evidence
2. Determine the verdict independently
3. Include the claim_index in your response

## Verdict Categories
- SUPPORTED: Evidence fully entails the claim (all facts present, numbers match)
- PARTIAL: Evidence partially supports (some aspects not mentioned). Use PARTIAL when the core fact is confirmed but the claim also contains editorial interpretation or commentary beyond the evidence
- UNSUPPORTED: Evidence doesn't address the PRIMARY factual assertion in the claim. Do NOT use if the main fact is confirmed but editorial commentary isn't
- CONTRADICTED: Evidence directly opposes the claim

## Response Format (JSON)
```json
{{
  "results": [
    {{
      "claim_index": 0,
      "verdict": "SUPPORTED" | "PARTIAL" | "UNSUPPORTED" | "CONTRADICTED",
      "verification_confidence": 0.0-1.0,
      "reasoning": "Brief explanation (max 100 words)",
      "key_match": "Specific quote from evidence if relevant"
    }},
    ...
  ]
}}
```

CRITICAL: Return one result per claim in the same order as input. \
Include claim_index to handle any reordering.

Verify all claims:"""


BATCH_QUICK_VERIFICATION_PROMPT = """\
Quickly verify if each claim matches its evidence.

## Claims to Verify
{claims_section}

## Quick Check Rules
- SUPPORTED: Core fact present in evidence, no contradiction
- PARTIAL: Some aspects supported, some not mentioned
- UNSUPPORTED: Evidence doesn't address the claim
- CONTRADICTED: Evidence says the opposite

## Response Format (JSON)
```json
{{
  "results": [
    {{"claim_index": 0, "verdict": "SUPPORTED"}},
    {{"claim_index": 1, "verdict": "PARTIAL"}},
    ...
  ]
}}
```

Verify all claims:"""


# ---------------------------------------------------------------------------
# IsolatedVerifier
# ---------------------------------------------------------------------------


class IsolatedVerifier:
    """Stage 4: Isolated Verification.

    Verifies claims against evidence IN ISOLATION using the CoVe
    (Chain of Verification) pattern to prevent bias propagation.

    The verifier operates in two modes:

    - **Single-claim**: Full or quick verification of one claim at a time.
    - **Batch**: Groups multiple claims into a single LLM call (up to
      ``DEFAULT_BATCH_SIZE`` pairs) with an MD5-based cache to skip
      previously-verified claim+evidence pairs.
    """

    def __init__(
        self,
        llm_client: FrameworkLLMClient,
        config: IsolatedVerificationConfig | None = None,
    ) -> None:
        """Initialise the isolated verifier.

        Args:
            llm_client: Framework LLM client for verification calls.
            config: Optional stage-4 configuration.  When *None* the
                default ``IsolatedVerificationConfig()`` is used.
        """
        self._llm = llm_client
        self._config = config or IsolatedVerificationConfig()

    # -- public single-claim API -------------------------------------------

    async def verify_with_isolation(
        self,
        claim_text: str,
        evidence: RankedEvidence,
        *,
        use_quick_verification: bool = False,
    ) -> VerificationResult:
        """Verify a claim against evidence in isolation.

        CRITICAL: This method receives NO generation context to prevent
        the LLM from "remembering" what it generated and confirming bias.

        Args:
            claim_text: The claim to verify.
            evidence: The supporting evidence span.
            use_quick_verification: Use fast verification for
                high-confidence claims.

        Returns:
            ``VerificationResult`` with verdict and reasoning.
        """
        if use_quick_verification:
            return await self._quick_verify(claim_text, evidence)
        return await self._full_verify(claim_text, evidence)

    # -- public batch API --------------------------------------------------

    async def verify_batch(
        self,
        claims: list[tuple[str, RankedEvidence]],
        confidence_levels: list[str] | None = None,
    ) -> list[VerificationResult]:
        """Verify multiple claims sequentially.

        Args:
            claims: List of ``(claim_text, evidence)`` tuples.
            confidence_levels: Optional confidence levels for routing
                (``"high"`` -> quick verification).

        Returns:
            List of ``VerificationResult`` objects in the same order.
        """
        results: list[VerificationResult] = []
        for i, (claim_text, evidence) in enumerate(claims):
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

    async def verify_batch_grouped(
        self,
        claims: list[tuple[str, RankedEvidence]],
        batch_size: int = DEFAULT_BATCH_SIZE,
        *,
        use_quick_verification: bool = False,
        verification_cache: dict[str, VerificationResult] | None = None,
    ) -> list[VerificationResult]:
        """Verify multiple claims using batched LLM calls.

        This is a **token-optimisation** method that processes multiple
        claims in a single LLM call, reducing overhead significantly.

        Args:
            claims: List of ``(claim_text, evidence)`` tuples.
            batch_size: Number of claims per batch (default 10).
            use_quick_verification: Use faster, simpler verification.
            verification_cache: Optional dict keyed by MD5 fingerprint
                for result re-use across invocations.

        Returns:
            List of ``VerificationResult`` objects in the same order as
            the input *claims*.
        """
        if not claims:
            return []

        results: list[VerificationResult | None] = [None] * len(claims)
        uncached_indices: list[int] = []

        # Phase 1: check cache ------------------------------------------------
        if verification_cache is not None:
            for i, (claim_text, evidence) in enumerate(claims):
                fp = self.fingerprint_pair(claim_text, evidence.quote_text)
                if fp in verification_cache:
                    results[i] = verification_cache[fp]
                    logger.debug(
                        "VERIFICATION_CACHE_HIT claim_index=%d fingerprint=%s",
                        i,
                        fp,
                    )
                else:
                    uncached_indices.append(i)
        else:
            uncached_indices = list(range(len(claims)))

        if not uncached_indices:
            return [r for r in results if r is not None]

        # Phase 2: group uncached claims into batches --------------------------
        batches: list[list[int]] = []
        for start in range(0, len(uncached_indices), batch_size):
            batches.append(uncached_indices[start : start + batch_size])

        logger.info(
            "BATCH_VERIFICATION_START total_claims=%d cached=%d "
            "uncached=%d batches=%d",
            len(claims),
            len(claims) - len(uncached_indices),
            len(uncached_indices),
            len(batches),
        )

        # Phase 3: process each batch ------------------------------------------
        for batch_num, batch_indices in enumerate(batches):
            batch_claims = [claims[i] for i in batch_indices]
            try:
                batch_results = await self._process_batch(
                    batch_claims, use_quick_verification
                )
                for j, idx in enumerate(batch_indices):
                    if j < len(batch_results):
                        results[idx] = batch_results[j]
                        if verification_cache is not None:
                            fp = self.fingerprint_pair(
                                claims[idx][0], claims[idx][1].quote_text
                            )
                            verification_cache[fp] = batch_results[j]
                    else:
                        results[idx] = VerificationResult(
                            verdict=VerificationVerdict.UNSUPPORTED,
                            reasoning="Batch verification returned no result",
                            abstained=True,
                        )
            except Exception as exc:
                logger.warning(
                    "BATCH_VERIFICATION_ERROR batch_num=%d error=%s",
                    batch_num,
                    str(exc)[:100],
                )
                # Fall back to sequential verification for this batch.
                for idx in batch_indices:
                    claim_text, evidence = claims[idx]
                    results[idx] = await self.verify_with_isolation(
                        claim_text=claim_text,
                        evidence=evidence,
                        use_quick_verification=use_quick_verification,
                    )

        # Fill any remaining Nones with abstained results ----------------------
        for i, result in enumerate(results):
            if result is None:
                results[i] = VerificationResult(
                    verdict=VerificationVerdict.UNSUPPORTED,
                    reasoning="Verification incomplete",
                    abstained=True,
                )

        logger.info(
            "BATCH_VERIFICATION_COMPLETE total_claims=%d results=%d",
            len(claims),
            len([r for r in results if r is not None]),
        )

        return [r for r in results if r is not None]

    # -- NEI heuristic check -----------------------------------------------

    def check_nei(
        self,
        claim_text: str,
        evidence: RankedEvidence,
    ) -> bool:
        """Check if Not Enough Information (NEI) verdict likely applies.

        Quick heuristic check before full verification based on word
        overlap between claim and evidence.

        Args:
            claim_text: The claim to check.
            evidence: The evidence span.

        Returns:
            ``True`` if NEI likely applies (< 20 % word overlap).
        """
        if not self._config.enable_nei_verdict:
            return False

        claim_words = set(re.findall(r"\b\w{4,}\b", claim_text.lower()))
        evidence_words = set(
            re.findall(r"\b\w{4,}\b", evidence.quote_text.lower())
        )
        if not claim_words:
            return True

        overlap = len(claim_words & evidence_words) / len(claim_words)
        return overlap < 0.2

    # -- fingerprinting / caching ------------------------------------------

    @staticmethod
    def fingerprint_claim(claim_text: str) -> str:
        """Create normalised fingerprint for claim caching.

        Normalisation: lowercase, remove punctuation, sort words, then
        MD5-hash to a 16-character hex digest.

        Args:
            claim_text: The claim text to fingerprint.

        Returns:
            16-character MD5 hex digest.
        """
        normalised = re.sub(r"[^\w\s]", "", claim_text.lower())
        words = sorted(normalised.split())
        return hashlib.md5(" ".join(words).encode()).hexdigest()[:16]

    @staticmethod
    def fingerprint_pair(claim_text: str, evidence_text: str) -> str:
        """Create normalised fingerprint for a claim+evidence pair.

        Hashing both claim and evidence avoids false cache hits when the
        same claim is checked against different evidence spans.

        Args:
            claim_text: The claim text.
            evidence_text: The evidence quote text.

        Returns:
            16-character MD5 hex digest.
        """
        claim_norm = re.sub(r"[^\w\s]", "", claim_text.lower())
        evidence_norm = re.sub(r"[^\w\s]", "", evidence_text.lower())
        combined = f"{sorted(claim_norm.split())}|{sorted(evidence_norm.split())}"
        return hashlib.md5(combined.encode()).hexdigest()[:16]

    # -- verdict parsing ---------------------------------------------------

    @staticmethod
    def parse_verdict(verdict_text: str) -> VerificationVerdict:
        """Parse verdict from raw LLM text.

        The parser is deliberately tolerant of surrounding whitespace,
        mixed case, and extra words.

        Args:
            verdict_text: Raw verdict text from LLM.

        Returns:
            Parsed ``VerificationVerdict`` enum value.
        """
        text = verdict_text.upper().strip()
        if "SUPPORTED" in text and "UNSUPPORTED" not in text:
            return VerificationVerdict.SUPPORTED
        if "PARTIAL" in text:
            return VerificationVerdict.PARTIAL
        if "CONTRADICTED" in text:
            return VerificationVerdict.CONTRADICTED
        return VerificationVerdict.UNSUPPORTED

    # ======================================================================
    # Private helpers
    # ======================================================================

    # -- single-claim verification -----------------------------------------

    async def _full_verify(
        self,
        claim_text: str,
        evidence: RankedEvidence,
    ) -> VerificationResult:
        """Full verification with detailed reasoning."""
        prompt = ISOLATED_VERIFICATION_PROMPT.format(
            claim_text=claim_text,
            source_title=evidence.source_title or "Unknown",
            source_url=evidence.source_url,
            evidence_quote=evidence.quote_text,
        )

        tier = self._resolve_tier(self._config.verification_model_tier)

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
                    confidence=(
                        output.verification_confidence
                        if output.verification_confidence is not None
                        else self._default_confidence(verdict)
                    ),
                )

            # Fallback: parse from raw content.
            return self._parse_verification_response(response.content)

        except Exception as exc:
            logger.error("Full verification failed: %s", exc)
            return VerificationResult(
                verdict=VerificationVerdict.UNSUPPORTED,
                reasoning=f"Verification failed: {exc}",
                confidence=0.0,
                abstained=True,
            )

    async def _quick_verify(
        self,
        claim_text: str,
        evidence: RankedEvidence,
    ) -> VerificationResult:
        """Quick verification for high-confidence claims."""
        prompt = QUICK_VERIFICATION_PROMPT.format(
            claim_text=claim_text,
            evidence_quote=evidence.quote_text,
        )

        tier = self._resolve_tier(self._config.quick_verification_tier)

        try:
            response = await self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                tier=tier,
            )
            verdict = self.parse_verdict(response.content.strip().upper())
            return VerificationResult(
                verdict=verdict,
                reasoning="Quick verification",
                confidence=self._default_confidence(verdict, quick=True),
            )
        except Exception as exc:
            logger.error("Quick verification failed: %s", exc)
            return VerificationResult(
                verdict=VerificationVerdict.UNSUPPORTED,
                reasoning=f"Quick verification failed: {exc}",
                confidence=0.0,
                abstained=True,
            )

    # -- batch verification ------------------------------------------------

    def _format_claims_for_batch(
        self,
        claims: list[tuple[str, RankedEvidence]],
    ) -> str:
        """Format claims for the batch verification prompt."""
        sections: list[str] = []
        for i, (claim_text, evidence) in enumerate(claims):
            section = (
                f"### Claim {i}\n"
                f'**Claim:** "{claim_text}"\n'
                f"**Source:** {evidence.source_title or 'Unknown'}\n"
                f'**Evidence:** "{evidence.quote_text[:1000]}"\n'
            )
            sections.append(section)
        return "\n".join(sections)

    async def _process_batch(
        self,
        claims: list[tuple[str, RankedEvidence]],
        use_quick_verification: bool = False,
    ) -> list[VerificationResult]:
        """Process a single batch of claims via one LLM call."""
        if not claims:
            return []

        claims_section = self._format_claims_for_batch(claims)

        if use_quick_verification:
            prompt = BATCH_QUICK_VERIFICATION_PROMPT.format(
                claims_section=claims_section,
            )
            tier = self._resolve_tier(self._config.quick_verification_tier)
        else:
            prompt = BATCH_VERIFICATION_PROMPT.format(
                claims_section=claims_section,
            )
            tier = self._resolve_tier(self._config.verification_model_tier)

        try:
            response = await self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                tier=tier,
                structured_output=BatchVerificationOutput,
            )

            if response.structured:
                output: BatchVerificationOutput = response.structured
                return self._parse_batch_results(output, len(claims))

            # Fallback: try to parse from raw content.
            return self._parse_batch_response_content(
                response.content, len(claims)
            )

        except Exception as exc:
            logger.error("Batch processing failed: %s", exc)
            raise

    def _parse_batch_results(
        self,
        output: BatchVerificationOutput,
        expected_count: int,
    ) -> list[VerificationResult]:
        """Parse structured batch output into results list.

        Handles potential reordering by using ``claim_index`` from output.
        """
        results: list[VerificationResult] = [
            VerificationResult(
                verdict=VerificationVerdict.UNSUPPORTED,
                reasoning="No result in batch output",
                abstained=True,
            )
            for _ in range(expected_count)
        ]

        for item in output.results:
            if 0 <= item.claim_index < expected_count:
                verdict = self.parse_verdict(item.verdict)
                results[item.claim_index] = VerificationResult(
                    verdict=verdict,
                    reasoning=item.reasoning,
                    key_match=item.key_match,
                    confidence=(
                        item.verification_confidence
                        if item.verification_confidence is not None
                        else self._default_confidence(verdict)
                    ),
                )
            else:
                logger.warning(
                    "BATCH_RESULT_INDEX_OUT_OF_RANGE claim_index=%d "
                    "expected_count=%d",
                    item.claim_index,
                    expected_count,
                )

        return results

    def _parse_batch_response_content(
        self,
        content: str,
        expected_count: int,
    ) -> list[VerificationResult]:
        """Fallback parser for batch response when structured output fails."""
        try:
            json_match = re.search(r"\{[\s\S]*\}", content)
            if json_match:
                data = json.loads(json_match.group())
                if "results" in data and isinstance(data["results"], list):
                    results: list[VerificationResult] = [
                        VerificationResult(
                            verdict=VerificationVerdict.UNSUPPORTED,
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
                                confidence=float(
                                    item.get(
                                        "verification_confidence",
                                        self._default_confidence(verdict),
                                    )
                                ),
                            )
                    return results

        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Failed to parse batch response: %s", exc)

        return [
            VerificationResult(
                verdict=VerificationVerdict.UNSUPPORTED,
                reasoning="Failed to parse batch response",
                abstained=True,
            )
            for _ in range(expected_count)
        ]

    # -- response parsing --------------------------------------------------

    def _parse_verification_response(self, response: str) -> VerificationResult:
        """Parse a full verification response from raw LLM text."""
        try:
            json_match = re.search(r"\{[\s\S]*?\}", response)
            if json_match:
                data = json.loads(json_match.group())
                verdict = self.parse_verdict(
                    data.get("verdict", "UNSUPPORTED")
                )
                return VerificationResult(
                    verdict=verdict,
                    reasoning=data.get("reasoning", ""),
                    key_match=data.get("key_match"),
                    issues=data.get("issues"),
                    confidence=float(
                        data.get(
                            "verification_confidence",
                            self._default_confidence(verdict),
                        )
                    ),
                )
        except (json.JSONDecodeError, AttributeError) as exc:
            logger.warning("Failed to parse verification JSON: %s", exc)

        # Last resort: extract verdict from raw text.
        verdict = self.parse_verdict(response)
        return VerificationResult(
            verdict=verdict,
            reasoning=response[:500],
            confidence=self._default_confidence(verdict),
        )

    @staticmethod
    def _default_confidence(
        verdict: VerificationVerdict,
        *,
        quick: bool = False,
    ) -> float:
        """Provide an explicit fallback confidence when the verifier omits one."""
        if quick:
            mapping = {
                VerificationVerdict.SUPPORTED: 0.75,
                VerificationVerdict.PARTIAL: 0.65,
                VerificationVerdict.UNSUPPORTED: 0.55,
                VerificationVerdict.CONTRADICTED: 0.75,
            }
        else:
            mapping = {
                VerificationVerdict.SUPPORTED: 0.85,
                VerificationVerdict.PARTIAL: 0.7,
                VerificationVerdict.UNSUPPORTED: 0.6,
                VerificationVerdict.CONTRADICTED: 0.85,
            }
        return mapping[verdict]

    # -- tier resolution ---------------------------------------------------

    @staticmethod
    def _resolve_tier(tier_str: str) -> str | ModelTier:
        """Map a tier string to a ``ModelTier`` if it matches, else pass through."""
        try:
            return ModelTier(tier_str)
        except ValueError:
            return tier_str
