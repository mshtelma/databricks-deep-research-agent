"""Post-verification of claims using existing verification pipeline (stages 4-6).

This module provides post-generation verification for structured output,
reusing the framework's existing verification components.

Usage:
    verifier = PostVerifier(llm, sources, evidence_pool)
    result = await verifier.verify_claims(claims)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from deep_research.core.app_config import get_app_config
from deep_research.core.logging_utils import get_logger, truncate
from deep_research.services.citation.citation_corrector import (
    CitationCorrector,
    CorrectionType,
)
from deep_research.services.citation.evidence_selector import RankedEvidence
from deep_research.services.citation.isolated_verifier import (
    IsolatedVerifier,
    Verdict,
)
from deep_research.services.citation.numeric_verifier import NumericVerifier
from deep_research.services.llm.client import LLMClient

if TYPE_CHECKING:
    from deep_research.services.citation.claim_extractor import ExtractedClaim

logger = get_logger(__name__)


@dataclass
class VerifiedClaim:
    """A claim after verification with verdict and optional corrections."""

    original: ExtractedClaim
    verdict: Verdict  # SUPPORTED, PARTIAL, UNSUPPORTED, CONTRADICTED
    corrected_source_refs: list[str] | None = None
    corrected_text: str | None = None
    reasoning: str | None = None
    numeric_verified: bool | None = None


@dataclass
class PostVerificationResult:
    """Summary of post-verification run."""

    total_claims: int
    verified_claims: list[VerifiedClaim]
    supported_count: int
    partial_count: int
    unsupported_count: int
    skipped_count: int
    corrections_applied: int

    @property
    def support_rate(self) -> float:
        """Percentage of claims that are fully supported."""
        if self.total_claims == 0:
            return 0.0
        return self.supported_count / self.total_claims


@dataclass
class SourceInfo:
    """Minimal source info for lookup."""

    url: str
    title: str | None = None
    snippet: str | None = None


class PostVerifier:
    """Run verification stages 4-6 on pre-extracted claims.

    This class reuses the framework's existing verification components:
    - Stage 4: IsolatedVerifier (CoVe pattern - verify without generation context)
    - Stage 5: CitationCorrector (CiteFix pattern - find better citations)
    - Stage 6: NumericVerifier (QAFactEval pattern - verify numeric claims)

    Stage 7 (VerificationRetriever) is NOT included by default as it requires
    external web searches and is optimized for ReClaim-style generation.
    """

    def __init__(
        self,
        llm: LLMClient,
        sources: list[SourceInfo],
        evidence_pool: list[RankedEvidence] | None = None,
    ) -> None:
        """Initialize post-verifier with LLM and source context.

        Args:
            llm: LLM client for verification calls.
            sources: List of sources from research (for reference lookup).
            evidence_pool: Pre-built evidence pool. If None, will be built from sources.
        """
        self._llm = llm
        self._sources = sources
        self._evidence_pool = evidence_pool or []
        self._config = get_app_config().citation_verification.post_verification

        # Initialize verifiers lazily
        self._isolated_verifier: IsolatedVerifier | None = None
        self._citation_corrector: CitationCorrector | None = None
        self._numeric_verifier: NumericVerifier | None = None

        # Build source lookup for quick access (1-indexed like citations)
        self._source_lookup: dict[str, SourceInfo] = {
            str(i + 1): src for i, src in enumerate(sources)
        }

    def _get_isolated_verifier(self) -> IsolatedVerifier:
        """Lazy initialization of IsolatedVerifier."""
        if self._isolated_verifier is None:
            self._isolated_verifier = IsolatedVerifier(self._llm)
        return self._isolated_verifier

    def _get_citation_corrector(self) -> CitationCorrector:
        """Lazy initialization of CitationCorrector."""
        if self._citation_corrector is None:
            self._citation_corrector = CitationCorrector(self._llm)
        return self._citation_corrector

    def _get_numeric_verifier(self) -> NumericVerifier:
        """Lazy initialization of NumericVerifier."""
        if self._numeric_verifier is None:
            self._numeric_verifier = NumericVerifier(self._llm)
        return self._numeric_verifier

    def _get_evidence_for_refs(
        self,
        source_refs: list[str],
    ) -> RankedEvidence | None:
        """Get evidence from evidence pool matching source references.

        Args:
            source_refs: List of source indices like ["1", "2"].

        Returns:
            Best matching RankedEvidence or None if not found.
        """
        if not self._evidence_pool or not source_refs:
            return None

        # Find evidence matching any of the source refs
        for ref in source_refs:
            source = self._source_lookup.get(ref)
            if not source:
                continue
            # Find evidence from this source
            for evidence in self._evidence_pool:
                if source.url and evidence.source_url == source.url:
                    return evidence
        return None

    async def verify_claims(
        self,
        claims: list[ExtractedClaim],
    ) -> PostVerificationResult:
        """Run verification on all claims.

        Args:
            claims: Claims extracted from structured output.

        Returns:
            PostVerificationResult with all verification outcomes.
        """
        if not self._config.enabled:
            logger.info("POST_VERIFICATION_DISABLED")
            return PostVerificationResult(
                total_claims=len(claims),
                verified_claims=[],
                supported_count=0,
                partial_count=0,
                unsupported_count=0,
                skipped_count=len(claims),
                corrections_applied=0,
            )

        logger.info(
            "POST_VERIFICATION_START",
            total_claims=len(claims),
            max_claims=self._config.max_claims_to_verify,
        )

        verified: list[VerifiedClaim] = []
        skipped = 0
        corrections = 0

        # Filter and limit claims
        claims_to_verify = self._filter_claims(claims)

        for claim in claims_to_verify:
            try:
                result = await self._verify_single_claim(claim)
                verified.append(result)
                if result.corrected_source_refs or result.corrected_text:
                    corrections += 1
            except Exception as e:
                logger.error(
                    "POST_VERIFICATION_CLAIM_ERROR",
                    field_path=claim.field_path,
                    error=str(e)[:200],
                )
                # Mark as skipped on error
                skipped += 1

        # Count verdicts
        supported = sum(1 for v in verified if v.verdict == Verdict.SUPPORTED)
        partial = sum(1 for v in verified if v.verdict == Verdict.PARTIAL)
        unsupported = sum(
            1 for v in verified if v.verdict in (Verdict.UNSUPPORTED, Verdict.CONTRADICTED)
        )
        skipped += len(claims) - len(claims_to_verify)

        logger.info(
            "POST_VERIFICATION_COMPLETE",
            verified=len(verified),
            supported=supported,
            partial=partial,
            unsupported=unsupported,
            skipped=skipped,
            corrections=corrections,
        )

        return PostVerificationResult(
            total_claims=len(claims),
            verified_claims=verified,
            supported_count=supported,
            partial_count=partial,
            unsupported_count=unsupported,
            skipped_count=skipped,
            corrections_applied=corrections,
        )

    def _filter_claims(self, claims: list[ExtractedClaim]) -> list[ExtractedClaim]:
        """Filter and limit claims for verification."""
        filtered = list(claims)

        # Skip low-priority short claims if configured
        if self._config.skip_low_priority_claims:
            filtered = [
                c for c in filtered if c.priority != "low" or c.char_count >= 300
            ]

        # Limit to max claims
        if len(filtered) > self._config.max_claims_to_verify:
            # Prioritize: high > medium > low, then by length
            filtered = sorted(
                filtered,
                key=lambda c: (
                    {"high": 0, "medium": 1, "low": 2}.get(c.priority, 1),
                    -c.char_count,
                ),
            )[: self._config.max_claims_to_verify]

        return filtered

    async def _verify_single_claim(
        self,
        claim: ExtractedClaim,
    ) -> VerifiedClaim:
        """Verify a single claim through stages 4-6."""
        evidence = self._get_evidence_for_refs(claim.source_refs)

        # Stage 4: Isolated verification
        verdict = Verdict.UNSUPPORTED
        reasoning = None

        if self._config.include_stage4_isolation and evidence:
            verifier = self._get_isolated_verifier()
            result = await verifier.verify_with_isolation(
                claim_text=claim.text,
                evidence=evidence,
            )
            verdict = result.verdict
            reasoning = result.reasoning

        corrected_refs = None
        corrected_text = None

        # Stage 5: Citation correction if needed
        if (
            self._config.include_stage5_correction
            and verdict in (Verdict.UNSUPPORTED, Verdict.PARTIAL)
            and self._evidence_pool
        ):
            corrector = self._get_citation_corrector()
            correction = await corrector.correct_single_citation(
                claim=claim.text,
                current_evidence=evidence,
                evidence_pool=self._evidence_pool,
                current_verdict=verdict.value if verdict else None,
            )
            if (
                correction.correction_type == CorrectionType.REPLACE
                and correction.corrected_evidence
            ):
                # Find source index for corrected evidence
                for idx, src in self._source_lookup.items():
                    if src.url == correction.corrected_evidence.source_url:
                        corrected_refs = [idx]
                        break

        # Stage 6: Numeric verification
        numeric_verified = None
        if self._config.include_stage6_numeric and evidence:
            verifier = self._get_numeric_verifier()
            if verifier.detect_numeric_claims(claim.text):
                result = await verifier.verify_numeric_claim(
                    claim_text=claim.text,
                    evidence=evidence,
                )
                numeric_verified = result.overall_match
                # Downgrade verdict if numeric mismatch
                if not result.overall_match and verdict == Verdict.SUPPORTED:
                    verdict = Verdict.PARTIAL

        return VerifiedClaim(
            original=claim,
            verdict=verdict,
            corrected_source_refs=corrected_refs,
            corrected_text=corrected_text,
            reasoning=reasoning,
            numeric_verified=numeric_verified,
        )
