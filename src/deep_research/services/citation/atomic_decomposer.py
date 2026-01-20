"""Stage 7: Atomic Fact Decomposer service.

Implements FActScore-style atomic fact decomposition for ARE
(Atomic fact decomposition-based Retrieval and Editing) verification.

Scientific basis:
- FActScore: https://arxiv.org/abs/2305.14251 (EMNLP 2023)
- SAFE: https://arxiv.org/abs/2403.18802 (Google DeepMind)
- ARE: https://arxiv.org/abs/2410.16708

The decomposer breaks complex claims into independent, self-contained
atomic facts that can be verified individually. This enables:
1. More precise evidence retrieval (facts as queries)
2. Granular verification (per-fact verdicts)
3. Targeted revision (soften only unverified parts)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

import mlflow
from mlflow.entities import SpanType
from pydantic import BaseModel, Field

from deep_research.agent.prompts.citation.verification_retrieval import (
    ATOMIC_DECOMPOSITION_PROMPT,
)
from deep_research.core.app_config import VerificationRetrievalConfig, get_app_config
from deep_research.core.logging_utils import get_logger, truncate
from deep_research.core.tracing_constants import (
    ATTR_BATCH_FILTERED,
    ATTR_BATCH_TOTAL,
    ATTR_CLAIM_INDEX,
    ATTR_CLAIM_TEXT,
    ATTR_DECOMPOSITION_FACT_COUNT,
    ATTR_DECOMPOSITION_REASONING,
    ATTR_DECOMPOSITION_SKIPPED,
    STAGE_7_ARE,
    citation_span_name,
    truncate_for_attr,
)
from deep_research.services.llm.client import LLMClient
from deep_research.services.llm.types import ModelTier

if TYPE_CHECKING:
    from deep_research.agent.state import ClaimInfo
    from deep_research.services.citation.evidence_selector import RankedEvidence

logger = get_logger(__name__)


# =============================================================================
# Data Models
# =============================================================================


class EvidenceSource(str, Enum):
    """Source of evidence supporting an atomic fact."""

    INTERNAL = "internal"  # From existing evidence pool
    EXTERNAL = "external"  # From new Brave search + crawl
    NONE = "none"  # No evidence found


@dataclass
class AtomicFact:
    """Single atomic fact decomposed from a claim.

    Each atomic fact is:
    - Self-contained (no pronouns, explicit references)
    - Independently verifiable
    - A single, simple statement

    Example:
        Original claim: "OpenAI released GPT-4 in March 2023, achieving 90% on bar exam."
        Atomic facts:
        1. "OpenAI released GPT-4."
        2. "GPT-4 was released in March 2023."
        3. "GPT-4 achieved 90% on the bar exam."
    """

    fact_text: str
    fact_index: int
    parent_claim_id: int  # Index of parent claim in claims list

    # Verification results (populated during verification)
    is_verified: bool = False
    evidence: RankedEvidence | None = None
    evidence_source: EvidenceSource = EvidenceSource.NONE
    entailment_score: float = 0.0
    search_queries: list[str] = field(default_factory=list)

    # Pre-assigned citation key for Stage 7 (set before reconstruction to ensure consistency)
    assigned_citation_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "fact_text": self.fact_text,
            "fact_index": self.fact_index,
            "parent_claim_id": self.parent_claim_id,
            "is_verified": self.is_verified,
            "evidence_source": self.evidence_source.value,
            "entailment_score": self.entailment_score,
            "search_queries": self.search_queries,
            "assigned_citation_key": self.assigned_citation_key,
            "evidence": {
                "source_url": self.evidence.source_url if self.evidence else None,
                "quote_text": truncate(self.evidence.quote_text, 200) if self.evidence else None,
            } if self.evidence else None,
        }


@dataclass
class ClaimDecomposition:
    """Result of decomposing a claim into atomic facts.

    Tracks both the decomposition and aggregated verification status.
    """

    original_claim: ClaimInfo
    atomic_facts: list[AtomicFact]
    decomposition_reasoning: str = ""

    # Aggregated verification results (updated during verification)
    all_verified: bool = False
    partial_verified: bool = False
    verified_count: int = 0
    total_count: int = 0

    def update_verification_status(self) -> None:
        """Recalculate aggregated verification status from atomic facts."""
        self.total_count = len(self.atomic_facts)
        self.verified_count = sum(1 for f in self.atomic_facts if f.is_verified)
        self.all_verified = self.verified_count == self.total_count and self.total_count > 0
        self.partial_verified = 0 < self.verified_count < self.total_count

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "original_claim": self.original_claim.claim_text,
            "atomic_facts": [f.to_dict() for f in self.atomic_facts],
            "decomposition_reasoning": self.decomposition_reasoning,
            "all_verified": self.all_verified,
            "partial_verified": self.partial_verified,
            "verified_count": self.verified_count,
            "total_count": self.total_count,
        }


@dataclass
class ClaimRevision:
    """Result of revising a claim based on atomic fact verification.

    Contains both the revised text and tracking information for
    applying the revision to the report.
    """

    original_claim: str
    revised_claim: str
    revision_type: Literal["fully_verified", "partially_softened", "fully_softened"]

    # Position tracking for report replacement
    original_position_start: int
    original_position_end: int

    # Detailed breakdown
    decomposition: ClaimDecomposition
    verified_facts: list[AtomicFact] = field(default_factory=list)
    softened_facts: list[AtomicFact] = field(default_factory=list)

    # New citations added during verification
    new_citations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "original_claim": self.original_claim,
            "revised_claim": self.revised_claim,
            "revision_type": self.revision_type,
            "original_position_start": self.original_position_start,
            "original_position_end": self.original_position_end,
            "verified_facts": [f.to_dict() for f in self.verified_facts],
            "softened_facts": [f.to_dict() for f in self.softened_facts],
            "new_citations": self.new_citations,
        }


@dataclass
class DecompositionMetrics:
    """Aggregate metrics for atomic decomposition stage."""

    total_claims_processed: int = 0
    total_atomic_facts: int = 0
    single_fact_claims: int = 0  # Claims that didn't decompose
    multi_fact_claims: int = 0
    decomposition_failures: int = 0
    avg_facts_per_claim: float = 0.0

    def compute_avg(self) -> None:
        """Compute average facts per claim."""
        if self.total_claims_processed > 0:
            self.avg_facts_per_claim = (
                self.total_atomic_facts / self.total_claims_processed
            )


# =============================================================================
# Pydantic Models for Structured LLM Output
# =============================================================================


class AtomicDecompositionOutput(BaseModel):
    """Output from atomic fact decomposition LLM call."""

    atomic_facts: list[str] = Field(
        default_factory=list,
        description="List of atomic, self-contained facts",
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of the decomposition",
    )


# =============================================================================
# Atomic Decomposer Service
# =============================================================================


class AtomicDecomposer:
    """Decomposes claims into atomic, self-contained facts.

    Implements FActScore-style decomposition:
    1. Split claim into independent factual statements
    2. Make each fact self-contained (decontextualize)
    3. Ensure each fact is verifiable independently

    This is the first step in the ARE (Atomic fact decomposition-based
    Retrieval and Editing) pattern for claim verification.
    """

    def __init__(
        self,
        llm: LLMClient,
        config: VerificationRetrievalConfig | None = None,
    ) -> None:
        """Initialize the decomposer.

        Args:
            llm: LLM client for decomposition.
            config: Verification retrieval configuration.
        """
        self.llm = llm
        self.config = config or get_app_config().citation_verification.verification_retrieval

    async def decompose(
        self,
        claim: ClaimInfo,
        claim_index: int,
        context: str | None = None,
    ) -> ClaimDecomposition:
        """Decompose a claim into atomic facts.

        Follows FActScore methodology:
        1. Split into independent factual statements
        2. Make each self-contained (resolve pronouns, add context)
        3. Ensure each is verifiable independently

        Args:
            claim: The claim to decompose.
            claim_index: Index of this claim in the claims list.
            context: Optional surrounding context for decontextualization.

        Returns:
            ClaimDecomposition with list of atomic facts.
        """
        span_name = citation_span_name(STAGE_7_ARE, "decompose", claim_index)

        with mlflow.start_span(name=span_name, span_type=SpanType.CHAIN) as span:
            claim_text = claim.claim_text

            # Set input attributes
            span.set_attributes({
                ATTR_CLAIM_INDEX: claim_index,
                ATTR_CLAIM_TEXT: truncate_for_attr(claim_text, 150),
                "claim.word_count": len(claim_text.split()),
            })

            # Skip very short claims (likely already atomic)
            if len(claim_text.split()) <= 8:
                logger.debug(
                    "DECOMPOSITION_SKIP_SHORT",
                    claim=truncate(claim_text, 50),
                    word_count=len(claim_text.split()),
                )
                span.set_attributes({
                    ATTR_DECOMPOSITION_SKIPPED: True,
                    ATTR_DECOMPOSITION_FACT_COUNT: 1,
                    ATTR_DECOMPOSITION_REASONING: "Claim is short enough to be atomic",
                })
                return ClaimDecomposition(
                    original_claim=claim,
                    atomic_facts=[
                        AtomicFact(
                            fact_text=claim_text,
                            fact_index=0,
                            parent_claim_id=claim_index,
                        )
                    ],
                    decomposition_reasoning="Claim is short enough to be atomic",
                )

            try:
                # Build prompt
                prompt = ATOMIC_DECOMPOSITION_PROMPT.format(claim_text=claim_text)

                # Get tier from config string
                tier = self._get_model_tier(self.config.decomposition_tier)

                # Call LLM with structured output
                response = await asyncio.wait_for(
                    self.llm.complete(
                        messages=[{"role": "user", "content": prompt}],
                        tier=tier,
                        structured_output=AtomicDecompositionOutput,
                    ),
                    timeout=self.config.decomposition_timeout_seconds,
                )

                if not response.structured:
                    logger.warning(
                        "DECOMPOSITION_NO_STRUCTURED_OUTPUT",
                        claim=truncate(claim_text, 50),
                    )
                    result = self._fallback_decomposition(claim, claim_index)
                    span.set_attributes({
                        ATTR_DECOMPOSITION_SKIPPED: False,
                        ATTR_DECOMPOSITION_FACT_COUNT: len(result.atomic_facts),
                        ATTR_DECOMPOSITION_REASONING: "Fallback: no structured output",
                        "decomposition.fallback": True,
                    })
                    return result

                output: AtomicDecompositionOutput = response.structured

                # Validate and create atomic facts
                atomic_facts = []
                seen_facts: set[str] = set()  # Deduplicate

                for i, fact_text in enumerate(output.atomic_facts):
                    # Skip empty or duplicate facts
                    fact_normalized = fact_text.strip().lower()
                    if not fact_text.strip() or fact_normalized in seen_facts:
                        continue
                    seen_facts.add(fact_normalized)

                    # Decontextualize if needed
                    decontextualized = self._decontextualize(fact_text.strip(), claim_text)

                    atomic_facts.append(
                        AtomicFact(
                            fact_text=decontextualized,
                            fact_index=i,
                            parent_claim_id=claim_index,
                        )
                    )

                # Enforce max facts limit
                truncated = False
                if len(atomic_facts) > self.config.max_atomic_facts_per_claim:
                    logger.debug(
                        "DECOMPOSITION_TRUNCATED",
                        claim=truncate(claim_text, 50),
                        original_count=len(atomic_facts),
                        limit=self.config.max_atomic_facts_per_claim,
                    )
                    truncated = True
                    atomic_facts = atomic_facts[: self.config.max_atomic_facts_per_claim]

                # If decomposition produced no facts, use original claim
                if not atomic_facts:
                    atomic_facts = [
                        AtomicFact(
                            fact_text=claim_text,
                            fact_index=0,
                            parent_claim_id=claim_index,
                        )
                    ]

                logger.info(
                    "CLAIM_DECOMPOSED",
                    claim=truncate(claim_text, 50),
                    fact_count=len(atomic_facts),
                )

                # Set output attributes
                span.set_attributes({
                    ATTR_DECOMPOSITION_SKIPPED: False,
                    ATTR_DECOMPOSITION_FACT_COUNT: len(atomic_facts),
                    ATTR_DECOMPOSITION_REASONING: truncate_for_attr(output.reasoning, 200),
                    "decomposition.truncated": truncated,
                    "decomposition.fallback": False,
                })

                return ClaimDecomposition(
                    original_claim=claim,
                    atomic_facts=atomic_facts,
                    decomposition_reasoning=output.reasoning,
                )

            except asyncio.TimeoutError:
                logger.warning(
                    "DECOMPOSITION_TIMEOUT",
                    claim=truncate(claim_text, 50),
                    timeout=self.config.decomposition_timeout_seconds,
                )
                result = self._fallback_decomposition(claim, claim_index)
                span.set_attributes({
                    ATTR_DECOMPOSITION_FACT_COUNT: len(result.atomic_facts),
                    ATTR_DECOMPOSITION_REASONING: "Fallback: timeout",
                    "decomposition.timeout": True,
                    "decomposition.fallback": True,
                })
                return result

            except Exception as e:
                logger.error(
                    "DECOMPOSITION_ERROR",
                    claim=truncate(claim_text, 50),
                    error=str(e)[:100],
                )
                result = self._fallback_decomposition(claim, claim_index)
                span.set_attributes({
                    ATTR_DECOMPOSITION_FACT_COUNT: len(result.atomic_facts),
                    ATTR_DECOMPOSITION_REASONING: f"Fallback: {str(e)[:100]}",
                    "decomposition.error": str(e)[:100],
                    "decomposition.fallback": True,
                })
                return result

    def _fallback_decomposition(
        self,
        claim: ClaimInfo,
        claim_index: int,
    ) -> ClaimDecomposition:
        """Create a fallback decomposition (treat claim as single atomic fact).

        Used when LLM decomposition fails or times out.
        """
        return ClaimDecomposition(
            original_claim=claim,
            atomic_facts=[
                AtomicFact(
                    fact_text=claim.claim_text,
                    fact_index=0,
                    parent_claim_id=claim_index,
                )
            ],
            decomposition_reasoning="Fallback: treating entire claim as single atomic fact",
        )

    def _decontextualize(self, fact: str, claim: str) -> str:
        """Make a fact self-contained without the original claim context.

        Applies simple heuristic rules:
        1. Resolve common pronouns if context allows
        2. Ensure the fact makes sense standalone

        For complex decontextualization, the LLM prompt already instructs
        to make facts self-contained, so this is a lightweight fallback.

        Args:
            fact: The atomic fact text.
            claim: The original claim for context.

        Returns:
            Decontextualized fact text.
        """
        # Basic check: if fact starts with pronoun-like words and is short,
        # it might need context. But since we instruct the LLM to avoid pronouns,
        # this is mostly a safety check.
        pronouns = {"it", "this", "that", "they", "he", "she", "its", "their"}
        first_word = fact.split()[0].lower() if fact.split() else ""

        if first_word in pronouns:
            # Log but don't modify - trust LLM to have context
            logger.debug(
                "FACT_MAY_NEED_CONTEXT",
                fact=truncate(fact, 50),
                starts_with=first_word,
            )

        return fact

    def _get_model_tier(self, tier_name: str) -> ModelTier:
        """Convert tier name string to ModelTier enum."""
        tier_map = {
            "simple": ModelTier.SIMPLE,
            "analytical": ModelTier.ANALYTICAL,
            "complex": ModelTier.COMPLEX,
        }
        return tier_map.get(tier_name.lower(), ModelTier.SIMPLE)

    async def decompose_claims(
        self,
        claims: list[ClaimInfo],
        verdicts_to_process: set[str] | None = None,
    ) -> tuple[list[ClaimDecomposition], DecompositionMetrics]:
        """Decompose multiple claims into atomic facts.

        Args:
            claims: List of claims to decompose.
            verdicts_to_process: Set of verdicts to filter (default: unsupported, partial).

        Returns:
            Tuple of (decompositions, metrics).
        """
        span_name = citation_span_name(STAGE_7_ARE, "decompose_batch")

        with mlflow.start_span(name=span_name, span_type=SpanType.CHAIN) as span:
            if verdicts_to_process is None:
                verdicts_to_process = set(self.config.trigger_on_verdicts)

            metrics = DecompositionMetrics()
            decompositions: list[ClaimDecomposition] = []

            # Filter claims by verdict
            claims_to_process = [
                (i, c) for i, c in enumerate(claims)
                if c.verification_verdict in verdicts_to_process
            ]

            # Set input attributes
            span.set_attributes({
                ATTR_BATCH_TOTAL: len(claims),
                ATTR_BATCH_FILTERED: len(claims_to_process),
                "batch.verdicts": str(list(verdicts_to_process)),
            })

            logger.info(
                "DECOMPOSITION_BATCH_START",
                total_claims=len(claims),
                filtered_claims=len(claims_to_process),
                verdicts=list(verdicts_to_process),
            )

            for claim_index, claim in claims_to_process:
                decomposition = await self.decompose(claim, claim_index)
                decompositions.append(decomposition)

                # Update metrics
                metrics.total_claims_processed += 1
                metrics.total_atomic_facts += len(decomposition.atomic_facts)

                if len(decomposition.atomic_facts) == 1:
                    metrics.single_fact_claims += 1
                else:
                    metrics.multi_fact_claims += 1

            metrics.compute_avg()

            # Set output attributes
            span.set_attributes({
                "metrics.claims_processed": metrics.total_claims_processed,
                "metrics.total_facts": metrics.total_atomic_facts,
                "metrics.avg_facts_per_claim": round(metrics.avg_facts_per_claim, 2),
                "metrics.single_fact_claims": metrics.single_fact_claims,
                "metrics.multi_fact_claims": metrics.multi_fact_claims,
            })

            logger.info(
                "DECOMPOSITION_BATCH_COMPLETE",
                claims_processed=metrics.total_claims_processed,
                total_facts=metrics.total_atomic_facts,
                avg_facts_per_claim=f"{metrics.avg_facts_per_claim:.1f}",
                single_fact=metrics.single_fact_claims,
                multi_fact=metrics.multi_fact_claims,
            )

            return decompositions, metrics
