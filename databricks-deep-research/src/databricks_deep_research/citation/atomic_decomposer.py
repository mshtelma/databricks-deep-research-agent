"""Atomic Fact Decomposer for ARE-style verification.

FActScore-style atomic fact decomposition for ARE (Atomic fact
decomposition-based Retrieval and Editing) verification.

References:
- FActScore: https://arxiv.org/abs/2305.14251 (EMNLP 2023)
- SAFE: https://arxiv.org/abs/2403.18802 (Google DeepMind)
- ARE: https://arxiv.org/abs/2410.16708

Breaks complex claims into independent, self-contained atomic facts
that can be verified individually.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from databricks_deep_research.citation.types import ClaimInfo, RankedEvidence
from databricks_deep_research.llm.client import FrameworkLLMClient, ModelTier

logger = logging.getLogger(__name__)

DEFAULT_DECOMPOSITION_BATCH_SIZE = 5


def _truncate(text: str, max_len: int = 200) -> str:
    """Truncate *text* for log messages."""
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ATOMIC_DECOMPOSITION_PROMPT = """You are decomposing a claim into atomic facts.

## Claim to Decompose
"{claim_text}"

## Instructions
Break this claim into independent, self-contained atomic facts.

Rules for atomic facts:
1. Each fact should be a single, simple statement
2. Each fact should be independently verifiable
3. Replace all pronouns with explicit references
4. Each fact should make sense without the original claim
5. Do NOT add information not present in the original claim
6. Do NOT duplicate facts - each should be unique

## Examples

Input: "OpenAI released GPT-4 in March 2023, which scored 90% on the bar exam."
Output:
1. "OpenAI released GPT-4."
2. "GPT-4 was released in March 2023."
3. "GPT-4 scored 90% on the bar exam."

Input: "Tesla sold 500,000 vehicles in Q3 2024 and became the most valuable automaker."
Output:
1. "Tesla sold 500,000 vehicles in Q3 2024."
2. "Tesla became the most valuable automaker."

Input: "The study found that 75% of participants improved."
Output:
1. "A study was conducted."
2. "75% of participants in the study improved."

## Response Format (JSON)
```json
{{
  "atomic_facts": ["fact 1", "fact 2", ...],
  "reasoning": "Brief explanation of how you decomposed the claim"
}}
```

Decompose the claim into atomic facts:"""


BATCH_DECOMPOSITION_PROMPT = """Decompose each claim into atomic, independently verifiable facts.

## Claims to Decompose
{claims_section}

## Instructions
For EACH claim:
1. Break into independent, self-contained atomic facts
2. Each fact should be verifiable on its own
3. Replace all pronouns with explicit references
4. Do NOT add information not present in the original claim
5. Include claim_index to identify which claim each decomposition belongs to

## Rules for Atomic Facts
- Single, simple statement
- Self-contained (no pronouns)
- Independently verifiable
- 3-7 facts per claim (typically)

## Response Format (JSON)
```json
{{
  "decompositions": [
    {{
      "claim_index": 0,
      "atomic_facts": ["Fact 1 about the claim", "Fact 2 about the claim", ...],
      "reasoning": "Brief explanation"
    }},
    ...
  ]
}}
```

Decompose all claims:"""


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class EvidenceSource(StrEnum):
    """Source of evidence supporting an atomic fact."""

    INTERNAL = "internal"
    EXTERNAL = "external"
    NONE = "none"


@dataclass
class AtomicFact:
    """Single atomic, self-contained fact decomposed from a claim."""

    fact_text: str
    fact_index: int
    parent_claim_id: int

    # Verification results (populated later)
    is_verified: bool = False
    evidence: RankedEvidence | None = None
    evidence_source: EvidenceSource = EvidenceSource.NONE
    entailment_score: float = 0.0
    search_queries: list[str] = field(default_factory=list)
    assigned_citation_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
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
                "source_url": self.evidence.source_url,
                "quote_text": _truncate(self.evidence.quote_text, 200),
            } if self.evidence else None,
        }


@dataclass
class ClaimDecomposition:
    """Result of decomposing a claim into atomic facts."""

    original_claim: ClaimInfo
    atomic_facts: list[AtomicFact]
    decomposition_reasoning: str = ""
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
    """Result of revising a claim based on atomic fact verification."""

    original_claim: str
    revised_claim: str
    revision_type: Literal["fully_verified", "partially_softened", "fully_softened"]
    original_position_start: int
    original_position_end: int
    decomposition: ClaimDecomposition
    verified_facts: list[AtomicFact] = field(default_factory=list)
    softened_facts: list[AtomicFact] = field(default_factory=list)
    new_citations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
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
    """Aggregate metrics for the atomic decomposition stage."""

    total_claims_processed: int = 0
    total_atomic_facts: int = 0
    single_fact_claims: int = 0
    multi_fact_claims: int = 0
    decomposition_failures: int = 0
    avg_facts_per_claim: float = 0.0

    def compute_avg(self) -> None:
        if self.total_claims_processed > 0:
            self.avg_facts_per_claim = self.total_atomic_facts / self.total_claims_processed


# ---------------------------------------------------------------------------
# Pydantic structured-output schemas
# ---------------------------------------------------------------------------


class AtomicDecompositionOutput(BaseModel):
    """Structured output from single-claim decomposition."""

    atomic_facts: list[str] = Field(default_factory=list)
    reasoning: str = Field(default="")


class BatchDecompositionItem(BaseModel):
    """Single claim result within a batch decomposition."""

    claim_index: int = Field(description="0-based index in input batch")
    atomic_facts: list[str] = Field(default_factory=list, max_length=7)
    reasoning: str = Field(default="", max_length=200)


class BatchDecompositionOutput(BaseModel):
    """Structured output for batched decomposition."""

    decompositions: list[BatchDecompositionItem]


# ---------------------------------------------------------------------------
# AtomicDecomposer
# ---------------------------------------------------------------------------


class AtomicDecomposer:
    """Decomposes claims into atomic, self-contained facts (FActScore-style).

    First step in the ARE pattern: decompose -> retrieve -> edit.
    """

    def __init__(
        self,
        llm: FrameworkLLMClient,
        *,
        decomposition_timeout_seconds: float = 30.0,
        max_atomic_facts_per_claim: int = 7,
        decomposition_batch_size: int = DEFAULT_DECOMPOSITION_BATCH_SIZE,
        trigger_on_verdicts: list[str] | None = None,
        decomposition_tier: str = "simple",
    ) -> None:
        self.llm = llm
        self.decomposition_timeout_seconds = decomposition_timeout_seconds
        self.max_atomic_facts_per_claim = max_atomic_facts_per_claim
        self.decomposition_batch_size = decomposition_batch_size
        self.trigger_on_verdicts: list[str] = (
            trigger_on_verdicts if trigger_on_verdicts is not None
            else ["unsupported", "partial"]
        )
        self.decomposition_tier = decomposition_tier

    # -- single claim -------------------------------------------------------

    async def decompose(
        self,
        claim: ClaimInfo,
        claim_index: int,
        _context: str | None = None,
    ) -> ClaimDecomposition:
        """Decompose *claim* into atomic facts.

        Short claims (<=8 words) are returned as-is.  On timeout or error
        the claim is wrapped as a single atomic fact (fallback).
        """
        claim_text = claim.claim_text

        if len(claim_text.split()) <= 8:
            logger.debug("DECOMPOSITION_SKIP_SHORT claim=%s", _truncate(claim_text, 50))
            return self._atomic_passthrough(claim, claim_index)

        try:
            prompt = ATOMIC_DECOMPOSITION_PROMPT.format(claim_text=claim_text)
            tier = _resolve_tier(self.decomposition_tier)

            response = await asyncio.wait_for(
                self.llm.complete(
                    messages=[{"role": "user", "content": prompt}],
                    tier=tier,
                    structured_output=AtomicDecompositionOutput,
                ),
                timeout=self.decomposition_timeout_seconds,
            )

            if not response.structured:
                logger.warning("DECOMPOSITION_NO_STRUCTURED claim=%s", _truncate(claim_text, 50))
                return self._fallback_decomposition(claim, claim_index)

            output: AtomicDecompositionOutput = response.structured
            atomic_facts = self._build_facts(output.atomic_facts, claim_text, claim_index)

            logger.info("CLAIM_DECOMPOSED claim=%s facts=%d", _truncate(claim_text, 50), len(atomic_facts))
            return ClaimDecomposition(
                original_claim=claim,
                atomic_facts=atomic_facts,
                decomposition_reasoning=output.reasoning,
            )

        except TimeoutError:
            logger.warning("DECOMPOSITION_TIMEOUT claim=%s", _truncate(claim_text, 50))
            return self._fallback_decomposition(claim, claim_index)
        except Exception as e:
            logger.error("DECOMPOSITION_ERROR claim=%s error=%s", _truncate(claim_text, 50), str(e)[:100])
            return self._fallback_decomposition(claim, claim_index)

    # -- multi-claim (sequential) -------------------------------------------

    async def decompose_claims(
        self,
        claims: list[ClaimInfo],
        verdicts_to_process: set[str] | None = None,
    ) -> tuple[list[ClaimDecomposition], DecompositionMetrics]:
        """Decompose multiple claims, filtering by verification verdict."""
        if verdicts_to_process is None:
            verdicts_to_process = set(self.trigger_on_verdicts)

        metrics = DecompositionMetrics()
        decompositions: list[ClaimDecomposition] = []

        to_process = [
            (i, c) for i, c in enumerate(claims)
            if c.verification_verdict in verdicts_to_process
        ]

        logger.info("DECOMPOSITION_START total=%d filtered=%d", len(claims), len(to_process))

        for claim_index, claim in to_process:
            decomp = await self.decompose(claim, claim_index)
            decompositions.append(decomp)
            metrics.total_claims_processed += 1
            metrics.total_atomic_facts += len(decomp.atomic_facts)
            if len(decomp.atomic_facts) == 1:
                metrics.single_fact_claims += 1
            else:
                metrics.multi_fact_claims += 1

        metrics.compute_avg()
        logger.info(
            "DECOMPOSITION_COMPLETE claims=%d facts=%d avg=%.1f",
            metrics.total_claims_processed, metrics.total_atomic_facts, metrics.avg_facts_per_claim,
        )
        return decompositions, metrics

    # -- batch decomposition (token-optimized) ------------------------------

    async def batch_decompose(
        self,
        claims: list[tuple[int, ClaimInfo]],
        batch_size: int | None = None,
    ) -> list[ClaimDecomposition]:
        """Decompose claims with batched LLM calls (multiple claims per call).

        Falls back to sequential ``decompose()`` on per-batch failure.
        """
        if not claims:
            return []

        bs = batch_size or self.decomposition_batch_size
        results: list[ClaimDecomposition | None] = [None] * len(claims)

        batches = [list(range(i, min(i + bs, len(claims)))) for i in range(0, len(claims), bs)]
        logger.info("BATCH_DECOMPOSE_START total=%d batches=%d", len(claims), len(batches))

        for batch_num, indices in enumerate(batches):
            batch_claims = [claims[i] for i in indices]
            try:
                batch_results = await self._process_batch(batch_claims)
                for j, idx in enumerate(indices):
                    if j < len(batch_results):
                        results[idx] = batch_results[j]
                    else:
                        ci, cl = claims[idx]
                        results[idx] = self._fallback_decomposition(cl, ci)
            except Exception as e:
                logger.warning("BATCH_ERROR batch=%d error=%s", batch_num, str(e)[:100])
                for idx in indices:
                    ci, cl = claims[idx]
                    results[idx] = await self.decompose(cl, ci)

        # Fill remaining Nones
        for i, r in enumerate(results):
            if r is None:
                ci, cl = claims[i]
                results[i] = self._fallback_decomposition(cl, ci)

        final = [r for r in results if r is not None]
        logger.info("BATCH_DECOMPOSE_DONE results=%d facts=%d", len(final), sum(len(r.atomic_facts) for r in final))
        return final

    # -- private helpers ----------------------------------------------------

    def _atomic_passthrough(self, claim: ClaimInfo, claim_index: int) -> ClaimDecomposition:
        """Return claim as a single atomic fact (already atomic)."""
        return ClaimDecomposition(
            original_claim=claim,
            atomic_facts=[AtomicFact(fact_text=claim.claim_text, fact_index=0, parent_claim_id=claim_index)],
            decomposition_reasoning="Claim is short enough to be atomic",
        )

    def _fallback_decomposition(self, claim: ClaimInfo, claim_index: int) -> ClaimDecomposition:
        """Wrap the whole claim as a single atomic fact (LLM failure fallback)."""
        return ClaimDecomposition(
            original_claim=claim,
            atomic_facts=[AtomicFact(fact_text=claim.claim_text, fact_index=0, parent_claim_id=claim_index)],
            decomposition_reasoning="Fallback: treating entire claim as single atomic fact",
        )

    def _build_facts(self, raw_facts: list[str], claim_text: str, claim_index: int) -> list[AtomicFact]:
        """Deduplicate, decontextualize, and cap a list of raw fact strings."""
        facts: list[AtomicFact] = []
        seen: set[str] = set()
        for i, text in enumerate(raw_facts):
            norm = text.strip().lower()
            if not text.strip() or norm in seen:
                continue
            seen.add(norm)
            facts.append(AtomicFact(
                fact_text=_decontextualize(text.strip(), claim_text),
                fact_index=i,
                parent_claim_id=claim_index,
            ))
        if len(facts) > self.max_atomic_facts_per_claim:
            facts = facts[: self.max_atomic_facts_per_claim]
        if not facts:
            facts = [AtomicFact(fact_text=claim_text, fact_index=0, parent_claim_id=claim_index)]
        return facts

    def _format_claims_for_batch(self, claims: list[tuple[int, ClaimInfo]]) -> str:
        parts: list[str] = []
        for i, (idx, claim) in enumerate(claims):
            parts.append(f'### Claim {i} (original index: {idx})\n"{claim.claim_text}"\n')
        return "\n".join(parts)

    async def _process_batch(self, claims: list[tuple[int, ClaimInfo]]) -> list[ClaimDecomposition]:
        """Process a single batch of claims via one LLM call."""
        if not claims:
            return []

        short: dict[int, ClaimDecomposition] = {}
        to_llm: list[tuple[int, int, ClaimInfo]] = []

        for bidx, (cidx, c) in enumerate(claims):
            if len(c.claim_text.split()) <= 8:
                short[bidx] = self._atomic_passthrough(c, cidx)
            else:
                to_llm.append((bidx, cidx, c))

        if not to_llm:
            return [short[i] for i in range(len(claims))]

        prompt_claims = [(cidx, c) for _, cidx, c in to_llm]
        prompt = BATCH_DECOMPOSITION_PROMPT.format(claims_section=self._format_claims_for_batch(prompt_claims))
        tier = _resolve_tier(self.decomposition_tier)

        response = await asyncio.wait_for(
            self.llm.complete(
                messages=[{"role": "user", "content": prompt}],
                tier=tier,
                structured_output=BatchDecompositionOutput,
            ),
            timeout=self.decomposition_timeout_seconds * len(to_llm),
        )

        llm_map: dict[int, ClaimDecomposition] = {}
        if response.structured:
            llm_map = self._parse_structured(response.structured, to_llm)
        else:
            llm_map = self._parse_content(response.content, to_llm)

        result: list[ClaimDecomposition] = []
        for bidx in range(len(claims)):
            if bidx in short:
                result.append(short[bidx])
            elif bidx in llm_map:
                result.append(llm_map[bidx])
            else:
                cidx, c = claims[bidx]
                result.append(self._fallback_decomposition(c, cidx))
        return result

    def _parse_structured(
        self,
        output: BatchDecompositionOutput,
        to_llm: list[tuple[int, int, ClaimInfo]],
    ) -> dict[int, ClaimDecomposition]:
        """Parse structured batch output into ``{batch_idx: decomposition}``."""
        idx_map = dict(enumerate(to_llm))
        results: dict[int, ClaimDecomposition] = {}

        for item in output.decompositions:
            if item.claim_index not in idx_map:
                logger.warning("BATCH_INDEX_ERROR idx=%d", item.claim_index)
                continue
            bidx, cidx, claim = idx_map[item.claim_index]
            facts = self._build_facts(item.atomic_facts, claim.claim_text, cidx)
            results[bidx] = ClaimDecomposition(
                original_claim=claim, atomic_facts=facts, decomposition_reasoning=item.reasoning,
            )
        return results

    def _parse_content(
        self,
        content: str,
        to_llm: list[tuple[int, int, ClaimInfo]],
    ) -> dict[int, ClaimDecomposition]:
        """Fallback JSON parser when structured output is unavailable."""
        results: dict[int, ClaimDecomposition] = {}
        try:
            m = re.search(r"\{[\s\S]*\}", content)
            if not m:
                return results
            data = json.loads(m.group())
            if not isinstance(data.get("decompositions"), list):
                return results

            idx_map = dict(enumerate(to_llm))
            for item in data["decompositions"]:
                pidx = item.get("claim_index", -1)
                if pidx not in idx_map:
                    continue
                bidx, cidx, claim = idx_map[pidx]
                raw = item.get("atomic_facts", [])
                facts = self._build_facts(raw, claim.claim_text, cidx)
                results[bidx] = ClaimDecomposition(
                    original_claim=claim, atomic_facts=facts, decomposition_reasoning=item.get("reasoning", ""),
                )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("BATCH_PARSE_FAIL error=%s", e)
        return results


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _resolve_tier(tier_name: str) -> ModelTier:
    """Convert tier name string to ``ModelTier`` enum."""
    return {"simple": ModelTier.simple, "analytical": ModelTier.analytical, "complex": ModelTier.complex}.get(
        tier_name.lower(), ModelTier.simple
    )


def _decontextualize(fact: str, _claim: str) -> str:
    """Lightweight check that *fact* is self-contained.

    The LLM prompt already instructs to avoid pronouns; this logs a warning
    if the heuristic detects a leading pronoun.
    """
    pronouns = {"it", "this", "that", "they", "he", "she", "its", "their"}
    first = fact.split()[0].lower() if fact.split() else ""
    if first in pronouns:
        logger.debug("FACT_MAY_NEED_CONTEXT fact=%s", _truncate(fact, 50))
    return fact
