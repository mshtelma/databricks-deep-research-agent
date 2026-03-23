"""Stage 7: ARE-style Verification Retrieval & Claim Revision.

Implements the ARE (Atomic fact decomposition-based Retrieval and Editing)
pattern for verifying and revising unsupported/partial claims.

Pipeline:
1. Filter claims (verdict = unsupported | partial)
2. Decompose each claim into atomic facts
3. For each atomic fact:
   a. Search internal evidence pool (BM25)
   b. If not found: External web search + crawl
   c. NLI entailment check
   d. Mark: verified or unverified
4. Reconstruct claim with verified/softened facts
5. Apply revision to report (position-based replacement)

Scientific basis:
- ARE: https://arxiv.org/abs/2410.16708
- FActScore: https://arxiv.org/abs/2305.14251
- SAFE: https://arxiv.org/abs/2403.18802

Token Optimization Features:
- Batch entailment: Process multiple fact-evidence pairs in single LLM call
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from databricks_deep_research.citation.atomic_decomposer import (
    AtomicDecomposer,
    AtomicFact,
    ClaimDecomposition,
    ClaimRevision,
    EvidenceSource,
)
from databricks_deep_research.citation.config import SofteningStrategy
from databricks_deep_research.citation.types import ClaimInfo, RankedEvidence
from databricks_deep_research.citation.utils import truncate as _truncate
from databricks_deep_research.llm.client import FrameworkLLMClient, ModelTier

logger = logging.getLogger(__name__)


# =============================================================================
# Prompts (inlined -- no external prompt imports)
# =============================================================================

ENTAILMENT_CHECK_PROMPT = """\
You are checking if evidence supports a factual claim.

## Fact to Verify
"{fact_text}"

## Evidence
Source: {source_url}
Quote: "{evidence_quote}"

## Task
Determine if the evidence ENTAILS (supports) the fact.

## Entailment Scoring
- 1.0: Evidence directly and explicitly states the fact
- 0.8: Evidence strongly implies the fact with minimal inference
- 0.6: Evidence partially supports the fact (some aspects covered)
- 0.4: Evidence is tangentially related but doesn't confirm
- 0.2: Evidence is about the same topic but doesn't support the fact
- 0.0: Evidence contradicts the fact

## Key Considerations
- Numbers must match exactly (or be close enough to be rounding)
- Entity names must match
- Time periods must align
- Causal claims need explicit support

## Response Format (JSON)
```json
{{
  "entails": true/false,
  "score": 0.0-1.0,
  "reasoning": "Brief explanation of the assessment",
  "key_match": "Quote the specific part that matches or conflicts"
}}
```

Assess whether the evidence entails the fact:"""


EVIDENCE_EXTRACTION_PROMPT = """\
You are extracting evidence to verify a fact from web content.

## Fact to Verify
"{fact_text}"

## Source Content
URL: {source_url}
Title: {source_title}

Content:
{source_content}

## Task
Find the MOST RELEVANT quote (1-3 sentences) that could verify or refute the fact.

## Guidelines
1. Look for explicit statements about the fact
2. Prioritize quotes with specific numbers, dates, or names
3. Include enough context for the quote to make sense
4. If multiple relevant quotes exist, choose the most authoritative
5. If no relevant content exists, indicate this clearly

## Response Format (JSON)
```json
{{
  "quote_text": "Exact quote from source (or null if none found)",
  "relevance_score": 0.0-1.0,
  "has_numeric_content": true/false,
  "section_heading": "Section name if available (or null)",
  "reasoning": "Why this quote is (or is not) relevant"
}}
```

Extract the most relevant evidence:"""


VERIFICATION_QUERY_PROMPT = """\
You are generating a search query to verify a factual claim.

## Fact to Verify
"{fact_text}"

## Original Research Query (for context)
"{research_query}"

## Previous Query Attempts (if any)
{previous_queries}

## Task
Generate a specific search query to find authoritative evidence that would:
1. Directly support OR refute this fact
2. Come from reliable sources (news, academic, official)
3. Contain the specific details mentioned in the fact

## Guidelines for Good Queries
- Focus on the CORE FACT being claimed
- Include key entities (names, organizations, dates)
- Include specific numbers or metrics if present
- Avoid using the exact wording (find independent sources)
- If this is a retry, try different phrasings or synonyms

{reformulation_guidance}

## Response Format (JSON)
```json
{{
  "query": "your search query here",
  "reasoning": "why this query will find relevant evidence",
  "search_strategy": "what type of source you expect to find"
}}
```

Generate the search query:"""


REFORMULATION_GUIDANCE = """
## REFORMULATION REQUIRED
Previous queries did not find supporting evidence. Try:
- Different synonyms or alternative phrasings
- Broader scope (e.g., "Tesla vehicle sales" instead of "Tesla Q3 2024 sales")
- Narrower scope (more specific entity or time period)
- Alternative source types (official reports, press releases, news articles)
- Different language or terminology used in the industry
"""


CLAIM_RECONSTRUCTION_PROMPT = """\
You are reconstructing a claim based on verification results.

## Original Claim
"{original_claim}"

## Atomic Facts with Verification Status
{facts_with_status}

## Instructions
Reconstruct the claim following these rules:

### For VERIFIED facts (entailment_score >= 0.6):
- Keep the fact as-is
- Add [Citation] marker if new evidence was found
- Example: "Tesla sold 500K vehicles [Reuters]"

### For UNVERIFIED facts (entailment_score < 0.6):
- Add hedging language to indicate uncertainty
- Options:
  - "reportedly" - "Tesla reportedly became the most valuable..."
  - "according to some sources" - "According to some sources, Tesla..."
  - "it is claimed that" - "It is claimed that Tesla..."
  - "allegedly" - "Tesla allegedly became..."
- Do NOT remove the fact entirely - keep the information but mark uncertainty

### General Guidelines:
- Maintain natural sentence flow and readability
- Preserve the original claim's structure where possible
- Combine multiple atomic facts back into coherent sentences
- If ALL facts are unverified, soften the entire claim
- If ALL facts are verified, the claim can stand as-is with citations

## Example
Original: "Tesla sold 500,000 vehicles in Q3 2024 and became the most valuable automaker."

Facts:
1. "Tesla sold 500,000 vehicles in Q3 2024" - VERIFIED [Reuters]
2. "Tesla became the most valuable automaker" - UNVERIFIED

Output: "Tesla sold 500,000 vehicles in Q3 2024 [Reuters], though its claim to \
being the world's most valuable automaker remains disputed."

## Response Format
Return ONLY the reconstructed claim text (no JSON, no explanation).

Reconstruct the claim:"""


CLAIM_SOFTENING_HEDGE_PROMPT = """\
You are softening a claim that lacks supporting evidence.

## Original Claim
"{claim_text}"

## Task
Rewrite this claim using HEDGING language to indicate uncertainty.

## Hedging Techniques
- "reportedly" - indicates unconfirmed reports
- "allegedly" - indicates unverified allegations
- "according to some sources" - indicates disputed information
- "it is believed that" - indicates unconfirmed belief
- "may have" / "might have" - indicates possibility
- "appears to" - indicates uncertain observation

## Guidelines
- Maintain the claim's informational value
- Make clear this is not definitively established
- Do NOT make up false sources or citations
- Keep similar length to original
- Preserve the core meaning while adding uncertainty

## Response
Return ONLY the softened claim text:"""


CLAIM_SOFTENING_QUALIFY_PROMPT = """\
You are softening a claim that lacks supporting evidence.

## Original Claim
"{claim_text}"

## Task
Rewrite this claim using QUALIFYING phrases to indicate uncertainty.

## Qualifying Techniques
- "Some evidence suggests that..." - partial support
- "It is believed that..." - unconfirmed belief
- "There are indications that..." - tentative evidence
- "Early reports indicate that..." - unconfirmed reports
- "Preliminary findings suggest..." - initial evidence

## Guidelines
- Start with a qualifying phrase
- Maintain the claim's informational value
- Keep similar length to original
- Preserve the core meaning

## Response
Return ONLY the qualified claim text:"""


CLAIM_SOFTENING_PARENTHETICAL_PROMPT = """\
You are softening a claim that lacks supporting evidence.

## Original Claim
"{claim_text}"

## Task
Add PARENTHETICAL markers to indicate the claim needs verification.

## Parenthetical Options
- "(unverified)" - at the end of claim
- "(needs citation)" - for missing source
- "(disputed)" - for contested claims
- "(approximate)" - for uncertain numbers

## Guidelines
- Keep the original claim intact
- Add one or more parenthetical markers
- Use sparingly - don't overload with markers
- Preserve readability

## Response
Return ONLY the claim with parenthetical markers:"""


BATCH_ENTAILMENT_PROMPT = """\
Check if each fact is entailed by its evidence.

## Facts to Check
{facts_section}

## Instructions
For EACH fact above:
1. Determine if the evidence ENTAILS (supports) the fact
2. Provide a confidence score (0.0-1.0)
3. Include fact_index to identify which fact each result belongs to

## Entailment Scoring
- 1.0: Evidence directly and explicitly states the fact
- 0.8: Evidence strongly implies the fact
- 0.6: Evidence partially supports the fact
- 0.4: Evidence is tangentially related
- 0.2: Evidence is about the same topic but doesn't support
- 0.0: Evidence contradicts the fact

## Response Format (JSON)
```json
{{
  "results": [
    {{
      "fact_index": 0,
      "entails": true,
      "score": 0.8,
      "reasoning": "Brief explanation",
      "supporting_quote": "Relevant quote from evidence"
    }},
    ...
  ]
}}
```

CRITICAL: Return one result per fact. Include fact_index to handle any reordering.

Check all facts:"""


# =============================================================================
# Pydantic Models for Structured LLM Output
# =============================================================================


class EntailmentCheckOutput(BaseModel):
    """Output from entailment check LLM call."""

    entails: bool = Field(description="Whether evidence entails the fact")
    score: float = Field(ge=0.0, le=1.0, description="Entailment confidence score")
    reasoning: str = Field(default="", description="Brief explanation")
    key_match: str = Field(default="", description="Quote that matches or conflicts")


class EvidenceExtractionOutput(BaseModel):
    """Output from evidence extraction LLM call."""

    quote_text: str | None = Field(
        default=None, description="Exact quote from source"
    )
    relevance_score: float = Field(
        ge=0.0, le=1.0, default=0.0, description="Relevance to fact"
    )
    has_numeric_content: bool = Field(default=False)
    section_heading: str | None = Field(default=None)
    reasoning: str = Field(default="")


class VerificationQueryOutput(BaseModel):
    """Output from verification query generation LLM call."""

    query: str = Field(description="Search query to verify fact")
    reasoning: str = Field(default="")
    search_strategy: str = Field(default="")


class BatchEntailmentItem(BaseModel):
    """Single entailment check result in a batch."""

    fact_index: int = Field(description="0-based index of fact in input batch")
    entails: bool = Field(description="Whether evidence entails the fact")
    score: float = Field(ge=0.0, le=1.0, description="Entailment confidence score")
    reasoning: str = Field(default="", max_length=200)
    supporting_quote: str | None = Field(default=None, max_length=300)


class BatchEntailmentOutput(BaseModel):
    """Output for batched entailment checks."""

    results: list[BatchEntailmentItem] = Field(
        description="Entailment results in same order as input facts"
    )


# Default batch size for entailment checks
DEFAULT_ENTAILMENT_BATCH_SIZE = 10


# =============================================================================
# Metrics and Events
# =============================================================================


@dataclass
class VerificationRetrievalMetrics:
    """Aggregate metrics for Stage 7 verification retrieval."""

    total_claims_processed: int = 0
    total_atomic_facts: int = 0
    facts_verified: int = 0
    facts_softened: int = 0
    claims_fully_verified: int = 0
    claims_partially_softened: int = 0
    claims_fully_softened: int = 0
    internal_searches: int = 0
    external_searches: int = 0
    external_crawls: int = 0
    entailment_checks: int = 0
    new_sources_added: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_claims_processed": self.total_claims_processed,
            "total_atomic_facts": self.total_atomic_facts,
            "facts_verified": self.facts_verified,
            "facts_softened": self.facts_softened,
            "claims_fully_verified": self.claims_fully_verified,
            "claims_partially_softened": self.claims_partially_softened,
            "claims_fully_softened": self.claims_fully_softened,
            "internal_searches": self.internal_searches,
            "external_searches": self.external_searches,
            "external_crawls": self.external_crawls,
            "entailment_checks": self.entailment_checks,
            "new_sources_added": self.new_sources_added,
        }


@dataclass
class VerificationEvent:
    """Event emitted during verification for progress updates."""

    event_type: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class NewExternalEvidence:
    """New evidence found by Stage 7 external search with pre-assigned citation key.

    The citation key is pre-assigned BEFORE reconstruction to ensure
    consistency between what the LLM outputs and what we register.
    """

    evidence: RankedEvidence
    citation_key: str
    fact_text: str
    source_url: str


# =============================================================================
# Internal Evidence Pool Searcher (BM25)
# =============================================================================


class InternalPoolSearcher:
    """Searches the internal evidence pool using BM25 scoring."""

    def __init__(self, evidence_pool: list[RankedEvidence]) -> None:
        self.evidence_pool = evidence_pool
        self._build_index()

    def _build_index(self) -> None:
        """Build inverted index for BM25 scoring."""
        self._doc_freq: dict[str, int] = {}
        self._doc_lengths: list[int] = []
        self._doc_terms: list[set[str]] = []

        for evidence in self.evidence_pool:
            terms = self._tokenize(evidence.quote_text)
            self._doc_terms.append(terms)
            self._doc_lengths.append(len(terms))
            for term in terms:
                self._doc_freq[term] = self._doc_freq.get(term, 0) + 1

        self._avg_doc_length = (
            sum(self._doc_lengths) / len(self._doc_lengths)
            if self._doc_lengths
            else 0
        )

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Tokenize text for BM25 scoring."""
        tokens = re.findall(r"\b[a-z0-9]+\b", text.lower())
        return {t for t in tokens if len(t) > 2}

    def _bm25_score(self, query_terms: set[str], doc_idx: int) -> float:
        """Calculate BM25 score for a document."""
        k1 = 1.2
        b = 0.75
        n = len(self.evidence_pool)

        score = 0.0
        doc_terms = self._doc_terms[doc_idx]
        doc_len = self._doc_lengths[doc_idx]

        for term in query_terms:
            if term in doc_terms:
                df = self._doc_freq.get(term, 0)
                idf = max(0, (n - df + 0.5) / (df + 0.5))
                tf = 1  # Binary TF
                norm = (1 - b) + b * (doc_len / max(self._avg_doc_length, 1))
                score += idf * ((tf * (k1 + 1)) / (tf + k1 * norm))

        return score

    def search(
        self,
        fact_text: str,
        threshold: float = 0.7,
        top_k: int = 3,
    ) -> list[tuple[RankedEvidence, float]]:
        """Search internal pool for evidence supporting a fact.

        Args:
            fact_text: The atomic fact to find evidence for.
            threshold: Minimum BM25 score threshold.
            top_k: Maximum number of results.

        Returns:
            List of ``(evidence, score)`` tuples sorted by score descending.
        """
        if not self.evidence_pool:
            return []

        query_terms = self._tokenize(fact_text)
        if not query_terms:
            return []

        scores: list[tuple[int, float]] = []
        for i in range(len(self.evidence_pool)):
            score = self._bm25_score(query_terms, i)
            if score >= threshold:
                scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        return [
            (self.evidence_pool[idx], score)
            for idx, score in scores[:top_k]
        ]


# =============================================================================
# Verification Retriever
# =============================================================================


class VerificationRetriever:
    """ARE-style verification and revision for failed claims.

    Orchestrates the full Stage 7 pipeline:
    1. Filter claims by verdict
    2. Decompose into atomic facts
    3. Verify each fact (internal pool -> external search)
    4. Reconstruct with verified/softened facts
    5. Apply revisions to report

    Constructor accepts framework-native dependencies rather than app-specific
    ones: ``FrameworkLLMClient`` for LLM calls, generic ``search_tool`` /
    ``crawl_tool`` callables for web interaction.
    """

    def __init__(
        self,
        llm: FrameworkLLMClient,
        search_tool: Any | None = None,
        crawl_tool: Any | None = None,
        *,
        # Trigger conditions
        trigger_on_verdicts: list[str] | None = None,
        # Decomposition config
        max_atomic_facts_per_claim: int = 5,
        decomposition_timeout_seconds: float = 10.0,
        # Search budget (per atomic fact)
        max_searches_per_fact: int = 2,
        max_external_urls_per_search: int = 3,
        # Entailment thresholds
        entailment_threshold: float = 0.6,
        internal_search_threshold: float = 0.7,
        # Reconstruction
        softening_strategy: SofteningStrategy = SofteningStrategy.HEDGE,
        # Timeouts
        search_timeout_seconds: float = 10.0,
        crawl_timeout_seconds: float = 15.0,
        # Model tiers
        decomposition_tier: str = "simple",
        entailment_tier: str = "simple",
        reconstruction_tier: str = "analytical",
        softening_tier: str = "simple",
    ) -> None:
        """Initialize the verification retriever.

        Args:
            llm: Framework LLM client.
            search_tool: Tool for web search (``ResearchTool`` or callable).
                Must accept ``arguments={"query": ...}, context=...`` and
                return a result with ``.content`` and ``.sources``.
            crawl_tool: Tool for web crawl (``ResearchTool`` or callable).
                Must accept ``arguments={"urls": ...}, context=...`` and
                return a result with ``.content``.
            trigger_on_verdicts: Verdicts that trigger processing
                (default: ``["unsupported", "partial"]``).
            max_atomic_facts_per_claim: Maximum facts per decomposed claim.
            decomposition_timeout_seconds: Timeout for decomposition LLM call.
            max_searches_per_fact: Max search attempts per atomic fact.
            max_external_urls_per_search: Max URLs to crawl per search.
            entailment_threshold: Minimum entailment score to accept evidence.
            internal_search_threshold: BM25 threshold for internal pool.
            softening_strategy: Strategy for softening unverified facts.
            search_timeout_seconds: Timeout per external search call.
            crawl_timeout_seconds: Timeout per crawl call.
            decomposition_tier: Model tier for decomposition.
            entailment_tier: Model tier for entailment checks.
            reconstruction_tier: Model tier for claim reconstruction.
            softening_tier: Model tier for claim softening.
        """
        self.llm = llm
        self.search_tool = search_tool
        self.crawl_tool = crawl_tool

        self.trigger_on_verdicts: set[str] = set(
            trigger_on_verdicts
            if trigger_on_verdicts is not None
            else ["unsupported", "partial"]
        )
        self.max_searches_per_fact = max_searches_per_fact
        self.max_external_urls_per_search = max_external_urls_per_search
        self.entailment_threshold = entailment_threshold
        self.internal_search_threshold = internal_search_threshold
        self.softening_strategy = softening_strategy
        self.search_timeout_seconds = search_timeout_seconds
        self.crawl_timeout_seconds = crawl_timeout_seconds
        self.decomposition_tier = decomposition_tier
        self.entailment_tier = entailment_tier
        self.reconstruction_tier = reconstruction_tier
        self.softening_tier = softening_tier

        self.decomposer = AtomicDecomposer(
            llm,
            decomposition_timeout_seconds=decomposition_timeout_seconds,
            max_atomic_facts_per_claim=max_atomic_facts_per_claim,
            decomposition_tier=decomposition_tier,
        )
        self.metrics = VerificationRetrievalMetrics()

        # Track new external evidence with pre-assigned citation keys
        self._new_external_evidence_with_keys: list[NewExternalEvidence] = []
        self._existing_citation_keys: set[str] = set()

    # =========================================================================
    # Main entry point
    # =========================================================================

    async def retrieve_and_revise(
        self,
        claims: list[ClaimInfo],
        evidence_pool: list[RankedEvidence],
        _report_content: str,
        research_query: str,
    ) -> AsyncGenerator[VerificationEvent | ClaimRevision, None]:
        """Main entry point for Stage 7 verification retrieval.

        Args:
            claims: List of claims from previous stages.
            evidence_pool: Pre-selected evidence from Stage 1.
            report_content: Current report content.
            research_query: Original research query for context.

        Yields:
            ``VerificationEvent`` for progress updates and ``ClaimRevision``
            for results.
        """
        # Reset per-run state
        self.metrics = VerificationRetrievalMetrics()
        self._new_external_evidence_with_keys = []
        self._existing_citation_keys = {
            self._extract_domain(ev.source_url)
            for ev in evidence_pool
            if ev.source_url
        }

        # Filter claims by verdict
        claims_to_process = self._filter_claims(claims)

        if not claims_to_process:
            logger.info("STAGE_7_SKIP reason='No claims to process'")
            yield VerificationEvent(
                "stage_7_skipped",
                {"reason": "No unsupported or partial claims"},
            )
            return

        yield VerificationEvent(
            "stage_7_started",
            {
                "total_claims": len(claims_to_process),
                "verdicts": list(self.trigger_on_verdicts),
            },
        )

        # Build internal BM25 searcher
        internal_searcher = InternalPoolSearcher(evidence_pool)

        # Process claims in REVERSE position order to avoid drift
        claims_to_process.sort(
            key=lambda x: x[1].position_start, reverse=True
        )

        for claim_index, claim in claims_to_process:
            self.metrics.total_claims_processed += 1

            yield VerificationEvent(
                "claim_verification_started",
                {
                    "claim_index": claim_index,
                    "claim_text": _truncate(claim.claim_text, 100),
                    "verdict": claim.verification_verdict,
                },
            )

            try:
                revision = await self._process_claim(
                    claim=claim,
                    claim_index=claim_index,
                    internal_searcher=internal_searcher,
                    research_query=research_query,
                )

                if revision.revision_type == "fully_verified":
                    self.metrics.claims_fully_verified += 1
                elif revision.revision_type == "partially_softened":
                    self.metrics.claims_partially_softened += 1
                else:
                    self.metrics.claims_fully_softened += 1

                yield revision

            except Exception as e:
                logger.error(
                    "CLAIM_VERIFICATION_ERROR claim_index=%d error=%s",
                    claim_index, str(e)[:100],
                )
                yield VerificationEvent(
                    "claim_verification_error",
                    {"claim_index": claim_index, "error": str(e)[:100]},
                )

        # Final summary
        yield VerificationEvent("stage_7_complete", self.metrics.to_dict())

        logger.info(
            "STAGE_7_COMPLETE claims=%d verified=%d softened=%d",
            self.metrics.total_claims_processed,
            self.metrics.facts_verified,
            self.metrics.facts_softened,
        )

    # =========================================================================
    # Claim filtering
    # =========================================================================

    def _filter_claims(
        self, claims: list[ClaimInfo],
    ) -> list[tuple[int, ClaimInfo]]:
        """Filter claims by verdict to process."""
        filtered = [
            (i, c) for i, c in enumerate(claims)
            if c.verification_verdict in self.trigger_on_verdicts
        ]
        logger.info(
            "CLAIMS_FILTERED total=%d filtered=%d verdicts=%s",
            len(claims), len(filtered), list(self.trigger_on_verdicts),
        )
        return filtered

    # =========================================================================
    # Single-claim processing (ARE pipeline)
    # =========================================================================

    async def _process_claim(
        self,
        claim: ClaimInfo,
        claim_index: int,
        internal_searcher: InternalPoolSearcher,
        research_query: str,
    ) -> ClaimRevision:
        """Process a single claim through the ARE pipeline."""
        # Step 1: Decompose claim into atomic facts
        decomposition = await self.decomposer.decompose(claim, claim_index)
        self.metrics.total_atomic_facts += len(decomposition.atomic_facts)

        logger.debug(
            "CLAIM_DECOMPOSED claim_index=%d fact_count=%d",
            claim_index, len(decomposition.atomic_facts),
        )

        # Step 2: Verify each atomic fact
        for fact in decomposition.atomic_facts:
            await self._verify_atomic_fact(
                fact=fact,
                claim_index=claim_index,
                internal_searcher=internal_searcher,
                research_query=research_query,
            )
            if fact.is_verified:
                self.metrics.facts_verified += 1
            else:
                self.metrics.facts_softened += 1

        # Update decomposition status
        decomposition.update_verification_status()

        # Step 3: Reconstruct claim
        revised_claim = await self._reconstruct_claim(decomposition, claim_index)

        # Determine revision type
        if decomposition.all_verified:
            revision_type: Literal[
                "fully_verified", "partially_softened", "fully_softened"
            ] = "fully_verified"
        elif decomposition.partial_verified:
            revision_type = "partially_softened"
        else:
            revision_type = "fully_softened"

        return ClaimRevision(
            original_claim=claim.claim_text,
            revised_claim=revised_claim,
            revision_type=revision_type,
            original_position_start=claim.position_start,
            original_position_end=claim.position_end,
            decomposition=decomposition,
            verified_facts=[
                f for f in decomposition.atomic_facts if f.is_verified
            ],
            softened_facts=[
                f for f in decomposition.atomic_facts if not f.is_verified
            ],
            new_citations=[
                f.evidence.source_url
                for f in decomposition.atomic_facts
                if f.is_verified
                and f.evidence
                and f.evidence_source == EvidenceSource.EXTERNAL
            ],
        )

    # =========================================================================
    # Atomic fact verification
    # =========================================================================

    async def _verify_atomic_fact(
        self,
        fact: AtomicFact,
        claim_index: int,
        internal_searcher: InternalPoolSearcher,
        research_query: str,
    ) -> None:
        """Verify a single atomic fact: internal pool first, then external."""
        # Step 1: Search internal evidence pool
        self.metrics.internal_searches += 1
        internal_results = internal_searcher.search(
            fact.fact_text, threshold=self.internal_search_threshold,
        )

        for evidence, _bm25_score in internal_results:
            entails, ent_score = await self._check_entailment(
                fact, evidence, claim_index,
            )
            self.metrics.entailment_checks += 1

            if entails and ent_score >= self.entailment_threshold:
                fact.is_verified = True
                fact.evidence = evidence
                fact.evidence_source = EvidenceSource.INTERNAL
                fact.entailment_score = ent_score
                logger.debug(
                    "FACT_VERIFIED_INTERNAL fact=%s score=%.2f",
                    _truncate(fact.fact_text, 50), ent_score,
                )
                return

        # Step 2: External search if internal didn't verify
        if not fact.is_verified and self.search_tool is not None:
            await self._search_external(
                fact=fact,
                claim_index=claim_index,
                research_query=research_query,
            )

        if not fact.is_verified:
            logger.debug(
                "FACT_UNVERIFIED fact=%s searches=%d",
                _truncate(fact.fact_text, 50), len(fact.search_queries),
            )

    # =========================================================================
    # External search
    # =========================================================================

    async def _search_external(
        self,
        fact: AtomicFact,
        claim_index: int,
        research_query: str,
    ) -> None:
        """Search external sources for evidence supporting a fact.

        Uses ``self.search_tool`` and optionally ``self.crawl_tool`` to
        find and verify evidence from the web.
        """
        if self.search_tool is None:
            return

        for search_attempt in range(self.max_searches_per_fact):
            # Generate search query
            query = await self._generate_search_query(
                fact=fact,
                research_query=research_query,
                previous_queries=fact.search_queries,
                is_reformulation=search_attempt > 0,
            )

            if not query:
                continue

            fact.search_queries.append(query)
            self.metrics.external_searches += 1

            try:
                # Execute web search via tool protocol
                search_result = await asyncio.wait_for(
                    self._execute_search(query),
                    timeout=self.search_timeout_seconds,
                )

                if not search_result:
                    continue

                # If crawl tool available, crawl top URLs for full content
                if self.crawl_tool is not None:
                    urls = [
                        s.get("url", "") for s in search_result
                        if s.get("url")
                    ][: self.max_external_urls_per_search]

                    if urls:
                        self.metrics.external_crawls += 1
                        crawl_results = await asyncio.wait_for(
                            self._execute_crawl(urls),
                            timeout=self.crawl_timeout_seconds,
                        )

                        for crawl_item in crawl_results:
                            url = crawl_item.get("url", "")
                            content = crawl_item.get("content", "")
                            title = crawl_item.get("title", "Unknown")

                            if not content:
                                continue

                            evidence = await self._extract_evidence(
                                fact=fact, url=url, title=title, content=content,
                            )
                            if not evidence:
                                continue

                            entails, ent_score = await self._check_entailment(
                                fact, evidence, claim_index,
                            )
                            self.metrics.entailment_checks += 1

                            if entails and ent_score >= self.entailment_threshold:
                                self._mark_fact_verified_external(
                                    fact, evidence, ent_score, url,
                                )
                                return
                else:
                    # No crawl tool -- use search snippets as evidence
                    for item in search_result:
                        snippet = item.get("snippet", "")
                        url = item.get("url", "")
                        if not snippet:
                            continue

                        evidence = RankedEvidence(
                            source_id=None,
                            source_url=url,
                            source_title=item.get("title"),
                            quote_text=snippet,
                            start_offset=None,
                            end_offset=None,
                            section_heading=None,
                            relevance_score=0.5,
                            has_numeric_content=False,
                            is_snippet_based=True,
                        )

                        entails, ent_score = await self._check_entailment(
                            fact, evidence, claim_index,
                        )
                        self.metrics.entailment_checks += 1

                        if entails and ent_score >= self.entailment_threshold:
                            self._mark_fact_verified_external(
                                fact, evidence, ent_score, url,
                            )
                            return

            except TimeoutError:
                logger.warning(
                    "EXTERNAL_SEARCH_TIMEOUT query=%s attempt=%d",
                    _truncate(query, 50), search_attempt + 1,
                )
            except Exception as e:
                logger.warning(
                    "EXTERNAL_SEARCH_ERROR query=%s error=%s",
                    _truncate(query, 50), str(e)[:100],
                )

    def _mark_fact_verified_external(
        self,
        fact: AtomicFact,
        evidence: RankedEvidence,
        ent_score: float,
        source_url: str,
    ) -> None:
        """Mark a fact as verified via external evidence and assign citation key."""
        fact.is_verified = True
        fact.evidence = evidence
        fact.evidence_source = EvidenceSource.EXTERNAL
        fact.entailment_score = ent_score
        self.metrics.new_sources_added += 1

        citation_key = self._generate_citation_key(source_url)
        fact.assigned_citation_key = citation_key

        self._new_external_evidence_with_keys.append(
            NewExternalEvidence(
                evidence=evidence,
                citation_key=citation_key,
                fact_text=fact.fact_text,
                source_url=source_url,
            )
        )

        logger.debug(
            "FACT_VERIFIED_EXTERNAL fact=%s source=%s score=%.2f key=%s",
            _truncate(fact.fact_text, 50), source_url, ent_score, citation_key,
        )

    async def _execute_search(self, query: str) -> list[dict[str, Any]]:
        """Execute web search via the search tool.

        Returns a list of dicts with ``url``, ``title``, ``snippet`` keys.
        """
        if self.search_tool is None:
            return []

        from databricks_deep_research.tools.protocol import ToolContext

        result = await self.search_tool.execute(
            {"query": query},
            ToolContext(query=query),
        )

        if not result.success:
            return []

        # Prefer structured sources if available
        if result.sources:
            return [
                {"url": s.url, "title": s.title, "snippet": s.snippet}
                for s in result.sources
            ]

        # Fallback: try to parse content as JSON list
        try:
            data = json.loads(result.content)
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, TypeError):
            pass

        return []

    async def _execute_crawl(
        self, urls: list[str],
    ) -> list[dict[str, Any]]:
        """Execute web crawl via the crawl tool.

        Returns a list of dicts with ``url``, ``title``, ``content`` keys.
        """
        if self.crawl_tool is None:
            return []

        from databricks_deep_research.tools.protocol import ToolContext

        result = await self.crawl_tool.execute(
            {"urls": urls},
            ToolContext(),
        )

        if not result.success:
            return []

        # The crawl tool typically returns structured data
        if result.data and "results" in result.data:
            return list(result.data["results"])

        # Fallback: try to parse content
        try:
            data = json.loads(result.content)
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, TypeError):
            pass

        # Single-page fallback
        if result.content and urls:
            return [{"url": urls[0], "title": "", "content": result.content}]

        return []

    # =========================================================================
    # Search query generation
    # =========================================================================

    async def _generate_search_query(
        self,
        fact: AtomicFact,
        research_query: str,
        previous_queries: list[str],
        is_reformulation: bool = False,
    ) -> str | None:
        """Generate a search query to verify a fact."""
        try:
            reformulation_text = REFORMULATION_GUIDANCE if is_reformulation else ""
            previous_text = (
                "\n".join(f"- {q}" for q in previous_queries)
                if previous_queries
                else "None"
            )

            prompt = VERIFICATION_QUERY_PROMPT.format(
                fact_text=fact.fact_text,
                research_query=research_query,
                previous_queries=previous_text,
                reformulation_guidance=reformulation_text,
            )

            tier = self._resolve_tier(self.decomposition_tier)

            response = await self.llm.complete(
                messages=[{"role": "user", "content": prompt}],
                tier=tier,
                structured_output=VerificationQueryOutput,
            )

            if response.structured:
                output: VerificationQueryOutput = response.structured
                return output.query

        except Exception as e:
            logger.warning(
                "QUERY_GENERATION_ERROR fact=%s error=%s",
                _truncate(fact.fact_text, 50), str(e)[:100],
            )

        # Fallback: use fact text as query
        return fact.fact_text

    # =========================================================================
    # Evidence extraction from crawled content
    # =========================================================================

    async def _extract_evidence(
        self,
        fact: AtomicFact,
        url: str,
        title: str,
        content: str,
    ) -> RankedEvidence | None:
        """Extract relevant evidence from crawled content."""
        try:
            # Truncate content for LLM context
            content_truncated = content[:15000] if content else ""

            prompt = EVIDENCE_EXTRACTION_PROMPT.format(
                fact_text=fact.fact_text,
                source_url=url,
                source_title=title or "Unknown",
                source_content=content_truncated,
            )

            tier = self._resolve_tier(self.entailment_tier)

            response = await self.llm.complete(
                messages=[{"role": "user", "content": prompt}],
                tier=tier,
                structured_output=EvidenceExtractionOutput,
            )

            if not response.structured:
                return None

            output: EvidenceExtractionOutput = response.structured

            if not output.quote_text or output.relevance_score < 0.3:
                return None

            return RankedEvidence(
                source_id=None,
                source_url=url,
                source_title=title,
                quote_text=output.quote_text,
                start_offset=None,
                end_offset=None,
                section_heading=output.section_heading,
                relevance_score=output.relevance_score,
                has_numeric_content=output.has_numeric_content,
            )

        except Exception as e:
            logger.warning(
                "EVIDENCE_EXTRACTION_ERROR url=%s error=%s",
                url, str(e)[:100],
            )
            return None

    # =========================================================================
    # Entailment checking (single + batch)
    # =========================================================================

    async def _check_entailment(
        self,
        fact: AtomicFact,
        evidence: RankedEvidence,
        _claim_index: int,
    ) -> tuple[bool, float]:
        """Check if evidence entails the atomic fact (NLI-style)."""
        try:
            prompt = ENTAILMENT_CHECK_PROMPT.format(
                fact_text=fact.fact_text,
                source_url=evidence.source_url,
                evidence_quote=evidence.quote_text,
            )

            tier = self._resolve_tier(self.entailment_tier)

            response = await self.llm.complete(
                messages=[{"role": "user", "content": prompt}],
                tier=tier,
                structured_output=EntailmentCheckOutput,
            )

            if response.structured:
                output: EntailmentCheckOutput = response.structured
                return output.entails, output.score

        except Exception as e:
            logger.warning(
                "ENTAILMENT_CHECK_ERROR fact=%s error=%s",
                _truncate(fact.fact_text, 50), str(e)[:100],
            )

        return False, 0.0

    async def check_entailment_batch(
        self,
        fact_evidence_pairs: list[tuple[AtomicFact, RankedEvidence, int]],
        batch_size: int = DEFAULT_ENTAILMENT_BATCH_SIZE,
    ) -> list[tuple[bool, float]]:
        """Check entailment for multiple fact-evidence pairs in batches.

        Token optimization: processes multiple pairs in a single LLM call,
        reducing overhead by 80-90%.

        Args:
            fact_evidence_pairs: ``(fact, evidence, claim_index)`` tuples.
            batch_size: Pairs per batch (default: 10).

        Returns:
            ``(entails, score)`` tuples in same order as input.
        """
        if not fact_evidence_pairs:
            return []

        results: list[tuple[bool, float] | None] = [None] * len(fact_evidence_pairs)

        batches: list[list[int]] = []
        for i in range(0, len(fact_evidence_pairs), batch_size):
            batches.append(
                list(range(i, min(i + batch_size, len(fact_evidence_pairs))))
            )

        logger.info(
            "BATCH_ENTAILMENT_START total=%d batch_size=%d batches=%d",
            len(fact_evidence_pairs), batch_size, len(batches),
        )

        for batch_num, batch_indices in enumerate(batches):
            batch_pairs = [
                (fact_evidence_pairs[i][0], fact_evidence_pairs[i][1])
                for i in batch_indices
            ]

            try:
                batch_results = await self._process_entailment_batch(batch_pairs)
                for j, idx in enumerate(batch_indices):
                    results[idx] = (
                        batch_results[j] if j < len(batch_results) else (False, 0.0)
                    )

            except Exception as e:
                logger.warning(
                    "BATCH_ENTAILMENT_ERROR batch=%d error=%s",
                    batch_num, str(e)[:100],
                )
                # Fallback to sequential
                for idx in batch_indices:
                    fact, evidence, claim_index = fact_evidence_pairs[idx]
                    results[idx] = await self._check_entailment(
                        fact, evidence, claim_index,
                    )

        # Fill remaining None values
        for i, result in enumerate(results):
            if result is None:
                results[i] = (False, 0.0)

        self.metrics.entailment_checks += len(fact_evidence_pairs)

        logger.info(
            "BATCH_ENTAILMENT_COMPLETE total=%d entailed=%d",
            len(fact_evidence_pairs),
            sum(1 for r in results if r and r[0]),
        )

        return [(r[0], r[1]) if r else (False, 0.0) for r in results]

    async def _process_entailment_batch(
        self,
        fact_evidence_pairs: list[tuple[AtomicFact, RankedEvidence]],
    ) -> list[tuple[bool, float]]:
        """Process a single batch of entailment checks."""
        if not fact_evidence_pairs:
            return []

        facts_section = self._format_facts_for_batch_entailment(fact_evidence_pairs)
        prompt = BATCH_ENTAILMENT_PROMPT.format(facts_section=facts_section)
        tier = self._resolve_tier(self.entailment_tier)

        response = await self.llm.complete(
            messages=[{"role": "user", "content": prompt}],
            tier=tier,
            structured_output=BatchEntailmentOutput,
        )

        if response.structured:
            output: BatchEntailmentOutput = response.structured
            return self._parse_batch_entailment_results(
                output, len(fact_evidence_pairs),
            )

        # Fallback: parse from content
        return self._parse_batch_entailment_content(
            response.content, len(fact_evidence_pairs),
        )

    @staticmethod
    def _format_facts_for_batch_entailment(
        fact_evidence_pairs: list[tuple[AtomicFact, RankedEvidence]],
    ) -> str:
        """Format fact-evidence pairs for batch entailment prompt."""
        sections: list[str] = []
        for i, (fact, evidence) in enumerate(fact_evidence_pairs):
            section = (
                f'### Fact {i}\n'
                f'**Fact:** "{fact.fact_text}"\n'
                f'**Source:** {evidence.source_url}\n'
                f'**Evidence:** "{evidence.quote_text[:400]}"\n'
            )
            sections.append(section)
        return "\n".join(sections)

    @staticmethod
    def _parse_batch_entailment_results(
        output: BatchEntailmentOutput,
        expected_count: int,
    ) -> list[tuple[bool, float]]:
        """Parse batch entailment output into results list."""
        results: list[tuple[bool, float]] = [(False, 0.0)] * expected_count
        for item in output.results:
            if 0 <= item.fact_index < expected_count:
                results[item.fact_index] = (item.entails, item.score)
            else:
                logger.warning(
                    "BATCH_ENTAILMENT_INDEX_OUT_OF_RANGE idx=%d max=%d",
                    item.fact_index, expected_count,
                )
        return results

    @staticmethod
    def _parse_batch_entailment_content(
        content: str,
        expected_count: int,
    ) -> list[tuple[bool, float]]:
        """Fallback parser for batch entailment when structured output fails."""
        results: list[tuple[bool, float]] = [(False, 0.0)] * expected_count
        try:
            json_match = re.search(r"\{[\s\S]*\}", content)
            if json_match:
                data = json.loads(json_match.group())
                if "results" in data and isinstance(data["results"], list):
                    for item in data["results"]:
                        idx = item.get("fact_index", -1)
                        if 0 <= idx < expected_count:
                            entails = item.get("entails", False)
                            score = float(item.get("score", 0.0))
                            results[idx] = (entails, score)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning("BATCH_ENTAILMENT_PARSE_FAILURE error=%s", e)
        return results

    # =========================================================================
    # Claim reconstruction
    # =========================================================================

    async def _reconstruct_claim(
        self,
        decomposition: ClaimDecomposition,
        claim_index: int,
    ) -> str:
        """Reconstruct a claim from verified/softened atomic facts."""
        # If all verified, return original
        if decomposition.all_verified:
            return decomposition.original_claim.claim_text

        # If all unverified, apply simple softening
        if not decomposition.partial_verified and decomposition.total_count > 0:
            return await self._apply_softening(
                decomposition.original_claim.claim_text, claim_index,
            )

        # Mixed verified/unverified: use reconstruction prompt
        facts_status = self._format_facts_with_status(decomposition)

        try:
            prompt = CLAIM_RECONSTRUCTION_PROMPT.format(
                original_claim=decomposition.original_claim.claim_text,
                facts_with_status=facts_status,
            )

            tier = self._resolve_tier(self.reconstruction_tier)

            response = await self.llm.complete(
                messages=[{"role": "user", "content": prompt}],
                tier=tier,
            )

            if response.content:
                return response.content.strip()

        except Exception as e:
            logger.warning("RECONSTRUCTION_ERROR error=%s", str(e)[:100])

        # Fallback: hedge prefix
        return f"Reportedly, {decomposition.original_claim.claim_text.lower()}"

    def _format_facts_with_status(
        self, decomposition: ClaimDecomposition,
    ) -> str:
        """Format atomic facts with verification status for reconstruction.

        Uses pre-assigned citation keys to ensure the LLM outputs the exact
        same key we registered in the evidence pool.
        """
        lines: list[str] = []
        for fact in decomposition.atomic_facts:
            status = "VERIFIED" if fact.is_verified else "UNVERIFIED"
            source = ""
            if fact.is_verified and fact.evidence:
                if fact.assigned_citation_key:
                    source = f" [{fact.assigned_citation_key}]"
                else:
                    source = f" [{self._extract_domain(fact.evidence.source_url)}]"
            lines.append(f'- "{fact.fact_text}" - {status}{source}')
        return "\n".join(lines)

    # =========================================================================
    # Claim softening
    # =========================================================================

    async def _apply_softening(
        self, claim_text: str, _claim_index: int,
    ) -> str:
        """Apply softening to an entirely unverified claim."""
        strategy = self.softening_strategy

        if strategy == SofteningStrategy.HEDGE:
            prompt = CLAIM_SOFTENING_HEDGE_PROMPT.format(claim_text=claim_text)
        elif strategy == SofteningStrategy.QUALIFY:
            prompt = CLAIM_SOFTENING_QUALIFY_PROMPT.format(claim_text=claim_text)
        elif strategy == SofteningStrategy.PARENTHETICAL:
            prompt = CLAIM_SOFTENING_PARENTHETICAL_PROMPT.format(
                claim_text=claim_text,
            )
        else:
            prompt = CLAIM_SOFTENING_HEDGE_PROMPT.format(claim_text=claim_text)

        try:
            tier = self._resolve_tier(self.softening_tier)

            response = await self.llm.complete(
                messages=[{"role": "user", "content": prompt}],
                tier=tier,
            )

            if response.content:
                return response.content.strip()

        except Exception as e:
            logger.warning("SOFTENING_ERROR error=%s", str(e)[:100])

        # Fallback based on strategy
        if strategy == SofteningStrategy.PARENTHETICAL:
            return f"{claim_text} (unverified)"
        elif strategy == SofteningStrategy.QUALIFY:
            return f"Some evidence suggests that {claim_text.lower()}"
        else:
            return f"Reportedly, {claim_text.lower()}"

    # =========================================================================
    # Report revision helpers
    # =========================================================================

    def apply_revision_to_report(
        self,
        report: str,
        revision: ClaimRevision,
        position_offset: int = 0,
    ) -> tuple[str, int]:
        """Apply a single revision to the report.

        Process revisions in reverse position order to avoid drift.

        Args:
            report: Current report content.
            revision: The claim revision to apply.
            position_offset: Cumulative offset from previous revisions.

        Returns:
            ``(updated_report, new_offset)`` tuple.
        """
        start = revision.original_position_start + position_offset
        end = revision.original_position_end + position_offset

        if start < 0 or end > len(report) or start >= end:
            logger.warning(
                "REVISION_POSITION_ERROR start=%d end=%d report_len=%d",
                start, end, len(report),
            )
            return report, position_offset

        updated = report[:start] + revision.revised_claim + report[end:]
        length_diff = len(revision.revised_claim) - (end - start)
        new_offset = position_offset + length_diff

        logger.debug(
            "REVISION_APPLIED original_len=%d revised_len=%d offset_change=%d",
            end - start, len(revision.revised_claim), length_diff,
        )

        return updated, new_offset

    def apply_all_revisions(
        self,
        report: str,
        revisions: list[ClaimRevision],
    ) -> str:
        """Apply all revisions to the report.

        Processes in reverse position order to handle position drift.

        Args:
            report: Original report content.
            revisions: List of claim revisions.

        Returns:
            Updated report with all revisions applied.
        """
        if not revisions:
            return report

        sorted_revisions = sorted(
            revisions,
            key=lambda r: r.original_position_start,
            reverse=True,
        )

        updated = report
        for revision in sorted_revisions:
            updated, _ = self.apply_revision_to_report(updated, revision, 0)

        return updated

    # =========================================================================
    # Citation key management
    # =========================================================================

    def get_new_evidence_with_keys(self) -> list[NewExternalEvidence]:
        """Get new external evidence discovered during Stage 7.

        Returns pre-assigned citation keys that should be added to the
        evidence pool by the caller.
        """
        return self._new_external_evidence_with_keys

    def _generate_citation_key(self, url: str) -> str:
        """Generate a unique citation key for new external evidence.

        Ensures no conflict with existing keys by adding numeric suffixes.
        """
        base_key = self._extract_domain(url)

        if base_key not in self._existing_citation_keys:
            self._existing_citation_keys.add(base_key)
            return base_key

        count = 2
        while f"{base_key}-{count}" in self._existing_citation_keys:
            count += 1

        unique_key = f"{base_key}-{count}"
        self._existing_citation_keys.add(unique_key)

        logger.debug(
            "GENERATED_UNIQUE_CITATION_KEY base=%s unique=%s",
            base_key, unique_key,
        )

        return unique_key

    @staticmethod
    def _extract_domain(url: str) -> str:
        """Extract domain name from URL for citation key."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.replace("www.", "")
            return domain.split(".")[0].capitalize()
        except Exception:
            return "Source"

    # =========================================================================
    # Tier resolution
    # =========================================================================

    @staticmethod
    def _resolve_tier(tier_name: str) -> ModelTier:
        """Convert tier name string to ``ModelTier`` enum."""
        tier_map = {
            "simple": ModelTier.simple,
            "analytical": ModelTier.analytical,
            "complex": ModelTier.complex,
        }
        return tier_map.get(tier_name.lower(), ModelTier.simple)
