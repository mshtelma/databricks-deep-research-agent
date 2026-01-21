"""Stage 7: ARE-style Verification Retrieval & Claim Revision service.

Implements the ARE (Atomic fact decomposition-based Retrieval and Editing)
pattern for verifying and revising unsupported/partial claims.

Pipeline:
1. Filter claims (verdict = unsupported | partial)
2. Decompose each claim into atomic facts
3. For each atomic fact:
   a. Search internal evidence pool (BM25 + semantic similarity)
   b. If not found: External Brave search + web crawl
   c. NLI entailment check
   d. Mark: verified or unverified
4. Reconstruct claim with verified/softened facts
5. Apply revision to report (position-based replacement)

Scientific basis:
- ARE: https://arxiv.org/abs/2410.16708
- FActScore: https://arxiv.org/abs/2305.14251
- SAFE: https://arxiv.org/abs/2403.18802

Token Optimization Features:
- Batch entailment: Process multiple fact-evidence pairs in single LLM call (10 pairs per batch)
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import mlflow
from mlflow.entities import SpanType
from pydantic import BaseModel, Field

from deep_research.agent.prompts.citation.verification_retrieval import (
    BATCH_ENTAILMENT_PROMPT,
    CLAIM_RECONSTRUCTION_PROMPT,
    CLAIM_SOFTENING_HEDGE_PROMPT,
    CLAIM_SOFTENING_PARENTHETICAL_PROMPT,
    CLAIM_SOFTENING_QUALIFY_PROMPT,
    ENTAILMENT_CHECK_PROMPT,
    EVIDENCE_EXTRACTION_PROMPT,
    REFORMULATION_GUIDANCE,
    VERIFICATION_QUERY_PROMPT,
)
from deep_research.core.app_config import (
    SofteningStrategy,
    VerificationRetrievalConfig,
    get_app_config,
)
from deep_research.core.logging_utils import get_logger, truncate
from deep_research.core.tracing_constants import (
    ATTR_CLAIM_INDEX,
    ATTR_CLAIM_TEXT,
    ATTR_CLAIM_VERDICT,
    ATTR_DECOMPOSITION_FACT_COUNT,
    ATTR_ENTAILMENT_REASONING,
    ATTR_ENTAILMENT_SCORE,
    ATTR_ENTAILS,
    ATTR_EVIDENCE_SOURCE,
    ATTR_FACT_INDEX,
    ATTR_FACT_TEXT,
    ATTR_INPUT_CLAIMS_COUNT,
    ATTR_INPUT_EVIDENCE_POOL_SIZE,
    ATTR_OUTPUT_FULLY_SOFTENED,
    ATTR_OUTPUT_FULLY_VERIFIED,
    ATTR_OUTPUT_PARTIALLY_SOFTENED,
    ATTR_REVISION_TYPE,
    ATTR_SEARCH_ATTEMPT,
    ATTR_SEARCH_QUERY,
    ATTR_SEARCH_RESULTS_COUNT,
    ATTR_SOFTENED_COUNT,
    ATTR_SOFTENING_INPUT,
    ATTR_SOFTENING_OUTPUT,
    ATTR_SOFTENING_STRATEGY,
    ATTR_VERIFIED,
    ATTR_VERIFIED_COUNT,
    OP_ENTAILMENT_CHECK,
    OP_EXTERNAL_SEARCH,
    OP_INTERNAL_SEARCH,
    OP_PROCESS,
    OP_RECONSTRUCT,
    OP_RETRIEVE_AND_REVISE,
    OP_SOFTEN,
    OP_VERIFY,
    STAGE_7_ARE,
    citation_span_name,
    truncate_for_attr,
)
from deep_research.services.citation.atomic_decomposer import (
    AtomicDecomposer,
    AtomicFact,
    ClaimDecomposition,
    ClaimRevision,
    DecompositionMetrics,
    EvidenceSource,
)
from deep_research.services.citation.evidence_selector import RankedEvidence
from deep_research.services.llm.client import LLMClient
from deep_research.services.llm.types import ModelTier

if TYPE_CHECKING:
    from uuid import UUID

    from deep_research.agent.state import ClaimInfo, SourceInfo
    from deep_research.agent.tools.web_crawler import CrawlResult, WebCrawler
    from deep_research.services.search.brave import BraveSearchClient, SearchResult

logger = get_logger(__name__)


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


# Batch entailment models (Token Optimization - Phase 3)
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
    """SSE event for frontend updates during verification."""

    event_type: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class NewExternalEvidence:
    """New evidence found by Stage 7 external search with pre-assigned citation key.

    This dataclass tracks evidence discovered during Stage 7 verification that
    came from external Brave search (not the internal evidence pool). The citation
    key is pre-assigned BEFORE reconstruction to ensure consistency between what
    the LLM outputs and what we register in the evidence pool.
    """

    evidence: RankedEvidence
    citation_key: str  # Pre-assigned key that the LLM will use in reconstruction
    fact_text: str  # The atomic fact this evidence verified
    source_url: str  # URL of the external source


# =============================================================================
# Internal Evidence Pool Searcher
# =============================================================================


class InternalPoolSearcher:
    """Searches the internal evidence pool using BM25 + semantic similarity."""

    def __init__(self, evidence_pool: list[RankedEvidence]) -> None:
        """Initialize with evidence pool.

        Args:
            evidence_pool: List of pre-selected evidence from Stage 1.
        """
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

    def _tokenize(self, text: str) -> set[str]:
        """Tokenize text for BM25 scoring."""
        # Simple tokenization: lowercase, alphanumeric only
        tokens = re.findall(r"\b[a-z0-9]+\b", text.lower())
        # Filter very short tokens
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
            threshold: Minimum similarity threshold.
            top_k: Maximum number of results.

        Returns:
            List of (evidence, score) tuples, sorted by score descending.
        """
        if not self.evidence_pool:
            return []

        query_terms = self._tokenize(fact_text)
        if not query_terms:
            return []

        # Score all documents
        scores: list[tuple[int, float]] = []
        for i in range(len(self.evidence_pool)):
            score = self._bm25_score(query_terms, i)
            if score >= threshold:
                scores.append((i, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top_k results with evidence
        return [
            (self.evidence_pool[idx], score)
            for idx, score in scores[:top_k]
        ]


# =============================================================================
# Verification Retriever Service
# =============================================================================


class VerificationRetriever:
    """ARE-style verification and revision for failed claims.

    Orchestrates the full Stage 7 pipeline:
    1. Filter claims by verdict
    2. Decompose into atomic facts
    3. Verify each fact (internal → external search)
    4. Reconstruct with verified/softened facts
    5. Apply to report
    """

    def __init__(
        self,
        llm: LLMClient,
        brave_client: BraveSearchClient | None = None,
        web_crawler: WebCrawler | None = None,
        config: VerificationRetrievalConfig | None = None,
    ) -> None:
        """Initialize the verification retriever.

        Args:
            llm: LLM client for entailment checks and reconstruction.
            brave_client: Brave Search client for external search.
            web_crawler: Web crawler for fetching external pages.
            config: Verification retrieval configuration.
        """
        self.llm = llm
        self.brave_client = brave_client
        self.web_crawler = web_crawler
        self.config = (
            config or get_app_config().citation_verification.verification_retrieval
        )
        self.decomposer = AtomicDecomposer(llm, self.config)
        self.metrics = VerificationRetrievalMetrics()

        # Track new external evidence with pre-assigned citation keys
        # This is populated during _search_external() when new evidence is found
        self._new_external_evidence_with_keys: list[NewExternalEvidence] = []
        # Track existing citation keys to avoid duplicates during key generation
        self._existing_citation_keys: set[str] = set()

    async def retrieve_and_revise(
        self,
        claims: list[ClaimInfo],
        evidence_pool: list[RankedEvidence],
        report_content: str,
        research_query: str,
        sources: list[SourceInfo] | None = None,
    ) -> AsyncGenerator[VerificationEvent | ClaimRevision, None]:
        """Main entry point for Stage 7 verification retrieval.

        Args:
            claims: List of claims from previous stages.
            evidence_pool: Pre-selected evidence from Stage 1.
            report_content: Current report content.
            research_query: Original research query for context.
            sources: List of sources (for adding new sources).

        Yields:
            VerificationEvent for progress updates and ClaimRevision for results.
        """
        span_name = citation_span_name(STAGE_7_ARE, OP_RETRIEVE_AND_REVISE)

        with mlflow.start_span(name=span_name, span_type=SpanType.CHAIN) as span:
            # Initialize metrics and tracking
            self.metrics = VerificationRetrievalMetrics()
            self._new_external_evidence_with_keys = []

            # Initialize existing citation keys from evidence pool for deduplication
            self._existing_citation_keys = {
                self._extract_domain(ev.source_url)
                for ev in evidence_pool
                if ev.source_url
            }

            # Filter claims by verdict
            claims_to_process = self._filter_claims(claims)

            # Set input attributes
            span.set_attributes({
                ATTR_INPUT_CLAIMS_COUNT: len(claims),
                "input.filtered_claims": len(claims_to_process),
                ATTR_INPUT_EVIDENCE_POOL_SIZE: len(evidence_pool),
                "input.trigger_verdicts": str(list(self.config.trigger_on_verdicts)),
            })

            if not claims_to_process:
                logger.info("STAGE_7_SKIP", reason="No claims to process")
                span.set_attributes({
                    "output.skipped": True,
                    "output.skip_reason": "No unsupported or partial claims",
                })
                yield VerificationEvent(
                    "stage_7_skipped",
                    {"reason": "No unsupported or partial claims"},
                )
                return

            yield VerificationEvent(
                "stage_7_started",
                {
                    "total_claims": len(claims_to_process),
                    "verdicts": list(self.config.trigger_on_verdicts),
                },
            )

            # Build internal searcher
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
                        "claim_text": truncate(claim.claim_text, 100),
                        "verdict": claim.verification_verdict,
                    },
                )

                try:
                    revision = await self._process_claim(
                        claim=claim,
                        claim_index=claim_index,
                        internal_searcher=internal_searcher,
                        research_query=research_query,
                        sources=sources,
                    )

                    # Update metrics based on revision type
                    if revision.revision_type == "fully_verified":
                        self.metrics.claims_fully_verified += 1
                    elif revision.revision_type == "partially_softened":
                        self.metrics.claims_partially_softened += 1
                    else:
                        self.metrics.claims_fully_softened += 1

                    yield revision

                except Exception as e:
                    logger.error(
                        "CLAIM_VERIFICATION_ERROR",
                        claim_index=claim_index,
                        error=str(e)[:100],
                    )
                    yield VerificationEvent(
                        "claim_verification_error",
                        {"claim_index": claim_index, "error": str(e)[:100]},
                    )

            # Set output attributes
            span.set_attributes({
                "output.skipped": False,
                "output.claims_processed": self.metrics.total_claims_processed,
                "output.facts_verified": self.metrics.facts_verified,
                "output.facts_softened": self.metrics.facts_softened,
                ATTR_OUTPUT_FULLY_VERIFIED: self.metrics.claims_fully_verified,
                ATTR_OUTPUT_PARTIALLY_SOFTENED: self.metrics.claims_partially_softened,
                ATTR_OUTPUT_FULLY_SOFTENED: self.metrics.claims_fully_softened,
                "output.internal_searches": self.metrics.internal_searches,
                "output.external_searches": self.metrics.external_searches,
                "output.entailment_checks": self.metrics.entailment_checks,
            })

            # Final summary
            yield VerificationEvent("stage_7_complete", self.metrics.to_dict())

            logger.info(
                "STAGE_7_COMPLETE",
                claims_processed=self.metrics.total_claims_processed,
                facts_verified=self.metrics.facts_verified,
                facts_softened=self.metrics.facts_softened,
            )

    def _filter_claims(
        self,
        claims: list[ClaimInfo],
    ) -> list[tuple[int, ClaimInfo]]:
        """Filter claims by verdict to process.

        Args:
            claims: All claims from previous stages.

        Returns:
            List of (index, claim) tuples for claims to process.
        """
        trigger_verdicts = set(self.config.trigger_on_verdicts)

        filtered = [
            (i, c)
            for i, c in enumerate(claims)
            if c.verification_verdict in trigger_verdicts
        ]

        logger.info(
            "CLAIMS_FILTERED",
            total=len(claims),
            filtered=len(filtered),
            verdicts=list(trigger_verdicts),
        )

        return filtered

    async def _process_claim(
        self,
        claim: ClaimInfo,
        claim_index: int,
        internal_searcher: InternalPoolSearcher,
        research_query: str,
        sources: list[SourceInfo] | None = None,
    ) -> ClaimRevision:
        """Process a single claim through the ARE pipeline.

        Args:
            claim: The claim to process.
            claim_index: Index in claims list.
            internal_searcher: Internal evidence pool searcher.
            research_query: Original research query.
            sources: List of sources (for adding new sources).

        Returns:
            ClaimRevision with revised text.
        """
        span_name = citation_span_name(STAGE_7_ARE, OP_PROCESS, claim_index)

        with mlflow.start_span(name=span_name, span_type=SpanType.CHAIN) as span:
            # Set input attributes
            span.set_attributes({
                ATTR_CLAIM_INDEX: claim_index,
                ATTR_CLAIM_TEXT: truncate_for_attr(claim.claim_text, 150),
                ATTR_CLAIM_VERDICT: claim.verification_verdict or "unknown",
            })

            # Step 1: Decompose claim into atomic facts
            decomposition = await self.decomposer.decompose(claim, claim_index)
            self.metrics.total_atomic_facts += len(decomposition.atomic_facts)

            logger.debug(
                "CLAIM_DECOMPOSED",
                claim_index=claim_index,
                fact_count=len(decomposition.atomic_facts),
            )

            span.set_attributes({
                ATTR_DECOMPOSITION_FACT_COUNT: len(decomposition.atomic_facts),
            })

            # Step 2: Verify each atomic fact
            for fact in decomposition.atomic_facts:
                await self._verify_atomic_fact(
                    fact=fact,
                    claim_index=claim_index,
                    internal_searcher=internal_searcher,
                    research_query=research_query,
                    sources=sources,
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

            # Set output attributes
            span.set_attributes({
                ATTR_REVISION_TYPE: revision_type,
                ATTR_VERIFIED_COUNT: decomposition.verified_count,
                ATTR_SOFTENED_COUNT: decomposition.total_count - decomposition.verified_count,
                "revision.all_verified": decomposition.all_verified,
                "revision.partial_verified": decomposition.partial_verified,
            })

            return ClaimRevision(
                original_claim=claim.claim_text,
                revised_claim=revised_claim,
                revision_type=revision_type,
                original_position_start=claim.position_start,
                original_position_end=claim.position_end,
                decomposition=decomposition,
                verified_facts=[f for f in decomposition.atomic_facts if f.is_verified],
                softened_facts=[f for f in decomposition.atomic_facts if not f.is_verified],
                new_citations=[
                    f.evidence.source_url
                    for f in decomposition.atomic_facts
                    if f.is_verified
                    and f.evidence
                    and f.evidence_source == EvidenceSource.EXTERNAL
                ],
            )

    async def _verify_atomic_fact(
        self,
        fact: AtomicFact,
        claim_index: int,
        internal_searcher: InternalPoolSearcher,
        research_query: str,
        sources: list[SourceInfo] | None = None,
    ) -> None:
        """Verify a single atomic fact.

        Search strategy: Internal pool first, then external if needed.

        Args:
            fact: The atomic fact to verify.
            claim_index: Index of parent claim for span naming.
            internal_searcher: Internal evidence pool searcher.
            research_query: Original research query for context.
            sources: List of sources (for adding new sources).
        """
        span_name = citation_span_name(STAGE_7_ARE, OP_VERIFY, claim_index, fact.fact_index)

        with mlflow.start_span(name=span_name, span_type=SpanType.CHAIN) as span:
            # Set input attributes
            span.set_attributes({
                ATTR_CLAIM_INDEX: claim_index,
                ATTR_FACT_INDEX: fact.fact_index,
                ATTR_FACT_TEXT: truncate_for_attr(fact.fact_text, 150),
            })

            # Step 1: Search internal evidence pool
            internal_span_name = citation_span_name(STAGE_7_ARE, OP_INTERNAL_SEARCH, claim_index, fact.fact_index)
            with mlflow.start_span(name=internal_span_name, span_type=SpanType.RETRIEVER) as internal_span:
                self.metrics.internal_searches += 1
                internal_results = internal_searcher.search(
                    fact.fact_text,
                    threshold=self.config.internal_search_threshold,
                )
                internal_span.set_attributes({
                    "search.results_count": len(internal_results),
                    "search.threshold": self.config.internal_search_threshold,
                })

                for evidence, score in internal_results:
                    # Check entailment
                    entails, ent_score = await self._check_entailment(
                        fact, evidence, claim_index
                    )
                    self.metrics.entailment_checks += 1

                    if entails and ent_score >= self.config.entailment_threshold:
                        fact.is_verified = True
                        fact.evidence = evidence
                        fact.evidence_source = EvidenceSource.INTERNAL
                        fact.entailment_score = ent_score
                        logger.debug(
                            "FACT_VERIFIED_INTERNAL",
                            fact=truncate(fact.fact_text, 50),
                            score=f"{ent_score:.2f}",
                        )
                        internal_span.set_attributes({
                            ATTR_VERIFIED: True,
                            ATTR_ENTAILMENT_SCORE: ent_score,
                        })
                        span.set_attributes({
                            ATTR_VERIFIED: True,
                            ATTR_EVIDENCE_SOURCE: "internal",
                            ATTR_ENTAILMENT_SCORE: ent_score,
                        })
                        return

                internal_span.set_attributes({ATTR_VERIFIED: False})

            # Step 2: External search if internal didn't find supporting evidence
            if not fact.is_verified and self.brave_client and self.web_crawler:
                await self._search_external(
                    fact=fact,
                    claim_index=claim_index,
                    research_query=research_query,
                    sources=sources,
                )

            if fact.is_verified:
                span.set_attributes({
                    ATTR_VERIFIED: True,
                    ATTR_EVIDENCE_SOURCE: fact.evidence_source.value,
                    ATTR_ENTAILMENT_SCORE: fact.entailment_score,
                })
            else:
                logger.debug(
                    "FACT_UNVERIFIED",
                    fact=truncate(fact.fact_text, 50),
                    searches=len(fact.search_queries),
                )
                span.set_attributes({
                    ATTR_VERIFIED: False,
                    ATTR_EVIDENCE_SOURCE: "none",
                    "search.external_attempts": len(fact.search_queries),
                })

    async def _search_external(
        self,
        fact: AtomicFact,
        claim_index: int,
        research_query: str,
        sources: list[SourceInfo] | None = None,
    ) -> None:
        """Search external sources for evidence supporting a fact.

        Args:
            fact: The atomic fact to find evidence for.
            claim_index: Index of parent claim for span naming.
            research_query: Original research query for context.
            sources: List of sources (for adding new sources).
        """
        if not self.brave_client or not self.web_crawler:
            return

        span_name = citation_span_name(STAGE_7_ARE, OP_EXTERNAL_SEARCH, claim_index, fact.fact_index)

        with mlflow.start_span(name=span_name, span_type=SpanType.RETRIEVER) as span:
            span.set_attributes({
                ATTR_FACT_INDEX: fact.fact_index,
                ATTR_FACT_TEXT: truncate_for_attr(fact.fact_text, 100),
                "search.max_attempts": self.config.max_searches_per_fact,
            })

            for search_attempt in range(self.config.max_searches_per_fact):
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
                    # Brave search
                    search_response = await asyncio.wait_for(
                        self.brave_client.search(
                            query,
                            count=self.config.max_external_urls_per_search,
                        ),
                        timeout=self.config.search_timeout_seconds,
                    )

                    if not search_response.results:
                        continue

                    # Crawl top URLs
                    urls = [r.url for r in search_response.results]
                    self.metrics.external_crawls += 1

                    crawl_output = await asyncio.wait_for(
                        self.web_crawler.crawl(urls),
                        timeout=self.config.crawl_timeout_seconds,
                    )

                    # Extract and verify evidence from crawled content
                    for crawl_result in crawl_output.results:
                        if not crawl_result.success or not crawl_result.content:
                            continue

                        evidence = await self._extract_evidence(
                            fact=fact,
                            crawl_result=crawl_result,
                        )

                        if not evidence:
                            continue

                        # Check entailment
                        entails, ent_score = await self._check_entailment(
                            fact, evidence, claim_index
                        )
                        self.metrics.entailment_checks += 1

                        if entails and ent_score >= self.config.entailment_threshold:
                            fact.is_verified = True
                            fact.evidence = evidence
                            fact.evidence_source = EvidenceSource.EXTERNAL
                            fact.entailment_score = ent_score
                            self.metrics.new_sources_added += 1

                            # Generate and assign unique citation key BEFORE reconstruction
                            # This ensures the LLM uses the exact key we register
                            citation_key = self._generate_citation_key_for_new_evidence(
                                crawl_result.url
                            )
                            fact.assigned_citation_key = citation_key

                            # Track new evidence with its pre-assigned key
                            self._new_external_evidence_with_keys.append(
                                NewExternalEvidence(
                                    evidence=evidence,
                                    citation_key=citation_key,
                                    fact_text=fact.fact_text,
                                    source_url=crawl_result.url,
                                )
                            )

                            logger.debug(
                                "FACT_VERIFIED_EXTERNAL",
                                fact=truncate(fact.fact_text, 50),
                                source=crawl_result.url,
                                score=f"{ent_score:.2f}",
                                citation_key=citation_key,
                            )
                            span.set_attributes({
                                ATTR_VERIFIED: True,
                                ATTR_SEARCH_ATTEMPT: search_attempt + 1,
                                ATTR_SEARCH_QUERY: truncate_for_attr(query, 100),
                                ATTR_ENTAILMENT_SCORE: ent_score,
                                "search.source_url": crawl_result.url,
                                "search.citation_key": citation_key,
                            })
                            return

                except asyncio.TimeoutError:
                    logger.warning(
                        "EXTERNAL_SEARCH_TIMEOUT",
                        query=truncate(query, 50),
                        attempt=search_attempt + 1,
                    )
                except Exception as e:
                    logger.warning(
                        "EXTERNAL_SEARCH_ERROR",
                        query=truncate(query, 50),
                        error=str(e)[:100],
                    )

            # If we get here, external search didn't verify the fact
            span.set_attributes({
                ATTR_VERIFIED: False,
                "search.attempts": len(fact.search_queries),
                "search.queries": str(fact.search_queries[:3]),  # First 3 queries
            })

    async def _generate_search_query(
        self,
        fact: AtomicFact,
        research_query: str,
        previous_queries: list[str],
        is_reformulation: bool = False,
    ) -> str | None:
        """Generate a search query to verify a fact.

        Args:
            fact: The atomic fact to verify.
            research_query: Original research query for context.
            previous_queries: List of previously tried queries.
            is_reformulation: Whether this is a reformulation attempt.

        Returns:
            Search query string or None if generation fails.
        """
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

            tier = self._get_model_tier(self.config.decomposition_tier)

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
                "QUERY_GENERATION_ERROR",
                fact=truncate(fact.fact_text, 50),
                error=str(e)[:100],
            )

        # Fallback: use fact text as query
        return fact.fact_text

    async def _extract_evidence(
        self,
        fact: AtomicFact,
        crawl_result: CrawlResult,
    ) -> RankedEvidence | None:
        """Extract relevant evidence from crawled content.

        Args:
            fact: The atomic fact to find evidence for.
            crawl_result: Crawled web page content.

        Returns:
            RankedEvidence if relevant quote found, else None.
        """
        try:
            # Truncate content for LLM context
            content = crawl_result.content[:15000] if crawl_result.content else ""

            prompt = EVIDENCE_EXTRACTION_PROMPT.format(
                fact_text=fact.fact_text,
                source_url=crawl_result.url,
                source_title=crawl_result.title or "Unknown",
                source_content=content,
            )

            tier = self._get_model_tier(self.config.entailment_tier)

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
                source_url=crawl_result.url,
                source_title=crawl_result.title,
                quote_text=output.quote_text,
                start_offset=None,
                end_offset=None,
                section_heading=output.section_heading,
                relevance_score=output.relevance_score,
                has_numeric_content=output.has_numeric_content,
            )

        except Exception as e:
            logger.warning(
                "EVIDENCE_EXTRACTION_ERROR",
                url=crawl_result.url,
                error=str(e)[:100],
            )
            return None

    async def _check_entailment(
        self,
        fact: AtomicFact,
        evidence: RankedEvidence,
        claim_index: int,
    ) -> tuple[bool, float]:
        """Check if evidence entails the atomic fact.

        Uses NLI-style verification with configurable threshold.

        Args:
            fact: The atomic fact to verify.
            evidence: Evidence to check against.
            claim_index: Index of parent claim for span naming.

        Returns:
            Tuple of (entails: bool, score: float).
        """
        span_name = citation_span_name(STAGE_7_ARE, OP_ENTAILMENT_CHECK, claim_index, fact.fact_index)

        with mlflow.start_span(name=span_name, span_type=SpanType.CHAIN) as span:
            span.set_attributes({
                ATTR_FACT_TEXT: truncate_for_attr(fact.fact_text, 100),
                "evidence.url": evidence.source_url,
                "evidence.quote": truncate_for_attr(evidence.quote_text, 150),
            })

            try:
                prompt = ENTAILMENT_CHECK_PROMPT.format(
                    fact_text=fact.fact_text,
                    source_url=evidence.source_url,
                    evidence_quote=evidence.quote_text,
                )

                tier = self._get_model_tier(self.config.entailment_tier)

                response = await self.llm.complete(
                    messages=[{"role": "user", "content": prompt}],
                    tier=tier,
                    structured_output=EntailmentCheckOutput,
                )

                if response.structured:
                    output: EntailmentCheckOutput = response.structured
                    span.set_attributes({
                        ATTR_ENTAILS: output.entails,
                        ATTR_ENTAILMENT_SCORE: output.score,
                        ATTR_ENTAILMENT_REASONING: truncate_for_attr(output.reasoning, 150),
                        "entailment.threshold": self.config.entailment_threshold,
                        "entailment.passes_threshold": output.score >= self.config.entailment_threshold,
                    })
                    return output.entails, output.score

                span.set_attributes({
                    ATTR_ENTAILS: False,
                    ATTR_ENTAILMENT_SCORE: 0.0,
                    "entailment.error": "no structured output",
                })

            except Exception as e:
                logger.warning(
                    "ENTAILMENT_CHECK_ERROR",
                    fact=truncate(fact.fact_text, 50),
                    error=str(e)[:100],
                )
                span.set_attributes({
                    ATTR_ENTAILS: False,
                    ATTR_ENTAILMENT_SCORE: 0.0,
                    "entailment.error": str(e)[:100],
                })

            return False, 0.0

    # =========================================================================
    # Token Optimization: Batch Entailment Checks (Phase 3)
    # =========================================================================

    def _format_facts_for_batch_entailment(
        self,
        fact_evidence_pairs: list[tuple[AtomicFact, RankedEvidence]],
    ) -> str:
        """Format fact-evidence pairs for batch entailment prompt.

        Args:
            fact_evidence_pairs: List of (fact, evidence) tuples.

        Returns:
            Formatted string for the batch prompt.
        """
        sections = []
        for i, (fact, evidence) in enumerate(fact_evidence_pairs):
            section = f"""### Fact {i}
**Fact:** "{fact.fact_text}"
**Source:** {evidence.source_url}
**Evidence:** "{evidence.quote_text[:400]}"
"""
            sections.append(section)
        return "\n".join(sections)

    async def check_entailment_batch(
        self,
        fact_evidence_pairs: list[tuple[AtomicFact, RankedEvidence, int]],
        batch_size: int = DEFAULT_ENTAILMENT_BATCH_SIZE,
    ) -> list[tuple[bool, float]]:
        """Check entailment for multiple fact-evidence pairs in batches.

        This is a TOKEN OPTIMIZATION method that processes multiple fact-evidence
        pairs in a single LLM call, reducing overhead by 80-90%.

        Args:
            fact_evidence_pairs: List of (fact, evidence, claim_index) tuples.
            batch_size: Number of pairs per batch (default: 10).

        Returns:
            List of (entails, score) tuples in same order as input.
        """
        if not fact_evidence_pairs:
            return []

        results: list[tuple[bool, float] | None] = [None] * len(fact_evidence_pairs)

        # Group into batches
        batches: list[list[int]] = []
        for i in range(0, len(fact_evidence_pairs), batch_size):
            batches.append(list(range(i, min(i + batch_size, len(fact_evidence_pairs)))))

        logger.info(
            "BATCH_ENTAILMENT_START",
            total_pairs=len(fact_evidence_pairs),
            batch_size=batch_size,
            num_batches=len(batches),
        )

        # Process each batch
        for batch_num, batch_indices in enumerate(batches):
            batch_pairs = [
                (fact_evidence_pairs[i][0], fact_evidence_pairs[i][1])
                for i in batch_indices
            ]

            try:
                batch_results = await self._process_entailment_batch(batch_pairs)

                # Map results back to original indices
                for j, idx in enumerate(batch_indices):
                    if j < len(batch_results):
                        results[idx] = batch_results[j]
                    else:
                        results[idx] = (False, 0.0)

            except Exception as e:
                logger.warning(
                    "BATCH_ENTAILMENT_ERROR",
                    batch_num=batch_num,
                    error=str(e)[:100],
                )
                # Fall back to sequential entailment for this batch
                for idx in batch_indices:
                    fact, evidence, claim_index = fact_evidence_pairs[idx]
                    results[idx] = await self._check_entailment(
                        fact, evidence, claim_index
                    )

        # Fill any remaining None values
        for i, result in enumerate(results):
            if result is None:
                results[i] = (False, 0.0)

        self.metrics.entailment_checks += len(fact_evidence_pairs)

        logger.info(
            "BATCH_ENTAILMENT_COMPLETE",
            total_pairs=len(fact_evidence_pairs),
            entailed_count=sum(1 for r in results if r and r[0]),
        )

        return [(r[0], r[1]) if r else (False, 0.0) for r in results]

    async def _process_entailment_batch(
        self,
        fact_evidence_pairs: list[tuple[AtomicFact, RankedEvidence]],
    ) -> list[tuple[bool, float]]:
        """Process a single batch of fact-evidence pairs for entailment.

        Args:
            fact_evidence_pairs: List of (fact, evidence) tuples.

        Returns:
            List of (entails, score) tuples.
        """
        if not fact_evidence_pairs:
            return []

        # Format for batch prompt
        facts_section = self._format_facts_for_batch_entailment(fact_evidence_pairs)

        prompt = BATCH_ENTAILMENT_PROMPT.format(facts_section=facts_section)
        tier = self._get_model_tier(self.config.entailment_tier)

        try:
            response = await self.llm.complete(
                messages=[{"role": "user", "content": prompt}],
                tier=tier,
                structured_output=BatchEntailmentOutput,
            )

            if response.structured:
                output: BatchEntailmentOutput = response.structured
                return self._parse_batch_entailment_results(
                    output, len(fact_evidence_pairs)
                )

            # Fallback: try to parse from content
            return self._parse_batch_entailment_content(
                response.content, len(fact_evidence_pairs)
            )

        except Exception as e:
            logger.error(f"Batch entailment processing failed: {e}")
            raise

    def _parse_batch_entailment_results(
        self,
        output: BatchEntailmentOutput,
        expected_count: int,
    ) -> list[tuple[bool, float]]:
        """Parse batch entailment output into results list.

        Args:
            output: Structured batch output from LLM.
            expected_count: Expected number of results.

        Returns:
            List of (entails, score) tuples in original order.
        """
        # Initialize with defaults
        results: list[tuple[bool, float]] = [(False, 0.0)] * expected_count

        # Map results by fact_index to handle any reordering
        for item in output.results:
            if 0 <= item.fact_index < expected_count:
                results[item.fact_index] = (item.entails, item.score)
            else:
                logger.warning(
                    "BATCH_ENTAILMENT_INDEX_OUT_OF_RANGE",
                    fact_index=item.fact_index,
                    expected_count=expected_count,
                )

        return results

    def _parse_batch_entailment_content(
        self,
        content: str,
        expected_count: int,
    ) -> list[tuple[bool, float]]:
        """Fallback parser for batch entailment when structured output fails.

        Args:
            content: Raw LLM response content.
            expected_count: Expected number of results.

        Returns:
            List of (entails, score) tuples.
        """
        results: list[tuple[bool, float]] = [(False, 0.0)] * expected_count

        try:
            # Try to extract JSON from response
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
            logger.warning(f"Failed to parse batch entailment response: {e}")

        return results

    async def _reconstruct_claim(
        self,
        decomposition: ClaimDecomposition,
        claim_index: int,
    ) -> str:
        """Reconstruct a claim from verified/softened atomic facts.

        Strategy based on config:
        - Verified facts: Keep with new citations
        - Unverified facts: Apply softening strategy (hedge/qualify/parenthetical)

        Args:
            decomposition: Claim decomposition with verification results.
            claim_index: Index of the claim for span naming.

        Returns:
            Reconstructed claim text.
        """
        span_name = citation_span_name(STAGE_7_ARE, OP_RECONSTRUCT, claim_index)

        with mlflow.start_span(name=span_name, span_type=SpanType.CHAIN) as span:
            span.set_attributes({
                ATTR_CLAIM_INDEX: claim_index,
                ATTR_VERIFIED_COUNT: decomposition.verified_count,
                ATTR_SOFTENED_COUNT: decomposition.total_count - decomposition.verified_count,
                "reconstruction.all_verified": decomposition.all_verified,
                "reconstruction.partial_verified": decomposition.partial_verified,
                ATTR_SOFTENING_STRATEGY: self.config.softening_strategy,
            })

            # If all verified, return original with any new citations
            if decomposition.all_verified:
                span.set_attributes({"reconstruction.action": "keep_original"})
                return decomposition.original_claim.claim_text

            # If all unverified, apply simple softening
            if not decomposition.partial_verified and decomposition.total_count > 0:
                span.set_attributes({"reconstruction.action": "full_softening"})
                return await self._apply_softening(
                    decomposition.original_claim.claim_text,
                    claim_index,
                )

            # Mixed verified/unverified: use reconstruction prompt
            span.set_attributes({"reconstruction.action": "mixed_reconstruction"})
            facts_status = self._format_facts_with_status(decomposition)

            try:
                prompt = CLAIM_RECONSTRUCTION_PROMPT.format(
                    original_claim=decomposition.original_claim.claim_text,
                    facts_with_status=facts_status,
                )

                tier = self._get_model_tier(self.config.reconstruction_tier)

                response = await self.llm.complete(
                    messages=[{"role": "user", "content": prompt}],
                    tier=tier,
                )

                if response.content:
                    span.set_attributes({
                        "reconstruction.success": True,
                        "reconstruction.output_length": len(response.content),
                    })
                    return response.content.strip()

            except Exception as e:
                logger.warning(
                    "RECONSTRUCTION_ERROR",
                    error=str(e)[:100],
                )
                span.set_attributes({
                    "reconstruction.success": False,
                    "reconstruction.error": str(e)[:100],
                })

            # Fallback: return original with hedge prefix
            span.set_attributes({"reconstruction.fallback": True})
            return f"Reportedly, {decomposition.original_claim.claim_text.lower()}"

    def _format_facts_with_status(
        self,
        decomposition: ClaimDecomposition,
    ) -> str:
        """Format atomic facts with verification status for reconstruction prompt.

        CRITICAL: Uses pre-assigned citation keys to ensure the LLM outputs
        the exact same key we registered in the evidence pool. This prevents
        orphaned citation markers.
        """
        lines = []
        for fact in decomposition.atomic_facts:
            status = "VERIFIED" if fact.is_verified else "UNVERIFIED"
            source = ""
            if fact.is_verified and fact.evidence:
                # Use PRE-ASSIGNED key if available (for external evidence)
                # This ensures key consistency between reconstruction and registration
                if fact.assigned_citation_key:
                    source = f" [{fact.assigned_citation_key}]"
                else:
                    # Fallback for internal evidence (no pre-assigned key needed)
                    source = f" [{self._extract_domain(fact.evidence.source_url)}]"
            lines.append(f'- "{fact.fact_text}" - {status}{source}')
        return "\n".join(lines)

    def _extract_domain(self, url: str) -> str:
        """Extract domain name from URL for citation."""
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            domain = parsed.netloc.replace("www.", "")
            # Take first part before dot for short name
            return domain.split(".")[0].capitalize()
        except Exception:
            return "Source"

    def _generate_citation_key_for_new_evidence(self, url: str) -> str:
        """Generate a unique citation key for new external evidence.

        Ensures the key doesn't conflict with existing keys by adding
        a numeric suffix if needed (e.g., "Reuters" -> "Reuters-2").

        Args:
            url: URL of the new external source.

        Returns:
            Unique citation key string.
        """
        base_key = self._extract_domain(url)

        # Check if base key is already used
        if base_key not in self._existing_citation_keys:
            self._existing_citation_keys.add(base_key)
            return base_key

        # Generate unique key with suffix
        count = 2
        while f"{base_key}-{count}" in self._existing_citation_keys:
            count += 1

        unique_key = f"{base_key}-{count}"
        self._existing_citation_keys.add(unique_key)

        logger.debug(
            "GENERATED_UNIQUE_CITATION_KEY",
            base_key=base_key,
            unique_key=unique_key,
            existing_count=count - 1,
        )

        return unique_key

    def get_new_evidence_with_keys(self) -> list[NewExternalEvidence]:
        """Get new external evidence discovered during Stage 7 with pre-assigned citation keys.

        Returns:
            List of NewExternalEvidence objects containing evidence and their
            pre-assigned citation keys. These should be added to the evidence pool
            and state by the caller (pipeline.py).
        """
        return self._new_external_evidence_with_keys

    async def _apply_softening(self, claim_text: str, claim_index: int) -> str:
        """Apply softening to an entirely unverified claim.

        Args:
            claim_text: The original claim text.
            claim_index: Index of the claim for span naming.

        Returns:
            Softened claim text.
        """
        span_name = citation_span_name(STAGE_7_ARE, OP_SOFTEN, claim_index)

        with mlflow.start_span(name=span_name, span_type=SpanType.CHAIN) as span:
            strategy = self.config.softening_strategy
            span.set_attributes({
                ATTR_CLAIM_INDEX: claim_index,
                ATTR_SOFTENING_STRATEGY: strategy,
                ATTR_SOFTENING_INPUT: truncate_for_attr(claim_text, 150),
            })

            # Select prompt based on strategy
            if strategy == SofteningStrategy.HEDGE:
                prompt = CLAIM_SOFTENING_HEDGE_PROMPT.format(claim_text=claim_text)
            elif strategy == SofteningStrategy.QUALIFY:
                prompt = CLAIM_SOFTENING_QUALIFY_PROMPT.format(claim_text=claim_text)
            elif strategy == SofteningStrategy.PARENTHETICAL:
                prompt = CLAIM_SOFTENING_PARENTHETICAL_PROMPT.format(claim_text=claim_text)
            else:
                # Default to hedge
                prompt = CLAIM_SOFTENING_HEDGE_PROMPT.format(claim_text=claim_text)

            try:
                tier = self._get_model_tier(self.config.softening_tier)

                response = await self.llm.complete(
                    messages=[{"role": "user", "content": prompt}],
                    tier=tier,
                )

                if response.content:
                    output = response.content.strip()
                    span.set_attributes({
                        "softening.success": True,
                        ATTR_SOFTENING_OUTPUT: truncate_for_attr(output, 150),
                    })
                    return output

            except Exception as e:
                logger.warning(
                    "SOFTENING_ERROR",
                    error=str(e)[:100],
                )
                span.set_attributes({
                    "softening.success": False,
                    "softening.error": str(e)[:100],
                })

            # Fallback based on strategy
            span.set_attributes({"softening.fallback": True})
            if strategy == SofteningStrategy.PARENTHETICAL:
                output = f"{claim_text} (unverified)"
            elif strategy == SofteningStrategy.QUALIFY:
                output = f"Some evidence suggests that {claim_text.lower()}"
            else:
                output = f"Reportedly, {claim_text.lower()}"

            span.set_attributes({ATTR_SOFTENING_OUTPUT: truncate_for_attr(output, 150)})
            return output

    def _get_model_tier(self, tier_name: str) -> ModelTier:
        """Convert tier name string to ModelTier enum."""
        tier_map = {
            "simple": ModelTier.SIMPLE,
            "analytical": ModelTier.ANALYTICAL,
            "complex": ModelTier.COMPLEX,
            "bulk_analysis": ModelTier.BULK_ANALYSIS,
            "fast": ModelTier.FAST,
        }
        return tier_map.get(tier_name.lower(), ModelTier.SIMPLE)

    def apply_revision_to_report(
        self,
        report: str,
        revision: ClaimRevision,
        position_offset: int = 0,
    ) -> tuple[str, int]:
        """Apply a single revision to the report.

        IMPORTANT: Process revisions in reverse position order to avoid drift.

        Args:
            report: Current report content.
            revision: The claim revision to apply.
            position_offset: Cumulative offset from previous revisions.

        Returns:
            Tuple of (updated_report, new_offset).
        """
        start = revision.original_position_start + position_offset
        end = revision.original_position_end + position_offset

        # Safety check
        if start < 0 or end > len(report) or start >= end:
            logger.warning(
                "REVISION_POSITION_ERROR",
                start=start,
                end=end,
                report_len=len(report),
            )
            return report, position_offset

        # Apply replacement
        updated = report[:start] + revision.revised_claim + report[end:]

        # Calculate new offset
        length_diff = len(revision.revised_claim) - (end - start)
        new_offset = position_offset + length_diff

        logger.debug(
            "REVISION_APPLIED",
            original_len=end - start,
            revised_len=len(revision.revised_claim),
            offset_change=length_diff,
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
            revisions: List of claim revisions to apply.

        Returns:
            Updated report with all revisions applied.
        """
        if not revisions:
            return report

        # Sort by position descending (process from end to start)
        sorted_revisions = sorted(
            revisions,
            key=lambda r: r.original_position_start,
            reverse=True,
        )

        updated = report
        for revision in sorted_revisions:
            # No offset needed when processing reverse order
            updated, _ = self.apply_revision_to_report(updated, revision, 0)

        return updated
