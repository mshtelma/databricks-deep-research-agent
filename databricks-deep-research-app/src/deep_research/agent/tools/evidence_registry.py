"""Evidence Registry for index-based evidence access in ReAct synthesis.

Provides a mapping layer between numeric indices and evidence spans,
preventing the LLM from seeing URLs directly (security) and tracking
which evidence has been accessed for grounding inference.

This is the synthesis counterpart to UrlRegistry used in research.

Supports hybrid search combining:
- BM25S: Fast keyword search (exact terms, numbers, names)
- GTE embeddings: Semantic search (conceptual similarity)
- Hybrid fusion: Weighted combination (default: 0.6 vector, 0.4 BM25)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Lock
from typing import TYPE_CHECKING, Any
from uuid import UUID

import numpy as np

from deep_research.core.logging_utils import get_logger
from deep_research.services.citation.evidence_selector import RankedEvidence

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)


@dataclass
class IndexedEvidence:
    """An evidence entry in the registry."""

    index: int
    source_url: str
    source_title: str | None
    quote_text: str
    relevance_score: float
    has_numeric_content: bool
    section_heading: str | None = None
    source_id: UUID | None = None
    start_offset: int | None = None
    end_offset: int | None = None


@dataclass
class EvidenceAccess:
    """Record of an evidence access for grounding inference."""

    evidence_index: int
    access_type: str  # "search" or "read"
    timestamp: datetime
    query: str | None = None  # Search query that surfaced this evidence


@dataclass
class RetrievalContext:
    """Tracks recent evidence retrievals for grounding inference.

    Uses a sliding window to determine which evidence is "active"
    when a claim is generated. Claims are grounded if they match
    recently retrieved evidence.
    """

    recent_retrievals: list[tuple[int, str]] = field(default_factory=list)
    window_size: int = 3

    def add_retrieval(self, index: int, quote_text: str) -> None:
        """Add a retrieval to the context window."""
        self.recent_retrievals.append((index, quote_text))
        if len(self.recent_retrievals) > self.window_size:
            self.recent_retrievals.pop(0)

    def get_active_evidence(self) -> list[tuple[int, str]]:
        """Get currently active evidence in the window."""
        return list(self.recent_retrievals)

    def clear(self) -> None:
        """Clear the retrieval context."""
        self.recent_retrievals.clear()


@dataclass
class HybridSearchResult:
    """Result from hybrid search with detailed scores."""

    index: int
    score: float
    bm25_score: float
    vector_score: float


class HybridSearchIndex:
    """In-memory hybrid search index using BM25S + vector embeddings.

    Combines keyword search (BM25) with semantic search (cosine similarity)
    for evidence retrieval. BM25 catches exact terms like "GPT-4" or "86.4%",
    while vectors catch conceptual similarity.

    The fusion formula is:
        hybrid_score = α * vector_score + (1-α) * bm25_score

    where α defaults to 0.6 (slightly preferring semantic matches).
    """

    def __init__(
        self,
        texts: list[str],
        embeddings: NDArray[np.float32] | None = None,
        alpha: float = 0.6,
    ):
        """Initialize hybrid search index.

        Args:
            texts: List of text documents to index.
            embeddings: Pre-computed embeddings array of shape (n_docs, embed_dim).
                       If None, only BM25 search is used.
            alpha: Weight for vector scores (0.0 = pure BM25, 1.0 = pure vector).
        """
        self._texts = texts
        self._embeddings = embeddings
        self._alpha = alpha
        self._n_docs = len(texts)

        # Build BM25 index using bm25s
        try:
            import bm25s

            # Tokenize corpus
            self._corpus_tokens = bm25s.tokenize(texts, stopwords="en")
            self._bm25 = bm25s.BM25()
            self._bm25.index(self._corpus_tokens)
            self._has_bm25 = True
            logger.debug("BM25_INDEX_BUILT", n_docs=self._n_docs)
        except ImportError:
            logger.warning("BM25S_NOT_AVAILABLE", fallback="keyword_only")
            self._has_bm25 = False
            self._bm25 = None
            self._corpus_tokens = None
        except Exception as e:
            logger.warning("BM25_INDEX_FAILED", error=str(e)[:100])
            self._has_bm25 = False
            self._bm25 = None
            self._corpus_tokens = None

    def search(
        self,
        query: str,
        query_embedding: NDArray[np.float32] | None = None,
        limit: int = 5,
    ) -> list[HybridSearchResult]:
        """Search using hybrid BM25 + vector similarity.

        Args:
            query: Text query for BM25 search.
            query_embedding: Pre-computed query embedding for vector search.
                            If None and embeddings exist, only BM25 is used.
            limit: Maximum number of results.

        Returns:
            List of HybridSearchResult sorted by hybrid score descending.
        """
        if self._n_docs == 0:
            return []

        # Compute BM25 scores
        bm25_scores = self._compute_bm25_scores(query)

        # Compute vector scores
        vector_scores = self._compute_vector_scores(query_embedding)

        # Hybrid fusion
        bm25_has_results = bm25_scores is not None and float(np.max(bm25_scores)) > 0
        vector_has_results = vector_scores is not None and float(np.max(vector_scores)) > 0.5

        # Determine which scores to use
        hybrid_scores: NDArray[np.float32]
        if bm25_has_results and vector_has_results:
            # Both available with meaningful scores - blend
            assert bm25_scores is not None and vector_scores is not None
            hybrid_scores = self._alpha * vector_scores + (1 - self._alpha) * bm25_scores
        elif vector_has_results:
            assert vector_scores is not None
            hybrid_scores = vector_scores
        elif bm25_has_results:
            assert bm25_scores is not None
            hybrid_scores = bm25_scores
        else:
            # Fallback to simple keyword matching when no meaningful scores
            return self._fallback_keyword_search(query, limit)

        # Top-k indices
        top_indices = np.argsort(hybrid_scores)[::-1][:limit]

        results = []
        for idx in top_indices:
            idx_int = int(idx)
            if float(hybrid_scores[idx_int]) > 0:
                results.append(
                    HybridSearchResult(
                        index=idx_int,
                        score=float(hybrid_scores[idx_int]),
                        bm25_score=float(bm25_scores[idx_int]) if bm25_scores is not None else 0.0,
                        vector_score=float(vector_scores[idx_int]) if vector_scores is not None else 0.0,
                    )
                )

        return results

    def _compute_bm25_scores(self, query: str) -> NDArray[np.float32] | None:
        """Compute normalized BM25 scores for query."""
        if not self._has_bm25 or self._bm25 is None:
            return None

        try:
            import bm25s

            query_tokens = bm25s.tokenize([query], stopwords="en")
            # Get scores for all documents
            # Note: bm25.retrieve returns (indices, scores), not (scores, indices)
            result_indices, result_scores = self._bm25.retrieve(query_tokens, k=self._n_docs)
            indices = result_indices[0]  # First query
            scores = result_scores[0]

            # Reconstruct full score array (bm25.retrieve returns sorted results)
            bm25_scores = np.zeros(self._n_docs, dtype=np.float32)
            for idx, score in zip(indices, scores):
                bm25_scores[int(idx)] = float(score)

            # Normalize to 0-1 range
            max_score = np.max(bm25_scores)
            if max_score > 0:
                bm25_scores = bm25_scores / max_score
            return bm25_scores
        except Exception as e:
            logger.warning("BM25_SEARCH_FAILED", error=str(e)[:100])
            return None

    def _compute_vector_scores(
        self,
        query_embedding: NDArray[np.float32] | None,
    ) -> NDArray[np.float32] | None:
        """Compute normalized cosine similarity scores."""
        if self._embeddings is None or query_embedding is None:
            return None

        try:
            # Cosine similarity (embeddings should be normalized, but we normalize anyway)
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            doc_norms = self._embeddings / (
                np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-8
            )
            similarities = np.dot(doc_norms, query_norm)

            # Normalize to 0-1 range (cosine is already -1 to 1, shift and scale)
            vector_scores_result: NDArray[np.float32] = ((similarities + 1) / 2).astype(np.float32)
            return vector_scores_result
        except Exception as e:
            logger.warning("VECTOR_SEARCH_FAILED", error=str(e)[:100])
            return None

    def _fallback_keyword_search(
        self,
        query: str,
        limit: int,
    ) -> list[HybridSearchResult]:
        """Fallback to simple keyword overlap when BM25/vectors unavailable."""
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Score all documents to ensure we return something
        scored: list[tuple[float, int]] = []
        for idx, text in enumerate(self._texts):
            text_lower = text.lower()
            text_words = set(text_lower.split())

            # Word overlap
            overlap = len(query_words & text_words)
            # Substring bonus
            substring_bonus = 1.0 if query_lower in text_lower else 0.0
            # Base score to ensure all docs get some rank
            base_score = 0.1

            score = base_score + overlap * 0.4 + substring_bonus * 0.5
            scored.append((score, idx))

        # Sort and limit (always return top results even with low scores)
        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            HybridSearchResult(
                index=idx,
                score=score,
                bm25_score=0.0,
                vector_score=0.0,
            )
            for score, idx in scored[:limit]
        ]


class EvidenceRegistry:
    """Thread-safe registry managing evidence for ReAct synthesis.

    Provides:
    - Index-based access (LLM only sees indices, never URLs)
    - Keyword + semantic search over evidence snippets
    - Access tracking for grounding inference
    - Citation key generation for final report

    Example usage:
        registry = EvidenceRegistry(evidence_pool)

        # During search_evidence tool call
        results = registry.search("GPT-4 performance", limit=5)
        # Returns: [{"index": 0, "title": "...", "snippet_preview": "..."}, ...]

        # During read_snippet tool call
        evidence = registry.get(index=0)
        # Records access for grounding inference

        # After synthesis - get citation audit
        audit = registry.get_access_audit()
    """

    def __init__(
        self,
        evidence_pool: list[RankedEvidence],
        retrieval_window_size: int = 3,
        embeddings: NDArray[np.float32] | None = None,
        hybrid_alpha: float = 0.6,
    ) -> None:
        """Initialize registry from pre-selected evidence pool.

        Args:
            evidence_pool: List of RankedEvidence from Stage 1.
            retrieval_window_size: Size of sliding window for grounding.
            embeddings: Pre-computed embeddings for evidence texts.
                       Shape: (n_evidence, embed_dim). If None, keyword-only search.
            hybrid_alpha: Weight for vector vs BM25 (0.6 = 60% vector, 40% BM25).
        """
        self._evidence: dict[int, IndexedEvidence] = {}
        self._access_log: list[EvidenceAccess] = []
        self._retrieval_context = RetrievalContext(window_size=retrieval_window_size)
        self._lock = Lock()
        self._embeddings = embeddings

        # Build index from evidence pool
        texts: list[str] = []
        for idx, ev in enumerate(evidence_pool):
            self._evidence[idx] = IndexedEvidence(
                index=idx,
                source_url=ev.source_url,
                source_title=ev.source_title,
                quote_text=ev.quote_text,
                relevance_score=ev.relevance_score,
                has_numeric_content=ev.has_numeric_content,
                section_heading=ev.section_heading,
                source_id=ev.source_id,
                start_offset=ev.start_offset,
                end_offset=ev.end_offset,
            )
            texts.append(ev.quote_text)

        # Build hybrid search index
        self._search_index = HybridSearchIndex(
            texts=texts,
            embeddings=embeddings,
            alpha=hybrid_alpha,
        )

        logger.info(
            "EVIDENCE_REGISTRY_INITIALIZED",
            n_evidence=len(evidence_pool),
            has_embeddings=embeddings is not None,
            hybrid_alpha=hybrid_alpha,
        )

    def search(
        self,
        query: str,
        limit: int = 5,
        claim_type: str | None = None,
        query_embedding: NDArray[np.float32] | None = None,
    ) -> list[dict[str, Any]]:
        """Search evidence pool for relevant snippets using hybrid search.

        Uses BM25 + vector similarity for ranking (when embeddings available).
        Records search access for all returned results.

        Args:
            query: Search query from LLM.
            limit: Maximum results to return.
            claim_type: Optional type hint ("factual", "numeric", etc.)
            query_embedding: Pre-computed embedding for vector search.

        Returns:
            List of result dicts with index, title, snippet_preview.
        """
        with self._lock:
            # Use hybrid search index
            search_results = self._search_index.search(
                query=query,
                query_embedding=query_embedding,
                limit=limit * 2,  # Get extra for post-filtering
            )

            # Post-process with numeric boost if needed
            final_results: list[dict[str, Any]] = []
            for sr in search_results:
                ev = self._evidence.get(sr.index)
                if not ev:
                    continue

                # Apply numeric boost if claim_type suggests it
                adjusted_score = sr.score
                if claim_type == "numeric" and ev.has_numeric_content:
                    adjusted_score *= 1.3

                # Record access
                self._access_log.append(
                    EvidenceAccess(
                        evidence_index=sr.index,
                        access_type="search",
                        timestamp=datetime.now(UTC),
                        query=query,
                    )
                )

                # Return preview (first 150 chars)
                preview = ev.quote_text[:150] + "..." if len(ev.quote_text) > 150 else ev.quote_text

                final_results.append({
                    "index": sr.index,
                    "title": ev.source_title or "Unknown Source",
                    "snippet_preview": preview,
                    "relevance_score": ev.relevance_score,
                    "has_numeric": ev.has_numeric_content,
                    "hybrid_score": adjusted_score,
                    "bm25_score": sr.bm25_score,
                    "vector_score": sr.vector_score,
                })

                if len(final_results) >= limit:
                    break

            # Sort by adjusted score
            final_results.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)

            logger.debug(
                "EVIDENCE_SEARCH_COMPLETE",
                query=query[:50],
                results=len(final_results),
                claim_type=claim_type,
                has_embedding=query_embedding is not None,
            )

            return final_results

    def get(self, index: int) -> IndexedEvidence | None:
        """Get evidence by index and record read access.

        This adds the evidence to the retrieval context window
        for grounding inference.

        Args:
            index: Evidence index.

        Returns:
            IndexedEvidence or None if not found.
        """
        with self._lock:
            evidence = self._evidence.get(index)
            if evidence:
                # Record read access
                self._access_log.append(
                    EvidenceAccess(
                        evidence_index=index,
                        access_type="read",
                        timestamp=datetime.now(UTC),
                    )
                )
                # Add to retrieval context for grounding
                self._retrieval_context.add_retrieval(index, evidence.quote_text)

            return evidence

    def get_retrieval_context(self) -> RetrievalContext:
        """Get the current retrieval context for grounding checks.

        Returns:
            The RetrievalContext with recently read evidence.
        """
        return self._retrieval_context

    def get_active_retrievals(self) -> list[tuple[int, str]]:
        """Get currently active evidence in the sliding window.

        Returns:
            List of (index, quote_text) tuples.
        """
        return self._retrieval_context.get_active_evidence()

    def get_access_audit(self) -> list[dict[str, Any]]:
        """Get full access audit for provenance tracking.

        Returns:
            List of access records with timestamps.
        """
        with self._lock:
            return [
                {
                    "evidence_index": a.evidence_index,
                    "access_type": a.access_type,
                    "timestamp": a.timestamp.isoformat(),
                    "query": a.query,
                }
                for a in self._access_log
            ]

    def get_read_indices(self) -> set[int]:
        """Get set of all evidence indices that were read.

        Useful for determining which evidence was actually consulted.

        Returns:
            Set of indices that had read access.
        """
        with self._lock:
            return {a.evidence_index for a in self._access_log if a.access_type == "read"}

    def get_entry(self, index: int) -> IndexedEvidence | None:
        """Get evidence by index without recording access.

        Use this for internal operations where we don't want
        to affect the grounding context.

        Args:
            index: Evidence index.

        Returns:
            IndexedEvidence or None.
        """
        with self._lock:
            return self._evidence.get(index)

    def get_count(self) -> int:
        """Get number of evidence entries in registry.

        Returns:
            Number of evidence entries.
        """
        with self._lock:
            return len(self._evidence)

    def build_citation_key(self, index: int) -> str:
        """Build a human-readable citation key for an evidence index.

        Extracts domain from URL and adds suffix for duplicates.

        Args:
            index: Evidence index.

        Returns:
            Citation key like "Arxiv", "Github-2".
        """
        evidence = self.get_entry(index)
        if not evidence:
            return f"Source-{index}"

        # Extract domain from URL
        url = evidence.source_url
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.replace("www.", "")
            # Get first part of domain (e.g., "arxiv" from "arxiv.org")
            key = domain.split(".")[0].capitalize()
        except Exception:
            key = "Source"

        # Check for duplicates in registry and add suffix if needed
        count = 0
        with self._lock:
            for idx, ev in self._evidence.items():
                if idx < index:
                    other_key = self._extract_domain_key(ev.source_url)
                    if other_key == key:
                        count += 1

        if count > 0:
            return f"{key}-{count + 1}"
        return key

    def _extract_domain_key(self, url: str) -> str:
        """Extract domain key from URL."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.replace("www.", "")
            return domain.split(".")[0].capitalize()
        except Exception:
            return "Source"

    def list_all(self) -> list[IndexedEvidence]:
        """Get all evidence entries.

        Returns:
            List of all IndexedEvidence, sorted by index.
        """
        with self._lock:
            return sorted(self._evidence.values(), key=lambda x: x.index)

    def __len__(self) -> int:
        """Return number of evidence entries."""
        return self.get_count()

    def __contains__(self, index: int) -> bool:
        """Check if index exists in registry."""
        with self._lock:
            return index in self._evidence
