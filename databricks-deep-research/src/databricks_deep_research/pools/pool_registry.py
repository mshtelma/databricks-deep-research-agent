"""Pool registry: manages pool name -> PoolState mapping and initialization.

Pools are created lazily on first access or eagerly from config.
Provides hybrid search (BM25 + vector) with 4-tier graceful degradation:

1. Full hybrid (BM25 + vector): bm25s + embedding model configured
2. BM25 only: bm25s installed, no embedding model
3. Keyword fallback: neither available, simple word overlap
4. Chronological: get_recent() always works
"""

from __future__ import annotations

import json
import logging
from typing import Any

from databricks_deep_research.pools.pool_state import PoolConfig, PoolState

logger = logging.getLogger(__name__)

# Optional search dependencies -- degrade gracefully when not installed.
_HAS_BM25 = False
try:
    import bm25s
    import numpy as np

    _HAS_BM25 = True
except ImportError:
    pass


class PoolRegistry:
    """Registry of named pools for a workflow execution.

    Pools are created lazily on first access or eagerly from config.
    Provides hybrid search (BM25 + vector) with 4-tier graceful degradation:
    1. Full hybrid (BM25 + vector): bm25s + embedding model configured
    2. BM25 only: bm25s installed, no embedding model
    3. Keyword fallback: neither available, simple word overlap
    4. Chronological: get_recent() always works
    """

    def __init__(self, *, llm_client: Any = None, alpha: float = 0.6) -> None:
        """
        Args:
            llm_client: Optional FrameworkLLMClient for embedding-based search.
                       Typed as Any to avoid circular import.
            alpha: Weight for BM25 vs vector in hybrid search (0=vector only, 1=BM25 only).
        """
        self._pools: dict[str, PoolState] = {}
        self._llm_client = llm_client
        self._alpha = alpha

    def initialize_from_configs(self, configs: list[dict[str, Any]]) -> None:
        """Create pools from config dicts (from WorkflowDefinition.pools)."""
        for cfg_dict in configs:
            pool_config = PoolConfig(**cfg_dict)
            if pool_config.name in self._pools:
                logger.warning("POOL_REGISTRY_DUPLICATE name=%s", pool_config.name)
                continue
            self._pools[pool_config.name] = PoolState(pool_config)
            logger.debug("POOL_REGISTRY_INIT name=%s", pool_config.name)

    def get(self, name: str) -> PoolState:
        """Get pool by name. Raises KeyError if not found."""
        if name not in self._pools:
            raise KeyError(f"Pool '{name}' not found. Available: {list(self._pools.keys())}")
        return self._pools[name]

    def get_or_create(self, name: str, **kwargs: Any) -> PoolState:
        """Get existing pool or create with defaults."""
        if name not in self._pools:
            config = PoolConfig(name=name, **kwargs)
            self._pools[name] = PoolState(config)
            logger.debug("POOL_REGISTRY_LAZY_CREATE name=%s", name)
        return self._pools[name]

    def has(self, name: str) -> bool:
        """Check if a pool exists in the registry."""
        return name in self._pools

    def all_pools(self) -> dict[str, PoolState]:
        """Return a shallow copy of all registered pools."""
        return dict(self._pools)

    async def search(self, pool_name: str, query: str, top_k: int = 10) -> list[Any]:
        """Search a pool with best available method (hybrid > BM25 > keyword > recent).

        Graceful degradation:
        - If bm25s is installed and an embedding model is configured: hybrid search.
        - If bm25s is installed but no embedding model: BM25 only.
        - If bm25s is not installed: simple keyword overlap (PoolState.search).
        - If the pool is empty or query yields no results: get_recent() fallback.
        """
        pool = self.get(pool_name)

        if pool.count() == 0:
            return []

        # Tier 1 & 2: BM25 (possibly combined with vector)
        if _HAS_BM25 and pool.count() > 0:
            bm25_results = self._bm25_search(pool, query, top_k)

            # Tier 1: Full hybrid -- re-rank with vector similarity
            if (
                self._llm_client is not None
                and getattr(self._llm_client, "supports_embeddings", False)
                and bm25_results
            ):
                try:
                    return await self._hybrid_rerank(bm25_results, query, top_k)
                except Exception:
                    logger.warning("POOL_SEARCH_VECTOR_FAILED pool=%s", pool_name, exc_info=True)
                    # Fall through to BM25-only results

            if bm25_results:
                return bm25_results

        # Tier 3: Keyword fallback
        keyword_results = pool.search(query, limit=top_k)
        if keyword_results:
            return keyword_results

        # Tier 4: Chronological fallback
        return pool.get_recent(top_k)

    # -- Private helpers -----------------------------------------------------

    @staticmethod
    def _item_to_text(item: Any) -> str:
        """Convert a pool item to a searchable text string."""
        if isinstance(item, str):
            return item
        return json.dumps(item, default=str)

    def _bm25_search(self, pool: PoolState, query: str, top_k: int) -> list[Any]:
        """Run BM25 search over pool items. Returns scored results."""
        if not _HAS_BM25:
            return []

        corpus = [self._item_to_text(item) for item in pool.items]
        if not corpus:
            return []

        tokenized_corpus = bm25s.tokenize(corpus)
        retriever = bm25s.BM25()
        retriever.index(tokenized_corpus)

        tokenized_query = bm25s.tokenize([query])
        effective_k = min(top_k, len(corpus))
        results, scores = retriever.retrieve(tokenized_query, corpus=corpus, k=effective_k)

        # results and scores are 2D arrays (one row per query)
        scored_items: list[tuple[float, Any]] = []
        for doc, score in zip(results[0], scores[0], strict=True):
            if score > 0:
                scored_items.append((float(score), doc))

        scored_items.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored_items]

    async def _hybrid_rerank(
        self, bm25_results: list[Any], query: str, top_k: int
    ) -> list[Any]:
        """Re-rank BM25 results using vector similarity for hybrid scoring."""
        texts = [self._item_to_text(item) for item in bm25_results]
        all_texts = [query] + texts
        embeddings = await self._llm_client.embed(all_texts)

        query_vec = np.array(embeddings[0])
        doc_vecs = np.array(embeddings[1:])

        # Cosine similarity
        query_norm = np.linalg.norm(query_vec)
        doc_norms = np.linalg.norm(doc_vecs, axis=1)
        # Avoid division by zero
        doc_norms = np.where(doc_norms == 0, 1.0, doc_norms)
        cosine_scores = doc_vecs @ query_vec / (doc_norms * query_norm)

        # Normalize BM25 rank scores to [0, 1] (rank-based, not raw BM25 scores)
        n = len(bm25_results)
        bm25_rank_scores = np.array([(n - i) / n for i in range(n)])

        # Hybrid score: alpha * BM25_rank + (1 - alpha) * cosine
        hybrid_scores = self._alpha * bm25_rank_scores + (1 - self._alpha) * cosine_scores

        ranked_indices = np.argsort(-hybrid_scores)
        return [bm25_results[i] for i in ranked_indices[:top_k]]
