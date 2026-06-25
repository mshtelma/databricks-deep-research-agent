"""Chat-level source pool with hybrid BM25 + semantic search.

This service manages the accumulation and retrieval of sources across
all messages in a chat conversation. It enables:

1. **Source Accumulation**: Sources are persisted with chat_id for direct lookup
2. **Hybrid Search**: BM25 keyword + embedding similarity for intelligent retrieval
3. **URL Deduplication**: Same URL within a chat is upserted, not duplicated
4. **Follow-up Support**: Existing sources are available for citation in follow-ups

Usage:
    pool = ChatSourcePoolService(db, embedder=get_embedder())

    # Load existing sources for a chat
    sources = await pool.get_all_sources(chat_id)

    # Build searchable index (BM25 + embeddings)
    await pool.build_search_index(chat_id)

    # Search for relevant sources
    relevant = await pool.search("PyTorch benchmarks", limit=5)

    # Add new source (upsert by URL)
    await pool.add_or_update_source(chat_id, url, title, content)
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from deep_research.core.logging_utils import get_logger
from deep_research.models.source import Source

if TYPE_CHECKING:
    from typing import Any

    import numpy as np
    from numpy.typing import NDArray

    from deep_research.agent.tools.evidence_registry import HybridSearchIndex, HybridSearchResult
    from deep_research.services.llm.embedder import Embedder
    from deep_research.storage.factory import StorageStack

logger = get_logger(__name__)


class ChatSourcePoolService:
    """Manage chat-level source pool with hybrid BM25 + semantic search.

    This service provides efficient access to all sources accumulated across
    a chat conversation. It supports:

    - Loading all sources for a chat with O(1) lookup via chat_id
    - Building an in-memory hybrid search index (BM25 + embeddings)
    - Searching sources with hybrid fusion (α=0.6)
    - Adding/updating sources with atomic upsert by URL
    """

    def __init__(
        self,
        session: AsyncSession,
        embedder: Embedder | None = None,
        *,
        storage_stack: StorageStack | None = None,
    ):
        """Initialize the chat source pool service.

        Args:
            session: Async database session (used only on the legacy SQL path).
            embedder: Optional embedder for semantic search.
                     If None, only BM25 keyword search is used.
            storage_stack: Event-sourced storage stack. When provided, sources
                     are read from ``ChatState.sources[]`` (the store the
                     synthesizer writes to) instead of the legacy ``sources``
                     table, which is dropped on event-sourced deployments.
        """
        self._session = session
        self._embedder = embedder
        self._storage_stack = storage_stack
        self._index: HybridSearchIndex | None = None
        self._sources: list[Source] = []
        self._source_embeddings: NDArray[np.float32] | None = None

    async def get_all_sources(
        self,
        chat_id: UUID,
        limit: int = 200,
    ) -> list[Source]:
        """Get all sources for a chat.

        Sources are returned in reverse chronological order (newest first).

        Args:
            chat_id: UUID of the chat.
            limit: Maximum number of sources to return.

        Returns:
            List of Source objects for the chat.
        """
        # Event-sourced path: read from ChatState.sources[] (the store the
        # synthesizer writes to). The legacy `sources` table is dropped on
        # event-sourced deployments, so the SQL path below would raise
        # UndefinedTableError there. Mirrors the cached-first read in
        # framework_orchestrator._load_existing_sources.
        if self._storage_stack is not None:
            cached = await self._load_sources_from_cache(chat_id, limit)
            if cached is not None:
                self._sources = cached
                logger.debug(
                    "CHAT_SOURCES_LOADED",
                    chat_id=str(chat_id),
                    count=len(self._sources),
                    path="cached",
                )
                return self._sources

        query = (
            select(Source)
            .where(Source.chat_id == chat_id)
            .order_by(Source.fetched_at.desc())
            .limit(limit)
        )
        result = await self._session.execute(query)
        self._sources = list(result.scalars().all())

        logger.debug(
            "CHAT_SOURCES_LOADED",
            chat_id=str(chat_id),
            count=len(self._sources),
            path="sql",
        )

        return self._sources

    async def _load_sources_from_cache(
        self,
        chat_id: UUID,
        limit: int,
    ) -> list[Source] | None:
        """Load chat sources from the event-sourced ChatDocument.

        Builds detached ``Source`` data-holders from ``ChatState.sources[]``
        (never added to a DB session, so the in-memory BM25/embedding index and
        ``search`` path are unchanged — they only read scalar attributes).

        Returns ``None`` on a cache error so the caller can fall back to the SQL
        path; returns ``[]`` when the chat document simply has no sources (so
        the caller does NOT touch the dropped ``sources`` table).
        """
        from deep_research.storage.documents import Source as DocSource

        stack = self._storage_stack
        if stack is None:  # pragma: no cover — guarded by caller
            return None
        try:
            doc = await stack.cache.get(chat_id)
        except Exception as exc:  # noqa: BLE001 — degrade to SQL fallback
            logger.warning(
                "CHAT_SOURCES_CACHE_LOAD_FAILED",
                chat_id=str(chat_id),
                error=str(exc)[:200],
            )
            return None

        doc_sources: list[DocSource] = list(doc.state.sources)

        def _key(s: DocSource) -> tuple[int, float, int]:
            meta = s.metadata or {}
            raw = meta.get("relevance_score")
            try:
                score = float(raw) if raw is not None else None
            except (TypeError, ValueError):
                score = None
            # relevance desc (nulls last), then most-recently-used desc.
            return (0 if score is not None else 1, -(score or 0.0), -s.last_used_step)

        doc_sources.sort(key=_key)

        out: list[Source] = []
        for ds in doc_sources[:limit]:
            meta = ds.metadata or {}
            out.append(
                Source(
                    chat_id=chat_id,
                    url=ds.url,
                    title=ds.title,
                    snippet=meta.get("snippet"),
                    content=meta.get("content"),
                    relevance_score=meta.get("relevance_score"),
                    source_type=ds.source_type,
                )
            )
        return out

    async def add_or_update_source(
        self,
        chat_id: UUID,
        url: str,
        title: str | None = None,
        content: str | None = None,
        snippet: str | None = None,
        relevance_score: float | None = None,
        research_session_id: UUID | None = None,
        **kwargs: Any,
    ) -> Source:
        """Add a source to the pool or update if URL exists.

        Uses atomic upsert (ON CONFLICT) to handle race conditions.
        If the URL already exists for this chat, updates content and fetched_at.

        Args:
            chat_id: UUID of the chat.
            url: Source URL (unique per chat).
            title: Source title.
            content: Source content (truncated if >50KB).
            snippet: Short snippet from search results.
            relevance_score: Computed relevance score.
            research_session_id: ID of the research session that found this source.
            **kwargs: Additional source fields.

        Returns:
            The created or updated Source object.
        """
        # Truncate content if too long
        if content and len(content) > 50000:
            content = content[:50000]

        stmt = pg_insert(Source).values(
            chat_id=chat_id,
            research_session_id=research_session_id,
            url=url,
            title=title,
            content=content,
            snippet=snippet,
            relevance_score=relevance_score,
            **kwargs,
        ).on_conflict_do_update(
            constraint="uq_sources_chat_url",
            set_={
                "content": content,
                "snippet": snippet,
                "relevance_score": relevance_score,
                "fetched_at": func.now(),
            },
        ).returning(Source)

        result = await self._session.execute(stmt)
        source = result.scalar_one()

        logger.debug(
            "CHAT_SOURCE_UPSERTED",
            chat_id=str(chat_id),
            url=url[:80],
            source_id=str(source.id),
        )

        return source

    async def build_search_index(
        self,
        chat_id: UUID,
        compute_embeddings: bool = True,
    ) -> HybridSearchIndex:
        """Build an in-memory hybrid search index for the chat sources.

        Combines BM25 keyword search with embeddings for hybrid retrieval.
        If embedder is not available, falls back to BM25-only search.

        Args:
            chat_id: UUID of the chat.
            compute_embeddings: Whether to compute embeddings.

        Returns:
            The built HybridSearchIndex.

        Raises:
            RuntimeError: If sources haven't been loaded yet.
        """
        from deep_research.agent.tools.evidence_registry import HybridSearchIndex

        # Load sources if not already loaded
        if not self._sources:
            await self.get_all_sources(chat_id)

        if not self._sources:
            # Return empty index
            self._index = HybridSearchIndex(texts=[], embeddings=None, alpha=0.6)
            return self._index

        # Build text representation for each source
        texts = [
            self._source_to_text(source)
            for source in self._sources
        ]

        # Compute embeddings if requested and embedder available
        embeddings = None
        if compute_embeddings and self._embedder and texts:
            try:
                embeddings = await self._embedder.embed_batch(texts)
                self._source_embeddings = embeddings
                logger.debug(
                    "CHAT_SOURCE_EMBEDDINGS_COMPUTED",
                    chat_id=str(chat_id),
                    count=len(texts),
                )
            except Exception as e:
                logger.warning(
                    "CHAT_SOURCE_EMBEDDINGS_FAILED",
                    chat_id=str(chat_id),
                    error=str(e)[:100],
                )
                # Continue with BM25-only search

        self._index = HybridSearchIndex(
            texts=texts,
            embeddings=embeddings,
            alpha=0.6,  # 60% semantic, 40% keyword
        )

        logger.info(
            "CHAT_SOURCE_INDEX_BUILT",
            chat_id=str(chat_id),
            n_sources=len(self._sources),
            has_embeddings=embeddings is not None,
        )

        return self._index

    async def search(
        self,
        query: str,
        limit: int = 10,
        query_embedding: NDArray[np.float32] | None = None,
    ) -> list[Source]:
        """Search sources with hybrid BM25 + vector similarity.

        Args:
            query: Text query for search.
            limit: Maximum number of results.
            query_embedding: Pre-computed query embedding (optional).
                            If None and embedder available, computes on-demand.

        Returns:
            List of relevant Source objects sorted by hybrid score.

        Raises:
            RuntimeError: If index hasn't been built yet.
        """
        if not self._index:
            raise RuntimeError("Call build_search_index() before searching")

        # Compute query embedding if not provided and embedder available
        if query_embedding is None and self._embedder:
            try:
                query_embedding = await self._embedder.embed(query)
            except Exception as e:
                logger.warning(
                    "QUERY_EMBEDDING_FAILED",
                    query=query[:50],
                    error=str(e)[:100],
                )
                # Continue with BM25-only search

        # Search the index
        results: list[HybridSearchResult] = self._index.search(
            query=query,
            query_embedding=query_embedding,
            limit=limit,
        )

        # Map results back to Source objects
        sources = []
        for result in results:
            if result.index < len(self._sources):
                sources.append(self._sources[result.index])

        logger.debug(
            "CHAT_SOURCE_SEARCH",
            query=query[:50],
            results=len(sources),
            has_embedding=query_embedding is not None,
        )

        return sources

    def _source_to_text(self, source: Source) -> str:
        """Convert a Source to searchable text representation.

        Combines title, snippet, and content (truncated) for indexing.

        Args:
            source: The Source object.

        Returns:
            Text representation for BM25/embedding.
        """
        parts = []

        if source.title:
            parts.append(source.title)

        if source.snippet:
            parts.append(source.snippet)

        if source.content:
            # Use first 5000 chars of content for indexing
            content_preview = source.content[:5000]
            parts.append(content_preview)

        return " ".join(parts) if parts else ""

    @property
    def sources(self) -> list[Source]:
        """Get the loaded sources."""
        return self._sources

    @property
    def has_index(self) -> bool:
        """Check if the search index has been built."""
        return self._index is not None
