"""File search tool — searches a pre-built file index for relevant passages.

Implements the ``ResearchTool`` protocol.  The actual file storage and
indexing are handled by the caller — this tool only needs a ``FileIndex``
that supports a ``search(query, top_k)`` method.

Uses BM25 for ranking when the ``[search]`` extra is installed (``bm25s``),
falling back to simple keyword overlap otherwise.

Example usage::

    index = MyFileIndex(chunks)  # implements FileIndex protocol
    tool = FileSearchTool(index, top_k=10)
    result = await tool.execute({"query": "neural networks"}, context)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Protocol, runtime_checkable

from databricks_deep_research.tools.protocol import (
    SourceInfo,
    ToolContext,
    ToolDefinition,
    ToolResult,
)

__all__ = ["FileSearchTool"]

logger = logging.getLogger(__name__)

# Optional BM25 dependency — degrade gracefully when not installed.
_HAS_BM25 = False
try:
    import bm25s

    _HAS_BM25 = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# FileIndex protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class FileIndex(Protocol):
    """Protocol for a pre-built file index that the tool searches over.

    Each item returned by ``get_chunks`` must be a dict with at least:
      - ``content`` (str): The chunk text.
      - ``source`` (str): Human-readable source identifier (e.g. filename).

    Optional fields: ``chunk_index`` (int), ``page_number`` (int),
    ``section`` (str), ``file_id`` (str).
    """

    def get_chunks(self) -> list[dict[str, Any]]:
        """Return all indexed chunks for search."""
        ...


# ---------------------------------------------------------------------------
# FileSearchTool
# ---------------------------------------------------------------------------


class FileSearchTool:
    """Search user-provided files for relevant passages.

    Constructor DI — the caller provides a ``FileIndex`` with the documents
    already chunked and ready for search.  This tool handles ranking
    (BM25 or keyword fallback) and result formatting.
    """

    def __init__(self, file_index: FileIndex, *, top_k: int = 5) -> None:
        self._index = file_index
        self._top_k = top_k

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="file_search",
            description=(
                "Search through user-provided documents and files. "
                "Returns matching passages with source information."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to find relevant content in files.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": f"Maximum results to return (default: {self._top_k}).",
                        "default": self._top_k,
                    },
                },
                "required": ["query"],
            },
            source_type="file",
            source_kind="file",
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Validate and clean arguments. Raises ValueError on invalid input."""
        query = arguments.get("query")
        if not query or not isinstance(query, str):
            raise ValueError("'query' is required and must be a non-empty string.")
        query = query.strip()
        if len(query) < 2:
            raise ValueError("'query' must be at least 2 characters.")
        if len(query) > 500:
            raise ValueError("'query' must be 500 characters or less.")

        top_k = arguments.get("top_k", self._top_k)
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("'top_k' must be a positive integer.")
        top_k = min(top_k, 50)

        return {"query": query, "top_k": top_k}

    async def execute(
        self, arguments: dict[str, Any], context: ToolContext  # noqa: ARG002
    ) -> ToolResult:
        """Search the file index and return matching passages."""
        query: str = arguments["query"]
        top_k: int = arguments["top_k"]

        chunks = self._index.get_chunks()
        if not chunks:
            return ToolResult(
                content=f"No files available to search for: {query}",
                data={"query": query, "num_results": 0},
            )

        scored = self._rank(query, chunks, top_k)

        if not scored:
            return ToolResult(
                content=f"No results found in files for: {query}",
                data={"query": query, "num_results": 0},
            )

        # Build sources and formatted output
        sources: list[SourceInfo] = []
        lines: list[str] = []

        for idx, (score, chunk) in enumerate(scored):
            source_label = chunk.get("source", "unknown")
            file_id = chunk.get("file_id", "")
            chunk_index = chunk.get("chunk_index", idx)
            page = chunk.get("page_number")
            content_text: str = chunk.get("content", "")

            url = f"file://{file_id}#chunk-{chunk_index}" if file_id else f"file://{source_label}"
            sources.append(SourceInfo(
                url=url,
                title=source_label,
                snippet=content_text[:300],
                source_type="file",
            ))

            location = f" (page {page})" if page else ""
            highlight = self._highlight(content_text, query)
            lines.append(
                f"[{idx}] {source_label}{location} (score: {score:.2f})\n"
                f"    {highlight}"
            )

        return ToolResult(
            content="\n\n".join(lines),
            sources=sources,
            data={"query": query, "num_results": len(scored)},
        )

    # -- Ranking -------------------------------------------------------------

    def _rank(
        self, query: str, chunks: list[dict[str, Any]], top_k: int
    ) -> list[tuple[float, dict[str, Any]]]:
        """Rank chunks by relevance. Uses BM25 if available, else keyword overlap."""
        if _HAS_BM25:
            return self._bm25_rank(query, chunks, top_k)
        return self._keyword_rank(query, chunks, top_k)

    @staticmethod
    def _bm25_rank(
        query: str, chunks: list[dict[str, Any]], top_k: int
    ) -> list[tuple[float, dict[str, Any]]]:
        """Rank using BM25 via bm25s library."""
        corpus = [c.get("content", "") for c in chunks]
        tokenized_corpus = bm25s.tokenize(corpus)
        retriever = bm25s.BM25()
        retriever.index(tokenized_corpus)

        tokenized_query = bm25s.tokenize([query])
        effective_k = min(top_k, len(corpus))
        results, scores = retriever.retrieve(
            tokenized_query, corpus=corpus, k=effective_k
        )

        scored: list[tuple[float, dict[str, Any]]] = []
        for doc_text, score in zip(results[0], scores[0], strict=True):
            if score > 0:
                # Map the retrieved text back to its chunk dict
                for chunk in chunks:
                    if chunk.get("content", "") == doc_text:
                        scored.append((float(score), chunk))
                        break

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    @staticmethod
    def _keyword_rank(
        query: str, chunks: list[dict[str, Any]], top_k: int
    ) -> list[tuple[float, dict[str, Any]]]:
        """Simple keyword overlap ranking (fallback when bm25s not installed)."""
        query_tokens = set(re.findall(r"\b\w{3,}\b", query.lower()))
        if not query_tokens:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        for chunk in chunks:
            text = chunk.get("content", "").lower()
            matches = sum(1 for t in query_tokens if t in text)
            if matches > 0:
                scored.append((matches / len(query_tokens), chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    # -- Highlighting --------------------------------------------------------

    @staticmethod
    def _highlight(content: str, query: str, context_chars: int = 150) -> str:
        """Create a snippet around the first query term match."""
        content_lower = content.lower()
        terms = re.findall(r"\b\w{3,}\b", query.lower())

        best_pos = -1
        best_len = 0
        for term in terms:
            pos = content_lower.find(term)
            if pos != -1 and (best_pos == -1 or pos < best_pos):
                best_pos = pos
                best_len = len(term)

        if best_pos == -1:
            return content[:context_chars * 2] + ("..." if len(content) > context_chars * 2 else "")

        start = max(0, best_pos - context_chars)
        end = min(len(content), best_pos + best_len + context_chars)
        snippet = content[start:end]

        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."

        return snippet
