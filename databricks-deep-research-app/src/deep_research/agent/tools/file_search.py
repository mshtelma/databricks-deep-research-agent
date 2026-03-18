"""File Search Tool for user-uploaded documents.

Provides keyword search across uploaded file chunks for research.
Returns citations with filename, chunk location, and matched content.

Part of 007-enterprise-data-sources feature (T089).
"""

import re
from typing import Any
from uuid import UUID

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from deep_research.agent.tools.base import (
    ResearchContext,
    ToolDefinition,
    ToolResult,
)
from deep_research.core.logging_utils import get_logger
from deep_research.models.uploaded_file import FileChunk, FileProcessingStatus, UploadedFile

logger = get_logger(__name__)


class FileSearchTool:
    """
    File search tool implementing the ResearchTool protocol.

    Searches across user-uploaded file chunks using simple keyword matching.
    Returns results with source citations for integration with research.

    Requires database session for chunk access and optionally filters by session.
    """

    def __init__(
        self,
        session: AsyncSession,
        *,
        owner_id: str | None = None,
        session_id: UUID | None = None,
        file_ids: list[str] | None = None,
        max_results: int = 10,
    ) -> None:
        """Initialize the file search tool.

        Args:
            session: Async SQLAlchemy session for database access.
            owner_id: Filter files to this owner. If None, must be set per query.
            session_id: Optional session ID to filter session-scoped files.
            file_ids: Optional explicit file IDs to limit searches to.
            max_results: Default maximum results to return.
        """
        self._db_session = session
        self._owner_id = owner_id
        self._session_id = session_id
        self._file_ids = self._normalize_file_ids(file_ids)
        self._max_results = max_results

        self._definition = ToolDefinition(
            name="file_search",
            description=(
                "Search through user-uploaded documents (PDFs, text files, Markdown, Word docs). "
                "Use this to find relevant information in files the user has provided. "
                "Returns matched passages with file name and location."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search query. Use specific keywords or phrases to find relevant content. "
                            "Searches across all text in uploaded documents."
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "description": f"Maximum number of results to return (default: {max_results})",
                        "default": max_results,
                    },
                },
                "required": ["query"],
            },
            source_type="file_search",
        )

    @property
    def definition(self) -> ToolDefinition:
        """Return tool definition for LLM function calling."""
        return self._definition

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ResearchContext,
    ) -> ToolResult:
        """Execute file search and return results.

        Args:
            arguments: Tool arguments containing 'query' and optional 'max_results'
            context: Research context with user identity and session info

        Returns:
            ToolResult with formatted search results and source tracking
        """
        query = arguments.get("query", "")
        max_results = arguments.get("max_results", self._max_results)

        # Determine owner_id - prefer context, fall back to configured
        owner_id = context.user_id or self._owner_id
        if not owner_id:
            return ToolResult(
                content="Cannot search files: user identity not available.",
                success=False,
                error="Missing user identity for file search",
            )

        try:
            # Search files
            results = await self._search_files(
                query=query,
                owner_id=owner_id,
                session_id=self._session_id,
                file_ids=self._file_ids,
                max_results=max_results,
            )

            if not results:
                return ToolResult(
                    content=f"No results found in uploaded files for query: {query}",
                    success=True,
                    sources=[],
                    data={"query": query, "num_results": 0},
                )

            # Build sources list for citation tracking
            sources: list[dict[str, Any]] = []
            formatted_results: list[str] = []

            for idx, result in enumerate(results):
                # Build source for citation
                sources.append({
                    "type": "uploaded_file",
                    "url": f"uploaded-file://{result['file_id']}#chunk-{result['chunk_index']}",
                    "file_id": str(result["file_id"]),
                    "filename": result["filename"],
                    "chunk_id": str(result["chunk_id"]),
                    "chunk_index": result["chunk_index"],
                    "content": result["content"][:500],
                    "relevance_score": result["score"],
                    "page_number": result.get("page_number"),
                    "section": result.get("section"),
                    "search_index": idx,
                })

                # Format result for LLM
                location = f" (page {result['page_number']})" if result.get("page_number") else ""
                formatted_results.append(
                    f"[{idx}] **{result['filename']}**{location} (score: {result['score']:.2f})\n"
                    f"    {result['highlight'] or result['content'][:300]}..."
                )

            content = "\n\n".join(formatted_results)

            return ToolResult(
                content=content,
                success=True,
                sources=sources,
                data={
                    "query": query,
                    "num_results": len(results),
                },
            )

        except Exception as e:
            logger.error("File search error", error=str(e))
            return ToolResult(
                content=f"File search failed: {e}",
                success=False,
                error=str(e),
            )

    async def _search_files(
        self,
        query: str,
        owner_id: str,
        session_id: UUID | None = None,
        file_ids: set[UUID] | None = None,
        max_results: int = 10,
    ) -> list[dict[str, Any]]:
        """Search file chunks using keyword matching.

        Uses simple text matching (case-insensitive). For better results,
        consider implementing TF-IDF or BM25 scoring.

        Args:
            query: Search query string.
            owner_id: Owner user ID to filter files.
            session_id: Optional session ID to filter.
            file_ids: Optional explicit file IDs to search.
            max_results: Maximum results to return.

        Returns:
            List of search results with file info and matching content.
        """
        # Build file filter conditions
        file_conditions = [
            UploadedFile.owner_id == owner_id,
            UploadedFile.processing_status == FileProcessingStatus.READY.value,
        ]
        if session_id is not None:
            file_conditions.append(UploadedFile.session_id == session_id)
        if file_ids:
            file_conditions.append(UploadedFile.id.in_(file_ids))

        # Get eligible files
        file_result = await self._db_session.execute(
            select(UploadedFile).where(and_(*file_conditions))
        )
        files = {f.id: f for f in file_result.scalars().all()}

        if not files:
            return []

        # Search chunks
        # Note: This is a naive implementation. For production, consider:
        # - PostgreSQL full-text search (tsvector)
        # - Embedding-based similarity search
        # - BM25 scoring

        chunk_result = await self._db_session.execute(
            select(FileChunk).where(
                FileChunk.file_id.in_(files.keys())
            )
        )
        chunks = list(chunk_result.scalars().all())

        # Score chunks by keyword matching
        query_terms = self._tokenize(query.lower())
        scored_results: list[dict[str, Any]] = []

        for chunk in chunks:
            if chunk.file_id not in files:
                continue

            chunk_text = chunk.content.lower()
            chunk_terms = set(self._tokenize(chunk_text))

            # Simple TF-IDF-like scoring
            score = 0.0
            matched_terms: list[str] = []

            for term in query_terms:
                if term in chunk_text:
                    # Term frequency (normalized)
                    tf = chunk_text.count(term) / max(len(chunk_terms), 1)
                    # Boost for exact phrase match
                    if term in chunk_terms:
                        tf *= 1.5
                    score += tf
                    matched_terms.append(term)

            if score > 0:
                uploaded_file = files[chunk.file_id]
                highlight = self._create_highlight(chunk.content, matched_terms)

                scored_results.append({
                    "file_id": chunk.file_id,
                    "filename": uploaded_file.filename,
                    "chunk_id": chunk.id,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "score": score,
                    "page_number": chunk.metadata_.get("page_number"),
                    "section": chunk.metadata_.get("section"),
                    "highlight": highlight,
                })

        # Sort by score and limit
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:max_results]

    @staticmethod
    def _normalize_file_ids(file_ids: list[str] | None) -> set[UUID] | None:
        """Normalize file ID strings to UUIDs, dropping invalid values."""
        if not file_ids:
            return None

        normalized: set[UUID] = set()
        for file_id in file_ids:
            try:
                normalized.add(UUID(str(file_id)))
            except (TypeError, ValueError):
                logger.warning(
                    "FILE_SEARCH_INVALID_FILE_ID",
                    file_id=str(file_id),
                )

        return normalized or None

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization for search.

        Args:
            text: Text to tokenize.

        Returns:
            List of tokens (words).
        """
        # Split on non-word characters
        tokens = re.findall(r"\b\w+\b", text)
        # Filter very short tokens
        return [t for t in tokens if len(t) > 2]

    def _create_highlight(
        self,
        content: str,
        matched_terms: list[str],
        context_chars: int = 150,
    ) -> str | None:
        """Create a highlighted snippet around matched terms.

        Args:
            content: Full content text.
            matched_terms: Terms that were matched.
            context_chars: Characters to include around match.

        Returns:
            Highlighted snippet or None.
        """
        if not matched_terms:
            return None

        content_lower = content.lower()
        best_pos = -1
        best_term = ""

        # Find first match position
        for term in matched_terms:
            pos = content_lower.find(term.lower())
            if pos != -1 and (best_pos == -1 or pos < best_pos):
                best_pos = pos
                best_term = term

        if best_pos == -1:
            return None

        # Extract context around match
        start = max(0, best_pos - context_chars)
        end = min(len(content), best_pos + len(best_term) + context_chars)

        snippet = content[start:end]

        # Add ellipsis if truncated
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."

        return snippet

    def validate_arguments(self, arguments: dict[str, Any]) -> list[str]:
        """Validate search arguments.

        Args:
            arguments: Raw arguments from LLM

        Returns:
            List of error messages (empty if valid)
        """
        errors: list[str] = []

        # Required: query
        query = arguments.get("query")
        if not query:
            errors.append("'query' is required")
        elif not isinstance(query, str):
            errors.append("'query' must be a string")
        elif len(query) > 500:
            errors.append("'query' must be 500 characters or less")
        elif len(query) < 2:
            errors.append("'query' must be at least 2 characters")

        # Optional: max_results
        max_results = arguments.get("max_results")
        if max_results is not None:
            if not isinstance(max_results, int):
                errors.append("'max_results' must be an integer")
            elif max_results < 1 or max_results > 50:
                errors.append("'max_results' must be between 1 and 50")

        return errors


def create_file_search_tool(
    session: AsyncSession,
    owner_id: str,
    session_id: UUID | None = None,
    file_ids: list[str] | None = None,
) -> FileSearchTool:
    """Factory function to create a FileSearchTool.

    Args:
        session: Database session.
        owner_id: User ID for file filtering.
        session_id: Optional session ID for session-scoped files.
        file_ids: Optional explicit file IDs to limit searches to.

    Returns:
        Configured FileSearchTool instance.
    """
    return FileSearchTool(
        session=session,
        owner_id=owner_id,
        session_id=session_id,
        file_ids=file_ids,
    )
