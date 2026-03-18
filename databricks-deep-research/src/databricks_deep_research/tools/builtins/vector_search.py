"""Databricks Vector Search tool — queries vector search indexes.

Wraps the Databricks SDK ``VectorSearchIndexes.query_index()`` API into the
``ResearchTool`` protocol.  Configuration (index name, columns, query type)
is injected via the constructor; the LLM only provides a query string.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from databricks_deep_research.tools.protocol import (
    SourceInfo,
    SourceKind,
    ToolContext,
    ToolDefinition,
    ToolResult,
)

logger = logging.getLogger(__name__)

_PAGE_CHUNK_SUFFIX = re.compile(r'_page\d+(?:_chunk\d+)?$')
_PDF_SUFFIX = re.compile(r'_?_?pdf$', re.IGNORECASE)
_PATH_KEYWORDS = frozenset({
    'dbfs', 'volumes', 'mnt', 'workspace', 'users', 'documents',
    'uploads', 'upload', 'data', 'files', 'shared',
})


def _title_from_chunk_id(chunk_id: str) -> str:
    """Best-effort: extract a human-readable document title from a VS chunk_id.

    Handles Databricks auto-chunking paths like:
      dbfs__Volumes_users_joe_documents_upload__Sales_Battlecard__AWS_pdf_page13_chunk0
    → "Sales Battlecard: AWS (p.13)"

    Returns empty string if no meaningful title can be extracted.
    """
    if not chunk_id or '_pdf' not in chunk_id.lower():
        return ""

    page_m = re.search(r'_page(\d+)', chunk_id)
    page = page_m.group(1) if page_m else None

    base = _PAGE_CHUNK_SUFFIX.sub('', chunk_id)
    base = _PDF_SUFFIX.sub('', base)

    segments = [s for s in base.split('__') if s]

    doc_segments: list[str] = []
    for seg in segments:
        seg_lower = seg.lower()
        is_path = any(kw in seg_lower for kw in _PATH_KEYWORDS)
        if is_path:
            doc_segments = []
        else:
            doc_segments.append(seg)

    if not doc_segments:
        return ""

    title = ': '.join(seg.replace('_', ' ') for seg in doc_segments)
    if len(title) < 3:
        return ""

    return f"{title} (p.{page})" if page else title


class DatabricksVectorSearchTool:
    """Queries a Databricks Vector Search index.

    Implements the :class:`ResearchTool` protocol.
    """

    def __init__(
        self,
        workspace_client: Any,
        name: str,
        index_name: str,
        columns: list[str] | None = None,
        num_results: int = 10,
        query_type: str | None = None,
        filters_json: str | None = None,
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._ws = workspace_client
        self._name = name
        self._index_name = index_name
        self._columns = columns
        self._num_results = num_results
        self._query_type = query_type
        self._filters_json = filters_json
        self._description = description or f"Vector search over {index_name}"
        self._pk_col: str | None = None
        self._content_col: str | None = None

        self._definition = ToolDefinition(
            name=name,
            description=self._description,
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query text.",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return.",
                        "default": num_results,
                    },
                    "filters": {
                        "type": "object",
                        "description": "Optional filter conditions.",
                    },
                },
                "required": ["query"],
            },
            source_type="enterprise",
            source_kind=SourceKind.vector_index,
            metadata=metadata or {},
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = arguments.get("query")
        if not query or not isinstance(query, str):
            raise ValueError("'query' is required and must be a non-empty string")
        if len(query) > 1000:
            raise ValueError("'query' must be at most 1000 characters")

        num_results = arguments.get("num_results", self._num_results)
        if not isinstance(num_results, int) or num_results < 1:
            num_results = self._num_results

        validated: dict[str, Any] = {"query": query.strip(), "num_results": num_results}
        if "filters" in arguments and arguments["filters"]:
            validated["filters"] = arguments["filters"]
        return validated

    async def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        query = arguments["query"]
        num_results = arguments.get("num_results", self._num_results)

        try:
            if self._columns is None:
                self._columns = self._discover_columns()

            kwargs: dict[str, Any] = {
                "index_name": self._index_name,
                "query_text": query,
                "num_results": num_results,
            }
            if self._columns:
                kwargs["columns"] = self._columns
            if self._query_type:
                kwargs["query_type"] = self._query_type

            # Use filters from arguments or from config
            filters = arguments.get("filters")
            if filters:
                import json as _json
                kwargs["filters_json"] = _json.dumps(filters)
            elif self._filters_json:
                kwargs["filters_json"] = self._filters_json

            result = self._ws.vector_search_indexes.query_index(**kwargs)

            # Format results
            lines: list[str] = []
            sources: list[SourceInfo] = []

            manifest = getattr(result, "manifest", None)
            col_names: list[str] = []
            if manifest and hasattr(manifest, "columns"):
                col_names = [c.name for c in manifest.columns if hasattr(c, "name")]

            data_array = getattr(result, "result", None)
            if data_array and hasattr(data_array, "data_array"):
                rows = data_array.data_array or []
            else:
                rows = []

            for idx, row in enumerate(rows):
                row_values = row

                def _col(name: str, default: object = "", _row: list[object] = row_values) -> object:
                    col_idx = col_names.index(name) if name in col_names else -1
                    if 0 <= col_idx < len(_row):
                        value = _row[col_idx]
                        return value if value is not None else default
                    return default

                entry_parts: list[str] = []
                source_url = f"enterprise://vector_search/{self._name}/{idx}"
                canonical_source_url: str | None = None
                content_text = ""
                _content_candidates = ("content", "text", "chunk_text", "chunk_content", "page_content")
                content_cols = (self._content_col,) + _content_candidates if self._content_col else _content_candidates
                for content_col in content_cols:
                    raw_content = _col(content_col, "")
                    if isinstance(raw_content, str) and raw_content:
                        content_text = raw_content
                        break

                source_title = ""
                for title_col in ("title", "source_title", "doc_title", "name", "document_title"):
                    raw_title = _col(title_col, "")
                    if isinstance(raw_title, str) and raw_title:
                        source_title = raw_title
                        break

                # Extract document title from primary key value (typically chunk_id)
                if not source_title and self._pk_col:
                    pk_val = _col(self._pk_col, "")
                    if isinstance(pk_val, str) and pk_val:
                        source_title = _title_from_chunk_id(pk_val)

                for col_idx, val in enumerate(row):
                    col_name = col_names[col_idx] if col_idx < len(col_names) else f"col_{col_idx}"
                    entry_parts.append(f"{col_name}: {val}")

                    # Register URL in shared registry
                    if col_name in ("url", "source_url", "doc_url") and val and isinstance(val, str):
                        canonical_source_url = val

                if not source_title:
                    source_title = f"Vector search result {idx + 1}"
                    logger.info(
                        "VS_GENERIC_TITLE tool=%s index=%s row=%d cols=%s",
                        self._name, self._index_name, idx, col_names,
                    )

                lines.append(f"[{idx + 1}] {'; '.join(entry_parts)}")

                if context.url_registry:
                    context.url_registry.register(source_url)

                if not content_text:
                    content_text = "; ".join(entry_parts)

                sources.append(SourceInfo(
                    url=source_url,
                    canonical_url=canonical_source_url or source_url,
                    title=source_title,
                    snippet=content_text[:1500],
                    content=content_text,
                    source_type="enterprise",
                    source_kind=SourceKind.vector_index,
                    relevance_score=float(_col("score", 0.0) or 0.0),
                ))

            content = "\n".join(lines) if lines else "No results found."

            logger.info(
                "VECTOR_SEARCH_RESULTS tool=%s index=%s query=%s results=%d",
                self._name, self._index_name, query[:100], len(rows),
            )

            return ToolResult(
                content=content,
                success=True,
                sources=sources,
                data={
                    "result_count": len(rows),
                    "source_kind": SourceKind.vector_index,
                    "empty_result": len(rows) == 0,
                },
            )

        except Exception as exc:
            logger.exception(
                "VECTOR_SEARCH_ERROR tool=%s index=%s query=%s",
                self._name, self._index_name, query[:100],
            )
            return ToolResult(
                content=f"Vector search failed: {exc}",
                success=False,
                error=str(exc),
            )

    def _discover_columns(self) -> list[str]:
        """Discover a minimal usable column set from index metadata."""
        try:
            index_info = self._ws.vector_search_indexes.get_index(self._index_name)
        except Exception:
            return []

        columns: list[str] = []
        primary_key = getattr(index_info, "primary_key", None)
        if primary_key:
            columns.append(str(primary_key))
            self._pk_col = str(primary_key)

        delta_sync = getattr(index_info, "delta_sync_index_spec", None)
        if delta_sync:
            pk = getattr(delta_sync, "primary_key_columns", None)
            if pk:
                for col in (pk if isinstance(pk, list) else [pk]):
                    if col and col not in columns:
                        columns.append(str(col))
                # Use first PK column if top-level primary_key was missing
                if not self._pk_col and pk:
                    first_pk = pk[0] if isinstance(pk, list) else pk
                    if first_pk:
                        self._pk_col = str(first_pk)

            src_cols = getattr(delta_sync, "embedding_source_columns", None) or []
            for sc in src_cols:
                col_name = (
                    sc.get("name") if isinstance(sc, dict)
                    else getattr(sc, "name", None)
                )
                if col_name and col_name not in columns:
                    columns.append(col_name)
                # Store first embedding source column for content extraction
                if col_name and not self._content_col:
                    self._content_col = str(col_name)

        return columns or []
