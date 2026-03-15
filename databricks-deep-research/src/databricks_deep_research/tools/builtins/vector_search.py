"""Databricks Vector Search tool — queries vector search indexes.

Wraps the Databricks SDK ``VectorSearchIndexes.query_index()`` API into the
``ResearchTool`` protocol.  Configuration (index name, columns, query type)
is injected via the constructor; the LLM only provides a query string.
"""

from __future__ import annotations

import logging
from typing import Any

from databricks_deep_research.tools.protocol import (
    SourceInfo,
    SourceKind,
    ToolContext,
    ToolDefinition,
    ToolResult,
)

logger = logging.getLogger(__name__)


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
    ) -> None:
        self._ws = workspace_client
        self._name = name
        self._index_name = index_name
        self._columns = columns
        self._num_results = num_results
        self._query_type = query_type
        self._filters_json = filters_json
        self._description = description or f"Vector search over {index_name}"

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
                for content_col in ("content", "text", "chunk_text", "page_content"):
                    raw_content = _col(content_col, "")
                    if isinstance(raw_content, str) and raw_content:
                        content_text = raw_content
                        break

                source_title = ""
                for title_col in ("title", "source_title", "doc_title", "name"):
                    raw_title = _col(title_col, "")
                    if isinstance(raw_title, str) and raw_title:
                        source_title = raw_title
                        break

                for col_idx, val in enumerate(row):
                    col_name = col_names[col_idx] if col_idx < len(col_names) else f"col_{col_idx}"
                    entry_parts.append(f"{col_name}: {val}")

                    # Register URL in shared registry
                    if col_name in ("url", "source_url", "doc_url") and val and isinstance(val, str):
                        canonical_source_url = val

                if not source_title:
                    source_title = f"Vector search result {idx + 1}"

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

        delta_sync = getattr(index_info, "delta_sync_index_spec", None)
        if delta_sync:
            pk = getattr(delta_sync, "primary_key_columns", None)
            if pk:
                for col in (pk if isinstance(pk, list) else [pk]):
                    if col and col not in columns:
                        columns.append(str(col))

            src_cols = getattr(delta_sync, "embedding_source_columns", None) or []
            for sc in src_cols:
                col_name = (
                    sc.get("name") if isinstance(sc, dict)
                    else getattr(sc, "name", None)
                )
                if col_name and col_name not in columns:
                    columns.append(col_name)

        return columns or []
