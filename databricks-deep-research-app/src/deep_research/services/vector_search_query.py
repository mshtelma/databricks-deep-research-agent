"""Centralized Vector Search query execution using WorkspaceClient SDK.

Replaces direct VectorSearchClient usage with the official
WorkspaceClient.vector_search_indexes.query_index() API for consistent
authentication across local dev (profiles), deployed apps (service principal),
and OBO (user tokens).

All three call sites (UserVectorSearchTool, VectorSearchTool, background.py)
delegate to this service.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from databricks.sdk import WorkspaceClient

from deep_research.core.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ColumnRoles:
    """Deterministic mapping from index metadata to VectorSearchResult fields.

    Created by extract_queryable_columns() from a VectorIndex object.
    Used by _parse_response() to map arbitrary column names to semantic roles.
    """

    id_column: str
    """Primary key column — always known from index metadata."""

    content_column: str | None
    """First embedding_source_column — the embedded text."""

    all_columns: list[str]
    """Columns safe to pass to query_index()."""


def extract_queryable_columns(index: Any) -> ColumnRoles | None:
    """Extract queryable columns and their roles from a VectorIndex object.

    Deterministic logic:
    - id_column: always the index's primary_key
    - content_column: first embedding_source_column (the text that was embedded)
    - all_columns: primary_key + embedding_source_columns (DELTA_SYNC)
                   OR all schema columns (DIRECT_ACCESS via schema_json)
                   MINUS embedding_vector_columns (float arrays, not useful)

    Args:
        index: VectorIndex from client.vector_search_indexes.get_index()

    Returns:
        ColumnRoles if extraction succeeded, None otherwise.
    """
    primary_key = getattr(index, "primary_key", None)
    if not primary_key:
        return None

    spec = getattr(index, "delta_sync_index_spec", None) or \
        getattr(index, "direct_access_index_spec", None)
    if not spec:
        # No spec available — return minimal (just primary_key)
        return ColumnRoles(
            id_column=primary_key,
            content_column=None,
            all_columns=[primary_key],
        )

    columns: list[str] = []

    # 1. For DIRECT_ACCESS: extract ALL columns from schema_json
    schema_json_str = getattr(spec, "schema_json", None)
    if schema_json_str:
        try:
            schema = json.loads(schema_json_str)
            if isinstance(schema, dict):
                columns = list(schema.keys())
        except (json.JSONDecodeError, TypeError):
            pass

    # 2. Extract embedding source column names (the text that was embedded)
    embedding_source_names: list[str] = []
    if hasattr(spec, "embedding_source_columns") and spec.embedding_source_columns:
        for col in spec.embedding_source_columns:
            name = getattr(col, "name", None)
            if name:
                embedding_source_names.append(name)
                if name not in columns:
                    columns.append(name)

    # 3. Add primary key if not already present
    if primary_key not in columns:
        columns.insert(0, primary_key)

    # 3b. For DELTA_SYNC: include columns_to_sync (covers self-managed embeddings
    # where embedding_source_columns is empty).
    if hasattr(spec, "columns_to_sync") and spec.columns_to_sync:
        for col in spec.columns_to_sync:
            col_name = col.name if hasattr(col, "name") else str(col)
            if col_name and col_name not in columns:
                columns.append(col_name)

    # 4. Exclude embedding vector columns (array<float>) — useless for text results
    vector_col_names: set[str] = set()
    if hasattr(spec, "embedding_vector_columns") and spec.embedding_vector_columns:
        for col in spec.embedding_vector_columns:
            name = getattr(col, "name", None)
            if name:
                vector_col_names.add(name)
    columns = [c for c in columns if c not in vector_col_names]

    # 5. Determine content column (first embedding source that's still in columns)
    content_column: str | None = None
    for name in embedding_source_names:
        if name in columns and name != primary_key:
            content_column = name
            break

    # 5b. Heuristic fallback for self-managed embeddings where the API
    # doesn't identify the text column via embedding_source_columns.
    if content_column is None:
        _CONTENT_COL_CANDIDATES = (
            "content", "text", "chunk_text", "chunk_content", "page_content",
            "body", "passage", "document", "chunk", "paragraph",
        )
        for candidate in _CONTENT_COL_CANDIDATES:
            if candidate in columns and candidate != primary_key:
                content_column = candidate
                break

    return ColumnRoles(
        id_column=primary_key,
        content_column=content_column,
        all_columns=columns,
    )


@dataclass
class VectorSearchResult:
    """Standardized result from vector search."""

    id: str
    title: str
    content: str
    url: str | None
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorSearchQueryService:
    """Centralized vector search query execution using WorkspaceClient SDK.

    Uses ``client.vector_search_indexes.query_index()`` which:
    - Resolves the endpoint automatically from the three-level index name
    - Inherits auth from the WorkspaceClient (profile, SP, or OBO token)
    - Supports ``columns_to_rerank`` natively (no DatabricksReranker needed)
    """

    def query_sync(
        self,
        client: WorkspaceClient,
        index_name: str,
        query_text: str,
        columns: list[str],
        num_results: int = 10,
        query_type: str | None = None,
        filters_json: str | None = None,
        score_threshold: float | None = None,
        columns_to_rerank: list[str] | None = None,
        column_roles: ColumnRoles | None = None,
    ) -> list[VectorSearchResult]:
        """Execute a vector search query synchronously.

        This is the synchronous core called via ``run_in_executor`` from
        the async ``query()`` wrapper.

        Args:
            client: WorkspaceClient with appropriate auth.
            index_name: Fully qualified index name (catalog.schema.index).
            query_text: Natural language search query.
            columns: Columns to return.
            num_results: Number of results.
            query_type: "ANN", "HYBRID", or None (default ANN).
            filters_json: JSON-encoded filter string for the API.
            score_threshold: Minimum score cutoff.
            columns_to_rerank: Text columns for SDK-native reranking.
            column_roles: Deterministic column-to-field mapping from index metadata.

        Returns:
            Parsed list of VectorSearchResult.
        """
        kwargs: dict[str, Any] = {
            "index_name": index_name,
            "query_text": query_text,
            "columns": columns,
            "num_results": num_results,
        }
        if query_type:
            kwargs["query_type"] = query_type
        if filters_json:
            kwargs["filters_json"] = filters_json
        if score_threshold is not None:
            kwargs["score_threshold"] = score_threshold
        if columns_to_rerank:
            kwargs["columns_to_rerank"] = columns_to_rerank

        response = client.vector_search_indexes.query_index(**kwargs)

        return self._parse_response(response, columns, column_roles)

    async def query(
        self,
        client: WorkspaceClient,
        index_name: str,
        query_text: str,
        columns: list[str],
        num_results: int = 10,
        query_type: str | None = None,
        filters_json: str | None = None,
        score_threshold: float | None = None,
        columns_to_rerank: list[str] | None = None,
        column_roles: ColumnRoles | None = None,
    ) -> list[VectorSearchResult]:
        """Execute a vector search query asynchronously.

        Wraps the synchronous SDK call in ``run_in_executor``.

        Args:
            client: WorkspaceClient with appropriate auth.
            index_name: Fully qualified index name (catalog.schema.index).
            query_text: Natural language search query.
            columns: Columns to return.
            num_results: Number of results.
            query_type: "ANN", "HYBRID", or None (default ANN).
            filters_json: JSON-encoded filter string for the API.
            score_threshold: Minimum score cutoff.
            columns_to_rerank: Text columns for SDK-native reranking.
            column_roles: Deterministic column-to-field mapping from index metadata.

        Returns:
            Parsed list of VectorSearchResult.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.query_sync(
                client=client,
                index_name=index_name,
                query_text=query_text,
                columns=columns,
                num_results=num_results,
                query_type=query_type,
                filters_json=filters_json,
                score_threshold=score_threshold,
                columns_to_rerank=columns_to_rerank,
                column_roles=column_roles,
            ),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(
        response: Any,
        requested_columns: list[str],
        column_roles: ColumnRoles | None = None,
    ) -> list[VectorSearchResult]:
        """Parse QueryVectorIndexResponse into VectorSearchResult list.

        When column_roles is provided, uses deterministic mapping:
        - roles.id_column -> result.id
        - roles.content_column -> result.content
        - "score" -> result.score (always present in SDK response)
        - Everything else -> result.metadata

        When column_roles is None (backward compat), uses legacy hardcoded names.

        Handles both SDK response objects (attribute access) and plain
        dicts (for testing / older SDK versions).
        """
        # Extract manifest columns
        manifest = getattr(response, "manifest", None)
        if manifest is None and isinstance(response, dict):
            manifest = response.get("manifest")

        columns: list[str] = []
        if manifest is not None:
            raw_cols = getattr(manifest, "columns", None)
            if raw_cols is None and isinstance(manifest, dict):
                raw_cols = manifest.get("columns", [])
            if raw_cols:
                for c in raw_cols:
                    name = getattr(c, "name", None)
                    if name is None and isinstance(c, dict):
                        name = c.get("name", "")
                    columns.append(str(name))

        col_indices: dict[str, int] = {name: idx for idx, name in enumerate(columns)}

        # Extract data rows
        result_obj = getattr(response, "result", None)
        if result_obj is None and isinstance(response, dict):
            result_obj = response.get("result")

        data_array: list[list[Any]] = []
        if result_obj is not None:
            arr = getattr(result_obj, "data_array", None)
            if arr is None and isinstance(result_obj, dict):
                arr = result_obj.get("data_array", [])
            if arr:
                data_array = arr

        # Determine column mapping strategy
        if column_roles:
            id_col = column_roles.id_column
            content_col = column_roles.content_column
        else:
            # Legacy path: hardcoded names (backward compat for config-based tools)
            id_col = "id" if "id" in col_indices else None
            content_col = "content" if "content" in col_indices else None

        score_col = "score" if "score" in col_indices else None

        results: list[VectorSearchResult] = []
        for row_idx, row in enumerate(data_array):
            # ID: from roles or legacy
            doc_id = _safe_col(row, col_indices, id_col, f"doc_{row_idx}") if id_col else f"doc_{row_idx}"

            # Content: from roles or legacy
            content = _safe_col(row, col_indices, content_col, "") if content_col else ""

            # Score: always "score"
            score = _safe_col(row, col_indices, score_col, 0.0) if score_col else 0.0

            # Title: use "title" column if present, else derive from content
            title_from_col = _safe_col(row, col_indices, "title", None)
            if title_from_col:
                title = str(title_from_col)
            elif content:
                first_line = str(content).split("\n")[0][:80]
                title = first_line + "..." if len(str(content)) > 80 else first_line
            else:
                title = "Untitled"

            # URL: check for "url" column, else None
            url = _safe_col(row, col_indices, "url", None)

            # Metadata: everything not already mapped
            mapped_cols = {id_col, content_col, score_col, "title", "url"}
            metadata: dict[str, Any] = {}
            for col_name, col_idx in col_indices.items():
                if col_name not in mapped_cols and col_idx < len(row):
                    val = row[col_idx]
                    if val is not None:
                        metadata[col_name] = val

            results.append(
                VectorSearchResult(
                    id=str(doc_id),
                    title=title,
                    content=str(content) if content else "",
                    url=url,
                    score=float(score) if score else 0.0,
                    metadata=metadata,
                )
            )

        return results

    @staticmethod
    def build_filters_json(
        filters_sql: str | None = None,
        filters_dict: dict[str, Any] | None = None,
    ) -> str | None:
        """Convert filter representations to the ``filters_json`` string
        expected by ``query_index()``.

        The API accepts a JSON-encoded object like:
          ``{"column_name LIKE": "pattern"}`` or ``{"column_name": ["val1", "val2"]}``

        Args:
            filters_sql: SQL-like filter string (takes precedence).
            filters_dict: Dictionary-based filters.

        Returns:
            JSON string or None.
        """
        if filters_dict:
            return json.dumps(filters_dict)
        # For SQL filters, we wrap into a simple JSON string representation.
        # The query_index API also accepts filters_json as a JSON-encoded dict.
        # Callers should prefer build_filters_json(filters_dict=...) when possible.
        if filters_sql:
            return filters_sql
        return None


def _safe_col(
    row: list[Any],
    col_indices: dict[str, int],
    column: str,
    default: Any,
) -> Any:
    """Safely get column value from row."""
    idx = col_indices.get(column)
    if idx is not None and idx < len(row):
        return row[idx] if row[idx] is not None else default
    return default
