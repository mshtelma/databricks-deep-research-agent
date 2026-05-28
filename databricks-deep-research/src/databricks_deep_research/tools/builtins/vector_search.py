"""Databricks Vector Search tool — queries vector search indexes.

Wraps the Databricks SDK ``VectorSearchIndexes.query_index()`` API into the
``ResearchTool`` protocol.  Configuration (index name, columns, query type)
is injected via the constructor; the LLM only provides a query string.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable

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

# Comparison operators that require NUMERIC values (int/float/long/double).
# The Databricks Vector Search API rejects these with string values:
#   "Please use a numeric value: integer, float, double, long."
_COMPARISON_OPS = frozenset({'<', '<=', '>', '>='})


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
        exclude_chunk_types: list[str] | None = None,
    ) -> None:
        self._ws = workspace_client
        self._name = name
        self._index_name = index_name
        self._columns = columns
        self._num_results = num_results
        self._query_type = query_type
        self._filters_json = filters_json
        self._description = description or f"Vector search over {index_name}"
        self._exclude_chunk_types = exclude_chunk_types or []
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
                        "description": (
                            "Optional filter conditions as a JSON object. "
                            "Keys are column names optionally followed by an operator. "
                            'Supported: {"col": "val"} (equality), '
                            '{"col >": 5}, {"col >=": 5}, {"col <": 5}, {"col <=": 5} '
                            "(comparison — NUMERIC values only). "
                            "The IN, LIKE, and NOT operators are NOT supported. "
                            "Do NOT use comparison operators with string/date values."
                        ),
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

    @property
    def compute_name(self) -> str:
        return "vector_search"

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
            raw = arguments["filters"]
            if isinstance(raw, str):
                # LLM may double-encode filters as a JSON or Python dict string.
                raw = self._try_parse_filter_string(raw)
            if isinstance(raw, dict):
                normalized = self._normalize_filters(raw)
                if normalized:  # don't set empty dict — let execute() fall through to constructor default
                    validated["filters"] = normalized
            elif raw is not None:
                logger.warning(
                    "VECTOR_SEARCH_FILTER_IGNORED filters=%r (expected dict, got %s)",
                    arguments["filters"], type(raw).__name__,
                )
        return validated

    def to_compute_callable(self, *, compute: Any) -> Callable[..., list[dict[str, Any]]]:
        """Return a plain callable for use inside ``python_compute``.

        This mirrors the external vector-search query path but returns native
        row dictionaries instead of a ``ToolResult`` envelope.
        """
        del compute

        def _call(
            query: str,
            num_results: int | None = None,
            filters: dict[str, Any] | str | None = None,
        ) -> list[dict[str, Any]]:
            raw_args: dict[str, Any] = {"query": query}
            if num_results is not None:
                raw_args["num_results"] = num_results
            if filters:
                raw_args["filters"] = filters
            args = self.validate_arguments(raw_args)

            if self._columns is None:
                self._columns = self._discover_columns()
            if not self._columns:
                raise RuntimeError(
                    f"vector_search: no columns available for index "
                    f"{self._index_name!r}; set columns explicitly in the tool config"
                )

            requested = args.get("num_results", self._num_results)
            effective_num = (
                min(requested * 2, 50) if self._exclude_chunk_types else requested
            )
            kwargs: dict[str, Any] = {
                "index_name": self._index_name,
                "query_text": args["query"],
                "num_results": effective_num,
                "columns": self._columns,
            }
            if self._query_type:
                kwargs["query_type"] = self._query_type
            if args.get("filters"):
                import json as _json

                kwargs["filters_json"] = _json.dumps(args["filters"])
            elif self._filters_json:
                kwargs["filters_json"] = self._filters_json

            result = self._ws.vector_search_indexes.query_index(**kwargs)
            col_names = self._result_column_names(result)
            rows = self._result_rows(result)

            if self._exclude_chunk_types and rows and col_names:
                ct_idx = col_names.index("chunk_type") if "chunk_type" in col_names else -1
                if ct_idx >= 0:
                    rows = [r for r in rows if r[ct_idx] not in self._exclude_chunk_types]
            if self._exclude_chunk_types:
                rows = rows[:requested]

            out: list[dict[str, Any]] = []
            for row in rows:
                out.append(
                    {
                        col_names[i] if i < len(col_names) else f"col_{i}": value
                        for i, value in enumerate(row)
                    }
                )
            return out

        return _call

    @staticmethod
    def _normalize_filters(filters: dict[str, Any]) -> dict[str, Any]:
        """Normalize filter dict, removing operators unsupported by query_index().

        The Databricks Vector Search API dict-key format documents only
        equality and comparison operators.  This method gracefully handles
        unsupported operators that LLMs commonly generate:

        * ``IN`` with a non-empty list → equality with the first element.
        * Any other unsupported operator → key is dropped.

        Supported operators pass through unchanged.  A warning is logged
        for every rewritten or dropped key.
        """
        normalized: dict[str, Any] = {}
        for key, value in filters.items():
            parts = key.split(" ", 1)
            if len(parts) == 2 and parts[1].strip():
                col, op_raw = parts[0], parts[1].strip().upper()
                if op_raw == "IN":
                    if isinstance(value, list) and value:
                        logger.warning(
                            "VECTOR_SEARCH_FILTER_NORMALIZED key=%r "
                            "op=IN downgraded to equality with first value=%r",
                            key, value[0],
                        )
                        normalized[col] = value[0]
                    else:
                        logger.warning(
                            "VECTOR_SEARCH_FILTER_DROPPED key=%r value=%r "
                            "(IN requires a non-empty list)",
                            key, value,
                        )
                elif op_raw in _COMPARISON_OPS:
                    if isinstance(value, (int, float)):
                        normalized[key] = value
                    else:
                        logger.warning(
                            "VECTOR_SEARCH_FILTER_DROPPED key=%r value=%r "
                            "(comparison operators require numeric values)",
                            key, value,
                        )
                elif op_raw == "!=":
                    # != is not documented but commonly supported; pass through optimistically.
                    # If the API rejects it, execute() catches the exception.
                    normalized[key] = value
                else:
                    logger.warning(
                        "VECTOR_SEARCH_FILTER_DROPPED key=%r op=%r "
                        "(unsupported in dict-key format)",
                        key, op_raw,
                    )
            else:
                # Bare column name (equality) or trailing whitespace — always safe.
                normalized[key.strip()] = value
        return normalized

    @staticmethod
    def _try_parse_filter_string(raw: str) -> dict[str, Any] | None:
        """Attempt to parse a string-valued filters argument into a dict.

        LLMs sometimes double-encode filters (JSON string of a dict) or
        emit Python dict literal syntax.  This method tries JSON first,
        then ``ast.literal_eval`` as a safe fallback.
        """
        import ast
        import json as _json

        for parser, label in ((_json.loads, "JSON"), (ast.literal_eval, "Python literal")):
            try:
                parsed = parser(raw)
                if isinstance(parsed, dict):
                    logger.info(
                        "VECTOR_SEARCH_FILTER_PARSED format=%s raw=%r", label, raw[:200],
                    )
                    return parsed
            except (ValueError, SyntaxError, _json.JSONDecodeError):
                continue

        logger.warning(
            "VECTOR_SEARCH_FILTER_PARSE_FAILED raw=%r "
            "(not valid JSON or Python literal)",
            raw[:200],
        )
        return None

    @staticmethod
    def _result_column_names(result: Any) -> list[str]:
        manifest = getattr(result, "manifest", None)
        if manifest and hasattr(manifest, "columns"):
            return [c.name for c in manifest.columns if hasattr(c, "name")]
        return []

    @staticmethod
    def _result_rows(result: Any) -> list[list[Any]]:
        data_array = getattr(result, "result", None)
        if data_array and hasattr(data_array, "data_array"):
            return data_array.data_array or []
        return []

    async def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        query = arguments["query"]
        num_results = arguments.get("num_results", self._num_results)

        try:
            if self._columns is None:
                self._columns = self._discover_columns()

            # The Databricks SDK now requires ``columns`` on query_index
            # (previously optional). If we couldn't discover any columns
            # from index metadata — e.g. the user lacks read access, or
            # the index has no primary key / delta-sync source columns —
            # fail the call cleanly rather than propagating an opaque
            # TypeError from the SDK.
            if not self._columns:
                error_msg = (
                    f"vector_search: no columns available for index "
                    f"{self._index_name!r}. Discovery via get_index "
                    f"returned no primary_key or delta-sync source "
                    f"columns (likely missing read permission on the "
                    f"index, or an unsupported index type). Set "
                    f"``columns`` explicitly in the tool config."
                )
                logger.error(error_msg)
                return ToolResult(content=error_msg, success=False, error=error_msg)

            # Over-fetch when excluding chunk types to compensate for filtered rows
            effective_num = num_results
            if self._exclude_chunk_types:
                effective_num = min(num_results * 2, 50)

            kwargs: dict[str, Any] = {
                "index_name": self._index_name,
                "query_text": query,
                "num_results": effective_num,
                # ``columns`` is now a required positional arg on the
                # SDK. Pass it unconditionally.
                "columns": self._columns,
            }
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
            all_tables: list[dict[str, Any]] = []

            manifest = getattr(result, "manifest", None)
            col_names: list[str] = []
            if manifest and hasattr(manifest, "columns"):
                col_names = self._result_column_names(result)

            rows = self._result_rows(result)

            # Post-filter excluded chunk types and trim to requested count
            if self._exclude_chunk_types and rows and col_names:
                ct_idx = col_names.index("chunk_type") if "chunk_type" in col_names else -1
                if ct_idx >= 0:
                    rows = [r for r in rows if r[ct_idx] not in self._exclude_chunk_types]
            if self._exclude_chunk_types:
                rows = rows[:num_results]

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
                    relevance_score=float(str(_col("score", 0.0) or 0.0)),
                ))

                # Detect tables in result content
                if content_text and context.table_registry is not None:
                    try:
                        from databricks_deep_research.tools.builtins.text_utils import (  # noqa: PLC0415
                            detect_markdown_tables,
                        )

                        for pt in detect_markdown_tables(content_text):
                            tbl_entry: dict[str, Any] = {
                                "markdown": pt.markdown,
                                "table_json": pt.table_json,
                                "row_count": pt.row_count,
                                "col_count": pt.col_count,
                                "source": source_title,
                            }
                            try:
                                tbl_idx = context.table_registry.register(
                                    pt.table_json,
                                    source_kind="vector_index",
                                    source_label=source_title,
                                    markdown=pt.markdown,
                                )
                                tbl_entry["table_idx"] = tbl_idx
                            except ValueError:
                                logger.warning(
                                    "VS_TABLE_REGISTER_SKIPPED tool=%s reason=capacity",
                                    self._name,
                                )
                                break  # registry full — skip remaining tables
                            all_tables.append(tbl_entry)
                    except Exception:  # noqa: BLE001
                        logger.debug(
                            "VS_TABLE_DETECT_FAILED tool=%s",
                            self._name,
                            exc_info=True,
                        )

            content = "\n".join(lines) if lines else "No results found."

            logger.info(
                "VECTOR_SEARCH_RESULTS tool=%s index=%s query=%s results=%d",
                self._name, self._index_name, query[:100], len(rows),
            )

            data: dict[str, Any] = {
                "result_count": len(rows),
                "source_kind": SourceKind.vector_index,
                "empty_result": len(rows) == 0,
            }
            if all_tables:
                data["tables"] = all_tables
                data["table_count"] = len(all_tables)

            return ToolResult(
                content=content,
                success=True,
                sources=sources,
                data=data,
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
