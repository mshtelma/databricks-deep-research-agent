"""Delta table direct-read tools — file read, grep, and context via SQL Statement Execution.

Provides three tool kinds for precision retrieval when the LLM already knows which
document (file) contains the target data:

* ``DeltaReadTool``    — read ALL chunks from a file, ordered by position.
* ``DeltaGrepTool``    — search for text patterns (substring or regex) within a file.
* ``DeltaContextTool`` — fetch chunks surrounding a specific chunk_id for context expansion.

Both use the Databricks SQL Statement Execution API with parameterized queries
(no string interpolation — injection-safe).  Config-driven: table name, columns,
warehouse ID all come from the YAML workflow definition.

Complements ``VectorSearchTool`` which is semantic-similarity based.  Use
``DeltaReadTool`` when you need the *complete* document (e.g., a multi-row table
that was split across chunks) and ``DeltaGrepTool`` when you need to find an
exact value that semantic search misses (e.g., a row that is purely numeric).
"""

from __future__ import annotations

import json as _json
import logging
import re
from collections.abc import Callable
from typing import Any

from databricks_deep_research.tools.protocol import (
    SourceInfo,
    SourceKind,
    ToolContext,
    ToolDefinition,
    ToolResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_MAX_LIMIT = 100
_ALL_CHUNK_TYPES = ("table", "section", "text")


def _append_chunk_type_exclusion(
    where: str,
    params: list[dict[str, Any]],
    exclude_chunk_types: list[str],
) -> str:
    """Append ``chunk_type NOT IN (...)`` to a SQL WHERE clause (parameterized)."""
    if not exclude_chunk_types:
        return where
    placeholders = ", ".join(f":excl_ct_{i}" for i in range(len(exclude_chunk_types)))
    where += f" AND chunk_type NOT IN ({placeholders})"
    for i, ct in enumerate(exclude_chunk_types):
        params.append({"name": f"excl_ct_{i}", "value": ct, "type": "STRING"})
    return where


def _execute_sql(
    workspace_client: Any,
    warehouse_id: str,
    sql: str,
    params: list[dict[str, Any]],
) -> tuple[list[list[Any]], list[str]]:
    """Execute a parameterized SQL statement and return (rows, col_names).

    Raises on API errors — callers should catch and wrap in ToolResult.
    """
    # SDK expects StatementParameterListItem, not plain dicts.
    # Lazy import to keep module-level free of SDK dependency.
    from databricks.sdk.service.sql import StatementParameterListItem

    sdk_params = [
        StatementParameterListItem(
            name=p["name"],
            value=p["value"],
            type=p.get("type"),
        )
        for p in params
    ]
    response = workspace_client.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=warehouse_id,
        parameters=sdk_params,
        wait_timeout="30s",
    )
    rows: list[list[Any]] = []
    col_names: list[str] = []

    if response.result and response.result.data_array:
        rows = response.result.data_array
    if response.manifest and response.manifest.schema and response.manifest.schema.columns:
        col_names = [c.name for c in response.manifest.schema.columns]

    return rows, col_names


def _format_rows(
    rows: list[list[Any]],
    col_names: list[str],
    *,
    content_column: str,
    tool_name: str,
    table_name: str,
    context: ToolContext,
) -> tuple[str, list[SourceInfo]]:
    """Format SQL result rows into text output + SourceInfo list."""
    lines: list[str] = []
    sources: list[SourceInfo] = []

    for idx, row in enumerate(rows):
        row_dict = dict(zip(col_names, row)) if len(col_names) == len(row) else {}

        content_text = str(row_dict.get(content_column, row[0] if row else ""))
        chunk_type = str(row_dict.get("chunk_type", ""))
        page_info = str(row_dict.get("page_info", ""))
        file_name = str(row_dict.get("file_name", ""))
        chunk_id = str(row_dict.get("chunk_id", idx))

        lines.append(
            f"[{idx}] chunk_id={chunk_id} | chunk_type={chunk_type} | page_info={page_info}\n{content_text}"
        )

        source_url = f"delta://{table_name}/{chunk_id}"
        if context.url_registry is not None:
            context.url_registry.register(source_url)
        sources.append(
            SourceInfo(
                url=source_url,
                title=f"{file_name} [{chunk_type}]" if file_name else chunk_id,
                snippet=content_text[:500],
                content=content_text,
                source_type="enterprise",
                source_kind=SourceKind.delta_table,
            )
        )

    return "\n\n".join(lines), sources


# ---------------------------------------------------------------------------
# DeltaReadTool
# ---------------------------------------------------------------------------


class DeltaReadTool:
    """Read all chunks from a specific file in a Delta table.

    Implements the ``ResearchTool`` protocol.  Config-driven — works with any
    Delta table, not just Treasury Bulletins.  Uses parameterized SQL via the
    Databricks SQL Statement Execution API.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        table_name: str,
        columns: list[str],
        workspace_client: Any,
        warehouse_id: str,
        content_column: str = "content",
        order_by: str = "chunk_id",
        exclude_chunk_types: list[str] | None = None,
    ) -> None:
        self._name = name
        self._description = description
        self._table = table_name
        self._columns = columns
        self._ws = workspace_client
        self._warehouse_id = warehouse_id
        self._content_col = content_column
        self._order_by = order_by
        self._exclude_chunk_types = exclude_chunk_types or []

    @property
    def definition(self) -> ToolDefinition:
        properties: dict[str, Any] = {
            "file_name": {
                "type": "string",
                "description": "Exact filename to read (e.g., treasury_bulletin_1950_02.txt)",
            },
            "limit": {
                "type": "integer",
                "description": f"Max chunks to return (default 50, max {_MAX_LIMIT})",
                "default": 50,
            },
        }
        allowed_types = [t for t in _ALL_CHUNK_TYPES if t not in self._exclude_chunk_types]
        if allowed_types:
            properties["chunk_type"] = {
                "type": "string",
                "description": "Optional filter: " + ", ".join(allowed_types),
                "enum": list(allowed_types),
            }
        return ToolDefinition(
            name=self._name,
            description=self._description,
            parameters={
                "type": "object",
                "properties": properties,
                "required": ["file_name"],
                "additionalProperties": False,
            },
            source_type="enterprise",
            source_kind=SourceKind.delta_table,
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        file_name = arguments.get("file_name", "")
        if not isinstance(file_name, str) or not file_name.strip():
            raise ValueError("'file_name' must be a non-empty string")
        return {
            "file_name": file_name.strip(),
            "chunk_type": arguments.get("chunk_type"),
            "limit": min(int(arguments.get("limit", 50)), _MAX_LIMIT),
        }

    async def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        file_name = arguments["file_name"]
        chunk_type = arguments.get("chunk_type")
        limit = arguments.get("limit", 50)

        cols = ", ".join(self._columns)
        params: list[dict[str, Any]] = [
            {"name": "file_name", "value": file_name, "type": "STRING"},
        ]
        where = "WHERE file_name = :file_name"
        if chunk_type:
            where += " AND chunk_type = :chunk_type"
            params.append({"name": "chunk_type", "value": chunk_type, "type": "STRING"})
        where = _append_chunk_type_exclusion(where, params, self._exclude_chunk_types)

        sql = f"SELECT {cols} FROM {self._table} {where} ORDER BY {self._order_by} LIMIT {limit}"

        try:
            rows, col_names = _execute_sql(self._ws, self._warehouse_id, sql, params)
        except Exception as exc:
            logger.exception(
                "DELTA_READ_ERROR tool=%s table=%s file=%s",
                self._name, self._table, file_name,
            )
            return ToolResult(
                content=f"Delta read failed: {exc}",
                success=False,
                error=str(exc),
            )

        if not rows:
            return ToolResult(
                content=f"No chunks found for file_name='{file_name}'"
                + (f" chunk_type='{chunk_type}'" if chunk_type else ""),
                success=True,
            )

        content, sources = _format_rows(
            rows, col_names,
            content_column=self._content_col,
            tool_name=self._name,
            table_name=self._table,
            context=context,
        )
        return ToolResult(
            content=content,
            success=True,
            sources=sources,
            data={"file_name": file_name, "row_count": len(rows)},
        )


# ---------------------------------------------------------------------------
# DeltaGrepTool
# ---------------------------------------------------------------------------


class DeltaGrepTool:
    """Search for text patterns within a specific file's chunks.

    Implements the ``ResearchTool`` protocol.  Supports two matching modes:

    * ``substring`` (default) — case-insensitive substring via ``ILIKE``.
    * ``regex`` — full regexp matching via ``RLIKE``.

    Uses parameterized SQL (injection-safe).
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        table_name: str,
        columns: list[str],
        workspace_client: Any,
        warehouse_id: str,
        content_column: str = "content",
        order_by: str = "chunk_id",
        exclude_chunk_types: list[str] | None = None,
        date_column: str | None = None,
    ) -> None:
        self._name = name
        self._description = description
        self._table = table_name
        self._columns = columns
        self._ws = workspace_client
        self._warehouse_id = warehouse_id
        self._content_col = content_column
        self._order_by = order_by
        self._exclude_chunk_types = exclude_chunk_types or []
        self._date_col = date_column

    @property
    def definition(self) -> ToolDefinition:
        properties: dict[str, Any] = {
            "file_name": {
                "type": "string",
                "description": "Filename to search within. Omit to search ALL files (cross-file search).",
            },
            "pattern": {
                "type": "string",
                "description": "Text pattern to search for",
            },
            "mode": {
                "type": "string",
                "description": "Match mode: 'substring' (case-insensitive, default) or 'regex'",
                "enum": ["substring", "regex"],
                "default": "substring",
            },
            "limit": {
                "type": "integer",
                "description": f"Max results (default 20, max {_MAX_LIMIT})",
                "default": 20,
            },
        }
        allowed_types = [t for t in _ALL_CHUNK_TYPES if t not in self._exclude_chunk_types]
        if allowed_types:
            properties["chunk_type"] = {
                "type": "string",
                "description": "Optional filter: " + ", ".join(allowed_types),
                "enum": list(allowed_types),
            }
        if self._date_col:
            properties["pub_year_start"] = {
                "type": "integer",
                "description": (
                    f"Filter: publication year >= value "
                    f"(derived from {self._date_col} column)"
                ),
            }
            properties["pub_year_end"] = {
                "type": "integer",
                "description": (
                    f"Filter: publication year <= value "
                    f"(derived from {self._date_col} column)"
                ),
            }
            properties["pub_month"] = {
                "type": "integer",
                "description": (
                    f"Filter: publication month = value (1-12, "
                    f"derived from {self._date_col} column)"
                ),
            }
        return ToolDefinition(
            name=self._name,
            description=self._description,
            parameters={
                "type": "object",
                "properties": properties,
                "required": ["pattern"],
                "additionalProperties": False,
            },
            source_type="enterprise",
            source_kind=SourceKind.delta_table,
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raw_file_name = arguments.get("file_name", "")
        if raw_file_name and isinstance(raw_file_name, str) and raw_file_name.strip():
            file_name: str | None = raw_file_name.strip()
        else:
            file_name = None  # Cross-file search

        pattern = arguments.get("pattern", "")
        if not isinstance(pattern, str) or not pattern.strip():
            raise ValueError("'pattern' must be a non-empty string")

        mode = arguments.get("mode", "substring")
        if mode not in ("substring", "regex"):
            mode = "substring"

        # Validate regex syntax early to give a clear error
        if mode == "regex":
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"Invalid regex pattern: {exc}") from exc

        default_limit = 20 if file_name else 30
        validated: dict[str, Any] = {
            "file_name": file_name,
            "pattern": pattern.strip(),
            "mode": mode,
            "chunk_type": arguments.get("chunk_type"),
            "limit": min(int(arguments.get("limit", default_limit)), _MAX_LIMIT),
        }
        # Date filters (only active when date_column is configured)
        if self._date_col:
            for key in ("pub_year_start", "pub_year_end", "pub_month"):
                raw = arguments.get(key)
                if raw is not None:
                    try:
                        validated[key] = int(raw)
                    except (ValueError, TypeError):
                        pass  # ignore unparseable date filter
        return validated

    async def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        file_name = arguments.get("file_name")
        pattern = arguments["pattern"]
        mode = arguments.get("mode", "substring")
        chunk_type = arguments.get("chunk_type")
        limit = arguments.get("limit", 20)

        cols = ", ".join(self._columns)
        params: list[dict[str, Any]] = [
            {"name": "pattern", "value": pattern, "type": "STRING"},
        ]

        if file_name:
            where = "WHERE file_name = :file_name"
            params.insert(0, {"name": "file_name", "value": file_name, "type": "STRING"})
        else:
            where = "WHERE 1=1"

        if mode == "regex":
            where += f" AND {self._content_col} RLIKE :pattern"
        else:
            # ILIKE with escaped wildcards for safe substring matching
            escaped = pattern.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            params[-1]["value"] = f"%{escaped}%"
            where += f" AND {self._content_col} ILIKE :pattern"

        if chunk_type:
            where += " AND chunk_type = :chunk_type"
            params.append({"name": "chunk_type", "value": chunk_type, "type": "STRING"})

        # Date filters (config-driven: only active when date_column is set)
        if self._date_col:
            pub_year_start = arguments.get("pub_year_start")
            pub_year_end = arguments.get("pub_year_end")
            pub_month = arguments.get("pub_month")
            if pub_year_start is not None:
                where += (
                    f" AND CAST(SUBSTRING({self._date_col}, 1, 4) AS INT)"
                    " >= :pub_year_start"
                )
                params.append({
                    "name": "pub_year_start",
                    "value": str(pub_year_start),
                    "type": "INT",
                })
            if pub_year_end is not None:
                where += (
                    f" AND CAST(SUBSTRING({self._date_col}, 1, 4) AS INT)"
                    " <= :pub_year_end"
                )
                params.append({
                    "name": "pub_year_end",
                    "value": str(pub_year_end),
                    "type": "INT",
                })
            if pub_month is not None:
                where += (
                    f" AND CAST(SUBSTRING({self._date_col}, 6, 2) AS INT)"
                    " = :pub_month"
                )
                params.append({
                    "name": "pub_month",
                    "value": str(pub_month),
                    "type": "INT",
                })

        where = _append_chunk_type_exclusion(where, params, self._exclude_chunk_types)

        # Cross-file: order chronologically by file_name; within-file: order by chunk_id
        order = self._order_by if file_name else f"file_name, {self._order_by}"
        sql = f"SELECT {cols} FROM {self._table} {where} ORDER BY {order} LIMIT {limit}"

        try:
            rows, col_names = _execute_sql(self._ws, self._warehouse_id, sql, params)
        except Exception as exc:
            logger.exception(
                "DELTA_GREP_ERROR tool=%s table=%s file=%s pattern=%s mode=%s",
                self._name, self._table, file_name or "(all)", pattern[:100], mode,
            )
            return ToolResult(
                content=f"Delta grep failed: {exc}",
                success=False,
                error=str(exc),
            )

        if not rows:
            scope = f"file_name='{file_name}'" if file_name else "all files"
            return ToolResult(
                content=f"No matches for pattern='{pattern}' ({mode}) in {scope}",
                success=True,
            )

        content, sources = _format_rows(
            rows, col_names,
            content_column=self._content_col,
            tool_name=self._name,
            table_name=self._table,
            context=context,
        )
        return ToolResult(
            content=content,
            success=True,
            sources=sources,
            data={
                "file_name": file_name,
                "pattern": pattern,
                "mode": mode,
                "row_count": len(rows),
            },
        )


# ---------------------------------------------------------------------------
# DeltaContextTool
# ---------------------------------------------------------------------------

_MAX_WINDOW = 5


class DeltaContextTool:
    """Fetch chunks surrounding a specific chunk_id for context expansion.

    Implements the ``ResearchTool`` protocol.  Use after a vector search returns
    a chunk from a multi-row table — pass the ``chunk_id`` and ``file_name`` to
    see adjacent chunks without reading the entire file.

    Uses parameterized SQL (injection-safe).

    Accepts chunk_id as either a full string ID (e.g., ``prefix_c0027``) or a
    bare numeric index (e.g., ``27``).  When a bare number is passed, the tool
    reconstructs the compound ID from the file_name stem and retries
    automatically.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        table_name: str,
        columns: list[str],
        workspace_client: Any,
        warehouse_id: str,
        content_column: str = "content",
        order_by: str = "chunk_id",
    ) -> None:
        self._name = name
        self._description = description
        self._table = table_name
        self._columns = columns
        self._ws = workspace_client
        self._warehouse_id = warehouse_id
        self._content_col = content_column
        self._order_by = order_by

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description=self._description,
            parameters={
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "Exact filename (from search results)",
                    },
                    "chunk_id": {
                        "type": "string",
                        "description": (
                            "Center chunk ID (from search results). "
                            "Can be the full ID (e.g., 'treasury_bulletin_1941_01_c0027') "
                            "or just the numeric index (e.g., '27')."
                        ),
                    },
                    "window": {
                        "type": "integer",
                        "description": f"Chunks before/after center (default 2, max {_MAX_WINDOW})",
                        "default": 2,
                    },
                },
                "required": ["file_name", "chunk_id"],
                "additionalProperties": False,
            },
            source_type="enterprise",
            source_kind=SourceKind.delta_table,
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        file_name = arguments.get("file_name", "")
        if not isinstance(file_name, str) or not file_name.strip():
            raise ValueError("'file_name' must be a non-empty string")

        chunk_id = str(arguments.get("chunk_id", "")).strip()
        if not chunk_id:
            raise ValueError("'chunk_id' must be a non-empty string or integer")

        window = min(max(int(arguments.get("window", 2)), 0), _MAX_WINDOW)

        return {
            "file_name": file_name.strip(),
            "chunk_id": chunk_id,
            "window": window,
        }

    @staticmethod
    def _compute_range(chunk_id: str, window: int) -> tuple[str, str]:
        """Compute start/end IDs for a range query, preserving the ID format.

        Handles two formats:
        1. Compound: ``prefix_cNNNN`` → decrement/increment the numeric suffix.
        2. Pure numeric string: ``"67"`` → decrement/increment the integer.

        Falls back to an exact-match range (start == end) when the format is
        unrecognised.
        """
        # Compound format: anything ending with _cNNNN (zero-padded)
        match = re.match(r"^(.+_c)(\d+)$", chunk_id)
        if match:
            prefix, num_str = match.group(1), match.group(2)
            width = len(num_str)
            center = int(num_str)
            start = max(0, center - window)
            end = center + window
            return f"{prefix}{start:0{width}d}", f"{prefix}{end:0{width}d}"

        # Pure numeric string
        try:
            center = int(chunk_id)
            start = max(0, center - window)
            end = center + window
            return str(start), str(end)
        except ValueError:
            pass

        # Unrecognised format — exact match only
        return chunk_id, chunk_id

    def _build_range_sql(self, cols: str) -> str:
        return (
            f"SELECT {cols} FROM {self._table} "
            f"WHERE file_name = :file_name "
            f"AND {self._order_by} >= :start_id AND {self._order_by} <= :end_id "
            f"ORDER BY {self._order_by}"
        )

    def _range_params(
        self, file_name: str, start_id: str, end_id: str,
    ) -> list[dict[str, Any]]:
        return [
            {"name": "file_name", "value": file_name, "type": "STRING"},
            {"name": "start_id", "value": start_id, "type": "STRING"},
            {"name": "end_id", "value": end_id, "type": "STRING"},
        ]

    async def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        file_name: str = arguments["file_name"]
        chunk_id_raw: str = str(arguments["chunk_id"])
        window: int = arguments.get("window", 2)

        cols = ", ".join(self._columns)
        range_sql = self._build_range_sql(cols)

        try:
            rows, col_names = await self._resolve_context_rows(
                file_name, chunk_id_raw, window, cols, range_sql,
            )
        except Exception as exc:
            logger.exception(
                "DELTA_CONTEXT_ERROR tool=%s table=%s file=%s chunk_id=%s window=%d",
                self._name, self._table, file_name, chunk_id_raw, window,
            )
            return ToolResult(
                content=f"Delta context failed: {exc}",
                success=False,
                error=str(exc),
            )

        if not rows:
            return ToolResult(
                content=f"No chunks found for file_name='{file_name}' around chunk_id={chunk_id_raw}",
                success=True,
            )

        content, sources = _format_rows(
            rows, col_names,
            content_column=self._content_col,
            tool_name=self._name,
            table_name=self._table,
            context=context,
        )
        return ToolResult(
            content=content,
            success=True,
            sources=sources,
            data={
                "file_name": file_name,
                "chunk_id": chunk_id_raw,
                "window": window,
                "row_count": len(rows),
            },
        )

    # -- Private helpers for multi-strategy chunk resolution ------------------

    async def _resolve_context_rows(
        self,
        file_name: str,
        chunk_id_raw: str,
        window: int,
        cols: str,
        range_sql: str,
    ) -> tuple[list[list[Any]], list[str]]:
        """Try multiple strategies to resolve surrounding chunks.

        Strategy 1: Direct range query using chunk_id as-is.
        Strategy 2: Reconstruct compound ID from file_name stem + bare number.
        Strategy 3: Fallback — fetch all file chunks and window client-side.
        """
        # Strategy 1: use chunk_id directly
        start_id, end_id = self._compute_range(chunk_id_raw, window)
        rows, col_names = _execute_sql(
            self._ws, self._warehouse_id, range_sql,
            self._range_params(file_name, start_id, end_id),
        )
        if len(rows) > 1:
            logger.info(
                "DELTA_CONTEXT_RESOLVED strategy=direct rows=%d chunk_id=%s",
                len(rows), chunk_id_raw,
            )
            return rows, col_names

        # Strategy 2: bare number → reconstruct compound ID from file stem
        if chunk_id_raw.lstrip("-").isdigit():
            stem = file_name.rsplit(".", 1)[0]  # strip extension
            reconstructed = f"{stem}_c{int(chunk_id_raw):04d}"
            start_id, end_id = self._compute_range(reconstructed, window)
            rows, col_names = _execute_sql(
                self._ws, self._warehouse_id, range_sql,
                self._range_params(file_name, start_id, end_id),
            )
            if len(rows) > 1:
                logger.info(
                    "DELTA_CONTEXT_RESOLVED strategy=reconstruct rows=%d "
                    "reconstructed=%s",
                    len(rows), reconstructed,
                )
                return rows, col_names

        # Strategy 3: fetch all chunks for this file and window client-side
        all_sql = (
            f"SELECT {cols} FROM {self._table} "
            f"WHERE file_name = :file_name "
            f"ORDER BY {self._order_by} LIMIT 500"
        )
        all_params = [{"name": "file_name", "value": file_name, "type": "STRING"}]
        all_rows, col_names = _execute_sql(
            self._ws, self._warehouse_id, all_sql, all_params,
        )
        if not all_rows:
            return [], col_names

        center_idx = self._find_center_row(all_rows, col_names, chunk_id_raw)
        if center_idx is not None:
            start_idx = max(0, center_idx - window)
            end_idx = min(len(all_rows), center_idx + window + 1)
            logger.info(
                "DELTA_CONTEXT_RESOLVED strategy=fallback rows=%d center=%d "
                "total_file_chunks=%d",
                end_idx - start_idx, center_idx, len(all_rows),
            )
            return all_rows[start_idx:end_idx], col_names

        # Nothing matched — return whatever Strategy 1 found (may be 0-1 rows)
        logger.warning(
            "DELTA_CONTEXT_NO_MATCH chunk_id=%s file=%s strategies_tried=3",
            chunk_id_raw, file_name,
        )
        return rows, col_names

    @staticmethod
    def _find_center_row(
        rows: list[list[Any]],
        col_names: list[str],
        chunk_id_raw: str,
    ) -> int | None:
        """Find the row index matching chunk_id_raw by exact or suffix match."""
        cid_col = col_names.index("chunk_id") if "chunk_id" in col_names else 0
        needle = chunk_id_raw.lower()

        for idx, row in enumerate(rows):
            val = str(row[cid_col]).lower()
            # Exact match
            if val == needle:
                return idx
            # Suffix match: chunk_id_raw is a bare number, val ends with _cNNNN
            if needle.isdigit():
                suffix_match = re.search(r"_c0*(\d+)$", val)
                if suffix_match and suffix_match.group(1) == needle.lstrip("0"):
                    return idx
        return None


# ---------------------------------------------------------------------------
# DeltaTableReadTool — read a single row by primary key
# ---------------------------------------------------------------------------


class DeltaTableReadTool:
    """Read a single row from a Delta table by primary key lookup.

    Generic framework tool — works with any Delta table, not just treasury.
    Designed for structured-data-beside-chunks patterns where a separate table
    stores enriched representations (e.g., JSON table structures) keyed by the
    same chunk_id used in the main chunks table.

    Implements the ``ResearchTool`` protocol.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        table_name: str,
        columns: list[str],
        workspace_client: Any,
        warehouse_id: str,
        content_column: str = "content",
        pk_column: str = "chunk_id",
        store_in_compute: str | None = None,
        compute_resolver: Callable[[], Any] | None = None,
        structural_analysis: bool = False,
    ) -> None:
        self._name = name
        self._description = description
        self._table = table_name
        self._columns = columns
        self._ws = workspace_client
        self._warehouse_id = warehouse_id
        self._content_col = content_column
        self._pk_col = pk_column
        self._store_as = store_in_compute
        self._resolve_compute = compute_resolver
        self._enable_analysis = structural_analysis

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description=self._description,
            parameters={
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "string",
                        "description": (
                            f"Primary key value to look up in the "
                            f"'{self._pk_col}' column "
                            f"(e.g., treasury_bulletin_1977_03_c0015)"
                        ),
                    },
                    "file_name": {
                        "type": "string",
                        "description": "Optional safety filter: exact filename",
                    },
                },
                "required": ["chunk_id"],
                "additionalProperties": False,
            },
            source_type="enterprise",
            source_kind=SourceKind.delta_table,
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        pk_value = arguments.get("chunk_id", "")
        if not isinstance(pk_value, str) or not pk_value.strip():
            raise ValueError("'chunk_id' must be a non-empty string")
        return {
            "chunk_id": pk_value.strip(),
            "file_name": (arguments.get("file_name") or "").strip() or None,
        }

    async def execute(
        self, arguments: dict[str, Any], context: ToolContext,
    ) -> ToolResult:
        pk_value = arguments["chunk_id"]
        file_name = arguments.get("file_name")

        cols = ", ".join(self._columns)
        params: list[dict[str, Any]] = [
            {"name": "pk_value", "value": pk_value, "type": "STRING"},
        ]
        where = f"WHERE {self._pk_col} = :pk_value"
        if file_name:
            where += " AND file_name = :file_name"
            params.append(
                {"name": "file_name", "value": file_name, "type": "STRING"},
            )

        sql = f"SELECT {cols} FROM {self._table} {where} LIMIT 1"

        try:
            rows, col_names = _execute_sql(
                self._ws, self._warehouse_id, sql, params,
            )
        except Exception as exc:
            logger.exception(
                "DELTA_TABLE_READ_ERROR tool=%s table=%s pk=%s",
                self._name, self._table, pk_value,
            )
            return ToolResult(
                content=f"Table read failed: {exc}",
                success=False,
                error=str(exc),
            )

        if not rows:
            return ToolResult(
                content=(
                    f"No row found for {self._pk_col}='{pk_value}'"
                    + (f" file_name='{file_name}'" if file_name else "")
                ),
                success=True,
            )

        row_dict = dict(zip(col_names, rows[0])) if len(col_names) == len(rows[0]) else {}

        # Parse JSON + inject into compute namespace
        parsed, table_wrapped = self._parse_and_inject(row_dict, pk_value)

        # Build source info for URL registry
        source_url = f"delta://{self._table}/{pk_value}"
        if context.url_registry is not None:
            context.url_registry.register(source_url)
        fn = str(row_dict.get("file_name", ""))
        source = SourceInfo(
            url=source_url,
            title=f"{fn} [{row_dict.get('table_title', '')}]" if fn else pk_value,
            snippet=str(row_dict.get("content", ""))[:500],
            content="",
            source_type="enterprise",
            source_kind=SourceKind.delta_table,
        )

        # When structural analysis is enabled and JSON parsed successfully,
        # output ONLY the analysis + row labels. Raw values are hidden —
        # accessible only via compute namespace. This forces the agent to
        # use compute() for value extraction instead of eyeballing text.
        if self._enable_analysis and parsed:
            analysis = self._analyze_table_structure(
                parsed, row_dict, pk_value, table_wrapped=table_wrapped,
            )
            return ToolResult(
                content=analysis,
                success=True,
                sources=[source],
                data={"chunk_id": pk_value, "row_count": 1},
            )

        # Fallback: raw content via _format_rows (no analysis, or JSON parse failed)
        content, sources_fmt = _format_rows(
            rows, col_names,
            content_column=self._content_col,
            tool_name=self._name,
            table_name=self._table,
            context=context,
        )
        return ToolResult(
            content=content,
            success=True,
            sources=sources_fmt,
            data={"chunk_id": pk_value, "row_count": 1},
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_and_inject(
        self, row_dict: dict[str, Any], pk_value: str,
    ) -> tuple[dict[str, Any] | None, bool]:
        """Parse JSON from content column, inject into compute.

        Returns ``(parsed_dict, table_wrapped)`` where *table_wrapped* is
        ``True`` only when a :class:`Table` instance was successfully
        injected (not a raw dict fallback).
        """
        json_str = str(row_dict.get(self._content_col, ""))
        if not json_str or json_str == "None":
            return None, False
        try:
            parsed = _json.loads(json_str)
        except (ValueError, TypeError):
            logger.warning(
                "DELTA_TABLE_READ_JSON_PARSE_FAIL tool=%s pk=%s",
                self._name, pk_value,
            )
            return None, False

        table_wrapped = False
        if self._store_as and self._resolve_compute:
            compute_tool = self._resolve_compute()
            if compute_tool is not None:
                injectable: Any = parsed
                # Wrap in Table class when the parsed dict has table structure.
                # Table.__getitem__ and .get() preserve backward compatibility
                # with existing code that accesses table['rows'], table['headers'].
                if (
                    isinstance(parsed, dict)
                    and "headers" in parsed
                    and "rows" in parsed
                ):
                    try:
                        from databricks_deep_research.tools.builtins.table_api import (  # noqa: PLC0415
                            Table,
                        )

                        injectable = Table(
                            parsed,
                            chunk_id=pk_value,
                            file_name=str(row_dict.get("file_name", "")),
                            title=str(row_dict.get("table_title", "")),
                            annotation=str(row_dict.get("annotation", "")),
                        )
                        table_wrapped = True
                    except Exception:  # noqa: BLE001
                        logger.error(
                            "DELTA_TABLE_READ_TABLE_WRAP_FAIL pk=%s — "
                            "falling back to raw dict; LLM will NOT have "
                            "Table API methods (cell, series, find_rows, …)",
                            pk_value,
                            exc_info=True,
                        )
                compute_tool.inject_variable(self._store_as, injectable)
                logger.info(
                    "DELTA_TABLE_READ_INJECTED tool=%s var=%s type=%s keys=%s",
                    self._name,
                    self._store_as,
                    type(injectable).__name__,
                    list(parsed.keys())[:5] if isinstance(parsed, dict) else type(parsed).__name__,
                )
        return parsed, table_wrapped

    @staticmethod
    def _analyze_table_structure(
        parsed: dict[str, Any],
        row_dict: dict[str, Any],
        pk_value: str,
        *,
        table_wrapped: bool,
    ) -> str:
        """Produce structural diagnostics for a parsed table JSON.

        Shows header parents, columns, decomposition status, row labels,
        and period range.  Contains NO raw cell values — the agent must
        use ``compute()`` on the namespace variable to access data.

        When *table_wrapped* is ``False`` (Table construction failed),
        the method hints reference raw dict access instead of Table API
        methods so the LLM doesn't call methods that don't exist.

        All checks are generic (pure structure and math, no domain keywords).
        """
        headers = parsed.get("headers", [])
        rows = parsed.get("rows", [])

        lines: list[str] = ["STRUCTURAL ANALYSIS:"]

        # -- Source edition (helps detect cross-edition mixing) --
        src_file = str(row_dict.get("file_name", ""))
        src_date = str(row_dict.get("bulletin_date", ""))
        if src_file:
            edition = f"  Source: {src_file}"
            if src_date:
                edition += f" ({src_date})"
            lines.append(edition)

        # -- Table title from pre-table text (TSO-3, CM-I-1, etc.) --
        table_title = str(row_dict.get("table_title", ""))
        if table_title:
            lines.append(f"  Table: {table_title}")

        # -- Header parents (section context, shown as-is) --
        parents = sorted({h["parent"] for h in headers if h.get("parent")})
        for p in parents:
            lines.append(f"  Header context: {p}")

        # -- Column names --
        col_names = [h.get("name", "") for h in headers]
        if col_names:
            lines.append(f"  Columns: {' | '.join(col_names)}")

        # -- Annotation (from surrounding metadata) --
        annotation = str(row_dict.get("annotation", ""))
        if annotation:
            lines.append(f"  Annotation: {annotation}")

        # -- Total rows + decomposition check (pure math) --
        data_rows = [r for r in rows
                     if not r.get("is_group_header") and not r.get("is_total")]
        total_rows = [r for r in rows if r.get("is_total")]

        for total_row in total_rows:
            total_label = total_row.get("label", "")
            decomp_found = False
            for cname, total_val_str in total_row.get("cells", {}).items():
                try:
                    total_val = float(
                        str(total_val_str).replace(",", "").replace("$", "").strip()
                    )
                except (ValueError, TypeError):
                    continue
                if total_val == 0:
                    continue
                data_sum = 0.0
                valid_count = 0
                for dr in data_rows:
                    val_str = dr.get("cells", {}).get(cname, "")
                    try:
                        data_sum += float(
                            str(val_str).replace(",", "").replace("$", "").strip()
                        )
                        valid_count += 1
                    except (ValueError, TypeError):
                        continue
                if valid_count > 1:
                    ratio = data_sum / total_val
                    if 0.95 <= ratio <= 1.05:
                        lines.append(
                            f"  Total row: \"{total_label}\" [{cname}] = {total_val_str}"
                        )
                        lines.append(
                            f"  ⚠ DECOMPOSITION: {valid_count} data values in \"{cname}\" "
                            f"sum to {data_sum:,.0f} ≈ total {total_val:,.0f} "
                            f"(ratio {ratio:.3f}). "
                            f"Values are COMPONENTS of the total row, "
                            f"not independent observations."
                        )
                        decomp_found = True
                        break
            if not decomp_found and total_label:
                lines.append(f"  Total row: \"{total_label}\"")

        # -- Shape + period range --
        if data_rows:
            first_label = data_rows[0].get("label", "")
            last_label = data_rows[-1].get("label", "")
            period = (
                f"{first_label} — {last_label}"
                if first_label != last_label
                else first_label
            )
            lines.append(f"  Data rows: {len(data_rows)} | Period: {period}")

        # -- Row labels (structure only, no values) --
        labels = [r.get("label", "") for r in data_rows if r.get("label")]
        _MAX_LABELS = 30
        _HALF = _MAX_LABELS // 2
        if labels:
            if len(labels) <= _MAX_LABELS:
                label_text = ", ".join(labels)
            else:
                label_text = (
                    ", ".join(labels[:_HALF])
                    + ", ..., "
                    + ", ".join(labels[-_HALF:])
                )
            lines.append(f"  Row labels: {label_text}")

        # -- Orientation detection --
        # Heuristic: if the first data row labels are years/months, the
        # "real" entity names are column headers, not row labels.
        _TEMPORAL_RE = re.compile(
            r"^\d{4}$|^\d{4}\s*\(|^\d{4}-"
            r"|^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)",
            re.IGNORECASE,
        )
        sample_labels = [
            r.get("label", "").strip()
            for r in data_rows[:5]
            if r.get("label", "").strip()
        ]
        temporal_count = sum(
            1 for lbl in sample_labels if _TEMPORAL_RE.match(lbl)
        )
        if temporal_count >= 3 and len(sample_labels) >= 3:
            lines.append("")
            lines.append(
                "ORIENTATION: entities-as-COLUMNS (rows are time periods)"
            )
            if table_wrapped:
                lines.append(
                    "  To extract data for a named entity, use: "
                    "table.series('<column_name>', as_float=True)"
                )
            else:
                lines.append(
                    "  To extract a column, iterate: "
                    "[row['cells']['<column>'] for row in table['rows']]"
                )
        else:
            lines.append("")
            lines.append(
                "ORIENTATION: entities-as-ROWS (rows are named entities)"
            )
            if table_wrapped:
                lines.append(
                    "  To extract data for a named entity, use: "
                    "table.cell('<row_label>', '<column>', as_float=True)"
                )
            else:
                lines.append(
                    "  To extract a cell, find the row by label and index "
                    "into row['cells']['<column>']"
                )

        # -- Compute namespace note --
        lines.append("")
        if table_wrapped:
            lines.append(
                "Data stored in compute namespace as 'table'. "
                "Additional access methods:"
            )
            lines.append("  table.cell('row_label', 'column', as_float=True)")
            lines.append("  table.series('column', as_float=True)")
            lines.append("  table.find_rows('pattern')")
            lines.append("  table.find_columns('pattern')")
            lines.append("  table.to_dataframe()")
        else:
            lines.append(
                "Data stored in compute namespace as 'table' (raw dict). "
                "Access methods:"
            )
            lines.append("  table['headers']  — list of column header dicts")
            lines.append("  table['rows']     — list of row dicts")
            lines.append("  row['label']      — row label string")
            lines.append("  row['cells']      — dict mapping column name → value")

        return "\n".join(lines)
