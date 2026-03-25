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

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_MAX_LIMIT = 100


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
            f"[{idx}] chunk_type={chunk_type} | page_info={page_info}\n{content_text}"
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
                        "description": "Exact filename to read (e.g., treasury_bulletin_1950_02.txt)",
                    },
                    "chunk_type": {
                        "type": "string",
                        "description": "Optional filter: table, section, or text",
                        "enum": ["table", "section", "text"],
                    },
                    "limit": {
                        "type": "integer",
                        "description": f"Max chunks to return (default 50, max {_MAX_LIMIT})",
                        "default": 50,
                    },
                },
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
                        "description": "Exact filename to search within",
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
                    "chunk_type": {
                        "type": "string",
                        "description": "Optional filter: table, section, or text",
                        "enum": ["table", "section", "text"],
                    },
                    "limit": {
                        "type": "integer",
                        "description": f"Max results (default 20, max {_MAX_LIMIT})",
                        "default": 20,
                    },
                },
                "required": ["file_name", "pattern"],
                "additionalProperties": False,
            },
            source_type="enterprise",
            source_kind=SourceKind.delta_table,
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        file_name = arguments.get("file_name", "")
        if not isinstance(file_name, str) or not file_name.strip():
            raise ValueError("'file_name' must be a non-empty string")

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

        return {
            "file_name": file_name.strip(),
            "pattern": pattern.strip(),
            "mode": mode,
            "chunk_type": arguments.get("chunk_type"),
            "limit": min(int(arguments.get("limit", 20)), _MAX_LIMIT),
        }

    async def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        file_name = arguments["file_name"]
        pattern = arguments["pattern"]
        mode = arguments.get("mode", "substring")
        chunk_type = arguments.get("chunk_type")
        limit = arguments.get("limit", 20)

        cols = ", ".join(self._columns)
        params: list[dict[str, Any]] = [
            {"name": "file_name", "value": file_name, "type": "STRING"},
            {"name": "pattern", "value": pattern, "type": "STRING"},
        ]
        where = "WHERE file_name = :file_name"

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

        sql = f"SELECT {cols} FROM {self._table} {where} ORDER BY {self._order_by} LIMIT {limit}"

        try:
            rows, col_names = _execute_sql(self._ws, self._warehouse_id, sql, params)
        except Exception as exc:
            logger.exception(
                "DELTA_GREP_ERROR tool=%s table=%s file=%s pattern=%s mode=%s",
                self._name, self._table, file_name, pattern[:100], mode,
            )
            return ToolResult(
                content=f"Delta grep failed: {exc}",
                success=False,
                error=str(exc),
            )

        if not rows:
            return ToolResult(
                content=f"No matches for pattern='{pattern}' ({mode}) in file_name='{file_name}'",
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
