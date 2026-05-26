"""TableReadTool — load any registered table into compute for structured access.

Source-agnostic: works with tables from web_crawl, web_search, file_search,
vector_search, or any other tool that registers tables in the shared
:class:`~databricks_deep_research.tools.protocol.TableRegistry`.

Follows the same pattern as ``DeltaTableReadTool``:

1. Agent provides ``table_index`` (from ``[table_idx=N]`` in tool output).
2. Resolve from ``TableRegistry`` via ``ToolContext``.
3. Wrap in :class:`~databricks_deep_research.tools.builtins.text_table.table_api.Table`.
4. Inject into compute namespace.
5. Return structural analysis (no raw values — forces agent to use compute).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from databricks_deep_research.tools.protocol import (
    SourceKind,
    ToolContext,
    ToolDefinition,
    ToolResult,
)

logger = logging.getLogger(__name__)


class TableReadTool:
    """Load a registered table into compute for structured access.

    Implements the ``ResearchTool`` protocol.
    """

    def __init__(
        self,
        *,
        name: str = "table_read",
        description: str | None = None,
        store_in_compute: str = "web_table",
        compute_resolver: Callable[[], Any] | None = None,
    ) -> None:
        self._name = name
        self._description = description or (
            "Load a table detected in web pages, search results, or uploaded "
            "files into the compute namespace for structured access. "
            "Provide the table_index from [table_idx=N] in tool output."
        )
        self._store_as = store_in_compute
        self._resolve_compute = compute_resolver

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description=self._description,
            parameters={
                "type": "object",
                "properties": {
                    "table_index": {
                        "type": "integer",
                        "description": (
                            "Index of the table to load, from [table_idx=N] "
                            "in tool output."
                        ),
                    },
                },
                "required": ["table_index"],
                "additionalProperties": False,
            },
            source_type="builtin",
            source_kind=SourceKind.builtin,
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        idx = arguments.get("table_index")
        if idx is None:
            raise ValueError("'table_index' is required")
        if not isinstance(idx, int):
            try:
                idx = int(idx)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"'table_index' must be an integer, got {type(idx).__name__}"
                ) from exc
        if idx < 0:
            raise ValueError(f"'table_index' must be non-negative, got {idx}")
        return {"table_index": idx}

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        table_index: int = arguments["table_index"]

        # -- Resolve from TableRegistry --
        if context.table_registry is None:
            return ToolResult(
                content="No table registry available in this context.",
                success=False,
                error="table_registry_unavailable",
            )

        entry = context.table_registry.resolve(table_index)
        if entry is None:
            total = len(context.table_registry)
            return ToolResult(
                content=(
                    f"Table index {table_index} not found. "
                    f"Registry contains {total} table(s) "
                    f"(valid indices: 0–{total - 1})."
                    if total > 0
                    else f"Table index {table_index} not found. Registry is empty."
                ),
                success=False,
                error="table_not_found",
            )

        table_json = entry.table_json

        # -- Wrap in Table class and inject into compute --
        injectable: Any = table_json
        try:
            from databricks_deep_research.tools.builtins.text_table.table_api import (  # noqa: PLC0415
                Table,
            )

            injectable = Table(
                table_json,
                title=f"Table from {entry.source_label}" if entry.source_label else "",
            )
        except Exception:  # noqa: BLE001
            logger.debug(
                "TABLE_READ_TABLE_WRAP_FAIL index=%d", table_index, exc_info=True,
            )

        injected = False
        if self._store_as and self._resolve_compute:
            compute_tool = self._resolve_compute()
            if compute_tool is not None:
                compute_tool.inject_variable(self._store_as, injectable)
                injected = True
                logger.info(
                    "TABLE_READ_INJECTED var=%s index=%d source=%s type=%s",
                    self._store_as,
                    table_index,
                    entry.source_kind,
                    type(injectable).__name__,
                )

        # -- Build structural analysis --
        analysis = self._analyze(table_json, entry, injected=injected)

        return ToolResult(
            content=analysis,
            success=True,
            data={
                "table_index": table_index,
                "source_kind": entry.source_kind,
                "source_label": entry.source_label,
                "row_count": table_json.get("row_count", 0),
                "data_row_count": table_json.get("data_row_count", 0),
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _analyze(
        self,
        table_json: dict[str, Any],
        entry: Any,
        *,
        injected: bool = False,
    ) -> str:
        """Produce structural diagnostics — no raw cell values."""
        headers = table_json.get("headers", [])
        rows = table_json.get("rows", [])

        lines: list[str] = ["STRUCTURAL ANALYSIS:"]

        if entry.source_label:
            lines.append(f"  Source: {entry.source_label} ({entry.source_kind})")

        # -- Header parents --
        parents = sorted({h["parent"] for h in headers if h.get("parent")})
        for p in parents:
            lines.append(f"  Header context: {p}")

        # -- Column names --
        col_names = [h.get("name", "") for h in headers]
        if col_names:
            lines.append(f"  Columns: {' | '.join(col_names)}")

        # -- Row statistics --
        data_rows = [
            r for r in rows
            if not r.get("is_group_header") and not r.get("is_total")
        ]
        total_rows = [r for r in rows if r.get("is_total")]

        if total_rows:
            for tr in total_rows:
                lines.append(f"  Total row: \"{tr.get('label', '')}\"")

        if data_rows:
            first_label = data_rows[0].get("label", "")
            last_label = data_rows[-1].get("label", "")
            period = (
                f"{first_label} — {last_label}"
                if first_label != last_label
                else first_label
            )
            lines.append(f"  Data rows: {len(data_rows)} | Range: {period}")

        # -- Row labels --
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

        # -- Compute namespace note (only when injection succeeded) --
        if injected and self._store_as:
            lines.append("")
            lines.append(
                f"Data stored in compute namespace as '{self._store_as}'. "
                "Access methods:"
            )
            lines.append(f"  {self._store_as}.cell('row_label', 'column', as_float=True)")
            lines.append(f"  {self._store_as}.series('column', as_float=True)")
            lines.append(f"  {self._store_as}.find_rows('pattern')")
            lines.append(f"  {self._store_as}.find_columns('pattern')")
            lines.append(f"  {self._store_as}.to_dataframe()")
        elif not injected:
            lines.append("")
            lines.append(
                "Note: Table loaded but compute namespace injection was not "
                "available. Raw table data is in the result metadata."
            )

        return "\n".join(lines)
