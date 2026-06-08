"""HTML table extraction with exact cell text preservation.

Ported from ``benchmarks/officeqa/html_table_parser.py``.  Uses stdlib
``html.parser`` only — no external dependencies.

**Two public entry points**:

* ``extract_tables_from_html(html)`` — extract ``<table>`` elements from HTML
  into :class:`ParsedTable` objects carrying both Markdown and Table-compatible
  ``dict`` representations.  This is **Layer 1** of the two-layer table
  extraction strategy (HTML rescue before trafilatura destroys structure).

* ``truncate_markdown_table(markdown, max_rows)`` — truncate a large Markdown
  pipe-table for LLM context, preserving header + first/last rows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from typing import Any

# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class ParsedTable:
    """A table extracted from HTML or detected in markdown.

    ``table_json`` is a *dict* (not a JSON string) using the same schema as
    :class:`~databricks_deep_research.tools.builtins.text_table.table_api.Table`:

    .. code-block:: python

        {
            "headers": [{"name": str, "parent": str | None, "index": int}, ...],
            "rows": [{"label": str, "cells": {col: val}, "is_total": bool,
                       "is_group_header": bool}, ...],
            "row_count": int,
            "data_row_count": int,
        }
    """

    markdown: str
    table_json: dict[str, Any]
    row_count: int
    col_count: int


# ---------------------------------------------------------------------------
# Markdown cell escaping
# ---------------------------------------------------------------------------


def _escape_md_cell(text: str) -> str:
    """Escape pipe characters and normalize whitespace for Markdown cells."""
    return text.replace("|", "\\|").replace("\n", " ").strip()


# ---------------------------------------------------------------------------
# HTML table parser
# ---------------------------------------------------------------------------


class _HTMLTableParser(HTMLParser):
    """Parse ``<table>`` blocks into 2D grids of exact cell text.

    Handles ``rowspan`` / ``colspan`` by maintaining an ``_occupied`` dict
    that tracks which ``(row, col)`` positions are filled.  Cell text is
    preserved exactly as it appears in the HTML source — no type coercion.

    Only the **outermost** table is parsed — nested tables have their text
    folded into the enclosing cell.
    """

    def __init__(self) -> None:
        super().__init__()
        self.tables: list[tuple[list[list[str]], int]] = []

        # Per-table state
        self._occupied: dict[tuple[int, int], str] = {}
        self._header_rows: int = 0
        self._in_table: bool = False
        self._table_depth: int = 0
        self._row_idx: int = -1
        self._col_idx: int = 0

        # Per-cell state
        self._in_cell: bool = False
        self._cell_text: str = ""
        self._cell_is_header: bool = False
        self._cell_rowspan: int = 1
        self._cell_colspan: int = 1

    # -- HTMLParser overrides ------------------------------------------------

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attrs_dict = dict(attrs)

        if tag == "table":
            self._table_depth += 1
            if self._table_depth == 1:
                self._occupied = {}
                self._header_rows = 0
                self._row_idx = -1
                self._in_table = True

        elif tag == "tr" and self._in_table:
            self._row_idx += 1
            self._col_idx = 0

        elif tag in ("td", "th") and self._in_table:
            self._in_cell = True
            self._cell_text = ""
            self._cell_is_header = tag == "th"
            self._cell_rowspan = int(attrs_dict.get("rowspan") or 1)
            self._cell_colspan = int(attrs_dict.get("colspan") or 1)
            # Advance past positions already occupied by rowspan/colspan
            while (self._row_idx, self._col_idx) in self._occupied:
                self._col_idx += 1

        elif tag == "br" and self._in_cell:
            self._cell_text += " "

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()

        if tag in ("td", "th") and self._in_cell:
            self._in_cell = False
            text = self._cell_text.strip()

            for r in range(self._cell_rowspan):
                for c in range(self._cell_colspan):
                    pos = (self._row_idx + r, self._col_idx + c)
                    self._occupied[pos] = text if (r == 0 and c == 0) else ""

            if self._cell_is_header:
                self._header_rows = max(self._header_rows, self._row_idx + 1)

            self._col_idx += self._cell_colspan

        elif tag == "table":
            if self._table_depth == 1 and self._in_table:
                self._in_table = False
                grid = self._build_grid()
                if grid:
                    self.tables.append((grid, self._header_rows))
            self._table_depth = max(0, self._table_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_text += data

    def handle_entityref(self, name: str) -> None:
        if self._in_cell:
            self._cell_text += unescape(f"&{name};")

    def handle_charref(self, name: str) -> None:
        if self._in_cell:
            self._cell_text += unescape(f"&#{name};")

    # -- Grid construction ---------------------------------------------------

    def _build_grid(self) -> list[list[str]]:
        if not self._occupied:
            return []
        max_row = max(r for r, _ in self._occupied)
        max_col = max(c for _, c in self._occupied)
        return [
            [self._occupied.get((r, c), "") for c in range(max_col + 1)]
            for r in range(max_row + 1)
        ]


# ---------------------------------------------------------------------------
# Shared header builder
# ---------------------------------------------------------------------------

_TOTAL_PATTERN = re.compile(r"^\s*(grand\s+)?total\b", re.IGNORECASE)


def _build_typed_headers(
    grid: list[list[str]],
    header_rows: int,
    num_cols: int,
) -> tuple[list[str], list[dict[str, Any]], int]:
    """Build header names and typed metadata from a 2D grid.

    Returns ``(flat_names, typed_headers, data_start_row)``.
    """
    raw_names: list[str] = []
    parents: list[str | None] = []

    if header_rows == 0:
        raw_names = ["Category"] + [f"col_{i + 1}" for i in range(1, num_cols)]
        parents = [None] * num_cols
        data_start = 0
    elif header_rows == 1:
        raw_names = [grid[0][i] if i < len(grid[0]) else "" for i in range(num_cols)]
        parents = [None] * num_cols
        data_start = 1
    else:
        for col_idx in range(num_cols):
            parts: list[str] = []
            for row_idx in range(header_rows):
                val = grid[row_idx][col_idx] if col_idx < len(grid[row_idx]) else ""
                if val:
                    parts.append(val)
            raw_names.append(" > ".join(parts) if parts else "")

        for col_idx in range(num_cols):
            parent_parts: list[str] = []
            for row_idx in range(header_rows - 1):
                val = grid[row_idx][col_idx] if col_idx < len(grid[row_idx]) else ""
                if not val:
                    for left in range(col_idx - 1, -1, -1):
                        left_val = grid[row_idx][left] if left < len(grid[row_idx]) else ""
                        if left_val:
                            val = left_val
                            break
                if val:
                    parent_parts.append(val)
            parents.append(" > ".join(parent_parts) if parent_parts else None)
        data_start = header_rows

    # First-column heuristic: if empty, label as "Category"
    if raw_names and not raw_names[0]:
        raw_names[0] = "Category"

    # Ensure unique, non-empty header names
    flat_names: list[str] = []
    seen: dict[str, int] = {}
    for i, h in enumerate(raw_names):
        name = h if h else f"col_{i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        flat_names.append(name)

    # Build typed headers
    typed: list[dict[str, Any]] = []
    for i in range(num_cols):
        leaf = flat_names[i]
        if parents[i] and " > " in leaf:
            leaf = leaf.split(" > ")[-1].strip()
        typed.append({"name": leaf, "parent": parents[i], "index": i})

    # Ensure unique leaf names
    leaf_seen: dict[str, int] = {}
    for th in typed:
        name = th["name"]
        if name in leaf_seen:
            leaf_seen[name] += 1
            th["name"] = f"{name}_{leaf_seen[name]}"
        else:
            leaf_seen[name] = 1

    return flat_names, typed, data_start


# ---------------------------------------------------------------------------
# Grid → Markdown
# ---------------------------------------------------------------------------


def _grid_to_markdown(grid: list[list[str]], header_rows: int) -> str:
    """Convert a 2D grid of cell text into a Markdown pipe-table string."""
    if not grid:
        return ""

    num_cols = max(len(row) for row in grid)
    for row in grid:
        while len(row) < num_cols:
            row.append("")

    flat_names, _, data_start = _build_typed_headers(grid, header_rows, num_cols)
    data_rows = grid[data_start:]

    lines: list[str] = []
    lines.append("| " + " | ".join(_escape_md_cell(h) for h in flat_names) + " |")
    lines.append("| " + " | ".join(["---"] * len(flat_names)) + " |")

    for row in data_rows:
        cells = [_escape_md_cell(row[i] if i < len(row) else "") for i in range(len(flat_names))]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Grid → Table-compatible dict
# ---------------------------------------------------------------------------


def _grid_to_table_json(grid: list[list[str]], header_rows: int) -> dict[str, Any]:
    """Convert a 2D grid into a Table-compatible dict.

    Returns a *dict* (not a JSON string) with ``headers``, ``rows``,
    ``row_count``, ``data_row_count`` — ready to wrap in ``Table()``.
    """
    if not grid:
        return {}

    num_cols = max(len(row) for row in grid)
    for row in grid:
        while len(row) < num_cols:
            row.append("")

    _, typed_headers, data_start = _build_typed_headers(grid, header_rows, num_cols)
    data_rows_grid = grid[data_start:]

    rows: list[dict[str, Any]] = []
    data_row_count = 0

    for row in data_rows_grid:
        label = row[0] if row else ""
        cells: dict[str, str] = {}
        for col_idx in range(1, num_cols):
            header_name = typed_headers[col_idx]["name"]
            cells[header_name] = row[col_idx] if col_idx < len(row) else ""

        non_empty_data = sum(1 for v in cells.values() if v.strip())
        is_group_header = non_empty_data == 0 and bool(label.strip())
        is_total = bool(_TOTAL_PATTERN.match(label.strip())) if label else False

        rows.append({
            "label": label,
            "cells": cells,
            "is_group_header": is_group_header,
            "is_total": is_total,
        })
        if not is_group_header:
            data_row_count += 1

    return {
        "headers": typed_headers,
        "rows": rows,
        "row_count": len(rows),
        "data_row_count": data_row_count,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _feed_and_flush(html: str) -> tuple[_HTMLTableParser, bool]:
    """Feed HTML to parser, flush in-progress tables."""
    parser = _HTMLTableParser()
    try:
        parser.feed(html)
    except Exception:
        return parser, False

    # Flush any in-progress table — handles truncated HTML missing </table>.
    if parser._in_table and parser._occupied:
        grid = parser._build_grid()
        if grid:
            parser.tables.append((grid, parser._header_rows))
        parser._in_table = False

    return parser, True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_tables_from_html(html: str) -> list[ParsedTable]:
    """Extract ``<table>`` elements from HTML into :class:`ParsedTable` objects.

    **Layer 1** of the two-layer table extraction strategy.  Runs *before*
    trafilatura strips table structure from the raw HTML.

    Uses stdlib ``html.parser`` — no external dependencies.  Handles
    ``rowspan`` and ``colspan``.

    Tables with fewer than 2 rows or fewer than 2 columns are skipped
    (not useful as structured data).

    Parameters
    ----------
    html:
        Raw HTML string potentially containing ``<table>`` blocks.

    Returns
    -------
    list[ParsedTable]:
        One per qualifying table found.  Empty list if no tables or
        if parsing fails.
    """
    parser, ok = _feed_and_flush(html)

    if not ok or not parser.tables:
        return []

    results: list[ParsedTable] = []
    for grid, hrows in parser.tables:
        num_rows = len(grid)
        num_cols = max((len(row) for row in grid), default=0)

        # Skip trivial tables
        if num_rows < 2 or num_cols < 2:
            continue

        markdown = _grid_to_markdown(grid, hrows)
        table_json = _grid_to_table_json(grid, hrows)

        if not markdown or not table_json:
            continue

        results.append(ParsedTable(
            markdown=markdown,
            table_json=table_json,
            row_count=num_rows,
            col_count=num_cols,
        ))

    return results


def truncate_markdown_table(markdown: str, *, max_rows: int = 20) -> str:
    """Truncate a large Markdown pipe-table for LLM context.

    Preserves the header row, separator row, first *max_rows* data rows,
    and appends a ``... N more rows`` indicator when truncated.

    Parameters
    ----------
    markdown:
        A Markdown pipe-table string (header + separator + data rows).
    max_rows:
        Maximum number of data rows to include.

    Returns
    -------
    str:
        The (possibly truncated) Markdown table.
    """
    lines = markdown.split("\n")
    if len(lines) <= 2:
        return markdown  # header + separator only, or empty

    # First two lines are header + separator
    header_lines = lines[:2]
    data_lines = lines[2:]

    if len(data_lines) <= max_rows:
        return markdown

    kept = data_lines[:max_rows]
    omitted = len(data_lines) - max_rows
    kept.append(f"| ... {omitted} more rows |")

    return "\n".join(header_lines + kept)
