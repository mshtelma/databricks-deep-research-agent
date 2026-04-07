"""HTML table → Markdown conversion with exact cell text preservation.

Replaces ``pd.read_html()`` which corrupts data (NaN injection, comma stripping,
float coercion, Unnamed: headers).  Uses stdlib ``html.parser`` only — no pandas
or external dependencies for table parsing.

Public API:
    ``parse_html_tables(html) -> list[str]``  — returns Markdown table strings.
"""

from __future__ import annotations

import json as _json
import re
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from typing import Any

# ---------------------------------------------------------------------------
# OCR corrections for header text
# ---------------------------------------------------------------------------

_OCR_CORRECTIONS: dict[str, str] = {
    "Receipte": "Receipts",
    "Expendituree": "Expenditures",
    "Expendltures": "Expenditures",
    "Unexpanded": "Unexpended",
}


def _apply_ocr_corrections(text: str) -> str:
    for wrong, right in _OCR_CORRECTIONS.items():
        text = text.replace(wrong, right)
    return text


def _escape_md_cell(text: str) -> str:
    """Escape pipe characters and normalize whitespace for Markdown table cells."""
    return text.replace("|", "\\|").replace("\n", " ").strip()


# ---------------------------------------------------------------------------
# HTML table parser
# ---------------------------------------------------------------------------


class _HTMLTableParser(HTMLParser):
    """Parse ``<table>`` blocks into 2D grids of exact cell text.

    Handles ``rowspan`` / ``colspan`` by maintaining an ``_occupied`` dict
    that tracks which ``(row, col)`` positions are filled.  Cell text is
    preserved exactly as it appears in the HTML source — no type coercion.
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
                # Only parse outermost table
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

            # Fill occupied positions for rowspan/colspan
            for r in range(self._cell_rowspan):
                for c in range(self._cell_colspan):
                    pos = (self._row_idx + r, self._col_idx + c)
                    # Origin cell gets the text; spanned positions get ""
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
# Parsed table dataclass
# ---------------------------------------------------------------------------


@dataclass
class ParsedTable:
    """A parsed HTML table with both Markdown and structured JSON representations."""

    markdown: str
    table_json: str  # JSON string of the structured representation; empty if parse failed


# ---------------------------------------------------------------------------
# Shared header builder (used by both Markdown and JSON paths)
# ---------------------------------------------------------------------------


def _build_typed_headers(
    grid: list[list[str]], header_rows: int, num_cols: int,
) -> tuple[list[str], list[dict[str, Any]], int]:
    """Build header names and typed metadata from a 2D grid.

    Encapsulates multi-row flattening, OCR corrections, deduplication,
    "Category" heuristic, and unique-name enforcement — used by both
    ``_grid_to_markdown`` and ``grid_to_table_json``.

    Returns
    -------
    tuple of:
        flat_names: list[str]
            Unique, OCR-corrected header name strings (for markdown output).
        typed_headers: list[dict]
            ``{"name": str, "parent": str | None, "index": int}`` per column.
            ``name`` is the leaf header, ``parent`` is the ancestor chain
            (joined with `` > ``) or ``None`` for single-row headers.
        data_start_row: int
            Grid row index where data rows begin.
    """
    raw_names: list[str] = []
    parents: list[str | None] = []

    if header_rows == 0:
        # No <th> found — generate positional headers
        raw_names = ["Category"] + [f"col_{i + 1}" for i in range(1, num_cols)]
        parents = [None] * num_cols
        data_start = 0
    elif header_rows == 1:
        raw_names = [grid[0][i] if i < len(grid[0]) else "" for i in range(num_cols)]
        parents = [None] * num_cols
        data_start = 1
    else:
        # Multi-row headers: flatten by column (for markdown-compatible names).
        # NOTE: We do NOT deduplicate consecutive values.  The HTML parser
        # fills rowspan-spanned positions with "" (line 119), so true rowspan
        # repetitions never appear as consecutive duplicates — they appear as
        # empty strings and are skipped by the ``if val:`` check.  Deduplicating
        # consecutive values would silently destroy data when the parser
        # mis-classifies data rows as header rows (e.g., two months both
        # having value "73").
        for col_idx in range(num_cols):
            parts: list[str] = []
            for row_idx in range(header_rows):
                val = grid[row_idx][col_idx] if col_idx < len(grid[row_idx]) else ""
                if val:
                    parts.append(val)
            raw_names.append(" > ".join(parts) if parts else "")

        # Build parents separately with colspan-aware propagation.
        # In the grid, colspan fills spanned cells with "" — we recover the
        # spanning parent by looking left for the nearest non-empty cell in
        # each ancestor header row.  No consecutive-duplicate dropping here
        # either — see comment above for reasoning.
        for col_idx in range(num_cols):
            parent_parts: list[str] = []
            for row_idx in range(header_rows - 1):  # ancestor rows only (not leaf)
                val = grid[row_idx][col_idx] if col_idx < len(grid[row_idx]) else ""
                if not val:
                    # Colspan span — look left for the spanning cell's value
                    for left in range(col_idx - 1, -1, -1):
                        left_val = grid[row_idx][left] if left < len(grid[row_idx]) else ""
                        if left_val:
                            val = left_val
                            break
                if val:
                    parent_parts.append(val)
            parents.append(" > ".join(parent_parts) if parent_parts else None)
        data_start = header_rows

    # --- Clean headers ---
    raw_names = [_apply_ocr_corrections(h) for h in raw_names]

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

    # Build typed headers (leaf name from flat_names, parent from parents list)
    typed: list[dict[str, Any]] = []
    for i in range(num_cols):
        leaf = flat_names[i]
        # For multi-row headers, extract the leaf (last part) from the flat name
        if parents[i] and " > " in leaf:
            leaf = leaf.split(" > ")[-1].strip()
        typed.append({"name": leaf, "parent": parents[i], "index": i})

    # Ensure leaf names are unique — prevent cells dict key collision in
    # grid_to_table_json when two columns share the same leaf after
    # parent-prefix extraction (e.g., "FY1986 > Oct" and "FY1987 > Oct"
    # both extract leaf "Oct").  Mirrors the flat_names pattern above.
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
# Grid → Markdown conversion
# ---------------------------------------------------------------------------


def _grid_to_markdown(grid: list[list[str]], header_rows: int) -> str:
    """Convert a 2D grid of cell text into a Markdown table string."""
    if not grid:
        return ""

    num_cols = max(len(row) for row in grid)

    # Pad rows to uniform column count
    for row in grid:
        while len(row) < num_cols:
            row.append("")

    flat_names, _, data_start = _build_typed_headers(grid, header_rows, num_cols)

    data_rows = grid[data_start:]

    # --- Build Markdown ---
    lines: list[str] = []
    lines.append("| " + " | ".join(_escape_md_cell(h) for h in flat_names) + " |")
    lines.append("| " + " | ".join(["---"] * len(flat_names)) + " |")

    for row in data_rows:
        cells = [_escape_md_cell(row[i] if i < len(row) else "") for i in range(len(flat_names))]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Grid → Structured JSON conversion
# ---------------------------------------------------------------------------

_TOTAL_PATTERN = re.compile(r"^\s*(grand\s+)?total\b", re.IGNORECASE)


def grid_to_table_json(grid: list[list[str]], header_rows: int) -> str:
    """Convert a 2D grid into a structured JSON string.

    Uses the same header-building logic as ``_grid_to_markdown`` (via the
    shared ``_build_typed_headers`` helper) to ensure consistency.

    Returns
    -------
    str
        JSON string with ``headers``, ``rows``, ``row_count``, ``data_row_count``.
    """
    if not grid:
        return "{}"

    num_cols = max(len(row) for row in grid)

    # Pad rows to uniform column count (idempotent if already padded)
    for row in grid:
        while len(row) < num_cols:
            row.append("")

    flat_names, typed_headers, data_start = _build_typed_headers(
        grid, header_rows, num_cols,
    )

    data_rows_grid = grid[data_start:]
    rows: list[dict[str, Any]] = []
    data_row_count = 0

    for row in data_rows_grid:
        label = row[0] if row else ""
        # Build cells dict keyed by header name (skip first column = label)
        cells: dict[str, str] = {}
        for col_idx in range(1, num_cols):
            header_name = typed_headers[col_idx]["name"]
            cells[header_name] = row[col_idx] if col_idx < len(row) else ""

        # Classify row
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

    result = {
        "headers": typed_headers,
        "rows": rows,
        "row_count": len(rows),
        "data_row_count": data_row_count,
    }
    return _json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _feed_and_flush(html: str) -> tuple[_HTMLTableParser, bool]:
    """Feed HTML to parser, flush in-progress tables.  Returns (parser, ok)."""
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


def parse_html_tables(html: str) -> list[str]:
    """Parse HTML containing ``<table>`` elements into Markdown table strings.

    Preserves exact cell text — no NaN injection, no comma stripping, no float
    coercion.  Handles ``rowspan`` and ``colspan``.

    Parameters
    ----------
    html:
        HTML string potentially containing one or more ``<table>`` blocks.

    Returns
    -------
    list[str]:
        One Markdown table string per ``<table>`` found.  If parsing fails,
        returns the HTML wrapped in a code fence as fallback.
    """
    parser, ok = _feed_and_flush(html)

    if not ok or not parser.tables:
        return ["```html\n" + html.strip() + "\n```"]

    return [_grid_to_markdown(grid, hrows) for grid, hrows in parser.tables]


def parse_html_tables_structured(html: str) -> list[ParsedTable]:
    """Parse HTML tables into both Markdown and structured JSON representations.

    Same parsing as :func:`parse_html_tables` but returns :class:`ParsedTable`
    objects carrying both the Markdown string and a structured JSON string.
    The JSON captures header hierarchy, row classification (group header / total),
    and cell-to-header mapping for downstream programmatic access.

    Parameters
    ----------
    html:
        HTML string potentially containing one or more ``<table>`` blocks.

    Returns
    -------
    list[ParsedTable]:
        One ``ParsedTable`` per ``<table>`` found.
    """
    parser, ok = _feed_and_flush(html)

    if not ok or not parser.tables:
        return [ParsedTable(
            markdown="```html\n" + html.strip() + "\n```",
            table_json="",
        )]

    return [
        ParsedTable(
            markdown=_grid_to_markdown(grid, hrows),
            table_json=grid_to_table_json(grid, hrows),
        )
        for grid, hrows in parser.tables
    ]
