"""Structured table wrapper for LLM-generated compute code.

Wraps the ``table_json`` dict produced by ``html_table_parser.grid_to_table_json()``
with typed extraction methods.  Injected into the compute namespace by
``TableLoadTool`` so that agent code can call ``table.cell()``,
``table.series()``, etc. instead of manually traversing nested dicts.

Backward-compatible with raw dict access via ``__getitem__`` / ``get()``.
"""

from __future__ import annotations

import difflib
import math
import re
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MISSING_SENTINELS = frozenset(
    {"", "-", "--", "\u2014", "*", "(*)", "n.a.", "N/A", "(X)", "..."}
)

_FOOTNOTE_SUFFIX_RE = re.compile(r"\s+\d+/\s*$")


# ---------------------------------------------------------------------------
# Table class
# ---------------------------------------------------------------------------


class Table:
    """Structured table with typed extraction API.

    Parameters
    ----------
    data
        Parsed ``table_json`` dict with ``headers`` and ``rows`` keys.
    chunk_id, file_name, title, annotation
        Optional metadata propagated from the Delta table row.
    """

    def __init__(
        self,
        data: dict[str, Any],
        *,
        chunk_id: str = "",
        file_name: str = "",
        title: str = "",
        annotation: str = "",
    ) -> None:
        self._data = data
        self.chunk_id = chunk_id
        self.file_name = file_name
        self.title = title
        self.annotation = annotation

        self._headers: list[dict[str, Any]] = data.get("headers", [])
        self._rows: list[dict[str, Any]] = data.get("rows", [])

        # Pre-build indexes for O(1) lookup.
        self._row_index: dict[str, int] = {}
        for i, r in enumerate(self._rows):
            label = r.get("label", "").strip()
            if label and label not in self._row_index:
                self._row_index[label] = i

        self._col_names: list[str] = [
            h.get("name", "").strip() for h in self._headers
        ]
        self._col_set: set[str] = {c for c in self._col_names if c}

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def columns(self) -> list[str]:
        """Column names in document order."""
        return list(self._col_names)

    @property
    def parents(self) -> dict[str, str]:
        """Map column name → parent header chain (empty string if none)."""
        return {
            h["name"]: h.get("parent") or ""
            for h in self._headers
            if h.get("name")
        }

    @property
    def labels(self) -> list[str]:
        """All row labels (including totals and group headers)."""
        return [r.get("label", "").strip() for r in self._rows]

    @property
    def entity_labels(self) -> list[str]:
        """Non-total, non-group-header row labels."""
        return [
            r["label"].strip()
            for r in self._rows
            if not r.get("is_total")
            and not r.get("is_group_header")
            and r.get("label", "").strip()
        ]

    @property
    def total_labels(self) -> list[str]:
        """Labels of rows marked as totals."""
        return [r["label"].strip() for r in self._rows if r.get("is_total")]

    @property
    def row_count(self) -> int:
        return int(self._data.get("row_count", len(self._rows)))

    @property
    def data_row_count(self) -> int:
        return int(self._data.get("data_row_count", len(self.entity_labels)))

    # ------------------------------------------------------------------
    # Cell access
    # ------------------------------------------------------------------

    def cell(
        self,
        row_label: str,
        column: str,
        *,
        as_float: bool = False,
    ) -> str | float:
        """Get one cell by row label and column name.

        Uses fuzzy matching (cutoff 0.6) when an exact match is not found.

        Raises
        ------
        KeyError
            With a list of available labels/columns when no match is found.
        """
        row = self._resolve_row(row_label)
        col = self._resolve_column(column)
        raw = row.get("cells", {}).get(col, "")
        return to_float(raw) if as_float else raw

    def row_dict(
        self,
        row_label: str,
        *,
        as_float: bool = False,
    ) -> dict[str, str | float]:
        """All cells for a row as ``{column: value}``."""
        row = self._resolve_row(row_label)
        cells = row.get("cells", {})
        if as_float:
            return {k: to_float(v) for k, v in cells.items()}
        return dict(cells)

    # ------------------------------------------------------------------
    # Series access
    # ------------------------------------------------------------------

    def series(
        self,
        column: str,
        *,
        as_float: bool = False,
        exclude_totals: bool = True,
        exclude_headers: bool = True,
    ) -> list[tuple[str, Any]]:
        """Extract a column as ``[(label, value), ...]`` pairs."""
        col = self._resolve_column(column)
        result: list[tuple[str, Any]] = []
        for r in self._rows:
            if exclude_totals and r.get("is_total"):
                continue
            if exclude_headers and r.get("is_group_header"):
                continue
            label = r.get("label", "").strip()
            raw = r.get("cells", {}).get(col, "")
            val = to_float(raw) if as_float else raw
            result.append((label, val))
        return result

    def column_values(
        self,
        column: str,
        *,
        as_float: bool = True,
        exclude_totals: bool = True,
    ) -> list[float]:
        """Extract only the numeric values from a column (no labels).

        Non-numeric cells (NaN) are excluded from the result.
        """
        pairs = self.series(
            column, as_float=as_float, exclude_totals=exclude_totals
        )
        return [v for _, v in pairs if isinstance(v, (int, float)) and not math.isnan(v)]

    # ------------------------------------------------------------------
    # Search helpers
    # ------------------------------------------------------------------

    def find_rows(self, pattern: str) -> list[str]:
        """Find row labels containing *pattern* (case-insensitive)."""
        pat = pattern.lower()
        return [label for label in self.labels if pat in label.lower()]

    def find_columns(self, pattern: str) -> list[str]:
        """Find column names containing *pattern* (case-insensitive)."""
        pat = pattern.lower()
        return [c for c in self._col_names if pat in c.lower()]

    def has_label(self, label: str) -> bool:
        """Check whether an exact row label exists."""
        return label.strip() in self._row_index

    # ------------------------------------------------------------------
    # Structural helpers
    # ------------------------------------------------------------------

    def column_parent(self, column: str) -> str:
        """Get the parent header chain for a column."""
        col = self._resolve_column(column)
        return self.parents.get(col, "")

    def describe(self) -> str:
        """Return a concise structural summary string."""
        lines: list[str] = [
            f"Table: {self.title}" if self.title else "Table (untitled)"
        ]
        if self.annotation:
            lines.append(f"  {self.annotation}")
        lines.append(f"  {self.row_count}R x {len(self._col_names)}C")
        cols_preview = ", ".join(self._col_names[:8])
        if len(self._col_names) > 8:
            cols_preview += ", ..."
        lines.append(f"  Columns: {cols_preview}")
        if self.entity_labels:
            sample = self.entity_labels[:5]
            labels_preview = ", ".join(sample)
            if len(self.entity_labels) > 5:
                labels_preview += ", ..."
            lines.append(f"  Row labels: {labels_preview}")
        if self.total_labels:
            lines.append(f"  Totals: {', '.join(self.total_labels)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_dataframe(self, *, include_totals: bool = True) -> Any:
        """Convert to a ``pandas.DataFrame``.  Pandas is imported lazily."""
        import pandas as pd  # noqa: PLC0415

        records: list[dict[str, Any]] = []
        for r in self._rows:
            if not include_totals and r.get("is_total"):
                continue
            if r.get("is_group_header"):
                continue
            record: dict[str, Any] = {"_label": r.get("label", "").strip()}
            record.update(r.get("cells", {}))
            records.append(record)
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Dict-like access (backward compatibility with raw dict)
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        """Allow ``table['headers']``, ``table['rows']`` etc."""
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-compatible ``.get()``."""
        return self._data.get(key, default)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"Table({self.chunk_id!r}, {self.row_count}R x {len(self._col_names)}C)"

    # ------------------------------------------------------------------
    # Internal resolution helpers
    # ------------------------------------------------------------------

    def _resolve_row(self, label: str) -> dict[str, Any]:
        """Find a row by exact match first, then fuzzy fallback."""
        stripped = label.strip()
        if stripped in self._row_index:
            return self._rows[self._row_index[stripped]]
        # Fuzzy fallback
        matches = difflib.get_close_matches(
            stripped, self._row_index.keys(), n=3, cutoff=0.6
        )
        if matches:
            return self._rows[self._row_index[matches[0]]]
        available = list(self._row_index.keys())
        preview = available[:15]
        suffix = ", ..." if len(available) > 15 else ""
        raise KeyError(
            f"Row label {label!r} not found. Available: {preview}{suffix}"
        )

    def _resolve_column(self, column: str) -> str:
        """Find a column by exact match first, then fuzzy fallback."""
        stripped = column.strip()
        if stripped in self._col_set:
            return stripped
        matches = difflib.get_close_matches(
            stripped, self._col_set, n=3, cutoff=0.6
        )
        if matches:
            return matches[0]
        raise KeyError(
            f"Column {column!r} not found. Available: {sorted(self._col_set)}"
        )


# ---------------------------------------------------------------------------
# Module-level utility (also used by TableLoadTool)
# ---------------------------------------------------------------------------


def to_float(raw: str | Any) -> float:
    """Convert a table cell value to ``float``.

    Handles thousands commas, parenthetical negatives, dashes, asterisks,
    footnote markers, and common "not available" sentinels.  Returns
    ``float('nan')`` for unparseable or missing values.
    """
    if not raw or not isinstance(raw, str):
        return float("nan")
    cleaned = raw.strip()
    if cleaned in _MISSING_SENTINELS:
        return float("nan")
    # Thousands separators
    cleaned = cleaned.replace(",", "")
    # Parenthetical negatives: (500) → -500
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = "-" + cleaned[1:-1]
    # Trailing footnote markers: "123 1/" → "123"
    cleaned = _FOOTNOTE_SUFFIX_RE.sub("", cleaned)
    try:
        return float(cleaned)
    except ValueError:
        return float("nan")
