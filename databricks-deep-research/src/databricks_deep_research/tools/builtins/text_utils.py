"""Table-aware text utilities for content chunking and markdown table detection.

Provides **Layer 2** of the two-layer table extraction strategy: detect
markdown pipe-tables in text from *any* source (web_crawl, web_search,
file_search, vector_search) regardless of how the text was produced.

Public API:

* ``detect_markdown_tables(text)`` — find pipe-tables in arbitrary text
* ``chunk_content(text)`` — split mixed text+table content into typed chunks
* ``sanitize_text(content)`` — remove control characters
* ``is_table_line(line)`` — test whether a line is part of a pipe-table
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from databricks_deep_research.tools.builtins.html_tables import ParsedTable

# ---------------------------------------------------------------------------
# Chunk type enum
# ---------------------------------------------------------------------------


class ChunkType(StrEnum):
    """Content type of a chunk."""

    text = "text"
    table = "table"
    section = "section"


# ---------------------------------------------------------------------------
# Text sanitization
# ---------------------------------------------------------------------------

# Control chars except \t (0x09), \n (0x0A), \r (0x0D)
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_text(content: str) -> str:
    """Remove null bytes and control characters, normalize newlines.

    Preserves tabs, newlines, and all printable content.  Suitable for
    cleaning text before database storage or LLM consumption.
    """
    content = _CONTROL_CHAR_RE.sub("", content)
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    return content


# ---------------------------------------------------------------------------
# Pipe-table line detection
# ---------------------------------------------------------------------------

# Separator row pattern: | --- | --- | (with optional colons for alignment)
_SEPARATOR_RE = re.compile(r"^\s*\|[\s:]*-{2,}[\s:]*(\|[\s:]*-{2,}[\s:]*)+\|?\s*$")


def is_table_line(line: str) -> bool:
    """Return ``True`` if *line* looks like part of a markdown pipe-table.

    A pipe-table line starts with ``|`` and contains at least 2 ``|`` chars.
    Does NOT match lines that merely contain pipes (shell commands, etc.).
    """
    stripped = line.strip()
    if not stripped.startswith("|"):
        return False
    return stripped.count("|") >= 2


def _is_separator_line(line: str) -> bool:
    """Return ``True`` if *line* is a pipe-table separator (``|---|---|``)."""
    return bool(_SEPARATOR_RE.match(line))


# ---------------------------------------------------------------------------
# Markdown table detection (Layer 2)
# ---------------------------------------------------------------------------


def detect_markdown_tables(text: str) -> list[ParsedTable]:
    """Detect markdown pipe-tables in arbitrary text.

    **Layer 2** of the two-layer table extraction strategy.  Works on text
    from *any* source — Jina crawl, Jina search, file search results,
    vector search results, or text where Layer 1 (HTML rescue) already
    embedded markdown tables.

    Algorithm:

    1. Split text into lines.
    2. Find consecutive runs of pipe-table lines (via :func:`is_table_line`).
    3. Skip runs inside fenced code blocks (``````` or ``~~~``).
    4. Require a separator row (``|---|---|``) to confirm the run is a table.
    5. Parse confirmed tables into :class:`ParsedTable` with markdown +
       Table-compatible ``table_json`` dict.
    6. Skip tables with fewer than 2 data rows (header-only = not useful).

    Parameters
    ----------
    text:
        Arbitrary text that may contain markdown pipe-tables.

    Returns
    -------
    list[ParsedTable]:
        Detected tables.  Empty list if none found.
    """
    lines = text.split("\n")
    tables: list[ParsedTable] = []

    in_fence = False
    run: list[str] = []

    for line in lines:
        stripped = line.strip()

        # Track fenced code blocks
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            if run:
                _flush_run(run, tables)
                run = []
            continue

        if in_fence:
            continue

        if is_table_line(stripped):
            run.append(line)
        else:
            if run:
                _flush_run(run, tables)
                run = []

    # Flush final run
    if run:
        _flush_run(run, tables)

    return tables


def _flush_run(run: list[str], tables: list[ParsedTable]) -> None:
    """Validate a run of pipe-table lines and append to *tables* if valid."""
    if len(run) < 3:
        # Need at least header + separator + 1 data row
        run.clear()
        return

    # Must contain a separator row
    has_separator = False
    sep_idx = -1
    for i, line in enumerate(run):
        if _is_separator_line(line):
            has_separator = True
            sep_idx = i
            break

    if not has_separator:
        run.clear()
        return

    # Parse: everything before separator = header rows, after = data rows
    header_lines = run[:sep_idx]
    data_lines = run[sep_idx + 1:]

    if not header_lines or not data_lines:
        run.clear()
        return

    # Build grid from pipe-delimited lines
    all_content_lines = header_lines + data_lines
    grid = _pipe_lines_to_grid(all_content_lines)
    header_row_count = len(header_lines)

    if not grid:
        run.clear()
        return

    num_rows = len(grid)
    num_cols = max((len(row) for row in grid), default=0)

    if num_cols < 2:
        run.clear()
        return

    # Build markdown (the original lines, joined)
    markdown = "\n".join(run)

    # Build table_json via the shared header builder
    from databricks_deep_research.tools.builtins.html_tables import (
        _TOTAL_PATTERN,
        _build_typed_headers,
    )

    for row in grid:
        while len(row) < num_cols:
            row.append("")

    _, typed_headers, data_start = _build_typed_headers(grid, header_row_count, num_cols)
    data_rows_grid = grid[data_start:]

    rows: list[dict[str, Any]] = []
    data_row_count = 0
    for row in data_rows_grid:
        label = row[0] if row else ""
        cells: dict[str, str] = {}
        for col_idx in range(1, num_cols):
            header_name = typed_headers[col_idx]["name"]
            cells[header_name] = row[col_idx] if col_idx < len(row) else ""

        non_empty = sum(1 for v in cells.values() if v.strip())
        is_group_header = non_empty == 0 and bool(label.strip())
        is_total = bool(_TOTAL_PATTERN.match(label.strip())) if label else False

        rows.append({
            "label": label,
            "cells": cells,
            "is_group_header": is_group_header,
            "is_total": is_total,
        })
        if not is_group_header:
            data_row_count += 1

    table_json: dict[str, Any] = {
        "headers": typed_headers,
        "rows": rows,
        "row_count": len(rows),
        "data_row_count": data_row_count,
    }

    tables.append(ParsedTable(
        markdown=markdown,
        table_json=table_json,
        row_count=num_rows,
        col_count=num_cols,
    ))

    run.clear()


def _pipe_lines_to_grid(lines: list[str]) -> list[list[str]]:
    """Parse pipe-delimited lines into a 2D grid of cell text."""
    grid: list[list[str]] = []
    for line in lines:
        stripped = line.strip()
        # Remove leading/trailing pipes and split
        if stripped.startswith("|"):
            stripped = stripped[1:]
        if stripped.endswith("|"):
            stripped = stripped[:-1]
        cells = [c.strip() for c in stripped.split("|")]
        grid.append(cells)
    return grid


# ---------------------------------------------------------------------------
# Content splitting into typed blocks
# ---------------------------------------------------------------------------


def split_into_blocks(text: str) -> list[tuple[str, str]]:
    """Split text into typed blocks: ``(content, chunk_type)``.

    Detects markdown pipe-table blocks and labels them as
    :attr:`ChunkType.table`; everything else is :attr:`ChunkType.text`.
    Fenced code blocks are kept as text (not scanned for tables).
    """
    lines = text.split("\n")
    blocks: list[tuple[str, str]] = []

    current_lines: list[str] = []
    current_type: str = ChunkType.text
    in_fence = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            current_lines.append(line)
            continue

        if in_fence:
            current_lines.append(line)
            continue

        line_is_table = is_table_line(stripped) or _is_separator_line(stripped)

        if line_is_table and current_type == ChunkType.text:
            # Start of table — flush text block
            if current_lines:
                content = "\n".join(current_lines).strip()
                if content:
                    blocks.append((content, ChunkType.text))
            current_lines = [line]
            current_type = ChunkType.table

        elif not line_is_table and current_type == ChunkType.table:
            # End of table — flush table block
            if current_lines:
                content = "\n".join(current_lines).strip()
                if content:
                    blocks.append((content, ChunkType.table))
            current_lines = [line]
            current_type = ChunkType.text

        else:
            current_lines.append(line)

    # Flush remaining
    if current_lines:
        content = "\n".join(current_lines).strip()
        if content:
            blocks.append((content, current_type))

    return blocks


# ---------------------------------------------------------------------------
# Table-aware chunking
# ---------------------------------------------------------------------------


def chunk_table(table_text: str, *, max_chars: int = 4000) -> list[str]:
    """Split a large markdown table by rows, preserving header in each chunk.

    If the table fits within *max_chars*, returns it as a single chunk.
    Otherwise, splits at row boundaries, repeating the header row and
    separator in each chunk.
    """
    lines = table_text.split("\n")
    if len(lines) < 3:
        return [table_text]

    header_lines = lines[:2]  # header + separator
    data_lines = lines[2:]
    header_text = "\n".join(header_lines)
    header_len = len(header_text) + 1  # +1 for newline

    if len(table_text) <= max_chars:
        return [table_text]

    chunks: list[str] = []
    current_rows: list[str] = []
    current_len = header_len

    for row in data_lines:
        row_len = len(row) + 1
        if current_rows and current_len + row_len > max_chars:
            chunks.append(header_text + "\n" + "\n".join(current_rows))
            current_rows = []
            current_len = header_len

        current_rows.append(row)
        current_len += row_len

    if current_rows:
        chunks.append(header_text + "\n" + "\n".join(current_rows))

    return chunks if chunks else [table_text]


def chunk_text(
    text: str,
    *,
    max_chars: int = 2000,
    overlap: int = 200,
) -> list[str]:
    """Split text at paragraph boundaries with overlap.

    Splits on double-newline boundaries.  Each chunk is at most *max_chars*.
    Adjacent chunks share *overlap* characters for context continuity.
    """
    if len(text) <= max_chars:
        return [text]

    paragraphs = re.split(r"\n\n+", text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para) + 2  # +2 for \n\n separator
        if current and current_len + para_len > max_chars:
            chunk = "\n\n".join(current)
            chunks.append(chunk)
            # Overlap: keep last paragraph(s) that fit within overlap budget
            overlap_parts: list[str] = []
            overlap_len = 0
            for p in reversed(current):
                if overlap_len + len(p) + 2 > overlap:
                    break
                overlap_parts.insert(0, p)
                overlap_len += len(p) + 2
            current = overlap_parts
            current_len = overlap_len

        current.append(para)
        current_len += para_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks if chunks else [text]


def chunk_content(
    content: str,
    *,
    chunk_max_chars: int = 2000,
    table_max_chars: int = 4000,
    overlap: int = 200,
) -> list[tuple[str, str]]:
    """Split mixed text+table content into table-aware typed chunks.

    Tables are kept whole when they fit within *table_max_chars*; large
    tables are split by rows with headers preserved in each chunk.  Text
    blocks are split at paragraph boundaries with overlap.

    Parameters
    ----------
    content:
        Mixed text that may contain markdown pipe-tables.
    chunk_max_chars:
        Maximum chars per text chunk.
    table_max_chars:
        Maximum chars per table chunk.
    overlap:
        Character overlap between adjacent text chunks.

    Returns
    -------
    list[tuple[str, str]]:
        ``(chunk_content, chunk_type)`` pairs where ``chunk_type`` is
        ``"text"`` or ``"table"``.
    """
    blocks = split_into_blocks(content)
    result: list[tuple[str, str]] = []

    for block_content, block_type in blocks:
        if block_type == ChunkType.table:
            for chunk in chunk_table(block_content, max_chars=table_max_chars):
                result.append((chunk, ChunkType.table))
        else:
            for chunk in chunk_text(block_content, max_chars=chunk_max_chars, overlap=overlap):
                result.append((chunk, ChunkType.text))

    return result
