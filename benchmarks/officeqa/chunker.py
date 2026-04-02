"""Table-aware Markdown chunking for Treasury Bulletin text files.

The OfficeQA `transform_parsed_files.py` produces files named
`treasury_bulletin_{YEAR}_{MONTH}.txt` with:
- Plain text paragraphs (separated by blank lines)
- Markdown tables with `|` delimiters (header + `|---|---|` separator + rows)
- Multi-level headers flattened with " > " separator
- Escaped pipes in cells (`\\|`)
- Occasional HTML fallback blocks in ```html``` fences
"""

from __future__ import annotations

import json as _json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    """Chunking parameters."""

    chunk_max_chars: int = 2000
    chunk_overlap_chars: int = 200
    table_max_chars: int = 4000
    section_max_chars: int = 8000  # heading + table + footnotes grouped as one chunk


@dataclass
class Chunk:
    """A single chunk of text or table content."""

    chunk_id: str = ""
    file_name: str = ""
    bulletin_date: str = ""  # "YYYY-MM" from filename
    page_info: str = ""  # nearest heading before this chunk
    content: str = ""
    chunk_type: str = "text"  # "text" | "table" | "html_fallback"
    char_count: int = 0


def _is_table_line(line: str) -> bool:
    """A line is a table line if it starts with '|' and contains 2+ pipes."""
    stripped = line.strip()
    if not stripped.startswith("|"):
        return False
    return stripped.count("|") >= 2


def _extract_date(filename: str) -> str:
    """Extract YYYY-MM from filename like treasury_bulletin_2024_03.txt."""
    match = re.search(r"(\d{4})_(\d{2})", filename)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    logger.warning("CHUNKER_NO_DATE filename=%s", filename)
    return ""


def _split_into_blocks(
    text: str,
) -> list[tuple[str, str, str]]:
    """Split text into (block_type, block_text, heading_context) tuples.

    Table detection: a line is a table line if it starts with '|' after
    stripping whitespace AND contains at least 2 pipe characters.
    A table block is a maximal run of consecutive table lines (blank lines
    between table rows tolerated up to 1).

    HTML fallback: ```html ... ``` fenced blocks.

    Heading tracking: lines starting with '#' update the current heading context.
    """
    lines = text.split("\n")
    blocks: list[tuple[str, str, str]] = []
    current_heading = ""
    i = 0

    while i < len(lines):
        line = lines[i]

        # Track headings
        if line.strip().startswith("#"):
            current_heading = line.strip().lstrip("#").strip()
            i += 1
            continue

        # HTML fallback block
        if line.strip().startswith("```html"):
            html_lines = [line]
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                html_lines.append(lines[i])
                i += 1
            if i < len(lines):
                html_lines.append(lines[i])
                i += 1
            blocks.append(("html_fallback", "\n".join(html_lines), current_heading))
            continue

        # Table block
        if _is_table_line(line):
            table_lines = [line]
            i += 1
            while i < len(lines):
                if _is_table_line(lines[i]):
                    table_lines.append(lines[i])
                    i += 1
                elif lines[i].strip() == "" and i + 1 < len(lines) and _is_table_line(
                    lines[i + 1]
                ):
                    # Tolerate 1 blank line within a table
                    table_lines.append(lines[i])
                    i += 1
                else:
                    break
            blocks.append(("table", "\n".join(table_lines), current_heading))
            continue

        # Text line — accumulate until next non-text block
        text_lines = []
        while i < len(lines):
            if lines[i].strip().startswith("#"):
                break
            if _is_table_line(lines[i]):
                break
            if lines[i].strip().startswith("```html"):
                break
            text_lines.append(lines[i])
            i += 1

        text_block = "\n".join(text_lines).strip()
        if text_block:
            blocks.append(("text", text_block, current_heading))

    return blocks


def _chunk_table(
    table_text: str, heading: str, config: ChunkConfig
) -> list[Chunk]:
    """Keep tables whole if possible. Split large tables by rows, preserving header."""
    if len(table_text) <= config.table_max_chars:
        return [
            Chunk(
                page_info=heading,
                content=table_text,
                chunk_type="table",
                char_count=len(table_text),
            )
        ]

    # Split: extract header (line 0) + separator (line 1), split rows
    lines = table_text.split("\n")
    header_lines: list[str] = []
    data_lines: list[str] = []

    # Find header + separator
    for j, line in enumerate(lines):
        if re.match(r"^\s*\|[\s\-:|]+\|\s*$", line):
            header_lines = lines[: j + 1]
            data_lines = lines[j + 1 :]
            break
    else:
        # No separator found — treat first line as header
        header_lines = lines[:1]
        data_lines = lines[1:]

    header_text = "\n".join(header_lines)
    header_len = len(header_text) + 1  # +1 for newline

    chunks: list[Chunk] = []
    current_rows: list[str] = []
    current_len = header_len

    for row in data_lines:
        row_len = len(row) + 1
        if current_len + row_len > config.table_max_chars and current_rows:
            chunk_content = header_text + "\n" + "\n".join(current_rows)
            chunks.append(
                Chunk(
                    page_info=heading,
                    content=chunk_content,
                    chunk_type="table",
                    char_count=len(chunk_content),
                )
            )
            current_rows = []
            current_len = header_len

        current_rows.append(row)
        current_len += row_len

    # Flush remaining rows
    if current_rows:
        chunk_content = header_text + "\n" + "\n".join(current_rows)
        chunks.append(
            Chunk(
                page_info=heading,
                content=chunk_content,
                chunk_type="table",
                char_count=len(chunk_content),
            )
        )

    if not chunks:
        # Edge case: single row larger than table_max_chars — emit as-is
        logger.warning(
            "CHUNKER_OVERSIZED_TABLE chars=%d heading=%s", len(table_text), heading
        )
        chunks.append(
            Chunk(
                page_info=heading,
                content=table_text,
                chunk_type="table",
                char_count=len(table_text),
            )
        )

    return chunks


def _chunk_text(
    text: str, heading: str, config: ChunkConfig
) -> list[Chunk]:
    """Split at paragraph boundaries with overlap."""
    paragraphs = re.split(r"\n\n+", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    if not paragraphs:
        return []

    chunks: list[Chunk] = []
    current_parts: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)

        if current_len + para_len + 2 > config.chunk_max_chars and current_parts:
            chunk_content = "\n\n".join(current_parts)
            chunks.append(
                Chunk(
                    page_info=heading,
                    content=chunk_content,
                    chunk_type="text",
                    char_count=len(chunk_content),
                )
            )
            # Overlap: carry the last part if it fits in overlap budget
            if config.chunk_overlap_chars > 0 and current_parts:
                last = current_parts[-1]
                if len(last) <= config.chunk_overlap_chars:
                    current_parts = [last]
                    current_len = len(last)
                else:
                    current_parts = []
                    current_len = 0
            else:
                current_parts = []
                current_len = 0

        current_parts.append(para)
        current_len += para_len + 2  # +2 for "\n\n" separator

    # Flush remaining
    if current_parts:
        chunk_content = "\n\n".join(current_parts)
        chunks.append(
            Chunk(
                page_info=heading,
                content=chunk_content,
                chunk_type="text",
                char_count=len(chunk_content),
            )
        )

    return chunks


def _merge_into_sections(
    blocks: list[tuple[str, str, str]],
    config: ChunkConfig,
) -> list[tuple[str, str, str]]:
    """Merge consecutive text→table→text blocks into section blocks.

    A *section* is a heading/description text block, followed by a table, and
    optionally followed by footnote/source text.  When the combined size fits
    within ``config.section_max_chars`` the three are merged into a single
    ``("section", combined_text, heading)`` block.  If the combined size
    exceeds the limit the blocks are returned individually (fallback to
    per-block chunking).
    """
    merged: list[tuple[str, str, str]] = []
    i = 0
    while i < len(blocks):
        btype, btext, heading = blocks[i]

        if btype == "text" and i + 1 < len(blocks) and blocks[i + 1][0] == "table":
            # Candidate: text_before + table (+ optional text_after)
            text_before = btext
            table_block = blocks[i + 1]
            text_after = ""
            consumed = 2  # text_before + table

            # Look ahead for a trailing text block (footnotes/source line)
            if (
                i + 2 < len(blocks)
                and blocks[i + 2][0] == "text"
                # Don't consume if the next block is very large (probably a
                # separate narrative section, not a footnote)
                and len(blocks[i + 2][1]) < 1000
            ):
                text_after = blocks[i + 2][1]
                consumed = 3

            combined = "\n\n".join(
                part for part in [text_before, table_block[1], text_after] if part
            )
            if len(combined) <= config.section_max_chars:
                merged.append(("section", combined, heading or table_block[2]))
                i += consumed
                continue

        # Default — pass through unchanged
        merged.append((btype, btext, heading))
        i += 1

    return merged


def _prepend_context(content: str, file_name: str, bulletin_date: str, heading: str) -> str:
    """Prepend document-level context to chunk content for better embeddings.

    Per OfficeQA paper (arxiv 2603.08655, Section D.4) contextual embeddings
    that include document name, date, and section title yield +21% accuracy.
    """
    parts: list[str] = []
    if file_name:
        parts.append(f"Document: {file_name}")
    if bulletin_date:
        parts.append(f"Bulletin date: {bulletin_date}")
    if heading:
        parts.append(f"Section: {heading}")
    if parts:
        return "\n".join(parts) + "\n\n" + content
    return content


def chunk_file(file_path: Path, config: ChunkConfig | None = None) -> list[Chunk]:
    """Chunk a single Treasury Bulletin text file.

    Parameters
    ----------
    file_path:
        Path to the .txt file.
    config:
        Chunking parameters. Uses defaults if None.

    Returns
    -------
    list[Chunk]:
        Chunks with IDs, metadata, and content.
    """
    if config is None:
        config = ChunkConfig()

    text = file_path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        logger.warning("CHUNKER_EMPTY_FILE path=%s", file_path)
        return []

    blocks = _split_into_blocks(text)

    # Merge heading + table + footnote blocks into sections when they fit
    blocks = _merge_into_sections(blocks, config)

    bulletin_date = _extract_date(file_path.name)
    chunks: list[Chunk] = []

    for block_type, block_text, heading_context in blocks:
        if block_type == "section":
            # Already merged — emit as a single chunk with contextual prefix
            content = _prepend_context(
                block_text, file_path.name, bulletin_date, heading_context
            )
            chunks.append(
                Chunk(
                    page_info=heading_context,
                    content=content,
                    chunk_type="section",
                    char_count=len(content),
                )
            )
        elif block_type == "table":
            table_chunks = _chunk_table(block_text, heading_context, config)
            for c in table_chunks:
                c.content = _prepend_context(
                    c.content, file_path.name, bulletin_date, heading_context
                )
                c.char_count = len(c.content)
            chunks.extend(table_chunks)
        elif block_type == "html_fallback":
            html_chunks = _chunk_text(block_text, heading_context, config)
            for c in html_chunks:
                c.chunk_type = "html_fallback"
                c.content = _prepend_context(
                    c.content, file_path.name, bulletin_date, heading_context
                )
                c.char_count = len(c.content)
            chunks.extend(html_chunks)
        else:
            text_chunks = _chunk_text(block_text, heading_context, config)
            for c in text_chunks:
                c.content = _prepend_context(
                    c.content, file_path.name, bulletin_date, heading_context
                )
                c.char_count = len(c.content)
            chunks.extend(text_chunks)

    # Assign IDs and metadata
    for i, chunk in enumerate(chunks):
        chunk.chunk_id = f"{file_path.stem}_c{i:04d}"
        chunk.file_name = file_path.name
        chunk.bulletin_date = bulletin_date

    logger.info(
        "CHUNKER_DONE file=%s chunks=%d total_chars=%d",
        file_path.name,
        len(chunks),
        sum(c.char_count for c in chunks),
    )
    return chunks


# ---------------------------------------------------------------------------
# Structured table records (v9+)
# ---------------------------------------------------------------------------

_ANNOTATION_RE = re.compile(
    r"\([Ii]n\s+(millions|billions|thousands|percent|basis\s+points)[^)]*\)"
)


@dataclass
class TableRecord:
    """Structured table record for the ``treasury_tables`` Delta table.

    Each record mirrors a table-type (or table-containing section) chunk in
    ``treasury_chunks`` and carries the complete JSON representation of the
    table.  The ``chunk_id`` field matches the corresponding ``Chunk.chunk_id``
    so the agent can look up the JSON by chunk_id after finding the table via
    vector search or grep.
    """

    chunk_id: str = ""
    file_name: str = ""
    bulletin_date: str = ""
    page_info: str = ""
    table_title: str = ""
    annotation: str = ""
    content: str = ""  # brief summary for _format_rows display
    table_json: str = ""  # complete JSON string
    row_count: int = 0
    col_count: int = 0
    chunk_type: str = "table"
    char_count: int = 0


def _extract_annotation(text: str) -> str:
    """Extract unit annotation like '(In millions of dollars)' from text."""
    m = _ANNOTATION_RE.search(text)
    return m.group(0) if m else ""


def _chunk_contains_table(chunk: Chunk) -> bool:
    """Check if a chunk's content contains a markdown table."""
    return any(_is_table_line(line) for line in chunk.content.split("\n"))


def _extract_pre_table_title(content: str) -> str:
    """Extract the table title line from text preceding the markdown table.

    Treasury bulletins often have a title line like:
    ``## TABLE TSO-3 — Interest-Bearing Marketable Public Debt Securities``
    before the pipe-delimited table.  This title contains the table identifier
    (TSO-3, CM-I-1, etc.) that agents search for but which doesn't appear in
    ``page_info`` (section heading) or column names.
    """
    lines = content.split("\n")
    # Collect non-metadata text lines that appear before the first table line
    candidates: list[str] = []
    _METADATA_PREFIXES = ("Document:", "Bulletin date:", "Section:", "chunk_id:")
    for line in lines:
        stripped = line.strip()
        if _is_table_line(stripped):
            break
        if not stripped:
            continue
        if stripped.startswith(_METADATA_PREFIXES):
            continue
        cleaned = stripped.lstrip("#").strip().rstrip("*").strip()
        if len(cleaned) > 5:
            candidates.append(cleaned)
    # Return all candidate lines joined — these typically include the table
    # designation ("TABLE TSO-3 — ...") and/or the annotation ("(In millions...)")
    return " | ".join(candidates) if candidates else ""


_MAX_ENTITY_LABELS = 30
_ENTITY_HALF = _MAX_ENTITY_LABELS // 2


def _build_table_summary(
    table_data: dict[str, Any],
    file_name: str,
    bulletin_date: str,
    page_info: str,
    table_title: str,
    annotation: str,
    pre_table_title: str = "",
) -> str:
    """Build a rich, search-optimized summary for a structured table record.

    The summary is stored in the ``content`` column of ``treasury_tables``
    and serves as the search target for ``treasury_table_grep`` and the
    display body for ``treasury_table_list``.  It must contain ALL terms
    an agent might grep for: table titles, column names, header parent
    chains, entity row labels, period ranges, total row labels, and unit
    annotations.

    Returns
    -------
    str
        Multi-line summary text.  ``chunk_id:`` is NOT included here —
        it is prepended per-record by ``build_table_records`` since split
        table chunks have different chunk_ids.
    """
    headers = table_data.get("headers", [])
    rows = table_data.get("rows", [])
    row_count = table_data.get("row_count", len(rows))
    col_count = len(headers)

    parts: list[str] = []

    # Document context
    if file_name:
        parts.append(f"Document: {file_name}")
    if bulletin_date:
        parts.append(f"Bulletin date: {bulletin_date}")
    if page_info:
        parts.append(f"Section: {page_info}")

    # Table title
    if table_title:
        parts.append(f"TABLE: {table_title}")

    # Pre-table title (contains table identifiers like "TSO-3", "CM-I-1")
    if pre_table_title and pre_table_title != table_title:
        parts.append(f"Title: {pre_table_title}")

    # Unit annotation
    if annotation:
        parts.append(annotation)

    # Header parent chains (critical for flow-vs-stock: "maturing", "by Issue").
    # Preserve column order; deduplicate only exact-identical chains.
    parent_unique = list(dict.fromkeys(
        h["parent"] for h in headers if h.get("parent")
    ))
    if parent_unique:
        parts.append(f"Header context: {'; '.join(parent_unique)}")

    # Column names
    if headers:
        col_names = [h.get("name", "") for h in headers]
        parts.append(f"Columns: {' | '.join(col_names)}")

    # Classify rows for period range, totals, entities
    data_labels: list[str] = []
    total_labels: list[str] = []
    for r in rows:
        label = r.get("label", "").strip()
        if not label:
            continue
        if r.get("is_total"):
            total_labels.append(label)
        elif not r.get("is_group_header"):
            data_labels.append(label)

    # Period range (first and last data row labels)
    if data_labels:
        first, last = data_labels[0], data_labels[-1]
        if first != last:
            parts.append(f"Period range: {first} — {last}")
        else:
            parts.append(f"Period: {first}")

    # Shape
    parts.append(f"{row_count} data rows | {col_count} columns")

    # Total row labels
    if total_labels:
        parts.append(f"Total rows: {', '.join(total_labels)}")

    # Entity samples (all labels up to cap, first+last halves if over cap)
    if data_labels:
        if len(data_labels) <= _MAX_ENTITY_LABELS:
            entity_text = ", ".join(data_labels)
        else:
            head = data_labels[:_ENTITY_HALF]
            tail = data_labels[-_ENTITY_HALF:]
            entity_text = ", ".join(head) + ", ..., " + ", ".join(tail)
        parts.append(f"Entities: {entity_text}")

    return "\n".join(parts)


def build_table_records(
    chunks: list[Chunk],
    parsed_tables_by_file: dict[str, Any],
) -> list[TableRecord]:
    """Build :class:`TableRecord` objects by matching Chunks to ParsedTable data.

    Runs AFTER the existing ``chunk_file()`` pipeline — it does NOT modify
    any Chunks.  The ``chunk_id`` in each ``TableRecord`` is the SAME as
    the matched ``Chunk.chunk_id``, creating the FK link to ``treasury_chunks``.

    For split tables (a large table split across multiple consecutive chunks),
    all split chunks get a ``TableRecord`` with the same ``table_json``, so the
    agent can look up ANY chunk_id and get the complete table.

    Parameters
    ----------
    chunks:
        All chunks produced by ``chunk_file()`` for one or more files.
    parsed_tables_by_file:
        Mapping from file stem (e.g., ``"treasury_bulletin_1941_01"``) to a
        list of ``ParsedTable`` objects (from ``parse_html_tables_structured``),
        in document order.

    Returns
    -------
    list[TableRecord]
    """
    from collections import defaultdict

    # Group chunks by file
    by_file: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        stem = Path(chunk.file_name).stem if chunk.file_name else ""
        if stem:
            by_file[stem].append(chunk)

    records: list[TableRecord] = []

    for stem, file_chunks in by_file.items():
        parsed_tables = list(parsed_tables_by_file.get(stem, []))
        if not parsed_tables:
            continue

        table_idx = 0  # index into parsed_tables for this file

        # Track consecutive table chunks for split-table handling
        i = 0
        while i < len(file_chunks):
            chunk = file_chunks[i]
            is_table_chunk = chunk.chunk_type == "table"
            is_section_with_table = (
                chunk.chunk_type == "section" and _chunk_contains_table(chunk)
            )

            if not (is_table_chunk or is_section_with_table):
                i += 1
                continue

            if table_idx >= len(parsed_tables):
                logger.warning(
                    "TABLE_RECORD_MISMATCH file=%s chunk_idx=%d table_idx=%d "
                    "no_more_parsed_tables=%d",
                    stem, i, table_idx, len(parsed_tables),
                )
                i += 1
                continue

            pt = parsed_tables[table_idx]
            table_idx += 1

            # Parse JSON to extract metadata
            try:
                table_data = _json.loads(pt.table_json) if pt.table_json else {}
            except (ValueError, TypeError):
                table_data = {}

            row_count = table_data.get("row_count", 0)
            col_count = len(table_data.get("headers", []))
            annotation = _extract_annotation(chunk.content)
            table_title = chunk.page_info or ""
            pre_table_title = _extract_pre_table_title(chunk.content)

            # Build rich, searchable summary (without chunk_id — added per-record)
            base_summary = _build_table_summary(
                table_data,
                file_name=chunk.file_name,
                bulletin_date=chunk.bulletin_date,
                page_info=chunk.page_info,
                table_title=table_title,
                annotation=annotation,
                pre_table_title=pre_table_title,
            )

            # Collect all chunk_ids for this table (handles split tables)
            table_chunk_ids = [chunk.chunk_id]

            # Look ahead for consecutive table chunks from the same heading
            # (split table detection: same page_info, sequential chunk_ids)
            if is_table_chunk:
                j = i + 1
                while j < len(file_chunks):
                    next_chunk = file_chunks[j]
                    if (
                        next_chunk.chunk_type == "table"
                        and next_chunk.page_info == chunk.page_info
                    ):
                        table_chunk_ids.append(next_chunk.chunk_id)
                        j += 1
                    else:
                        break
                i = j  # skip past all split chunks
            else:
                i += 1

            # Create one TableRecord per chunk_id (same JSON for all split parts).
            # Each record gets its own chunk_id: line in content so the agent
            # can extract it from list/grep results for follow-up calls.
            for cid in table_chunk_ids:
                records.append(TableRecord(
                    chunk_id=cid,
                    file_name=chunk.file_name,
                    bulletin_date=chunk.bulletin_date,
                    page_info=chunk.page_info,
                    table_title=table_title,
                    annotation=annotation,
                    content=f"chunk_id: {cid}\n{base_summary}",
                    table_json=pt.table_json,
                    row_count=row_count,
                    col_count=col_count,
                    chunk_type="table",
                    char_count=len(pt.table_json),
                ))

    logger.info(
        "TABLE_RECORDS_BUILT total=%d files=%d",
        len(records), len(by_file),
    )
    return records
