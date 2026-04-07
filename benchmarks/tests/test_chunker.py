"""Tests for table-aware Markdown chunker."""

from pathlib import Path

import pytest

from benchmarks.officeqa.chunker import (
    Chunk,
    ChunkConfig,
    TableRecord,
    _build_table_summary,
    _chunk_table,
    _chunk_text,
    _extract_annotation,
    _extract_date,
    _is_table_line,
    _merge_into_sections,
    _prepend_context,
    _split_into_blocks,
    build_table_records,
    chunk_file,
)
from benchmarks.officeqa.html_table_parser import ParsedTable


class TestTableLineDetection:
    def test_valid_table_line(self) -> None:
        assert _is_table_line("| A | B |")
        assert _is_table_line("| 1,234 | 5,678 |")
        assert _is_table_line("|---|---|")

    def test_not_table_line(self) -> None:
        assert not _is_table_line("Revenue | expenses comparison")
        assert not _is_table_line("Normal text")
        assert not _is_table_line("")

    def test_pipe_at_start_required(self) -> None:
        assert not _is_table_line("A | B | C")

    def test_whitespace_before_pipe(self) -> None:
        assert _is_table_line("  | A | B |")


class TestDateExtraction:
    def test_standard_filename(self) -> None:
        assert _extract_date("treasury_bulletin_2024_03.txt") == "2024-03"

    def test_no_match(self) -> None:
        assert _extract_date("random_file.txt") == ""

    def test_year_month_in_middle(self) -> None:
        assert _extract_date("tb_1945_12_data.txt") == "1945-12"


class TestBlockSplitting:
    def test_text_only(self) -> None:
        text = "Hello world.\n\nSecond paragraph."
        blocks = _split_into_blocks(text)
        assert len(blocks) == 1
        assert blocks[0][0] == "text"

    def test_table_detected(self) -> None:
        text = "Intro text.\n| A | B |\n|---|---|\n| 1 | 2 |"
        blocks = _split_into_blocks(text)
        assert any(b[0] == "table" for b in blocks)

    def test_heading_tracked(self) -> None:
        text = "# Budget Summary\n| A | B |\n|---|---|\n| 1 | 2 |"
        blocks = _split_into_blocks(text)
        table_block = [b for b in blocks if b[0] == "table"][0]
        assert table_block[2] == "Budget Summary"

    def test_html_fallback(self) -> None:
        text = "```html\n<table><tr><td>1</td></tr></table>\n```"
        blocks = _split_into_blocks(text)
        assert blocks[0][0] == "html_fallback"

    def test_blank_line_in_table_tolerated(self) -> None:
        text = "| A | B |\n|---|---|\n\n| 1 | 2 |"
        blocks = _split_into_blocks(text)
        table_blocks = [b for b in blocks if b[0] == "table"]
        # Should be a single table (blank line tolerated)
        assert len(table_blocks) == 1


class TestTableChunking:
    def test_small_table_kept_whole(self) -> None:
        text = "| A | B |\n|---|---|\n| 1 | 2 |"
        chunks = _chunk_table(text, "heading", ChunkConfig(table_max_chars=1000))
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "table"

    def test_large_table_split_preserves_header(self) -> None:
        header = "| Col A |\n|---|"
        rows = "\n".join(f"| row{i} value here |" for i in range(100))
        big_table = header + "\n" + rows
        chunks = _chunk_table(big_table, "heading", ChunkConfig(table_max_chars=200))
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.content.startswith("| Col A |")

    def test_heading_preserved(self) -> None:
        text = "| A | B |\n|---|---|\n| 1 | 2 |"
        chunks = _chunk_table(text, "Budget Table", ChunkConfig())
        assert chunks[0].page_info == "Budget Table"


class TestTextChunking:
    def test_single_paragraph(self) -> None:
        chunks = _chunk_text("Short text.", "", ChunkConfig())
        assert len(chunks) == 1

    def test_split_at_paragraph_boundary(self) -> None:
        # Each paragraph is ~30 chars, so chunk_max_chars=35 forces one-per-chunk
        text = "Paragraph one has some text.\n\nParagraph two has more text.\n\nParagraph three also here."
        chunks = _chunk_text(text, "", ChunkConfig(chunk_max_chars=35, chunk_overlap_chars=0))
        assert len(chunks) >= 2
        # Each chunk should be a single paragraph (no double-newlines within)
        for c in chunks:
            assert "\n\n" not in c.content.strip()

    def test_empty_returns_nothing(self) -> None:
        assert _chunk_text("", "", ChunkConfig()) == []
        assert _chunk_text("   \n\n   ", "", ChunkConfig()) == []


class TestSectionMerging:
    def test_text_table_footnote_merged(self) -> None:
        """text + table + short footnote → single section block."""
        blocks: list[tuple[str, str, str]] = [
            ("text", "Budget Expenditures\n(in millions of dollars)", "Budget"),
            ("table", "| Year | Total |\n|---|---|\n| 2023 | 100 |", "Budget"),
            ("text", "Source: Daily Treasury Statements.", "Budget"),
        ]
        merged = _merge_into_sections(blocks, ChunkConfig())
        assert len(merged) == 1
        assert merged[0][0] == "section"
        assert "Budget Expenditures" in merged[0][1]
        assert "| Year | Total |" in merged[0][1]
        assert "Source: Daily Treasury" in merged[0][1]

    def test_text_table_merged_without_footnote(self) -> None:
        """text + table (no trailing text) → single section."""
        blocks: list[tuple[str, str, str]] = [
            ("text", "Table title here", "Heading"),
            ("table", "| A | B |\n|---|---|\n| 1 | 2 |", "Heading"),
        ]
        merged = _merge_into_sections(blocks, ChunkConfig())
        assert len(merged) == 1
        assert merged[0][0] == "section"

    def test_oversized_section_not_merged(self) -> None:
        """When combined exceeds section_max_chars, blocks stay separate."""
        big_table = "| A | B |\n|---|---|\n" + "\n".join(
            f"| row{i} | val{i} |" for i in range(500)
        )
        blocks: list[tuple[str, str, str]] = [
            ("text", "Title", "H"),
            ("table", big_table, "H"),
        ]
        config = ChunkConfig(section_max_chars=100)
        merged = _merge_into_sections(blocks, config)
        # Should not merge — stays as text + table
        assert len(merged) == 2
        assert merged[0][0] == "text"
        assert merged[1][0] == "table"

    def test_large_trailing_text_not_consumed(self) -> None:
        """Trailing text > 600 chars is NOT consumed as footnote."""
        blocks: list[tuple[str, str, str]] = [
            ("text", "Title", "H"),
            ("table", "| A |\n|---|\n| 1 |", "H"),
            ("text", "x" * 700, "H"),  # Too large to be a footnote
        ]
        merged = _merge_into_sections(blocks, ChunkConfig())
        # text+table merged, but big trailing text is separate
        assert len(merged) == 2
        assert merged[0][0] == "section"
        assert merged[1][0] == "text"

    def test_standalone_table_passes_through(self) -> None:
        """Table not preceded by text stays as-is."""
        blocks: list[tuple[str, str, str]] = [
            ("table", "| A | B |\n|---|---|\n| 1 | 2 |", "H"),
        ]
        merged = _merge_into_sections(blocks, ChunkConfig())
        assert len(merged) == 1
        assert merged[0][0] == "table"

    def test_standalone_text_passes_through(self) -> None:
        blocks: list[tuple[str, str, str]] = [
            ("text", "Just some text.", ""),
        ]
        merged = _merge_into_sections(blocks, ChunkConfig())
        assert len(merged) == 1
        assert merged[0][0] == "text"


class TestPrependContext:
    def test_all_fields(self) -> None:
        result = _prepend_context("Table data", "bulletin_2023_09.txt", "2023-09", "Budget")
        assert result.startswith("Document: bulletin_2023_09.txt\n")
        assert "Bulletin date: 2023-09" in result
        assert "Section: Budget" in result
        assert result.endswith("\n\nTable data")

    def test_no_metadata(self) -> None:
        result = _prepend_context("Table data", "", "", "")
        assert result == "Table data"

    def test_partial_metadata(self) -> None:
        result = _prepend_context("content", "file.txt", "", "")
        assert "Document: file.txt" in result
        assert "Bulletin date" not in result


class TestChunkFile:
    def test_full_file(self, tmp_path: Path) -> None:
        content = """# Summary
Total receipts for fiscal year 2023.

| Category | Amount |
|---|---|
| Individual Income | 2,176 |
| Corporate Income | 420 |

Some notes about the data.
"""
        fp = tmp_path / "treasury_bulletin_2023_09.txt"
        fp.write_text(content)

        chunks = chunk_file(fp)
        assert len(chunks) > 0
        assert all(c.file_name == "treasury_bulletin_2023_09.txt" for c in chunks)
        assert all(c.bulletin_date == "2023-09" for c in chunks)
        assert all(c.chunk_id.startswith("treasury_bulletin_2023_09_c") for c in chunks)

    def test_empty_file(self, tmp_path: Path) -> None:
        fp = tmp_path / "empty.txt"
        fp.write_text("")
        assert chunk_file(fp) == []

    def test_table_only_file(self, tmp_path: Path) -> None:
        content = "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"
        fp = tmp_path / "treasury_bulletin_2020_01.txt"
        fp.write_text(content)
        chunks = chunk_file(fp)
        assert len(chunks) >= 1
        assert any(c.chunk_type == "table" for c in chunks)

    def test_section_merging_in_full_file(self, tmp_path: Path) -> None:
        """text + table + footnote produces a 'section' chunk with context prefix."""
        content = """# Budget Summary
Budget Expenditures Classified as General
(in millions of dollars)

| Year | Total | Defense |
|---|---|---|
| 1940 | 9,589 | 1,590 |
| 1941 | 13,980 | 6,301 |

Source: Daily Treasury Statements.
"""
        fp = tmp_path / "treasury_bulletin_1941_01.txt"
        fp.write_text(content)

        chunks = chunk_file(fp)
        section_chunks = [c for c in chunks if c.chunk_type == "section"]
        assert len(section_chunks) >= 1

        sc = section_chunks[0]
        # Context prefix present
        assert "Document: treasury_bulletin_1941_01.txt" in sc.content
        assert "Bulletin date: 1941-01" in sc.content
        # Original content preserved
        assert "Budget Expenditures" in sc.content
        assert "| 1940 | 9,589 | 1,590 |" in sc.content
        assert "Source: Daily Treasury Statements." in sc.content

    def test_contextual_prefix_on_all_chunks(self, tmp_path: Path) -> None:
        """Every chunk gets the document/date/section prefix."""
        content = "# Receipts\nSome text about receipts in fiscal year 2023."
        fp = tmp_path / "treasury_bulletin_2023_06.txt"
        fp.write_text(content)

        chunks = chunk_file(fp)
        assert len(chunks) >= 1
        for c in chunks:
            assert "Document: treasury_bulletin_2023_06.txt" in c.content
            assert "Bulletin date: 2023-06" in c.content


# ---------------------------------------------------------------------------
# TableRecord / build_table_records tests (v9+)
# ---------------------------------------------------------------------------


_SAMPLE_TABLE_JSON = (
    '{"headers": [{"name": "Category", "parent": null, "index": 0}, '
    '{"name": "Amount", "parent": null, "index": 1}], '
    '"rows": [{"label": "Revenue", "cells": {"Amount": "1,234"}, '
    '"is_group_header": false, "is_total": false}], '
    '"row_count": 1, "data_row_count": 1}'
)


class TestExtractAnnotation:
    def test_millions_of_dollars(self) -> None:
        text = "Table 1\n(In millions of dollars)\n| A | B |"
        assert _extract_annotation(text) == "(In millions of dollars)"

    def test_percent(self) -> None:
        assert _extract_annotation("(In percent per annum)") == "(In percent per annum)"

    def test_basis_points(self) -> None:
        assert _extract_annotation("(In basis points)") == "(In basis points)"

    def test_no_annotation(self) -> None:
        assert _extract_annotation("Just plain text") == ""

    def test_case_insensitive(self) -> None:
        assert _extract_annotation("(in thousands of units)") == "(in thousands of units)"


class TestBuildTableRecords:
    def test_basic_matching(self) -> None:
        """One file with one text chunk and one table chunk."""
        chunks = [
            Chunk(chunk_id="test_c0000", file_name="test.txt", bulletin_date="2024-01",
                  page_info="Summary", content="Some text", chunk_type="text"),
            Chunk(chunk_id="test_c0001", file_name="test.txt", bulletin_date="2024-01",
                  page_info="Summary", content="| A | B |\n|---|---|\n| 1 | 2 |",
                  chunk_type="table"),
        ]
        parsed = {"test": [ParsedTable(markdown="...", table_json=_SAMPLE_TABLE_JSON)]}

        records = build_table_records(chunks, parsed)
        assert len(records) == 1
        assert records[0].chunk_id == "test_c0001"
        assert records[0].file_name == "test.txt"
        assert records[0].table_json == _SAMPLE_TABLE_JSON

    def test_multiple_tables_in_file(self) -> None:
        chunks = [
            Chunk(chunk_id="f_c0000", file_name="f.txt", page_info="A",
                  content="| X |\n|---|\n| 1 |", chunk_type="table"),
            Chunk(chunk_id="f_c0001", file_name="f.txt", page_info="B",
                  content="| Y |\n|---|\n| 2 |", chunk_type="table"),
        ]
        pt1 = ParsedTable(markdown="...", table_json='{"headers":[], "rows":[], "row_count":0, "data_row_count":0}')
        pt2 = ParsedTable(markdown="...", table_json='{"headers":[{"name":"Y","parent":null,"index":0}], "rows":[], "row_count":0, "data_row_count":0}')
        parsed = {"f": [pt1, pt2]}

        records = build_table_records(chunks, parsed)
        assert len(records) == 2
        assert records[0].chunk_id == "f_c0000"
        assert records[1].chunk_id == "f_c0001"

    def test_split_table_all_chunks_get_same_json(self) -> None:
        """A table split into 3 consecutive chunks should produce 3 records with same JSON."""
        chunks = [
            Chunk(chunk_id="f_c0000", file_name="f.txt", page_info="Debt",
                  content="| A | B |\n|---|---|\n| 1 | 2 |", chunk_type="table"),
            Chunk(chunk_id="f_c0001", file_name="f.txt", page_info="Debt",
                  content="| A | B |\n|---|---|\n| 3 | 4 |", chunk_type="table"),
            Chunk(chunk_id="f_c0002", file_name="f.txt", page_info="Debt",
                  content="| A | B |\n|---|---|\n| 5 | 6 |", chunk_type="table"),
        ]
        parsed = {"f": [ParsedTable(markdown="...", table_json=_SAMPLE_TABLE_JSON)]}

        records = build_table_records(chunks, parsed)
        assert len(records) == 3
        assert records[0].chunk_id == "f_c0000"
        assert records[1].chunk_id == "f_c0001"
        assert records[2].chunk_id == "f_c0002"
        # All have the same JSON
        assert records[0].table_json == records[1].table_json == records[2].table_json

    def test_section_with_embedded_table(self) -> None:
        """Section chunks that contain a table should be matched."""
        chunks = [
            Chunk(chunk_id="f_c0000", file_name="f.txt", page_info="Summary",
                  content="Title text\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\nFootnote.",
                  chunk_type="section"),
        ]
        parsed = {"f": [ParsedTable(markdown="...", table_json=_SAMPLE_TABLE_JSON)]}

        records = build_table_records(chunks, parsed)
        assert len(records) == 1
        assert records[0].chunk_id == "f_c0000"

    def test_no_parsed_tables_for_file(self) -> None:
        """If no ParsedTables exist for a file, no records are created."""
        chunks = [
            Chunk(chunk_id="f_c0000", file_name="f.txt", page_info="X",
                  content="| A |\n|---|\n| 1 |", chunk_type="table"),
        ]
        records = build_table_records(chunks, {})
        assert len(records) == 0

    def test_mismatch_more_chunks_than_tables(self) -> None:
        """More table chunks than ParsedTables — extra chunks are skipped with warning."""
        chunks = [
            Chunk(chunk_id="f_c0000", file_name="f.txt", page_info="A",
                  content="| X |\n|---|\n| 1 |", chunk_type="table"),
            Chunk(chunk_id="f_c0001", file_name="f.txt", page_info="B",
                  content="| Y |\n|---|\n| 2 |", chunk_type="table"),
        ]
        parsed = {"f": [ParsedTable(markdown="...", table_json=_SAMPLE_TABLE_JSON)]}

        records = build_table_records(chunks, parsed)
        assert len(records) == 1  # only first matches

    def test_text_chunks_ignored(self) -> None:
        """Text-only chunks are skipped."""
        chunks = [
            Chunk(chunk_id="f_c0000", file_name="f.txt", content="Just text",
                  chunk_type="text"),
        ]
        parsed = {"f": [ParsedTable(markdown="...", table_json=_SAMPLE_TABLE_JSON)]}
        records = build_table_records(chunks, parsed)
        assert len(records) == 0

    def test_annotation_extracted_from_content(self) -> None:
        """Annotation is extracted from the chunk content."""
        chunks = [
            Chunk(chunk_id="f_c0000", file_name="f.txt", page_info="Budget",
                  content="(In millions of dollars)\n| A | B |\n|---|---|\n| 1 | 2 |",
                  chunk_type="table"),
        ]
        parsed = {"f": [ParsedTable(markdown="...", table_json=_SAMPLE_TABLE_JSON)]}
        records = build_table_records(chunks, parsed)
        assert records[0].annotation == "(In millions of dollars)"

    def test_chunk_id_in_content(self) -> None:
        """Each record's content starts with its own chunk_id."""
        chunks = [
            Chunk(chunk_id="f_c0000", file_name="f.txt", page_info="X",
                  content="| A | B |\n|---|---|\n| 1 | 2 |", chunk_type="table"),
        ]
        parsed = {"f": [ParsedTable(markdown="...", table_json=_SAMPLE_TABLE_JSON)]}
        records = build_table_records(chunks, parsed)
        assert records[0].content.startswith("chunk_id: f_c0000\n")

    def test_split_table_different_chunk_ids_in_content(self) -> None:
        """Split table records have DIFFERENT chunk_id lines in content."""
        chunks = [
            Chunk(chunk_id="f_c0000", file_name="f.txt", page_info="D",
                  content="| A | B |\n|---|---|\n| 1 | 2 |", chunk_type="table"),
            Chunk(chunk_id="f_c0001", file_name="f.txt", page_info="D",
                  content="| A | B |\n|---|---|\n| 3 | 4 |", chunk_type="table"),
        ]
        parsed = {"f": [ParsedTable(markdown="...", table_json=_SAMPLE_TABLE_JSON)]}
        records = build_table_records(chunks, parsed)
        assert records[0].content.startswith("chunk_id: f_c0000\n")
        assert records[1].content.startswith("chunk_id: f_c0001\n")
        # Same base summary (everything after chunk_id line)
        body0 = records[0].content.split("\n", 1)[1]
        body1 = records[1].content.split("\n", 1)[1]
        assert body0 == body1


# ---------------------------------------------------------------------------
# _build_table_summary tests
# ---------------------------------------------------------------------------

_RICH_TABLE_DATA: dict = {
    "headers": [
        {"name": "Period", "parent": "Budget receipts and expenditures", "index": 0},
        {"name": "Net receipts", "parent": "Budget receipts and expenditures", "index": 1},
        {"name": "Expenditures", "parent": None, "index": 2},
    ],
    "rows": [
        {"label": "Fiscal years:", "cells": {}, "is_group_header": True, "is_total": False},
        {"label": "1942", "cells": {"Net receipts": "12,696", "Expenditures": "34,187"}, "is_group_header": False, "is_total": False},
        {"label": "1943", "cells": {"Net receipts": "22,208", "Expenditures": "79,622"}, "is_group_header": False, "is_total": False},
        {"label": "Total", "cells": {"Net receipts": "34,904", "Expenditures": "113,809"}, "is_group_header": False, "is_total": True},
    ],
    "row_count": 4,
    "data_row_count": 3,
}


class TestBuildTableSummary:
    def test_contains_document_context(self) -> None:
        s = _build_table_summary(_RICH_TABLE_DATA, "bulletin_1942.txt", "1942-01", "Budget", "Fiscal Ops", "")
        assert "Document: bulletin_1942.txt" in s
        assert "Bulletin date: 1942-01" in s
        assert "Section: Budget" in s

    def test_contains_table_title(self) -> None:
        s = _build_table_summary(_RICH_TABLE_DATA, "", "", "", "Fiscal Operations", "")
        assert "TABLE: Fiscal Operations" in s

    def test_contains_annotation(self) -> None:
        s = _build_table_summary(_RICH_TABLE_DATA, "", "", "", "", "(In millions of dollars)")
        assert "(In millions of dollars)" in s

    def test_contains_header_parents(self) -> None:
        s = _build_table_summary(_RICH_TABLE_DATA, "", "", "", "", "")
        assert "Header context:" in s
        assert "Budget receipts and expenditures" in s

    def test_no_header_context_when_no_parents(self) -> None:
        data = {"headers": [{"name": "A", "parent": None, "index": 0}], "rows": [], "row_count": 0}
        s = _build_table_summary(data, "", "", "", "", "")
        assert "Header context:" not in s

    def test_contains_column_names(self) -> None:
        s = _build_table_summary(_RICH_TABLE_DATA, "", "", "", "", "")
        assert "Columns: Period | Net receipts | Expenditures" in s

    def test_contains_period_range(self) -> None:
        s = _build_table_summary(_RICH_TABLE_DATA, "", "", "", "", "")
        assert "Period range: 1942 — 1943" in s

    def test_single_row_period(self) -> None:
        data = {"headers": [{"name": "X", "parent": None, "index": 0}],
                "rows": [{"label": "2020", "cells": {}, "is_group_header": False, "is_total": False}],
                "row_count": 1}
        s = _build_table_summary(data, "", "", "", "", "")
        assert "Period: 2020" in s

    def test_contains_total_rows(self) -> None:
        s = _build_table_summary(_RICH_TABLE_DATA, "", "", "", "", "")
        assert "Total rows: Total" in s

    def test_contains_entity_labels(self) -> None:
        s = _build_table_summary(_RICH_TABLE_DATA, "", "", "", "", "")
        assert "Entities: 1942, 1943" in s

    def test_group_headers_excluded_from_entities(self) -> None:
        s = _build_table_summary(_RICH_TABLE_DATA, "", "", "", "", "")
        # "Fiscal years:" is a group header — should NOT appear in Entities
        assert "Fiscal years:" not in s.split("Entities:")[-1] if "Entities:" in s else True

    def test_total_rows_excluded_from_entities(self) -> None:
        s = _build_table_summary(_RICH_TABLE_DATA, "", "", "", "", "")
        entities_line = [l for l in s.split("\n") if l.startswith("Entities:")]
        if entities_line:
            assert "Total" not in entities_line[0].replace("Total rows:", "")

    def test_entity_cap_at_30(self) -> None:
        """Tables with >30 rows get first 15 + ... + last 15."""
        rows = [{"label": f"Row_{i:03d}", "cells": {}, "is_group_header": False, "is_total": False}
                for i in range(50)]
        data = {"headers": [{"name": "X", "parent": None, "index": 0}],
                "rows": rows, "row_count": 50}
        s = _build_table_summary(data, "", "", "", "", "")
        assert "Row_000" in s  # first
        assert "Row_049" in s  # last
        assert "..." in s      # truncation indicator
        assert "Row_025" not in s  # middle excluded

    def test_header_context_deduplicates_exact_matches_only(self) -> None:
        """Identical parent chains collapsed; different chains in column order."""
        data = {
            "headers": [
                {"name": "A", "parent": "Group X", "index": 0},
                {"name": "B", "parent": "Group Y", "index": 1},
                {"name": "C", "parent": "Group X", "index": 2},
            ],
            "rows": [], "row_count": 0,
        }
        s = _build_table_summary(data, "", "", "", "", "")
        ctx = [line for line in s.split("\n") if line.startswith("Header context:")][0]
        assert ctx == "Header context: Group X; Group Y"
