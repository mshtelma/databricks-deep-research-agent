"""Tests for table-aware Markdown chunker."""

from pathlib import Path

import pytest

from benchmarks.officeqa.chunker import (
    Chunk,
    ChunkConfig,
    _chunk_table,
    _chunk_text,
    _extract_date,
    _is_table_line,
    _merge_into_sections,
    _prepend_context,
    _split_into_blocks,
    chunk_file,
)


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
