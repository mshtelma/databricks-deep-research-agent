"""Tests for text_utils.py — markdown table detection and table-aware chunking."""

from __future__ import annotations

from databricks_deep_research.tools.builtins.text_utils import (
    ChunkType,
    chunk_content,
    chunk_table,
    chunk_text,
    detect_markdown_tables,
    is_table_line,
    sanitize_text,
    split_into_blocks,
)

# ---------------------------------------------------------------------------
# sanitize_text
# ---------------------------------------------------------------------------


class TestSanitizeText:
    def test_removes_null_bytes(self) -> None:
        assert sanitize_text("hello\x00world") == "helloworld"

    def test_preserves_tabs_and_newlines(self) -> None:
        assert sanitize_text("a\tb\nc") == "a\tb\nc"

    def test_normalizes_crlf(self) -> None:
        assert sanitize_text("a\r\nb\rc") == "a\nb\nc"

    def test_removes_control_chars(self) -> None:
        assert sanitize_text("a\x01b\x7fc") == "abc"

    def test_empty_string(self) -> None:
        assert sanitize_text("") == ""


# ---------------------------------------------------------------------------
# is_table_line
# ---------------------------------------------------------------------------


class TestIsTableLine:
    def test_standard_pipe_table(self) -> None:
        assert is_table_line("| A | B |") is True

    def test_compact_pipe_table(self) -> None:
        assert is_table_line("|A|B|") is True

    def test_not_starting_with_pipe(self) -> None:
        assert is_table_line("echo a | grep b") is False

    def test_single_pipe(self) -> None:
        assert is_table_line("| only one") is False

    def test_plain_text(self) -> None:
        assert is_table_line("just text") is False

    def test_separator_line(self) -> None:
        assert is_table_line("| --- | --- |") is True

    def test_indented_pipe_table(self) -> None:
        assert is_table_line("  | A | B |") is True


# ---------------------------------------------------------------------------
# detect_markdown_tables
# ---------------------------------------------------------------------------


class TestDetectMarkdownTables:
    def test_table_in_prose(self) -> None:
        text = (
            "Some intro text.\n\n"
            "| Name | Value |\n"
            "|---|---|\n"
            "| GDP | 1.2T |\n"
            "| Pop | 330M |\n\n"
            "More text after."
        )
        tables = detect_markdown_tables(text)
        assert len(tables) == 1
        assert isinstance(tables[0].table_json, dict)
        assert tables[0].col_count == 2
        assert tables[0].table_json["rows"][0]["label"] == "GDP"

    def test_skip_fenced_code_blocks(self) -> None:
        text = (
            "Text before\n\n"
            "```\n"
            "| A | B |\n"
            "|---|---|\n"
            "| 1 | 2 |\n"
            "```\n\n"
            "Text after"
        )
        tables = detect_markdown_tables(text)
        assert len(tables) == 0

    def test_skip_tilde_fenced_code_blocks(self) -> None:
        text = (
            "~~~\n"
            "| A | B |\n"
            "|---|---|\n"
            "| 1 | 2 |\n"
            "~~~\n"
        )
        tables = detect_markdown_tables(text)
        assert len(tables) == 0

    def test_require_separator_row(self) -> None:
        text = "| A | B |\n| 1 | 2 |\n| 3 | 4 |"
        tables = detect_markdown_tables(text)
        assert len(tables) == 0, "Without separator row, should not detect table"

    def test_multiple_tables(self) -> None:
        text = (
            "| X | Y |\n|---|---|\n| 1 | 2 |\n\n"
            "Some text\n\n"
            "| P | Q |\n|---|---|\n| a | b |"
        )
        tables = detect_markdown_tables(text)
        assert len(tables) == 2

    def test_empty_text(self) -> None:
        assert detect_markdown_tables("") == []

    def test_no_tables(self) -> None:
        assert detect_markdown_tables("Just some plain text\nwith newlines.") == []

    def test_table_with_alignment_colons(self) -> None:
        text = "| Left | Center | Right |\n|:---|:---:|---:|\n| a | b | c |"
        tables = detect_markdown_tables(text)
        assert len(tables) == 1

    def test_table_at_start_of_text(self) -> None:
        text = "| A | B |\n|---|---|\n| 1 | 2 |"
        tables = detect_markdown_tables(text)
        assert len(tables) == 1

    def test_table_at_end_of_text(self) -> None:
        text = "Intro text\n\n| A | B |\n|---|---|\n| 1 | 2 |"
        tables = detect_markdown_tables(text)
        assert len(tables) == 1

    def test_table_json_schema(self) -> None:
        text = "| Name | Score |\n|---|---|\n| Alice | 95 |\n| Bob | 87 |"
        tables = detect_markdown_tables(text)
        assert len(tables) == 1
        tj = tables[0].table_json
        assert "headers" in tj
        assert "rows" in tj
        assert "row_count" in tj
        assert tj["row_count"] == 2
        assert tj["headers"][0]["name"] == "Name"


# ---------------------------------------------------------------------------
# split_into_blocks
# ---------------------------------------------------------------------------


class TestSplitIntoBlocks:
    def test_text_only(self) -> None:
        blocks = split_into_blocks("Hello\nworld")
        assert len(blocks) == 1
        assert blocks[0][1] == ChunkType.text

    def test_table_only(self) -> None:
        text = "| A | B |\n|---|---|\n| 1 | 2 |"
        blocks = split_into_blocks(text)
        assert any(t == ChunkType.table for _, t in blocks)

    def test_mixed_content(self) -> None:
        text = "Intro\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\nOutro"
        blocks = split_into_blocks(text)
        types = [t for _, t in blocks]
        assert ChunkType.text in types
        assert ChunkType.table in types

    def test_code_block_not_detected_as_table(self) -> None:
        text = "```\n| A | B |\n|---|---|\n| 1 | 2 |\n```"
        blocks = split_into_blocks(text)
        # Should be text (inside code block), not table
        assert all(t == ChunkType.text for _, t in blocks)


# ---------------------------------------------------------------------------
# chunk_table
# ---------------------------------------------------------------------------


class TestChunkTable:
    def test_small_table_single_chunk(self) -> None:
        md = "| A | B |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |"
        chunks = chunk_table(md)
        assert len(chunks) == 1
        assert chunks[0] == md

    def test_large_table_split_preserves_header(self) -> None:
        header = "| A | B |\n| --- | --- |"
        data = "\n".join(f"| row{i} | val{i} |" for i in range(200))
        md = f"{header}\n{data}"
        chunks = chunk_table(md, max_chars=500)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.startswith("| A | B |"), "Each chunk must have header"
            assert "| --- | --- |" in chunk, "Each chunk must have separator"

    def test_very_short_table(self) -> None:
        md = "| A |\n| --- |"
        chunks = chunk_table(md)
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------


class TestChunkText:
    def test_short_text_single_chunk(self) -> None:
        text = "Hello world"
        chunks = chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_split_at_paragraphs(self) -> None:
        text = "\n\n".join(f"Paragraph {i} " * 50 for i in range(20))
        chunks = chunk_text(text, max_chars=500)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 700  # some tolerance for overlap

    def test_overlap(self) -> None:
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunk_text(text, max_chars=30, overlap=20)
        assert len(chunks) >= 2


# ---------------------------------------------------------------------------
# chunk_content
# ---------------------------------------------------------------------------


class TestChunkContent:
    def test_preserves_table_boundaries(self) -> None:
        text = (
            "Intro paragraph.\n\n"
            "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n\n"
            "Closing paragraph."
        )
        chunks = chunk_content(text)
        table_chunks = [(c, t) for c, t in chunks if t == ChunkType.table]
        text_chunks = [(c, t) for c, t in chunks if t == ChunkType.text]
        assert len(table_chunks) >= 1
        assert "| A | B |" in table_chunks[0][0]
        assert len(text_chunks) >= 1

    def test_text_only_content(self) -> None:
        text = "Just some text\n\nWith paragraphs."
        chunks = chunk_content(text)
        assert all(t == ChunkType.text for _, t in chunks)

    def test_table_only_content(self) -> None:
        text = "| A | B |\n|---|---|\n| 1 | 2 |"
        chunks = chunk_content(text)
        assert any(t == ChunkType.table for _, t in chunks)

    def test_empty_content(self) -> None:
        chunks = chunk_content("")
        assert len(chunks) == 0
