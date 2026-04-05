"""Tests for html_tables.py — HTML table extraction."""

from __future__ import annotations

from databricks_deep_research.tools.builtins.html_tables import (
    ParsedTable,
    extract_tables_from_html,
    truncate_markdown_table,
)

# ---------------------------------------------------------------------------
# extract_tables_from_html
# ---------------------------------------------------------------------------


class TestExtractTablesFromHtml:
    """Tests for extract_tables_from_html()."""

    def test_simple_2x2_table(self) -> None:
        html = (
            "<table>"
            "<tr><th>Name</th><th>Value</th></tr>"
            "<tr><td>GDP</td><td>1.2T</td></tr>"
            "</table>"
        )
        tables = extract_tables_from_html(html)
        assert len(tables) == 1
        t = tables[0]
        assert isinstance(t, ParsedTable)
        assert isinstance(t.table_json, dict), "table_json must be dict, not str"
        assert "| Name | Value |" in t.markdown
        assert "| GDP | 1.2T |" in t.markdown
        assert t.row_count == 2
        assert t.col_count == 2
        assert t.table_json["headers"][0]["name"] == "Name"
        assert t.table_json["headers"][1]["name"] == "Value"
        assert t.table_json["rows"][0]["label"] == "GDP"
        assert t.table_json["rows"][0]["cells"]["Value"] == "1.2T"

    def test_table_json_is_dict_not_str(self) -> None:
        html = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
        tables = extract_tables_from_html(html)
        assert len(tables) == 1
        assert isinstance(tables[0].table_json, dict)
        assert "headers" in tables[0].table_json
        assert "rows" in tables[0].table_json

    def test_rowspan(self) -> None:
        html = (
            "<table>"
            "<tr><th>Cat</th><th>Val</th></tr>"
            "<tr><td rowspan='2'>X</td><td>1</td></tr>"
            "<tr><td>2</td></tr>"
            "</table>"
        )
        tables = extract_tables_from_html(html)
        assert len(tables) == 1
        assert tables[0].row_count == 3
        # First data row should have "X"
        assert "| X | 1 |" in tables[0].markdown

    def test_colspan(self) -> None:
        html = (
            "<table>"
            "<tr><th colspan='2'>Header</th></tr>"
            "<tr><th>A</th><th>B</th></tr>"
            "<tr><td>1</td><td>2</td></tr>"
            "</table>"
        )
        tables = extract_tables_from_html(html)
        assert len(tables) == 1
        assert tables[0].col_count == 2

    def test_empty_table_skipped(self) -> None:
        assert extract_tables_from_html("<table></table>") == []

    def test_header_only_table_skipped(self) -> None:
        html = "<table><tr><th>A</th><th>B</th></tr></table>"
        assert extract_tables_from_html(html) == []

    def test_single_column_table_skipped(self) -> None:
        html = "<table><tr><th>A</th></tr><tr><td>1</td></tr></table>"
        assert extract_tables_from_html(html) == []

    def test_no_tables_in_html(self) -> None:
        assert extract_tables_from_html("<p>Hello world</p>") == []

    def test_empty_string(self) -> None:
        assert extract_tables_from_html("") == []

    def test_malformed_html_missing_close_tag(self) -> None:
        html = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr>"
        tables = extract_tables_from_html(html)
        assert len(tables) == 1, "Should flush in-progress table"

    def test_nested_tables_outer_wins(self) -> None:
        html = (
            "<table>"
            "<tr><th>X</th><th>Y</th></tr>"
            "<tr><td>a</td><td>"
            "<table><tr><td>inner</td></tr></table>"
            "</td></tr>"
            "</table>"
        )
        tables = extract_tables_from_html(html)
        assert len(tables) == 1
        # Inner table text becomes part of cell content
        assert "inner" in tables[0].markdown

    def test_multiple_tables(self) -> None:
        html = (
            "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
            "<p>text between</p>"
            "<table><tr><th>X</th><th>Y</th></tr><tr><td>3</td><td>4</td></tr></table>"
        )
        tables = extract_tables_from_html(html)
        assert len(tables) == 2

    def test_entity_references_decoded(self) -> None:
        html = "<table><tr><th>A</th><th>B</th></tr><tr><td>a&amp;b</td><td>1&lt;2</td></tr></table>"
        tables = extract_tables_from_html(html)
        assert len(tables) == 1
        assert "a&b" in tables[0].markdown
        assert "1<2" in tables[0].markdown

    def test_pipe_characters_escaped_in_markdown(self) -> None:
        html = "<table><tr><th>A</th><th>B</th></tr><tr><td>x|y</td><td>z</td></tr></table>"
        tables = extract_tables_from_html(html)
        assert len(tables) == 1
        assert "x\\|y" in tables[0].markdown

    def test_large_table_extracted_fully(self) -> None:
        rows = "".join(f"<tr><td>r{i}</td><td>v{i}</td></tr>" for i in range(100))
        html = f"<table><tr><th>Key</th><th>Val</th></tr>{rows}</table>"
        tables = extract_tables_from_html(html)
        assert len(tables) == 1
        assert tables[0].row_count == 101  # 1 header + 100 data


# ---------------------------------------------------------------------------
# truncate_markdown_table
# ---------------------------------------------------------------------------


class TestTruncateMarkdownTable:
    """Tests for truncate_markdown_table()."""

    def test_small_table_unchanged(self) -> None:
        md = "| A | B |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |"
        assert truncate_markdown_table(md, max_rows=10) == md

    def test_large_table_truncated(self) -> None:
        header = "| A | B |\n| --- | --- |"
        data = "\n".join(f"| {i} | {i} |" for i in range(50))
        md = f"{header}\n{data}"
        result = truncate_markdown_table(md, max_rows=5)
        assert "... 45 more rows" in result
        # Should have header + separator + 5 data rows + indicator
        lines = result.split("\n")
        assert len(lines) == 8  # 2 header + 5 data + 1 indicator

    def test_header_only(self) -> None:
        md = "| A | B |\n| --- | --- |"
        assert truncate_markdown_table(md) == md

    def test_exact_max_rows(self) -> None:
        header = "| A | B |\n| --- | --- |"
        data = "\n".join(f"| {i} | {i} |" for i in range(5))
        md = f"{header}\n{data}"
        assert truncate_markdown_table(md, max_rows=5) == md
