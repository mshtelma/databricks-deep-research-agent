"""Tests for HTML table parser — structured JSON output and shared header builder."""

from __future__ import annotations

import json

import pytest

from benchmarks.officeqa.html_table_parser import (
    ParsedTable,
    _build_typed_headers,
    grid_to_table_json,
    parse_html_tables,
    parse_html_tables_structured,
)


# ---------------------------------------------------------------------------
# _build_typed_headers tests
# ---------------------------------------------------------------------------


class TestBuildTypedHeaders:
    """Tests for the shared header builder used by both markdown and JSON paths."""

    def test_zero_header_rows_generates_positional(self) -> None:
        grid = [["A", "1"], ["B", "2"]]
        flat, typed, data_start = _build_typed_headers(grid, header_rows=0, num_cols=2)
        assert flat == ["Category", "col_2"]
        assert data_start == 0
        assert typed[0]["name"] == "Category"
        assert typed[0]["parent"] is None

    def test_single_header_row(self) -> None:
        grid = [["Period", "Amount"], ["1942", "12696"]]
        flat, typed, data_start = _build_typed_headers(grid, header_rows=1, num_cols=2)
        assert flat == ["Period", "Amount"]
        assert data_start == 1
        assert typed[1]["name"] == "Amount"
        assert typed[1]["parent"] is None

    def test_multi_row_headers_with_parent(self) -> None:
        # Simulates: <th colspan=2>Budget</th> → grid[0] = ["", "Budget", ""]
        grid = [
            ["", "Budget", ""],
            ["Period", "Receipts", "Outlays"],
            ["1942", "100", "200"],
        ]
        flat, typed, data_start = _build_typed_headers(grid, header_rows=2, num_cols=3)
        assert data_start == 2
        # Column 1: parent = "Budget" (direct)
        assert typed[1]["parent"] == "Budget"
        assert typed[1]["name"] == "Receipts"
        # Column 2: parent = "Budget" (colspan propagation)
        assert typed[2]["parent"] == "Budget"
        assert typed[2]["name"] == "Outlays"
        # Column 0: no parent
        assert typed[0]["parent"] is None

    def test_empty_first_header_gets_category(self) -> None:
        grid = [["", "Val1", "Val2"], ["row1", "a", "b"]]
        flat, typed, data_start = _build_typed_headers(grid, header_rows=1, num_cols=3)
        assert flat[0] == "Category"
        assert typed[0]["name"] == "Category"

    def test_duplicate_header_names_made_unique(self) -> None:
        grid = [["Name", "Name", "Name"], ["a", "b", "c"]]
        flat, _, _ = _build_typed_headers(grid, header_rows=1, num_cols=3)
        assert flat == ["Name", "Name_2", "Name_3"]

    def test_ocr_corrections_applied(self) -> None:
        grid = [["Receipte", "Expendltures"], ["100", "200"]]
        flat, typed, _ = _build_typed_headers(grid, header_rows=1, num_cols=2)
        assert flat[0] == "Receipts"
        assert flat[1] == "Expenditures"
        assert typed[0]["name"] == "Receipts"

    def test_parent_chain_preserves_consecutive_duplicates(self) -> None:
        """Data values in misparsed header rows must not be deduplicated."""
        grid = [
            ["Dept A", "Dept B"],  # header row 0
            ["73", "100"],          # header row 1 (misparsed data)
            ["73", "200"],          # header row 2 (same value!)
            ["118", "300"],         # header row 3 (misparsed data)
            ["Oct", "Nov"],         # leaf header row
            ["x", "y"],            # data row
        ]
        _, typed, _ = _build_typed_headers(grid, header_rows=5, num_cols=2)
        # Column 0 parent should have BOTH "73" values
        assert typed[0]["parent"] == "Dept A > 73 > 73 > 118"
        assert typed[1]["parent"] == "Dept B > 100 > 200 > 300"

    def test_leaf_names_made_unique_after_parent_extraction(self) -> None:
        """Columns with same leaf but different parents get uniqueness suffix."""
        grid = [
            ["", "FY1986", "", "FY1987", ""],
            ["Category", "Oct", "Nov", "Oct", "Nov"],
            ["Row1", "1", "2", "3", "4"],
        ]
        _, typed, _ = _build_typed_headers(grid, header_rows=2, num_cols=5)
        leaf_names = [t["name"] for t in typed]
        assert leaf_names.count("Oct") == 1
        assert "Oct_2" in leaf_names
        assert leaf_names.count("Nov") == 1
        assert "Nov_2" in leaf_names


# ---------------------------------------------------------------------------
# grid_to_table_json tests
# ---------------------------------------------------------------------------


class TestGridToTableJson:
    """Tests for structured JSON table conversion."""

    def test_basic_table(self) -> None:
        grid = [
            ["Category", "Amount"],
            ["Revenue", "1,234"],
            ["Expenses", "5,678"],
        ]
        result = json.loads(grid_to_table_json(grid, header_rows=1))
        assert result["row_count"] == 2
        assert result["data_row_count"] == 2
        assert result["rows"][0]["label"] == "Revenue"
        assert result["rows"][0]["cells"]["Amount"] == "1,234"

    def test_total_row_detection(self) -> None:
        grid = [
            ["Item", "Value"],
            ["A", "10"],
            ["Total", "10"],
        ]
        result = json.loads(grid_to_table_json(grid, header_rows=1))
        assert result["rows"][0]["is_total"] is False
        assert result["rows"][1]["is_total"] is True

    def test_grand_total_detection(self) -> None:
        grid = [
            ["Item", "Value"],
            ["A", "10"],
            ["Grand Total", "10"],
        ]
        result = json.loads(grid_to_table_json(grid, header_rows=1))
        assert result["rows"][1]["is_total"] is True

    def test_total_case_insensitive(self) -> None:
        grid = [["X", "Y"], ["TOTAL LIABILITIES", "100"]]
        result = json.loads(grid_to_table_json(grid, header_rows=1))
        assert result["rows"][0]["is_total"] is True

    def test_total_prefix_only(self) -> None:
        """'Total' must be at the start of the label, not in the middle."""
        grid = [["X", "Y"], ["Sub Total", "50"], ["Not a total item", "25"]]
        result = json.loads(grid_to_table_json(grid, header_rows=1))
        # "Sub Total" does NOT start with "total" — should not be detected
        # (the regex is ^(grand\s+)?total\b)
        assert result["rows"][1]["is_total"] is False

    def test_group_header_detection(self) -> None:
        grid = [
            ["Category", "Amount"],
            ["Fiscal years:", ""],
            ["1942", "12,696"],
        ]
        result = json.loads(grid_to_table_json(grid, header_rows=1))
        assert result["rows"][0]["is_group_header"] is True
        assert result["rows"][0]["label"] == "Fiscal years:"
        assert result["rows"][1]["is_group_header"] is False

    def test_group_header_all_cells_empty(self) -> None:
        """Group header has a label but ALL data cells are empty/whitespace."""
        grid = [["Cat", "A", "B"], ["Section:", " ", "  "], ["Data", "1", "2"]]
        result = json.loads(grid_to_table_json(grid, header_rows=1))
        assert result["rows"][0]["is_group_header"] is True

    def test_multi_row_header_parent_in_json(self) -> None:
        grid = [
            ["", "Budget", ""],
            ["Period", "Receipts", "Outlays"],
            ["1942", "100", "200"],
        ]
        result = json.loads(grid_to_table_json(grid, header_rows=2))
        assert result["headers"][1]["parent"] == "Budget"
        assert result["headers"][2]["parent"] == "Budget"
        assert result["headers"][0]["parent"] is None

    def test_empty_grid(self) -> None:
        assert grid_to_table_json([], header_rows=0) == "{}"

    def test_data_row_count_excludes_group_headers(self) -> None:
        grid = [
            ["Cat", "Val"],
            ["Section:", ""],
            ["A", "1"],
            ["B", "2"],
            ["Total", "3"],
        ]
        result = json.loads(grid_to_table_json(grid, header_rows=1))
        assert result["row_count"] == 4  # all rows
        assert result["data_row_count"] == 3  # excludes group header

    def test_duplicate_leaf_names_preserve_all_cell_data(self) -> None:
        """When columns share a leaf name, uniqueness suffixes prevent data loss."""
        grid = [
            ["", "FY1986", "FY1987"],
            ["Category", "Amount", "Amount"],
            ["Row1", "100", "200"],
        ]
        result = json.loads(grid_to_table_json(grid, header_rows=2))
        row = result["rows"][0]
        assert len(row["cells"]) == 2
        assert "100" in row["cells"].values()
        assert "200" in row["cells"].values()


# ---------------------------------------------------------------------------
# parse_html_tables_structured tests
# ---------------------------------------------------------------------------


class TestParseHtmlTablesStructured:
    """Tests for the structured parsing API."""

    def test_returns_parsed_table_objects(self) -> None:
        html = "<table><tr><th>A</th></tr><tr><td>1</td></tr></table>"
        result = parse_html_tables_structured(html)
        assert len(result) == 1
        assert isinstance(result[0], ParsedTable)
        assert isinstance(result[0].markdown, str)
        assert isinstance(result[0].table_json, str)

    def test_markdown_identical_to_legacy(self) -> None:
        html = """<table>
        <tr><th>Category</th><th>Amount</th></tr>
        <tr><td>Revenue</td><td>1,234</td></tr>
        </table>"""
        old = parse_html_tables(html)
        new = parse_html_tables_structured(html)
        assert old[0] == new[0].markdown

    def test_fallback_returns_parsed_table_with_empty_json(self) -> None:
        html = "<p>No tables here</p>"
        result = parse_html_tables_structured(html)
        assert len(result) == 1
        assert result[0].table_json == ""
        assert "```html" in result[0].markdown

    def test_multiple_tables(self) -> None:
        html = """
        <table><tr><th>A</th></tr><tr><td>1</td></tr></table>
        <table><tr><th>B</th></tr><tr><td>2</td></tr></table>
        """
        result = parse_html_tables_structured(html)
        assert len(result) == 2
        t1 = json.loads(result[0].table_json)
        t2 = json.loads(result[1].table_json)
        assert t1["headers"][0]["name"] == "A"
        assert t2["headers"][0]["name"] == "B"

    def test_json_has_correct_cell_values(self) -> None:
        html = """<table>
        <tr><th>Year</th><th>GDP</th></tr>
        <tr><td>2020</td><td>21,433</td></tr>
        <tr><td>2021</td><td>23,315</td></tr>
        </table>"""
        result = parse_html_tables_structured(html)
        table = json.loads(result[0].table_json)
        assert table["rows"][0]["cells"]["GDP"] == "21,433"
        assert table["rows"][1]["cells"]["GDP"] == "23,315"

    def test_truncated_html_handled(self) -> None:
        """Truncated HTML missing </table> should still produce a result."""
        html = "<table><tr><th>X</th></tr><tr><td>42</td></tr>"
        result = parse_html_tables_structured(html)
        assert len(result) == 1
        table = json.loads(result[0].table_json)
        assert table["rows"][0]["cells"] == {}  # single-col table, label only
        assert table["rows"][0]["label"] == "42"
