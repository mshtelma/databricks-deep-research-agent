"""Tests for table_api.py — Table class for structured table access."""

from __future__ import annotations

import math

import pytest

from databricks_deep_research.tools.builtins.text_table.table_api import Table, to_float


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _simple_table() -> Table:
    """A small table for basic tests."""
    return Table(
        {
            "headers": [
                {"name": "Year", "index": 0},
                {"name": "Total", "index": 1, "parent": "Budget"},
                {"name": "Defense", "index": 2},
            ],
            "rows": [
                {"label": "1933", "cells": {"Total": "680", "Defense": "100"}, "is_total": False, "is_group_header": False},
                {"label": "1934", "cells": {"Total": "531", "Defense": "90"}, "is_total": False, "is_group_header": False},
                {"label": "1935", "cells": {"Total": "689", "Defense": "110"}, "is_total": False, "is_group_header": False},
                {"label": "Grand Total", "cells": {"Total": "1,900", "Defense": "300"}, "is_total": True, "is_group_header": False},
            ],
            "row_count": 4,
            "data_row_count": 3,
        },
        chunk_id="test_c0001",
        file_name="test_file.txt",
        title="Test Table",
        annotation="(In millions of dollars)",
    )


def _ocr_table() -> Table:
    """Table with an OCR-corrupted column name."""
    return Table({
        "headers": [{"name": "Heavy Department", "index": 0}],
        "rows": [
            {"label": "FY1940", "cells": {"Heavy Department": "891"}, "is_total": False, "is_group_header": False},
        ],
    })


def _entities_as_columns_table() -> Table:
    """Table where entities are columns and rows are time periods."""
    return Table({
        "headers": [
            {"name": "Fiscal year or month", "index": 0},
            {"name": "The Judiciary", "index": 1},
            {"name": "Agriculture Department", "index": 2},
        ],
        "rows": [
            {"label": "1980", "cells": {"The Judiciary": "564", "Agriculture Department": "5,000"}, "is_total": False, "is_group_header": False},
            {"label": "1981", "cells": {"The Judiciary": "637", "Agriculture Department": "5,200"}, "is_total": False, "is_group_header": False},
            {"label": "1984-Jan.", "cells": {"The Judiciary": "84", "Agriculture Department": "400"}, "is_total": False, "is_group_header": False},
        ],
    })


# ---------------------------------------------------------------------------
# Cell access tests
# ---------------------------------------------------------------------------


class TestCellAccess:
    def test_exact_match(self) -> None:
        t = _simple_table()
        assert t.cell("1933", "Total") == "680"

    def test_exact_match_as_float(self) -> None:
        t = _simple_table()
        assert t.cell("1933", "Total", as_float=True) == 680.0

    def test_missing_row_raises_keyerror(self) -> None:
        t = _simple_table()
        with pytest.raises(KeyError, match="not found"):
            t.cell("1999", "Total")

    def test_missing_column_raises_keyerror(self) -> None:
        t = _simple_table()
        with pytest.raises(KeyError, match="not found"):
            t.cell("1933", "Nonexistent")

    def test_keyerror_shows_available_labels(self) -> None:
        t = _simple_table()
        with pytest.raises(KeyError, match="1933"):
            t.cell("1999", "Total")

    def test_fuzzy_column_match(self) -> None:
        t = _ocr_table()
        # "Navy Department" fuzzy-matches "Heavy Department"
        assert t.cell("FY1940", "Navy Department") == "891"

    def test_row_dict(self) -> None:
        t = _simple_table()
        rd = t.row_dict("1933")
        assert rd["Total"] == "680"
        assert rd["Defense"] == "100"

    def test_row_dict_as_float(self) -> None:
        t = _simple_table()
        rd = t.row_dict("1933", as_float=True)
        assert rd["Total"] == 680.0


# ---------------------------------------------------------------------------
# Series access tests
# ---------------------------------------------------------------------------


class TestSeriesAccess:
    def test_series_excludes_totals(self) -> None:
        t = _simple_table()
        s = t.series("Total")
        assert len(s) == 3  # excludes Grand Total
        assert s[0] == ("1933", "680")

    def test_series_includes_totals(self) -> None:
        t = _simple_table()
        s = t.series("Total", exclude_totals=False)
        assert len(s) == 4

    def test_series_as_float(self) -> None:
        t = _simple_table()
        s = t.series("Total", as_float=True)
        assert s[0] == ("1933", 680.0)
        assert s[2] == ("1935", 689.0)

    def test_column_values(self) -> None:
        t = _simple_table()
        vals = t.column_values("Total")
        assert vals == [680.0, 531.0, 689.0]

    def test_column_values_excludes_nan(self) -> None:
        t = Table({
            "headers": [{"name": "A"}],
            "rows": [
                {"label": "x", "cells": {"A": "42"}, "is_total": False},
                {"label": "y", "cells": {"A": "-"}, "is_total": False},
                {"label": "z", "cells": {"A": "10"}, "is_total": False},
            ],
        })
        assert t.column_values("A") == [42.0, 10.0]


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------


class TestSearchHelpers:
    def test_find_rows(self) -> None:
        t = _simple_table()
        assert t.find_rows("total") == ["Grand Total"]
        assert t.find_rows("193") == ["1933", "1934", "1935"]

    def test_find_columns(self) -> None:
        t = _simple_table()
        assert t.find_columns("tot") == ["Total"]
        assert t.find_columns("def") == ["Defense"]

    def test_has_label(self) -> None:
        t = _simple_table()
        assert t.has_label("1933") is True
        assert t.has_label("1999") is False


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_columns(self) -> None:
        t = _simple_table()
        assert t.columns == ["Year", "Total", "Defense"]

    def test_parents(self) -> None:
        t = _simple_table()
        assert t.parents["Total"] == "Budget"
        assert t.parents.get("Defense", "") == ""

    def test_labels(self) -> None:
        t = _simple_table()
        assert "Grand Total" in t.labels

    def test_entity_labels(self) -> None:
        t = _simple_table()
        assert t.entity_labels == ["1933", "1934", "1935"]

    def test_total_labels(self) -> None:
        t = _simple_table()
        assert t.total_labels == ["Grand Total"]

    def test_row_count(self) -> None:
        t = _simple_table()
        assert t.row_count == 4

    def test_data_row_count(self) -> None:
        t = _simple_table()
        assert t.data_row_count == 3

    def test_metadata(self) -> None:
        t = _simple_table()
        assert t.chunk_id == "test_c0001"
        assert t.file_name == "test_file.txt"
        assert t.title == "Test Table"
        assert t.annotation == "(In millions of dollars)"


# ---------------------------------------------------------------------------
# Dict backward compatibility
# ---------------------------------------------------------------------------


class TestDictCompat:
    def test_getitem(self) -> None:
        t = _simple_table()
        assert t["row_count"] == 4
        assert len(t["headers"]) == 3

    def test_get(self) -> None:
        t = _simple_table()
        assert t.get("row_count") == 4
        assert t.get("missing", 42) == 42

    def test_contains(self) -> None:
        t = _simple_table()
        assert "headers" in t
        assert "nonexistent" not in t

    def test_iterate_rows(self) -> None:
        """Existing v93 pattern: for r in table['rows']."""
        t = _simple_table()
        labels = [r["label"] for r in t["rows"]]
        assert "1933" in labels

    def test_header_parent_access(self) -> None:
        """Existing v93 pattern: table['headers'][i]['parent']."""
        t = _simple_table()
        assert t["headers"][1]["parent"] == "Budget"


# ---------------------------------------------------------------------------
# to_float edge cases
# ---------------------------------------------------------------------------


class TestToFloat:
    @pytest.mark.parametrize("raw,expected", [
        ("42", 42.0),
        ("1,234", 1234.0),
        ("1,234,567", 1234567.0),
        ("-3.14", -3.14),
        ("(500)", -500.0),
        ("(1,234)", -1234.0),
        ("123 1/", 123.0),
        ("456 2/", 456.0),
    ])
    def test_valid_numbers(self, raw: str, expected: float) -> None:
        assert to_float(raw) == expected

    @pytest.mark.parametrize("raw", [
        "", "-", "--", "\u2014", "*", "(*)", "n.a.", "N/A", "(X)", "...",
    ])
    def test_missing_sentinels(self, raw: str) -> None:
        assert math.isnan(to_float(raw))

    def test_none_input(self) -> None:
        assert math.isnan(to_float(None))  # type: ignore[arg-type]

    def test_non_string_input(self) -> None:
        assert math.isnan(to_float(42))  # type: ignore[arg-type]

    def test_whitespace(self) -> None:
        assert to_float("  680  ") == 680.0

    def test_unparseable(self) -> None:
        assert math.isnan(to_float("hello"))


# ---------------------------------------------------------------------------
# Structural helpers
# ---------------------------------------------------------------------------


class TestStructuralHelpers:
    def test_column_parent(self) -> None:
        t = _simple_table()
        assert t.column_parent("Total") == "Budget"

    def test_describe(self) -> None:
        t = _simple_table()
        desc = t.describe()
        assert "Test Table" in desc
        assert "4R x 3C" in desc
        assert "Grand Total" in desc

    def test_repr(self) -> None:
        t = _simple_table()
        assert "test_c0001" in repr(t)
        assert "4R" in repr(t)


# ---------------------------------------------------------------------------
# Entities-as-columns table
# ---------------------------------------------------------------------------


class TestEntitiesAsColumns:
    def test_series_from_column(self) -> None:
        t = _entities_as_columns_table()
        s = t.series("The Judiciary", as_float=True)
        assert len(s) == 3
        assert s[0] == ("1980", 564.0)

    def test_find_columns_judiciary(self) -> None:
        t = _entities_as_columns_table()
        assert t.find_columns("judiciary") == ["The Judiciary"]
