"""Tests for bare ``{column: value}`` where-filter coercion.

Regression for the production crash where the LLM passed
``{"document_source": "treasury_bulletin_1945_10.txt"}`` and FlatTableFilter
rejected it with ``extra_forbidden``.
"""

from __future__ import annotations

from databricks_deep_research.tools.builtins.text_table.filter_dsl import (
    FlatTableFilter,
    coerce_flat_filter_shape,
)
from databricks_deep_research.tools.builtins.text_table.tools.read import (
    _parse_filter as read_parse_filter,
)
from databricks_deep_research.tools.builtins.text_table.tools.search import (
    _parse_filter as search_parse_filter,
)


def test_bare_column_mapping_coerced_to_eq() -> None:
    assert coerce_flat_filter_shape({"document_source": "x.txt"}) == {
        "eq": {"document_source": "x.txt"}
    }


def test_multiple_bare_columns_all_under_eq() -> None:
    assert coerce_flat_filter_shape({"a": 1, "b": 2}) == {"eq": {"a": 1, "b": 2}}


def test_existing_dsl_passthrough_unchanged() -> None:
    dsl = {"eq": {"a": 1}}
    assert coerce_flat_filter_shape(dsl) is dsl


def test_composite_passthrough_unchanged() -> None:
    for key in ("and", "or", "not"):
        raw = {key: []}
        assert coerce_flat_filter_shape(raw) is raw


def test_empty_dict_unchanged() -> None:
    assert coerce_flat_filter_shape({}) == {}


def test_read_parse_filter_accepts_bare_column() -> None:
    parsed = read_parse_filter({"document_source": "treasury_bulletin_1945_10.txt"})
    assert isinstance(parsed, FlatTableFilter)
    assert parsed.eq == {"document_source": "treasury_bulletin_1945_10.txt"}


def test_search_parse_filter_accepts_bare_column() -> None:
    parsed = search_parse_filter({"document_source": "treasury_bulletin_1945_10.txt"})
    assert isinstance(parsed, FlatTableFilter)
    assert parsed.eq == {"document_source": "treasury_bulletin_1945_10.txt"}


def test_parse_filter_still_accepts_dsl() -> None:
    parsed = read_parse_filter({"eq": {"year": 1945}})
    assert isinstance(parsed, FlatTableFilter)
    assert parsed.eq == {"year": 1945}
