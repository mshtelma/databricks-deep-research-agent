"""Tests for structured-passage parsers (html / markdown / json)."""

from __future__ import annotations

import pytest

from databricks_deep_research.tools.builtins.text_table.error_codes import (
    ErrorCode,
    ToolErrorException,
)
from databricks_deep_research.tools.builtins.text_table.parsers import (
    StructuredPassage,
    get_parser,
)


def test_html_table_parsed_to_list_of_row_dicts() -> None:
    parse = get_parser("html")
    html = (
        "<table>"
        "<tr><th>a</th><th>b</th></tr>"
        "<tr><td>1</td><td>2</td></tr>"
        "<tr><td>3</td><td>4</td></tr>"
        "</table>"
    )
    result: StructuredPassage = parse(html)
    assert result["parser"] == "html"
    assert result["raw"] == html
    parsed = result["parsed"]
    assert isinstance(parsed, list)
    assert parsed == [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]


def test_html_non_table_returns_text_string() -> None:
    parse = get_parser("html")
    html = "<div><p>Hello <b>world</b></p></div>"
    result = parse(html)
    assert result["parser"] == "html"
    assert isinstance(result["parsed"], str)
    assert "Hello" in result["parsed"]
    assert "world" in result["parsed"]


def test_markdown_passes_through() -> None:
    parse = get_parser("markdown")
    md = "# Heading\n\nSome body text with $x = y$"
    result = parse(md)
    assert result["parser"] == "markdown"
    assert result["raw"] == md
    parsed = result["parsed"]
    # Either a sectioned dict or pass-through string is acceptable per spec.
    assert isinstance(parsed, (dict, str))


def test_json_valid_parsed_to_dict() -> None:
    parse = get_parser("json")
    s = '{"a": 1, "b": [2, 3]}'
    result = parse(s)
    assert result["parser"] == "json"
    assert result["raw"] == s
    assert result["parsed"] == {"a": 1, "b": [2, 3]}


def test_json_valid_array() -> None:
    parse = get_parser("json")
    s = '[{"a": 1}, {"a": 2}]'
    result = parse(s)
    assert result["parsed"] == [{"a": 1}, {"a": 2}]


def test_json_invalid_raises_tool_error_exception() -> None:
    parse = get_parser("json")
    with pytest.raises(ToolErrorException) as exc:
        parse("not json{{}")
    assert exc.value.error.error_code is ErrorCode.INVALID_BINDING


def test_unknown_parser_raises_value_error() -> None:
    with pytest.raises(ValueError):
        get_parser("xml")  # type: ignore[arg-type]
