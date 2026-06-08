"""Google-style docstring parsing for the ``@tool`` decorator."""

from __future__ import annotations

from databricks_deep_research.tools.api import tool
from databricks_deep_research.tools.introspect import parse_google_docstring


def test_summary_only() -> None:
    parsed = parse_google_docstring("Just a summary line.")
    assert parsed.summary == "Just a summary line."
    assert parsed.args == {}


def test_summary_with_args() -> None:
    doc = """Search the web.

    Args:
        query: The search query.
        top_k: Number of results.

    Returns:
        A list of results.
    """
    parsed = parse_google_docstring(doc)
    assert parsed.summary == "Search the web."
    assert parsed.args["query"] == "The search query."
    assert parsed.args["top_k"] == "Number of results."
    assert "list of results" in parsed.returns


def test_args_with_types_in_parens() -> None:
    doc = """X.

    Args:
        a (int): integer arg.
        b (str): string arg.
    """
    parsed = parse_google_docstring(doc)
    assert parsed.args["a"] == "integer arg."
    assert parsed.args["b"] == "string arg."


def test_multiline_arg_description() -> None:
    doc = """X.

    Args:
        long_arg: this is the first line
            and this continues the description.
    """
    parsed = parse_google_docstring(doc)
    assert "first line" in parsed.args["long_arg"]
    assert "continues" in parsed.args["long_arg"]


def test_unknown_section_falls_back_to_summary() -> None:
    doc = """The summary.

    Notes:
        random
    """
    parsed = parse_google_docstring(doc)
    assert parsed.summary == "The summary."


def test_empty_doc_returns_empty_parsed() -> None:
    parsed = parse_google_docstring("")
    assert parsed.summary == ""
    assert parsed.args == {}


def test_decorator_propagates_arg_descriptions() -> None:
    @tool
    def search(query: str, top_k: int = 5) -> str:
        """Search the web.

        Args:
            query: The search query.
            top_k: Number of results.
        """
        return query

    props = search.parameters_schema.get("properties", {})
    assert "search query" in props["query"].get("description", "")
    assert "Number of results" in props["top_k"].get("description", "")
