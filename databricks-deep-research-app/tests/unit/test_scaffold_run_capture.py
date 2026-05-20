"""Unit tests for ``tests/complex/_scaffold_run_capture.emit_console_report``.

The helper is exercised end-to-end by the slow live scaffold-and-run test;
these unit tests pin its behaviour around shape drift, Pydantic state
values, and oversized reports so regressions are caught fast and locally.
"""
from __future__ import annotations

import io
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from tests.complex._scaffold_run_capture import (
    _format_source_entry,
    _format_state_value,
    emit_console_report,
)


# ---------------------------------------------------------------------------
# emit_console_report — happy path + edge cases
# ---------------------------------------------------------------------------


def test_emit_console_report_prints_report() -> None:
    """A non-empty report sits between banner lines."""
    buf = io.StringIO()
    emit_console_report(
        case_id="case_x",
        report="# Title\n\nBody paragraph.",
        sources=[],
        runtime_state=None,
        stream=buf,
    )
    out = buf.getvalue()
    assert "GENERATED REPORT  case=case_x  chars=" in out
    assert "# Title" in out
    assert "Body paragraph." in out
    # Banner lines present (===... at default width).
    assert "=" * 100 in out


def test_emit_console_report_handles_empty_report() -> None:
    """Empty report still emits banners and a placeholder."""
    buf = io.StringIO()
    emit_console_report(
        case_id="case_y",
        report="",
        sources=None,
        runtime_state=None,
        stream=buf,
    )
    out = buf.getvalue()
    assert "(empty report)" in out
    assert "SOURCES  count=0" in out


def test_emit_console_report_handles_none_report() -> None:
    """``report=None`` is treated as empty, never crashes."""
    buf = io.StringIO()
    emit_console_report(
        case_id="case_z",
        report=None,
        sources=None,
        runtime_state=None,
        stream=buf,
    )
    out = buf.getvalue()
    assert "(empty report)" in out


def test_emit_console_report_prints_source_entries() -> None:
    """Source list prints with index, title, url for each item shape."""
    buf = io.StringIO()
    sources = [
        {"title": "Dict source", "url": "https://example.com/a"},
        SimpleNamespace(title="Obj source", url="https://example.com/b"),
        "https://example.com/c",  # bare URL string
    ]
    emit_console_report(
        case_id="case_src",
        report="report",
        sources=sources,  # type: ignore[arg-type]
        runtime_state=None,
        stream=buf,
    )
    out = buf.getvalue()
    assert "[ 1] Dict source" in out
    assert "[ 2] Obj source" in out
    assert "https://example.com/a" in out
    assert "https://example.com/b" in out
    assert "https://example.com/c" in out


def test_emit_console_report_handles_pydantic_state_value() -> None:
    """A Pydantic BaseModel state value is dumped as indented JSON, not as repr."""

    class FakeReview(BaseModel):
        decision: str
        reasoning: str

    state = SimpleNamespace(
        values={"coverage_review": FakeReview(decision="adjust", reasoning="needs work")},
    )
    buf = io.StringIO()
    emit_console_report(
        case_id="case_pyd",
        report="r",
        sources=[],
        runtime_state=state,
        stream=buf,
    )
    out = buf.getvalue()
    assert "STATE  key=coverage_review" in out
    # JSON dump preserves keys verbatim
    assert '"decision"' in out
    assert '"adjust"' in out
    assert '"reasoning"' in out
    # The Pydantic repr ``FakeReview(decision='adjust' ...)`` MUST NOT appear
    assert "FakeReview(" not in out


def test_emit_console_report_handles_missing_runtime_state() -> None:
    """``runtime_state=None`` is silent — no STATE block, no exception."""
    buf = io.StringIO()
    emit_console_report(
        case_id="case_no_rs",
        report="r",
        sources=[],
        runtime_state=None,
        stream=buf,
    )
    out = buf.getvalue()
    assert "STATE  key=" not in out


def test_emit_console_report_handles_runtime_state_without_values() -> None:
    """``runtime_state`` lacking ``.values`` does not raise."""
    state = object()  # bare object — no attributes
    buf = io.StringIO()
    emit_console_report(
        case_id="case_bare",
        report="r",
        sources=[],
        runtime_state=state,
        stream=buf,
    )
    out = buf.getvalue()
    assert "STATE  key=" not in out


def test_emit_console_report_skips_empty_state_values() -> None:
    """Keys whose values are None/empty are not surfaced."""
    state = SimpleNamespace(values={"coverage_review": None, "directives": []})
    buf = io.StringIO()
    emit_console_report(
        case_id="case_empty_state",
        report="r",
        sources=[],
        runtime_state=state,
        stream=buf,
    )
    assert "STATE  key=coverage_review" not in buf.getvalue()
    assert "STATE  key=directives" not in buf.getvalue()


def test_emit_console_report_truncates_huge_reports() -> None:
    """Reports beyond ``line_cap`` get a truncation footer."""
    long_report = "\n".join(f"line {i}" for i in range(6000))
    buf = io.StringIO()
    emit_console_report(
        case_id="case_long",
        report=long_report,
        sources=[],
        runtime_state=None,
        line_cap=5000,
        stream=buf,
    )
    out = buf.getvalue()
    assert "truncated at 5000 lines of 6000" in out
    # Last expected printed line is line 4999 (zero-indexed); line 5000 itself
    # is replaced by the ellipsis + footer.
    assert "line 4999" in out
    assert "line 5500" not in out


def test_emit_console_report_extra_keys_filter() -> None:
    """Only keys listed in ``extra_keys`` are surfaced."""
    state = SimpleNamespace(values={"foo": "shown", "bar": "hidden"})
    buf = io.StringIO()
    emit_console_report(
        case_id="case_keys",
        report="r",
        sources=[],
        runtime_state=state,
        extra_keys=("foo",),
        stream=buf,
    )
    out = buf.getvalue()
    assert "STATE  key=foo" in out
    assert "shown" in out
    assert "STATE  key=bar" not in out
    assert "hidden" not in out


# ---------------------------------------------------------------------------
# _format_source_entry — explicit shape coverage
# ---------------------------------------------------------------------------


def test_format_source_entry_dict() -> None:
    assert "Foo" in _format_source_entry(1, {"title": "Foo", "url": "u"})
    assert "u" in _format_source_entry(1, {"title": "Foo", "url": "u"})


def test_format_source_entry_string() -> None:
    line = _format_source_entry(2, "https://example.com")
    assert "[ 2]" in line
    assert "https://example.com" in line


def test_format_source_entry_object() -> None:
    src = SimpleNamespace(title="ObjTitle", url="ObjUrl")
    line = _format_source_entry(3, src)
    assert "ObjTitle" in line
    assert "ObjUrl" in line


def test_format_source_entry_pydantic_model() -> None:
    class S(BaseModel):
        title: str
        url: str

    line = _format_source_entry(4, S(title="PydTitle", url="PydUrl"))
    assert "PydTitle" in line
    assert "PydUrl" in line


def test_format_source_entry_unknown_shape() -> None:
    """An object missing ``title``/``url`` still renders without raising."""
    line = _format_source_entry(5, object())
    assert "[ 5]" in line
    assert "(no title)" in line


def test_format_source_entry_truncates_long_title() -> None:
    long_title = "x" * 200
    line = _format_source_entry(6, {"title": long_title, "url": "u"})
    # Title trimmed to 90 chars (default) + ellipsis.
    assert "x" * 90 not in line  # the raw long title is not present
    assert "…" in line


# ---------------------------------------------------------------------------
# _format_state_value — shape coverage
# ---------------------------------------------------------------------------


def test_format_state_value_none() -> None:
    assert _format_state_value(None) == "(none)"


def test_format_state_value_string_passthrough() -> None:
    assert _format_state_value("hello") == "hello"


def test_format_state_value_dict_json() -> None:
    out = _format_state_value({"a": 1, "b": [2, 3]})
    assert '"a"' in out
    assert "1" in out


def test_format_state_value_pydantic_json() -> None:
    class M(BaseModel):
        x: int

    out = _format_state_value(M(x=42))
    assert '"x"' in out
    assert "42" in out


def test_format_state_value_fallback_str() -> None:
    class Weird:
        def __str__(self) -> str:
            return "weird-repr"

    assert _format_state_value(Weird()) == "weird-repr"


@pytest.mark.parametrize("empty", ["", [], {}])
def test_format_state_value_empty_collection_is_not_treated_as_none(empty: object) -> None:
    """``_format_state_value`` itself does not short-circuit empties; the
    caller (``emit_console_report``) does. We just check it renders without
    raising for these inputs."""
    out = _format_state_value(empty)
    assert isinstance(out, str)
