"""Tests for MemEx-first tool I/O offload helpers (spec §1.1).

All pure/mocked — no network, no LLM, no real compute kernel. A tiny fake sink
records injected variables into a dict so we can assert retrievability.
"""

from __future__ import annotations

import json
from typing import Any

from databricks_deep_research.agents.config import ToolOutputBudgetConfig
from databricks_deep_research.agents.tool_offload import (
    build_preview,
    coerce_to_object,
    describe_object,
    hard_clip,
    line_preserving_truncate,
    maybe_offload,
    should_offload,
    snap_to_line_boundary,
)


class FakeComputeSink:
    """Records injected variables into a dict (implements ComputeSink)."""

    def __init__(self) -> None:
        self.injected: dict[str, Any] = {}

    def inject_variable(self, name: str, value: Any) -> None:
        self.injected[name] = value


def _cfg(**overrides: Any) -> ToolOutputBudgetConfig:
    return ToolOutputBudgetConfig(**overrides)


# --------------------------------------------------------------------------
# coerce_to_object
# --------------------------------------------------------------------------


def test_coerce_json_dict_returns_dict() -> None:
    obj = coerce_to_object('{"a": 1, "b": [1, 2, 3]}')
    assert obj == {"a": 1, "b": [1, 2, 3]}
    assert isinstance(obj, dict)


def test_coerce_json_list_returns_list() -> None:
    obj = coerce_to_object("[1, 2, 3]")
    assert obj == [1, 2, 3]
    assert isinstance(obj, list)


def test_coerce_tabular_string_returns_string() -> None:
    text = "| col_a | col_b |\n| 1 | 2 |\n| 3 | 4 |"
    obj = coerce_to_object(text)
    assert obj == text
    assert isinstance(obj, str)


def test_coerce_plain_prose_returns_string() -> None:
    text = "This is a paragraph of prose with no structure at all."
    obj = coerce_to_object(text)
    assert obj == text
    assert isinstance(obj, str)


def test_coerce_never_raises_on_garbage() -> None:
    # Invalid JSON, single pipe (below the >=2 tabular threshold) => raw string.
    text = "not json { and only one | pipe here"
    assert coerce_to_object(text) == text


# --------------------------------------------------------------------------
# describe_object
# --------------------------------------------------------------------------


def test_describe_dict_lists_keys() -> None:
    desc = describe_object({"alpha": 1, "beta": 2})
    assert "dict keys=" in desc
    assert "alpha" in desc and "beta" in desc


def test_describe_list_reports_len_and_elem_type() -> None:
    desc = describe_object([1, 2, 3])
    assert "list len=3" in desc
    assert "int" in desc


def test_describe_str_reports_chars() -> None:
    desc = describe_object("hello world")
    assert desc == "str chars=11"


# --------------------------------------------------------------------------
# snap_to_line_boundary
# --------------------------------------------------------------------------


def test_snap_head_cuts_at_newline() -> None:
    text = "line one\nline two\nline three\n" + ("x" * 100)
    head = snap_to_line_boundary(text, 20)
    # Should cut at a newline boundary within the first 20 chars.
    assert head == "line one\nline two"


def test_snap_tail_cuts_at_newline() -> None:
    text = ("x" * 100) + "\nlast line one\nlast line two"
    tail = snap_to_line_boundary(text, 20, from_end=True)
    assert tail.endswith("last line two")
    assert "x" * 100 not in tail


def test_snap_returns_whole_text_when_shorter_than_length() -> None:
    assert snap_to_line_boundary("short", 100) == "short"


# --------------------------------------------------------------------------
# build_preview
# --------------------------------------------------------------------------


def test_build_preview_contains_marker_head_and_tail() -> None:
    head_part = "HEADLINE\n" + ("h" * 1990)
    middle = "\n" + ("m" * 30000) + "\n"
    tail_part = ("t" * 990) + "\nTAILLINE"
    content = head_part + middle + tail_part
    cfg = _cfg(preview_head_chars=2000, preview_tail_chars=1000)
    obj = coerce_to_object(content)
    preview = build_preview(content, "web_crawl_0", obj, cfg)

    # Marker references the handle, the full size, and the read-it-back hint.
    assert "`web_crawl_0`" in preview
    assert f"({len(content)} chars)" in preview
    assert "Use `compute` to read/operate on it." in preview
    # Head and tail content survive; the giant middle does not.
    assert "HEADLINE" in preview
    assert "TAILLINE" in preview
    assert "m" * 30000 not in preview
    # describe_object info is embedded in the marker (prose => str chars=N).
    assert f"str chars={len(content)}" in preview


def test_build_preview_marker_includes_dict_keys() -> None:
    payload = {f"key_{i}": i for i in range(3)}
    content = json.dumps(payload)
    cfg = _cfg()
    obj = coerce_to_object(content)
    preview = build_preview(content, "vector_search_2", obj, cfg)
    assert "dict keys=" in preview
    assert "key_0" in preview


# --------------------------------------------------------------------------
# should_offload
# --------------------------------------------------------------------------


def test_should_offload_under_threshold_false() -> None:
    cfg = _cfg(externalize_min_chars=100)
    assert should_offload("x" * 50, tool="web_crawl", mode="auto", cfg=cfg) is False


def test_should_offload_over_threshold_true() -> None:
    cfg = _cfg(externalize_min_chars=100)
    assert should_offload("x" * 200, tool="web_crawl", mode="auto", cfg=cfg) is True


def test_should_offload_mode_off_always_false() -> None:
    cfg = _cfg(externalize_min_chars=10)
    assert should_offload("x" * 9999, tool="web_crawl", mode="off", cfg=cfg) is False


def test_should_offload_exempt_tool_false() -> None:
    cfg = _cfg(externalize_min_chars=10)
    assert should_offload("x" * 9999, tool="read_file", mode="auto", cfg=cfg) is False


def test_should_offload_per_tool_override_threshold() -> None:
    cfg = _cfg(externalize_min_chars=100, tool_overrides={"web_crawl": 5000})
    content = "x" * 1000  # over global default, under the per-tool override
    assert should_offload(content, tool="web_crawl", mode="auto", cfg=cfg) is False
    # A different tool still uses the global threshold.
    assert should_offload(content, tool="vector_search", mode="auto", cfg=cfg) is True


# --------------------------------------------------------------------------
# maybe_offload
# --------------------------------------------------------------------------


def test_maybe_offload_under_threshold_passthrough() -> None:
    sink = FakeComputeSink()
    cfg = _cfg(externalize_min_chars=100)
    content = "small result"
    text, handle = maybe_offload(
        content, tool="web_crawl", idx=0, mode="auto", compute=sink, cfg=cfg
    )
    assert handle is None
    assert text == content
    assert sink.injected == {}


def test_maybe_offload_over_threshold_mints_handle_and_preview() -> None:
    sink = FakeComputeSink()
    cfg = _cfg(externalize_min_chars=100, preview_head_chars=50, preview_tail_chars=20)
    content = "HEAD\n" + ("x" * 5000) + "\nTAIL"
    text, handle = maybe_offload(
        content, tool="web_crawl", idx=7, mode="auto", compute=sink, cfg=cfg
    )
    assert handle == "web_crawl_7"
    assert text != content
    # Preview carries the marker, head, and tail.
    assert "`web_crawl_7`" in text
    assert "HEAD" in text
    assert "TAIL" in text
    assert "x" * 5000 not in text
    # The full object is retrievable from the sink by handle.
    assert handle in sink.injected
    assert sink.injected[handle] == content


def test_maybe_offload_mode_off_always_passthrough() -> None:
    sink = FakeComputeSink()
    cfg = _cfg(externalize_min_chars=10)
    content = "x" * 9999
    text, handle = maybe_offload(
        content, tool="web_crawl", idx=0, mode="off", compute=sink, cfg=cfg
    )
    assert handle is None
    assert text == content
    assert sink.injected == {}


def test_maybe_offload_no_compute_sink_passthrough() -> None:
    cfg = _cfg(externalize_min_chars=10)
    content = "x" * 9999
    text, handle = maybe_offload(
        content, tool="web_crawl", idx=0, mode="auto", compute=None, cfg=cfg
    )
    assert handle is None
    assert text == content


def test_maybe_offload_exempt_tool_passthrough() -> None:
    sink = FakeComputeSink()
    cfg = _cfg(externalize_min_chars=10)
    content = "x" * 9999
    text, handle = maybe_offload(
        content, tool="read_file", idx=0, mode="auto", compute=sink, cfg=cfg
    )
    assert handle is None
    assert text == content
    assert sink.injected == {}


def test_maybe_offload_handle_naming_uses_idx_and_is_unique() -> None:
    sink = FakeComputeSink()
    cfg = _cfg(externalize_min_chars=10)
    content = "y" * 100

    _t0, h0 = maybe_offload(
        content, tool="web_crawl", idx=0, mode="auto", compute=sink, cfg=cfg
    )
    _t1, h1 = maybe_offload(
        content, tool="web_crawl", idx=1, mode="auto", compute=sink, cfg=cfg
    )
    assert h0 == "web_crawl_0"
    assert h1 == "web_crawl_1"
    assert h0 != h1
    # Both objects are retrievable independently.
    assert set(sink.injected) == {"web_crawl_0", "web_crawl_1"}


def test_maybe_offload_coerces_json_object_into_sink() -> None:
    sink = FakeComputeSink()
    cfg = _cfg(externalize_min_chars=10)
    payload = {"rows": list(range(50)), "note": "z" * 200}
    content = json.dumps(payload)
    text, handle = maybe_offload(
        content, tool="vector_search", idx=3, mode="auto", compute=sink, cfg=cfg
    )
    assert handle == "vector_search_3"
    # The stored object is the parsed dict, not the raw string.
    assert sink.injected[handle] == payload
    assert isinstance(sink.injected[handle], dict)
    # Preview describes it as a dict with keys.
    assert "dict keys=" in text


def test_maybe_offload_per_tool_override_respected() -> None:
    sink = FakeComputeSink()
    cfg = _cfg(externalize_min_chars=10, tool_overrides={"web_crawl": 100000})
    content = "x" * 5000  # over global, under the per-tool override
    text, handle = maybe_offload(
        content, tool="web_crawl", idx=0, mode="auto", compute=sink, cfg=cfg
    )
    assert handle is None
    assert text == content
    assert sink.injected == {}


# --------------------------------------------------------------------------
# Budget-ladder rungs (spec §1.2)
#
# These goldens are frozen literals captured from the pre-1.2 in-module
# ``react_loop._summarize_tool_result`` / truncate-branch implementations.
# They pin byte-for-byte output (Codex F7: OfficeQA depends on ``mask``).
# Embedding them as literals — rather than re-deriving from the function under
# test — makes the regression net independent of the implementation.
# --------------------------------------------------------------------------


def test_line_preserving_truncate_table_golden() -> None:
    content = "| A | B |\n| 1 | 2 |\nSome narrative text.\n| 3 | 4 |"
    assert line_preserving_truncate(content, max_chars=500) == (
        "[Compacted from 50 chars — key data preserved:]\n"
        "| A | B |\n| 1 | 2 |\n| 3 | 4 |"
    )


def test_line_preserving_truncate_numeric_golden() -> None:
    content = "Description text without numbers\nValue: 2,602\nMore description"
    assert line_preserving_truncate(content, max_chars=500) == (
        "[Compacted from 62 chars — key data preserved:]\nValue: 2,602"
    )


def test_line_preserving_truncate_unit_header_golden() -> None:
    content = (
        "Table 1 — Summary of Budget Results\n"
        "[In millions of dollars]\n"
        "| Category | FY 1940 | FY 1941 |\n"
        "| National defense | 2,602 | 3,100 |\n"
        "Some narrative explanation here.\n"
    )
    assert line_preserving_truncate(content, max_chars=500) == (
        "[Compacted from 164 chars — key data preserved:]\n"
        "Table 1 — Summary of Budget Results\n"
        "[In millions of dollars]\n"
        "| Category | FY 1940 | FY 1941 |\n"
        "| National defense | 2,602 | 3,100 |"
    )


def test_line_preserving_truncate_empty_golden() -> None:
    assert line_preserving_truncate("   \n\n  ", max_chars=200) == (
        "[Prior results — 7 chars, no tabular data]"
    )


def test_line_preserving_truncate_structural_cap_golden() -> None:
    """Structural-only lines are capped at 120 chars (verbatim pre-1.2 output)."""
    content = (
        "Document: a really really really really really really really really "
        "really really really long title that exceeds one hundred and twenty "
        "characters total here\n| x | 9 |"
    )
    assert line_preserving_truncate(content, max_chars=500) == (
        "[Compacted from 167 chars — key data preserved:]\n"
        "Document: a really really really really really really really really "
        "really really really long title that exceeds one hun\n| x | 9 |"
    )


def test_line_preserving_truncate_respects_max_chars_golden() -> None:
    content = "\n".join(f"| row{i} | {i * 100} |" for i in range(100))
    assert line_preserving_truncate(content, max_chars=200) == (
        "[Compacted from 1677 chars — key data preserved:]\n"
        "| row0 | 0 |\n| row1 | 100 |\n| row2 | 200 |\n| row3 | 300 |\n"
        "| row4 | 400 |\n| row5 | 500 |\n| row6 | 600 |\n| row7 | 700 |\n"
        "| row8 | 800 |\n| row9 | 900 |\n| row10 | 1000 |\n| row11 | 1100 |\n"
        "| row12 | 1200 |\n| row13 | 1300 |\n...[additional data truncated]"
    )


def test_line_preserving_truncate_default_max_chars_is_800() -> None:
    """The default keyword (800) matches the pre-1.2 signature."""
    content = "| A | B |\n| 1 | 2 |"
    assert line_preserving_truncate(content) == line_preserving_truncate(
        content, max_chars=800
    )


def test_summarize_tool_result_delegates_to_line_preserving() -> None:
    """react_loop._summarize_tool_result must stay an alias (single source)."""
    from databricks_deep_research.agents.react_loop import _summarize_tool_result

    samples = [
        "| A | B |\n| 1 | 2 |\nNarrative.\n| 3 | 4 |",
        "Description without numbers\nValue: 9,001\nMore text",
        "   \n\n  ",
    ]
    for content in samples:
        for mc in (200, 500, 800):
            assert _summarize_tool_result(content, max_chars=mc) == (
                line_preserving_truncate(content, max_chars=mc)
            )


def test_hard_clip_golden() -> None:
    content = "x" * 6048
    assert hard_clip(content, 4000) == (
        "x" * 4000 + "\n...[truncated from 6048 chars]"
    )


def test_hard_clip_reports_original_length_not_clipped_length() -> None:
    content = "abcdefghij"  # 10 chars
    assert hard_clip(content, 4) == "abcd\n...[truncated from 10 chars]"
