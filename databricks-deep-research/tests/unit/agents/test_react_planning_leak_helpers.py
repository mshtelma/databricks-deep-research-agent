"""Phase 3.2 tests — JSON-value planning-text detection.

The bare-string detector ``_looks_like_planning`` returns False on
``{...}``-shaped content; the new ``_planning_text_in_json_value`` reaches
into parsed JSON values so a researcher emitting
``{"findings": "Let me search..."}`` is caught and routed through the
existing planning-leak REPLACE handling instead of leaking planning text
into the synthesizer's observations pool.
"""
from __future__ import annotations

import json

from databricks_deep_research.agents.react_loop import (
    _looks_like_planning,
    _planning_text_in_json_value,
    _value_looks_like_planning,
)


def test_bare_string_planning_detected() -> None:
    assert _looks_like_planning("Let me search for the Q4 transcript")
    assert _looks_like_planning("I'll now crawl seekingalpha.com")
    assert _looks_like_planning("First, I need to identify the entities")


def test_bare_string_concrete_findings_not_flagged() -> None:
    findings = (
        "Revenue grew 32% YoY to $60.9B in fiscal 2025. Operating margin "
        "expanded 540bps to 65.2%, driven by data-center demand."
    )
    assert not _looks_like_planning(findings)


def test_bare_json_not_flagged_by_bare_detector() -> None:
    payload = json.dumps({"findings": "Let me search for the Q4 transcript"})
    assert not _looks_like_planning(payload)


def test_json_value_planning_in_findings_detected() -> None:
    payload = json.dumps({"findings": "Let me search for more sources"})
    assert _planning_text_in_json_value(payload)


def test_json_value_planning_in_observation_detected() -> None:
    payload = json.dumps(
        {"observation": "I'll now crawl the relevant pages", "key_points": []}
    )
    assert _planning_text_in_json_value(payload)


def test_json_value_planning_in_lane_suffixed_field_detected() -> None:
    payload = json.dumps(
        {
            "findings_lane_5": "Let me get the actual Q4 2025 earnings call transcript",
            "observation": "Let me get the actual...",
        }
    )
    assert _planning_text_in_json_value(payload)


def test_json_concrete_findings_not_flagged() -> None:
    payload = json.dumps(
        {
            "findings": (
                "Revenue grew 32% YoY to $60.9B. Operating margin expanded "
                "540bps. Data-center segment up 122% YoY."
            ),
            "key_points": ["Strong DC growth", "Margin expansion"],
            "sources_used": ["https://example.com"],
        }
    )
    assert not _planning_text_in_json_value(payload)


def test_non_json_returns_false() -> None:
    # Bare strings are the bare detector's job, not this one.
    assert not _planning_text_in_json_value("plain prose with no braces")
    assert not _planning_text_in_json_value("")


def test_malformed_json_returns_false() -> None:
    assert not _planning_text_in_json_value("{not real json")
    assert not _planning_text_in_json_value("{\"unterminated\": \"string")


def test_json_array_not_flagged() -> None:
    payload = json.dumps([{"x": "Let me search"}])
    # Top-level arrays don't fit the researcher contract; we only flag dicts.
    assert not _planning_text_in_json_value(payload)


def test_value_looks_like_planning_handles_non_strings() -> None:
    assert not _value_looks_like_planning(None)
    assert not _value_looks_like_planning(123)
    assert not _value_looks_like_planning(["Let me search"])
    assert not _value_looks_like_planning({"nested": "Let me search"})


def test_long_value_not_flagged_as_planning() -> None:
    long_value = "Let me explain " + ("data " * 100)  # Long, contains "Let me"
    # The heuristic only flags SHORT planning-prefixed text — long content
    # is presumed to be a real (if rambling) answer.
    assert not _value_looks_like_planning(long_value)
