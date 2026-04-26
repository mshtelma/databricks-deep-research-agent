"""Unit tests for :func:`_is_suspicious_repair`.

The helper guards against a class of ``json_repair`` false successes
where substantive LLM content (thousands of chars of research notes)
gets compressed into a tiny structured output (a 59-char list, an
empty dict, …) and silently shipped downstream. These tests pin the
behavior so the sanity check doesn't drift.
"""

from __future__ import annotations

import logging

import pytest

from databricks_deep_research.agents.harness import (
    _JSON_REPAIR_MIN_SIZE_RATIO,
    _is_suspicious_repair,
    _parse_output,
    _UnparsedJSONOutput,
)
from databricks_deep_research.agents.config import AgentNodeConfig


# ---------------------------------------------------------------------------
# _is_suspicious_repair behavior
# ---------------------------------------------------------------------------


def test_healthy_parse_passes() -> None:
    content = '{"x": "a substantive value populated by the agent reasoning"}'
    parsed = {"x": "a substantive value populated by the agent reasoning"}
    assert _is_suspicious_repair(parsed, content, subtype="researcher") is None


def test_empty_scalar_from_non_empty_content_rejected() -> None:
    """Aggressive: any non-whitespace content that parses to "" or None
    is suspicious. Preserves pre-existing behavior so WorkflowError
    contracts don't regress for short-but-malformed payloads."""
    content = "```not-json"
    assert _is_suspicious_repair("", content, subtype="planner") == (
        "empty scalar from non-empty content"
    )
    assert _is_suspicious_repair(None, content, subtype="planner") == (
        "empty scalar from non-empty content"
    )


def test_empty_scalar_from_whitespace_only_content_passes() -> None:
    """If the content is empty/whitespace, an empty parse is consistent
    with it — not a ``json_repair`` failure."""
    assert _is_suspicious_repair("", "   ", subtype="researcher") is None
    assert _is_suspicious_repair(None, "", subtype="researcher") is None


def test_empty_dict_from_researcher_rejected() -> None:
    content = "x" * 500
    result = _is_suspicious_repair({}, content, subtype="researcher")
    assert result == "empty dict from substantive text"


def test_empty_list_from_researcher_rejected() -> None:
    content = "x" * 500
    result = _is_suspicious_repair([], content, subtype="researcher")
    assert result == "empty list from substantive text"


def test_empty_container_from_non_researcher_passes() -> None:
    content = "x" * 500
    assert _is_suspicious_repair({}, content, subtype="coordinator") is None
    assert _is_suspicious_repair([], content, subtype="synthesizer") is None


def test_tiny_list_from_long_text_rejected() -> None:
    """Prod reproducer: 20 000 chars of reasoning → 59-char list.
    Ratio 0.003, well below the 0.1 threshold."""
    content = "A" * 20_000
    parsed = [{"k": "v"}]   # str repr is ~20 chars
    reason = _is_suspicious_repair(parsed, content, subtype="researcher")
    assert reason is not None
    assert reason.startswith("size collapse:")
    assert "ratio=" in reason


def test_tiny_output_from_short_text_passes() -> None:
    """Short input shouldn't trigger the size-ratio check — 499 chars is
    below the 500-char floor."""
    content = "A" * 499
    parsed = [{"k": "v"}]
    assert _is_suspicious_repair(parsed, content, subtype="researcher") is None


def test_size_collapse_fires_regardless_of_subtype() -> None:
    """Size-collapse is a structural check: a 39-char scalar from 5 000
    chars of content is pathological even for non-researcher subtypes.

    This is intentional — if a coordinator or synthesizer emits an LLM
    output string that json_repair crushes from 5 000 chars to 39 chars,
    we prefer raw preservation over a false-success structured value.
    """
    content = "A" * 5_000
    parsed = "a reasonable-length single string field"
    reason = _is_suspicious_repair(parsed, content, subtype="coordinator")
    assert reason is not None
    assert reason.startswith("size collapse:")


def test_ratio_threshold_boundary() -> None:
    """Exactly at the threshold: ratio == 0.1 is still accepted (strict
    less-than). Just below rejects."""
    content = "x" * 1_000
    # parsed_len needs to be < 100 to trigger
    just_above = "y" * 100            # ratio 0.1, str(parsed) repr includes quotes -> len 102
    just_below = "y" * 95             # ratio ~0.097
    assert _is_suspicious_repair(just_above, content, subtype="researcher") is None
    reason = _is_suspicious_repair(just_below, content, subtype="researcher")
    assert reason is not None and reason.startswith("size collapse:")


def test_threshold_constant_sane() -> None:
    """Guard against accidental threshold drift."""
    assert 0.0 < _JSON_REPAIR_MIN_SIZE_RATIO <= 0.5


# ---------------------------------------------------------------------------
# _parse_output integration: suspicious repair → _UnparsedJSONOutput
# ---------------------------------------------------------------------------


def _researcher_config() -> AgentNodeConfig:
    return AgentNodeConfig(
        subtype="researcher",
        model_tier="analytical",
        output_key="t",
        output_format="json",
    )


def test_parse_output_size_collapse_falls_back_to_unparsed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="databricks_deep_research.agents.harness")

    # 20 000 chars of plausible-looking garbage that json_repair will
    # partially recover as a small structure.
    content = (
        "Here is a lot of reasoning text produced by the LLM. "
        * 500
        + "\n[{}]"
    )
    result = _parse_output(content, _researcher_config())
    assert isinstance(result, _UnparsedJSONOutput)
    assert len(result) == len(content)

    parse_failures = [r for r in caplog.records if "JSON_PARSE_FAILURE" in r.message]
    assert parse_failures, "Expected JSON_PARSE_FAILURE log line"
    assert "reason=" in parse_failures[0].message


def test_parse_output_healthy_json_returns_parsed() -> None:
    content = '{"competitor_analyses": [{"name": "Sample", "source_refs": ["1"]}]}'
    result = _parse_output(content, _researcher_config())
    assert isinstance(result, dict)
    assert "competitor_analyses" in result


def test_parse_output_kill_switch_restores_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting HARNESS_RESILIENT_JSON_REPAIR=false restores the narrower
    sanity check — a tiny-but-non-empty list from substantive text is
    accepted again."""
    monkeypatch.setenv("HARNESS_RESILIENT_JSON_REPAIR", "false")

    content = "A" * 20_000 + "\n[1]"
    result = _parse_output(content, _researcher_config())
    # With legacy behavior, json_repair's result is accepted even though
    # the size ratio is pathological.
    # Note: json_repair may surface different shapes; assert we did NOT
    # fall through to _UnparsedJSONOutput in the kill-switch path.
    assert not isinstance(result, _UnparsedJSONOutput)
