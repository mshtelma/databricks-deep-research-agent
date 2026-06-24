"""Unit tests for the provider safety-termination detector (feature 2.3).

Verifies the provider -> normalized-reason strategy map: each provider's safety /
content-policy finish reason classifies as safety-terminated, while benign
reasons (normal stop, output-length truncation, tool/control flow) never do.
Pure — no LLM, no network.
"""

from __future__ import annotations

import pytest

from databricks_deep_research.agents.safety_termination_detectors import (
    classify_termination_reason,
    is_safety_terminated,
    safety_termination_reason,
)
from databricks_deep_research.llm.client import LLMResponse, ToolCall

# ---------------------------------------------------------------------------
# Safety reasons -> True (per provider, native + OpenAI-normalized).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reason",
    [
        "content_filter",      # OpenAI / gateway-normalized
        "refusal",             # Anthropic native
        "safety",              # Gemini native
        "recitation",          # Gemini native
        "blocklist",           # Gemini native
        "prohibited_content",  # Gemini native
        "spii",                # Gemini native
    ],
)
def test_provider_safety_reasons_classified_as_safety(reason: str) -> None:
    assert classify_termination_reason(reason) == "safety_terminated"


@pytest.mark.parametrize(
    "reason",
    [
        "CONTENT_FILTER",
        "Refusal",
        "SAFETY",
        "Recitation",
        "PROHIBITED_CONTENT",
        "  refusal  ",  # surrounding whitespace tolerated
    ],
)
def test_safety_reasons_are_case_and_whitespace_insensitive(reason: str) -> None:
    assert classify_termination_reason(reason) == "safety_terminated"


# ---------------------------------------------------------------------------
# Benign reasons -> None (normal stop + length truncation must NOT be safety).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reason",
    [
        "stop",          # OpenAI normal completion
        "end_turn",      # Anthropic normal completion
        "stop_sequence",  # Anthropic stop string
        "length",        # OpenAI output-length truncation
        "max_tokens",    # Anthropic / Gemini output-length truncation
        "MAX_TOKENS",    # Gemini uppercases — still benign
        "tool_calls",    # OpenAI tool request
        "tool_use",      # Anthropic tool request
        "function_call",  # OpenAI legacy
        "pause_turn",    # Anthropic resumable pause
        "other",         # Gemini catch-all — not a confirmed safety stop
    ],
)
def test_benign_reasons_not_classified_as_safety(reason: str) -> None:
    assert classify_termination_reason(reason) is None


def test_empty_and_none_reason_are_benign() -> None:
    assert classify_termination_reason("") is None
    assert classify_termination_reason(None) is None


def test_unknown_reason_fails_open_to_none() -> None:
    # Fail-open: an unrecognized reason is NOT treated as safety (we only
    # suppress on a confirmed signal).
    assert classify_termination_reason("some_future_reason") is None


# ---------------------------------------------------------------------------
# Response-level wrappers over a real LLMResponse.
# ---------------------------------------------------------------------------


def _response(finish_reason: str) -> LLMResponse:
    return LLMResponse(
        content="partial",
        tool_calls=[ToolCall(id="tc1", function_name="web_search", arguments="{}")],
        model="test-model",
        finish_reason=finish_reason,
    )


def test_is_safety_terminated_true_on_safety_response() -> None:
    assert is_safety_terminated(_response("content_filter")) is True
    assert safety_termination_reason(_response("refusal")) == "safety_terminated"


def test_is_safety_terminated_false_on_benign_response() -> None:
    assert is_safety_terminated(_response("stop")) is False
    assert is_safety_terminated(_response("max_tokens")) is False
    assert safety_termination_reason(_response("tool_calls")) is None


def test_default_llmresponse_finish_reason_is_benign() -> None:
    # The default finish_reason is "stop" (e.g. the streaming path builds
    # responses without a reason) — must never be treated as safety.
    assert is_safety_terminated(LLMResponse(content="x")) is False
