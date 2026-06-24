"""Multi-model conversation-shape compatibility (Layer F).

GPT-family endpoints (databricks-gpt-5-*) reject conversations ending with an
assistant role:

    BadRequestError: This model does not support assistant message prefill.
    The conversation must end with a user message.

Claude tolerates the assistant suffix (used for prefill prompting). The
framework runs both providers from the same harness, so the LLM client
auto-appends a no-op user turn when targeting GPT and the last role is
assistant. This test pins that behavior.
"""

from __future__ import annotations

from databricks_deep_research.llm.client import (
    _ensure_user_suffix_for_gpt,
    _is_gpt_endpoint,
)

# ---------------------------------------------------------------------------
# _is_gpt_endpoint heuristic
# ---------------------------------------------------------------------------


def test_is_gpt_endpoint_recognizes_databricks_gpt_5() -> None:
    assert _is_gpt_endpoint("databricks-gpt-5-4")
    assert _is_gpt_endpoint("databricks-gpt-5-mini")
    assert _is_gpt_endpoint("databricks-gpt-5-nano")


def test_is_gpt_endpoint_recognizes_openai_naming() -> None:
    assert _is_gpt_endpoint("gpt-4o")
    assert _is_gpt_endpoint("gpt-5-turbo")


def test_is_gpt_endpoint_rejects_claude() -> None:
    assert not _is_gpt_endpoint("databricks-claude-opus-4-6")
    assert not _is_gpt_endpoint("databricks-claude-sonnet-4-6")
    assert not _is_gpt_endpoint("databricks-claude-haiku-4-5")


def test_is_gpt_endpoint_rejects_gemini_and_llama() -> None:
    assert not _is_gpt_endpoint("databricks-gemini-3-pro")
    assert not _is_gpt_endpoint("databricks-gemini-3-flash")
    assert not _is_gpt_endpoint("databricks-meta-llama-3-1-70b-instruct")


def test_is_gpt_endpoint_handles_none_and_empty() -> None:
    assert not _is_gpt_endpoint("")
    assert not _is_gpt_endpoint(None)  # type: ignore[arg-type]


def test_is_gpt_endpoint_case_insensitive() -> None:
    assert _is_gpt_endpoint("Databricks-GPT-5")
    assert _is_gpt_endpoint("DATABRICKS-GPT-5-MINI")


# ---------------------------------------------------------------------------
# _ensure_user_suffix_for_gpt
# ---------------------------------------------------------------------------


def test_claude_messages_pass_through_unchanged() -> None:
    messages = [
        {"role": "system", "content": "you are…"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    out = _ensure_user_suffix_for_gpt(messages, "databricks-claude-opus-4-6")
    assert out == messages  # exact same shape preserved


def test_gpt_with_assistant_suffix_gets_user_appended() -> None:
    messages = [
        {"role": "system", "content": "you are…"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "partial reply"},
    ]
    out = _ensure_user_suffix_for_gpt(messages, "databricks-gpt-5-4")
    assert len(out) == 4
    assert out[-1] == {"role": "user", "content": "Continue."}
    # earlier messages preserved
    assert out[:3] == messages


def test_gpt_with_user_suffix_passes_through_unchanged() -> None:
    messages = [
        {"role": "system", "content": "you are…"},
        {"role": "user", "content": "hi"},
    ]
    out = _ensure_user_suffix_for_gpt(messages, "databricks-gpt-5-4")
    assert out == messages


def test_gpt_with_tool_suffix_passes_through_unchanged() -> None:
    """Tool-result messages are role='tool'; the loop should continue
    via another user/assistant turn naturally — no patch required."""
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "tool_calls": [{"id": "x"}]},
        {"role": "tool", "tool_call_id": "x", "content": "{}"},
    ]
    out = _ensure_user_suffix_for_gpt(messages, "databricks-gpt-5-4")
    assert out == messages


def test_empty_messages_pass_through_unchanged() -> None:
    out = _ensure_user_suffix_for_gpt([], "databricks-gpt-5-4")
    assert out == []


def test_helper_does_not_mutate_input() -> None:
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
    ]
    snapshot = [dict(m) for m in messages]
    _ensure_user_suffix_for_gpt(messages, "databricks-gpt-5-4")
    assert messages == snapshot, "Input list was mutated"


def test_appended_message_uses_continue_marker() -> None:
    """Pin the exact `Continue.` content so changes are intentional."""
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
    ]
    out = _ensure_user_suffix_for_gpt(messages, "databricks-gpt-5-4")
    assert out[-1]["content"] == "Continue."
