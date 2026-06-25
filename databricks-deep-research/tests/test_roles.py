"""Unit tests for the OpenAI chat-role normalizer (llm/roles.py).

Regression guard for the AIS "Invalid role in the chat message" 400: a
conversation-history message stored with a non-OpenAI role (e.g. an app's
``"agent"``) must be coerced to ``"assistant"`` before reaching the gateway.
"""

from __future__ import annotations

from databricks_deep_research.llm.roles import (
    OPENAI_CHAT_ROLES,
    normalize_history_roles,
    sanitize_history_messages,
)


def test_agent_role_mapped_to_assistant() -> None:
    out = normalize_history_roles([{"role": "agent", "content": "report"}])
    assert out == [{"role": "assistant", "content": "report"}]


def test_valid_roles_pass_through_unchanged() -> None:
    history = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
        {"role": "system", "content": "s"},
        {"role": "tool", "content": "t", "tool_call_id": "c1"},
        {"role": "developer", "content": "d"},
        {"role": "function", "content": "f", "name": "fn"},
    ]
    assert normalize_history_roles(history) == history
    # Every emitted role is gateway-valid.
    assert all(m["role"] in OPENAI_CHAT_ROLES for m in normalize_history_roles(history))


def test_unknown_empty_and_missing_roles_become_assistant() -> None:
    out = normalize_history_roles(
        [
            {"role": "model", "content": "x"},
            {"role": "", "content": "y"},
            {"content": "z"},  # role missing entirely
        ]
    )
    assert [m["role"] for m in out] == ["assistant", "assistant", "assistant"]


def test_extra_keys_preserved_when_role_rewritten() -> None:
    out = normalize_history_roles(
        [{"role": "agent", "content": "c", "name": "n", "tool_calls": []}]
    )
    assert out[0] == {
        "role": "assistant",
        "content": "c",
        "name": "n",
        "tool_calls": [],
    }


def test_input_is_not_mutated() -> None:
    original = [{"role": "agent", "content": "c"}]
    snapshot = [dict(m) for m in original]
    result = normalize_history_roles(original)
    # Input list and its dicts are untouched (RuntimeState is append-only).
    assert original == snapshot
    assert result is not original
    assert result[0] is not original[0]


def test_idempotent() -> None:
    once = normalize_history_roles([{"role": "agent", "content": "c"}])
    twice = normalize_history_roles(once)
    assert once == twice
    # Already-valid dicts are returned by identity on the second pass.
    assert twice[0] is once[0]


def test_empty_history() -> None:
    assert normalize_history_roles([]) == []


# ---------------------------------------------------------------------------
# sanitize_history_messages — flatten cross-turn tool mechanics.
# Regression guard for the AIS designer multi-turn 400:
#   "messages.1.tool_calls: Extra inputs are not permitted"
# (a replayed user/history message carrying a tool_calls key) AND the
# tool-call pairing/orphan-tool-message 400 class.
# ---------------------------------------------------------------------------


def _has_tool_keys(messages: list[dict]) -> bool:
    return any(("tool_calls" in m or "tool_call_id" in m or "name" in m) for m in messages)


def test_sanitize_strips_tool_calls_key_from_user_message() -> None:
    # The exact AIS failure: a user-role history message carrying tool_calls.
    out = sanitize_history_messages(
        [{"role": "user", "content": "q", "tool_calls": []}]
    )
    assert out == [{"role": "user", "content": "q"}]
    assert not _has_tool_keys(out)


def test_sanitize_drops_tool_and_function_messages() -> None:
    out = sanitize_history_messages(
        [
            {"role": "user", "content": "q"},
            {"role": "tool", "content": "result", "tool_call_id": "c1"},
            {"role": "function", "content": "fres", "name": "fn"},
        ]
    )
    assert out == [{"role": "user", "content": "q"}]


def test_sanitize_strips_assistant_tool_calls_keeps_text() -> None:
    out = sanitize_history_messages(
        [{"role": "assistant", "content": "here is the plan", "tool_calls": [{"id": "c1"}]}]
    )
    assert out == [{"role": "assistant", "content": "here is the plan"}]


def test_sanitize_drops_tool_call_only_assistant_turn() -> None:
    # Assistant turn that carried only tool_calls (no text) → dropped (an empty
    # assistant message is invalid and carries no forward value).
    out = sanitize_history_messages(
        [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
            {"role": "assistant", "content": "   ", "tool_calls": [{"id": "c2"}]},
        ]
    )
    assert out == []


def test_sanitize_agent_role_mapped_to_assistant() -> None:
    out = sanitize_history_messages([{"role": "agent", "content": "report"}])
    assert out == [{"role": "assistant", "content": "report"}]


def test_sanitize_multi_turn_designer_regression() -> None:
    # The exact role sequence captured on AIS that 400'd at intent_grounding.
    history = [
        {"role": "user", "content": "build best-of-n", "tool_calls": []},
        {"role": "assistant", "content": "designed it", "tool_calls": [{"id": "a1"}]},
        {"role": "tool", "content": "discover result", "tool_call_id": "a1"},
        {"role": "tool", "content": "more", "tool_call_id": "a2"},
        {"role": "tool", "content": "more2", "tool_call_id": "a3"},
        {"role": "user", "content": "use the same tools as before", "tool_calls": []},
    ]
    out = sanitize_history_messages(history)
    # No tool mechanics survive; gateway-safe shape only.
    assert not _has_tool_keys(out)
    assert all(m["role"] in OPENAI_CHAT_ROLES for m in out)
    assert all(m["role"] not in {"tool", "function"} for m in out)
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert out[0] == {"role": "user", "content": "build best-of-n"}


def test_sanitize_input_is_not_mutated() -> None:
    original = [{"role": "user", "content": "q", "tool_calls": []}]
    snapshot = [dict(m) for m in original]
    result = sanitize_history_messages(original)
    assert original == snapshot  # input untouched (RuntimeState is append-only)
    assert result is not original
    assert result[0] is not original[0]


def test_sanitize_empty_history() -> None:
    assert sanitize_history_messages([]) == []
