"""Behavior-preserving tests for the consolidated message-hygiene helpers.

Feature 2.3 extracts two pre-existing hygiene operations into
``agents.message_hygiene``:

1. ``fill_missing_tool_results`` — the ReAct loop's "every tool_call gets a
   tool_result" dangling-fill (was inline at ``react_loop.py``).
2. ``sanitize_conversation_history`` — the harness replay-strip (re-export of
   ``llm.roles.sanitize_history_messages``).

These tests PIN that the extraction is behavior-preserving: ``fill_*`` is
compared against a literal copy of the original inline loop (the golden oracle),
and the re-export is asserted to be the same function object.
"""

from __future__ import annotations

from typing import Any

from databricks_deep_research.agents.message_hygiene import (
    fill_missing_tool_results,
    sanitize_conversation_history,
)
from databricks_deep_research.llm.client import ToolCall
from databricks_deep_research.llm.roles import sanitize_history_messages

# ---------------------------------------------------------------------------
# Golden oracle: a verbatim copy of the ORIGINAL inline react_loop logic,
# including the exact _tool_msg message shape. fill_missing_tool_results must
# produce byte-identical output to this.
# ---------------------------------------------------------------------------


def _tool_msg(tool_call_id: str, content: str) -> dict[str, Any]:
    """Verbatim copy of ReactLoop._tool_msg."""
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def _golden_inline_fill(
    messages: list[dict[str, Any]],
    tool_calls: list[ToolCall],
    responded_tc_ids: set[str],
) -> None:
    """The original inline loop, copied verbatim from react_loop.py:

        for tc in response.tool_calls:
            if tc.id not in responded_tc_ids:
                messages.append(self._tool_msg(tc.id, ""))
    """
    for tc in tool_calls:
        if tc.id not in responded_tc_ids:
            messages.append(_tool_msg(tc.id, ""))


def _tc(tc_id: str) -> ToolCall:
    return ToolCall(id=tc_id, function_name="web_search", arguments="{}")


# ---------------------------------------------------------------------------
# fill_missing_tool_results — golden behavior-preserving cases.
# ---------------------------------------------------------------------------


def _run_both(
    seed: list[dict[str, Any]],
    tool_calls: list[ToolCall],
    responded: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the golden oracle and the extracted helper on identical inputs."""
    golden = [dict(m) for m in seed]
    _golden_inline_fill(golden, tool_calls, set(responded))

    actual = [dict(m) for m in seed]
    fill_missing_tool_results(actual, tool_calls, set(responded), _tool_msg)
    return golden, actual


def test_fill_all_unanswered_matches_golden() -> None:
    tool_calls = [_tc("a"), _tc("b"), _tc("c")]
    golden, actual = _run_both([], tool_calls, responded=set())
    assert actual == golden
    assert actual == [
        {"role": "tool", "tool_call_id": "a", "content": ""},
        {"role": "tool", "tool_call_id": "b", "content": ""},
        {"role": "tool", "tool_call_id": "c", "content": ""},
    ]


def test_fill_some_already_answered_matches_golden() -> None:
    tool_calls = [_tc("a"), _tc("b"), _tc("c")]
    golden, actual = _run_both([], tool_calls, responded={"a", "c"})
    assert actual == golden
    # Only the unanswered "b" gets an empty result.
    assert actual == [{"role": "tool", "tool_call_id": "b", "content": ""}]


def test_fill_none_missing_is_noop_matches_golden() -> None:
    tool_calls = [_tc("a"), _tc("b")]
    golden, actual = _run_both([], tool_calls, responded={"a", "b"})
    assert actual == golden
    assert actual == []


def test_fill_preserves_existing_messages_and_order() -> None:
    seed = [
        {"role": "assistant", "content": "thinking", "tool_calls": [{"id": "a"}]},
        {"role": "tool", "tool_call_id": "a", "content": "real result"},
    ]
    tool_calls = [_tc("a"), _tc("b")]
    golden, actual = _run_both(seed, tool_calls, responded={"a"})
    assert actual == golden
    # "a" already answered (its real result preserved); only "b" appended.
    assert actual[-1] == {"role": "tool", "tool_call_id": "b", "content": ""}
    assert actual[0]["content"] == "thinking"
    assert actual[1]["content"] == "real result"


def test_fill_no_tool_calls_is_noop() -> None:
    golden, actual = _run_both(
        [{"role": "user", "content": "q"}], [], responded=set()
    )
    assert actual == golden
    assert actual == [{"role": "user", "content": "q"}]


def test_fill_duplicate_ids_match_golden() -> None:
    # Two calls share an id; once the first appends a result the id is NOT in
    # responded (the helper, like the original, checks against the passed set
    # only) — so both append. Pinning the original's exact behavior.
    tool_calls = [_tc("dup"), _tc("dup")]
    golden, actual = _run_both([], tool_calls, responded=set())
    assert actual == golden
    assert len(actual) == 2


# ---------------------------------------------------------------------------
# sanitize_conversation_history — confirm it IS the roles implementation
# (re-export, no behavior change).
# ---------------------------------------------------------------------------


def test_sanitize_conversation_history_is_roles_impl() -> None:
    assert sanitize_conversation_history is sanitize_history_messages


def test_sanitize_conversation_history_flattens_tool_mechanics() -> None:
    # Same contract as roles.sanitize_history_messages (spot-check the re-export
    # actually runs the strip — full coverage lives in tests/test_roles.py).
    history = [
        {"role": "user", "content": "q", "tool_calls": []},
        {"role": "assistant", "content": "answer", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "content": "result", "tool_call_id": "c1"},
    ]
    out = sanitize_conversation_history(history)
    assert out == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "answer"},
    ]
