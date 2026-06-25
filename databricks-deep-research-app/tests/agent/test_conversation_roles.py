"""Tests for app-side conversation role normalization.

Guards the app→framework boundary fix for the AIS "Invalid role" 400: the
app stores assistant turns as ``MessageRole.AGENT = "agent"``, which the LLM
gateway rejects. ``normalize_history_roles`` maps it to ``assistant`` before
handing history to the framework.
"""

from __future__ import annotations

import pytest

from deep_research.agent.utils.conversation import (
    build_messages_with_history,
    normalize_history_roles,
)


def test_normalize_maps_agent_to_assistant() -> None:
    assert normalize_history_roles([{"role": "agent", "content": "r"}]) == [
        {"role": "assistant", "content": "r"}
    ]


def test_normalize_leaves_user_and_assistant_intact() -> None:
    history = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
        {"role": "system", "content": "s"},
    ]
    assert normalize_history_roles(history) == history


@pytest.mark.parametrize("history", [None, []])
def test_normalize_handles_empty(history: list[dict[str, str]] | None) -> None:
    assert normalize_history_roles(history) == []


def test_normalize_is_idempotent_and_pure() -> None:
    original = [{"role": "agent", "content": "r"}]
    snapshot = [dict(m) for m in original]
    once = normalize_history_roles(original)
    twice = normalize_history_roles(once)
    assert once == twice
    assert original == snapshot  # input not mutated


def test_build_messages_with_history_still_maps_agent() -> None:
    """The legacy path keeps working after refactoring to the shared helper."""
    messages = build_messages_with_history(
        system_prompt="sys",
        user_query="follow-up",
        history=[
            {"role": "user", "content": "q1"},
            {"role": "agent", "content": "13K report"},
            {"role": "system", "content": "should be skipped in history"},
        ],
    )
    roles = [m["role"] for m in messages]
    # system(prompt), user(q1), assistant(was agent), user(current);
    # the history "system" turn is skipped.
    assert roles == ["system", "user", "assistant", "user"]
