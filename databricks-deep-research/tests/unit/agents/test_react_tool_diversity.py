"""Tests for ReactLoop tool-diversity nudge."""

from __future__ import annotations

from unittest.mock import MagicMock

from databricks_deep_research.agents.react_loop import ReactLoop


def _make_two_tool_loop() -> ReactLoop:
    """Build a ReactLoop with 2 mock tools for diversity testing."""
    llm = MagicMock()
    tool_a = MagicMock()
    tool_a.definition.name = "tool_a"
    tool_a.definition.metadata = {}
    tool_a.definition.parameters = {}
    tool_a.definition.description = "test tool a"
    tool_b = MagicMock()
    tool_b.definition.name = "tool_b"
    tool_b.definition.metadata = {}
    tool_b.definition.parameters = {}
    tool_b.definition.description = "test tool b"
    return ReactLoop(llm_client=llm, tools=[tool_a, tool_b], node_id="test")


def _make_single_tool_loop() -> ReactLoop:
    """Build a ReactLoop with 1 mock tool."""
    llm = MagicMock()
    tool_a = MagicMock()
    tool_a.definition.name = "tool_a"
    tool_a.definition.metadata = {}
    tool_a.definition.parameters = {}
    tool_a.definition.description = "test tool a"
    return ReactLoop(llm_client=llm, tools=[tool_a], node_id="test")


def _simulate_round(loop: ReactLoop, tool_names: list[str], responded_tc_ids: set[str], messages: list[dict]) -> None:
    """Simulate the diversity-tracking portion of a single execute() round.

    Mimics the code in execute() Phase 3 that tracks tool diversity and
    optionally injects a nudge message.
    """
    # Build fake response.tool_calls
    tool_calls = []
    for i, name in enumerate(tool_names):
        tc = MagicMock()
        tc.id = f"tc_{i}"
        tc.function_name = name
        tool_calls.append(tc)

    # Track tool diversity (same logic as execute())
    round_tool_names: set[str] = set()
    for tc in tool_calls:
        if tc.id in responded_tc_ids:
            round_tool_names.add(tc.function_name)

    if len(round_tool_names) == 1 and len(loop._tools) > 1:
        only_tool = next(iter(round_tool_names))
        if only_tool == loop._last_round_tool:
            loop._same_tool_consecutive_rounds += 1
        else:
            loop._same_tool_consecutive_rounds = 1
        loop._last_round_tool = only_tool
    else:
        loop._same_tool_consecutive_rounds = 0
        loop._last_round_tool = ""

    if loop._same_tool_consecutive_rounds >= 3 and len(loop._tools) > 1:
        other_tools = [n for n in loop._tools if n != loop._last_round_tool]
        messages.append({
            "role": "system",
            "content": (
                f"You have used only '{loop._last_round_tool}' for the last "
                f"{loop._same_tool_consecutive_rounds} rounds. "
                f"Other tools are available: {', '.join(other_tools)}. "
                "For cross-validation and broader coverage, try querying "
                "a different tool before concluding this step."
            ),
        })
        loop._same_tool_consecutive_rounds = 0


def test_diversity_nudge_after_3_same_tool_rounds() -> None:
    """Nudge fires after 3 consecutive rounds using only one tool."""
    loop = _make_two_tool_loop()
    messages: list[dict] = []
    responded = {"tc_0"}

    # Rounds 1-2: no nudge yet
    _simulate_round(loop, ["tool_a"], responded, messages)
    assert len(messages) == 0
    _simulate_round(loop, ["tool_a"], responded, messages)
    assert len(messages) == 0

    # Round 3: nudge fires
    _simulate_round(loop, ["tool_a"], responded, messages)
    assert len(messages) == 1
    assert "Other tools are available" in messages[0]["content"]
    assert "tool_b" in messages[0]["content"]


def test_no_diversity_nudge_when_tools_vary() -> None:
    """No nudge when LLM alternates between tools."""
    loop = _make_two_tool_loop()
    messages: list[dict] = []
    responded = {"tc_0"}

    _simulate_round(loop, ["tool_a"], responded, messages)
    _simulate_round(loop, ["tool_b"], responded, messages)
    _simulate_round(loop, ["tool_a"], responded, messages)
    _simulate_round(loop, ["tool_b"], responded, messages)

    assert len(messages) == 0


def test_no_diversity_nudge_with_single_tool() -> None:
    """No nudge when only 1 tool is available, regardless of repetition."""
    loop = _make_single_tool_loop()
    messages: list[dict] = []
    responded = {"tc_0"}

    for _ in range(5):
        _simulate_round(loop, ["tool_a"], responded, messages)

    assert len(messages) == 0


def test_diversity_nudge_counter_resets_after_nudge() -> None:
    """After nudge fires, counter resets to 0."""
    loop = _make_two_tool_loop()
    messages: list[dict] = []
    responded = {"tc_0"}

    # Trigger nudge
    for _ in range(3):
        _simulate_round(loop, ["tool_a"], responded, messages)
    assert len(messages) == 1
    assert loop._same_tool_consecutive_rounds == 0

    # Need 3 more rounds to trigger again
    messages.clear()
    _simulate_round(loop, ["tool_a"], responded, messages)
    assert len(messages) == 0
    assert loop._same_tool_consecutive_rounds == 1


def test_diversity_counter_resets_on_fallback_expansion() -> None:
    """Calling _enable_fallback_tools resets diversity counters."""
    loop = _make_two_tool_loop()
    loop._same_tool_consecutive_rounds = 2
    loop._last_round_tool = "tool_a"

    # Add a fallback tool so _enable_fallback_tools works
    fallback = MagicMock()
    fallback.definition.name = "tool_c"
    fallback.definition.metadata = {}
    fallback.definition.parameters = {}
    fallback.definition.description = "fallback tool"
    loop._fallback_tools = {"tool_c": fallback}

    messages: list[dict] = []
    loop._enable_fallback_tools(messages, reason="test")

    assert loop._same_tool_consecutive_rounds == 0
    assert loop._last_round_tool == ""


def test_diversity_counter_resets_when_multi_tool_round() -> None:
    """Round with 2 different tools resets the counter to 0."""
    loop = _make_two_tool_loop()
    messages: list[dict] = []
    responded = {"tc_0", "tc_1"}

    # Build up 2 same-tool rounds
    _simulate_round(loop, ["tool_a"], {"tc_0"}, messages)
    _simulate_round(loop, ["tool_a"], {"tc_0"}, messages)
    assert loop._same_tool_consecutive_rounds == 2

    # A round with both tools resets counter
    _simulate_round(loop, ["tool_a", "tool_b"], responded, messages)
    assert loop._same_tool_consecutive_rounds == 0
    assert loop._last_round_tool == ""
    assert len(messages) == 0
