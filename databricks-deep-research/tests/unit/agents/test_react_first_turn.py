"""Tests for first-turn no-tool retry logic in ReactLoop.

When an evidence-gathering subtype (researcher, background) produces no tool
calls on its very first LLM turn, the loop injects a nudge message and retries
once.  Non-evidence subtypes (synthesizer, etc.) should exit immediately.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from databricks_deep_research.agents.react_loop import ReactLoop, ReactResult
from databricks_deep_research.events.types import (
    AgentStreamChunkEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from databricks_deep_research.llm.client import LLMResponse, ToolCall
from databricks_deep_research.tools.protocol import ToolContext, ToolDefinition, ToolResult


# ---------------------------------------------------------------------------
# Helpers (mirrors patterns from tests/test_react_loop.py)
# ---------------------------------------------------------------------------


def _llm_response(
    content: str = "",
    tool_calls: list[ToolCall] | None = None,
) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        model="test-model",
    )


def _tool_call(
    name: str = "web_search",
    args: dict[str, Any] | None = None,
    tc_id: str = "tc1",
) -> ToolCall:
    return ToolCall(
        id=tc_id,
        function_name=name,
        arguments=json.dumps(args or {"query": "test"}),
    )


def _make_tool(
    name: str = "web_search",
    result_content: str = "tool output",
) -> MagicMock:
    """Create a mock ResearchTool."""
    tool = MagicMock()
    type(tool).definition = PropertyMock(
        return_value=ToolDefinition(
            name=name,
            description=f"Mock {name}",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        )
    )
    tool.validate_arguments.side_effect = lambda args: args
    tool.execute = AsyncMock(
        return_value=ToolResult(content=result_content, sources=[])
    )
    return tool


# ---------------------------------------------------------------------------
# Test: researcher subtype gets one retry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_researcher_retries_on_first_turn_no_tools() -> None:
    """Researcher subtype gets a second chance when the first turn has no tool calls."""
    tool = _make_tool("web_search", result_content="search results")
    llm = AsyncMock()
    llm.complete = AsyncMock(
        side_effect=[
            # First call: no tool calls (should trigger retry)
            _llm_response(content="I can answer that directly."),
            # Second call (after nudge): uses a tool
            _llm_response(
                tool_calls=[_tool_call("web_search", {"query": "test"})],
            ),
            # Third call: final answer
            _llm_response(content="Based on search results..."),
        ]
    )

    loop = ReactLoop(
        llm, tools=[tool], max_tool_calls=5, subtype="researcher",
    )
    result = await loop.execute([{"role": "user", "content": "hello"}])

    assert isinstance(result, ReactResult)
    # The tool was called during the retried turn
    assert result.tool_calls_made == 1
    assert result.content == "Based on search results..."
    tool.execute.assert_awaited_once()
    # LLM was called 3 times: initial (no tools) + retry (tool call) + final
    assert llm.complete.await_count == 3


# ---------------------------------------------------------------------------
# Test: background subtype gets one retry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_background_retries_on_first_turn_no_tools() -> None:
    """Background subtype also receives the first-turn retry nudge."""
    tool = _make_tool("web_search", result_content="background data")
    llm = AsyncMock()
    llm.complete = AsyncMock(
        side_effect=[
            _llm_response(content="Let me think..."),
            _llm_response(
                tool_calls=[_tool_call("web_search", {"query": "bg"})],
            ),
            _llm_response(content="Background complete."),
        ]
    )

    loop = ReactLoop(
        llm, tools=[tool], max_tool_calls=5, subtype="background",
    )
    result = await loop.execute([{"role": "user", "content": "research this"}])

    assert result.tool_calls_made == 1
    assert result.content == "Background complete."
    assert llm.complete.await_count == 3


# ---------------------------------------------------------------------------
# Test: non-evidence subtypes exit normally
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesizer_exits_without_retry_on_no_tools() -> None:
    """Synthesizer (non-evidence subtype) exits immediately when no tool calls."""
    llm = AsyncMock()
    llm.complete = AsyncMock(
        return_value=_llm_response(content="Here is the synthesis."),
    )

    loop = ReactLoop(
        llm, tools=[], max_tool_calls=5, subtype="synthesizer",
    )
    result = await loop.execute([{"role": "user", "content": "synthesize"}])

    assert result.content == "Here is the synthesis."
    assert result.tool_calls_made == 0
    # Only one LLM call -- no retry
    llm.complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_empty_subtype_exits_without_retry() -> None:
    """Default (empty) subtype exits immediately when no tool calls."""
    llm = AsyncMock()
    llm.complete = AsyncMock(
        return_value=_llm_response(content="Done."),
    )

    loop = ReactLoop(llm, tools=[], max_tool_calls=5, subtype="")
    result = await loop.execute([{"role": "user", "content": "go"}])

    assert result.content == "Done."
    assert result.tool_calls_made == 0
    llm.complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_reflector_exits_without_retry() -> None:
    """Reflector subtype exits immediately when no tool calls."""
    llm = AsyncMock()
    llm.complete = AsyncMock(
        return_value=_llm_response(content="Reflection complete."),
    )

    loop = ReactLoop(
        llm, tools=[], max_tool_calls=5, subtype="reflector",
    )
    result = await loop.execute([{"role": "user", "content": "reflect"}])

    assert result.content == "Reflection complete."
    llm.complete.assert_awaited_once()


# ---------------------------------------------------------------------------
# Test: retry does not duplicate streaming events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_turn_does_not_stream() -> None:
    """After first_turn_retried is set, the retry LLM call uses complete() not stream().

    When stream=True, the first call uses _stream_call.  If that first call
    triggers the retry, the retried call should use plain complete() (because
    the condition ``call_count == 0 and not first_turn_retried`` is False on
    the retry), preventing duplicate streaming chunks from reaching the UI.
    """
    tool = _make_tool("web_search", result_content="data")
    llm = AsyncMock()

    # _stream_call path: first call returns no tool calls
    first_stream_response = _llm_response(content="Let me think...")
    stream_chunk_event = AgentStreamChunkEvent(
        node_id="r0",
        timestamp="2025-01-01T00:00:00Z",
        chunk="Let me think...",
        subtype="researcher",
    )

    # complete() path: second call (retry) returns tool call, third is final
    llm.complete = AsyncMock(
        side_effect=[
            _llm_response(
                tool_calls=[_tool_call("web_search", {"query": "q"})],
            ),
            _llm_response(content="Final answer."),
        ]
    )

    loop = ReactLoop(
        llm,
        tools=[tool],
        max_tool_calls=5,
        subtype="researcher",
        stream=True,
        node_id="r0",
    )

    # Patch _stream_call so we control exactly what the first streaming call returns
    loop._stream_call = AsyncMock(  # type: ignore[method-assign]
        return_value=(first_stream_response, [stream_chunk_event]),
    )

    result = await loop.execute([{"role": "user", "content": "go"}])

    # _stream_call was invoked exactly once (the very first turn)
    loop._stream_call.assert_awaited_once()  # type: ignore[union-attr]
    # complete() was invoked for the retry turn and the final turn
    assert llm.complete.await_count == 2
    assert result.content == "Final answer."

    # Streaming events from the first (retried) turn are still in the result
    # but no additional stream chunks from the retry turn
    stream_chunks = [
        e for e in result.events if isinstance(e, AgentStreamChunkEvent)
    ]
    assert len(stream_chunks) == 1
    assert stream_chunks[0].chunk == "Let me think..."


# ---------------------------------------------------------------------------
# Test: second consecutive no-tool response exits (no infinite loop)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_second_no_tool_response_exits_normally() -> None:
    """If the retry also produces no tool calls, the loop exits -- no infinite loop."""
    llm = AsyncMock()
    llm.complete = AsyncMock(
        side_effect=[
            # First call: no tool calls -> triggers retry
            _llm_response(content="Thinking..."),
            # Retry call: still no tool calls -> should exit
            _llm_response(content="I really cannot use tools."),
        ]
    )

    loop = ReactLoop(
        llm,
        tools=[_make_tool("web_search")],
        max_tool_calls=5,
        subtype="researcher",
    )
    result = await loop.execute([{"role": "user", "content": "go"}])

    assert result.content == "I really cannot use tools."
    assert result.tool_calls_made == 0
    # Exactly two LLM calls: initial + one retry
    assert llm.complete.await_count == 2


@pytest.mark.asyncio
async def test_background_second_no_tool_response_exits() -> None:
    """Background subtype also exits after the retry fails to produce tool calls."""
    llm = AsyncMock()
    llm.complete = AsyncMock(
        side_effect=[
            _llm_response(content="Hmm..."),
            _llm_response(content="Still no tools."),
        ]
    )

    loop = ReactLoop(
        llm,
        tools=[_make_tool("web_search")],
        max_tool_calls=5,
        subtype="background",
    )
    result = await loop.execute([{"role": "user", "content": "go"}])

    assert result.content == "Still no tools."
    assert result.tool_calls_made == 0
    assert llm.complete.await_count == 2


# ---------------------------------------------------------------------------
# Test: nudge message content is injected correctly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nudge_message_injected_on_retry() -> None:
    """After the first no-tool turn, the messages list should contain the nudge."""
    captured_messages: list[list[dict[str, Any]]] = []

    llm = AsyncMock()

    async def capture_complete(
        msgs: list[dict[str, Any]], *args: Any, **kwargs: Any,
    ) -> LLMResponse:
        # Deep-copy the current messages list to capture the state at each call
        captured_messages.append([dict(m) for m in msgs])
        if len(captured_messages) == 1:
            return _llm_response(content="No tools yet.")
        if len(captured_messages) == 2:
            return _llm_response(content="Still no tools.")
        return _llm_response(content="done")

    llm.complete = capture_complete  # type: ignore[assignment]

    loop = ReactLoop(
        llm,
        tools=[_make_tool("web_search")],
        max_tool_calls=5,
        subtype="researcher",
    )
    await loop.execute([{"role": "user", "content": "go"}])

    # The second call's messages should include the nudge system message
    assert len(captured_messages) == 2
    retry_messages = captured_messages[1]
    system_msgs = [m for m in retry_messages if m.get("role") == "system"]
    assert any("search tools available" in m["content"] for m in system_msgs)

    # The assistant response from the first turn should also be present
    assistant_msgs = [m for m in retry_messages if m.get("role") == "assistant"]
    assert any("No tools yet." in m.get("content", "") for m in assistant_msgs)


# ---------------------------------------------------------------------------
# Test: retry only happens on first turn (call_count == 0)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_retry_on_later_turns() -> None:
    """If a researcher makes tool calls first, then produces no tool calls later,
    there is no retry -- the retry only applies to the very first turn."""
    tool = _make_tool("web_search", result_content="data")
    llm = AsyncMock()
    llm.complete = AsyncMock(
        side_effect=[
            # First call: has tool calls (no retry needed)
            _llm_response(
                tool_calls=[_tool_call("web_search", {"query": "q1"})],
            ),
            # Second call: no tool calls -- should exit, NOT retry
            _llm_response(content="All done."),
        ]
    )

    loop = ReactLoop(
        llm, tools=[tool], max_tool_calls=5, subtype="researcher",
    )
    result = await loop.execute([{"role": "user", "content": "go"}])

    assert result.content == "All done."
    assert result.tool_calls_made == 1
    # Exactly two LLM calls: one with tools, one final
    assert llm.complete.await_count == 2
