"""ReactLoop safety-termination integration tests (feature 2.3).

When the LLM response is safety-terminated (provider content-policy stop) AND
carries partial tool calls, the loop must SUPPRESS those calls (never dispatch
them) and surface a terminal ``safety_termination`` status. Benign finish
reasons (normal stop, MAX_TOKENS / length) must NOT be suppressed — their tool
calls are dispatched as usual.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from databricks_deep_research.agents.react_loop import ReactLoop, ReactResult
from databricks_deep_research.events.types import NodeErrorEvent
from databricks_deep_research.llm.client import LLMResponse, ToolCall
from databricks_deep_research.tools.protocol import ToolDefinition, ToolResult

# ---------------------------------------------------------------------------
# Helpers (mirror tests/unit/agents/test_react_first_turn.py).
# ---------------------------------------------------------------------------


def _llm_response(
    content: str = "",
    tool_calls: list[ToolCall] | None = None,
    finish_reason: str = "stop",
) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        model="test-model",
        finish_reason=finish_reason,
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
# Safety-terminated + partial tool calls -> suppressed.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "safety_reason",
    ["content_filter", "refusal", "safety", "recitation", "prohibited_content"],
)
@pytest.mark.asyncio
async def test_safety_termination_suppresses_partial_tool_calls(
    safety_reason: str,
) -> None:
    """A safety-terminated turn with tool calls must NOT dispatch them."""
    tool = _make_tool("web_search")
    llm = AsyncMock()
    llm.complete = AsyncMock(
        return_value=_llm_response(
            content="I cannot help with that.",
            tool_calls=[_tool_call("web_search", {"query": "x"})],
            finish_reason=safety_reason,
        )
    )

    loop = ReactLoop(llm, tools=[tool], max_tool_calls=5, subtype="researcher")
    result = await loop.execute([{"role": "user", "content": "hello"}])

    assert isinstance(result, ReactResult)
    # The partial tool call was SUPPRESSED — the tool never executed.
    tool.execute.assert_not_awaited()
    # The LLM was called exactly once (no loop continuation).
    assert llm.complete.await_count == 1
    # The safety-terminated content is returned as-is.
    assert result.content == "I cannot help with that."
    # A terminal safety_termination status was emitted.
    node_errors = [e for e in result.events if isinstance(e, NodeErrorEvent)]
    assert len(node_errors) == 1
    assert node_errors[0].status == "safety_termination"


@pytest.mark.asyncio
async def test_safety_termination_without_tool_calls_falls_through() -> None:
    """A safety stop with NO tool calls is not the dangling-call case.

    There are no partial calls to suppress, so the guard does not fire and the
    loop exits through the normal no-tool-calls path (no NodeErrorEvent).
    """
    tool = _make_tool("web_search")
    llm = AsyncMock()
    llm.complete = AsyncMock(
        return_value=_llm_response(
            content="Refused.", tool_calls=[], finish_reason="refusal"
        )
    )

    loop = ReactLoop(llm, tools=[tool], max_tool_calls=5, subtype="synthesizer")
    result = await loop.execute([{"role": "user", "content": "hello"}])

    assert result.content == "Refused."
    tool.execute.assert_not_awaited()
    assert not [e for e in result.events if isinstance(e, NodeErrorEvent)]


# ---------------------------------------------------------------------------
# Benign reasons -> NOT suppressed; tool calls dispatched normally.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("benign_reason", ["tool_calls", "max_tokens", "length", "stop"])
@pytest.mark.asyncio
async def test_benign_finish_reason_dispatches_tool_calls(
    benign_reason: str,
) -> None:
    """MAX_TOKENS / length / normal-stop turns must dispatch their tool calls."""
    tool = _make_tool("web_search", result_content="search results")
    llm = AsyncMock()
    llm.complete = AsyncMock(
        side_effect=[
            # First call: tool request with a benign finish reason — must run.
            _llm_response(
                tool_calls=[_tool_call("web_search", {"query": "q"})],
                finish_reason=benign_reason,
            ),
            # Second call: final answer, no tools.
            _llm_response(content="Based on results...", finish_reason="stop"),
        ]
    )

    loop = ReactLoop(llm, tools=[tool], max_tool_calls=5, subtype="researcher")
    result = await loop.execute([{"role": "user", "content": "hello"}])

    # The tool WAS dispatched (not suppressed).
    tool.execute.assert_awaited_once()
    assert result.tool_calls_made == 1
    assert result.content == "Based on results..."
    # No safety status emitted for a benign reason.
    assert not [e for e in result.events if isinstance(e, NodeErrorEvent)]


@pytest.mark.asyncio
async def test_max_tokens_with_tool_calls_is_not_safety() -> None:
    """Explicit regression: an output-length truncation that happens to carry a
    tool call is recoverable and must be dispatched, never suppressed."""
    tool = _make_tool("web_search", result_content="data")
    llm = AsyncMock()
    llm.complete = AsyncMock(
        side_effect=[
            _llm_response(
                tool_calls=[_tool_call("web_search", {"query": "q"})],
                finish_reason="max_tokens",
            ),
            _llm_response(content="done", finish_reason="stop"),
        ]
    )

    loop = ReactLoop(llm, tools=[tool], max_tool_calls=5, subtype="researcher")
    result = await loop.execute([{"role": "user", "content": "hello"}])

    tool.execute.assert_awaited_once()
    assert not [e for e in result.events if isinstance(e, NodeErrorEvent)]
    assert result.content == "done"
