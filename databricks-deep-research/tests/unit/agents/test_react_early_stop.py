"""Tests for ReactLoop diminishing-returns early stop (Change 5)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from databricks_deep_research.agents.react_loop import ReactLoop
from databricks_deep_research.llm.client import LLMResponse, ToolCall
from databricks_deep_research.tools.protocol import ToolContext, ToolDefinition

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(name: str = "vs_tool") -> MagicMock:
    """Build a mock ResearchTool."""
    tool = MagicMock()
    tool.definition = ToolDefinition(
        name=name,
        description="test tool",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        source_kind="vector_index",
    )
    tool.validate_arguments = MagicMock(side_effect=lambda a: a)
    return tool


def _make_tool_call(tc_id: str, name: str, query: str) -> ToolCall:
    return ToolCall(id=tc_id, function_name=name, arguments=json.dumps({"query": query}))


def _make_response(content: str = "", tool_calls: list[ToolCall] | None = None) -> LLMResponse:
    return LLMResponse(content=content, tool_calls=tool_calls or [], model="analytical")


def _source_dict(url: str) -> dict[str, str]:
    return {"url": url, "title": "t", "content": "c" * 100}


def _exec_result(tc_id: str, urls: list[str]) -> tuple[str, str, list[dict], dict]:
    """Build a mock _execute_single_tool return value."""
    sources = [_source_dict(u) for u in urls]
    meta = {
        "tool_success": True, "tool_error": "",
        "raw_source_count": len(sources),
        "accepted_source_count": len(sources),
        "accepted_substantive_count": len(sources),
        "accepted_low_value_count": 0,
        "rejected_source_count": 0,
        "evidence_quality": "full_text", "failure_mode": "",
        "needs_adaptation": False,
    }
    return tc_id, "results", sources, meta


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEarlyStopNudge:
    """Verify the diminishing-returns system message injection."""

    @pytest.mark.asyncio
    async def test_nudge_after_two_zero_novel_rounds(self) -> None:
        """Two consecutive rounds with no novel URLs -> system nudge injected."""
        tool = _make_tool()

        # Mock _execute_single_tool to return same URLs every round
        same_urls = ["https://a.com", "https://b.com"]
        exec_mock = AsyncMock(side_effect=[
            _exec_result("tc1", same_urls),
            _exec_result("tc2", same_urls),
            _exec_result("tc3", same_urls),
        ])

        call_sequence = [
            # Round 1: returns 2 sources (novel)
            _make_response(tool_calls=[_make_tool_call("tc1", "vs_tool", "query 1")]),
            # Round 2: same sources (0 novel) -> counter=1
            _make_response(tool_calls=[_make_tool_call("tc2", "vs_tool", "query 2")]),
            # Round 3: same sources again (0 novel) -> counter=2 -> nudge
            _make_response(tool_calls=[_make_tool_call("tc3", "vs_tool", "query 3")]),
            # Round 4: LLM complies with nudge
            _make_response(content="Here is my synthesis."),
        ]

        llm = MagicMock()
        llm.complete = AsyncMock(side_effect=call_sequence)

        ctx = ToolContext(query="test")
        loop = ReactLoop(
            llm_client=llm,
            tools=[tool],
            tool_context=ctx,
            node_id="test",
            max_tool_calls=10,
            force_convergence=True,  # Nudge requires force_convergence
        )
        loop._apply_step_tool_selection = MagicMock()
        loop._execute_single_tool = exec_mock

        result = await loop.execute([{"role": "user", "content": "test"}])

        assert result.content == "Here is my synthesis."
        # Verify the nudge system message was injected
        all_messages = llm.complete.call_args_list
        last_call_messages = all_messages[-1].args[0]
        # Mid-conversation nudges use role:user (Databricks proxy drops
        # mid-stream system messages — see commit history).
        nudge_msgs = [m for m in last_call_messages
                      if m.get("role") == "user" and "no new unique sources" in m.get("content", "")]
        assert len(nudge_msgs) >= 1

    @pytest.mark.asyncio
    async def test_no_nudge_when_novel_sources_found(self) -> None:
        """Rounds with new URLs reset the counter — no nudge."""
        tool = _make_tool()

        # Each round returns different URLs (all novel)
        exec_mock = AsyncMock(side_effect=[
            _exec_result("tc1", ["https://a.com"]),
            _exec_result("tc2", ["https://b.com"]),  # novel
        ])

        call_sequence = [
            _make_response(tool_calls=[_make_tool_call("tc1", "vs_tool", "query 1")]),
            _make_response(tool_calls=[_make_tool_call("tc2", "vs_tool", "query 2")]),
            _make_response(content="Done."),
        ]

        llm = MagicMock()
        llm.complete = AsyncMock(side_effect=call_sequence)

        ctx = ToolContext(query="test")
        loop = ReactLoop(
            llm_client=llm,
            tools=[tool],
            tool_context=ctx,
            node_id="test",
            max_tool_calls=10,
        )
        loop._apply_step_tool_selection = MagicMock()
        loop._execute_single_tool = exec_mock

        result = await loop.execute([{"role": "user", "content": "test"}])

        assert result.content == "Done."
        # No nudge should have been injected
        all_messages = llm.complete.call_args_list
        last_call_messages = all_messages[-1].args[0]
        # Mid-conversation nudges use role:user (Databricks proxy drops
        # mid-stream system messages — see commit history).
        nudge_msgs = [m for m in last_call_messages
                      if m.get("role") == "user" and "no new unique sources" in m.get("content", "")]
        assert len(nudge_msgs) == 0


class TestFallbackResetsCounter:
    """Verify _enable_fallback_tools resets the counter."""

    def test_counter_reset_on_fallback(self) -> None:
        llm = MagicMock()
        loop = ReactLoop(llm_client=llm, tools=[], node_id="test")
        loop._consecutive_zero_novel_rounds = 3
        loop._fallback_tools = {"fallback": MagicMock()}
        loop._fallback_enabled = False

        messages: list[dict] = []
        loop._enable_fallback_tools(messages, reason="test")

        assert loop._consecutive_zero_novel_rounds == 0
        assert loop._fallback_enabled is True
