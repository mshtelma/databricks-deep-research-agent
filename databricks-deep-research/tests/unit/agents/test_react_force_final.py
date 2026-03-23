"""Tests for budget-aware prompting and compaction summarization."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from databricks_deep_research.agents.react_loop import (
    ReactLoop,
    ReactResult,
    _summarize_tool_result,
)
from databricks_deep_research.llm.client import LLMResponse, ToolCall
from databricks_deep_research.tools.protocol import (
    ToolContext,
    ToolDefinition,
    ToolResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(name: str = "compute", source_kind: str = "builtin") -> MagicMock:
    tool = MagicMock()
    tool.definition = ToolDefinition(
        name=name,
        description="test tool",
        parameters={"type": "object", "properties": {"code": {"type": "string"}}},
        source_kind=source_kind,
    )
    tool.validate_arguments = MagicMock(side_effect=lambda a: a)
    tool.execute = AsyncMock(return_value=ToolResult(
        content="42", success=True, sources=[],
    ))
    return tool


def _make_search_tool() -> MagicMock:
    return _make_tool(name="treasury_search", source_kind="vector_index")


def _tc(tc_id: str, name: str = "compute", code: str = "2+2") -> ToolCall:
    return ToolCall(id=tc_id, function_name=name, arguments=json.dumps({"code": code}))


def _resp(content: str = "", tool_calls: list[ToolCall] | None = None) -> LLMResponse:
    return LLMResponse(content=content, tool_calls=tool_calls or [], model="test")


def _messages() -> list[dict[str, str]]:
    return [{"role": "user", "content": "test query"}]


# ---------------------------------------------------------------------------
# Budget guidance
# ---------------------------------------------------------------------------


class TestBudgetGuidance:
    def test_no_guidance_at_high_budget(self) -> None:
        loop = ReactLoop(MagicMock(), [_make_tool()], max_tool_calls=20)
        msgs: list[dict] = []
        result = loop._inject_budget_guidance(msgs, remaining=15)
        assert result is None
        assert len(msgs) == 0

    def test_warning_at_25_percent(self) -> None:
        loop = ReactLoop(MagicMock(), [_make_tool()], max_tool_calls=20)
        msgs: list[dict] = []
        loop._inject_budget_guidance(msgs, remaining=5)
        assert len(msgs) == 1
        assert "BUDGET" in msgs[0]["content"]

    def test_warning_fires_once(self) -> None:
        loop = ReactLoop(MagicMock(), [_make_tool()], max_tool_calls=20)
        msgs: list[dict] = []
        loop._inject_budget_guidance(msgs, remaining=5)
        loop._inject_budget_guidance(msgs, remaining=4)
        assert len(msgs) == 1  # Only the first warning

    def test_critical_at_remaining_2(self) -> None:
        compute = _make_tool()
        loop = ReactLoop(MagicMock(), [compute], max_tool_calls=20)
        msgs: list[dict] = []
        result = loop._inject_budget_guidance(msgs, remaining=2)
        assert len(msgs) == 1
        assert "CRITICAL" in msgs[0]["content"]
        assert result is not None

    def test_restriction_only_keeps_compute(self) -> None:
        search = _make_search_tool()
        compute = _make_tool()
        loop = ReactLoop(MagicMock(), [search, compute], max_tool_calls=20)
        msgs: list[dict] = []
        restricted = loop._inject_budget_guidance(msgs, remaining=1)
        assert restricted is not None
        assert len(restricted) == 1
        assert restricted[0]["function"]["name"] == "compute"

    def test_restriction_returns_none_when_no_compute(self) -> None:
        search = _make_search_tool()
        loop = ReactLoop(MagicMock(), [search], max_tool_calls=20)
        msgs: list[dict] = []
        result = loop._inject_budget_guidance(msgs, remaining=1)
        assert result is None  # No compute → None → text only

    def test_no_guidance_at_remaining_zero(self) -> None:
        loop = ReactLoop(MagicMock(), [_make_tool()], max_tool_calls=20)
        msgs: list[dict] = []
        result = loop._inject_budget_guidance(msgs, remaining=0)
        assert result is None
        assert len(msgs) == 0


# ---------------------------------------------------------------------------
# Namespace fallback
# ---------------------------------------------------------------------------


class TestNamespaceFallback:
    @pytest.mark.asyncio
    async def test_namespace_fallback_on_empty_content(self) -> None:
        """When max_calls with empty content, dump compute namespace."""
        tool = _make_tool()
        tool._namespace = {"defense_1940": 2602, "defense_1953": 44463}

        llm = MagicMock()
        call_count = 0

        async def mock_complete(messages, tier, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _resp(content="", tool_calls=[_tc("tc1")])
            # Last call: empty content, tool calls
            return _resp(content="", tool_calls=[_tc("tc2")])

        llm.complete = AsyncMock(side_effect=mock_complete)

        loop = ReactLoop(llm, [tool], max_tool_calls=1, node_id="test")
        result = await loop.execute(_messages())

        assert "defense_1940" in result.content
        assert "2602" in result.content

    @pytest.mark.asyncio
    async def test_no_fallback_when_content_present(self) -> None:
        """When max_calls but content exists, use that content."""
        tool = _make_tool()
        tool._namespace = {"x": 42}

        llm = MagicMock()
        call_count = 0

        async def mock_complete(messages, tier, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _resp(content="", tool_calls=[_tc("tc1")])
            return _resp(content="The answer is 42.", tool_calls=[_tc("tc2")])

        llm.complete = AsyncMock(side_effect=mock_complete)

        loop = ReactLoop(llm, [tool], max_tool_calls=1, node_id="test")
        result = await loop.execute(_messages())

        assert "answer is 42" in result.content
        assert "Extracted data" not in result.content


# ---------------------------------------------------------------------------
# Compaction summarization
# ---------------------------------------------------------------------------


class TestSummarizeToolResult:
    def test_preserves_table_rows(self) -> None:
        content = "| A | B |\n| 1 | 2 |\nSome narrative text.\n| 3 | 4 |"
        result = _summarize_tool_result(content, max_chars=500)
        assert "| A | B |" in result
        assert "| 1 | 2 |" in result
        assert "narrative" not in result

    def test_preserves_numeric_lines(self) -> None:
        content = "Description text without numbers\nValue: 2,602\nMore description"
        result = _summarize_tool_result(content, max_chars=500)
        assert "2,602" in result
        assert "Description text without numbers" not in result

    def test_respects_max_chars(self) -> None:
        content = "\n".join(f"| row{i} | {i * 100} |" for i in range(100))
        result = _summarize_tool_result(content, max_chars=200)
        assert "truncated" in result
        assert len(result) < 500

    def test_empty_content_returns_placeholder(self) -> None:
        result = _summarize_tool_result("   \n\n  ", max_chars=200)
        assert "no tabular data" in result

    def test_preserves_metadata_headers(self) -> None:
        content = "[0] chunk_type=table | file=test.txt\nNarrative only."
        result = _summarize_tool_result(content, max_chars=500)
        assert "[0] chunk_type=table" in result

    def test_compacted_prefix(self) -> None:
        content = "| A | B |\n| 1 | 2 |"
        result = _summarize_tool_result(content, max_chars=500)
        assert result.startswith("[Compacted from")

    def test_no_false_positive_on_short_lines(self) -> None:
        content = "ab\ncd\n\n"
        result = _summarize_tool_result(content, max_chars=500)
        assert "no tabular data" in result


class TestDelayedCompaction:
    def test_compact_after_rounds_auto_calculated(self) -> None:
        loop = ReactLoop(MagicMock(), [_make_tool()], max_tool_calls=25)
        assert loop._compact_after_rounds == 10

    def test_compact_after_rounds_small_budget(self) -> None:
        loop = ReactLoop(MagicMock(), [_make_tool()], max_tool_calls=5)
        assert loop._compact_after_rounds == 2
