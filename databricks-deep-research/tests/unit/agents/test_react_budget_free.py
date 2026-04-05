"""Tests for budget-free tool mechanism in ReactLoop.

Budget-free tools (metadata ``budget_free=True``) bypass the global
``max_tool_calls`` counter so lightweight housekeeping operations
(e.g. ``compute_namespace_list``) never starve the research budget.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from databricks_deep_research.agents.react_loop import ReactLoop
from databricks_deep_research.llm.client import LLMResponse, ToolCall
from databricks_deep_research.tools.protocol import ToolDefinition, ToolResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_budget_free_tool(name: str = "compute_namespace_list") -> MagicMock:
    """Create a mock tool with ``budget_free=True`` metadata."""
    tool = MagicMock()
    tool.definition = ToolDefinition(
        name=name,
        description="test budget-free tool",
        parameters={"type": "object", "properties": {}},
        source_kind="builtin",
        metadata={"budget_free": True},
    )
    tool.validate_arguments = MagicMock(side_effect=lambda a: a)
    tool.execute = AsyncMock(return_value=ToolResult(
        content="namespace info", success=True, sources=[],
    ))
    return tool


def _make_regular_tool(name: str = "treasury_search", source_kind: str = "vector_index") -> MagicMock:
    """Create a regular (non-budget-free) mock tool."""
    tool = MagicMock()
    tool.definition = ToolDefinition(
        name=name,
        description="test regular tool",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        source_kind=source_kind,
        metadata={},
    )
    tool.validate_arguments = MagicMock(side_effect=lambda a: a)
    tool.execute = AsyncMock(return_value=ToolResult(
        content="search results", success=True, sources=[],
    ))
    return tool


def _make_compute_tool() -> MagicMock:
    """Create a mock compute tool (regular, not budget-free)."""
    tool = MagicMock()
    tool.definition = ToolDefinition(
        name="compute",
        description="test compute tool",
        parameters={"type": "object", "properties": {"code": {"type": "string"}}},
        source_kind="builtin",
        metadata={},
    )
    tool.validate_arguments = MagicMock(side_effect=lambda a: a)
    tool.execute = AsyncMock(return_value=ToolResult(
        content="42", success=True, sources=[],
    ))
    return tool


def _tc(tc_id: str, name: str, args: str = "{}") -> ToolCall:
    return ToolCall(id=tc_id, function_name=name, arguments=args)


def _resp(content: str = "", tool_calls: list[ToolCall] | None = None) -> LLMResponse:
    return LLMResponse(content=content, tool_calls=tool_calls or [], model="test")


def _messages() -> list[dict[str, str]]:
    return [{"role": "user", "content": "test query"}]


# ---------------------------------------------------------------------------
# Budget-free frozenset construction
# ---------------------------------------------------------------------------


class TestBudgetFreeToolsFrozenset:
    """Verify ``_budget_free_tools`` is computed correctly from tool metadata."""

    def test_single_budget_free_tool(self) -> None:
        free_tool = _make_budget_free_tool("ns_list")
        regular = _make_regular_tool("search")
        loop = ReactLoop(MagicMock(), [free_tool, regular], max_tool_calls=10)
        assert loop._budget_free_tools == frozenset({"ns_list"})

    def test_multiple_budget_free_tools(self) -> None:
        free1 = _make_budget_free_tool("ns_list")
        free2 = _make_budget_free_tool("ns_get")
        regular = _make_regular_tool("search")
        loop = ReactLoop(MagicMock(), [free1, free2, regular], max_tool_calls=10)
        assert loop._budget_free_tools == frozenset({"ns_list", "ns_get"})

    def test_no_budget_free_tools(self) -> None:
        regular1 = _make_regular_tool("search")
        regular2 = _make_compute_tool()
        loop = ReactLoop(MagicMock(), [regular1, regular2], max_tool_calls=10)
        assert loop._budget_free_tools == frozenset()

    def test_all_budget_free_tools(self) -> None:
        free1 = _make_budget_free_tool("ns_list")
        free2 = _make_budget_free_tool("ns_get")
        loop = ReactLoop(MagicMock(), [free1, free2], max_tool_calls=10)
        assert loop._budget_free_tools == frozenset({"ns_list", "ns_get"})

    def test_type_is_frozenset(self) -> None:
        free_tool = _make_budget_free_tool()
        loop = ReactLoop(MagicMock(), [free_tool], max_tool_calls=10)
        assert isinstance(loop._budget_free_tools, frozenset)


# ---------------------------------------------------------------------------
# Budget-free tool not counted against call budget
# ---------------------------------------------------------------------------


class TestBudgetFreeToolNotCounted:
    """Budget-free tools must not increment ``call_count`` in Phase 1."""

    @pytest.mark.asyncio
    async def test_budget_free_tool_not_counted(self) -> None:
        """Calling a budget-free tool should not consume the global budget."""
        free_tool = _make_budget_free_tool("ns_list")
        regular = _make_regular_tool("search")

        llm = MagicMock()
        call_num = 0

        async def mock_complete(messages, tier, **kwargs):
            nonlocal call_num
            call_num += 1
            if call_num == 1:
                # First LLM call: invoke the budget-free tool
                return _resp(tool_calls=[_tc("tc1", "ns_list")])
            # Second call: produce final answer
            return _resp(content="Done.")

        llm.complete = AsyncMock(side_effect=mock_complete)

        loop = ReactLoop(llm, [free_tool, regular], max_tool_calls=5, node_id="test")
        result = await loop.execute(_messages())

        # Budget-free call should NOT have consumed any of the 5 budget slots
        assert result.tool_calls_made == 0

    @pytest.mark.asyncio
    async def test_multiple_budget_free_calls_not_counted(self) -> None:
        """Multiple budget-free calls in sequence leave call_count at zero."""
        free_tool = _make_budget_free_tool("ns_list")

        llm = MagicMock()
        call_num = 0

        async def mock_complete(messages, tier, **kwargs):
            nonlocal call_num
            call_num += 1
            if call_num <= 3:
                return _resp(tool_calls=[_tc(f"tc{call_num}", "ns_list")])
            return _resp(content="Done.")

        llm.complete = AsyncMock(side_effect=mock_complete)

        loop = ReactLoop(llm, [free_tool], max_tool_calls=5, node_id="test")
        result = await loop.execute(_messages())

        assert result.tool_calls_made == 0


# ---------------------------------------------------------------------------
# Regular tool still counted
# ---------------------------------------------------------------------------


class TestRegularToolStillCounted:
    """Non-budget-free tools must increment ``call_count`` normally."""

    @pytest.mark.asyncio
    async def test_regular_tool_still_counted(self) -> None:
        """A regular tool call should increment the global budget counter."""
        free_tool = _make_budget_free_tool("ns_list")
        regular = _make_regular_tool("search")

        llm = MagicMock()
        call_num = 0

        async def mock_complete(messages, tier, **kwargs):
            nonlocal call_num
            call_num += 1
            if call_num == 1:
                return _resp(tool_calls=[
                    _tc("tc1", "search", json.dumps({"query": "test"})),
                ])
            return _resp(content="Done.")

        llm.complete = AsyncMock(side_effect=mock_complete)

        loop = ReactLoop(llm, [free_tool, regular], max_tool_calls=5, node_id="test")
        result = await loop.execute(_messages())

        # Regular tool uses 1 budget slot
        assert result.tool_calls_made == 1

    @pytest.mark.asyncio
    async def test_mixed_calls_only_counts_regular(self) -> None:
        """In a batch with both free and regular calls, only regular is counted."""
        free_tool = _make_budget_free_tool("ns_list")
        regular = _make_regular_tool("search")

        llm = MagicMock()
        call_num = 0

        async def mock_complete(messages, tier, **kwargs):
            nonlocal call_num
            call_num += 1
            if call_num == 1:
                # LLM emits one budget-free + one regular in the same turn
                return _resp(tool_calls=[
                    _tc("tc1", "ns_list"),
                    _tc("tc2", "search", json.dumps({"query": "test"})),
                ])
            return _resp(content="Done.")

        llm.complete = AsyncMock(side_effect=mock_complete)

        loop = ReactLoop(llm, [free_tool, regular], max_tool_calls=5, node_id="test")
        result = await loop.execute(_messages())

        # Only the regular tool counts against the budget
        assert result.tool_calls_made == 1

    @pytest.mark.asyncio
    async def test_regular_tool_exhausts_budget(self) -> None:
        """Regular tools can exhaust the budget while budget-free keeps working."""
        free_tool = _make_budget_free_tool("ns_list")
        regular = _make_regular_tool("search")

        llm = MagicMock()
        call_num = 0

        async def mock_complete(messages, tier, **kwargs):
            nonlocal call_num
            call_num += 1
            if call_num == 1:
                # 2 regular calls to exhaust max_tool_calls=2
                return _resp(tool_calls=[
                    _tc("tc1", "search", json.dumps({"query": "q1"})),
                    _tc("tc2", "search", json.dumps({"query": "q2"})),
                ])
            if call_num == 2:
                # Now try 1 more regular (should be rejected) + 1 free (should work)
                return _resp(tool_calls=[
                    _tc("tc3", "ns_list"),
                    _tc("tc4", "search", json.dumps({"query": "q3"})),
                ])
            return _resp(content="Final answer.")

        llm.complete = AsyncMock(side_effect=mock_complete)

        loop = ReactLoop(llm, [free_tool, regular], max_tool_calls=2, node_id="test")
        result = await loop.execute(_messages())

        # call_count = 2 (the two regular calls); the budget-free call doesn't add
        assert result.tool_calls_made == 2


# ---------------------------------------------------------------------------
# Budget-free available during convergence
# ---------------------------------------------------------------------------


class TestBudgetFreeDuringConvergence:
    """During forced convergence, budget-free tools remain available alongside compute."""

    def test_forced_convergence_phase1_includes_budget_free(self) -> None:
        """Phase 1 forced convergence keeps compute + budget-free tools."""
        compute = _make_compute_tool()
        compute._namespace = {"val": 42}
        free_tool = _make_budget_free_tool("ns_list")
        search = _make_regular_tool("search")

        loop = ReactLoop(
            MagicMock(), [search, compute, free_tool],
            max_tool_calls=80, force_convergence=True,
        )
        loop._consecutive_zero_novel_rounds = 4

        msgs: list[dict] = []
        restricted = loop._inject_budget_guidance(msgs, remaining=60)

        assert any("FORCED CONVERGENCE" in m.get("content", "") for m in msgs)
        assert restricted is not None
        restricted_names = {td["function"]["name"] for td in restricted}
        # Both compute and budget-free tool should be present
        assert "compute" in restricted_names
        assert "ns_list" in restricted_names
        # Regular search should be excluded
        assert "search" not in restricted_names

    def test_budget_critical_includes_budget_free(self) -> None:
        """Budget-critical restriction (remaining<=2) keeps compute + budget-free tools."""
        compute = _make_compute_tool()
        free_tool = _make_budget_free_tool("ns_list")
        search = _make_regular_tool("search")

        loop = ReactLoop(
            MagicMock(), [search, compute, free_tool], max_tool_calls=20,
        )

        msgs: list[dict] = []
        restricted = loop._inject_budget_guidance(msgs, remaining=2)

        assert any("CRITICAL" in m.get("content", "") for m in msgs)
        assert restricted is not None
        restricted_names = {td["function"]["name"] for td in restricted}
        assert "compute" in restricted_names
        assert "ns_list" in restricted_names
        assert "search" not in restricted_names

    def test_budget_critical_remaining_1_includes_budget_free(self) -> None:
        """At remaining=1, budget-free tools are still available."""
        compute = _make_compute_tool()
        free_tool = _make_budget_free_tool("ns_list")

        loop = ReactLoop(
            MagicMock(), [compute, free_tool], max_tool_calls=20,
        )

        msgs: list[dict] = []
        restricted = loop._inject_budget_guidance(msgs, remaining=1)

        assert restricted is not None
        restricted_names = {td["function"]["name"] for td in restricted}
        assert "compute" in restricted_names
        assert "ns_list" in restricted_names

    def test_only_budget_free_no_compute_during_convergence(self) -> None:
        """If there is no compute tool, budget-free tools alone can be returned."""
        free_tool = _make_budget_free_tool("ns_list")
        search = _make_regular_tool("search")

        loop = ReactLoop(
            MagicMock(), [search, free_tool], max_tool_calls=20,
        )

        msgs: list[dict] = []
        restricted = loop._inject_budget_guidance(msgs, remaining=2)

        assert restricted is not None
        restricted_names = {td["function"]["name"] for td in restricted}
        assert "ns_list" in restricted_names
        assert "search" not in restricted_names

    def test_forced_convergence_phase2_text_only_ignores_budget_free(self) -> None:
        """Phase 2 (text-only) returns empty list even when budget-free tools exist."""
        compute = _make_compute_tool()
        compute._namespace = {"val": 42}
        free_tool = _make_budget_free_tool("ns_list")
        search = _make_regular_tool("search")

        loop = ReactLoop(
            MagicMock(), [search, compute, free_tool],
            max_tool_calls=80, force_convergence=True,
        )
        loop._consecutive_zero_novel_rounds = 5  # Phase 2: past convergence_rounds

        msgs: list[dict] = []
        restricted = loop._inject_budget_guidance(msgs, remaining=60)

        assert any("FINAL WARNING" in m.get("content", "") for m in msgs)
        assert restricted == []
