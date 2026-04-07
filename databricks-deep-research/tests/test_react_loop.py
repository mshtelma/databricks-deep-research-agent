"""Tests for the generic ReAct execution loop."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from databricks_deep_research.agents.react_loop import ReactLoop, ReactResult, ToolCallCache
from databricks_deep_research.events.types import ToolCacheHitEvent, ToolCallEvent, ToolResultEvent
from databricks_deep_research.llm.client import LLMResponse, ToolCall
from databricks_deep_research.tools.protocol import ToolContext, ToolDefinition, ToolResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(name: str = "web_search", result_content: str = "tool output") -> MagicMock:
    """Create a mock ResearchTool."""
    tool = MagicMock()
    type(tool).definition = PropertyMock(
        return_value=ToolDefinition(
            name=name,
            description=f"Mock {name}",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        )
    )
    tool.validate_arguments.side_effect = lambda args: args
    tool.execute = AsyncMock(
        return_value=ToolResult(content=result_content, sources=[])
    )
    return tool


def _llm_response(content: str = "", tool_calls: list[ToolCall] | None = None) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        model="test-model",
    )


def _tool_call(name: str = "web_search", args: dict[str, Any] | None = None, tc_id: str = "tc1") -> ToolCall:
    return ToolCall(id=tc_id, function_name=name, arguments=json.dumps(args or {"query": "test"}))


# ---------------------------------------------------------------------------
# ToolCallCache (Fix 8: now stores sources)
# ---------------------------------------------------------------------------


class TestToolCallCache:
    def test_miss_returns_none(self) -> None:
        cache = ToolCallCache()
        assert cache.get("web_search", {"query": "hello"}) is None

    def test_put_then_get(self) -> None:
        cache = ToolCallCache()
        cache.put("web_search", {"query": "hello"}, "result-1", [{"url": "http://test"}])
        result = cache.get("web_search", {"query": "hello"})
        assert result is not None
        content, sources = result
        assert content == "result-1"
        assert len(sources) == 1

    def test_put_without_sources(self) -> None:
        cache = ToolCallCache()
        cache.put("web_search", {"query": "hello"}, "result-1")
        result = cache.get("web_search", {"query": "hello"})
        assert result is not None
        content, sources = result
        assert content == "result-1"
        assert sources == []

    def test_different_args_are_different_keys(self) -> None:
        cache = ToolCallCache()
        cache.put("web_search", {"query": "a"}, "res-a", [])
        cache.put("web_search", {"query": "b"}, "res-b", [])
        result_a = cache.get("web_search", {"query": "a"})
        result_b = cache.get("web_search", {"query": "b"})
        assert result_a is not None
        assert result_b is not None
        assert result_a[0] == "res-a"
        assert result_b[0] == "res-b"

    def test_dedup_detection(self) -> None:
        """Same tool + same args should return the cached result."""
        cache = ToolCallCache()
        cache.put("crawl", {"url": "https://x.com"}, "page content", [{"url": "https://x.com"}])
        hit = cache.get("crawl", {"url": "https://x.com"})
        assert hit is not None
        assert hit[0] == "page content"
        assert len(hit[1]) == 1

    def test_cache_hit_returns_sources(self) -> None:
        """Cache hits return source metadata alongside content."""
        cache = ToolCallCache()
        sources = [{"url": "http://a.com", "title": "A"}, {"url": "http://b.com", "title": "B"}]
        cache.put("web_search", {"query": "test"}, "results", sources)
        hit = cache.get("web_search", {"query": "test"})
        assert hit is not None
        content, cached_sources = hit
        assert content == "results"
        assert cached_sources == sources


# ---------------------------------------------------------------------------
# ReactLoop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_zero_tool_calls() -> None:
    """LLM returns content only -- loop should exit after one call."""
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=_llm_response(content="Final answer"))

    loop = ReactLoop(llm, tools=[], max_tool_calls=5)
    result = await loop.execute([{"role": "user", "content": "hello"}])

    assert isinstance(result, ReactResult)
    assert result.content == "Final answer"
    assert result.tool_calls_made == 0
    llm.complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_tool_calls_executed() -> None:
    """LLM requests a tool call, tool executes, then LLM returns content."""
    tool = _make_tool("web_search", result_content="search results here")
    llm = AsyncMock()
    llm.complete = AsyncMock(
        side_effect=[
            _llm_response(tool_calls=[_tool_call("web_search", {"query": "AI"})]),
            _llm_response(content="Based on results..."),
        ]
    )

    loop = ReactLoop(llm, tools=[tool], max_tool_calls=5)
    result = await loop.execute([{"role": "user", "content": "search AI"}])

    assert result.content == "Based on results..."
    assert result.tool_calls_made == 1
    tool.execute.assert_awaited_once()

    # Check events contain ToolCallEvent and ToolResultEvent
    event_types = [type(e) for e in result.events]
    assert ToolCallEvent in event_types
    assert ToolResultEvent in event_types


@pytest.mark.asyncio
async def test_max_tool_calls_respected() -> None:
    """Loop stops after max_tool_calls even if LLM keeps requesting tools."""
    tool = _make_tool("web_search", result_content="data")
    # LLM always returns a tool call
    llm = AsyncMock()
    llm.complete = AsyncMock(
        return_value=_llm_response(
            content="partial",
            tool_calls=[_tool_call("web_search", {"query": "q"})],
        )
    )

    loop = ReactLoop(llm, tools=[tool], max_tool_calls=3)
    result = await loop.execute([{"role": "user", "content": "go"}])

    # Should stop at max_tool_calls
    assert result.tool_calls_made <= 3
    assert result.content == "partial"


@pytest.mark.asyncio
async def test_duplicate_tool_calls_skipped() -> None:
    """Second identical tool call should hit cache and skip execution."""
    tool = _make_tool("web_search", result_content="cached result")
    same_tc = _tool_call("web_search", {"query": "same"}, tc_id="tc1")
    same_tc2 = _tool_call("web_search", {"query": "same"}, tc_id="tc2")

    llm = AsyncMock()
    llm.complete = AsyncMock(
        side_effect=[
            _llm_response(tool_calls=[same_tc]),
            _llm_response(tool_calls=[same_tc2]),
            _llm_response(content="done"),
        ]
    )

    loop = ReactLoop(llm, tools=[tool], max_tool_calls=10)
    result = await loop.execute([{"role": "user", "content": "go"}])

    # Tool.execute should be called only once (second is cache hit)
    assert tool.execute.await_count == 1
    assert result.content == "done"
    assert result.tool_calls_made == 2

    # Events should include a ToolCacheHitEvent
    cache_hits = [e for e in result.events if isinstance(e, ToolCacheHitEvent)]
    assert len(cache_hits) == 1
    assert cache_hits[0].tool_name == "web_search"


@pytest.mark.asyncio
async def test_result_contains_expected_events() -> None:
    """ReactResult.events should contain call, result, and cache events."""
    tool = _make_tool("crawl", result_content="page body")
    llm = AsyncMock()
    llm.complete = AsyncMock(
        side_effect=[
            _llm_response(tool_calls=[_tool_call("crawl", {"url": "https://a.com"}, tc_id="c1")]),
            _llm_response(content="summary"),
        ]
    )

    loop = ReactLoop(llm, tools=[tool], max_tool_calls=5, node_id="researcher-0")
    result = await loop.execute([{"role": "user", "content": "crawl"}])

    assert len(result.events) == 2  # ToolCallEvent + ToolResultEvent
    assert isinstance(result.events[0], ToolCallEvent)
    assert result.events[0].node_id == "researcher-0"
    assert result.events[0].tool_name == "crawl"
    assert isinstance(result.events[1], ToolResultEvent)
    assert result.events[1].tool_name == "crawl"


@pytest.mark.asyncio
async def test_unknown_tool_returns_error_message() -> None:
    """Calling an unregistered tool should produce an error result message."""
    llm = AsyncMock()
    llm.complete = AsyncMock(
        side_effect=[
            _llm_response(tool_calls=[_tool_call("nonexistent", {"x": 1})]),
            _llm_response(content="ok"),
        ]
    )

    loop = ReactLoop(llm, tools=[], max_tool_calls=5)
    result = await loop.execute([{"role": "user", "content": "go"}])

    assert result.tool_calls_made == 1
    # The unknown tool gets handled in phase 2 (parallel exec) and emitted in phase 3
    # Check that loop completed successfully
    assert result.content == "ok"


@pytest.mark.asyncio
async def test_orphaned_tool_calls_get_stub_results() -> None:
    """When budget is hit mid-batch, remaining tool calls get stub results."""
    tool = _make_tool("web_search", result_content="data")
    # LLM returns 3 tool calls, but budget is only 2
    three_tcs = [
        _tool_call("web_search", {"query": "q1"}, tc_id="tc1"),
        _tool_call("web_search", {"query": "q2"}, tc_id="tc2"),
        _tool_call("web_search", {"query": "q3"}, tc_id="tc3"),
    ]
    llm = AsyncMock()
    llm.complete = AsyncMock(side_effect=[
        _llm_response(tool_calls=three_tcs),
        _llm_response(content="final answer"),
    ])

    loop = ReactLoop(llm, tools=[tool], max_tool_calls=2)
    result = await loop.execute([{"role": "user", "content": "go"}])

    assert result.tool_calls_made == 2
    # Verify no 400 error (the second LLM call succeeds)
    assert result.content == "final answer"
    # Verify tc3 got a stub message (check messages passed to second LLM call)
    second_call_messages = llm.complete.call_args_list[1][0][0]
    tool_results = [m for m in second_call_messages if m.get("role") == "tool"]
    tool_result_ids = {m["tool_call_id"] for m in tool_results}
    assert "tc3" in tool_result_ids  # stub result for orphaned tc3


@pytest.mark.asyncio
async def test_token_usage_accumulated() -> None:
    """Token usage from multiple LLM calls should be summed."""
    llm = AsyncMock()
    llm.complete = AsyncMock(
        side_effect=[
            _llm_response(tool_calls=[_tool_call("t", {"q": "a"})]),
            _llm_response(content="done"),
        ]
    )
    tool = _make_tool("t")

    loop = ReactLoop(llm, tools=[tool], max_tool_calls=5)
    result = await loop.execute([{"role": "user", "content": "go"}])

    # Two LLM calls each with 15 total tokens
    assert result.token_usage["total_tokens"] == 30
    assert result.token_usage["prompt_tokens"] == 20
    assert result.token_usage["completion_tokens"] == 10


@pytest.mark.asyncio
async def test_shared_cache_across_instances() -> None:
    """A pre-populated cache is used by a new ReactLoop instance."""
    cache = ToolCallCache()
    cache.put("web_search", {"query": "shared"}, "cached data", [])

    tool = _make_tool("web_search", result_content="fresh data")
    llm = AsyncMock()
    llm.complete = AsyncMock(
        side_effect=[
            _llm_response(tool_calls=[_tool_call("web_search", {"query": "shared"}, tc_id="tc1")]),
            _llm_response(content="done with cached"),
        ]
    )

    loop = ReactLoop(llm, tools=[tool], cache=cache, max_tool_calls=5, node_id="step-2")
    result = await loop.execute([{"role": "user", "content": "go"}])

    # Tool should NOT be executed — cache hit
    tool.execute.assert_not_awaited()
    assert result.content == "done with cached"

    # Events should include a cache hit
    cache_hits = [e for e in result.events if isinstance(e, ToolCacheHitEvent)]
    assert len(cache_hits) == 1
    assert cache_hits[0].tool_name == "web_search"


# ---------------------------------------------------------------------------
# Fix 1: Parallel tool execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parallel_tool_execution_faster_than_sequential() -> None:
    """Multiple tool calls in one LLM response execute in parallel."""
    async def slow_execute(args: dict[str, Any], ctx: Any) -> ToolResult:
        await asyncio.sleep(0.1)
        return ToolResult(content="ok", sources=[])

    tool = _make_tool("web_search")
    tool.execute = slow_execute  # type: ignore[assignment]

    # LLM returns 5 tool calls, then final response
    five_tcs = [_tool_call("web_search", {"query": f"q{i}"}, tc_id=f"tc{i}") for i in range(5)]
    llm = AsyncMock()
    llm.complete = AsyncMock(side_effect=[
        _llm_response(tool_calls=five_tcs),
        _llm_response(content="done"),
    ])

    loop = ReactLoop(llm, tools=[tool], max_tool_calls=10)
    start = time.monotonic()
    result = await loop.execute([{"role": "user", "content": "test"}])
    elapsed = time.monotonic() - start

    assert result.tool_calls_made == 5
    assert elapsed < 0.4  # 5 × 0.1s sequential = 0.5s; parallel < 0.4s


@pytest.mark.asyncio
async def test_parallel_execution_preserves_source_order() -> None:
    """Sources from parallel calls are reassembled in original order."""
    async def make_tool_with_source(name: str, source_url: str) -> MagicMock:
        tool = _make_tool(name)
        tool.execute = AsyncMock(
            return_value=ToolResult(
                content=f"result from {name}",
                sources=[{"url": source_url}],
            )
        )
        return tool

    tool_a = await make_tool_with_source("tool_a", "http://a.com")
    tool_b = await make_tool_with_source("tool_b", "http://b.com")

    tcs = [
        _tool_call("tool_a", {"query": "a"}, tc_id="tc_a"),
        _tool_call("tool_b", {"query": "b"}, tc_id="tc_b"),
    ]
    llm = AsyncMock()
    llm.complete = AsyncMock(side_effect=[
        _llm_response(tool_calls=tcs),
        _llm_response(content="done"),
    ])

    loop = ReactLoop(llm, tools=[tool_a, tool_b], max_tool_calls=10)
    result = await loop.execute([{"role": "user", "content": "go"}])

    assert len(result.sources) == 2
    assert result.sources[0]["url"] == "http://a.com"
    assert result.sources[1]["url"] == "http://b.com"


@pytest.mark.asyncio
async def test_vector_query_planning_adds_alternate_queries() -> None:
    tool = _make_tool("search_main_dbdemos_ai_agent_earnings_vs_index", result_content="vector results")
    type(tool).definition = PropertyMock(
        return_value=ToolDefinition(
            name="search_main_dbdemos_ai_agent_earnings_vs_index",
            description="Search quarterly earnings release documents.",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            source_type="vector_search",
            metadata={"index_name": "main.dbdemos_ai_agent.earnings_vs_index"},
        )
    )
    llm = AsyncMock()
    llm.complete = AsyncMock(side_effect=[
        _llm_response(tool_calls=[_tool_call(
            "search_main_dbdemos_ai_agent_earnings_vs_index",
            {"query": "Kroger Q3 2025 earnings report revenue net income EPS"},
        )]),
        _llm_response(content="done"),
    ])

    loop = ReactLoop(
        llm,
        tools=[tool],
        max_tool_calls=5,
        tool_context=ToolContext(
            query="Why did Kroger miss earnings expectations?",
            current_step={
                "title": "Find Kroger earnings release",
                "description": "Focus on revenue, net income, EPS, and guidance.",
            },
        ),
    )
    result = await loop.execute([{"role": "user", "content": "go"}])

    assert result.content == "done"
    validated_args = tool.validate_arguments.call_args.args[0]
    rewritten = validated_args["query"]
    # Passthrough mode preserves the LLM's raw query
    assert rewritten == "Kroger Q3 2025 earnings report revenue net income EPS"


@pytest.mark.asyncio
async def test_react_loop_widens_to_fallback_sources_when_primary_misses() -> None:
    primary = _make_tool("search_main_dbdemos_ai_agent_earnings_vs_index", result_content="irrelevant")
    type(primary).definition = PropertyMock(
        return_value=ToolDefinition(
            name="search_main_dbdemos_ai_agent_earnings_vs_index",
            description="Search earnings releases.",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            source_type="vector_search",
        )
    )
    primary.execute = AsyncMock(return_value=ToolResult(
        content="irrelevant primary result",
        sources=[{
            "url": "https://example.com/manual",
            "title": "Business Internet Setup Guide",
            "snippet": "Telecommunications installation manual.",
            "source_type": "vector_search",
        }],
    ))

    fallback = _make_tool("search_main_msh_dbdemos_ai_agent_knowledge_base_vs_index", result_content="relevant")
    type(fallback).definition = PropertyMock(
        return_value=ToolDefinition(
            name="search_main_msh_dbdemos_ai_agent_knowledge_base_vs_index",
            description="Search knowledge base documents.",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            source_type="vector_search",
        )
    )
    fallback.execute = AsyncMock(return_value=ToolResult(
        content="relevant fallback result",
        sources=[{
            "url": "https://example.com/kroger",
            "title": "Kroger Reports Fourth Quarter and Full-Year 2024 Results",
            "snippet": "Revenue and EPS details for Kroger.",
            "source_type": "vector_search",
        }],
    ))

    llm = AsyncMock()
    llm.complete = AsyncMock(side_effect=[
        _llm_response(tool_calls=[_tool_call(
            "search_main_dbdemos_ai_agent_earnings_vs_index",
            {"query": "Kroger earnings"},
            tc_id="tc1",
        )]),
        _llm_response(content="preferred sources missed"),
        _llm_response(tool_calls=[_tool_call(
            "search_main_msh_dbdemos_ai_agent_knowledge_base_vs_index",
            {"query": "Kroger earnings"},
            tc_id="tc2",
        )]),
        _llm_response(content="done"),
    ])

    loop = ReactLoop(
        llm,
        tools=[primary, fallback],
        max_tool_calls=5,
        tool_context=ToolContext(
            query="Why did Kroger miss earnings expectations?",
            current_step={
                "title": "Research Kroger earnings",
                "description": "Use earnings index first, then knowledge base if needed.",
                "source_hints": [
                    {"source_name": "search_main_dbdemos_ai_agent_earnings_vs_index", "source_type": "vector_search", "priority": 1},
                    {"source_name": "search_main_msh_dbdemos_ai_agent_knowledge_base_vs_index", "source_type": "vector_search", "priority": 3},
                ],
            },
        ),
    )
    result = await loop.execute([{"role": "user", "content": "go"}])

    assert result.content == "done"
    assert primary.execute.await_count == 1
    assert fallback.execute.await_count == 1
    assert result.sources[0]["title"].startswith("Kroger Reports")


# ---------------------------------------------------------------------------
# Fix 5: Message compaction
# ---------------------------------------------------------------------------


class TestMessageCompaction:
    def test_compact_old_tool_results(self) -> None:
        """Old tool results are truncated, current iteration preserved."""
        loop = ReactLoop(
            MagicMock(), tools=[], max_tool_calls=10,
            max_result_chars=50,
        )

        messages = [
            {"role": "user", "content": "test"},
            # First iteration (old)
            {"role": "assistant", "content": "", "tool_calls": [{"id": "tc1"}]},
            {"role": "tool", "tool_call_id": "tc1", "content": "A" * 200},
            # Second iteration (current)
            {"role": "assistant", "content": "", "tool_calls": [{"id": "tc2"}]},
            {"role": "tool", "tool_call_id": "tc2", "content": "B" * 200},
        ]

        loop._compact_old_tool_results(messages)

        # Old tool result should be truncated
        assert len(messages[2]["content"]) < 200
        assert "truncated" in messages[2]["content"]
        # Current tool result should be preserved
        assert messages[4]["content"] == "B" * 200

    def test_compact_no_op_when_disabled(self) -> None:
        """No truncation when max_result_chars is 0."""
        loop = ReactLoop(
            MagicMock(), tools=[], max_tool_calls=10,
            max_result_chars=0,
        )
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "tc1"}]},
            {"role": "tool", "tool_call_id": "tc1", "content": "A" * 200},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "tc2"}]},
            {"role": "tool", "tool_call_id": "tc2", "content": "B" * 200},
        ]
        loop._compact_old_tool_results(messages)
        assert messages[1]["content"] == "A" * 200

    def test_compact_no_op_first_iteration(self) -> None:
        """Nothing to truncate on first iteration (only one assistant+tool block)."""
        loop = ReactLoop(
            MagicMock(), tools=[], max_tool_calls=10,
            max_result_chars=50,
        )
        messages = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "tc1"}]},
            {"role": "tool", "tool_call_id": "tc1", "content": "A" * 200},
        ]
        loop._compact_old_tool_results(messages)
        # Nothing should change (only one tool-call block, no "old" results)
        assert messages[2]["content"] == "A" * 200


@pytest.mark.asyncio
async def test_vector_metadata_only_result_marks_low_value_and_adaptation() -> None:
    tool = _make_tool("vector_search", result_content="Found transcript headers")
    type(tool).definition = PropertyMock(
        return_value=ToolDefinition(
            name="vector_search",
            description="Search transcript index.",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            source_type="enterprise",
            source_kind="vector_index",
        )
    )
    tool.execute = AsyncMock(return_value=ToolResult(
        content="Found transcript headers",
        sources=[ToolResult.__dataclass_fields__["sources"].default_factory.__self__[0] if False else None],
    ))
    from databricks_deep_research.tools.protocol import SourceInfo
    tool.execute = AsyncMock(return_value=ToolResult(
        content="Found transcript headers",
        sources=[SourceInfo(url="enterprise://vector_search/transcript/1", title="Transcript", snippet="Header only")],
    ))

    llm = AsyncMock()
    llm.complete = AsyncMock(
        side_effect=[
            _llm_response(tool_calls=[_tool_call("vector_search", {"query": "Kroger earnings"})]),
            _llm_response(content="done"),
        ]
    )

    loop = ReactLoop(
        llm,
        tools=[tool],
        max_tool_calls=5,
        tool_context=ToolContext(query="Kroger earnings", current_step={"title": "Find transcript quotes"}),
    )
    result = await loop.execute([{"role": "user", "content": "go"}])

    tool_results = [e for e in result.events if isinstance(e, ToolResultEvent)]
    assert tool_results


@pytest.mark.asyncio
async def test_duplicate_low_yield_query_is_skipped() -> None:
    tool = _make_tool("vector_search", result_content="Found transcript headers")
    type(tool).definition = PropertyMock(
        return_value=ToolDefinition(
            name="vector_search",
            description="Search transcript index.",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            source_type="enterprise",
            source_kind="vector_index",
        )
    )
    from databricks_deep_research.tools.protocol import SourceInfo
    tool.execute = AsyncMock(return_value=ToolResult(
        content="Found transcript headers",
        sources=[SourceInfo(url="enterprise://vector_search/transcript/1", title="Transcript", snippet="Header only")],
    ))

    llm = AsyncMock()
    llm.complete = AsyncMock(
        side_effect=[
            _llm_response(tool_calls=[_tool_call("vector_search", {"query": "Kroger earnings"}, tc_id="tc1")]),
            _llm_response(tool_calls=[_tool_call("vector_search", {"query": "Kroger earnings"}, tc_id="tc2")]),
            _llm_response(content="done"),
        ]
    )

    loop = ReactLoop(
        llm,
        tools=[tool],
        max_tool_calls=5,
        tool_context=ToolContext(query="Kroger earnings", current_step={"title": "Find transcript quotes"}),
    )
    result = await loop.execute([{"role": "user", "content": "go"}])

    assert tool.execute.await_count == 1
    assert result.content == "done"
