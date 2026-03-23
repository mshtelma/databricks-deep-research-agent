"""Tests for ReactLoop dedup: _is_low_yield_duplicate, _dedup_check_and_register, VS path."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from databricks_deep_research.agents.react_loop import ReactLoop
from databricks_deep_research.llm.client import ToolCall


def _make_loop() -> ReactLoop:
    """Build a minimal ReactLoop for dedup testing."""
    llm = MagicMock()
    return ReactLoop(llm_client=llm, tools=[], node_id="test")


def test_exact_duplicate_detected() -> None:
    """Second identical query is detected as duplicate."""
    loop = _make_loop()
    sig = loop._normalize_query_signature("tool_a", "databricks agent")
    loop._step_query_signatures.add(sig)

    assert loop._is_low_yield_duplicate("tool_a", "databricks agent") is True


def test_first_occurrence_passes() -> None:
    """First occurrence of a query is NOT a duplicate."""
    loop = _make_loop()
    assert loop._is_low_yield_duplicate("tool_a", "databricks agent") is False


def test_jaccard_near_duplicate_detected() -> None:
    """Queries with >80% word overlap are near-duplicates."""
    loop = _make_loop()
    # Registered: {databricks, agent, deploy, guide, setup} = 5 words
    sig = loop._normalize_query_signature("tool_a", "databricks agent deploy guide setup")
    loop._step_query_signatures.add(sig)

    # Query: {databricks, agent, deploy, guide, config} = 5 words
    # Intersection: {databricks, agent, deploy, guide} = 4
    # Union: {databricks, agent, deploy, guide, setup, config} = 6
    # Jaccard = 4/6 = 0.667 -> below threshold
    assert loop._is_low_yield_duplicate("tool_a", "databricks agent deploy guide config") is False

    # Now test a true near-duplicate: subset + 1 extra word
    loop._step_query_signatures.clear()
    # Registered: {databricks, agent, deploy, guide, setup, config} = 6 words
    sig = loop._normalize_query_signature("tool_a", "databricks agent deploy guide setup config")
    loop._step_query_signatures.add(sig)

    # Query: {databricks, agent, deploy, guide, setup} = 5 words (subset)
    # Intersection: 5, Union: 6, Jaccard = 5/6 = 0.833 > 0.8
    assert loop._is_low_yield_duplicate("tool_a", "databricks agent deploy guide setup") is True


def test_jaccard_below_threshold_passes() -> None:
    """Queries with low word overlap are NOT duplicates."""
    loop = _make_loop()
    sig = loop._normalize_query_signature("tool_a", "databricks agent deploy")
    loop._step_query_signatures.add(sig)

    assert loop._is_low_yield_duplicate("tool_a", "kubernetes pod orchestration") is False


def test_cross_tool_not_duplicate() -> None:
    """Same query on different tools is NOT a duplicate."""
    loop = _make_loop()
    sig = loop._normalize_query_signature("tool_a", "databricks agent")
    loop._step_query_signatures.add(sig)

    assert loop._is_low_yield_duplicate("tool_b", "databricks agent") is False


def test_short_query_not_jaccard_checked() -> None:
    """Single-word queries skip Jaccard (avoid instability on tiny sets)."""
    loop = _make_loop()
    sig = loop._normalize_query_signature("tool_a", "databricks")
    loop._step_query_signatures.add(sig)

    # Exact match still works
    assert loop._is_low_yield_duplicate("tool_a", "databricks") is True

    # But near-match on short query (different single word) won't trigger Jaccard
    # because query_words has < 2 elements
    assert loop._is_low_yield_duplicate("tool_a", "kubernetes") is False


# ---------------------------------------------------------------------------
# _dedup_check_and_register helper tests
# ---------------------------------------------------------------------------


def test_dedup_check_and_register_novel_query() -> None:
    """Novel query returns None and registers signature."""
    loop = _make_loop()
    result = loop._dedup_check_and_register("tool_a", "databricks agent setup")
    assert result is None
    assert loop._normalize_query_signature("tool_a", "databricks agent setup") in loop._step_query_signatures


def test_dedup_check_and_register_exact_duplicate() -> None:
    """Exact duplicate returns skip_meta with failure_mode."""
    loop = _make_loop()
    # Register first
    loop._dedup_check_and_register("tool_a", "databricks agent setup")
    # Second call is duplicate
    result = loop._dedup_check_and_register("tool_a", "databricks agent setup")
    assert result is not None
    assert result["failure_mode"] == "duplicate_low_yield"
    assert result["needs_adaptation"] is True
    assert result["raw_source_count"] == 0


def test_dedup_check_and_register_jaccard_duplicate() -> None:
    """Near-duplicate (>80% Jaccard) returns skip_meta."""
    loop = _make_loop()
    # Register 6-word query
    loop._dedup_check_and_register("tool_a", "databricks agent deploy guide setup config")
    # 5/6 overlap → Jaccard = 5/6 = 0.833 > 0.8
    result = loop._dedup_check_and_register("tool_a", "databricks agent deploy guide setup")
    assert result is not None
    assert result["failure_mode"] == "duplicate_low_yield"


# ---------------------------------------------------------------------------
# VS branch integration test
# ---------------------------------------------------------------------------


def _make_vs_tool(name: str = "vs_index_1") -> MagicMock:
    """Create a mock VS tool with passthrough query_policy."""
    tool = MagicMock()
    tool.definition.name = name
    tool.definition.metadata = {"query_policy": "passthrough", "source_kind": "vector_index"}
    tool.definition.parameters = {}
    tool.definition.description = "test vs index"
    return tool


def test_vs_branch_calls_dedup() -> None:
    """VS-optimized path dedup-skips when same query is called twice."""
    llm = MagicMock()
    vs_tool = _make_vs_tool("vs_index_1")
    loop = ReactLoop(llm_client=llm, tools=[vs_tool], node_id="test")

    tc1 = ToolCall(id="tc1", function_name="vs_index_1", arguments=json.dumps({"query": "databricks agents"}))
    tc2 = ToolCall(id="tc2", function_name="vs_index_1", arguments=json.dumps({"query": "databricks agents"}))
    args = {"query": "databricks agents"}

    # Mock _execute_vs_optimized to return success
    mock_result = (tc1.id, "some content", [{"url": "http://example.com"}], {
        "tool_success": True, "tool_error": "", "raw_source_count": 1,
        "accepted_source_count": 1, "accepted_substantive_count": 1,
        "accepted_low_value_count": 0, "rejected_source_count": 0,
        "evidence_quality": "good", "failure_mode": "", "needs_adaptation": False,
    })

    with (
        patch.object(loop, "_execute_vs_optimized", new_callable=AsyncMock, return_value=mock_result),
        patch("databricks_deep_research.agents.react_loop.tool_source_kind", return_value="vector_index"),
    ):
            # First call should execute
            result1 = asyncio.get_event_loop().run_until_complete(
                loop._execute_single_tool(tc1, args)
            )
            assert result1[3].get("failure_mode", "") != "duplicate_low_yield"

            # Second call with same query should be dedup-skipped
            result2 = asyncio.get_event_loop().run_until_complete(
                loop._execute_single_tool(tc2, args)
            )
            assert result2[3]["failure_mode"] == "duplicate_low_yield"


def test_standard_branch_uses_shared_helper() -> None:
    """Standard (non-VS) path still dedup-skips via shared helper."""
    loop = _make_loop()
    # Pre-register a query signature
    loop._dedup_check_and_register("web_search", "how to build databricks apps")

    # Verify same query is detected as duplicate
    result = loop._dedup_check_and_register("web_search", "how to build databricks apps")
    assert result is not None
    assert result["failure_mode"] == "duplicate_low_yield"
