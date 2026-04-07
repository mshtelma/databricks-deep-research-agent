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


def test_vs_branch_dedup_via_canonical_signature() -> None:
    """VS tool: same query is detected as duplicate via canonical signature in Phase 1."""
    loop = _make_loop()
    args = {"query": "databricks agents", "num_results": "10"}

    # First call: register canonical signature (simulates Phase 1)
    sig = loop._canonical_dedup_signature("vs_index_1", args)
    assert not loop._is_known_duplicate_sig(sig, "vs_index_1")
    loop._step_query_signatures.add(sig)

    # Second call with same query: detected as duplicate
    assert loop._is_known_duplicate_sig(sig, "vs_index_1")


def test_standard_branch_uses_shared_helper() -> None:
    """Standard (non-VS) path still dedup-skips via shared helper."""
    loop = _make_loop()
    # Pre-register a query signature
    loop._dedup_check_and_register("web_search", "how to build databricks apps")

    # Verify same query is detected as duplicate
    result = loop._dedup_check_and_register("web_search", "how to build databricks apps")
    assert result is not None
    assert result["failure_mode"] == "duplicate_low_yield"


# ---------------------------------------------------------------------------
# Canonical dedup signature tests
# ---------------------------------------------------------------------------


def test_canonical_sig_same_file_different_pattern_not_duplicate() -> None:
    """treasury_grep with same file_name but different pattern → different sigs."""
    loop = _make_loop()
    sig1 = loop._canonical_dedup_signature("treasury_grep", {
        "file_name": "treasury_bulletin_1986_12.txt", "pattern": "Judiciary",
    })
    sig2 = loop._canonical_dedup_signature("treasury_grep", {
        "file_name": "treasury_bulletin_1986_12.txt", "pattern": "FFO-3",
    })
    assert sig1 != sig2


def test_canonical_sig_same_file_same_pattern_is_duplicate() -> None:
    """Identical file + pattern → same sig."""
    loop = _make_loop()
    sig1 = loop._canonical_dedup_signature("treasury_grep", {
        "file_name": "x.txt", "pattern": "foo",
    })
    sig2 = loop._canonical_dedup_signature("treasury_grep", {
        "file_name": "x.txt", "pattern": "foo",
    })
    assert sig1 == sig2


def test_canonical_sig_arg_order_independent() -> None:
    """Args sorted by key → order-independent."""
    loop = _make_loop()
    sig1 = loop._canonical_dedup_signature("tool", {
        "file_name": "x.txt", "pattern": "foo",
    })
    sig2 = loop._canonical_dedup_signature("tool", {
        "pattern": "foo", "file_name": "x.txt",
    })
    assert sig1 == sig2


def test_canonical_sig_empty_values_omitted() -> None:
    """Empty/None args excluded from signature."""
    loop = _make_loop()
    sig1 = loop._canonical_dedup_signature("tool", {
        "file_name": "x.txt", "pattern": "foo",
    })
    sig2 = loop._canonical_dedup_signature("tool", {
        "file_name": "x.txt", "pattern": "foo", "mode": None, "limit": "",
    })
    assert sig1 == sig2


def test_canonical_sig_query_tool_uses_query_text() -> None:
    """Tools with 'query' key use normalized query text, ignoring other args."""
    loop = _make_loop()
    sig = loop._canonical_dedup_signature("treasury_search", {
        "query": "judiciary outlays 1984", "num_results": "10",
    })
    assert "judiciary outlays 1984" in sig
    assert "num_results" not in sig


def test_canonical_sig_case_insensitive() -> None:
    """Signatures are case-insensitive."""
    loop = _make_loop()
    sig1 = loop._canonical_dedup_signature("treasury_grep", {
        "file_name": "X.txt", "pattern": "Judiciary",
    })
    sig2 = loop._canonical_dedup_signature("treasury_grep", {
        "file_name": "x.txt", "pattern": "judiciary",
    })
    assert sig1 == sig2


def test_canonical_sig_whitespace_normalized() -> None:
    """Extra whitespace in values is collapsed."""
    loop = _make_loop()
    sig1 = loop._canonical_dedup_signature("tool", {
        "pattern": "foo  bar",
    })
    sig2 = loop._canonical_dedup_signature("tool", {
        "pattern": "foo bar",
    })
    assert sig1 == sig2


def test_is_known_duplicate_exact_match() -> None:
    """_is_known_duplicate_sig detects exact matches."""
    loop = _make_loop()
    sig = loop._canonical_dedup_signature("treasury_grep", {
        "file_name": "x.txt", "pattern": "foo",
    })
    loop._step_query_signatures.add(sig)
    assert loop._is_known_duplicate_sig(sig, "treasury_grep") is True


def test_is_known_duplicate_different_tool_not_matched() -> None:
    """Same args but different tool name → not duplicate."""
    loop = _make_loop()
    sig = loop._canonical_dedup_signature("treasury_grep", {
        "file_name": "x.txt", "pattern": "foo",
    })
    loop._step_query_signatures.add(sig)

    sig2 = loop._canonical_dedup_signature("treasury_file_read", {
        "file_name": "x.txt", "pattern": "foo",
    })
    assert loop._is_known_duplicate_sig(sig2, "treasury_file_read") is False


def test_is_known_duplicate_jaccard_near_match() -> None:
    """Jaccard near-duplicate detection for canonical signatures."""
    loop = _make_loop()
    # Register a 6-word signature
    sig1 = "treasury_search:judiciary outlays monthly 1984 fiscal year"
    loop._step_query_signatures.add(sig1)

    # 5/6 overlap → Jaccard = 5/7 = 0.714 — below default 0.8
    sig2 = "treasury_search:judiciary outlays monthly 1984 calendar year budget"
    assert loop._is_known_duplicate_sig(sig2, "treasury_search") is False

    # 5/6 overlap → Jaccard = 5/6 = 0.833 — above 0.8
    sig3 = "treasury_search:judiciary outlays monthly 1984 fiscal"
    assert loop._is_known_duplicate_sig(sig3, "treasury_search") is True


def test_delta_grep_different_patterns_not_jaccard_deduped() -> None:
    """Different patterns on same file should not be Jaccard-deduped.

    Canonical sigs: 'treasury_grep:chunk_type=table file_name=x.txt pattern=judiciary'
    vs 'treasury_grep:chunk_type=table file_name=x.txt pattern=ffo-3'
    Jaccard = 2/4 = 0.5 — well below 0.9 threshold.
    """
    loop = _make_loop()
    loop._jaccard_threshold = 0.9  # as in the benchmark YAML

    sig1 = loop._canonical_dedup_signature("treasury_grep", {
        "file_name": "x.txt", "pattern": "judiciary", "chunk_type": "table",
    })
    loop._step_query_signatures.add(sig1)

    sig2 = loop._canonical_dedup_signature("treasury_grep", {
        "file_name": "x.txt", "pattern": "ffo-3", "chunk_type": "table",
    })
    assert loop._is_known_duplicate_sig(sig2, "treasury_grep") is False
