"""Unit tests for the source-specific query rewriter module.

Tests cover:
- Vector Search multi_query strategy
- Vector Search query2doc strategy
- Genie schema-aware rewriting
- Knowledge Assistant step-back with previous observations
- Fallback behavior on LLM failure and timeout
- Disabled config passthrough
- Unknown source type handling
- Direct strategy for vector search
- None config naive concatenation
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_research.agent.tools.query_rewriter import (
    QueryRewriteConfig,
    RewrittenQuery,
    rewrite_for_source_type,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm_response(content: str, structured: Any = None) -> MagicMock:
    """Build a mock LLMResponse matching the real LLMResponse dataclass.

    The LLMClient.complete() returns an LLMResponse with `.content` (str)
    and `.structured` (parsed Pydantic model or None).
    """
    resp = MagicMock()
    resp.content = content
    resp.structured = structured
    return resp


def _make_mock_llm() -> AsyncMock:
    """Create a mock LLMClient with an async complete() method."""
    mock_llm = AsyncMock()
    return mock_llm


# ---------------------------------------------------------------------------
# Test: VS multi_query returns 3 queries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vs_multi_query_returns_3_queries() -> None:
    """Mock LLM to return JSON with 3 queries; verify alternate_queries has
    3 items (primary + 2 alternates) and strategy_used is 'multi_query'."""
    mock_llm = _make_mock_llm()

    # The rewriter requests structured output (_VSMultiQueryOutput).
    # When structured output is parsed, response.structured is set.
    # Simulate the case where structured is populated.
    structured_obj = MagicMock()
    structured_obj.queries = ["semantic search for docs", "document retrieval methods", "finding relevant passages"]

    mock_llm.complete.return_value = _make_llm_response(
        content='{"queries": ["semantic search for docs", "document retrieval methods", "finding relevant passages"]}',
        structured=structured_obj,
    )

    config = QueryRewriteConfig(enabled=True, strategy="multi_query", max_alternate_queries=3)

    result = await rewrite_for_source_type(
        llm=mock_llm,
        source_type="vector_search",
        step_title="Find document retrieval techniques",
        step_description="Search for modern document retrieval approaches",
        original_query="How do modern search systems retrieve documents?",
        config=config,
    )

    assert isinstance(result, RewrittenQuery)
    assert result.strategy_used == "multi_query"
    # Primary query is the first, alternates are the rest
    assert result.primary_query == "semantic search for docs"
    assert len(result.alternate_queries) == 2
    assert result.alternate_queries[0] == "document retrieval methods"
    assert result.alternate_queries[1] == "finding relevant passages"

    # LLM was called
    mock_llm.complete.assert_called_once()


# ---------------------------------------------------------------------------
# Test: VS query2doc concatenates expansion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vs_query2doc_concatenates_expansion() -> None:
    """Mock LLM to return a pseudo-document passage; verify query2doc
    format: 'original_query\\n\\npseudo_doc'."""
    mock_llm = _make_mock_llm()

    pseudo_doc = "Modern search systems use dense embeddings and approximate nearest neighbor algorithms to find relevant documents efficiently."

    structured_obj = MagicMock()
    structured_obj.passage = pseudo_doc

    mock_llm.complete.return_value = _make_llm_response(
        content=f'{{"passage": "{pseudo_doc}"}}',
        structured=structured_obj,
    )

    config = QueryRewriteConfig(enabled=True, strategy="query2doc")

    original_query = "How do modern search systems work?"
    result = await rewrite_for_source_type(
        llm=mock_llm,
        source_type="vector_search",
        step_title="Understand search systems",
        step_description="Research modern search architectures",
        original_query=original_query,
        config=config,
    )

    assert isinstance(result, RewrittenQuery)
    assert result.strategy_used == "query2doc"
    # query2doc format: original_query + \n\n + pseudo_doc
    assert result.primary_query == f"{original_query}\n\n{pseudo_doc}"
    assert result.alternate_queries == []


# ---------------------------------------------------------------------------
# Test: Genie extracts entities/metrics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_genie_extracts_entities_metrics() -> None:
    """Mock LLM to return a precise data question; verify primary_query
    differs from the naive step_title: step_description concatenation."""
    mock_llm = _make_mock_llm()

    precise_question = "What was the total revenue by product category for Q4 2024?"

    structured_obj = MagicMock()
    structured_obj.question = precise_question

    mock_llm.complete.return_value = _make_llm_response(
        content=f'{{"question": "{precise_question}"}}',
        structured=structured_obj,
    )

    config = QueryRewriteConfig(enabled=True, strategy="schema_aware")

    result = await rewrite_for_source_type(
        llm=mock_llm,
        source_type="genie",
        step_title="Analyze Q4 revenue",
        step_description="Look at revenue breakdown by product for Q4",
        original_query="Tell me about Q4 2024 revenue",
        source_description="Enterprise sales analytics database",
        config=config,
    )

    assert isinstance(result, RewrittenQuery)
    assert result.strategy_used == "schema_aware"
    assert result.primary_query == precise_question
    # The primary_query should differ from the naive concatenation
    naive = "Analyze Q4 revenue: Look at revenue breakdown by product for Q4"
    assert result.primary_query != naive


# ---------------------------------------------------------------------------
# Test: KA includes previous observations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ka_includes_previous_observations() -> None:
    """Mock LLM, pass previous_observations, verify they appear in the
    prompt sent to LLM."""
    mock_llm = _make_mock_llm()

    focused_question = "Based on the identified bottleneck in the data pipeline, what are the recommended optimization strategies?"

    structured_obj = MagicMock()
    structured_obj.question = focused_question

    mock_llm.complete.return_value = _make_llm_response(
        content=f'{{"question": "{focused_question}"}}',
        structured=structured_obj,
    )

    config = QueryRewriteConfig(enabled=True, strategy="step_back")
    observations = [
        "The data pipeline processes 1TB/day with an average latency of 30 minutes",
        "Main bottleneck identified in the transformation stage",
        "Current architecture uses batch processing with Spark",
    ]

    result = await rewrite_for_source_type(
        llm=mock_llm,
        source_type="knowledge_assistant",
        step_title="Optimize data pipeline",
        step_description="Find optimization strategies for the pipeline bottleneck",
        original_query="How can we speed up our data pipeline?",
        config=config,
        previous_observations=observations,
    )

    assert isinstance(result, RewrittenQuery)
    assert result.strategy_used == "step_back"
    assert result.primary_query == focused_question

    # Verify the LLM was called and the prompt includes observations
    mock_llm.complete.assert_called_once()
    call_kwargs = mock_llm.complete.call_args
    messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[0]
    prompt_content = messages[0]["content"]

    # Observations should appear in the prompt (last 3 are included, truncated to 300 chars)
    assert "data pipeline processes 1TB/day" in prompt_content
    assert "Main bottleneck identified" in prompt_content
    assert "batch processing with Spark" in prompt_content


# ---------------------------------------------------------------------------
# Test: Fallback on LLM failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fallback_on_llm_failure() -> None:
    """Mock LLM to raise Exception; verify fallback_on_failure=True returns
    original query with strategy_used='fallback'."""
    mock_llm = _make_mock_llm()
    mock_llm.complete.side_effect = RuntimeError("LLM service unavailable")

    config = QueryRewriteConfig(enabled=True, strategy="multi_query", fallback_on_failure=True)

    result = await rewrite_for_source_type(
        llm=mock_llm,
        source_type="vector_search",
        step_title="Search for documents",
        step_description="Find relevant research papers",
        original_query="What are the latest ML techniques?",
        config=config,
    )

    assert isinstance(result, RewrittenQuery)
    assert result.strategy_used == "fallback"
    # Fallback returns the naive concatenation
    assert result.primary_query == "Search for documents: Find relevant research papers"
    assert result.alternate_queries == []


# ---------------------------------------------------------------------------
# Test: Fallback on LLM failure raises when fallback disabled
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_raises_when_fallback_disabled() -> None:
    """Mock LLM to raise Exception; verify fallback_on_failure=False
    re-raises the exception."""
    mock_llm = _make_mock_llm()
    mock_llm.complete.side_effect = RuntimeError("LLM service unavailable")

    config = QueryRewriteConfig(enabled=True, strategy="direct", fallback_on_failure=False)

    with pytest.raises(RuntimeError, match="LLM service unavailable"):
        await rewrite_for_source_type(
            llm=mock_llm,
            source_type="vector_search",
            step_title="Search for documents",
            step_description="Find relevant research papers",
            original_query="What are the latest ML techniques?",
            config=config,
        )


# ---------------------------------------------------------------------------
# Test: Fallback on timeout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fallback_on_timeout() -> None:
    """Mock LLM to take forever (asyncio.sleep); verify timeout works
    and falls back gracefully."""
    mock_llm = _make_mock_llm()

    async def slow_complete(**kwargs: Any) -> MagicMock:
        await asyncio.sleep(60)  # Way longer than any timeout
        return _make_llm_response(content='{"query": "should not reach here"}')

    mock_llm.complete.side_effect = slow_complete

    # Very short timeout to trigger quickly in tests
    config = QueryRewriteConfig(
        enabled=True,
        strategy="direct",
        timeout_seconds=0.05,
        fallback_on_failure=True,
    )

    result = await rewrite_for_source_type(
        llm=mock_llm,
        source_type="vector_search",
        step_title="Search docs",
        step_description="Find papers",
        original_query="ML techniques",
        config=config,
    )

    assert isinstance(result, RewrittenQuery)
    assert result.strategy_used == "fallback"
    assert result.primary_query == "Search docs: Find papers"


# ---------------------------------------------------------------------------
# Test: Disabled config returns original
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disabled_config_returns_original() -> None:
    """Pass QueryRewriteConfig(enabled=False); verify no LLM call and
    returns original query with strategy_used='disabled'."""
    mock_llm = _make_mock_llm()

    config = QueryRewriteConfig(enabled=False)

    result = await rewrite_for_source_type(
        llm=mock_llm,
        source_type="vector_search",
        step_title="Search for docs",
        step_description="Find relevant papers",
        original_query="What are ML techniques?",
        config=config,
    )

    assert isinstance(result, RewrittenQuery)
    assert result.strategy_used == "disabled"
    assert result.primary_query == "Search for docs: Find relevant papers"
    assert result.alternate_queries == []

    # LLM should NOT be called when disabled
    mock_llm.complete.assert_not_called()


# ---------------------------------------------------------------------------
# Test: Unknown source type returns original
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_source_type_returns_original() -> None:
    """Call rewrite_for_source_type with source_type='unknown'; verify
    returns naive concatenation with strategy_used='unknown_source_type'."""
    mock_llm = _make_mock_llm()

    config = QueryRewriteConfig(enabled=True, strategy="direct")

    result = await rewrite_for_source_type(
        llm=mock_llm,
        source_type="unknown",
        step_title="Some step",
        step_description="Some description",
        original_query="Some query",
        config=config,
    )

    assert isinstance(result, RewrittenQuery)
    assert result.strategy_used == "unknown_source_type"
    assert result.primary_query == "Some step: Some description"
    assert result.alternate_queries == []

    # LLM should NOT be called for unknown source types
    mock_llm.complete.assert_not_called()


# ---------------------------------------------------------------------------
# Test: Direct strategy for vector search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_direct_strategy_for_vector_search() -> None:
    """Test VS with strategy='direct'; verify single query returned
    with no alternates."""
    mock_llm = _make_mock_llm()

    rewritten_query = "A comprehensive overview of modern embedding-based document retrieval systems and their architectures"

    structured_obj = MagicMock()
    structured_obj.query = rewritten_query

    mock_llm.complete.return_value = _make_llm_response(
        content=f'{{"query": "{rewritten_query}"}}',
        structured=structured_obj,
    )

    config = QueryRewriteConfig(enabled=True, strategy="direct")

    result = await rewrite_for_source_type(
        llm=mock_llm,
        source_type="vector_search",
        step_title="Search for retrieval systems",
        step_description="Find info on embedding-based retrieval",
        original_query="How do embedding retrieval systems work?",
        config=config,
    )

    assert isinstance(result, RewrittenQuery)
    assert result.strategy_used == "direct"
    assert result.primary_query == rewritten_query
    assert result.alternate_queries == []

    mock_llm.complete.assert_called_once()


# ---------------------------------------------------------------------------
# Test: config=None returns naive concatenation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_config_none_returns_naive_concatenation() -> None:
    """Call rewrite_for_source_type with config=None; verify it uses
    default QueryRewriteConfig (enabled=True, strategy='direct') and
    makes an LLM call."""
    mock_llm = _make_mock_llm()

    rewritten = "A detailed explanation of machine learning optimization techniques"

    structured_obj = MagicMock()
    structured_obj.query = rewritten

    mock_llm.complete.return_value = _make_llm_response(
        content=f'{{"query": "{rewritten}"}}',
        structured=structured_obj,
    )

    result = await rewrite_for_source_type(
        llm=mock_llm,
        source_type="vector_search",
        step_title="ML optimization",
        step_description="Research optimization techniques",
        original_query="What optimization techniques are used in ML?",
        config=None,  # Explicitly None
    )

    assert isinstance(result, RewrittenQuery)
    # Default config has strategy="direct", enabled=True, so it calls LLM
    assert result.strategy_used == "direct"
    assert result.primary_query == rewritten

    # With None config and known source type, the LLM is still called
    # (default config enables rewriting)
    mock_llm.complete.assert_called_once()


# ---------------------------------------------------------------------------
# Test: config=None with unknown source type returns naive concatenation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_config_none_unknown_source_returns_naive() -> None:
    """Call rewrite_for_source_type with config=None and unknown source;
    verify returns naive concatenation without LLM call."""
    mock_llm = _make_mock_llm()

    result = await rewrite_for_source_type(
        llm=mock_llm,
        source_type="some_new_source_type",
        step_title="Step title",
        step_description="Step description",
        original_query="Original query",
        config=None,
    )

    assert isinstance(result, RewrittenQuery)
    assert result.strategy_used == "unknown_source_type"
    assert result.primary_query == "Step title: Step description"
    mock_llm.complete.assert_not_called()


# ---------------------------------------------------------------------------
# Test: multi_query with query2doc enabled
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vs_multi_query_with_query2doc() -> None:
    """Test multi_query strategy with enable_query2doc=True; verify
    primary query is augmented with pseudo-document expansion."""
    mock_llm = _make_mock_llm()

    call_count = 0

    async def mock_complete(**kwargs: Any) -> MagicMock:
        nonlocal call_count
        call_count += 1
        structured_output = kwargs.get("structured_output")

        if call_count == 1:
            # First call: multi_query generation
            structured = MagicMock()
            structured.queries = ["query one", "query two", "query three"]
            return _make_llm_response(
                content='{"queries": ["query one", "query two", "query three"]}',
                structured=structured,
            )
        else:
            # Second call: query2doc expansion
            structured = MagicMock()
            structured.passage = "This is a pseudo-document about the topic."
            return _make_llm_response(
                content='{"passage": "This is a pseudo-document about the topic."}',
                structured=structured,
            )

    mock_llm.complete.side_effect = mock_complete

    config = QueryRewriteConfig(
        enabled=True,
        strategy="multi_query",
        enable_query2doc=True,
        max_alternate_queries=3,
    )

    result = await rewrite_for_source_type(
        llm=mock_llm,
        source_type="vector_search",
        step_title="Research topic",
        step_description="Find information about the topic",
        original_query="What is the topic about?",
        config=config,
    )

    assert isinstance(result, RewrittenQuery)
    assert result.strategy_used == "multi_query_query2doc"
    # Primary query should have the pseudo-doc appended
    assert "query one" in result.primary_query
    assert "This is a pseudo-document about the topic." in result.primary_query
    assert "\n\n" in result.primary_query
    assert len(result.alternate_queries) == 2
    # LLM was called twice (multi_query + query2doc)
    assert mock_llm.complete.call_count == 2


# ---------------------------------------------------------------------------
# Test: Fallback when structured output parsing fails
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vs_direct_falls_back_to_json_parsing() -> None:
    """When structured output is None, the rewriter falls back to
    parsing the JSON content string directly."""
    mock_llm = _make_mock_llm()

    # structured=None forces fallback to manual JSON parse
    mock_llm.complete.return_value = _make_llm_response(
        content='{"query": "parsed from raw JSON content"}',
        structured=None,
    )

    config = QueryRewriteConfig(enabled=True, strategy="direct")

    result = await rewrite_for_source_type(
        llm=mock_llm,
        source_type="vector_search",
        step_title="Test step",
        step_description="Test description",
        original_query="Test query",
        config=config,
    )

    assert result.primary_query == "parsed from raw JSON content"
    assert result.strategy_used == "direct"


# ---------------------------------------------------------------------------
# Test: Empty step_description uses original_query in naive fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_description_uses_original_query() -> None:
    """When step_description is empty, the naive builder should use
    original_query in the concatenation."""
    mock_llm = _make_mock_llm()

    config = QueryRewriteConfig(enabled=False)

    result = await rewrite_for_source_type(
        llm=mock_llm,
        source_type="vector_search",
        step_title="Find results",
        step_description="",
        original_query="My original query",
        config=config,
    )

    # _build_naive_query returns "step_title: step_description or original_query"
    # When step_description is empty string (falsy), it uses original_query
    assert result.primary_query == "Find results: My original query"
