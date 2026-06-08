"""Tests for the agent execution harness."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from databricks_deep_research.agents.builtins.registry import get_builtin
from databricks_deep_research.agents.config import (
    AgentNodeConfig,
    PoolInjectConfig,
    PoolWriteConfig,
)
from databricks_deep_research.agents.harness import (
    _build_input,
    _build_messages,
    _compute_citation_stats,
    _extract_pool_items,
    _normalize_research_output,
    _parse_output,
    _serialize_for_context,
    _serialize_source_for_pool,
    execute_agent,
)
from databricks_deep_research.agents.isolation import AgentInput
from databricks_deep_research.agents.output_models import BackgroundOutput
from databricks_deep_research.agents.prompt_context import CompiledPoolSection
from databricks_deep_research.agents.prompts.planner import (
    PLANNER_USER_PROMPT,
    SOURCE_AWARE_PLANNER_NO_LANDSCAPE_PROMPT,
    SOURCE_AWARE_PLANNER_USER_PROMPT,
)
from databricks_deep_research.agents.react_loop import ToolCallCache
from databricks_deep_research.errors import WorkflowError
from databricks_deep_research.llm.client import LLMResponse
from databricks_deep_research.pools.pool_state import PoolConfig, PoolState
from databricks_deep_research.templates.renderer import SafeTemplateRenderer
from databricks_deep_research.tools.protocol import SourceInfo
from databricks_deep_research.workflow.state import WorkflowState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides: Any) -> AgentNodeConfig:
    defaults: dict[str, Any] = {
        "subtype": "researcher",
        "model_tier": "analytical",
        "system_prompt": "You are a helper.",
        "user_prompt_template": "Research: {query}",
        "input_keys": [],
        "output_key": "findings",
        "output_format": "text",
    }
    defaults.update(overrides)
    return AgentNodeConfig(**defaults)


def _make_state(query: str = "test query") -> WorkflowState:
    return WorkflowState(query=query)


def _mock_llm(content: str = "LLM says hello", usage: dict[str, int] | None = None) -> MagicMock:
    client = MagicMock()
    resp = LLMResponse(content=content, usage=usage or {"total_tokens": 42}, structured=None)
    client.complete = AsyncMock(return_value=resp)
    return client


def _make_pool(name: str = "sources") -> PoolState:
    return PoolState(PoolConfig(name=name, dedup_content_hash=False))


# ---------------------------------------------------------------------------
# 0. citation stats
# ---------------------------------------------------------------------------

def test_compute_citation_stats_counts_markdown_numeric_markers(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO", logger="databricks_deep_research.agents.harness")
    report = "A grounded report cites source one [1] and source three [3], but not [99]."

    _compute_citation_stats(report, total_sources=3)

    assert "CITATION_STATS total=3 valid=2 invalid=1 coverage=100.0% fields=1/1" in caplog.text


# ---------------------------------------------------------------------------
# 1. execute_agent — simple LLM call (no tools)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_agent_simple_call_writes_state() -> None:
    config = _make_config(subtype="custom", output_format="text", output_key="findings")
    state = _make_state()
    llm = _mock_llm("result text")

    output = await execute_agent("node-1", config, state, llm, tools=[], pools={})

    assert output.content == "result text"
    assert output.output_key == "findings"
    assert state.get("findings") == "result text"
    assert any(e.event_type == "agent_output" for e in output.events)


# ---------------------------------------------------------------------------
# 2. execute_agent — tools trigger ReAct loop
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_agent_with_tools_uses_react_loop() -> None:
    config = _make_config(subtype="custom", output_format="text", max_tool_calls=5)
    state = _make_state()
    llm = _mock_llm()

    mock_tool = MagicMock()
    mock_tool.definition = MagicMock(name="web_search")

    with patch("databricks_deep_research.agents.harness.ReactLoop") as MockLoop:
        mock_result = MagicMock()
        mock_result.content = "react output"
        mock_result.events = []
        mock_result.sources = []
        mock_result.token_usage = {"total_tokens": 10}
        MockLoop.return_value.execute = AsyncMock(return_value=mock_result)

        output = await execute_agent("node-2", config, state, llm, tools=[mock_tool], pools={})

    assert output.content == "react output"
    MockLoop.assert_called_once()
    MockLoop.return_value.execute.assert_awaited_once()


# ---------------------------------------------------------------------------
# 3. _build_input — resolves input_keys from state
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_build_input_resolves_keys() -> None:
    config = _make_config(input_keys=["plan", "coordination.complexity"])
    state = _make_state("my query")
    state.append("planner", "plan", {"steps": [1, 2]})
    state.append("coord", "coordination", MagicMock(complexity="deep"))

    agent_input, _ = await _build_input("n1", config, state, _mock_llm(), tools=[], pools={})

    assert agent_input.query == "my query"
    assert agent_input.context["plan"] == {"steps": [1, 2]}
    assert agent_input.context["coordination.complexity"] == "deep"


@pytest.mark.asyncio
async def test_build_input_runtime_context_overrides_state() -> None:
    config = _make_config(input_keys=["query", "iteration", "completed_steps"])
    state = _make_state("my query")
    state.append("planner", "iteration", "1")

    agent_input, _ = await _build_input(
        "n1",
        config,
        state,
        _mock_llm(),
        tools=[],
        pools={},
        runtime_context={"iteration": "2", "completed_steps": "- Step 1: done"},
    )

    assert agent_input.query == "my query"
    assert agent_input.context["iteration"] == "2"
    assert agent_input.context["completed_steps"] == "- Step 1: done"
    assert state.get("completed_steps") is None


# ---------------------------------------------------------------------------
# 4. _build_messages — constructs OpenAI-format messages
# ---------------------------------------------------------------------------

def test_build_messages_system_and_user() -> None:
    inp = AgentInput(
        query="q",
        system_prompt="Be concise.",
        user_prompt="Tell me about X",
    )
    msgs = _build_messages(inp)

    assert msgs[0] == {"role": "system", "content": "Be concise."}
    assert msgs[1]["role"] == "user"
    assert "Tell me about X" in msgs[1]["content"]


def test_build_messages_includes_pool_content() -> None:
    inp = AgentInput(
        query="q",
        user_prompt="Research",
        pool_sections={
            "sources": CompiledPoolSection(
                pool_name="sources",
                rendered_text="- src1\n- src2",
                format="markdown",
                raw_items=["src1", "src2"],
                rendered_items=["src1", "src2"],
            )
        },
    )
    msgs = _build_messages(inp)
    user_msg = msgs[0]["content"]
    assert "## sources" in user_msg
    assert "src1" in user_msg


def test_build_messages_no_system_prompt() -> None:
    inp = AgentInput(query="q", user_prompt="just this")
    msgs = _build_messages(inp)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"


# ---------------------------------------------------------------------------
# 5. _parse_output — text, JSON, markdown-wrapped JSON
# ---------------------------------------------------------------------------

def test_parse_output_text_passthrough() -> None:
    config = _make_config(output_format="text")
    assert _parse_output("hello", config) == "hello"


def test_parse_output_json() -> None:
    config = _make_config(output_format="json")
    assert _parse_output('{"a": 1}', config) == {"a": 1}


def test_parse_output_markdown_json() -> None:
    config = _make_config(output_format="json")
    raw = 'Some text\n```json\n{"key": "val"}\n```\nmore text'
    assert _parse_output(raw, config) == {"key": "val"}


def test_parse_output_markdown_json_unclosed_codeblock() -> None:
    """Unclosed code block must not crash — falls through to json_repair."""
    config = _make_config(output_format="json")
    raw = 'Some text\n```json\n{"key": "val"}\nno closing backticks'
    result = _parse_output(raw, config)
    assert result is not None


def test_parse_output_markdown_json_multiple_blocks() -> None:
    """Multiple code blocks — first valid one should be extracted."""
    config = _make_config(output_format="json")
    raw = 'Intro\n```json\n{"a": 1}\n```\nextra\n```json\n{"b": 2}\n```'
    assert _parse_output(raw, config) == {"a": 1}


def test_serialize_source_for_pool_preserves_full_content() -> None:
    """Source pool admission should keep full tool content, not only snippets."""
    item = _serialize_source_for_pool(
        SourceInfo(
            url="enterprise://vector_search/index/1",
            title="Quarterly earnings",
            snippet="Short snippet",
            content="Full chunk text with multiple paragraphs and supporting figures.",
            source_type="vector_search",
            relevance_score=0.91,
        )
    )

    assert item["snippet"] == "Short snippet"
    assert item["content"] == "Full chunk text with multiple paragraphs and supporting figures."
    assert item["relevance_score"] == pytest.approx(0.91)


def test_parse_output_non_string_passthrough() -> None:
    config = _make_config(output_format="json")
    obj = {"already": "parsed"}
    assert _parse_output(obj, config) is obj


# ---------------------------------------------------------------------------
# 6. _extract_pool_items — dot-path navigation
# ---------------------------------------------------------------------------

def test_extract_pool_items_simple_list() -> None:
    pw = PoolWriteConfig(pool="sources", extract="sources")
    output = {"sources": ["a", "b"]}
    assert _extract_pool_items(output, pw) == ["a", "b"]


def test_extract_pool_items_nested_path() -> None:
    pw = PoolWriteConfig(pool="claims", extract="analysis.claims")
    output = {"analysis": {"claims": [1, 2, 3]}}
    assert _extract_pool_items(output, pw) == [1, 2, 3]


def test_extract_pool_items_missing_key() -> None:
    pw = PoolWriteConfig(pool="x", extract="missing.path")
    assert _extract_pool_items({"other": 1}, pw) == []


def test_extract_pool_items_scalar_wrapped() -> None:
    pw = PoolWriteConfig(pool="x", extract="val")
    assert _extract_pool_items({"val": "single"}, pw) == ["single"]


# ---------------------------------------------------------------------------
# 7. Auto-detection of input_keys from prompt templates
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_auto_detect_input_keys_from_prompts() -> None:
    """When input_keys is empty, harness detects keys from prompt templates."""
    config = _make_config(
        subtype="custom",
        system_prompt="You are helpful.",
        user_prompt_template="Query: {query}\nHistory: {conversation_history}",
        input_keys=[],
    )
    state = _make_state()
    llm = _mock_llm("ok")

    output = await execute_agent("node-auto", config, state, llm, tools=[], pools={})

    # Should succeed — auto-detection resolves keys from templates
    assert output.content == "ok"


@pytest.mark.asyncio
async def test_explicit_input_keys_are_augmented_by_prompt_detection() -> None:
    """Prompt variables are merged into explicit input_keys to avoid stale configs."""
    config = _make_config(
        subtype="custom",
        user_prompt_template="{query} {extra}",
        input_keys=["query"],
    )
    state = _make_state()
    state.append("ctx", "extra", "from state")
    llm = _mock_llm("ok")

    output = await execute_agent("node-explicit", config, state, llm, tools=[], pools={})

    assert output.content == "ok"
    llm.complete.assert_awaited_once()
    messages = llm.complete.await_args.args[0]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "from state" in messages[1]["content"]


@pytest.mark.asyncio
async def test_runtime_context_is_rendered_into_prompts() -> None:
    config = _make_config(
        subtype="custom",
        user_prompt_template="Iteration {iteration}\n{completed_steps}",
        input_keys=[],
    )
    state = _make_state()
    llm = _mock_llm("ok")

    output = await execute_agent(
        "node-runtime",
        config,
        state,
        llm,
        tools=[],
        pools={},
        runtime_context={"iteration": "2", "completed_steps": "- Step 1: done"},
    )

    assert output.content == "ok"
    messages = llm.complete.await_args.args[0]
    assert "Iteration 2" in messages[1]["content"]
    assert "- Step 1: done" in messages[1]["content"]


@pytest.mark.asyncio
async def test_auto_detect_noop_when_no_prompts() -> None:
    """Auto-detection is skipped when both prompts are empty."""
    config = _make_config(
        subtype="custom",
        system_prompt="",
        user_prompt_template="",
        input_keys=[],
    )
    state = _make_state()
    llm = _mock_llm("ok")

    output = await execute_agent("node-empty", config, state, llm, tools=[], pools={})

    assert output.content == "ok"


# ---------------------------------------------------------------------------
# 8. Pool writes are executed correctly
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_agent_pool_writes() -> None:
    pw = PoolWriteConfig(pool="sources", extract="sources")
    config = _make_config(output_format="json", pool_writes=[pw])
    state = _make_state()
    llm = _mock_llm(
        '{"sources": ['
        '{"url": "https://example.com/1", "snippet": "Evidence one."},'
        '{"url": "https://example.com/2", "snippet": "Evidence two."}'
        "]}"
    )
    pool = _make_pool("sources")

    output = await execute_agent("n3", config, state, llm, tools=[], pools={"sources": pool})

    assert pool.count() == 2
    assert output.pool_writes["sources"] == [
        {"url": "https://example.com/1", "snippet": "Evidence one."},
        {"url": "https://example.com/2", "snippet": "Evidence two."},
    ]
    assert pool.items == output.pool_writes["sources"]


@pytest.mark.asyncio
async def test_execute_agent_pool_writes_preserve_source_metadata() -> None:
    """ReAct source fallback should retain source_type and content for persistence."""
    config = _make_config(max_tool_calls=5, pool_writes=[PoolWriteConfig(pool="sources", extract="sources")])
    state = _make_state()
    llm = _mock_llm()
    pool = _make_pool("sources")
    mock_tool = MagicMock()
    mock_tool.definition = MagicMock(name="enterprise_search")

    with patch("databricks_deep_research.agents.harness.ReactLoop") as MockLoop:
        mock_result = MagicMock()
        mock_result.content = "react output"
        mock_result.events = []
        mock_result.sources = [SimpleNamespace(
            url="vs://main.finance_docs/doc-1",
            title="Quarterly Earnings",
            snippet="Revenue grew 10 percent year over year.",
            source_type="vector_search",
            content="Revenue grew 10 percent year over year.",
            relevance_score=0.91,
        )]
        mock_result.token_usage = {"total_tokens": 10}
        MockLoop.return_value.execute = AsyncMock(return_value=mock_result)

        output = await execute_agent("node-2", config, state, llm, tools=[mock_tool], pools={"sources": pool})

    assert output.pool_writes["sources"] == [{
        "url": "vs://main.finance_docs/doc-1",
        "title": "Quarterly Earnings",
        "snippet": "Revenue grew 10 percent year over year.",
        "source_type": "vector_search",
        "content": "Revenue grew 10 percent year over year.",
        "relevance_score": 0.91,
    }]
    assert pool.items == output.pool_writes["sources"]


# ---------------------------------------------------------------------------
# 9. Pool injection respects max_items config
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pool_inject_max_items() -> None:
    """Pool injection respects max_items config."""
    pi = PoolInjectConfig(pool="obs", threshold=0, max_items=2)
    config = _make_config(pool_inject=[pi])
    pool = _make_pool("obs")
    for i in range(5):
        pool.add(f"item {i}")

    agent_input, _ = await _build_input(
        "n1",
        config,
        _make_state(),
        _mock_llm(),
        tools=[],
        pools={"obs": pool},
    )

    assert len(agent_input.pool_sections["obs"].raw_items) == 2


# ---------------------------------------------------------------------------
# 10. Pool injection truncates items when max_item_chars > 0
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pool_inject_max_item_chars() -> None:
    """Pool injection truncates items when max_item_chars > 0."""
    pi = PoolInjectConfig(pool="obs", threshold=0, max_item_chars=10)
    config = _make_config(pool_inject=[pi])
    pool = _make_pool("obs")
    pool.add("a" * 100)

    agent_input, _ = await _build_input(
        "n1",
        config,
        _make_state(),
        _mock_llm(),
        tools=[],
        pools={"obs": pool},
    )

    rendered = agent_input.pool_sections["obs"].rendered_text
    assert "aaaaaaaaaa..." in rendered


@pytest.mark.asyncio
async def test_pool_inject_no_truncation_when_zero() -> None:
    """Pool injection does not truncate when max_item_chars is 0 (default)."""
    pi = PoolInjectConfig(pool="obs", threshold=0, max_item_chars=0)
    config = _make_config(pool_inject=[pi])
    pool = _make_pool("obs")
    long_text = "x" * 1000
    pool.add(long_text)

    agent_input, _ = await _build_input(
        "n1",
        config,
        _make_state(),
        _mock_llm(),
        tools=[],
        pools={"obs": pool},
    )

    assert agent_input.pool_sections["obs"].rendered_text == long_text


@pytest.mark.asyncio
async def test_pool_inject_threshold_uses_scored_search() -> None:
    pi = PoolInjectConfig(pool="obs", threshold=0.5, max_items=5)
    config = _make_config(pool_inject=[pi])
    pool = _make_pool("obs")
    pool.add("cats are agile")
    pool.add("dogs are loyal")

    agent_input, _ = await _build_input(
        "n1",
        config,
        _make_state("cats"),
        _mock_llm(),
        tools=[],
        pools={"obs": pool},
    )

    section = agent_input.pool_sections["obs"]
    assert section.raw_items == ["cats are agile"]
    assert "dogs are loyal" not in section.rendered_text


@pytest.mark.asyncio
async def test_pool_inject_json_format_renders_json() -> None:
    pi = PoolInjectConfig(pool="obs", format="json", max_items=2)
    config = _make_config(pool_inject=[pi])
    pool = _make_pool("obs")
    pool.add({"title": "A", "value": 1})

    agent_input, _ = await _build_input(
        "n1",
        config,
        _make_state(),
        _mock_llm(),
        tools=[],
        pools={"obs": pool},
    )

    rendered = agent_input.pool_sections["obs"].rendered_text
    assert rendered.startswith("[")
    assert '"value": 1' in rendered


# ---------------------------------------------------------------------------
# 11. Shared tool_call_cache forwarded to ReactLoop
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_agent_passes_shared_cache() -> None:
    """Shared ToolCallCache is forwarded to ReactLoop."""
    cache = ToolCallCache()
    config = _make_config(max_tool_calls=5)
    state = _make_state()
    llm = _mock_llm()

    mock_tool = MagicMock()
    mock_tool.definition = MagicMock(name="web_search")

    with patch("databricks_deep_research.agents.harness.ReactLoop") as MockLoop:
        mock_result = MagicMock()
        mock_result.content = "react output"
        mock_result.events = []
        mock_result.sources = []
        mock_result.token_usage = {"total_tokens": 10}
        MockLoop.return_value.execute = AsyncMock(return_value=mock_result)

        await execute_agent(
            "node-cache", config, state, llm,
            tools=[mock_tool], pools={},
            tool_call_cache=cache,
        )

    # Verify cache was passed to ReactLoop constructor
    call_kwargs = MockLoop.call_args
    assert call_kwargs.kwargs.get("cache") is cache


# ---------------------------------------------------------------------------
# 12. Fix 3: Background variable resolved in planner prompt
# ---------------------------------------------------------------------------

def test_background_variable_resolved_in_planner_prompt() -> None:
    """Planner prompt uses {background}, not {background_results}."""
    renderer = SafeTemplateRenderer()
    variables = renderer.extract_variables(PLANNER_USER_PROMPT)
    assert "background" in variables
    assert "background_results" not in variables


def test_source_aware_planner_uses_background_variable() -> None:
    """Source-aware planner prompt uses {background}, not {background_results}."""
    renderer = SafeTemplateRenderer()
    for prompt in [SOURCE_AWARE_PLANNER_USER_PROMPT, SOURCE_AWARE_PLANNER_NO_LANDSCAPE_PROMPT]:
        variables = renderer.extract_variables(prompt)
        assert "background" in variables
        assert "background_results" not in variables


# ---------------------------------------------------------------------------
# 13. Fix 12: Context serialization
# ---------------------------------------------------------------------------

def test_serialize_for_context_string() -> None:
    """Strings pass through unchanged."""
    assert _serialize_for_context("hello") == "hello"


def test_serialize_for_context_none() -> None:
    """None returns empty string."""
    assert _serialize_for_context(None) == ""


def test_serialize_for_context_dict() -> None:
    """Dicts serialize to JSON."""
    result = _serialize_for_context({"key": "value"})
    assert '"key"' in result
    assert '"value"' in result


def test_serialize_for_context_pydantic() -> None:
    """Pydantic models serialize to JSON, not repr."""
    output = BackgroundOutput(summary="Test summary", query_decomposition=["sub1"])
    result = _serialize_for_context(output)
    assert "Test summary" in result
    assert isinstance(result, str)
    # Should be JSON, not Python repr
    assert "BackgroundOutput(" not in result


def test_serialize_for_context_list() -> None:
    """Lists serialize to JSON."""
    result = _serialize_for_context([{"a": 1}])
    assert '"a"' in result


def test_planner_builtin_uses_source_aware_prompt_when_landscape_present() -> None:
    """Planner switches to the source-aware prompt when background discovery exists."""
    config = AgentNodeConfig(subtype="planner", input_keys=[])
    state = _make_state()
    state.append("background", "data_landscape", {"sources": [{"source_name": "earnings"}]})

    builtin = get_builtin("planner")
    assert builtin is not None
    assert builtin.enrich_config is not None
    enriched = builtin.enrich_config(config, state, None)

    assert "source_hints" in enriched.user_prompt_template
    assert "Data Landscape Summary" in enriched.user_prompt_template


def test_planner_builtin_uses_source_aware_prompt_when_sources_are_available() -> None:
    config = AgentNodeConfig(subtype="planner", input_keys=[])
    state = _make_state()

    builtin = get_builtin("planner")
    assert builtin is not None
    assert builtin.enrich_config is not None
    enriched = builtin.enrich_config(
        config,
        state,
        {"available_sources": "- vector_search [vector_search/vector_index]: Internal docs"},
    )

    assert "Available Source Catalog" in enriched.user_prompt_template
    assert "source_hints" in enriched.user_prompt_template


@pytest.mark.asyncio
async def test_background_react_output_is_enriched_with_discovered_sources() -> None:
    """Background nodes keep structured discovery data even in ReAct mode."""
    config = _make_config(
        subtype="background",
        output_key="background",
        output_format="json",
        max_tool_calls=5,
    )
    state = _make_state()
    llm = _mock_llm()
    pool = _make_pool("discovery_sources")
    mock_tool = MagicMock()
    mock_tool.definition = MagicMock(name="vector_search")

    with patch("databricks_deep_research.agents.harness.ReactLoop") as MockLoop:
        mock_result = MagicMock()
        mock_result.content = '{"summary": "Kroger docs found", "query_decomposition": ["earnings"]}'
        mock_result.events = []
        mock_result.sources = [SimpleNamespace(
            url="vs://earnings/doc-1",
            title="Kroger Reports Fourth Quarter and Full-Year 2024 Results",
            snippet="Revenue and guidance details.",
            source_type="vector_search",
            content="Revenue and guidance details.",
        )]
        mock_result.token_usage = {"total_tokens": 10}
        MockLoop.return_value.execute = AsyncMock(return_value=mock_result)

        output = await execute_agent(
            "background-node",
            config,
            state,
            llm,
            tools=[mock_tool],
            pools={"discovery_sources": pool},
        )

    assert isinstance(output.content, dict)
    assert output.content["summary"] == "Kroger docs found"
    assert output.content["discovered_sources"][0]["title"].startswith("Kroger Reports")
    assert output.content["data_landscape"]["sources"][0]["source_type"] == "vector_search"


@pytest.mark.asyncio
async def test_researcher_structured_output_preserves_output_key_for_pool_writes() -> None:
    """Structured researcher output should still satisfy extract=<output_key> pool writes."""
    config = _make_config(
        subtype="researcher",
        output_key="web_findings",
        output_format="json",
        max_tool_calls=5,
        pool_writes=[
            PoolWriteConfig(pool="observations", extract="web_findings"),
            PoolWriteConfig(pool="sources", extract="sources"),
        ],
    )
    state = _make_state()
    llm = _mock_llm()
    observations = _make_pool("observations")
    sources_pool = _make_pool("sources")
    mock_tool = MagicMock()
    mock_tool.definition = MagicMock(name="web_search")

    with patch("databricks_deep_research.agents.harness.ReactLoop") as MockLoop:
        mock_result = MagicMock()
        mock_result.content = (
            '{"observation": "Web findings about scalable AI systems.", '
            '"search_queries": ["scalable AI systems architecture"]}'
        )
        mock_result.events = []
        mock_result.sources = [SimpleNamespace(
            url="https://example.com/ai-systems",
            title="Scalable AI Systems",
            snippet="Best practices for scalable AI platforms.",
            source_type="web",
        )]
        mock_result.token_usage = {"total_tokens": 10}
        MockLoop.return_value.execute = AsyncMock(return_value=mock_result)

        output = await execute_agent(
            "researcher-web",
            config,
            state,
            llm,
            tools=[mock_tool],
            pools={"observations": observations, "sources": sources_pool},
        )

    # Normalizer now serializes the full dict (not just the "observation" key)
    assert "Web findings about scalable AI systems." in output.content
    assert "observation" in output.content  # Full dict includes original keys
    assert observations.count() == 1
    assert "Web findings about scalable AI systems." in observations.get_recent(1)[0]
    assert sources_pool.count() == 1


@pytest.mark.asyncio
async def test_researcher_malformed_json_blocks_and_preserves_tool_sources() -> None:
    config = _make_config(
        subtype="researcher",
        output_key="web_findings",
        output_format="json",
        max_tool_calls=5,
        pool_writes=[
            PoolWriteConfig(pool="observations", extract="web_findings"),
            PoolWriteConfig(pool="sources", extract="sources"),
        ],
    )
    state = _make_state()
    llm = _mock_llm()
    observations = _make_pool("observations")
    sources_pool = _make_pool("sources")
    mock_tool = MagicMock()
    mock_tool.definition = MagicMock(name="web_search")

    with patch("databricks_deep_research.agents.harness.ReactLoop") as MockLoop:
        mock_result = MagicMock()
        mock_result.content = "This is a malformed answer, but it still contains substantive findings from web research and should not be discarded."
        mock_result.events = []
        mock_result.sources = [SimpleNamespace(
            url="https://example.com/source",
            title="Example Source",
            snippet="Important snippet.",
            source_type="web",
        )]
        mock_result.token_usage = {"total_tokens": 10}
        MockLoop.return_value.execute = AsyncMock(return_value=mock_result)

        output = await execute_agent(
            "researcher-web",
            config,
            state,
            llm,
            tools=[mock_tool],
            pools={"observations": observations, "sources": sources_pool},
        )

    # Malformed content may be wrapped in a dict by the harness; the key point
    # is that the substantive text is preserved and sources are collected.
    assert "This is a malformed answer" in output.content
    assert observations.count() == 1
    assert sources_pool.count() == 1


@pytest.mark.asyncio
async def test_researcher_empty_structured_output_keeps_sources_only() -> None:
    config = _make_config(
        subtype="researcher",
        output_key="web_findings",
        output_format="json",
        max_tool_calls=5,
        pool_writes=[
            PoolWriteConfig(pool="observations", extract="web_findings"),
            PoolWriteConfig(pool="sources", extract="sources"),
        ],
    )
    state = _make_state()
    llm = _mock_llm()
    observations = _make_pool("observations")
    sources_pool = _make_pool("sources")
    mock_tool = MagicMock()
    mock_tool.definition = MagicMock(name="web_search")

    with patch("databricks_deep_research.agents.harness.ReactLoop") as MockLoop:
        mock_result = MagicMock()
        mock_result.content = "{}"
        mock_result.events = []
        mock_result.sources = [SimpleNamespace(
            url="https://example.com/source-only",
            title="Source Only",
            snippet="Snippet only.",
            source_type="web",
        )]
        mock_result.token_usage = {"total_tokens": 10}
        MockLoop.return_value.execute = AsyncMock(return_value=mock_result)

        output = await execute_agent(
            "researcher-web",
            config,
            state,
            llm,
            tools=[mock_tool],
            pools={"observations": observations, "sources": sources_pool},
        )

    # Normalizer serializes the full dict; sources are still collected separately
    assert output.content  # non-empty — full dict serialization
    assert observations.count() == 1
    assert sources_pool.count() == 1


@pytest.mark.asyncio
async def test_planner_malformed_json_raises_workflow_error() -> None:
    config = _make_config(
        subtype="planner",
        output_key="plan",
        output_format="json",
        max_tool_calls=0,
        input_keys=["query"],
    )
    state = _make_state()
    llm = _mock_llm("```not-json")

    with pytest.raises(WorkflowError, match="Malformed structured output for planner"):
        await execute_agent("planner-node", config, state, llm, tools=[], pools={})


@pytest.mark.asyncio
async def test_researcher_empty_structured_output_synthesizes_observation_from_sources() -> None:
    config = _make_config(
        subtype="researcher",
        output_key="web_findings",
        output_format="json",
        max_tool_calls=5,
        pool_writes=[PoolWriteConfig(pool="observations", extract="web_findings")],
    )
    state = _make_state()
    llm = _mock_llm()
    observations = _make_pool("observations")
    mock_tool = MagicMock()
    mock_tool.definition = MagicMock(name="web_search")

    with patch("databricks_deep_research.agents.harness.ReactLoop") as MockLoop:
        mock_result = MagicMock()
        mock_result.content = "{}"
        mock_result.events = []
        mock_result.sources = [SimpleNamespace(
            url="https://example.com/article",
            title="Kroger earnings outlook",
            snippet="Kroger reported updated guidance and recent quarterly performance.",
            source_type="web",
        )]
        mock_result.token_usage = {"total_tokens": 10}
        MockLoop.return_value.execute = AsyncMock(return_value=mock_result)

        output = await execute_agent(
            "researcher-web",
            config,
            state,
            llm,
            tools=[mock_tool],
            pools={"observations": observations},
        )

    # Full dict serialization — content includes source data
    assert output.content  # non-empty
    assert observations.count() == 1


def test_normalize_research_output_source_backed() -> None:
    config = _make_config(subtype="researcher", output_key="findings")
    parsed = {
        "findings": "",
        "observation": "",
        "sources": [
            {
                "url": "enterprise://doc/1",
                "title": "Doc 1",
                "snippet": "Enterprise source with substantive evidence.",
            }
        ],
    }
    normalized = _normalize_research_output(parsed, config, [])
    assert normalized is not None
    assert normalized.sources
    assert normalized.state_text  # full dict serialized


def test_normalize_research_output_merges_tool_sources() -> None:
    config = _make_config(subtype="researcher", output_key="findings")
    parsed = {"findings": "", "observation": "", "sources": []}
    tool_sources = [
        {
            "url": "enterprise://doc/2",
            "title": "Doc 2",
            "snippet": "Tool source with substantive evidence.",
        }
    ]

    normalized = _normalize_research_output(parsed, config, tool_sources)

    assert normalized is not None
    assert normalized.sources == tool_sources
    assert normalized.state_text  # full dict serialized
    assert normalized.skip_source_writes is False


@pytest.mark.asyncio
async def test_execute_agent_empty_structured_output_writes_tool_sources_to_pool() -> None:
    config = _make_config(
        subtype="researcher",
        output_key="findings",
        output_format="json",
        max_tool_calls=5,
        pool_writes=[PoolWriteConfig(pool="sources", extract="sources")],
    )
    state = _make_state()
    llm = _mock_llm()
    sources_pool = _make_pool("sources")
    mock_tool = MagicMock()
    mock_tool.definition = MagicMock(name="vector_search")

    with patch("databricks_deep_research.agents.harness.ReactLoop") as MockLoop:
        mock_result = MagicMock()
        mock_result.content = '{"findings": "", "observation": "", "sources": []}'
        mock_result.events = []
        mock_result.sources = [SimpleNamespace(
            url="enterprise://warehouse/docs/1",
            title="Architecture Runbook",
            snippet="Internal architecture evidence.",
            source_type="vector_search",
        )]
        mock_result.token_usage = {"total_tokens": 10}
        MockLoop.return_value.execute = AsyncMock(return_value=mock_result)

        output = await execute_agent(
            "researcher-enterprise",
            config,
            state,
            llm,
            tools=[mock_tool],
            pools={"sources": sources_pool},
        )

    assert sources_pool.count() == 1
    assert output.pool_writes["sources"][0]["url"] == "enterprise://warehouse/docs/1"


@pytest.mark.asyncio
async def test_researcher_unparsed_substantive_text_reaches_observation_fallback() -> None:
    config = _make_config(
        subtype="researcher",
        output_key="findings",
        output_format="json",
        max_tool_calls=0,
        pool_writes=[PoolWriteConfig(pool="observations", extract="findings")],
    )
    state = _make_state()
    observations = _make_pool("observations")
    llm = _mock_llm("This answer is malformed JSON but contains substantive research findings that should still be persisted to the observations pool.")

    output = await execute_agent(
        "researcher-text",
        config,
        state,
        llm,
        tools=[],
        pools={"observations": observations},
    )

    assert "This answer is malformed JSON" in output.content
    assert observations.count() == 1
