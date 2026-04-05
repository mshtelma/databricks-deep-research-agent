"""Tests for per-tool call limits in AgentNodeConfig and ReactLoop."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from databricks_deep_research.agents.config import AgentNodeConfig
from databricks_deep_research.agents.react_loop import ReactLoop
from databricks_deep_research.llm.client import ToolCall

# ---------------------------------------------------------------------------
# Config parsing tests
# ---------------------------------------------------------------------------


class TestAgentNodeConfigPerToolLimits:
    """Pydantic validation for per_tool_limits field."""

    def test_valid_limits(self) -> None:
        cfg = AgentNodeConfig(subtype="synthesizer", per_tool_limits={"foo": 3})
        assert cfg.per_tool_limits == {"foo": 3}

    def test_default_none(self) -> None:
        cfg = AgentNodeConfig(subtype="synthesizer")
        assert cfg.per_tool_limits is None

    def test_negative_limit_rejected(self) -> None:
        with pytest.raises(ValidationError, match="non-negative integers"):
            AgentNodeConfig(subtype="synthesizer", per_tool_limits={"foo": -1})

    def test_zero_limit_valid(self) -> None:
        cfg = AgentNodeConfig(subtype="synthesizer", per_tool_limits={"foo": 0})
        assert cfg.per_tool_limits == {"foo": 0}

    def test_multiple_tools(self) -> None:
        cfg = AgentNodeConfig(
            subtype="synthesizer",
            per_tool_limits={"tool_a": 5, "tool_b": 0, "tool_c": 100},
        )
        assert cfg.per_tool_limits == {"tool_a": 5, "tool_b": 0, "tool_c": 100}

    def test_empty_dict_valid(self) -> None:
        cfg = AgentNodeConfig(subtype="synthesizer", per_tool_limits={})
        assert cfg.per_tool_limits == {}


# ---------------------------------------------------------------------------
# ReactLoop per-tool limit enforcement tests
# ---------------------------------------------------------------------------


def _make_mock_tool(name: str) -> MagicMock:
    """Create a mock ResearchTool."""
    tool = MagicMock()
    tool.definition.name = name
    tool.definition.metadata = {}
    tool.definition.parameters = {}
    tool.definition.description = f"mock {name}"
    tool.execute = AsyncMock(return_value=MagicMock(
        content=f"result from {name}",
        sources=[],
        metadata={},
    ))
    return tool


def _make_loop_with_limits(
    per_tool_limits: dict[str, int],
    tool_names: list[str] | None = None,
) -> ReactLoop:
    """Build a ReactLoop with per-tool limits and mock tools."""
    llm = MagicMock()
    if tool_names is None:
        tool_names = list(per_tool_limits.keys())
    tools = [_make_mock_tool(n) for n in tool_names]
    return ReactLoop(
        llm_client=llm,
        tools=tools,
        node_id="test",
        per_tool_limits=per_tool_limits,
    )


class TestReactLoopPerToolLimits:
    """Enforcement of per-tool call limits in _execute_single_tool."""

    def test_calls_within_limit_not_blocked(self) -> None:
        """Calls within the limit are not blocked by per-tool check.

        We pre-set count to 1 with limit=2 — the call should proceed past the
        per-tool gate (even if it fails later due to mock tool execution).
        """
        loop = _make_loop_with_limits({"mock_tool": 2})
        loop._per_tool_counts["mock_tool"] = 1  # 1 < 2, so next call should pass

        tc = ToolCall(id="tc1", function_name="mock_tool", arguments=json.dumps({"q": "a"}))
        result = asyncio.get_event_loop().run_until_complete(
            loop._execute_single_tool(tc, {"q": "a"})
        )

        # Should NOT be blocked by per-tool limit
        assert result[3].get("tool_error", "") != "tool_budget_exhausted:mock_tool"
        # Counter incremented
        assert loop._per_tool_counts["mock_tool"] == 2

    def test_call_exceeding_limit_blocked(self) -> None:
        """Third call to a tool with limit=2 is blocked."""
        loop = _make_loop_with_limits({"mock_tool": 2})
        # Pre-set counts to simulate 2 previous calls
        loop._per_tool_counts["mock_tool"] = 2

        tc = ToolCall(id="tc3", function_name="mock_tool", arguments=json.dumps({"q": "c"}))
        result = asyncio.get_event_loop().run_until_complete(
            loop._execute_single_tool(tc, {"q": "c"})
        )

        # Should be blocked
        assert "budget exhausted" in result[1]
        assert result[3]["tool_error"] == "tool_budget_exhausted:mock_tool"
        assert result[3]["tool_success"] is False
        # Counter still incremented
        assert loop._per_tool_counts["mock_tool"] == 3

    def test_attempts_always_counted(self) -> None:
        """Both successful and blocked attempts increment the counter."""
        loop = _make_loop_with_limits({"mock_tool": 2})
        tc = ToolCall(id="tc1", function_name="mock_tool", arguments=json.dumps({"q": "a"}))

        # Simulate 4 attempts: 2 within limit, 2 blocked
        for i in range(4):
            tc_i = ToolCall(id=f"tc{i}", function_name="mock_tool", arguments=json.dumps({"q": f"q{i}"}))
            asyncio.get_event_loop().run_until_complete(
                loop._execute_single_tool(tc_i, {"q": f"q{i}"})
            )

        assert loop._per_tool_counts["mock_tool"] == 4

    def test_non_limited_tools_unaffected(self) -> None:
        """Tools not in per_tool_limits have no limit."""
        loop = _make_loop_with_limits(
            per_tool_limits={"tool_a": 2},
            tool_names=["tool_a", "tool_b"],
        )

        # Call tool_b 5 times — should never be blocked
        for i in range(5):
            tc = ToolCall(id=f"tc{i}", function_name="tool_b", arguments=json.dumps({"q": f"q{i}"}))
            result = asyncio.get_event_loop().run_until_complete(
                loop._execute_single_tool(tc, {"q": f"q{i}"})
            )
            assert result[3].get("tool_error", "") != "tool_budget_exhausted:tool_b"

        # tool_b should not appear in per_tool_counts
        assert "tool_b" not in loop._per_tool_counts

    def test_limit_zero_blocks_all(self) -> None:
        """Limit=0 blocks the very first call."""
        loop = _make_loop_with_limits({"mock_tool": 0})
        tc = ToolCall(id="tc1", function_name="mock_tool", arguments=json.dumps({"q": "a"}))

        result = asyncio.get_event_loop().run_until_complete(
            loop._execute_single_tool(tc, {"q": "a"})
        )

        assert "budget exhausted" in result[1]
        assert result[3]["tool_error"] == "tool_budget_exhausted:mock_tool"
        assert loop._per_tool_counts["mock_tool"] == 1

    def test_active_tool_names_fires_before_per_tool_limit(self) -> None:
        """Tool restriction from budget guidance fires before per-tool check."""
        loop = _make_loop_with_limits({"compute": 5})
        # Restrict to compute only — treasury_grep is not allowed
        loop._active_tool_names = {"compute"}

        tc = ToolCall(id="tc1", function_name="treasury_grep", arguments=json.dumps({"q": "a"}))
        result = asyncio.get_event_loop().run_until_complete(
            loop._execute_single_tool(tc, {"q": "a"})
        )

        # Blocked by active_tool_names, NOT per-tool limit
        assert "tool_restricted" in result[3].get("tool_error", "")
        # No per-tool count entry for treasury_grep
        assert "treasury_grep" not in loop._per_tool_counts

    def test_no_limits_set(self) -> None:
        """ReactLoop with no per_tool_limits works normally."""
        llm = MagicMock()
        tool = _make_mock_tool("my_tool")
        loop = ReactLoop(llm_client=llm, tools=[tool], node_id="test")

        assert loop._per_tool_limits == {}
        assert loop._per_tool_counts == {}

    def test_constructor_copies_limits(self) -> None:
        """Constructor copies the dict so mutations don't affect original."""
        limits = {"tool_a": 3}
        loop = _make_loop_with_limits(limits)
        loop._per_tool_limits["tool_a"] = 999
        assert limits["tool_a"] == 3  # original unchanged


# ---------------------------------------------------------------------------
# YAML prompt template validation
# ---------------------------------------------------------------------------


def test_officeqa_yaml_prompts_pass_template_validation() -> None:
    """All system_prompt values in the OfficeQA YAML must pass template security validation."""
    from pathlib import Path

    import yaml

    from databricks_deep_research.templates.renderer import SafeTemplateRenderer

    yaml_path = (
        Path(__file__).resolve().parents[4]
        / "benchmarks"
        / "officeqa"
        / "workflow-v7-with-tools.yaml"
    )
    if not yaml_path.exists():
        pytest.skip("OfficeQA YAML not found")

    with open(yaml_path) as f:
        defn = yaml.safe_load(f)

    renderer = SafeTemplateRenderer()

    def _collect_prompts(node: dict, prompts: list) -> None:  # type: ignore[type-arg]
        cfg = node.get("config", {})
        for key in ("system_prompt", "user_prompt_template"):
            if key in cfg and cfg[key]:
                prompts.append((node.get("id", "unknown"), key, cfg[key]))
        for child in node.get("children", []):
            _collect_prompts(child, prompts)

    prompts: list[tuple[str, str, str]] = []
    _collect_prompts(defn.get("root", {}), prompts)
    assert len(prompts) > 0, "No prompts found — YAML structure may have changed"

    for node_id, key, prompt in prompts:
        renderer.render(prompt, {})  # _validate runs inside render; missing vars → empty string
