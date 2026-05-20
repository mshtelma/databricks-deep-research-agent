"""Tests for the Designer framework-tool wrappers (US-08).

Covers the wrapped designer tools, the structural gate, the iter-2
``parse_architect_ast`` extractor, and the loop's ``extract_critic_approved``
flattener — plus registry registration so framework workflows can resolve
these tools by name.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from databricks_deep_research.tools.protocol import ToolContext
from databricks_deep_research.tools.registry import ToolRegistry
from databricks_deep_research.workflow.loader import load_workflow_from_dict

from deep_research.agent_designer.critic_types import CriticVerdict
from deep_research.agent_designer.framework_tools import (
    AddBlockTool,
    BindToolToBlockTool,
    DeclareToolTool,
    ExtractCriticApprovedTool,
    ParseArchitectAstTool,
    ProposeWorkflowTool,
    SetModelTierTool,
    UpdateBlockTool,
    ValidateTool,
    builtin_designer_tools,
    get_global_registry,
    register_designer_tools,
)
from deep_research.agent_designer.structural_gate import StructuralGateTool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def ctx() -> ToolContext:
    return ToolContext(query="")


def _specialized_ast() -> dict[str, Any]:
    """Hand-crafted AST whose synthesizer/reflector cover description nouns.

    Description = ``valuation fundamentals dividend yields balance sheet``.
    Both noun coverage thresholds rely on lower-case substring match, so the
    agents' prompts must include at least two of those tokens.
    """
    description = (
        "Investigate company fundamentals and valuation: dividend yields, "
        "balance sheet strength, and growth drivers."
    )
    synth_sp = (
        "You are the synthesizer agent for an investment analysis workflow. "
        "Produce a multi-section report covering fundamentals, valuation "
        "tradeoffs, and dividend trajectory. Cite supporting balance sheet "
        "sources for every numerical claim. Surface contradictions between "
        "lane researchers rather than silently picking one. Never invent "
        "figures or dates that lack a direct observation in the sources pool. "
        "Structure the report so the fundamentals section precedes the "
        "valuation discussion, and end with a dividend outlook recommendation."
    )
    reflector_sp = (
        "You are the reflector agent for an investment analysis workflow. "
        "Audit coverage for fundamentals analysis, dividend trends, and "
        "valuation drivers before declaring research complete. Specifically "
        "check that every observation in the pool maps back to one of the "
        "required output sections, and flag any lane that produced no "
        "evidence at all. Do not approve completion while any required "
        "fundamentals, dividend, or valuation aspect remains uncovered."
    )
    return {
        "id": "specialized",
        "name": "Specialized Workflow",
        "description": description,
        "version": 1,
        "root": {
            "id": "root",
            "type": "sequence",
            "label": "Pipeline",
            "config": {},
            "children": [
                {
                    "id": "synth",
                    "type": "agent",
                    "label": "Synthesizer",
                    "config": {
                        "subtype": "synthesizer",
                        "model_tier": "complex",
                        "input_keys": ["query"],
                        "output_key": "report",
                        "system_prompt": synth_sp,
                        "user_prompt_template": (
                            "Compose the fundamentals + valuation report for "
                            "{query}."
                        ),
                    },
                    "children": [],
                },
                {
                    "id": "ref",
                    "type": "agent",
                    "label": "Reflector",
                    "config": {
                        "subtype": "reflector",
                        "model_tier": "analytical",
                        "input_keys": ["query"],
                        "output_key": "reflection",
                        "system_prompt": reflector_sp,
                        "user_prompt_template": (
                            "Audit fundamentals + valuation coverage for "
                            "{query}."
                        ),
                    },
                    "children": [],
                },
            ],
        },
        "tools": [],
        "pools": [],
        "sources": [],
        "models": {},
        "required_inputs": ["query"],
        "output_keys": ["report"],
    }


def _legacy_structural_gate_ast() -> dict[str, Any]:
    return {
        "description": (
            "Build a workflow covering valuation, fundamentals, dividend "
            "yields, balance sheet strength, and growth drivers."
        ),
        "root": {
            "id": "root",
            "type": "sequence",
            "children": [
                {
                    "id": "synth",
                    "type": "agent",
                    "config": {
                        "subtype": "synthesizer",
                        "system_prompt": "You summarize the research.",
                        "user_prompt_template": "Write the final answer.",
                    },
                },
                {
                    "id": "reflector",
                    "type": "agent",
                    "config": {
                        "subtype": "reflector",
                        "system_prompt": "You review whether work is complete.",
                        "user_prompt_template": "Approve or request changes.",
                    },
                },
            ],
        },
        "tools": [],
        "pools": [],
        "sources": [],
        "models": {},
        "required_inputs": ["query"],
        "output_keys": ["report"],
    }


# ---------------------------------------------------------------------------
# propose_workflow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_propose_workflow_returns_valid_ast(ctx: ToolContext) -> None:
    tool = ProposeWorkflowTool()
    result = await tool.execute(
        {"intent": "Research the financial outlook for NVIDIA in 2025"},
        ctx,
    )
    assert result.success is True
    assert isinstance(result.data["current_ast"], dict)
    # Round-trip through the framework loader — the AST must satisfy the
    # WorkflowDefinition Pydantic schema.
    load_workflow_from_dict(result.data["current_ast"])
    # And the content payload mirrors the AST.
    assert json.loads(result.content) == result.data["current_ast"]


@pytest.mark.asyncio
async def test_propose_workflow_accepts_python_literal_design_brief(
    ctx: ToolContext,
) -> None:
    tool = ProposeWorkflowTool()
    brief = {
        "workflow_name": "Literal Brief Research",
        "topology": "parallel_lanes",
        "research_lanes": [
            {
                "description": "Authoritative evidence for the requested topic",
                "system_prompt": "Use the supplied lane brief to gather citeable evidence.",
                "user_prompt_template": (
                    "Collect concrete evidence for the requested topic and "
                    "flag unavailable data."
                ),
            }
        ],
    }

    result = await tool.execute(
        {
            "intent": "Research a named public company",
            "design_brief": repr(brief),
        },
        ctx,
    )

    assert result.success is True
    assert result.data["current_ast"]["name"] == "Literal Brief Research"


@pytest.mark.asyncio
async def test_propose_workflow_preserves_brief_with_custom_target_placeholder(
    ctx: ToolContext,
) -> None:
    tool = ProposeWorkflowTool()
    result = await tool.execute(
        {
            "intent": "Build a decision-grade research report for a supplied target",
            "design_brief": {
                "workflow_name": "Target Research Report",
                "domain": "specialized target research",
                "topology": "parallel_lanes",
                "required_outputs": ["Executive summary", "Evidence review"],
                "research_lanes": [
                    {
                        "description": "Primary evidence and current status",
                        "system_prompt": (
                            "You are a specialized researcher. Gather primary "
                            "evidence, current status, source dates, and gaps. "
                            * 4
                        ),
                        "user_prompt_template": (
                            "Produce the primary evidence lane for "
                            "{target_identifier}. Address official records, "
                            "recent authoritative updates, conflicting source "
                            "claims, and any missing data. Return only "
                            "evidence-backed findings."
                        ),
                    }
                ],
            },
        },
        ctx,
    )

    assert result.success is True
    ast = result.data["current_ast"]
    assert ast["name"] == "Target Research Report"
    coordinator_prompt = ast["root"]["children"][0]["config"]["system_prompt"]
    assert "Domain: specialized target research" in coordinator_prompt

    lane = ast["root"]["children"][1]["children"][0]
    template = lane["config"]["user_prompt_template"]
    assert "{target_identifier}" not in template
    assert "for {query}" in template
    assert "Produce the primary evidence lane" in template


@pytest.mark.asyncio
async def test_propose_workflow_rejects_empty_intent(ctx: ToolContext) -> None:
    tool = ProposeWorkflowTool()
    with pytest.raises(ValueError):
        tool.validate_arguments({"intent": ""})


# ---------------------------------------------------------------------------
# structural_gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_structural_gate_pass_on_specialized_ast(ctx: ToolContext) -> None:
    tool = StructuralGateTool()
    ast = _specialized_ast()
    result = await tool.execute({"ast": ast}, ctx)
    assert result.data["status"] == "pass"
    assert result.data["failures"] == []


@pytest.mark.asyncio
async def test_structural_gate_accepts_json_string_ast(ctx: ToolContext) -> None:
    tool = StructuralGateTool()
    ast = _specialized_ast()
    result = await tool.execute({"ast": json.dumps(ast)}, ctx)
    assert result.data["status"] == "pass"


@pytest.mark.asyncio
async def test_structural_gate_fail_on_legacy_artifact(ctx: ToolContext) -> None:
    tool = StructuralGateTool()
    result = await tool.execute({"ast": _legacy_structural_gate_ast()}, ctx)
    assert result.data["status"] == "fail"
    assert len(result.data["failures"]) >= 1
    # Each failure carries the shape promised in the docstring.
    for failure in result.data["failures"]:
        assert set(failure.keys()) == {
            "path",
            "kind",
            "message",
            "suggested_action",
        }


# ---------------------------------------------------------------------------
# parse_architect_ast
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parse_architect_ast_extracts_json_block(ctx: ToolContext) -> None:
    tool = ParseArchitectAstTool()
    msg = (
        "Here's the AST:\n"
        "```json\n"
        '{"root": {"type": "agent"}}\n'
        "```\nDone."
    )
    result = await tool.execute({"raw_message": msg}, ctx)
    assert result.data["parse_ok"] is True
    assert result.data["current_ast"] == {"root": {"type": "agent"}}


@pytest.mark.asyncio
async def test_parse_architect_ast_no_block_returns_empty(ctx: ToolContext) -> None:
    tool = ParseArchitectAstTool()
    result = await tool.execute(
        {"raw_message": "I cannot produce a workflow."}, ctx
    )
    assert result.data["parse_ok"] is False
    assert result.data["current_ast"] == {}
    assert result.content == "{}"


@pytest.mark.asyncio
async def test_parse_architect_ast_no_block_falls_back_to_cache(
    ctx: ToolContext,
) -> None:
    cached = {"root": {"id": "cached", "type": "agent", "config": {"subtype": "synthesizer"}}}
    tool = ParseArchitectAstTool(state_getter=lambda: cached)

    result = await tool.execute(
        {"raw_message": "The workflow is valid, but no JSON block was emitted."},
        ctx,
    )

    assert result.data["parse_ok"] is True
    assert result.data["parse_fallback"] == "state_cache"
    assert result.data["current_ast"] == cached


@pytest.mark.asyncio
async def test_parse_architect_ast_prefers_cache_over_stale_json_block(
    ctx: ToolContext,
) -> None:
    cached = {"root": {"id": "cached", "type": "agent", "config": {"subtype": "synthesizer"}}}
    stale_msg = (
        "The workflow is valid.\n"
        "```json\n"
        '{"root": {"id": "stale", "type": "agent", "config": {"subtype": "synthesizer"}}}\n'
        "```"
    )
    tool = ParseArchitectAstTool(state_getter=lambda: cached)

    result = await tool.execute({"raw_message": stale_msg}, ctx)

    assert result.data["parse_ok"] is True
    assert result.data["parse_fallback"] == "state_cache_preferred"
    assert result.data["current_ast"] == cached


@pytest.mark.asyncio
async def test_parse_architect_ast_malformed_json_returns_empty(
    ctx: ToolContext,
) -> None:
    tool = ParseArchitectAstTool()
    msg = "```json\n{not valid json}\n```"
    result = await tool.execute({"raw_message": msg}, ctx)
    assert result.data["parse_ok"] is False
    assert result.data["current_ast"] == {}


@pytest.mark.asyncio
async def test_parse_architect_ast_empty_json_block_falls_back_to_cache(
    ctx: ToolContext,
) -> None:
    cached = {"root": {"id": "cached", "type": "agent", "config": {"subtype": "researcher"}}}
    tool = ParseArchitectAstTool(state_getter=lambda: cached)

    result = await tool.execute({"raw_message": "```json\n{}\n```"}, ctx)

    assert result.data["parse_ok"] is True
    assert result.data["parse_fallback"] == "state_cache"
    assert result.data["current_ast"]["root"]["id"] == cached["root"]["id"]
    assert result.data["current_ast"]["root"]["config"]["subtype"] == "researcher"
    assert "web_research" in result.data["current_ast"]["root"]["config"]["tools"]


# ---------------------------------------------------------------------------
# extract_critic_approved
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_critic_approved_true(ctx: ToolContext) -> None:
    tool = ExtractCriticApprovedTool()
    payload = json.dumps({"approve": True, "directives": []})
    result = await tool.execute({"critic_verdict": payload}, ctx)
    assert result.data["critic_approved"] is True


@pytest.mark.asyncio
async def test_extract_critic_approved_true_from_pydantic_model(
    ctx: ToolContext,
) -> None:
    tool = ExtractCriticApprovedTool()
    verdict = CriticVerdict(approve=True, directives=[])
    result = await tool.execute({"critic_verdict": verdict}, ctx)
    assert result.data["critic_approved"] is True


@pytest.mark.asyncio
async def test_extract_critic_approved_true_from_pydantic_repr(
    ctx: ToolContext,
) -> None:
    tool = ExtractCriticApprovedTool()
    result = await tool.execute(
        {"critic_verdict": "approve=True directives=[]"},
        ctx,
    )
    assert result.data["critic_approved"] is True


@pytest.mark.asyncio
async def test_extract_critic_approved_false_from_dict(ctx: ToolContext) -> None:
    tool = ExtractCriticApprovedTool()
    result = await tool.execute(
        {"critic_verdict": {"approve": False, "directives": []}},
        ctx,
    )
    assert result.data["critic_approved"] is False


@pytest.mark.asyncio
async def test_extract_critic_approved_false_on_parse_failure(
    ctx: ToolContext,
) -> None:
    tool = ExtractCriticApprovedTool()
    result = await tool.execute({"critic_verdict": "garbage"}, ctx)
    assert result.data["critic_approved"] is False


# ---------------------------------------------------------------------------
# Mutation tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_block_patches_label(ctx: ToolContext) -> None:
    propose = ProposeWorkflowTool()
    initial = await propose.execute(
        {"intent": "Research climate policy trends"},
        ctx,
    )
    ast = initial.data["current_ast"]

    update = UpdateBlockTool()
    patched = await update.execute(
        {
            "current_ast": ast,
            "path": "root",
            "patches": {"label": "Customized Root"},
        },
        ctx,
    )
    assert patched.success is True
    assert patched.data["current_ast"]["root"]["label"] == "Customized Root"


@pytest.mark.asyncio
async def test_add_block_appends_node_and_updates_cache(ctx: ToolContext) -> None:
    cached: dict[str, Any] = {}

    def get_cached() -> dict[str, Any]:
        return cached

    def set_cached(ast: Any) -> None:
        nonlocal cached
        cached = ast

    propose = ProposeWorkflowTool(state_setter=set_cached)
    await propose.execute(
        {
            "intent": "Build a market research workflow",
            "design_brief": {
                "topology": "parallel_lanes",
                "research_lanes": ["Demand signals", "Competitive risks"],
            },
        },
        ctx,
    )

    add = AddBlockTool(state_getter=get_cached, state_setter=set_cached)
    result = await add.execute(
        {
            "parent_path": "main",
            "node_type": "agent",
            "label": "Coverage Reflector",
            "config": {
                "subtype": "reflector",
                "input_keys": ["query", "report"],
                "output_key": "coverage_review",
                "system_prompt": "Audit coverage for market research outputs.",
            },
        },
        ctx,
    )

    assert result.success is True
    assert result.data["new_node_path"].startswith("root.children.")
    assert cached == result.data["current_ast"]
    assert cached["root"]["children"][-1]["label"] == "Coverage Reflector"


@pytest.mark.asyncio
async def test_declare_and_bind_tool_flow(ctx: ToolContext) -> None:
    # Start from a tiny AST so we control the agent shape directly.
    ast: dict[str, Any] = {
        "id": "wf",
        "name": "wf",
        "description": "",
        "version": 1,
        "root": {
            "id": "agent",
            "type": "agent",
            "label": "Agent",
            "config": {"subtype": "researcher", "input_keys": ["query"], "output_key": "out"},
            "children": [],
        },
        "tools": [],
        "pools": [],
        "sources": [],
        "models": {},
        "required_inputs": ["query"],
        "output_keys": ["out"],
    }
    declare = DeclareToolTool()
    declared = await declare.execute(
        {
            "current_ast": ast,
            "kind": "web_search",
            "name": "web_search",
            "config": {"max_results": 5},
        },
        ctx,
    )
    assert declared.success is True
    new_ast = declared.data["current_ast"]
    assert any(t["name"] == "web_search" for t in new_ast["tools"])

    bind = BindToolToBlockTool()
    bound = await bind.execute(
        {
            "current_ast": new_ast,
            "node_path": "root",
            "tool_name": "web_search",
        },
        ctx,
    )
    assert bound.success is True
    bound_tools = bound.data["current_ast"]["root"]["config"]["tools"]
    assert "web_research" in bound_tools
    assert "web_crawl" in bound_tools
    assert "web_search" not in bound_tools


@pytest.mark.asyncio
async def test_update_block_reapplies_grounding_defaults_after_config_replace(
    ctx: ToolContext,
) -> None:
    cached: dict[str, Any] = {}

    def get_cached() -> dict[str, Any]:
        return cached

    def set_cached(ast: Any) -> None:
        nonlocal cached
        cached = ast

    propose = ProposeWorkflowTool(state_setter=set_cached)
    initial = await propose.execute(
        {
            "intent": "Build a web research report about renewable energy markets",
            "design_brief": {
                "workflow_name": "Energy Research",
                "topology": "parallel_lanes",
                "research_lanes": ["Market drivers"],
            },
        },
        ctx,
    )
    assert initial.data["current_ast"]["root"]["children"][-1]["config"][
        "grounding_mode"
    ] == "reclaim"

    update = UpdateBlockTool(state_getter=get_cached, state_setter=set_cached)
    patched = await update.execute(
        {
            "path": "synthesizer",
            "patches": {
                "config": {
                    "subtype": "synthesizer",
                    "input_keys": ["query"],
                    "output_key": "report",
                    "model_tier": "complex",
                    "tools": [],
                    "max_tool_calls": 0,
                    "system_prompt": "Write the final report from the lane findings.",
                    "user_prompt_template": "Summarize {query}.",
                }
            },
        },
        ctx,
    )

    config = patched.data["current_ast"]["root"]["children"][-1]["config"]
    assert config["grounding_mode"] == "reclaim"
    assert config["output_schema"]["claim_disposition"] == {
        "abstained": "remove"
    }
    assert {item["pool"] for item in config["pool_inject"]} == {
        "observations",
        "sources",
    }
    assert cached == patched.data["current_ast"]


@pytest.mark.asyncio
async def test_set_model_tier_invalid_returns_error(ctx: ToolContext) -> None:
    ast: dict[str, Any] = {
        "id": "wf",
        "name": "wf",
        "description": "",
        "version": 1,
        "root": {
            "id": "agent",
            "type": "agent",
            "label": "Agent",
            "config": {"subtype": "researcher", "input_keys": ["query"], "output_key": "out"},
            "children": [],
        },
        "tools": [],
        "pools": [],
        "sources": [],
        "models": {},
        "required_inputs": ["query"],
        "output_keys": ["out"],
    }
    tool = SetModelTierTool()
    result = await tool.execute(
        {
            "current_ast": ast,
            "node_path": "root",
            "tier": "definitely-not-a-tier",
        },
        ctx,
    )
    assert result.success is False
    assert "set_model_tier failed" in (result.error or "")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validate_reports_errors_on_broken_ast(ctx: ToolContext) -> None:
    tool = ValidateTool()
    result = await tool.execute({"current_ast": {}}, ctx)
    assert result.data["valid"] is False
    assert result.data["errors"]


# ---------------------------------------------------------------------------
# Registry registration
# ---------------------------------------------------------------------------


_REQUIRED_TOOL_NAMES = {
    "propose_workflow",
    "add_block",
    "update_block",
    "bind_tool_to_block",
    "set_model_tier",
    "declare_tool",
    "discover_sources",
    "validate",
    "structural_gate",
    "parse_architect_ast",
    "extract_critic_approved",
}


def test_builtin_designer_tools_lists_all_required() -> None:
    names = {t.definition.name for t in builtin_designer_tools()}
    assert _REQUIRED_TOOL_NAMES.issubset(names)


def test_register_designer_tools_into_registry() -> None:
    registry = ToolRegistry()
    register_designer_tools(registry)
    for name in _REQUIRED_TOOL_NAMES:
        assert registry.has(name), f"{name} not registered"


def test_global_registry_contains_new_tools() -> None:
    # Importing the package eagerly populates the global registry; the helper
    # also lets tests resolve it directly.
    import deep_research.agent_designer as ad  # noqa: F401 — import trigger

    reg = get_global_registry()
    assert reg.has("structural_gate")
    assert reg.has("parse_architect_ast")
    assert reg.has("extract_critic_approved")
    assert reg.has("propose_workflow")
    assert reg.has("add_block")


def test_global_registry_resolves_via_tool_ref() -> None:
    """The executor's _exec_tool calls ``registry.resolve(ToolRef(...))``.

    Verify resolution succeeds for both 'builtin' and 'enterprise' ref
    types — the global registry registers tools on BOTH stores so the
    designer_workflow.yaml's ``type: tool`` nodes can use either form.
    """
    from databricks_deep_research.tools.protocol import ToolRef

    reg = get_global_registry()
    assert reg.resolve(ToolRef(type="builtin", name="structural_gate")) is not None
    assert (
        reg.resolve(ToolRef(type="enterprise", name="parse_architect_ast"))
        is not None
    )
