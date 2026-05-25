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
    InspectAssetsTool,
    ParseArchitectAstTool,
    ProposeWorkflowTool,
    RecommendToolsForAssetsTool,
    RemoveToolTool,
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


def _find_node_by_id(ast: dict[str, Any], node_id: str) -> dict[str, Any]:
    stack = [ast.get("root")]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        if node.get("id") == node_id:
            return node
        children = node.get("children", [])
        if isinstance(children, list):
            stack.extend(children)
        config = node.get("config")
        if isinstance(config, dict):
            body = config.get("body")
            if isinstance(body, dict):
                stack.append(body)
    raise AssertionError(f"node {node_id!r} not found")


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
async def test_propose_workflow_with_selected_assets_requires_explicit_tool_plan(
    ctx: ToolContext,
) -> None:
    tool = ProposeWorkflowTool(
        asset_getter=lambda: [
            {
                "kind": "vector_index",
                "full_name": "main.cat.idx",
                "usage": "required",
            }
        ]
    )
    brief = {
        "workflow_name": "Asset Research",
        "topology": "single_agent",
        "research_lanes": [
            {
                "description": "Answer from the selected corpus",
                "system_prompt": "Use selected corpus evidence to answer the question.",
                "user_prompt_template": "Answer the user's question from selected evidence.",
            }
        ],
    }

    result = await tool.execute(
        {
            "intent": "Research selected assets to answer user questions",
            "design_brief": brief,
        },
        ctx,
    )

    ast = result.data["current_ast"]
    # Top-level declared tool list stays empty until the architect emits a
    # tool_plan (still the architect's job).
    assert ast["tools"] == []
    # P4-1 contract: when no architect tool_plan exists, _tool_plan_bindings
    # falls back to the caller's default (default_researcher_tools =
    # ["web_research"]) instead of silently returning []. This guarantees a
    # deployed workflow always has SOME tool to run with, even if the
    # architect didn't finalize tool_plan. The architect's job is now to
    # OVERRIDE the default with corpus-specific bindings, not to bootstrap
    # from empty.
    answer_node = _find_node_by_id(ast, "answer-agent")
    assert answer_node["config"]["tools"] == ["web_research"]


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
    # Plan v2.1 M10 adds the ``severity`` field so the gate can grade
    # blocking vs warning/info defects.
    for failure in result.data["failures"]:
        assert set(failure.keys()) == {
            "path",
            "kind",
            "message",
            "severity",
            "suggested_action",
        }


@pytest.mark.asyncio
async def test_structural_gate_passes_with_only_warning_severity_defects(
    ctx: ToolContext,
) -> None:
    """Plan v2.1 M10 — when every defect is severity=warning, the gate
    status is ``pass`` so the workflow ships. Warnings still surface in
    the failures list for observability.

    This is the path that lets the placeholder_pending lifecycle finding
    show up in traces without breaking workflows whose architect prompt
    is not yet tuned to satisfy it.
    """
    from deep_research.agent_designer.blueprint import build_blueprint

    sig = {
        "asset_signature": "web_only",
        "retrieval_pattern": "independent_lanes",
        "question_class": "open_research",
        "primary_evidence_kind": "web_articles",
        "expected_output_shape": "structured_report",
        "independent_workstreams_count": 2,
        "step_dependencies_present": False,
        "iteration_required": False,
        "output_aggregation_kind": "cross_concern_synthesis",
        "lane_descriptions": ["topic A", "topic B"],
    }
    ast = build_blueprint(sig, "q", [])
    # The deterministic blueprint leaves placeholder_pending_nodes set;
    # detect_unspecialized_agents emits warning-severity findings for
    # them. The gate must NOT fail on warnings alone.
    tool = StructuralGateTool()
    result = await tool.execute({"ast": ast}, ctx)
    # The blueprint may still have other blocking defects (e.g., the
    # synthesizer's generic prompt). For THIS test we assert ONLY that
    # placeholder_pending warnings — present or not — don't promote to
    # blocking on the gate.
    placeholder_warnings = [
        f for f in result.data["failures"]
        if f.get("kind") == "placeholder_pending"
    ]
    for failure in placeholder_warnings:
        assert failure["severity"] == "warning", (
            "placeholder_pending must be severity=warning so the gate "
            "doesn't block workflows the runtime can still execute"
        )


@pytest.mark.asyncio
async def test_structural_gate_fails_required_asset_without_bound_tool(
    ctx: ToolContext,
) -> None:
    tool = StructuralGateTool()
    ast = _specialized_ast()
    ast["tools"] = [{"name": "web_search", "kind": "web_search", "config": {}}]

    result = await tool.execute(
        {
            "ast": ast,
            "assets": [
                {
                    "kind": "vector_index",
                    "full_name": "main.cat.idx",
                    "usage": "required",
                }
            ],
            "intent": "Build a fixed-corpus assistant over selected assets.",
        },
        ctx,
    )

    assert result.data["status"] == "fail"
    assert any("Required asset" in failure["message"] for failure in result.data["failures"])


@pytest.mark.asyncio
async def test_structural_gate_fails_researcher_with_no_tools(ctx: ToolContext) -> None:
    tool = StructuralGateTool()
    ast = _specialized_ast()
    ast["root"]["children"].insert(
        0,
        {
            "id": "researcher",
            "type": "agent",
            "label": "Researcher",
            "config": {
                "subtype": "researcher",
                "input_keys": ["query"],
                "output_key": "findings",
                "tools": [],
                "system_prompt": (
                    "Investigate the user's topic with concrete evidence and "
                    "return cited findings for synthesis."
                ),
            },
            "children": [],
        },
    )

    result = await tool.execute({"ast": ast}, ctx)

    assert result.data["status"] == "fail"
    assert any(
        "no bound executable evidence tools" in failure["message"]
        for failure in result.data["failures"]
    )


@pytest.mark.asyncio
async def test_structural_gate_rejects_custom_tool_as_researcher_evidence(
    ctx: ToolContext,
) -> None:
    tool = StructuralGateTool()
    ast = _specialized_ast()
    ast["tools"] = [{"name": "legacy_tool", "kind": "custom", "config": {}}]
    ast["root"]["children"].insert(
        0,
        {
            "id": "researcher",
            "type": "agent",
            "label": "Researcher",
            "config": {
                "subtype": "researcher",
                "input_keys": ["query"],
                "output_key": "findings",
                "tools": ["legacy_tool"],
                "system_prompt": (
                    "Investigate the user's topic with concrete evidence and "
                    "return cited findings for synthesis."
                ),
            },
            "children": [],
        },
    )

    result = await tool.execute({"ast": ast}, ctx)
    messages = [failure["message"] for failure in result.data["failures"]]

    assert result.data["status"] == "fail"
    assert any("unsupported runtime kind 'custom'" in message for message in messages)
    assert any("no bound executable evidence tools" in message for message in messages)


@pytest.mark.asyncio
async def test_structural_gate_fails_unused_runtime_tool(ctx: ToolContext) -> None:
    tool = StructuralGateTool()
    ast = _specialized_ast()
    ast["tools"] = [{"name": "asset_search", "kind": "vector_search", "config": {}}]

    result = await tool.execute({"ast": ast}, ctx)

    assert result.data["status"] == "fail"
    assert any("declared but not bound" in failure["message"] for failure in result.data["failures"])


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
    assert result.data["current_ast"]["root"]["config"].get("tools", []) == []


@pytest.mark.asyncio
async def test_parse_architect_ast_preserves_llm_chosen_tools(
    ctx: ToolContext,
) -> None:
    cached = {
        "id": "wf",
        "name": "wf",
        "description": "Answer from selected assets only. Do not use public web tools.",
        "version": 1,
        "root": {
            "id": "agent",
            "type": "agent",
            "label": "Agent",
            "config": {
                "subtype": "researcher",
                "input_keys": ["query"],
                "output_key": "out",
                "tools": ["web_search", "asset_search"],
                "system_prompt": "Use selected assets only. Do not use public web tools.",
            },
            "children": [],
        },
        "tools": [
            {"name": "web_search", "kind": "web_search", "config": {}},
            {
                "name": "asset_search",
                "kind": "vector_search",
                "config": {"index_name": "main.cat.idx"},
            },
        ],
        "pools": [],
        "sources": [],
        "models": {},
        "required_inputs": ["query"],
        "output_keys": ["out"],
    }
    tool = ParseArchitectAstTool(state_getter=lambda: cached)

    result = await tool.execute(
        {
            "raw_message": "no json block",
            "assets": [
                {
                    "kind": "vector_index",
                    "full_name": "main.cat.idx",
                    "usage": "required",
                }
            ],
            "intent": "Build a fixed-corpus assistant over selected assets.",
        },
        ctx,
    )

    ast = result.data["current_ast"]
    kinds = {item["kind"] for item in ast["tools"]}
    assert {"web_search", "vector_search"}.issubset(kinds)
    assert "web_research" not in kinds
    assert ast["root"]["config"]["tools"] == ["web_search", "asset_search"]
    fix_kinds = {fix["kind"] for fix in result.data["normalization_fixes"]}
    assert "tool_consolidation" not in fix_kinds


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
    assert bound_tools == ["web_search"]


@pytest.mark.asyncio
async def test_remove_tool_unbinds_every_agent_reference(ctx: ToolContext) -> None:
    ast: dict[str, Any] = {
        "id": "wf",
        "name": "wf",
        "description": "",
        "version": 1,
        "root": {
            "id": "main",
            "type": "sequence",
            "label": "Main",
            "config": {},
            "children": [
                {
                    "id": "agent",
                    "type": "agent",
                    "label": "Agent",
                    "config": {
                        "subtype": "researcher",
                        "input_keys": ["query"],
                        "output_key": "out",
                        "tools": ["web_search", "asset_search"],
                    },
                    "children": [],
                }
            ],
        },
        "tools": [
            {"name": "web_search", "kind": "web_search", "config": {}},
            {"name": "asset_search", "kind": "vector_search", "config": {}},
        ],
        "pools": [],
        "sources": [],
        "models": {},
        "required_inputs": ["query"],
        "output_keys": ["out"],
    }
    tool = RemoveToolTool()

    result = await tool.execute({"current_ast": ast, "name": "web_search"}, ctx)

    new_ast = result.data["current_ast"]
    assert [item["name"] for item in new_ast["tools"]] == ["asset_search"]
    assert new_ast["root"]["children"][0]["config"]["tools"] == ["asset_search"]


@pytest.mark.asyncio
async def test_update_block_reapplies_grounding_defaults_after_config_replace(
    ctx: ToolContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Plan v2.1 PR-3: this test patches structural config keys
    # (``subtype``, ``input_keys``, ``output_key``) that the new
    # allow-list rejects under DESIGNER_DETERMINISTIC_BLUEPRINT=ON.
    # Disable the flag here to exercise the legacy unconstrained path.
    monkeypatch.setenv("DESIGNER_DETERMINISTIC_BLUEPRINT", "0")
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
# asset inspection / recommendation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inspect_assets_tool_reads_asset_getter(ctx: ToolContext) -> None:
    tool = InspectAssetsTool(
        asset_getter=lambda: {
            "assets": [
                {
                    "kind": "vector_index",
                    "full_name": "main.cat.idx",
                    "usage": "required",
                }
            ]
        }
    )

    result = await tool.execute({}, ctx)

    assert result.success is True
    assert result.data["count"] == 1
    assert result.data["assets"][0]["identity"] == "main.cat.idx"


@pytest.mark.asyncio
async def test_recommend_tools_for_assets_tool_reads_asset_getter(
    ctx: ToolContext,
) -> None:
    tool = RecommendToolsForAssetsTool(
        asset_getter=lambda: {
            "assets": [
                {
                    "kind": "delta_table",
                    "full_name": "main.cat.rows",
                    "usage": "required",
                    "field_roles": {"content": "content"},
                    "metadata": {"warehouse_id": "abc123"},
                }
            ]
        }
    )

    result = await tool.execute({"intent": "sum totals from table rows"}, ctx)

    assert result.success is True
    assert result.data["diagnostics"] == []
    kinds = {item["kind"] for item in result.data["recommended_tools"]}
    assert {"delta_read", "delta_grep", "compute", "compute_namespace"}.issubset(kinds)


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
    "remove_tool",
    "discover_sources",
    "inspect_assets",
    "recommend_tools_for_assets",
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
