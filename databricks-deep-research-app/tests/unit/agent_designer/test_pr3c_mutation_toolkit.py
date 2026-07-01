"""PR3-C Layer 2 — wider mutation toolkit, CriticDirective severity,
extract_critic_approved all-advisory shortcut."""
from __future__ import annotations

import json
from typing import Any

import pytest
from databricks_deep_research.tools.protocol import ToolContext

from deep_research.agent_designer.critic_types import CriticDirective, CriticVerdict
from deep_research.agent_designer.framework_tools import (
    DeleteBlockTool,
    ExtractCriticApprovedTool,
    InspectAstSummaryTool,
    MoveBlockTool,
    UpdatePoolTool,
    builtin_designer_tools,
)
from deep_research.agent_designer.mutations import update_pool


@pytest.fixture
def ctx() -> ToolContext:
    return ToolContext(query="")


def _ast_with_pool() -> dict[str, Any]:
    return {
        "id": "x",
        "name": "x",
        "version": 1,
        "required_inputs": ["query"],
        "output_keys": ["final"],
        "tools": [],
        "pools": [
            {"name": "sources", "dedup_key": "url", "max_items": 100},
        ],
        "root": {
            "id": "root",
            "type": "sequence",
            "label": "Root",
            "config": {},
            "children": [
                {
                    "id": "agent-a",
                    "type": "agent",
                    "label": "A",
                    "config": {
                        "subtype": "coordinator",
                        "model_tier": "complex",
                    },
                },
                {
                    "id": "agent-b",
                    "type": "agent",
                    "label": "B",
                    "config": {
                        "subtype": "researcher",
                        "model_tier": "analytical",
                    },
                },
            ],
        },
    }


# ---------------------------------------------------------------------------
# update_pool — mutation function + framework tool
# ---------------------------------------------------------------------------


def test_update_pool_changes_dedup_key() -> None:
    ast = _ast_with_pool()
    new_ast = update_pool(ast, "sources", {"dedup_key": "chunk_id"})
    assert new_ast["pools"][0]["dedup_key"] == "chunk_id"
    # Original AST unmutated (immutable update).
    assert ast["pools"][0]["dedup_key"] == "url"


def test_update_pool_rejects_unknown_pool() -> None:
    ast = _ast_with_pool()
    with pytest.raises(Exception, match="No pool with name"):
        update_pool(ast, "nope", {"dedup_key": "url"})


def test_update_pool_rejects_disallowed_patch_key() -> None:
    ast = _ast_with_pool()
    with pytest.raises(Exception, match="Disallowed pool patch keys"):
        update_pool(ast, "sources", {"name": "renamed"})


def test_update_pool_changes_max_items() -> None:
    ast = _ast_with_pool()
    new_ast = update_pool(ast, "sources", {"max_items": 50})
    assert new_ast["pools"][0]["max_items"] == 50
    assert new_ast["pools"][0]["dedup_key"] == "url"  # untouched


async def test_update_pool_tool_writes_state(ctx: ToolContext) -> None:
    ast = _ast_with_pool()
    written: list[Any] = []
    tool = UpdatePoolTool(
        state_getter=lambda: ast, state_setter=written.append
    )
    result = await tool.execute(
        {"pool_name": "sources", "patches": {"dedup_key": "chunk_id"}}, ctx
    )
    body = json.loads(result.content)
    assert body["pools"][0]["dedup_key"] == "chunk_id"
    assert written and written[0]["pools"][0]["dedup_key"] == "chunk_id"


async def test_update_pool_tool_idempotent(ctx: ToolContext) -> None:
    ast = _ast_with_pool()
    tool = UpdatePoolTool(state_getter=lambda: ast)
    result1 = await tool.execute(
        {"pool_name": "sources", "patches": {"dedup_key": "chunk_id"}}, ctx
    )
    new_ast = json.loads(result1.content)
    # Re-run from the new AST → no-op (same dedup_key, no error).
    tool2 = UpdatePoolTool(state_getter=lambda: new_ast)
    result2 = await tool2.execute(
        {"pool_name": "sources", "patches": {"dedup_key": "chunk_id"}}, ctx
    )
    body2 = json.loads(result2.content)
    assert body2["pools"][0]["dedup_key"] == "chunk_id"


# ---------------------------------------------------------------------------
# delete_block / move_block — already exist in mutations.py; tools wrap them.
# ---------------------------------------------------------------------------


async def test_delete_block_tool_removes_node(ctx: ToolContext) -> None:
    ast = _ast_with_pool()
    tool = DeleteBlockTool(state_getter=lambda: ast)
    result = await tool.execute({"path": "agent-b"}, ctx)
    body = json.loads(result.content)
    child_ids = [c["id"] for c in body["root"]["children"]]
    assert "agent-b" not in child_ids


async def test_delete_block_tool_rejects_root(ctx: ToolContext) -> None:
    ast = _ast_with_pool()
    tool = DeleteBlockTool(state_getter=lambda: ast)
    result = await tool.execute({"path": "root"}, ctx)
    body = json.loads(result.content)
    assert body.get("error") or body.get("ok") is False


async def test_move_block_tool_repositions_node(ctx: ToolContext) -> None:
    # Add a parent for the move target.
    ast = _ast_with_pool()
    ast["root"]["children"].append(
        {
            "id": "wrapper",
            "type": "sequence",
            "label": "Wrapper",
            "config": {},
            "children": [],
        }
    )
    tool = MoveBlockTool(state_getter=lambda: ast)
    result = await tool.execute(
        {"from_path": "agent-b", "to_path": "wrapper"}, ctx
    )
    body = json.loads(result.content)
    wrapper = next(c for c in body["root"]["children"] if c["id"] == "wrapper")
    inner_ids = [c["id"] for c in wrapper["children"]]
    assert "agent-b" in inner_ids


# ---------------------------------------------------------------------------
# inspect_ast_summary — compact summary, NOT full AST
# ---------------------------------------------------------------------------


async def test_inspect_ast_summary_returns_compact_payload(
    ctx: ToolContext,
) -> None:
    ast = _ast_with_pool()
    tool = InspectAstSummaryTool(state_getter=lambda: ast)
    result = await tool.execute({}, ctx)
    body = json.loads(result.content)
    # Required compact fields.
    assert "node_count" in body
    assert "tool_count" in body
    assert "pools" in body
    assert "agent_roles" in body
    assert "validation_errors" in body
    # NOT the full AST (no 'root' key returned).
    assert "root" not in body
    # Counts make sense.
    assert body["node_count"] >= 3  # root + agent-a + agent-b
    assert "agent-a" in body["agent_roles"]
    assert "agent-b" in body["agent_roles"]


async def test_inspect_ast_summary_includes_plan_execute_nested_roles(
    ctx: ToolContext,
) -> None:
    ast = {
        "id": "designer-draft",
        "tools": [],
        "pools": [],
        "root": {
            "id": "main",
            "type": "sequence",
            "label": "Main",
            "config": {},
            "children": [
                {
                    "id": "coordinator",
                    "type": "agent",
                    "label": "Coordinator",
                    "config": {"subtype": "coordinator"},
                },
                {
                    "id": "plan-and-execute",
                    "type": "plan_and_execute",
                    "label": "Plan and Execute",
                    "config": {
                        "planner": {"label": "Planner", "subtype": "planner"},
                        "evaluator": {
                            "label": "Coverage Evaluator",
                            "subtype": "reflector",
                        },
                        "body": {
                            "id": "body",
                            "type": "sequence",
                            "label": "Body",
                            "config": {},
                            "children": [
                                {
                                    "id": "researcher",
                                    "type": "agent",
                                    "label": "Researcher",
                                    "config": {"subtype": "researcher"},
                                }
                            ],
                        },
                    },
                },
                {
                    "id": "synthesizer",
                    "type": "agent",
                    "label": "Synthesizer",
                    "config": {"subtype": "synthesizer"},
                },
            ],
        },
    }
    tool = InspectAstSummaryTool(state_getter=lambda: ast)
    result = await tool.execute({}, ctx)
    body = json.loads(result.content)
    assert body["topology"] == "plan_and_execute"
    assert any("reflector" in role for role in body["agent_roles"])
    assert any("planner" in role for role in body["agent_roles"])
    assert body["node_count"] >= 6


# ---------------------------------------------------------------------------
# CriticDirective severity + tool_hint defaults
# ---------------------------------------------------------------------------


def test_critic_directive_default_severity_is_blocking() -> None:
    d = CriticDirective(issue="x", suggested_action="y")
    assert d.severity == "blocking"
    assert d.tool_hint is None


def test_critic_directive_accepts_advisory_with_tool_hint() -> None:
    d = CriticDirective(
        issue="cosmetic", suggested_action="rename", severity="advisory",
        tool_hint="update_block",
    )
    assert d.severity == "advisory"
    assert d.tool_hint == "update_block"


# ---------------------------------------------------------------------------
# extract_critic_approved: all-advisory directives = approved
# ---------------------------------------------------------------------------


async def test_extract_critic_approved_all_advisory_returns_true(
    ctx: ToolContext,
) -> None:
    verdict = CriticVerdict(
        approve=False,
        directives=[
            CriticDirective(
                issue="polish 1", suggested_action="rename",
                severity="advisory",
            ),
            CriticDirective(
                issue="polish 2", suggested_action="reword",
                severity="advisory",
            ),
        ],
    )
    tool = ExtractCriticApprovedTool()
    result = await tool.execute({"critic_verdict": verdict.model_dump()}, ctx)
    body = json.loads(result.content)
    assert body["critic_approved"] is True


async def test_extract_critic_approved_any_blocking_returns_false(
    ctx: ToolContext,
) -> None:
    verdict = CriticVerdict(
        approve=False,
        directives=[
            CriticDirective(
                issue="polish", suggested_action="rename",
                severity="advisory",
            ),
            CriticDirective(
                issue="defect", suggested_action="fix",
                severity="blocking",
            ),
        ],
    )
    tool = ExtractCriticApprovedTool()
    result = await tool.execute({"critic_verdict": verdict.model_dump()}, ctx)
    body = json.loads(result.content)
    assert body["critic_approved"] is False


async def test_extract_critic_approved_no_directives_returns_false(
    ctx: ToolContext,
) -> None:
    # Empty directives + approve=False does NOT trigger the all-advisory
    # shortcut — that case is only meant to disambiguate polish-only verdicts.
    verdict = CriticVerdict(approve=False, directives=[])
    tool = ExtractCriticApprovedTool()
    result = await tool.execute({"critic_verdict": verdict.model_dump()}, ctx)
    body = json.loads(result.content)
    assert body["critic_approved"] is False


async def test_extract_critic_approved_explicit_true_short_circuits(
    ctx: ToolContext,
) -> None:
    verdict = CriticVerdict(approve=True, directives=[])
    tool = ExtractCriticApprovedTool()
    result = await tool.execute({"critic_verdict": verdict.model_dump()}, ctx)
    body = json.loads(result.content)
    assert body["critic_approved"] is True


# ---------------------------------------------------------------------------
# Registry — every new tool is registered
# ---------------------------------------------------------------------------


def test_builtin_designer_tools_includes_pr3c_tools() -> None:
    names = {t.definition.name for t in builtin_designer_tools()}
    assert "update_pool" in names
    assert "delete_block" in names
    assert "move_block" in names
    assert "inspect_ast_summary" in names
