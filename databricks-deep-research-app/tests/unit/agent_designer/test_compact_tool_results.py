"""Edit-lane mutation-result compaction + batched bind + graceful error.

Root cause of the recurring "add a compute tool to the research candidates does
nothing, no error" bug (chat 2b470fcb): the EDIT agent emitted N parallel
``bind_tool_to_block`` calls and EVERY mutation tool echoed the FULL ~70KB AST as
its result, so the follow-up LLM call ballooned to ~140K prompt tokens, fell back
to a gateway 400 ("missing thought_signature in functionCall parts"), and the
edit aborted before any mutation was applied.

These tests pin the fix:
  * compact results in the edit lane (bounded summary, not the full AST),
  * full AST preserved in the build lane AND in result ``data`` (cache path),
  * batched bind: one tool, many nodes, in a SINGLE call,
  * the bounded-cumulative-result invariant the original 4-candidate probe missed,
  * a friendly, "left unchanged" error message instead of a raw gateway 400.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from databricks_deep_research.tools.protocol import ToolContext

from deep_research.agent_designer.framework_tools import (
    BindToolToBlockTool,
    compact_tool_results,
)
from deep_research.agent_designer.orchestrator import _edit_stream_error_message


@pytest.fixture
def ctx() -> ToolContext:
    return ToolContext(query="")


def _best_of_n_ast(n: int) -> dict[str, Any]:
    """A best_of_n workflow with *n* candidate agents, each carrying a LONG
    prompt so the full AST is large (mirrors the ~70KB live failure)."""
    long_prompt = "You are a research candidate. " * 400
    children = [
        {
            "id": f"candidate-{i}",
            "type": "agent",
            "label": f"Candidate {i}",
            "config": {
                "subtype": "synthesizer",
                "output_key": f"report_{i}",
                "system_prompt": long_prompt,
                "tools": ["web"],
            },
            "children": [],
        }
        for i in range(1, n + 1)
    ]
    return {
        "name": "Research Best-of-N",
        "root": {
            "id": "main",
            "type": "parallel",
            "label": "candidates",
            "config": {},
            "children": children,
        },
        "tools": [
            {"name": "web", "kind": "web_search"},
            {"name": "compute", "kind": "compute"},
        ],
    }


# --- compaction -------------------------------------------------------------


async def test_bind_result_full_ast_by_default(ctx: ToolContext) -> None:
    """Build lane (no compact scope active): result content carries the full AST
    — the architect relies on it, so default behavior is unchanged."""
    ast = _best_of_n_ast(2)
    result = await BindToolToBlockTool().execute(
        {"current_ast": ast, "node_path": "candidate-1", "tool_name": "compute"}, ctx
    )
    payload = json.loads(result.content)
    assert payload.get("root")  # full AST shape echoed
    assert "compute" in payload["root"]["children"][0]["config"]["tools"]


async def test_bind_result_compact_in_edit_scope(ctx: ToolContext) -> None:
    """Edit lane (compact scope): result content is a bounded summary, NOT the
    full AST, and dramatically smaller. ``data`` still carries the full AST."""
    ast = _best_of_n_ast(8)
    full_len = len(json.dumps(ast))
    with compact_tool_results(True):
        result = await BindToolToBlockTool().execute(
            {"current_ast": ast, "node_path": "candidate-1", "tool_name": "compute"},
            ctx,
        )
    payload = json.loads(result.content)
    assert payload["ok"] is True
    assert "nodes" in payload and "root" not in payload  # summary, not full AST
    # the long prompt text must NOT leak into the LLM-visible result content
    assert "research candidate" not in result.content.lower()
    assert len(result.content) < full_len // 5
    # full AST still available to the orchestrator cache via ``data``
    bound_tools = result.data["current_ast"]["root"]["children"][0]["config"]["tools"]
    assert "compute" in bound_tools


async def test_compact_summary_lists_nodes_and_bound_tools(ctx: ToolContext) -> None:
    ast = _best_of_n_ast(3)
    with compact_tool_results(True):
        result = await BindToolToBlockTool().execute(
            {"current_ast": ast, "node_path": "candidate-2", "tool_name": "compute"},
            ctx,
        )
    payload = json.loads(result.content)
    ids = {row["id"] for row in payload["nodes"]}
    assert {"candidate-1", "candidate-2", "candidate-3"} <= ids
    cand2 = next(r for r in payload["nodes"] if r["id"] == "candidate-2")
    assert "compute" in cand2["tools"]


# --- batched bind (parallel-call hardening) ---------------------------------


async def test_batched_bind_binds_all_targets_in_one_call(ctx: ToolContext) -> None:
    ast = _best_of_n_ast(8)
    targets = [f"candidate-{i}" for i in range(1, 9)]
    with compact_tool_results(True):
        result = await BindToolToBlockTool().execute(
            {"current_ast": ast, "node_paths": targets, "tool_name": "compute"}, ctx
        )
    assert result.success is not False
    bound = result.data["current_ast"]
    for child in bound["root"]["children"]:
        assert "compute" in child["config"]["tools"]


async def test_batched_bind_reports_bad_target(ctx: ToolContext) -> None:
    ast = _best_of_n_ast(3)
    result = await BindToolToBlockTool().execute(
        {
            "current_ast": ast,
            "node_paths": ["candidate-1", "does-not-exist"],
            "tool_name": "compute",
        },
        ctx,
    )
    assert result.success is False
    assert "does-not-exist" in result.content


def test_validate_requires_a_target() -> None:
    tool = BindToolToBlockTool()
    with pytest.raises(ValueError):
        tool.validate_arguments({"tool_name": "compute"})
    # either addressing form satisfies it
    tool.validate_arguments({"tool_name": "compute", "node_path": "candidate-1"})
    tool.validate_arguments({"tool_name": "compute", "node_paths": ["candidate-1"]})


async def test_cumulative_result_bounded_under_many_binds(ctx: ToolContext) -> None:
    """The invariant the original 4-candidate probe missed: with compaction, N
    sequential binds on a LARGE workflow keep the COMBINED result size below a
    single full-AST echo — so the transcript cannot reach the 140K-token blowup
    that triggered the gateway fallback 400."""
    ast = _best_of_n_ast(8)
    full_len = len(json.dumps(ast))
    cur = ast
    total = 0
    with compact_tool_results(True):
        for i in range(1, 9):
            result = await BindToolToBlockTool().execute(
                {
                    "current_ast": cur,
                    "node_path": f"candidate-{i}",
                    "tool_name": "compute",
                },
                ctx,
            )
            total += len(result.content)
            cur = result.data["current_ast"]
    assert total < full_len  # 8 compact results < ONE full-AST echo


# --- graceful error ---------------------------------------------------------


def test_edit_error_oversized_is_friendly_and_hides_gateway_noise() -> None:
    exc = Exception(
        "Error code: 400 - missing thought_signature in functionCall parts, "
        "position 5"
    )
    msg = _edit_stream_error_message(exc, lane="edit")
    assert "left unchanged" in msg.lower()
    assert "thought_signature" not in msg  # raw gateway noise not surfaced


def test_edit_error_generic_is_friendly() -> None:
    msg = _edit_stream_error_message(Exception("kaboom"), lane="edit")
    assert "unchanged" in msg.lower()
    assert "kaboom" not in msg


def test_build_lane_error_keeps_detail() -> None:
    msg = _edit_stream_error_message(Exception("kaboom"), lane="build")
    assert "kaboom" in msg
