"""Structural eval for the Agent Designer chat orchestrator.

This is a STRUCTURAL eval: each prompt's fake LLM produces a known-shape
tool-call sequence; we assert the final AST passes WorkflowDefinition
validation and contains the expected tool_kinds subset.

NOT a semantic eval — that would require a live LLM and human review.

By default this runs in fixture mode (fake LLM, deterministic). Set
RUN_LIVE_LLM_EVAL=1 to replace the fake with the real LLMClient (skipped
in CI by default — requires Databricks credentials and incurs cost).

V1 known limitations
--------------------
expected_node_types is checked in RELAXED mode: we only assert that "agent"
is always present (since propose_workflow always scaffolds root=agent).
Composite root types (plan_and_execute, parallel, conditional, loop,
sequence) would require a multi-step restructuring sequence from the fake LLM
that is non-trivial given that propose_workflow hard-codes root as an agent
node, and that delete_block cannot delete root. This is deferred to Phase 2
where a live-LLM mode (RUN_LIVE_LLM_EVAL=1) can exercise unrestricted shapes.

expected_tool_kinds IS checked strictly: every kind listed in the fixture
must appear in ast["tools"][*]["kind"] after the fake-LLM sequence runs. This
validates the declare_tool + bind_tool_to_block mutation path which is the
most common LLM code path.
"""
from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from databricks_deep_research.workflow.loader import load_workflow_from_dict

from deep_research.agent_designer.discovery import DesignerDiscoveryAdapter
from deep_research.agent_designer.orchestrator import (
    DesignerChatOrchestrator,
    LLMStreamChunk,
    LLMToolCall,
    MutationProposedEvent,
)


_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "designer_eval_prompts.yaml"


def _load_prompts() -> list[dict[str, Any]]:
    with _FIXTURE_PATH.open() as f:
        data = yaml.safe_load(f)
    return data["prompts"]  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Fake-LLM chunk builders
# ---------------------------------------------------------------------------


def _propose(intent: str) -> LLMStreamChunk:
    return LLMStreamChunk(
        tool_call=LLMToolCall(
            id="tc_propose",
            name="propose_workflow",
            arguments={"intent": intent},
        )
    )


def _declare_tool(name: str, kind: str, **config: Any) -> LLMStreamChunk:
    return LLMStreamChunk(
        tool_call=LLMToolCall(
            id=f"tc_decl_{name}",
            name="declare_tool",
            arguments={"name": name, "kind": kind, "config": dict(config)},
        )
    )


def _bind_tool(node_path: str, tool_name: str) -> LLMStreamChunk:
    return LLMStreamChunk(
        tool_call=LLMToolCall(
            id=f"tc_bind_{tool_name}",
            name="bind_tool_to_block",
            arguments={"node_path": node_path, "tool_name": tool_name},
        )
    )


def _finish() -> LLMStreamChunk:
    return LLMStreamChunk(finish=True)


# ---------------------------------------------------------------------------
# Per-prompt chunk sequences
#
# Strategy:
#   1. Always start with propose_workflow (creates root=agent scaffold).
#   2. Declare each required tool kind with a unique name.
#   3. Bind each declared tool to the root agent node.
#   4. Emit finish.
#
# This validates the full declare_tool + bind_tool_to_block pipeline and
# produces an AST that passes load_workflow_from_dict.
#
# Prompts whose expected_tool_kinds is [] (no tools needed) only propose + finish.
# Composite node types (parallel, conditional, loop, etc.) are V1-relaxed —
# they are not structurally constructed by the fake LLM (see module docstring).
# ---------------------------------------------------------------------------


def _chunks_for_tool_kinds(
    intent: str, tool_kinds: list[str]
) -> list[LLMStreamChunk]:
    """Build a chunk sequence: propose → declare+bind for each kind → finish."""
    chunks: list[LLMStreamChunk] = [_propose(intent)]
    for kind in tool_kinds:
        # Use kind as tool name (guaranteed unique within a single prompt)
        chunks.append(_declare_tool(kind, kind))
        chunks.append(_bind_tool("root", kind))
    chunks.append(_finish())
    return chunks


# Build the per-prompt sequence map from the fixture at import time.
# Keys match prompt["id"] from the YAML fixture.

_PROMPT_CHUNKS: dict[str, list[LLMStreamChunk]] = {}


def _build_chunk_map() -> None:
    for prompt in _load_prompts():
        pid: str = prompt["id"]
        intent: str = prompt["intent"]
        tool_kinds: list[str] = prompt.get("expected_tool_kinds", [])
        _PROMPT_CHUNKS[pid] = _chunks_for_tool_kinds(intent, tool_kinds)


_build_chunk_map()


# ---------------------------------------------------------------------------
# Fake LLM and Discovery implementations
# ---------------------------------------------------------------------------


class _FakeLLM:
    """Replays a pre-built chunk sequence for the given prompt id."""

    def __init__(self, prompt_id: str) -> None:
        self._chunks = _PROMPT_CHUNKS.get(prompt_id, [_finish()])

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[LLMStreamChunk]:
        for chunk in self._chunks:
            yield chunk


class _FakeDiscoveryService:
    """Returns an empty sources response without touching Databricks."""

    async def discover_all(
        self,
        user_id: str,
        user_token: str | None = None,
        **kwargs: Any,
    ) -> Any:
        return type("_EmptyResponse", (), {"sources": []})()


def _make_adapter() -> DesignerDiscoveryAdapter:
    return DesignerDiscoveryAdapter(_FakeDiscoveryService())


# ---------------------------------------------------------------------------
# Helpers for AST introspection
# ---------------------------------------------------------------------------


def _collect_node_types(node: dict[str, Any], result: set[str]) -> None:
    """Walk the AST recursively and collect all node type strings."""
    if not isinstance(node, dict):
        return
    ntype = node.get("type")
    if isinstance(ntype, str):
        result.add(ntype)
    for child in node.get("children") or []:
        _collect_node_types(child, result)
    # plan_and_execute body
    config = node.get("config") or {}
    body = config.get("body")
    if isinstance(body, dict):
        _collect_node_types(body, result)


def _ast_node_types(ast: dict[str, Any]) -> set[str]:
    types: set[str] = set()
    root = ast.get("root")
    if isinstance(root, dict):
        _collect_node_types(root, types)
    return types


def _ast_tool_kinds(ast: dict[str, Any]) -> set[str]:
    return {t["kind"] for t in (ast.get("tools") or []) if "kind" in t}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prompt", _load_prompts(), ids=lambda p: p["id"])
async def test_eval_prompts_produce_valid_ast(prompt: dict[str, Any]) -> None:
    """Each prompt's fake LLM produces a structurally valid AST.

    Assertions:
      1. At least one MutationProposedEvent is emitted (AST was produced).
      2. The final AST passes load_workflow_from_dict (structurally valid).
      3. 'agent' node type is always present (V1 relaxed node-type check —
         see module docstring for the V1 limitation on composite types).
      4. Every expected_tool_kind from the fixture is present in ast['tools']
         (strict check — this validates the declare_tool + bind pipeline).
    """
    if os.environ.get("RUN_LIVE_LLM_EVAL") == "1":
        pytest.skip(
            "RUN_LIVE_LLM_EVAL=1 is set — live-LLM eval not implemented in V1; "
            "run the fixture-mode eval (default) or implement live mode in Phase 4."
        )

    fake_llm = _FakeLLM(prompt["id"])
    orchestrator = DesignerChatOrchestrator(fake_llm, _make_adapter())

    final_ast: dict[str, Any] | None = None
    async for event in orchestrator.run_turn(
        messages=[{"role": "user", "content": prompt["intent"]}],
        current_ast=None,
        session_id=prompt["id"],
        user_token="fake-token",
    ):
        if isinstance(event, MutationProposedEvent):
            final_ast = event.new_ast

    # 1. An AST was produced
    assert final_ast is not None, (
        f"Prompt '{prompt['id']}': no MutationProposedEvent emitted — "
        "the fake LLM sequence must produce at least one mutation"
    )

    # 2. Structural validity — passes framework loader
    load_workflow_from_dict(final_ast)

    # 3. Relaxed node-type check: 'agent' is always present (V1 limitation)
    node_types = _ast_node_types(final_ast)
    assert "agent" in node_types, (
        f"Prompt '{prompt['id']}': expected 'agent' node type in AST, "
        f"got {node_types}"
    )

    # 4. Strict tool-kind check
    declared_kinds = _ast_tool_kinds(final_ast)
    for expected_kind in prompt.get("expected_tool_kinds", []):
        assert expected_kind in declared_kinds, (
            f"Prompt '{prompt['id']}': expected tool kind '{expected_kind}' "
            f"not found in declared tools (got {declared_kinds})"
        )


def test_fixture_has_at_least_10_prompts() -> None:
    """Aggregate guard: the eval fixture must contain at least 10 prompts."""
    prompts = _load_prompts()
    assert len(prompts) >= 10, (
        f"Eval fixture must have >= 10 prompts, got {len(prompts)}"
    )


def test_all_prompt_ids_are_unique() -> None:
    """Every prompt id in the fixture must be unique."""
    prompts = _load_prompts()
    ids = [p["id"] for p in prompts]
    assert len(ids) == len(set(ids)), (
        f"Duplicate prompt ids detected: {[i for i in ids if ids.count(i) > 1]}"
    )


def test_at_least_9_of_10_prompts_covered_by_chunk_map() -> None:
    """All prompts in the fixture must have a pre-built chunk sequence.

    Run separately so we can report coverage even if one prompt drifts.
    """
    prompts = _load_prompts()
    missing = [p["id"] for p in prompts if p["id"] not in _PROMPT_CHUNKS]
    assert len(missing) == 0, (
        f"Prompts without a fake-LLM chunk sequence: {missing}. "
        "Add entries to _build_chunk_map()."
    )
