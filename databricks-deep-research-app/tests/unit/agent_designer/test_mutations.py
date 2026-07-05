"""Unit tests for agent_designer.mutations — pure-function AST primitives."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from deep_research.agent_designer.mutations import (
    BlockMutationError,
    BlockPathError,
    add_block,
    bind_tool_to_block,
    declare_tool,
    delete_block,
    move_block,
    remove_tool,
    set_model_tier,
    set_surface,
    update_block,
)
from deep_research.agent_designer.registry import model_tiers_payload

# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _minimal_ast(root_type: str = "agent") -> dict[str, Any]:
    return {
        "id": "test",
        "name": "test",
        "version": 1,
        "root": {
            "id": "root",
            "type": root_type,
            "label": "root",
            "config": {"subtype": "researcher"} if root_type == "agent" else {},
            "children": [],
        },
        "tools": [],
        "pools": [],
        "sources": [],
        "models": {},
        "required_inputs": ["query"],
        "output_keys": ["output"],
        "token_budget": 0,
        "timeout_seconds": 1800,
    }


def _seq_ast() -> dict[str, Any]:
    """Root is sequence with one child agent."""
    ast = _minimal_ast("sequence")
    child: dict[str, Any] = {
        "id": "child0",
        "type": "agent",
        "label": "first agent",
        "config": {"subtype": "researcher"},
        "children": [],
    }
    ast["root"]["children"] = [child]
    return ast


def _nested_seq_ast() -> dict[str, Any]:
    """Root is sequence > inner sequence > agent leaf."""
    inner_child: dict[str, Any] = {
        "id": "leaf0",
        "type": "agent",
        "label": "leaf agent",
        "config": {"subtype": "researcher"},
        "children": [],
    }
    inner_seq: dict[str, Any] = {
        "id": "inner_seq",
        "type": "sequence",
        "label": "inner sequence",
        "config": {},
        "children": [inner_child],
    }
    ast = _minimal_ast("sequence")
    ast["root"]["children"] = [inner_seq]
    return ast


def _plan_and_execute_ast(body: Any = None) -> dict[str, Any]:
    """Root is plan_and_execute with optional body."""
    ast = _minimal_ast("plan_and_execute")
    ast["root"]["config"] = {"body": body}
    return ast


# ---------------------------------------------------------------------------
# add_block tests
# ---------------------------------------------------------------------------


def test_add_block_to_root_children() -> None:
    """Add a new agent node to the root sequence."""
    ast = _minimal_ast("sequence")
    new_ast, path = add_block(ast, "root", "agent", {"subtype": "synthesizer"}, "Synth")
    assert len(new_ast["root"]["children"]) == 1
    new_node = new_ast["root"]["children"][0]
    assert new_node["type"] == "agent"
    assert new_node["label"] == "Synth"
    assert new_node["config"]["subtype"] == "synthesizer"
    assert path == "root.children.0"


def test_add_block_replaces_generic_researcher_label() -> None:
    """Generic role ordinals are replaced with semantic generated-object names."""
    ast = _minimal_ast("sequence")
    new_ast, path = add_block(
        ast,
        "root",
        "agent",
        {"subtype": "researcher", "output_key": "market_risk_findings"},
        "Researcher 1",
    )
    new_node = new_ast["root"]["children"][0]
    assert path == "root.children.0"
    assert new_node["label"] == "Market Risk Findings Researcher"


def test_add_block_to_nested_composite() -> None:
    """Add a node to an inner sequence (sequence > sequence > add)."""
    ast = _nested_seq_ast()
    # inner_seq is at root.children.0; add a second child
    new_ast, path = add_block(
        ast, "root.children.0", "agent", {"subtype": "reflector"}, "Reflect"
    )
    inner = new_ast["root"]["children"][0]
    assert len(inner["children"]) == 2
    assert inner["children"][1]["type"] == "agent"
    assert path == "root.children.0.children.1"


def test_add_block_accepts_parent_node_id() -> None:
    """LLM-facing add_block should accept stable node ids, not only dot paths."""
    ast = _nested_seq_ast()
    new_ast, path = add_block(
        ast, "inner_seq", "agent", {"subtype": "reflector"}, "Reflect"
    )
    inner = new_ast["root"]["children"][0]
    assert len(inner["children"]) == 2
    assert inner["children"][1]["label"] == "Reflect"
    assert path == "root.children.0.children.1"


def test_add_block_to_plan_and_execute_body_none() -> None:
    """Body is None — new node is set directly as body."""
    ast = _plan_and_execute_ast(body=None)
    new_ast, path = add_block(
        ast,
        "root.config.body",
        "agent",
        {"subtype": "researcher"},
        "Researcher",
    )
    body = new_ast["root"]["config"]["body"]
    assert isinstance(body, dict)
    assert body["type"] == "agent"
    assert path == "root.config.body"


def test_add_block_to_plan_and_execute_body_wraps_in_sequence() -> None:
    """Body has a single non-sequence node — wrap in sequence when adding second."""
    existing_node: dict[str, Any] = {
        "id": "existing",
        "type": "agent",
        "label": "existing",
        "config": {},
        "children": [],
    }
    ast = _plan_and_execute_ast(body=existing_node)
    new_ast, path = add_block(
        ast,
        "root.config.body",
        "agent",
        {"subtype": "synthesizer"},
        "Synth",
    )
    body = new_ast["root"]["config"]["body"]
    assert body["type"] == "sequence"
    assert len(body["children"]) == 2
    assert body["children"][0]["id"] == "existing"
    assert body["children"][1]["type"] == "agent"
    assert path == "root.config.body.children.1"


def test_add_block_to_plan_and_execute_body_appends_to_sequence() -> None:
    """Body is already a sequence — append to its children."""
    seq_body: dict[str, Any] = {
        "id": "seq_body",
        "type": "sequence",
        "label": "body",
        "config": {},
        "children": [
            {"id": "c0", "type": "agent", "label": "c0", "config": {}, "children": []}
        ],
    }
    ast = _plan_and_execute_ast(body=seq_body)
    new_ast, path = add_block(
        ast,
        "root.config.body",
        "agent",
        {"subtype": "reflector"},
        "Reflect",
    )
    body = new_ast["root"]["config"]["body"]
    assert body["type"] == "sequence"
    assert len(body["children"]) == 2
    assert path == "root.config.body.children.1"


def test_add_block_returns_new_ast_path_correctly() -> None:
    """The returned path must resolve to the newly added node."""
    ast = _minimal_ast("sequence")
    new_ast, path = add_block(ast, "root", "tool", {"tool_name": "web_search"}, "Search")
    from deep_research.agent_designer.mutations import _get_at
    node = _get_at(new_ast, path)
    assert node["type"] == "tool"
    assert node["label"] == "Search"


def test_add_block_does_not_mutate_input() -> None:
    """The original ast must be unchanged after add_block."""
    ast = _minimal_ast("sequence")
    original = copy.deepcopy(ast)
    add_block(ast, "root", "agent", {"subtype": "planner"}, "Planner")
    assert ast == original


# ---------------------------------------------------------------------------
# update_block tests
# ---------------------------------------------------------------------------


def test_update_block_label() -> None:
    """Patching 'label' updates the node label."""
    ast = _seq_ast()
    new_ast = update_block(ast, "root.children.0", {"label": "Updated"})
    assert new_ast["root"]["children"][0]["label"] == "Updated"
    # original unchanged
    assert ast["root"]["children"][0]["label"] == "first agent"


def test_update_block_config_merge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patching 'config' replaces (shallow) the config dict.

    Plan v2.1 PR-3 narrowed config-level patches to a prompt-only
    allow-list when DESIGNER_DETERMINISTIC_BLUEPRINT is ON (the
    default). This test exercises the legacy unconstrained semantics
    via env opt-out — see test_architect_patches.py for the strict-mode
    coverage."""
    monkeypatch.setenv("DESIGNER_DETERMINISTIC_BLUEPRINT", "0")
    ast = _seq_ast()
    new_ast = update_block(
        ast, "root.children.0", {"config": {"subtype": "synthesizer", "model_tier": "complex"}}
    )
    cfg = new_ast["root"]["children"][0]["config"]
    assert cfg["subtype"] == "synthesizer"
    assert cfg["model_tier"] == "complex"


def test_update_block_rejects_type_change() -> None:
    """Patching 'type' must raise BlockMutationError."""
    ast = _seq_ast()
    with pytest.raises(BlockMutationError, match="type"):
        update_block(ast, "root.children.0", {"type": "tool"})


def test_update_block_rejects_children_change() -> None:
    """Patching 'children' must raise BlockMutationError."""
    ast = _seq_ast()
    with pytest.raises(BlockMutationError, match="children"):
        update_block(ast, "root.children.0", {"children": []})


def test_update_block_invalid_path() -> None:
    """Invalid path must raise BlockPathError."""
    ast = _seq_ast()
    with pytest.raises(BlockPathError):
        update_block(ast, "root.children.99", {"label": "x"})


# ---------------------------------------------------------------------------
# delete_block tests
# ---------------------------------------------------------------------------


def test_delete_leaf_node() -> None:
    """Deleting a leaf node removes it from its parent's children."""
    ast = _seq_ast()
    new_ast = delete_block(ast, "root.children.0")
    assert new_ast["root"]["children"] == []
    # original unchanged
    assert len(ast["root"]["children"]) == 1


def test_delete_composite_cascades_children() -> None:
    """Deleting a composite node removes the whole subtree."""
    ast = _nested_seq_ast()
    # root.children.0 is the inner_seq which itself has a child
    new_ast = delete_block(ast, "root.children.0")
    assert new_ast["root"]["children"] == []


def test_delete_root_raises() -> None:
    """Attempting to delete the root raises BlockMutationError."""
    ast = _minimal_ast("sequence")
    with pytest.raises(BlockMutationError, match="root"):
        delete_block(ast, "root")


def test_delete_invalid_path_raises() -> None:
    """Invalid path raises BlockPathError."""
    ast = _seq_ast()
    with pytest.raises(BlockPathError):
        delete_block(ast, "root.children.99")


# ---------------------------------------------------------------------------
# move_block tests
# ---------------------------------------------------------------------------


def _two_child_seq_ast() -> dict[str, Any]:
    """Root sequence with two agent children."""
    ast = _minimal_ast("sequence")
    ast["root"]["children"] = [
        {"id": "a0", "type": "agent", "label": "a0", "config": {"subtype": "researcher"}, "children": []},
        {"id": "a1", "type": "agent", "label": "a1", "config": {"subtype": "synthesizer"}, "children": []},
    ]
    return ast


def _parallel_and_seq_ast() -> dict[str, Any]:
    """Root sequence: [parallel([a0, a1]), sequence([a2])]."""
    ast = _minimal_ast("sequence")
    parallel: dict[str, Any] = {
        "id": "par",
        "type": "parallel",
        "label": "parallel",
        "config": {},
        "children": [
            {"id": "a0", "type": "agent", "label": "a0", "config": {}, "children": []},
            {"id": "a1", "type": "agent", "label": "a1", "config": {}, "children": []},
        ],
    }
    seq2: dict[str, Any] = {
        "id": "seq2",
        "type": "sequence",
        "label": "seq2",
        "config": {},
        "children": [
            {"id": "a2", "type": "agent", "label": "a2", "config": {}, "children": []},
        ],
    }
    ast["root"]["children"] = [parallel, seq2]
    return ast


def test_move_within_siblings() -> None:
    """Move first sibling to become last (reorder)."""
    ast = _two_child_seq_ast()
    # move a0 (index 0) into root (appended)
    new_ast = move_block(ast, "root.children.0", "root")
    children = new_ast["root"]["children"]
    assert len(children) == 2
    # After removal of index 0, a1 shifts to 0; a0 is appended at 1
    assert children[0]["id"] == "a1"
    assert children[1]["id"] == "a0"


def test_move_across_composites() -> None:
    """Move a node from one composite container to another."""
    ast = _parallel_and_seq_ast()
    # Move a1 (root.children.0.children.1) into seq2 (root.children.1)
    new_ast = move_block(ast, "root.children.0.children.1", "root.children.1")
    par = new_ast["root"]["children"][0]
    seq2 = new_ast["root"]["children"][1]
    assert len(par["children"]) == 1
    assert par["children"][0]["id"] == "a0"
    assert len(seq2["children"]) == 2
    assert seq2["children"][-1]["id"] == "a1"


def test_move_into_descendant_raises_cycle() -> None:
    """Moving a node into one of its own descendants must raise BlockMutationError."""
    ast = _nested_seq_ast()
    # root.children.0 (inner_seq) -> try moving into root.children.0.children.0
    with pytest.raises(BlockMutationError, match="descendant|cycle"):
        move_block(ast, "root.children.0", "root.children.0.children.0")


def test_move_invalid_path_raises() -> None:
    """Invalid from_path raises BlockPathError."""
    ast = _seq_ast()
    with pytest.raises(BlockPathError):
        move_block(ast, "root.children.99", "root")


# ---------------------------------------------------------------------------
# declare_tool tests
# ---------------------------------------------------------------------------


def test_declare_tool_appends() -> None:
    """declare_tool appends a new ToolDeclaration dict."""
    ast = _minimal_ast()
    new_ast = declare_tool(ast, "vector_search", "my_index", {"index": "prod.idx"})
    assert len(new_ast["tools"]) == 1
    tool = new_ast["tools"][0]
    assert tool["name"] == "my_index"
    assert tool["kind"] == "vector_search"
    assert tool["config"]["index"] == "prod.idx"
    # original unchanged
    assert ast["tools"] == []


def test_declare_tool_duplicate_name_raises() -> None:
    """Declaring a tool with a duplicate name raises BlockMutationError."""
    ast = _minimal_ast()
    new_ast = declare_tool(ast, "web_search", "brave", {})
    with pytest.raises(BlockMutationError, match="brave"):
        declare_tool(new_ast, "web_search", "brave", {})


# ---------------------------------------------------------------------------
# remove_tool tests
# ---------------------------------------------------------------------------


def test_remove_tool_existing() -> None:
    """remove_tool removes the named tool from ast['tools']."""
    ast = _minimal_ast()
    ast = declare_tool(ast, "web_search", "brave", {})
    ast["root"]["config"]["tools"] = ["brave"]
    new_ast = remove_tool(ast, "brave")
    assert new_ast["tools"] == []
    assert new_ast["root"]["config"]["tools"] == []
    # original unchanged
    assert len(ast["tools"]) == 1
    assert ast["root"]["config"]["tools"] == ["brave"]


def test_remove_tool_missing_is_noop() -> None:
    """remove_tool on a non-existent name returns ast without error."""
    ast = _minimal_ast()
    new_ast = remove_tool(ast, "nonexistent")
    assert new_ast["tools"] == []


# ---------------------------------------------------------------------------
# bind_tool_to_block tests
# ---------------------------------------------------------------------------


def test_bind_tool_to_agent() -> None:
    """bind_tool_to_block appends tool_name to agent config.tools."""
    ast = _seq_ast()
    ast = declare_tool(ast, "web_search", "brave", {})
    new_ast = bind_tool_to_block(ast, "root.children.0", "brave")
    bound = new_ast["root"]["children"][0]["config"]["tools"]
    assert "brave" in bound
    # original unchanged
    assert "tools" not in ast["root"]["children"][0].get("config", {})


def test_bind_tool_to_non_agent_raises() -> None:
    """Binding a tool to a non-agent node raises BlockMutationError."""
    ast = _minimal_ast("sequence")
    ast = declare_tool(ast, "web_search", "brave", {})
    with pytest.raises(BlockMutationError, match="agent"):
        bind_tool_to_block(ast, "root", "brave")


def test_bind_unknown_tool_raises() -> None:
    """Binding an undeclared tool raises BlockMutationError."""
    ast = _seq_ast()
    with pytest.raises(BlockMutationError, match="not declared"):
        bind_tool_to_block(ast, "root.children.0", "undeclared_tool")


# ---------------------------------------------------------------------------
# set_model_tier tests
# ---------------------------------------------------------------------------


def test_set_model_tier_valid() -> None:
    """set_model_tier sets config.model_tier on an agent node."""
    ast = _seq_ast()
    for tier in model_tiers_payload():
        new_ast = set_model_tier(ast, "root.children.0", tier)
        assert new_ast["root"]["children"][0]["config"]["model_tier"] == tier
    # original unchanged
    assert "model_tier" not in ast["root"]["children"][0]["config"]


def test_set_model_tier_invalid_tier_raises() -> None:
    """An invalid tier value raises BlockMutationError."""
    ast = _seq_ast()
    with pytest.raises(BlockMutationError, match="tier|simple|analytical"):
        set_model_tier(ast, "root.children.0", "ultra")


def test_set_model_tier_non_agent_raises() -> None:
    """set_model_tier on a non-agent node raises BlockMutationError."""
    ast = _minimal_ast("sequence")
    with pytest.raises(BlockMutationError, match="agent"):
        set_model_tier(ast, "root", "analytical")


# ---------------------------------------------------------------------------
# set_surface tests
# ---------------------------------------------------------------------------


def _ast_with_extras() -> dict[str, Any]:
    """Minimal AST that also has sibling top-level keys beyond the standard set."""
    ast = _minimal_ast()
    ast["lane_keys"] = ["lane_a", "lane_b"]
    ast["designer_signature"] = {"topology": "parallel_lanes"}
    return ast


def _minimal_surface() -> dict[str, Any]:
    """A valid minimal surface dict (version, components, data_model, bindings)."""
    return {
        "version": 1,
        "components": [
            {
                "id": "root",
                "component": "Column",
                "props": {"gap": "md"},
                "children": ["run_button"],
            },
            {
                "id": "run_button",
                "component": "Button",
                "props": {"label": "Run", "action": "run", "variant": "primary"},
                "children": [],
            },
        ],
        "data_model": {"results": {"run": None}},
        "bindings": [
            {
                "action": "run",
                "kind": "run_agent",
                "inputs": {"query": "What should I research?"},
                "options": {},
                "output": {"target": "/results/run", "mode": "report"},
                "concurrency": "replace",
            }
        ],
    }


def test_set_surface_sets_key_on_copy_preserves_siblings() -> None:
    """set_surface returns a new dict with surface set; sibling keys are preserved."""
    ast = _ast_with_extras()
    original = copy.deepcopy(ast)
    surface = _minimal_surface()

    new_ast = set_surface(ast, surface)

    # Input not mutated.
    assert ast == original
    assert "surface" not in ast

    # New AST carries the surface.
    assert new_ast["surface"] == surface

    # Sibling top-level keys are preserved.
    assert new_ast["lane_keys"] == ["lane_a", "lane_b"]
    assert new_ast["designer_signature"] == {"topology": "parallel_lanes"}

    # Standard fields still present.
    assert "root" in new_ast
    assert "tools" in new_ast


def test_set_surface_none_removes_surface() -> None:
    """set_surface(ast, None) removes an existing surface key."""
    ast = _ast_with_extras()
    ast["surface"] = _minimal_surface()

    new_ast = set_surface(ast, None)

    assert "surface" not in new_ast
    # Input not mutated.
    assert "surface" in ast
    # Other keys preserved.
    assert new_ast["lane_keys"] == ["lane_a", "lane_b"]


def test_set_surface_none_on_ast_without_surface_is_noop() -> None:
    """set_surface(ast, None) on an AST without a surface is a no-op (no error)."""
    ast = _minimal_ast()
    assert "surface" not in ast
    new_ast = set_surface(ast, None)
    assert "surface" not in new_ast


def test_set_surface_non_dict_raises_block_mutation_error() -> None:
    """Non-dict surface value raises BlockMutationError."""
    ast = _minimal_ast()
    with pytest.raises(BlockMutationError):
        set_surface(ast, "not a dict")  # type: ignore[arg-type]
    with pytest.raises(BlockMutationError):
        set_surface(ast, ["components"])  # type: ignore[arg-type]
    with pytest.raises(BlockMutationError):
        set_surface(ast, 42)  # type: ignore[arg-type]
