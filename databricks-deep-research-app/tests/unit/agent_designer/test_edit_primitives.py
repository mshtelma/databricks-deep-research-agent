"""Unit tests for the edit-lane mutation primitives.

edit_update_block (broader allow-list than the build guard), clone_block
(sibling clone with fresh ids + output_key uniquification), and
update_workflow_meta (top-level fields update_block cannot reach).
"""

from __future__ import annotations

from typing import Any

import pytest

from deep_research.agent_designer import mutations as m


def _ast() -> dict[str, Any]:
    return {
        "name": "wf",
        "description": "d",
        "root": {
            "id": "root",
            "type": "parallel",
            "label": "candidates",
            "config": {},
            "children": [
                {
                    "id": "cand-0",
                    "type": "agent",
                    "label": "Candidate 0",
                    "config": {
                        "subtype": "synthesizer",
                        "output_key": "report",
                        "system_prompt": "draft A",
                        "tools": ["web"],
                    },
                    "children": [],
                }
            ],
        },
        "tools": [{"name": "web", "kind": "web_search"}],
    }


# --- edit_update_block ------------------------------------------------------


def test_edit_update_block_allows_output_format_where_update_block_rejects() -> None:
    ast = _ast()
    # update_block enforces the prompt-only build guard (flag default-on).
    with pytest.raises(m.BlockMutationError):
        m.update_block(ast, "cand-0", {"config": {"output_format": "json"}})
    # edit_update_block permits the full editable config surface.
    out = m.edit_update_block(ast, "cand-0", {"config": {"output_format": "markdown"}})
    assert out["root"]["children"][0]["config"]["output_format"] == "markdown"
    # untouched fields preserved (deep-merge)
    assert out["root"]["children"][0]["config"]["system_prompt"] == "draft A"


def test_edit_update_block_rejects_structural_config_keys() -> None:
    ast = _ast()
    for bad in ("subtype", "body", "children", "type"):
        with pytest.raises(m.BlockMutationError):
            m.edit_update_block(ast, "cand-0", {"config": {bad: "x"}})


def test_edit_update_block_honors_explicit_allow_list() -> None:
    ast = _ast()
    # provider is in the default edit allow-list but NOT in this explicit one.
    with pytest.raises(m.BlockMutationError):
        m.edit_update_block(
            ast,
            "cand-0",
            {"config": {"provider": "brave"}},
            allowed_config_fields={"system_prompt"},
        )
    out = m.edit_update_block(
        ast,
        "cand-0",
        {"config": {"system_prompt": "new"}},
        allowed_config_fields={"system_prompt"},
    )
    assert out["root"]["children"][0]["config"]["system_prompt"] == "new"


# --- clone_block ------------------------------------------------------------


def test_clone_block_adds_sibling_with_fresh_id_and_unique_output_key() -> None:
    ast = _ast()
    out, new_id = m.clone_block(ast, "cand-0")
    kids = out["root"]["children"]
    assert len(kids) == 2
    assert kids[0]["id"] == "cand-0"  # original preserved
    assert new_id != "cand-0" and kids[1]["id"] == new_id
    # output_key auto-uniquified to avoid collision
    assert kids[0]["config"]["output_key"] == "report"
    assert kids[1]["config"]["output_key"] == f"report_{new_id}"
    # prompt content carried over verbatim
    assert kids[1]["config"]["system_prompt"] == "draft A"


def test_clone_block_applies_overrides() -> None:
    ast = _ast()
    out, _ = m.clone_block(
        ast, "cand-0", overrides={"label": "Candidate X", "config": {"output_key": "fixed"}}
    )
    clone = out["root"]["children"][1]
    assert clone["label"] == "Candidate X"
    assert clone["config"]["output_key"] == "fixed"  # override beats auto-suffix


def test_clone_block_requires_parent_for_non_child() -> None:
    ast = _ast()
    with pytest.raises(m.BlockMutationError):
        m.clone_block(ast, "root")  # root is not a list child


# --- update_workflow_meta ---------------------------------------------------


def test_update_workflow_meta_patches_top_level_fields() -> None:
    ast = _ast()
    out = m.update_workflow_meta(ast, {"name": "Renamed", "run_as": "caller"})
    assert out["name"] == "Renamed"
    assert out["run_as"] == "caller"
    # root node untouched
    assert out["root"]["id"] == "root"


def test_update_workflow_meta_rejects_unknown_or_structural_keys() -> None:
    ast = _ast()
    for bad in ({"tools": []}, {"root": {}}, {"id": "x"}, {}):
        with pytest.raises(m.BlockMutationError):
            m.update_workflow_meta(ast, bad)
