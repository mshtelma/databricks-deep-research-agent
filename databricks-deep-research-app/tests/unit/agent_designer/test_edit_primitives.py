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


# --- expected_count patch-semantics (DeerFlow skill_manage_tool.py:135) ------


def test_count_matching_nodes_counts_distinct_resolved_refs() -> None:
    ast = _ast()
    assert m.count_matching_nodes(ast, "cand-0") == 1
    assert m.count_matching_nodes(ast, ["cand-0", "root"]) == 2
    # duplicate refs to the same node count once
    assert m.count_matching_nodes(ast, ["cand-0", "cand-0"]) == 1
    # an unresolvable ref contributes zero
    assert m.count_matching_nodes(ast, "does-not-exist") == 0


def test_assert_match_count_passes_on_exact_match() -> None:
    ast = _ast()
    # No raise == pass.
    m.assert_match_count(ast, "cand-0", 1)


def test_assert_match_count_fails_loudly_on_wrong_count() -> None:
    ast = _ast()
    with pytest.raises(m.PatchCountError) as exc_info:
        m.assert_match_count(ast, "does-not-exist", 1)
    msg = str(exc_info.value)
    assert "expected to match 1" in msg
    assert "matched 0" in msg
    assert "NOT applied" in msg
    # PatchCountError is a BlockMutationError so existing handlers catch it.
    assert isinstance(exc_info.value, m.BlockMutationError)


def test_assert_match_count_rejects_negative_expected() -> None:
    ast = _ast()
    with pytest.raises(m.PatchCountError):
        m.assert_match_count(ast, "cand-0", -1)


def test_edit_update_block_with_correct_expected_count_lands() -> None:
    ast = _ast()
    out = m.edit_update_block(
        ast,
        "cand-0",
        {"config": {"system_prompt": "patched"}},
        expected_count=1,
    )
    assert out["root"]["children"][0]["config"]["system_prompt"] == "patched"


def test_edit_update_block_with_wrong_expected_count_fails_and_no_apply() -> None:
    ast = _ast()
    before = ast["root"]["children"][0]["config"]["system_prompt"]
    # The node resolves to exactly 1, so asserting 2 must fail loudly.
    with pytest.raises(m.PatchCountError, match="expected to match 2"):
        m.edit_update_block(
            ast,
            "cand-0",
            {"config": {"system_prompt": "should-not-apply"}},
            expected_count=2,
        )
    # Input AST untouched (pure functions) — patch was NOT mis-applied.
    assert ast["root"]["children"][0]["config"]["system_prompt"] == before


def test_edit_update_block_default_expected_count_is_byte_identical() -> None:
    ast = _ast()
    # Without expected_count, behavior is exactly as before (no assertion).
    out_default = m.edit_update_block(
        ast, "cand-0", {"config": {"model_tier": "complex"}}
    )
    out_explicit = m.edit_update_block(
        ast, "cand-0", {"config": {"model_tier": "complex"}}, expected_count=1
    )
    assert out_default == out_explicit
