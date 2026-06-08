"""ID-based addressing + smart errors for mutations.py.

Each mutation entry point (``update_block``, ``bind_tool_to_block``,
``set_model_tier``, ``delete_block``, ``move_block``) accepts EITHER a
semantic ``node.id`` (e.g. ``"lane-fundamentals"``) or a dot-notation
indexed path (e.g. ``"root.children.1.children.0"``). On a miss, the
``BlockPathError`` message includes the list of available IDs plus the
closest-match suggestion so LLM designers can self-correct.
"""

from __future__ import annotations

from typing import Any

import pytest

from deep_research.agent_designer.mutations import (
    BlockPathError,
    _collect_id_index,
    _resolve_node_ref,
    bind_tool_to_block,
    delete_block,
    move_block,
    set_model_tier,
    update_block,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_research_ast() -> dict[str, Any]:
    """Multi-lane research AST used by most tests."""
    return {
        "root": {
            "id": "root",
            "type": "sequence",
            "label": "Top",
            "config": {},
            "children": [
                {
                    "id": "coordinator",
                    "type": "agent",
                    "label": "Coord",
                    "config": {"subtype": "coordinator"},
                    "children": [],
                },
                {
                    "id": "parallel-research",
                    "type": "parallel",
                    "label": "Lanes",
                    "config": {},
                    "children": [
                        {
                            "id": "lane-fundamentals",
                            "type": "agent",
                            "label": "Fundamentals",
                            "config": {
                                "subtype": "researcher",
                                "model_tier": "analytical",
                                "tools": [],
                            },
                            "children": [],
                        },
                        {
                            "id": "lane-risk",
                            "type": "agent",
                            "label": "Risk",
                            "config": {
                                "subtype": "researcher",
                                "model_tier": "analytical",
                                "tools": [],
                            },
                            "children": [],
                        },
                    ],
                },
                {
                    "id": "synthesizer",
                    "type": "agent",
                    "label": "Synth",
                    "config": {
                        "subtype": "synthesizer",
                        "model_tier": "complex",
                    },
                    "children": [],
                },
            ],
        },
        "tools": [
            {"name": "web_search", "kind": "web_search", "config": {}},
            {"name": "web_crawl", "kind": "web_crawl", "config": {}},
        ],
        "pools": [],
    }


def _make_plan_and_execute_ast() -> dict[str, Any]:
    """Plan-and-execute body + evaluator are addressable by ID too."""
    return {
        "root": {
            "id": "pe-root",
            "type": "plan_and_execute",
            "label": "PE",
            "config": {
                "body": {
                    "id": "body-runner",
                    "type": "agent",
                    "label": "Runner",
                    "config": {
                        "subtype": "researcher",
                        "model_tier": "analytical",
                        "tools": [],
                    },
                    "children": [],
                },
                "evaluator": {
                    "id": "body-eval",
                    "type": "agent",
                    "label": "Eval",
                    "config": {
                        "subtype": "reflector",
                        "model_tier": "analytical",
                    },
                    "children": [],
                },
            },
            "children": [],
        },
        "tools": [],
        "pools": [],
    }


# ---------------------------------------------------------------------------
# _collect_id_index
# ---------------------------------------------------------------------------


class TestCollectIdIndex:
    def test_maps_every_top_level_node(self) -> None:
        ast = _make_research_ast()
        idx = _collect_id_index(ast)
        assert idx["root"] == "root"
        assert idx["coordinator"] == "root.children.0"
        assert idx["parallel-research"] == "root.children.1"
        assert idx["lane-fundamentals"] == "root.children.1.children.0"
        assert idx["lane-risk"] == "root.children.1.children.1"
        assert idx["synthesizer"] == "root.children.2"

    def test_maps_plan_and_execute_body_and_evaluator(self) -> None:
        ast = _make_plan_and_execute_ast()
        idx = _collect_id_index(ast)
        assert idx["pe-root"] == "root"
        assert idx["body-runner"] == "root.config.body"
        assert idx["body-eval"] == "root.config.evaluator"

    def test_empty_ast_returns_empty_map(self) -> None:
        assert _collect_id_index({}) == {}
        assert _collect_id_index({"root": None}) == {}
        assert _collect_id_index({"not-root": {}}) == {}


# ---------------------------------------------------------------------------
# _resolve_node_ref — direct unit tests
# ---------------------------------------------------------------------------


class TestResolveNodeRef:
    def test_dot_path_resolves_verbatim(self) -> None:
        ast = _make_research_ast()
        assert (
            _resolve_node_ref(ast, "root.children.1.children.0")
            == "root.children.1.children.0"
        )

    def test_node_id_resolves_to_indexed_path(self) -> None:
        ast = _make_research_ast()
        assert (
            _resolve_node_ref(ast, "lane-fundamentals")
            == "root.children.1.children.0"
        )
        assert _resolve_node_ref(ast, "synthesizer") == "root.children.2"

    def test_root_id_resolves_to_root(self) -> None:
        ast = _make_research_ast()
        assert _resolve_node_ref(ast, "root") == "root"

    def test_empty_ref_raises(self) -> None:
        ast = _make_research_ast()
        with pytest.raises(BlockPathError, match="Empty path/ref"):
            _resolve_node_ref(ast, "")

    def test_unknown_ref_lists_available_ids(self) -> None:
        ast = _make_research_ast()
        with pytest.raises(BlockPathError) as exc_info:
            _resolve_node_ref(ast, "bogus-node")
        msg = str(exc_info.value)
        assert "Available node ids" in msg
        assert "lane-fundamentals" in msg
        assert "synthesizer" in msg

    def test_typo_includes_closest_match_suggestion(self) -> None:
        ast = _make_research_ast()
        with pytest.raises(BlockPathError) as exc_info:
            _resolve_node_ref(ast, "lane-fundamental")  # typo: missing 's'
        msg = str(exc_info.value)
        assert "Did you mean 'lane-fundamentals'?" in msg
        # Suggestion includes the resolved indexed path so the LLM has both
        # paste-friendly options.
        assert "root.children.1.children.0" in msg

    def test_empty_ast_hint_mentions_propose_workflow(self) -> None:
        with pytest.raises(BlockPathError) as exc_info:
            _resolve_node_ref({}, "anything")
        assert "propose_workflow" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Mutation entry points — each accepts id OR dot-path
# ---------------------------------------------------------------------------


class TestUpdateBlock:
    def test_by_id(self) -> None:
        ast = _make_research_ast()
        new_ast = update_block(
            ast, "lane-fundamentals", {"label": "Updated Fundamentals"}
        )
        assert new_ast["root"]["children"][1]["children"][0]["label"] == "Updated Fundamentals"

    def test_by_dot_path(self) -> None:
        ast = _make_research_ast()
        new_ast = update_block(
            ast, "root.children.0", {"label": "Updated Coordinator"}
        )
        assert new_ast["root"]["children"][0]["label"] == "Updated Coordinator"

    def test_smart_error_on_typo(self) -> None:
        ast = _make_research_ast()
        with pytest.raises(BlockPathError) as exc_info:
            update_block(ast, "lane-fundamntls", {"label": "x"})
        assert "Did you mean 'lane-fundamentals'?" in str(exc_info.value)


class TestBindToolToBlock:
    def test_by_id(self) -> None:
        ast = _make_research_ast()
        new_ast = bind_tool_to_block(ast, "lane-fundamentals", "web_search")
        bound = new_ast["root"]["children"][1]["children"][0]["config"]["tools"]
        assert bound == ["web_search"]

    def test_by_dot_path(self) -> None:
        ast = _make_research_ast()
        new_ast = bind_tool_to_block(
            ast, "root.children.1.children.0", "web_crawl"
        )
        bound = new_ast["root"]["children"][1]["children"][0]["config"]["tools"]
        assert bound == ["web_crawl"]


class TestSetModelTier:
    def test_by_id(self) -> None:
        ast = _make_research_ast()
        new_ast = set_model_tier(ast, "synthesizer", "complex")
        assert new_ast["root"]["children"][2]["config"]["model_tier"] == "complex"

    def test_by_dot_path(self) -> None:
        ast = _make_research_ast()
        new_ast = set_model_tier(ast, "root.children.2", "complex")
        assert new_ast["root"]["children"][2]["config"]["model_tier"] == "complex"


class TestDeleteBlock:
    def test_by_id(self) -> None:
        ast = _make_research_ast()
        new_ast = delete_block(ast, "lane-risk")
        lanes = new_ast["root"]["children"][1]["children"]
        assert len(lanes) == 1
        assert lanes[0]["id"] == "lane-fundamentals"

    def test_by_dot_path(self) -> None:
        ast = _make_research_ast()
        new_ast = delete_block(ast, "root.children.1.children.1")
        assert len(new_ast["root"]["children"][1]["children"]) == 1


class TestMoveBlock:
    def test_by_id_both_endpoints(self) -> None:
        ast = _make_research_ast()
        new_ast = move_block(
            ast,
            from_path="lane-risk",
            to_path="root",
        )
        # lane-risk should now be a top-level child
        top_ids = [c["id"] for c in new_ast["root"]["children"]]
        assert "lane-risk" in top_ids


# ---------------------------------------------------------------------------
# Plan-and-execute body/evaluator addressing
# ---------------------------------------------------------------------------


class TestPlanAndExecuteAddressing:
    def test_body_id_resolves(self) -> None:
        ast = _make_plan_and_execute_ast()
        new_ast = update_block(ast, "body-runner", {"label": "New body label"})
        assert new_ast["root"]["config"]["body"]["label"] == "New body label"

    def test_evaluator_id_resolves(self) -> None:
        ast = _make_plan_and_execute_ast()
        new_ast = update_block(ast, "body-eval", {"label": "New eval label"})
        assert new_ast["root"]["config"]["evaluator"]["label"] == "New eval label"


# ---------------------------------------------------------------------------
# Backward-compat: existing dot-path callers must NOT regress
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    """A regression-only suite. If any of these break, the id-resolution
    helper broke the existing dot-path contract."""

    def test_root_path(self) -> None:
        ast = _make_research_ast()
        new_ast = update_block(ast, "root", {"label": "New root"})
        assert new_ast["root"]["label"] == "New root"

    def test_deep_path(self) -> None:
        ast = _make_research_ast()
        new_ast = update_block(
            ast,
            "root.children.1.children.1",
            {"label": "Renamed Risk"},
        )
        assert new_ast["root"]["children"][1]["children"][1]["label"] == "Renamed Risk"

    def test_unknown_dot_path_raises_with_hint(self) -> None:
        ast = _make_research_ast()
        with pytest.raises(BlockPathError):
            update_block(ast, "root.children.99", {"label": "x"})
