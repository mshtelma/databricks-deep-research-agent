"""Unit tests for the shared AST-introspection helpers."""

from __future__ import annotations

from deep_research.agent_designer.ast_introspection import (
    config_of,
    is_lane_researcher,
    iter_agent_nodes,
    iter_all_nodes,
    tool_kinds_for_lane,
    topology_of_ast,
    topology_of_node,
)


def test_config_of_returns_dict_or_empty() -> None:
    assert config_of({"config": {"a": 1}}) == {"a": 1}
    assert config_of({"config": None}) == {}
    assert config_of({}) == {}
    assert config_of("not a node") == {}


def test_iter_agent_nodes_walks_children_and_body() -> None:
    ast_root = {
        "type": "sequence",
        "children": [
            {"type": "agent", "id": "coordinator"},
            {
                "type": "parallel",
                "children": [
                    {"type": "agent", "id": "lane_0"},
                    {"type": "agent", "id": "lane_1"},
                ],
            },
            {
                "type": "plan_and_execute",
                "config": {"body": {"type": "agent", "id": "in-body"}},
            },
        ],
    }
    ids = [n["id"] for n in iter_agent_nodes(ast_root)]
    assert ids == ["coordinator", "lane_0", "lane_1", "in-body"]


def test_iter_all_nodes_includes_composites() -> None:
    ast_root = {
        "type": "sequence",
        "children": [{"type": "parallel", "children": [{"type": "agent", "id": "a"}]}],
    }
    types = [n["type"] for n in iter_all_nodes(ast_root)]
    assert types == ["sequence", "parallel", "agent"]


def test_tool_kinds_for_lane_resolves_by_name() -> None:
    lane = {"config": {"tools": ["web", "vec"]}}
    tools = [
        {"name": "web", "kind": "web_search"},
        {"name": "vec", "kind": "vector_search"},
        {"name": "unused", "kind": "compute"},
    ]
    assert tool_kinds_for_lane(lane, tools) == {"web_search", "vector_search"}


def test_tool_kinds_for_lane_ignores_unknown_names() -> None:
    lane = {"config": {"tools": ["ghost"]}}
    assert tool_kinds_for_lane(lane, [{"name": "web", "kind": "web_search"}]) == set()


def test_is_lane_researcher_by_subtype_or_id() -> None:
    assert is_lane_researcher({"config": {"subtype": "researcher"}})
    assert is_lane_researcher({"id": "lane_3-researcher"})
    assert not is_lane_researcher({"id": "synthesizer", "config": {"subtype": "synthesizer"}})
    assert not is_lane_researcher("nope")


def test_topology_single_agent() -> None:
    ast = {"root": {"type": "sequence", "children": [{"type": "agent", "id": "a"}]}}
    assert topology_of_ast(ast) == "single_agent"


def test_topology_parallel_lanes() -> None:
    ast = {
        "root": {
            "type": "sequence",
            "children": [
                {"type": "agent", "id": "coord"},
                {"type": "parallel", "children": [{"type": "agent", "id": "lane_0"}]},
            ],
        }
    }
    assert topology_of_ast(ast) == "parallel_lanes"


def test_topology_plan_and_execute() -> None:
    ast = {"root": {"type": "plan_and_execute", "config": {"body": {}}}}
    assert topology_of_ast(ast) == "plan_and_execute"


def test_topology_router_not_misread_as_parallel() -> None:
    # A conditional with an evidence parallel INSIDE a branch must still
    # classify as router, not parallel_lanes.
    ast = {
        "root": {
            "type": "sequence",
            "children": [
                {"type": "agent", "id": "classifier"},
                {
                    "type": "conditional",
                    "children": [
                        {
                            "type": "sequence",
                            "children": [
                                {"type": "parallel", "children": [{"type": "agent", "id": "b0"}]}
                            ],
                        }
                    ],
                },
            ],
        }
    }
    assert topology_of_ast(ast) == "router"


def test_topology_unknown_for_empty_or_malformed() -> None:
    assert topology_of_ast({}) == "unknown"
    assert topology_of_ast({"root": "nope"}) == "unknown"
    assert topology_of_node({}) == "unknown"
