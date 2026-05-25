"""Unit tests for the _bound_tool_names walk helper.

Verifies that the helper correctly descends into plan_and_execute body nodes
(which are full nodes with type/config/children) and agent-config dicts
(planner/evaluator) without missing tools bound under nested children.
"""
from __future__ import annotations

from tests.complex._scaffold_run_capture import _bound_tool_names


def test_bound_tool_names_descends_into_plan_and_execute_body() -> None:
    ast = {
        "root": {
            "type": "sequence",
            "config": {},
            "children": [
                {
                    "type": "plan_and_execute",
                    "config": {
                        "planner": {"tools": ["planner_tool"]},
                        "evaluator": {"tools": ["evaluator_tool"]},
                        "body": {
                            "type": "sequence",
                            "config": {},
                            "children": [
                                {
                                    "type": "conditional",
                                    "config": {},
                                    "children": [
                                        {
                                            "type": "agent",
                                            "config": {"tools": ["foo", "bar"]},
                                            "children": [],
                                        }
                                    ],
                                }
                            ],
                        },
                    },
                    "children": [],
                }
            ],
        }
    }
    assert _bound_tool_names(ast) == {"foo", "bar", "planner_tool", "evaluator_tool"}


def test_bound_tool_names_handles_simple_agent() -> None:
    ast = {
        "root": {
            "type": "agent",
            "config": {"tools": ["baz"]},
            "children": [],
        }
    }
    assert _bound_tool_names(ast) == {"baz"}


def test_bound_tool_names_empty_returns_empty() -> None:
    assert _bound_tool_names({}) == set()
