"""Unit tests for ResourceResolver (US-302/305).

Verifies AST → MLflow Resource list resolution per plan Section D.3,
including DatabricksSQLWarehouse for delta_*/table_*/compute* tool kinds.

All MLflow Resource subclasses expose ``.name`` as their identifier
attribute.
"""
from __future__ import annotations

import pytest
from mlflow.models.resources import (
    DatabricksGenieSpace,
    DatabricksServingEndpoint,
    DatabricksSQLWarehouse,
    DatabricksVectorSearchIndex,
)

from deep_research.services.deployment.resource_resolver import (
    ResourceResolver,
)


def _names_by_type(
    resources: list[object], cls: type
) -> list[str]:
    return [r.name for r in resources if isinstance(r, cls)]


class TestToolKindDispatch:
    def test_vector_search(self) -> None:
        defn = {
            "tools": [
                {"name": "kb", "kind": "vector_search", "config": {"index_name": "main.docs.kb"}}
            ]
        }
        result = ResourceResolver().resolve(defn)
        assert _names_by_type(result, DatabricksVectorSearchIndex) == ["main.docs.kb"]

    def test_genie(self) -> None:
        defn = {
            "tools": [{"name": "fin", "kind": "genie", "config": {"genie_space_id": "01ef-genie"}}]
        }
        result = ResourceResolver().resolve(defn)
        assert _names_by_type(result, DatabricksGenieSpace) == ["01ef-genie"]

    def test_knowledge_assistant(self) -> None:
        defn = {
            "tools": [
                {"name": "ka", "kind": "knowledge_assistant", "config": {"endpoint_name": "ka-ep"}}
            ]
        }
        result = ResourceResolver().resolve(defn)
        assert _names_by_type(result, DatabricksServingEndpoint) == ["ka-ep"]

    @pytest.mark.parametrize("kind", ["web_search", "web_crawl", "file_search"])
    def test_no_resource_kinds(self, kind: str) -> None:
        defn = {"tools": [{"name": kind, "kind": kind, "config": {}}]}
        result = ResourceResolver().resolve(defn)
        assert result == []

    @pytest.mark.parametrize(
        "kind",
        [
            "table_discovery",
            "table_search",
            "table_read",
            "table_neighbors",
            "table_load",
            "table_aggregate",
            "compute",
            "compute_namespace",
        ],
    )
    def test_sql_warehouse_dependent_kinds(self, kind: str) -> None:
        defn = {
            "tools": [
                {"name": "x", "kind": kind, "config": {"warehouse_id": "wh-abc"}}
            ]
        }
        result = ResourceResolver().resolve(defn)
        assert _names_by_type(result, DatabricksSQLWarehouse) == ["wh-abc"]

    def test_sql_warehouse_no_warehouse_id(self) -> None:
        # Missing warehouse_id => no resource emitted (safe fallback).
        defn = {"tools": [{"name": "x", "kind": "table_read", "config": {}}]}
        result = ResourceResolver().resolve(defn)
        assert result == []

    def test_custom_kind_no_resource(self) -> None:
        defn = {"tools": [{"name": "myplugin", "kind": "custom", "config": {}}]}
        result = ResourceResolver().resolve(defn)
        assert result == []

    def test_unknown_kind_no_resource(self) -> None:
        defn = {"tools": [{"name": "weird", "kind": "totally_made_up", "config": {}}]}
        result = ResourceResolver().resolve(defn)
        assert result == []


class TestModelEndpoints:
    def test_string_endpoint_in_models(self) -> None:
        defn = {
            "models": {
                "analytical": {"endpoints": ["databricks-claude-sonnet-4-5"]}
            }
        }
        result = ResourceResolver().resolve(defn)
        assert _names_by_type(result, DatabricksServingEndpoint) == [
            "databricks-claude-sonnet-4-5"
        ]

    def test_dict_endpoint_in_models(self) -> None:
        defn = {
            "models": {
                "simple": {"endpoints": [{"name": "databricks-claude-haiku"}]}
            }
        }
        result = ResourceResolver().resolve(defn)
        assert _names_by_type(result, DatabricksServingEndpoint) == [
            "databricks-claude-haiku"
        ]


class TestAgentNodeWalk:
    def test_root_agent_endpoint(self) -> None:
        defn = {
            "root": {
                "type": "agent",
                "config": {"endpoint": "gpt-5-pro"},
                "children": [],
            }
        }
        result = ResourceResolver().resolve(defn)
        assert _names_by_type(result, DatabricksServingEndpoint) == ["gpt-5-pro"]

    def test_nested_agent_endpoints(self) -> None:
        defn = {
            "root": {
                "type": "sequence",
                "children": [
                    {
                        "type": "agent",
                        "config": {"endpoint": "gpt-5"},
                        "children": [],
                    },
                    {
                        "type": "loop",
                        "children": [
                            {
                                "type": "agent",
                                "config": {"endpoint": "claude-opus-4"},
                                "children": [],
                            }
                        ],
                    },
                ],
            }
        }
        result = ResourceResolver().resolve(defn)
        assert sorted(
            _names_by_type(result, DatabricksServingEndpoint)
        ) == sorted(["gpt-5", "claude-opus-4"])

    def test_plan_and_execute_planner_evaluator_body_endpoints(self) -> None:
        """W11: plan_and_execute stores nested agents in config.planner /
        config.evaluator / config.body — NOT in children. Resolver must
        descend so every nested endpoint receives a grant at deploy time.
        """
        defn = {
            "root": {
                "type": "plan_and_execute",
                "config": {
                    "planner": {"endpoint": "planner-ep", "subtype": "planner"},
                    "evaluator": {
                        "endpoint": "evaluator-ep",
                        "subtype": "reflector",
                    },
                    "body": {
                        "type": "agent",
                        "config": {"endpoint": "body-ep"},
                        "children": [],
                    },
                },
                "children": [],
            }
        }
        result = ResourceResolver().resolve(defn)
        assert sorted(_names_by_type(result, DatabricksServingEndpoint)) == sorted(
            ["planner-ep", "evaluator-ep", "body-ep"]
        )

    def test_plan_and_execute_no_evaluator_is_ok(self) -> None:
        """Evaluator is optional; absence must not break the walk."""
        defn = {
            "root": {
                "type": "plan_and_execute",
                "config": {
                    "planner": {"endpoint": "planner-ep"},
                    "body": {
                        "type": "agent",
                        "config": {"endpoint": "body-ep"},
                        "children": [],
                    },
                },
                "children": [],
            }
        }
        result = ResourceResolver().resolve(defn)
        assert sorted(_names_by_type(result, DatabricksServingEndpoint)) == sorted(
            ["planner-ep", "body-ep"]
        )

    def test_plan_and_execute_body_is_composite_with_nested_agents(self) -> None:
        """Body may itself be a composite — recurse into its children too."""
        defn = {
            "root": {
                "type": "plan_and_execute",
                "config": {
                    "planner": {"endpoint": "planner-ep"},
                    "body": {
                        "type": "sequence",
                        "children": [
                            {
                                "type": "agent",
                                "config": {"endpoint": "inner-1"},
                                "children": [],
                            },
                            {
                                "type": "agent",
                                "config": {"endpoint": "inner-2"},
                                "children": [],
                            },
                        ],
                    },
                },
                "children": [],
            }
        }
        result = ResourceResolver().resolve(defn)
        assert sorted(_names_by_type(result, DatabricksServingEndpoint)) == sorted(
            ["planner-ep", "inner-1", "inner-2"]
        )


class TestDeduplication:
    def test_same_endpoint_in_models_and_agent_emitted_once(self) -> None:
        defn = {
            "models": {"a": {"endpoints": ["sonnet"]}},
            "root": {
                "type": "agent",
                "config": {"endpoint": "sonnet"},
                "children": [],
            },
        }
        result = ResourceResolver().resolve(defn)
        endpoint_names = _names_by_type(result, DatabricksServingEndpoint)
        assert endpoint_names == ["sonnet"]
        assert len(endpoint_names) == 1

    def test_distinct_resource_types_with_same_name_kept_separate(self) -> None:
        # An endpoint named "x" and a vector index named "x" must both appear.
        defn = {
            "tools": [
                {"name": "kb", "kind": "vector_search", "config": {"index_name": "x"}},
                {"name": "ka", "kind": "knowledge_assistant", "config": {"endpoint_name": "x"}},
            ]
        }
        result = ResourceResolver().resolve(defn)
        assert _names_by_type(result, DatabricksVectorSearchIndex) == ["x"]
        assert _names_by_type(result, DatabricksServingEndpoint) == ["x"]
