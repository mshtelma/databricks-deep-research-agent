"""Tests for workflow definition models."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from databricks_deep_research.workflow.definition import (
    ErrorConfig,
    NodeType,
    WorkflowDefinition,
    WorkflowNode,
)


class TestNodeType:
    def test_all_eight_types_exist(self) -> None:
        expected = {
            "agent", "tool", "sequence", "parallel",
            "loop", "conditional", "subworkflow", "plan_and_execute",
        }
        assert {t.value for t in NodeType} == expected

    def test_string_coercion(self) -> None:
        assert NodeType("agent") is NodeType.agent
        assert NodeType("plan_and_execute") is NodeType.plan_and_execute


class TestWorkflowNode:
    def test_leaf_node(self) -> None:
        node = WorkflowNode(id="n1", type=NodeType.agent, label="Agent")
        assert node.id == "n1"
        assert node.type is NodeType.agent
        assert node.children == []
        assert node.config == {}

    def test_composite_node_with_children(self) -> None:
        child = WorkflowNode(id="c1", type=NodeType.agent, label="Child")
        parent = WorkflowNode(
            id="seq1",
            type=NodeType.sequence,
            label="Seq",
            children=[child],
        )
        assert len(parent.children) == 1
        assert parent.children[0].id == "c1"

    def test_deep_nesting(self) -> None:
        leaf = WorkflowNode(id="leaf", type=NodeType.agent, label="Leaf")
        loop = WorkflowNode(id="loop", type=NodeType.loop, label="Loop", children=[leaf])
        seq = WorkflowNode(id="seq", type=NodeType.sequence, label="Seq", children=[loop])
        assert seq.children[0].children[0].id == "leaf"

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            WorkflowNode(
                id="n1", type=NodeType.agent, label="Agent",
                unknown_field="bad",  # type: ignore[call-arg]
            )

    def test_config_dict(self) -> None:
        node = WorkflowNode(
            id="a1", type=NodeType.agent, label="Agent",
            config={"subtype": "researcher", "model_tier": "analytical"},
        )
        assert node.config["subtype"] == "researcher"

    def test_error_handling(self) -> None:
        node = WorkflowNode(
            id="n1", type=NodeType.agent, label="Agent",
            error_handling=ErrorConfig(on_error="skip"),
        )
        assert node.error_handling is not None
        assert node.error_handling.on_error == "skip"


class TestErrorConfig:
    def test_defaults(self) -> None:
        cfg = ErrorConfig()
        assert cfg.on_error == "fail"
        assert cfg.max_retries == 2
        assert cfg.retry_delay_seconds == 1.0

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            ErrorConfig(bad_field=True)  # type: ignore[call-arg]


class TestWorkflowDefinition:
    def _make_root(self) -> WorkflowNode:
        return WorkflowNode(id="root", type=NodeType.sequence, label="Root")

    def test_minimal_definition(self) -> None:
        defn = WorkflowDefinition(
            id="test", name="Test Workflow", root=self._make_root()
        )
        assert defn.id == "test"
        assert defn.version == 1
        assert defn.token_budget == 0
        assert defn.required_inputs == ["query"]
        assert defn.output_keys == ["output"]

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            WorkflowDefinition(
                id="test", name="Test", root=self._make_root(),
                bad_field="nope",  # type: ignore[call-arg]
            )

    def test_from_yaml_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            WorkflowDefinition.from_yaml("nonexistent.yaml")

    def test_to_yaml_roundtrip(self, tmp_path: Any) -> None:
        defn = WorkflowDefinition(
            id="test", name="Test", root=self._make_root()
        )
        out_path = tmp_path / "out.yaml"
        defn.to_yaml(str(out_path))
        assert out_path.exists()

    def test_full_tree(self) -> None:
        """Build a realistic multi-level tree and verify structure."""
        researcher = WorkflowNode(
            id="researcher", type=NodeType.agent, label="Researcher",
            config={"subtype": "researcher"},
        )
        synthesizer = WorkflowNode(
            id="synth", type=NodeType.agent, label="Synthesizer",
            config={"subtype": "synthesizer"},
        )
        loop = WorkflowNode(
            id="research_loop", type=NodeType.loop, label="Loop",
            children=[researcher],
            config={"until": {"type": "state", "key": "done", "operator": "eq", "value": True}},
        )
        root = WorkflowNode(
            id="main", type=NodeType.sequence, label="Main",
            children=[loop, synthesizer],
        )
        defn = WorkflowDefinition(
            id="deep_research", name="Deep Research", root=root,
            pools=[{"name": "sources"}, {"name": "observations"}],
            required_inputs=["query"],
            output_keys=["report"],
        )
        assert defn.root.children[0].type is NodeType.loop
        assert defn.root.children[1].config["subtype"] == "synthesizer"
        assert len(defn.pools) == 2
