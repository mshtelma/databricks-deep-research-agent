"""Tests for the models: section in YAML workflow loading."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from databricks_deep_research.errors import WorkflowValidationError
from databricks_deep_research.workflow.loader import (
    load_workflow,
    load_workflow_from_dict,
    load_workflow_from_string,
    save_workflow,
)


def _minimal_yaml(models: dict | None = None) -> str:
    """Build a minimal valid YAML string with optional models section."""
    data: dict = {
        "id": "test-wf",
        "name": "Test",
        "version": 1,
        "required_inputs": ["query"],
        "output_keys": ["output"],
        "root": {
            "id": "root",
            "type": "agent",
            "label": "Root",
            "config": {"subtype": "synthesizer", "output_key": "output"},
        },
    }
    if models is not None:
        data["models"] = models
    return yaml.dump(data, default_flow_style=False)


class TestModelsSection:
    def test_yaml_with_models_populates_field(self) -> None:
        """YAML with models: section -> definition.models is populated."""
        yaml_str = _minimal_yaml(models={
            "simple": "fast-model",
            "complex": {
                "endpoints": ["big-a", "big-b"],
                "fallback_on_429": True,
            },
        })
        defn = load_workflow_from_string(yaml_str)
        assert defn.models["simple"] == "fast-model"
        assert defn.models["complex"]["endpoints"] == ["big-a", "big-b"]

    def test_yaml_without_models_defaults_empty(self) -> None:
        """Existing YAML without models: -> definition.models == {}."""
        yaml_str = _minimal_yaml()
        defn = load_workflow_from_string(yaml_str)
        assert defn.models == {}

    def test_round_trip_preserves_models(self) -> None:
        """load -> save -> load preserves the models dict."""
        models = {
            "simple": "model-a",
            "analytical": {
                "endpoints": ["ep1", "ep2"],
                "fallback_on_429": True,
                "rotation_strategy": "ROUND_ROBIN",
                "tokens_per_minute": 100000,
            },
        }
        yaml_str = _minimal_yaml(models=models)
        defn = load_workflow_from_string(yaml_str)

        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            tmp_path = f.name

        save_workflow(defn, tmp_path)
        reloaded = load_workflow(tmp_path)

        assert reloaded.models["simple"] == "model-a"
        assert reloaded.models["analytical"]["endpoints"] == ["ep1", "ep2"]
        assert reloaded.models["analytical"]["rotation_strategy"] == "ROUND_ROBIN"
        Path(tmp_path).unlink(missing_ok=True)


class TestLoadFromDict:
    def test_produces_valid_definition(self) -> None:
        """Happy path: minimal valid dict -> WorkflowDefinition."""
        data = {
            "id": "test-wf",
            "name": "Test",
            "root": {
                "id": "root",
                "type": "agent",
                "label": "Root",
                "config": {"subtype": "synthesizer", "output_key": "output"},
            },
        }
        defn = load_workflow_from_dict(data)
        assert defn.id == "test-wf"
        assert defn.root.id == "root"
        assert defn.required_inputs == ["query"]

    def test_rejects_non_dict(self) -> None:
        """Type guard: non-dict raises WorkflowValidationError."""
        with pytest.raises(WorkflowValidationError, match="Expected a dict"):
            load_workflow_from_dict("not a dict")  # type: ignore[arg-type]

    def test_validates_missing_required_fields(self) -> None:
        """Missing id/name/root raises WorkflowValidationError."""
        with pytest.raises(WorkflowValidationError, match="missing required"):
            load_workflow_from_dict({"id": "x"})

    def test_round_trip_from_model_dump(self) -> None:
        """model_dump(mode='json') -> load_workflow_from_dict round-trips."""
        yaml_str = _minimal_yaml(models={"simple": "fast-model"})
        original = load_workflow_from_string(yaml_str)
        raw = original.model_dump(mode="json")
        reloaded = load_workflow_from_dict(raw)
        assert reloaded.id == original.id
        assert reloaded.name == original.name
        assert reloaded.models == original.models

    def test_extra_top_level_keys_ignored(self) -> None:
        """Unknown keys at the top level are silently ignored (forward-compat)."""
        data = {
            "id": "test-wf",
            "name": "Test",
            "root": {
                "id": "root",
                "type": "agent",
                "label": "Root",
                "config": {"subtype": "synthesizer", "output_key": "output"},
            },
            "future_field": "some value",
        }
        defn = load_workflow_from_dict(data)
        assert defn.id == "test-wf"


# ---------------------------------------------------------------------------
# heal_node_bound_web_tools — auto-declare builtin web tools a node binds but
# the workflow-level ``tools`` omits (designer scaffold / shell-app export /
# API import). Topology-agnostic; idempotent; never mutates the input dict.
# ---------------------------------------------------------------------------

import copy  # noqa: E402

from databricks_deep_research.workflow.definition import (  # noqa: E402
    NodeType,
    ToolDeclaration,
    WorkflowDefinition,
    WorkflowNode,
)
from databricks_deep_research.workflow.loader import (  # noqa: E402
    heal_node_bound_web_tools,
)


def _agent(node_id: str, *tools: str, subtype: str = "researcher") -> WorkflowNode:
    return WorkflowNode(
        id=node_id,
        type=NodeType.agent,
        label=node_id,
        config={"subtype": subtype, "output_key": f"{node_id}_out", "tools": list(tools)},
    )


def _defn(root: WorkflowNode, tools: list[ToolDeclaration] | None = None) -> WorkflowDefinition:
    return WorkflowDefinition(id="t", name="t", root=root, tools=tools or [])


def _names(defn: WorkflowDefinition) -> set[str]:
    return {t.name for t in defn.tools}


def test_heal_declares_web_research_in_parallel_lanes() -> None:
    defn = _defn(
        WorkflowNode(
            id="root",
            type=NodeType.parallel,
            label="root",
            children=[_agent("lane_1-researcher", "web_research"),
                      _agent("lane_2-researcher", "web_research")],
        )
    )
    heal_node_bound_web_tools(defn)
    assert "web_research" in _names(defn)
    wr = next(t for t in defn.tools if t.name == "web_research")
    assert wr.kind == "web_research"
    assert wr.config == {"total_results": 10, "auto_fetch_top_k": 5}


def test_heal_traverses_plan_and_execute_body_and_nested() -> None:
    # plan_and_execute keeps its body (and planner/evaluator) inside ``config``,
    # NOT under ``children`` — the heal must descend into all three.
    root = WorkflowNode(
        id="root",
        type=NodeType.plan_and_execute,
        label="pe",
        config={
            "planner": {"subtype": "planner", "tools": []},
            "evaluator": {"subtype": "reflector", "tools": []},
            "body": {
                "id": "step",
                "type": NodeType.agent.value,
                "label": "step",
                "config": {"subtype": "researcher", "tools": ["web_research"]},
            },
        },
    )
    defn = _defn(root)
    heal_node_bound_web_tools(defn)
    assert "web_research" in _names(defn)


def test_heal_traverses_loop_and_conditional_children() -> None:
    for node_type in (NodeType.loop, NodeType.conditional, NodeType.sequence):
        root = WorkflowNode(
            id="root",
            type=node_type,
            label="root",
            children=[_agent("r", "web_search"), _agent("critic", subtype="reflector")],
        )
        defn = _defn(root)
        heal_node_bound_web_tools(defn)
        assert "web_search" in _names(defn), node_type
        ws = next(t for t in defn.tools if t.name == "web_search")
        assert ws.config == {"max_results": 10}


def test_heal_skips_non_web_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    defn = _defn(_agent("r", "web_crawl", "my_corpus_index"))
    with caplog.at_level("ERROR"):
        heal_node_bound_web_tools(defn)
    assert "web_crawl" in _names(defn)
    assert "my_corpus_index" not in _names(defn)  # never invent corpus config
    assert any("WORKFLOW_HEAL_UNDECLARED_NONWEB" in r.message for r in caplog.records)


def test_heal_is_idempotent() -> None:
    defn = _defn(_agent("r", "web_research"))
    heal_node_bound_web_tools(defn)
    first = sorted(_names(defn))
    heal_node_bound_web_tools(defn)  # second pass adds nothing
    assert sorted(_names(defn)) == first
    assert len(defn.tools) == 1


def test_heal_leaves_already_declared_unchanged() -> None:
    defn = _defn(
        _agent("r", "web_research"),
        tools=[ToolDeclaration(name="web_research", kind="web_research", config={"total_results": 99})],
    )
    heal_node_bound_web_tools(defn)
    assert len(defn.tools) == 1
    assert defn.tools[0].config == {"total_results": 99}  # author config preserved


def test_load_workflow_from_dict_heals_and_does_not_mutate_input() -> None:
    data = {
        "id": "w1",
        "name": "W",
        "tools": [],
        "root": {
            "id": "root",
            "type": "parallel",
            "label": "root",
            "children": [
                {"id": "lane_1-researcher", "type": "agent", "label": "R",
                 "config": {"subtype": "researcher", "tools": ["web_research"]}},
                {"id": "lane_2-researcher", "type": "agent", "label": "R2",
                 "config": {"subtype": "researcher", "tools": ["web_research"]}},
            ],
        },
    }
    snapshot = copy.deepcopy(data)
    defn = load_workflow_from_dict(data)
    assert "web_research" in _names(defn)
    assert data == snapshot, "load_workflow_from_dict must not mutate its input dict"


def test_heal_round_trip_idempotent() -> None:
    """A first load must equal a reload of the saved output: the synthesized web
    declaration participates in source auto-population so save→load is stable."""
    data = {
        "id": "w1",
        "name": "W",
        "tools": [],
        "root": {
            "id": "root",
            "type": "parallel",
            "label": "root",
            "children": [
                {"id": "lane_1-researcher", "type": "agent", "label": "R",
                 "config": {"subtype": "researcher", "tools": ["web_research", "web_crawl"]}},
                {"id": "lane_2-researcher", "type": "agent", "label": "R2",
                 "config": {"subtype": "researcher", "tools": ["web_research"]}},
            ],
        },
    }
    wf1 = load_workflow_from_dict(data)
    wf2 = load_workflow_from_dict(wf1.model_dump(mode="json"))
    assert wf1.model_dump(mode="json") == wf2.model_dump(mode="json")
