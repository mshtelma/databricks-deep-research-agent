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
