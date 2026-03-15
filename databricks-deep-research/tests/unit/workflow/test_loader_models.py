"""Tests for the models: section in YAML workflow loading."""

from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

from databricks_deep_research.workflow.loader import (
    load_workflow,
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
