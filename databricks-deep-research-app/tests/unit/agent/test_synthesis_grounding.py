"""Unit tests for the shared synthesizer cite-vs-verify stamper.

``apply_grounding_to_synth_config`` is the single source of truth used by both
``config_translator._build_synthesizer`` and the custom-agent verify override in
``framework_orchestrator._apply_runtime_overlays_to_workflow``.
"""
from __future__ import annotations

from typing import Any

import pytest

from deep_research.agent.synthesis_grounding import apply_grounding_to_synth_config


def test_full_verify_true_stamps_reclaim() -> None:
    node: dict[str, Any] = {"subtype": "synthesizer"}
    apply_grounding_to_synth_config(node, full_verify=True)

    assert node["grounding_mode"] == "reclaim"
    schema = node["output_schema"]
    assert schema["synthesis_mode"] == "interleaved"
    assert schema["enable_citation_verification"] is True
    assert schema["enable_isolated_verification"] is True
    # correction/numeric are left to the framework defaults (True) for reclaim.
    assert "enable_citation_correction" not in schema
    assert "enable_numeric_qa_verification" not in schema


def test_full_verify_false_stamps_classical_lite() -> None:
    node: dict[str, Any] = {"subtype": "synthesizer"}
    apply_grounding_to_synth_config(node, full_verify=False)

    assert node["grounding_mode"] == "classical_lite"
    schema = node["output_schema"]
    assert schema["enable_citation_verification"] is True
    assert schema["enable_isolated_verification"] is False
    assert schema["enable_citation_correction"] is False
    assert schema["enable_numeric_qa_verification"] is False


def test_preserves_unrelated_output_schema_keys() -> None:
    node: dict[str, Any] = {
        "subtype": "synthesizer",
        "output_schema": {"max_tokens": 4096, "target_word_count": 1200},
    }
    apply_grounding_to_synth_config(node, full_verify=False)

    schema = node["output_schema"]
    assert schema["max_tokens"] == 4096
    assert schema["target_word_count"] == 1200
    assert schema["enable_isolated_verification"] is False


def test_reclaim_to_classical_lite_disables_overlay() -> None:
    """Re-stamping a baked-reclaim node off → classical_lite + flags fully disabled."""
    node: dict[str, Any] = {
        "subtype": "synthesizer",
        "grounding_mode": "reclaim",
        "output_schema": {"enable_isolated_verification": True},
    }
    apply_grounding_to_synth_config(node, full_verify=False)

    assert node["grounding_mode"] == "classical_lite"
    assert node["output_schema"]["enable_isolated_verification"] is False
    assert node["output_schema"]["enable_citation_correction"] is False


def test_classical_lite_to_reclaim_clears_stale_disables() -> None:
    """Re-stamping a classical_lite node on → reclaim must clear the stale False
    correction/numeric flags so they fall back to the framework defaults (True)."""
    node: dict[str, Any] = {
        "subtype": "synthesizer",
        "grounding_mode": "classical_lite",
        "output_schema": {
            "enable_isolated_verification": False,
            "enable_citation_correction": False,
            "enable_numeric_qa_verification": False,
        },
    }
    apply_grounding_to_synth_config(node, full_verify=True)

    assert node["grounding_mode"] == "reclaim"
    assert node["output_schema"]["enable_isolated_verification"] is True
    assert "enable_citation_correction" not in node["output_schema"]
    assert "enable_numeric_qa_verification" not in node["output_schema"]


def test_preserves_author_synthesis_mode() -> None:
    node: dict[str, Any] = {
        "subtype": "synthesizer",
        "output_schema": {"synthesis_mode": "custom_mode"},
    }
    apply_grounding_to_synth_config(node, full_verify=True)
    assert node["output_schema"]["synthesis_mode"] == "custom_mode"


@pytest.mark.parametrize("full_verify", [True, False])
def test_creates_output_schema_when_missing(full_verify: bool) -> None:
    node: dict[str, Any] = {"subtype": "synthesizer"}
    apply_grounding_to_synth_config(node, full_verify=full_verify)
    assert isinstance(node["output_schema"], dict)
