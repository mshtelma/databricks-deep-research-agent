"""Tests for the three new structural detectors added in US-04."""
from __future__ import annotations

from deep_research.agent_designer.semantic_validation import (
    detect_generic_reflector_prompt,
    detect_generic_synthesizer_prompt,
    detect_unspecialized_fallback_researcher,
)


def _legacy_ast() -> dict:  # type: ignore[type-arg]
    return {
        "description": (
            "Build a workflow covering valuation, fundamentals, dividend "
            "yields, balance sheet strength, and growth drivers."
        ),
        "root": {
            "id": "root",
            "type": "sequence",
            "children": [
                {
                    "id": "synth",
                    "type": "agent",
                    "config": {
                        "subtype": "synthesizer",
                        "system_prompt": "You summarize the research.",
                        "user_prompt_template": "Write the final answer.",
                    },
                },
                {
                    "id": "reflector",
                    "type": "agent",
                    "config": {
                        "subtype": "reflector",
                        "system_prompt": "You review whether work is complete.",
                        "user_prompt_template": "Approve or request changes.",
                    },
                },
                {
                    "id": "router",
                    "type": "conditional",
                    "config": {"conditions": [{"type": "always"}]},
                    "children": [
                        {
                            "id": "specialized-researcher",
                            "type": "agent",
                            "config": {
                                "subtype": "researcher",
                                "system_prompt": (
                                    "You are a specialized researcher for "
                                    "valuation evidence."
                                ),
                            },
                        },
                        {
                            "id": "fallback-researcher",
                            "type": "agent",
                            "config": {
                                "subtype": "researcher",
                                "system_prompt": (
                                    "You are the Researcher agent for a deep "
                                    "research system. Your role is to execute "
                                    "individual research steps."
                                ),
                            },
                        },
                    ],
                },
            ],
        },
    }


def test_synthesizer_detector_flags_legacy_artifact() -> None:
    legacy_ast = _legacy_ast()
    errors = detect_generic_synthesizer_prompt(legacy_ast)
    # Legacy artifact has generic synthesizer prompt — should flag at least once.
    assert len(errors) >= 1, f"Expected at least 1 error, got {errors}"
    assert errors[0].kind == "unspecialized_synthesizer"


def test_reflector_detector_flags_legacy_artifact() -> None:
    legacy_ast = _legacy_ast()
    errors = detect_generic_reflector_prompt(legacy_ast)
    assert len(errors) >= 1, f"Expected at least 1 error, got {errors}"
    assert errors[0].kind == "unspecialized_reflector"


def test_fallback_researcher_detector_flags_legacy_artifact() -> None:
    legacy_ast = _legacy_ast()
    errors = detect_unspecialized_fallback_researcher(legacy_ast)
    assert len(errors) >= 1, f"Expected at least 1 error, got {errors}"
    assert errors[0].kind == "unspecialized_fallback_researcher"


def test_empty_description_does_not_crash() -> None:
    ast = {
        "description": "",
        "root": {
            "type": "agent",
            "config": {"subtype": "synthesizer", "system_prompt": "x"},
        },
    }
    # No nouns extractable → detector should silently return empty.
    errors = detect_generic_synthesizer_prompt(ast)
    assert errors == []


def test_specialized_synthesizer_passes() -> None:
    ast = {
        "description": (
            "Build an investment research assistant for company fundamentals "
            "and valuation analysis with risk assessment."
        ),
        "root": {
            "type": "agent",
            "config": {
                "subtype": "synthesizer",
                "system_prompt": (
                    "You write investment reports covering fundamentals, "
                    "valuation, and risk for the company."
                ),
                "user_prompt_template": (
                    "Synthesize findings on fundamentals and valuation."
                ),
            },
        },
    }
    errors = detect_generic_synthesizer_prompt(ast)
    assert errors == [], f"Expected no errors, got {errors}"


def test_empty_ast_does_not_crash() -> None:
    for detector in (
        detect_generic_synthesizer_prompt,
        detect_generic_reflector_prompt,
        detect_unspecialized_fallback_researcher,
    ):
        assert detector({}) == []
        assert detector({"root": None}) == []
