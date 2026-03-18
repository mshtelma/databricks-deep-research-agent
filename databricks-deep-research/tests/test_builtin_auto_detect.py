"""Tests for builtin auto-detection of input_keys from prompt templates."""

from __future__ import annotations

import pytest

import databricks_deep_research.agents.builtins  # noqa: F401 — register all builtins
from databricks_deep_research.agents.builtins.registry import get_builtin
from databricks_deep_research.agents.config import AgentNodeConfig
from databricks_deep_research.templates.renderer import SafeTemplateRenderer
from databricks_deep_research.workflow.state import WorkflowState


@pytest.mark.parametrize(
    "subtype,expected_min_keys",
    [
        ("coordinator", {"query", "conversation_history"}),
        ("background", {"query", "conversation_history"}),
        (
            "planner",
            {"query", "min_steps", "max_steps", "background"},
        ),
        (
            "researcher",
            {"query", "step_title", "step_description"},
        ),
        (
            "reflector",
            {
                "query",
                "plan_summary",
                "iteration",
                "remaining_steps",
                "total_steps",
                "steps_completed",
            },
        ),
        (
            "synthesizer",
            {
                "query",
                "research_depth",
                "min_words",
                "max_words",
                "all_observations",
                "sources_list",
                "fallback_discovery_sources",
            },
        ),
    ],
)
def test_builtin_auto_detected_keys(
    subtype: str, expected_min_keys: set[str]
) -> None:
    """Each builtin's enriched prompts produce the expected input keys."""
    config = AgentNodeConfig(subtype=subtype, input_keys=[])
    state = WorkflowState(query="test")
    builtin = get_builtin(subtype)
    assert builtin is not None
    assert builtin.enrich_config is not None

    enriched = builtin.enrich_config(config, state)
    renderer = SafeTemplateRenderer()
    detected = renderer.extract_variables(
        enriched.system_prompt
    ) | renderer.extract_variables(enriched.user_prompt_template)

    assert expected_min_keys.issubset(detected), (
        f"Missing keys for {subtype}: {expected_min_keys - detected}"
    )
