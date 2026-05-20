"""Unit tests for Agent Designer revision deployability checks."""

from __future__ import annotations

from deep_research.agent_designer.deployability import (
    classify_revision_deployability,
)


def _stock_definition(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "name": "Untitled Agent",
        "description": "",
        "root": {
            "type": "sequence",
            "children": [
                {"id": "coordinator", "type": "agent", "label": "Coordinator"},
                {
                    "id": "plan-and-execute",
                    "type": "plan_and_execute",
                    "label": "Plan & Execute",
                    "config": {},
                },
                {"id": "synthesizer", "type": "agent", "label": "Synthesizer"},
            ],
        },
    }
    base.update(overrides)
    return base


def test_blocks_empty_default_scaffold() -> None:
    result = classify_revision_deployability(_stock_definition())

    assert result.deployable is False
    assert result.classification == "default_scaffold"
    assert result.root_child_ids == (
        "coordinator",
        "plan-and-execute",
        "synthesizer",
    )
    assert result.root_child_summary == [
        "coordinator:agent:Coordinator",
        "plan-and-execute:plan_and_execute:Plan & Execute",
        "synthesizer:agent:Synthesizer",
    ]


def test_allows_default_shape_with_description() -> None:
    result = classify_revision_deployability(
        _stock_definition(description="Investigate release readiness.")
    )

    assert result.deployable is True
    assert result.classification == "deployable"


def test_allows_default_shape_with_planner_guidance() -> None:
    definition = _stock_definition()
    children = definition["root"]["children"]  # type: ignore[index]
    children[1]["config"] = {  # type: ignore[index]
        "planner_guidance": "Focus the plan on migration risks.",
    }

    result = classify_revision_deployability(definition)

    assert result.deployable is True
    assert result.planner_guidance_present is True


def test_allows_default_shape_with_top_level_planner_guidance() -> None:
    result = classify_revision_deployability(
        _stock_definition(planner_guidance="Use the saved planner instructions.")
    )

    assert result.deployable is True
    assert result.planner_guidance_present is True
