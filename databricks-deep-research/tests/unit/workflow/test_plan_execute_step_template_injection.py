"""Phase 1.3 tests — plan_execute_runner injects the step's
``user_prompt_template`` into the body researcher's config before
execution, mirroring what the workflow_builder does for static lanes.

Without this, planner-emitted per-step briefs never reach the runtime
researcher; the lane keeps using the generic builtin and produces the
same planning-text findings the NVDA trace exposed.
"""
from __future__ import annotations

import copy

from databricks_deep_research.workflow.runtime.plan_execute_runner import (
    _extract_step_user_prompt_template,
    _materialize_body_for_step,
    _patch_researcher_user_prompt,
)

_DEFAULT_RESEARCHER_BODY = {
    "id": "lane-researcher",
    "type": "agent",
    "label": "Lane Researcher",
    "config": {
        "subtype": "researcher",
        "system_prompt": "You are a researcher.",
        "user_prompt_template": "(default)",
    },
    "children": [],
}


def test_extract_returns_none_for_missing_field() -> None:
    assert _extract_step_user_prompt_template({"id": "step-1", "title": "x"}) is None
    assert _extract_step_user_prompt_template({}) is None
    assert _extract_step_user_prompt_template(None) is None


def test_extract_returns_none_for_empty_field() -> None:
    assert _extract_step_user_prompt_template({"user_prompt_template": ""}) is None
    assert _extract_step_user_prompt_template({"user_prompt_template": "   "}) is None
    assert _extract_step_user_prompt_template({"user_prompt_template": None}) is None


def test_extract_strips_whitespace() -> None:
    out = _extract_step_user_prompt_template(
        {"user_prompt_template": "  ## Brief\n\nbody  "}
    )
    assert out == "## Brief\n\nbody"


def test_extract_pulls_from_pydantic_like_object() -> None:
    class FakeStep:
        user_prompt_template = "## From attr\n\n{query}"

    assert (
        _extract_step_user_prompt_template(FakeStep())
        == "## From attr\n\n{query}"
    )


def test_patch_replaces_researcher_template_in_place() -> None:
    body = copy.deepcopy(_DEFAULT_RESEARCHER_BODY)
    _patch_researcher_user_prompt(body, "## Step Brief\n\n{query}")
    assert body["config"]["user_prompt_template"] == "## Step Brief\n\n{query}"


def test_patch_does_not_touch_non_researcher_agents() -> None:
    body = {
        "id": "synth",
        "type": "agent",
        "label": "Synthesizer",
        "config": {
            "subtype": "synthesizer",
            "user_prompt_template": "(synth default)",
        },
        "children": [],
    }
    _patch_researcher_user_prompt(body, "## Step Brief")
    # Synthesizer is untouched.
    assert body["config"]["user_prompt_template"] == "(synth default)"


def test_patch_recurses_into_conditional_children() -> None:
    # plan_and_execute body is typically a conditional that dispatches to
    # one of N lane researchers — the injection must reach every lane.
    body = {
        "id": "lane-router",
        "type": "conditional",
        "label": "Router",
        "config": {"conditions": [], "default_branch": 1},
        "children": [
            copy.deepcopy(_DEFAULT_RESEARCHER_BODY),
            copy.deepcopy(_DEFAULT_RESEARCHER_BODY),
        ],
    }
    _patch_researcher_user_prompt(body, "## Step Brief")
    for child in body["children"]:
        assert child["config"]["user_prompt_template"] == "## Step Brief"


def test_materialize_no_template_returns_original_object() -> None:
    body = copy.deepcopy(_DEFAULT_RESEARCHER_BODY)
    out = _materialize_body_for_step(body, {"id": "step-1", "title": "x"})
    # Same identity — zero-overhead path when the planner did not emit a
    # template.
    assert out is body
    assert out["config"]["user_prompt_template"] == "(default)"


def test_materialize_with_template_deep_copies_and_overrides() -> None:
    body = copy.deepcopy(_DEFAULT_RESEARCHER_BODY)
    step = {"id": "step-1", "user_prompt_template": "## Step Brief\n\n{query}"}
    out = _materialize_body_for_step(body, step)
    # Original body is unchanged (deep copy semantics).
    assert body["config"]["user_prompt_template"] == "(default)"
    assert out is not body
    assert out["config"]["user_prompt_template"] == "## Step Brief\n\n{query}"


def test_materialize_handles_pydantic_step_object() -> None:
    body = copy.deepcopy(_DEFAULT_RESEARCHER_BODY)

    class FakeStep:
        user_prompt_template = "## From step model\n\n{query}"

    out = _materialize_body_for_step(body, FakeStep())
    assert out["config"]["user_prompt_template"] == "## From step model\n\n{query}"
