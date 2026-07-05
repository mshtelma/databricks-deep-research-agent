"""Deterministic default surface for any agent workflow.

``scaffold_surface_from_workflow`` derives a working Form + Run + Results
surface purely from the definition's ``required_inputs`` (plus name) — no LLM,
no topology/domain assumptions — so every agent can get a functional UI with
zero authoring risk; the Designer's LLM refines from this baseline instead of
inventing structure from scratch.
"""

from __future__ import annotations

from typing import Any

from deep_research.surface.schema import is_valid_identifier
from deep_research.surface.validation import (
    RESERVED_INPUT_KEYS,
    has_blocking,
    validate_surface,
)


def _field_label(key: str) -> str:
    return key.replace("_", " ").strip().title() or key


def _pathref(pointer: str) -> dict[str, str]:
    return {"path": pointer}


def scaffold_surface_from_workflow(definition: dict[str, Any]) -> dict[str, Any]:
    """Build the default surface dict for *definition*.

    Raises ``ValueError`` when the workflow's ``required_inputs`` cannot be
    expressed as form fields (non-identifier or pipeline-reserved keys) or when
    the generated surface unexpectedly fails validation — the scaffold must
    never emit an invalid surface.
    """
    raw_required = definition.get("required_inputs")
    if isinstance(raw_required, list):
        required = [k for k in raw_required if isinstance(k, str) and k]
    else:
        required = []
    if not required:
        required = ["query"]

    bad_keys = [
        k
        for k in required
        if k != "query" and (not is_valid_identifier(k) or k in RESERVED_INPUT_KEYS)
    ]
    if bad_keys:
        raise ValueError(
            "cannot scaffold a surface: required_inputs contain keys that are "
            f"not valid form inputs: {', '.join(sorted(bad_keys))}"
        )

    components: list[dict[str, Any]] = []
    form_children: list[str] = []
    form_values: dict[str, Any] = {}
    binding_inputs: dict[str, Any] = {}

    for key in required:
        field_id = f"field_{key}"
        pointer = f"/form/{key}"
        component = "TextArea" if key == "query" else "TextField"
        props: dict[str, Any] = {
            "label": "Research request" if key == "query" else _field_label(key),
            "value": _pathref(pointer),
        }
        if key == "query":
            props["placeholder"] = "What should this agent investigate?"
        components.append(
            {"id": field_id, "component": component, "props": props, "children": []}
        )
        form_children.append(field_id)
        form_values[key] = ""
        binding_inputs[key] = _pathref(pointer)

    agent_name = definition.get("name")
    title = agent_name if isinstance(agent_name, str) and agent_name.strip() else "Run agent"

    components.extend(
        [
            {
                "id": "root",
                "component": "Column",
                "props": {"gap": "md"},
                "children": ["form_card", "results_card"],
            },
            {
                "id": "form_card",
                "component": "Card",
                "props": {"title": title},
                "children": form_children,
            },
            {
                "id": "results_card",
                "component": "Card",
                "props": {"title": "Results"},
                "children": ["run_status", "run_report"],
            },
            {
                "id": "run_status",
                "component": "StatusBadge",
                "props": {"source": _pathref("/results/run")},
                "children": [],
            },
            {
                "id": "run_report",
                "component": "ReportRegion",
                "props": {
                    "source": _pathref("/results/run"),
                    "empty_text": "Run the agent to see results here.",
                },
                "children": [],
            },
        ]
    )

    surface: dict[str, Any] = {
        "version": 1,
        "components": components,
        "data_model": {
            "form": form_values,
            "results": {"run": None},
        },
        "bindings": [
            {
                "action": "run",
                "kind": "run_agent",
                "inputs": binding_inputs,
                "options": {},
                "output": {"target": "/results/run", "mode": "report"},
                "concurrency": "replace",
            }
        ],
        "runtime_controls": {
            "effort": "show",
            "sources": "show",
            "verify_sources": "advanced",
            "plan_review": "advanced",
            "report_style": "advanced",
            "cross_session_memory": "advanced",
            "live_search": "advanced",
        },
        "layout": {
            "actions": "host_bar",
            "sections": [
                {
                    "id": "inputs",
                    "title": "Inputs",
                    "role": "inputs",
                    "children": ["form_card"],
                    "default_open": "before_first_run",
                },
                {
                    "id": "results",
                    "title": "Results",
                    "role": "results",
                    "children": ["results_card"],
                    "default_open": "after_run",
                },
            ],
        },
    }

    probe = dict(definition)
    probe["surface"] = surface
    findings = validate_surface(probe)
    if has_blocking(findings):
        detail = "; ".join(
            f"{f.path or '<surface>'}: {f.message}"
            for f in findings
            if f.severity == "blocking"
        )
        raise ValueError(f"scaffolded surface failed validation: {detail}")
    return surface


__all__ = ["scaffold_surface_from_workflow"]
