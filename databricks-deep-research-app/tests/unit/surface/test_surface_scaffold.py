"""The deterministic default surface must be valid for ANY workflow shape.

``scaffold_surface_from_workflow`` is the zero-LLM baseline every agent can
fall back to, so it must (a) always produce a surface that passes
``validate_surface`` with no blocking findings, (b) derive its form purely
from ``required_inputs`` (topology/domain agnostic), and (c) refuse loudly
when the workflow's inputs cannot be expressed as form fields.
"""

from __future__ import annotations

from typing import Any

import pytest

from deep_research.surface import (
    has_blocking,
    scaffold_surface_from_workflow,
    validate_surface,
)

pytestmark = pytest.mark.unit


def _definition(**overrides: Any) -> dict[str, Any]:
    definition: dict[str, Any] = {
        "id": "wf",
        "name": "Account Research",
        "root": {"id": "r", "type": "sequence", "label": "x", "config": {}},
    }
    definition.update(overrides)
    return definition


def _validate(definition: dict[str, Any], surface: dict[str, Any]) -> None:
    probe = dict(definition)
    probe["surface"] = surface
    findings = validate_surface(probe)
    assert not has_blocking(findings), [
        f"{f.path}: {f.message}" for f in findings if f.severity == "blocking"
    ]


def test_default_scaffold_is_valid_and_complete() -> None:
    definition = _definition()
    surface = scaffold_surface_from_workflow(definition)
    _validate(definition, surface)

    components = {c["id"]: c for c in surface["components"]}
    assert "root" in components
    # Form + Run + Results shape.
    assert components["field_query"]["component"] == "TextArea"
    # The action button is host-rendered (layout.actions='host_bar'); there is no
    # in-tree Button component.
    assert surface["layout"]["actions"] == "host_bar"
    assert surface["bindings"][0]["action"] == "run"
    assert components["run_report"]["component"] == "ReportRegion"
    assert components["run_status"]["component"] == "StatusBadge"
    # Card title derives from the agent name.
    assert components["form_card"]["props"]["title"] == "Account Research"
    # The generated field is actually attached to the form card (regression: the
    # scaffold used to orphan its fields, so the form rendered empty).
    assert components["form_card"]["children"] == ["field_query"]

    binding = surface["bindings"][0]
    assert binding["kind"] == "run_agent"
    assert binding["inputs"] == {"query": {"path": "/form/query"}}
    assert binding["output"] == {"target": "/results/run", "mode": "report"}
    assert surface["data_model"]["form"] == {"query": ""}
    assert surface["data_model"]["results"] == {"run": None}
    # Fields are reachable from root → no "will never render" (orphan) warning.
    warnings = [
        f.message
        for f in validate_surface({**definition, "surface": surface})
        if f.severity == "warning"
    ]
    assert not any("never render" in m for m in warnings), warnings


def test_scaffold_covers_custom_required_inputs() -> None:
    definition = _definition(required_inputs=["query", "ticker", "fiscal_year"])
    surface = scaffold_surface_from_workflow(definition)
    _validate(definition, surface)

    binding = surface["bindings"][0]
    assert set(binding["inputs"]) == {"query", "ticker", "fiscal_year"}
    components = {c["id"]: c for c in surface["components"]}
    assert components["field_ticker"]["component"] == "TextField"
    assert components["field_ticker"]["props"]["label"] == "Ticker"
    assert components["field_fiscal_year"]["props"]["label"] == "Fiscal Year"
    assert components["form_card"]["children"] == [
        "field_query",
        "field_ticker",
        "field_fiscal_year",
    ]
    assert surface["data_model"]["form"] == {
        "query": "",
        "ticker": "",
        "fiscal_year": "",
    }


def test_scaffold_without_query_input() -> None:
    # Workflows may declare non-query inputs only; the scaffold must not
    # invent a query field the binding does not need.
    definition = _definition(required_inputs=["company"])
    surface = scaffold_surface_from_workflow(definition)
    _validate(definition, surface)
    binding = surface["bindings"][0]
    assert set(binding["inputs"]) == {"company"}


def test_scaffold_is_deterministic() -> None:
    definition = _definition(required_inputs=["query", "ticker"])
    assert scaffold_surface_from_workflow(definition) == scaffold_surface_from_workflow(
        definition
    )


def test_scaffold_rejects_reserved_required_input() -> None:
    with pytest.raises(ValueError, match="plan"):
        scaffold_surface_from_workflow(_definition(required_inputs=["query", "plan"]))


def test_scaffold_rejects_non_identifier_required_input() -> None:
    with pytest.raises(ValueError, match="not valid form inputs"):
        scaffold_surface_from_workflow(
            _definition(required_inputs=["query", "bad-key"])
        )


def test_scaffold_falls_back_to_query_when_inputs_missing_or_empty() -> None:
    for definition in (_definition(), _definition(required_inputs=[])):
        surface = scaffold_surface_from_workflow(definition)
        assert set(surface["bindings"][0]["inputs"]) == {"query"}


def test_scaffold_untitled_agent_gets_generic_title() -> None:
    definition = _definition(name="   ")
    surface = scaffold_surface_from_workflow(definition)
    components = {c["id"]: c for c in surface["components"]}
    assert components["form_card"]["props"]["title"] == "Run agent"
