"""Validation matrix for structured-output components.

Exercises `_validate_output_components` through the public
``validate_surface`` entry point: Tabs/TabPane structure, Table column
rules, Chart key grammar, the one-segment slot pointer rule, slot contract
conflicts, and the new string_list/object_list prop kinds.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from deep_research.surface import validate_surface
from deep_research.surface.validation import SurfaceValidationError, has_blocking

pytestmark = pytest.mark.unit


def _definition(surface: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "wf",
        "name": "Test workflow",
        "version": 1,
        "required_inputs": ["query"],
        "surface": surface,
    }


def _surface() -> dict[str, Any]:
    return {
        "version": 1,
        "components": [
            {"id": "root", "component": "Column", "props": {}, "children": [
                "run_btn", "tabs",
            ]},
            {"id": "run_btn", "component": "Button",
             "props": {"label": "Run", "action": "run"}, "children": []},
            {"id": "tabs", "component": "Tabs", "props": {},
             "children": ["pane_a", "pane_b"]},
            {"id": "pane_a", "component": "TabPane",
             "props": {"label": "Overview"},
             "children": ["metrics", "findings"]},
            {"id": "pane_b", "component": "TabPane",
             "props": {"label": "Details"}, "children": ["tbl", "cht"]},
            {"id": "metrics", "component": "MetricGrid",
             "props": {"source": {"path": "/results/run/data/headline_metrics"}},
             "children": []},
            {"id": "findings", "component": "KeyFindings",
             "props": {"source": {"path": "/results/run/data/key_findings"}},
             "children": []},
            {"id": "tbl", "component": "Table",
             "props": {
                 "source": {"path": "/results/run/data/comparison"},
                 "columns": [
                     {"key": "item", "label": "Item", "type": "string"},
                     {"key": "score", "label": "Score", "type": "number"},
                 ],
             },
             "children": []},
            {"id": "cht", "component": "Chart",
             "props": {
                 "source": {"path": "/results/run/data/comparison"},
                 "kind": "bar", "x_key": "item", "y_keys": ["score"],
             },
             "children": []},
        ],
        "data_model": {"query": ""},
        "bindings": [
            {"action": "run", "kind": "run_agent",
             "inputs": {"query": {"path": "/query"}}, "options": {},
             "output": {"target": "/results/run", "mode": "report"},
             "concurrency": "replace"},
        ],
    }


def _blocking(errors: list[SurfaceValidationError]) -> list[str]:
    return [e.message for e in errors if e.severity == "blocking"]


def _warnings(errors: list[SurfaceValidationError]) -> list[str]:
    return [e.message for e in errors if e.severity == "warning"]


def _component(surface: dict[str, Any], comp_id: str) -> dict[str, Any]:
    for comp in surface["components"]:
        if comp["id"] == comp_id:
            return comp  # type: ignore[no-any-return]
    raise AssertionError(f"no component {comp_id!r}")


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_valid_structured_surface_has_no_blocking_errors() -> None:
    errors = validate_surface(_definition(_surface()))
    assert not has_blocking(errors), _blocking(errors)


# ---------------------------------------------------------------------------
# Tabs / TabPane
# ---------------------------------------------------------------------------


def test_tabs_without_children_blocks() -> None:
    surface = _surface()
    _component(surface, "tabs")["children"] = []
    # Orphan panes are warnings; the empty Tabs is the blocking finding.
    errors = validate_surface(_definition(surface))
    assert any("at least one TabPane" in m for m in _blocking(errors))


def test_tabs_child_must_be_tabpane() -> None:
    surface = _surface()
    _component(surface, "tabs")["children"] = ["pane_a", "run_btn"]
    surface["components"][0]["children"] = ["tabs"]
    errors = validate_surface(_definition(surface))
    assert any("must be a TabPane" in m for m in _blocking(errors))


def test_tabpane_outside_tabs_blocks() -> None:
    surface = _surface()
    # Move pane_b directly under root.
    _component(surface, "tabs")["children"] = ["pane_a"]
    surface["components"][0]["children"] = ["run_btn", "tabs", "pane_b"]
    errors = validate_surface(_definition(surface))
    assert any("direct child of a Tabs" in m for m in _blocking(errors))


# ---------------------------------------------------------------------------
# Table columns
# ---------------------------------------------------------------------------


def test_table_requires_columns_prop() -> None:
    surface = _surface()
    del _component(surface, "tbl")["props"]["columns"]
    errors = validate_surface(_definition(surface))
    assert any("requires prop 'columns'" in m for m in _blocking(errors))


def test_empty_columns_list_blocks() -> None:
    surface = _surface()
    _component(surface, "tbl")["props"]["columns"] = []
    errors = validate_surface(_definition(surface))
    assert any("'columns' must be object_list" in m for m in _blocking(errors))


def test_unknown_column_item_key_blocks() -> None:
    surface = _surface()
    _component(surface, "tbl")["props"]["columns"] = [
        {"key": "item", "label": "Item", "type": "string", "width": 12},
    ]
    errors = validate_surface(_definition(surface))
    assert any("'columns' must be object_list" in m for m in _blocking(errors))


def test_bad_column_type_enum_blocks() -> None:
    surface = _surface()
    _component(surface, "tbl")["props"]["columns"] = [
        {"key": "item", "label": "Item", "type": "money"},
    ]
    errors = validate_surface(_definition(surface))
    assert any("'columns' must be object_list" in m for m in _blocking(errors))


def test_duplicate_column_key_blocks() -> None:
    surface = _surface()
    _component(surface, "tbl")["props"]["columns"] = [
        {"key": "item", "label": "Item", "type": "string"},
        {"key": "item", "label": "Item again", "type": "string"},
    ]
    errors = validate_surface(_definition(surface))
    assert any("duplicate column key 'item'" in m for m in _blocking(errors))


def test_non_identifier_column_key_blocks() -> None:
    surface = _surface()
    _component(surface, "tbl")["props"]["columns"] = [
        {"key": "9item", "label": "Item", "type": "string"},
    ]
    errors = validate_surface(_definition(surface))
    assert any("must be a valid identifier" in m for m in _blocking(errors))


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------


def test_chart_empty_y_keys_blocks() -> None:
    surface = _surface()
    _component(surface, "cht")["props"]["y_keys"] = []
    errors = validate_surface(_definition(surface))
    assert any("'y_keys' must be string_list" in m for m in _blocking(errors))


def test_chart_non_identifier_key_blocks() -> None:
    surface = _surface()
    _component(surface, "cht")["props"]["x_key"] = "a-b"
    errors = validate_surface(_definition(surface))
    assert any("key 'a-b' must be a valid identifier" in m for m in _blocking(errors))


def test_chart_key_not_in_shared_table_columns_warns() -> None:
    surface = _surface()
    _component(surface, "cht")["props"]["y_keys"] = ["missing_series"]
    errors = validate_surface(_definition(surface))
    assert not has_blocking(errors), _blocking(errors)
    assert any("not columns of the Table" in m for m in _warnings(errors))


# ---------------------------------------------------------------------------
# Slot pointer grammar
# ---------------------------------------------------------------------------


def test_two_segment_slot_blocks() -> None:
    surface = _surface()
    _component(surface, "findings")["props"]["source"] = {
        "path": "/results/run/data/deep/nested"
    }
    errors = validate_surface(_definition(surface))
    assert any("exactly one segment after /data/" in m for m in _blocking(errors))


def test_pointer_under_target_but_not_slot_blocks() -> None:
    surface = _surface()
    _component(surface, "findings")["props"]["source"] = {
        "path": "/results/run/findings"
    }
    errors = validate_surface(_definition(surface))
    assert any("not as a slot" in m for m in _blocking(errors))


def test_output_component_outside_targets_warns() -> None:
    surface = _surface()
    _component(surface, "findings")["props"]["source"] = {"path": "/elsewhere/x"}
    errors = validate_surface(_definition(surface))
    assert not has_blocking(errors), _blocking(errors)
    assert any("the model will never fill it" in m for m in _warnings(errors))


def test_list_over_static_data_is_fine() -> None:
    surface = _surface()
    surface["components"].append(
        {"id": "static", "component": "List",
         "props": {"items": {"path": "/static_items"}}, "children": []}
    )
    _component(surface, "pane_a")["children"].append("static")
    surface["data_model"]["static_items"] = ["a", "b"]
    errors = validate_surface(_definition(surface))
    assert not has_blocking(errors), _blocking(errors)
    assert not any("never fill" in m for m in _warnings(errors))


def test_non_identifier_slot_name_blocks() -> None:
    surface = _surface()
    _component(surface, "findings")["props"]["source"] = {
        "path": "/results/run/data/9findings"
    }
    errors = validate_surface(_definition(surface))
    assert any(
        "slot name '9findings' must be a valid identifier" in m
        for m in _blocking(errors)
    )


# ---------------------------------------------------------------------------
# Slot contract conflicts
# ---------------------------------------------------------------------------


def test_kind_conflict_on_shared_slot_blocks() -> None:
    surface = _surface()
    surface["components"].append(
        {"id": "mg2", "component": "MetricGrid",
         "props": {"source": {"path": "/results/run/data/comparison"}},
         "children": []}
    )
    _component(surface, "pane_b")["children"].append("mg2")
    errors = validate_surface(_definition(surface))
    assert any("give each dataset its own slot" in m for m in _blocking(errors))


def test_conflicting_table_columns_block() -> None:
    surface = _surface()
    tbl2 = copy.deepcopy(_component(surface, "tbl"))
    tbl2["id"] = "tbl2"
    tbl2["props"]["columns"] = [
        {"key": "other", "label": "Other", "type": "string"},
    ]
    surface["components"].append(tbl2)
    _component(surface, "pane_b")["children"].append("tbl2")
    errors = validate_surface(_definition(surface))
    assert any("conflicting Table columns" in m for m in _blocking(errors))


# ---------------------------------------------------------------------------
# Part C — query bound to a non-free-text input (authoring warning)
# ---------------------------------------------------------------------------


def _select_query_surface() -> dict[str, Any]:
    """Query bound to a Select input — the 'pick-or-custom' anti-pattern."""
    return {
        "version": 1,
        "components": [
            {"id": "root", "component": "Column", "props": {},
             "children": ["ticker", "run_btn"]},
            {"id": "ticker", "component": "Select",
             "props": {"label": "Ticker", "value": {"path": "/inputs/ticker"},
                       "options": [{"label": "Apple", "value": "AAPL"}]},
             "children": []},
            {"id": "run_btn", "component": "Button",
             "props": {"label": "Run", "action": "run"}, "children": []},
        ],
        "data_model": {"inputs": {"ticker": "AAPL"}},
        "bindings": [
            {"action": "run", "kind": "run_agent",
             "inputs": {"query": {"path": "/inputs/ticker"}}, "options": {},
             "output": {"target": "/results/run", "mode": "report"},
             "concurrency": "replace"},
        ],
    }


def test_query_bound_to_select_warns_but_does_not_block() -> None:
    errors = validate_surface(_definition(_select_query_surface()))
    assert not has_blocking(errors), _blocking(errors)
    assert any("binds its query to a Select" in m for m in _warnings(errors))


def test_query_bound_to_free_text_produces_no_warning() -> None:
    surface = _select_query_surface()
    # Rebind the same query pointer to a free-text TextField instead of the Select.
    _component(surface, "ticker")["component"] = "TextField"
    errors = validate_surface(_definition(surface))
    assert not any("should come from a free-text field" in m for m in _warnings(errors))
