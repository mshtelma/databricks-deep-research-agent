"""Validation matrix for declarative agent UI surfaces.

Covers the rules ``validate_surface`` enforces: shape (Pydantic), root/ids/
children/cycles, catalog membership + props, pointer syntax, button↔binding
coverage, required-inputs coverage, reserved input keys, output-target
disjointness, and the size caps. Also pins the write gate in
``schemas/agent_v2.py`` and the identifier/pointer grammars.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from deep_research.surface import (
    MAX_SURFACE_BYTES,
    RESERVED_INPUT_KEYS,
    SurfaceValidationError,
    catalog_reference,
    component_names,
    has_blocking,
    is_valid_identifier,
    is_valid_pointer,
    resolve_pointer,
    validate_surface,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixture surface: minimal valid Form + Run + Results
# ---------------------------------------------------------------------------


def _definition(surface: dict[str, Any] | None = None, **overrides: Any) -> dict[str, Any]:
    definition: dict[str, Any] = {
        "id": "wf",
        "name": "Test agent",
        "root": {"id": "r", "type": "sequence", "label": "x", "config": {}},
    }
    definition.update(overrides)
    if surface is not None:
        definition["surface"] = surface
    return definition


def _valid_surface() -> dict[str, Any]:
    return {
        "version": 1,
        "components": [
            {
                "id": "root",
                "component": "Column",
                "props": {},
                "children": ["query_field", "run_button", "report"],
            },
            {
                "id": "query_field",
                "component": "TextField",
                "props": {"label": "Query", "value": {"path": "/form/query"}},
                "children": [],
            },
            {
                "id": "run_button",
                "component": "Button",
                "props": {"label": "Run", "action": "run"},
                "children": [],
            },
            {
                "id": "report",
                "component": "ReportRegion",
                "props": {"source": {"path": "/results/run"}},
                "children": [],
            },
        ],
        "data_model": {"form": {"query": ""}, "results": {"run": None}},
        "bindings": [
            {
                "action": "run",
                "inputs": {"query": {"path": "/form/query"}},
                "output": {"target": "/results/run"},
            }
        ],
    }


def _blocking(errors: list[SurfaceValidationError]) -> list[str]:
    return [e.message for e in errors if e.severity == "blocking"]


def _warnings(errors: list[SurfaceValidationError]) -> list[str]:
    return [e.message for e in errors if e.severity == "warning"]


# ---------------------------------------------------------------------------
# Happy path + absence
# ---------------------------------------------------------------------------


def test_absent_surface_is_valid() -> None:
    assert validate_surface(_definition()) == []


def test_valid_surface_has_no_findings() -> None:
    assert validate_surface(_definition(_valid_surface())) == []


def test_surface_runtime_controls_and_layout_metadata_are_valid() -> None:
    surface = _valid_surface()
    surface["runtime_controls"] = {
        "effort": "show",
        "sources": "show",
        "verify_sources": "advanced",
        "plan_review": "advanced",
        "report_style": "advanced",
        "cross_session_memory": "advanced",
        "live_search": "advanced",
    }
    surface["layout"] = {
        "actions": "host_bar",
        "sections": [
            {
                "id": "inputs",
                "title": "Inputs",
                "role": "inputs",
                "children": ["query_field", "run_button"],
                "default_open": "before_first_run",
            },
            {
                "id": "results",
                "title": "Results",
                "role": "results",
                "children": ["report"],
                "default_open": "after_run",
            },
        ],
    }
    assert validate_surface(_definition(surface)) == []


def test_invalid_runtime_control_policy_is_blocking() -> None:
    surface = _valid_surface()
    surface["runtime_controls"] = {"effort": "sometimes"}
    errors = validate_surface(_definition(surface))
    assert has_blocking(errors)
    assert any("surface.runtime_controls.effort" in (e.path or "") for e in errors)


def test_layout_section_unknown_child_is_blocking() -> None:
    surface = _valid_surface()
    surface["layout"] = {
        "sections": [
            {
                "id": "inputs",
                "title": "Inputs",
                "role": "inputs",
                "children": ["missing_component"],
            }
        ]
    }
    errors = validate_surface(_definition(surface))
    assert any("references unknown child 'missing_component'" in m for m in _blocking(errors))


def test_layout_section_empty_children_warns_not_blocking() -> None:
    # A declared inputs/results section with no children is host-inferred at render
    # time; validation warns (advisory) so the author can list children — never blocks.
    surface = _valid_surface()  # has a TextField (input) and a ReportRegion (result)
    surface["layout"] = {
        "actions": "host_bar",
        "sections": [
            {"id": "inputs", "title": "Inputs", "role": "inputs", "children": []},
            {"id": "results", "title": "Results", "role": "results", "children": []},
        ],
    }
    errors = validate_surface(_definition(surface))
    assert not has_blocking(errors)
    warns = _warnings(errors)
    assert any("layout section 'inputs'" in m and "no children" in m for m in warns)
    assert any("layout section 'results'" in m and "no children" in m for m in warns)


def test_layout_empty_section_no_warning_without_matching_content() -> None:
    # Results-only surface: no INPUT_COMPONENTS present, so an empty inputs section
    # must NOT warn (nothing to place); the empty results section still warns.
    surface = {
        "version": 1,
        "components": [
            {"id": "root", "component": "Column", "props": {}, "children": ["report"]},
            {
                "id": "report",
                "component": "ReportRegion",
                "props": {"source": {"path": "/results/run"}},
                "children": [],
            },
        ],
        "data_model": {"form": {"query": ""}, "results": {"run": None}},
        "bindings": [
            {
                "action": "run",
                "inputs": {"query": {"path": "/form/query"}},
                "output": {"target": "/results/run"},
            }
        ],
        "layout": {
            "actions": "host_bar",
            "sections": [
                {"id": "inputs", "title": "Inputs", "role": "inputs", "children": []},
                {"id": "results", "title": "Results", "role": "results", "children": []},
            ],
        },
    }
    warns = _warnings(validate_surface(_definition(surface)))
    assert not any("layout section 'inputs'" in m for m in warns)
    assert any("layout section 'results'" in m for m in warns)


def test_non_dict_surface_is_blocking() -> None:
    errors = validate_surface(_definition() | {"surface": "nope"})
    assert has_blocking(errors)
    assert "must be an object" in errors[0].message


# ---------------------------------------------------------------------------
# Structure: root / ids / children / cycles / orphans
# ---------------------------------------------------------------------------


def test_missing_root_is_blocking() -> None:
    surface = _valid_surface()
    surface["components"][0]["id"] = "not_root"
    surface["bindings"] = []
    errors = validate_surface(_definition(surface))
    assert any("id 'root'" in m for m in _blocking(errors))


def test_duplicate_component_id_is_blocking() -> None:
    surface = _valid_surface()
    surface["components"].append(copy.deepcopy(surface["components"][1]))
    errors = validate_surface(_definition(surface))
    assert any("duplicate component id" in m for m in _blocking(errors))


def test_unknown_child_reference_is_blocking() -> None:
    surface = _valid_surface()
    surface["components"][0]["children"].append("ghost")
    errors = validate_surface(_definition(surface))
    assert any("unknown child 'ghost'" in m for m in _blocking(errors))


def test_children_on_non_container_is_blocking() -> None:
    surface = _valid_surface()
    surface["components"][1]["children"] = ["run_button"]
    errors = validate_surface(_definition(surface))
    assert any("cannot have children" in m for m in _blocking(errors))


def test_cycle_is_blocking() -> None:
    surface = _valid_surface()
    surface["components"].append(
        {"id": "a", "component": "Column", "props": {}, "children": ["b"]}
    )
    surface["components"].append(
        {"id": "b", "component": "Column", "props": {}, "children": ["a"]}
    )
    surface["components"][0]["children"].append("a")
    errors = validate_surface(_definition(surface))
    assert any("cycle" in m for m in _blocking(errors))


def test_orphan_component_is_warning_only() -> None:
    surface = _valid_surface()
    surface["components"].append(
        {"id": "island", "component": "Text", "props": {"text": "hi"}, "children": []}
    )
    errors = validate_surface(_definition(surface))
    assert not has_blocking(errors)
    assert any("island" in m for m in _warnings(errors))


# ---------------------------------------------------------------------------
# Catalog + props
# ---------------------------------------------------------------------------


def test_unknown_component_is_blocking() -> None:
    surface = _valid_surface()
    surface["components"][1]["component"] = "IFrame"
    errors = validate_surface(_definition(surface))
    assert any("unknown component 'IFrame'" in m for m in _blocking(errors))


def test_unknown_prop_is_blocking() -> None:
    surface = _valid_surface()
    surface["components"][1]["props"]["on_click"] = "alert(1)"
    errors = validate_surface(_definition(surface))
    assert any("unknown prop 'on_click'" in m for m in _blocking(errors))


def test_missing_required_prop_is_blocking() -> None:
    surface = _valid_surface()
    del surface["components"][2]["props"]["action"]
    errors = validate_surface(_definition(surface))
    assert any("requires prop 'action'" in m for m in _blocking(errors))


def test_wrong_prop_type_is_blocking() -> None:
    surface = _valid_surface()
    surface["components"][2]["props"]["label"] = 42
    errors = validate_surface(_definition(surface))
    assert any("prop 'label' must be string" in m for m in _blocking(errors))


def test_enum_prop_out_of_range_is_blocking() -> None:
    surface = _valid_surface()
    surface["components"][0]["props"]["gap"] = "gigantic"
    errors = validate_surface(_definition(surface))
    assert any("must be one of" in m for m in _blocking(errors))


def test_malformed_pathref_prop_is_blocking() -> None:
    surface = _valid_surface()
    surface["components"][1]["props"]["value"] = {"path": "form/query"}
    errors = validate_surface(_definition(surface))
    assert has_blocking(errors)


# ---------------------------------------------------------------------------
# Bindings
# ---------------------------------------------------------------------------


def test_button_without_binding_is_blocking() -> None:
    surface = _valid_surface()
    surface["bindings"] = []
    errors = validate_surface(_definition(surface))
    assert any("no binding defines it" in m for m in _blocking(errors))


def test_binding_without_button_is_warning() -> None:
    surface = _valid_surface()
    surface["bindings"].append(
        {
            "action": "shadow",
            "inputs": {"query": "fixed question"},
            "output": {"target": "/results/shadow"},
        }
    )
    errors = validate_surface(_definition(surface))
    assert not has_blocking(errors)
    assert any("no Button that triggers it" in m for m in _warnings(errors))


def test_duplicate_binding_action_is_blocking() -> None:
    surface = _valid_surface()
    surface["bindings"].append(copy.deepcopy(surface["bindings"][0]))
    errors = validate_surface(_definition(surface))
    assert any("duplicate binding action" in m for m in _blocking(errors))


def test_missing_required_input_coverage_is_blocking() -> None:
    surface = _valid_surface()
    errors = validate_surface(
        _definition(surface, required_inputs=["query", "ticker"])
    )
    assert any(
        "does not provide required workflow input(s): ticker" in m
        for m in _blocking(errors)
    )


@pytest.mark.parametrize("reserved", sorted(RESERVED_INPUT_KEYS)[:5] + ["plan"])
def test_reserved_input_key_is_blocking(reserved: str) -> None:
    surface = _valid_surface()
    surface["bindings"][0]["inputs"][reserved] = "boom"
    errors = validate_surface(_definition(surface))
    assert any("reserved" in m for m in _blocking(errors))


def test_query_is_never_reserved() -> None:
    assert "query" not in RESERVED_INPUT_KEYS


def test_invalid_input_identifier_is_blocking() -> None:
    surface = _valid_surface()
    surface["bindings"][0]["inputs"]["bad-key"] = "x"
    errors = validate_surface(_definition(surface))
    assert any("not a valid identifier" in m for m in _blocking(errors))


def test_malformed_template_pointer_is_blocking() -> None:
    surface = _valid_surface()
    surface["bindings"][0]["inputs"]["query"] = "Scan {/form/} today"
    errors = validate_surface(_definition(surface))
    assert any("malformed pointer placeholder" in m for m in _blocking(errors))


def test_plain_brace_words_in_query_template_are_ignored() -> None:
    surface = _valid_surface()
    surface["bindings"][0]["inputs"]["query"] = "Scan {company} at {/form/query}"
    errors = validate_surface(_definition(surface))
    assert not has_blocking(errors)


def test_expanded_run_option_literals_are_valid() -> None:
    surface = _valid_surface()
    surface["bindings"][0]["options"] = {
        "research_depth": "extended",
        "verify_sources": False,
        "query_mode": "web_search",
        "source_scope": "all",
        "enable_plan_review": True,
        "turn_intent": "research",
        "tone": "objective",
        "output_language": "Spanish",
        "enable_cross_session_memory": False,
        "allow_live_search": True,
    }
    assert validate_surface(_definition(surface)) == []


def test_run_option_pathrefs_are_valid_dynamic_values() -> None:
    surface = _valid_surface()
    surface["data_model"]["options"] = {
        "research_depth": "light",
        "verify_sources": False,
        "query_mode": "deep_research",
        "source_scope": "enterprise_only",
        "turn_intent": "research",
    }
    surface["bindings"][0]["options"] = {
        "research_depth": {"path": "/options/research_depth"},
        "verify_sources": {"path": "/options/verify_sources"},
        "query_mode": {"path": "/options/query_mode"},
        "source_scope": {"path": "/options/source_scope"},
        "turn_intent": {"path": "/options/turn_intent"},
    }
    assert validate_surface(_definition(surface)) == []


@pytest.mark.parametrize(
    ("option_name", "invalid_value"),
    [
        ("research_depth", "maximum"),
        ("query_mode", "agent"),
        ("source_scope", "enterprise"),
        ("turn_intent", "rerun"),
    ],
)
def test_invalid_run_option_enum_literal_is_blocking(
    option_name: str,
    invalid_value: str,
) -> None:
    surface = _valid_surface()
    surface["bindings"][0]["options"] = {option_name: invalid_value}
    errors = validate_surface(_definition(surface))
    assert any(option_name in (e.path or "") for e in errors if e.severity == "blocking")
    assert any("must be one of" in m for m in _blocking(errors))


@pytest.mark.parametrize(
    "option_name",
    [
        "verify_sources",
        "enable_plan_review",
        "enable_cross_session_memory",
        "allow_live_search",
    ],
)
def test_boolean_run_option_literals_must_be_boolean(option_name: str) -> None:
    surface = _valid_surface()
    surface["bindings"][0]["options"] = {option_name: "false"}
    errors = validate_surface(_definition(surface))
    assert any(option_name in (e.path or "") for e in errors if e.severity == "blocking")
    assert any("must be a boolean" in m for m in _blocking(errors))


@pytest.mark.parametrize("option_name", ["tone", "output_language"])
def test_string_run_option_literals_must_be_strings(option_name: str) -> None:
    surface = _valid_surface()
    surface["bindings"][0]["options"] = {option_name: True}
    errors = validate_surface(_definition(surface))
    assert any(option_name in (e.path or "") for e in errors if e.severity == "blocking")
    assert any("must be a string" in m for m in _blocking(errors))


def test_overlapping_output_targets_are_blocking() -> None:
    surface = _valid_surface()
    surface["bindings"].append(
        {
            "action": "second",
            "inputs": {"query": "x"},
            "output": {"target": "/results/run/nested"},
        }
    )
    surface["components"].append(
        {
            "id": "second_button",
            "component": "Button",
            "props": {"label": "B", "action": "second"},
            "children": [],
        }
    )
    surface["components"][0]["children"].append("second_button")
    errors = validate_surface(_definition(surface))
    assert any("overlap" in m for m in _blocking(errors))


def test_uninitialized_input_pointer_is_warning() -> None:
    surface = _valid_surface()
    surface["components"][1]["props"]["value"] = {"path": "/form/missing"}
    surface["bindings"][0]["inputs"]["query"] = {"path": "/form/missing"}
    errors = validate_surface(_definition(surface))
    assert not has_blocking(errors)
    assert any("not initialized in data_model" in m for m in _warnings(errors))


# ---------------------------------------------------------------------------
# Caps
# ---------------------------------------------------------------------------


def test_component_cap_is_blocking() -> None:
    surface = _valid_surface()
    for i in range(101):
        surface["components"].append(
            {
                "id": f"extra_{i}",
                "component": "Text",
                "props": {"text": "x"},
                "children": [],
            }
        )
    errors = validate_surface(_definition(surface))
    assert any("cap is" in m for m in _blocking(errors))


def test_size_cap_is_blocking() -> None:
    surface = _valid_surface()
    surface["data_model"]["blob"] = "x" * (MAX_SURFACE_BYTES + 1)
    errors = validate_surface(_definition(surface))
    assert any("size cap" in m for m in _blocking(errors))


# ---------------------------------------------------------------------------
# Write gate (schemas/agent_v2.py)
# ---------------------------------------------------------------------------


def test_write_gate_raises_on_blocking_surface() -> None:
    from deep_research.schemas.agent_v2 import _enforce_surface_validation

    surface = _valid_surface()
    surface["components"][1]["component"] = "IFrame"
    with pytest.raises(ValueError, match="Surface validation failed"):
        _enforce_surface_validation(_definition(surface))


def test_write_gate_passes_warnings() -> None:
    from deep_research.schemas.agent_v2 import _enforce_surface_validation

    surface = _valid_surface()
    surface["components"].append(
        {"id": "island", "component": "Text", "props": {"text": "hi"}, "children": []}
    )
    _enforce_surface_validation(_definition(surface))  # warning-only → no raise


# ---------------------------------------------------------------------------
# Grammars + catalog reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pointer", "ok"),
    [
        ("/form/query", True),
        ("/a", True),
        ("/a/b_2/c", True),
        ("", False),
        ("form/query", False),
        ("/form/", False),
        ("/form//x", False),
        ("/form/qu ery", False),
        ("/form/x-y", False),
    ],
)
def test_pointer_grammar(pointer: str, ok: bool) -> None:
    assert is_valid_pointer(pointer) is ok


@pytest.mark.parametrize(
    ("name", "ok"),
    [("ticker", True), ("_x", True), ("Ticker9", True), ("9x", False), ("a-b", False)],
)
def test_identifier_grammar(name: str, ok: bool) -> None:
    assert is_valid_identifier(name) is ok


def test_resolve_pointer() -> None:
    data = {"form": {"query": "", "n": None}}
    assert resolve_pointer(data, "/form/query") == (True, "")
    assert resolve_pointer(data, "/form/n") == (True, None)
    assert resolve_pointer(data, "/form/missing") == (False, None)
    assert resolve_pointer(data, "bad") == (False, None)


def test_catalog_reference_is_serializable_and_complete() -> None:
    import json

    reference = catalog_reference()
    assert set(reference) == set(component_names())
    json.dumps(reference)  # must be JSON-serializable for the TS parity check
    # The v1 trust boundary: no URL/HTML/event props anywhere in the catalog.
    for entry in reference.values():
        for prop_name in entry["props"]:
            assert "url" not in prop_name.lower()
            assert "html" not in prop_name.lower()
            assert not prop_name.lower().startswith("on_")
