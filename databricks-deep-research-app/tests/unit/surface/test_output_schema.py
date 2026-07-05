"""Golden tests for the structured-output slot compiler.

Pins the slot grammar (`<target>/data/<slot>`), the per-binding grouping,
contract merging (Table+Chart sharing a slot, conflicts), and the compiled
Pydantic schema (closed, capped, typed) the structuring pass sends to the LLM.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from pydantic import ValidationError

from deep_research.surface.output_schema import (
    LIST_CAP,
    METRICS_CAP,
    ROWS_CAP,
    SOURCE_REFS_CAP,
    ColumnSpec,
    SlotSpec,
    build_output_model,
    build_slot_wire_model,
    collect_output_slots,
    resolve_binding_for_run,
    slot_docs,
    split_slot_pointer,
    wire_slot_docs,
)

pytestmark = pytest.mark.unit


def _surface() -> dict[str, Any]:
    return {
        "version": 1,
        "components": [
            {"id": "root", "component": "Column", "props": {}, "children": [
                "metrics", "findings", "tbl", "cht", "run_btn",
            ]},
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
                     {"key": "as_of", "label": "As of", "type": "date"},
                 ],
             },
             "children": []},
            {"id": "cht", "component": "Chart",
             "props": {
                 "source": {"path": "/results/run/data/comparison"},
                 "kind": "bar", "x_key": "item", "y_keys": ["score"],
             },
             "children": []},
            {"id": "run_btn", "component": "Button",
             "props": {"label": "Run", "action": "run"}, "children": []},
        ],
        "data_model": {"query": ""},
        "bindings": [
            {"action": "run", "kind": "run_agent",
             "inputs": {"query": {"path": "/query"}}, "options": {},
             "output": {"target": "/results/run", "mode": "report"},
             "concurrency": "replace"},
        ],
    }


# ---------------------------------------------------------------------------
# split_slot_pointer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pointer", "expected"),
    [
        ("/results/run/data/comparison", ("run", "comparison")),
        ("/results/run/data/a/b", None),  # two segments under /data/
        ("/results/run/other", None),  # under target, not a slot
        ("/elsewhere/data/x", None),  # not under any target
        ("/results/run/data/", None),
    ],
)
def test_split_slot_pointer(pointer: str, expected: tuple[str, str] | None) -> None:
    assert split_slot_pointer(pointer, {"run": "/results/run"}) == expected


# ---------------------------------------------------------------------------
# collect_output_slots
# ---------------------------------------------------------------------------


def test_collect_groups_slots_per_binding() -> None:
    collected = collect_output_slots(_surface())
    assert collected.issues == []
    slots = collected.slots_for("run")
    assert set(slots) == {"headline_metrics", "key_findings", "comparison"}
    assert slots["headline_metrics"].kind == "metrics"
    assert slots["key_findings"].kind == "strings"
    comparison = slots["comparison"]
    assert comparison.kind == "table"
    assert [c.key for c in comparison.columns] == ["item", "score", "as_of"]
    # Chart shares the Table's slot and contributes its plot keys.
    assert comparison.chart_keys == ("item", "score")
    assert set(comparison.component_ids) == {"tbl", "cht"}


def test_chart_only_slot_synthesizes_columns() -> None:
    surface = _surface()
    surface["components"] = [c for c in surface["components"] if c["id"] != "tbl"]
    slots = collect_output_slots(surface).slots_for("run")
    comparison = slots["comparison"]
    assert comparison.kind == "table"
    assert [(c.key, c.type) for c in comparison.columns] == [
        ("item", "string"),
        ("score", "number"),
    ]


def test_kind_conflict_recorded_as_issue() -> None:
    surface = _surface()
    surface["components"].append(
        {"id": "mg2", "component": "MetricGrid",
         "props": {"source": {"path": "/results/run/data/comparison"}},
         "children": []}
    )
    surface["components"][0]["children"].append("mg2")
    collected = collect_output_slots(surface)
    assert any("comparison" in i.message for i in collected.issues)


def test_conflicting_table_columns_recorded_as_issue() -> None:
    surface = _surface()
    surface["components"].append(
        {"id": "tbl2", "component": "Table",
         "props": {
             "source": {"path": "/results/run/data/comparison"},
             "columns": [{"key": "other", "label": "Other", "type": "string"}],
         },
         "children": []}
    )
    surface["components"][0]["children"].append("tbl2")
    collected = collect_output_slots(surface)
    assert any("conflicting Table columns" in i.message for i in collected.issues)


def test_non_slot_pointers_are_skipped() -> None:
    surface = _surface()
    surface["components"].append(
        {"id": "static_list", "component": "List",
         "props": {"items": {"path": "/static/items"}}, "children": []}
    )
    surface["components"][0]["children"].append("static_list")
    collected = collect_output_slots(surface)
    assert "static_items" not in collected.slots_for("run")
    assert set(collected.slots_for("run")) == {
        "headline_metrics", "key_findings", "comparison",
    }


def test_invalid_surface_dict_is_fail_soft() -> None:
    assert collect_output_slots({"nonsense": True}).by_action == {}


# ---------------------------------------------------------------------------
# build_output_model
# ---------------------------------------------------------------------------


def test_compiled_schema_is_closed_and_capped() -> None:
    slots = collect_output_slots(_surface()).slots_for("run")
    model = build_output_model(slots)
    schema = model.model_json_schema()

    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(slots)

    comparison = schema["properties"]["comparison"]
    assert comparison["maxItems"] == ROWS_CAP
    row_schema = schema["$defs"]["Row_comparison"]
    assert row_schema["additionalProperties"] is False
    assert row_schema["properties"]["score"]["type"] == "number"
    assert "citation markers" in row_schema["properties"]["item"]["description"]
    assert "ISO-8601" in row_schema["properties"]["as_of"]["description"]

    assert schema["properties"]["key_findings"]["maxItems"] == LIST_CAP
    metrics = schema["properties"]["headline_metrics"]
    assert metrics["maxItems"] == METRICS_CAP
    metric_schema = schema["$defs"]["Metric_headline_metrics"]
    assert set(metric_schema["required"]) == {"label", "value"}


def test_compiled_model_validates_and_rejects() -> None:
    slots = collect_output_slots(_surface()).slots_for("run")
    model = build_output_model(slots)
    payload = {
        "headline_metrics": [{"label": "Coverage", "value": "87", "unit": "%"}],
        "key_findings": ["Finding one [K1]."],
        "comparison": [{"item": "A [K1]", "score": 8.1, "as_of": "2026-01-01"}],
    }
    parsed = model.model_validate(payload)
    assert parsed.model_dump()["comparison"][0]["score"] == 8.1

    with pytest.raises(ValidationError):
        model.model_validate({**payload, "extra_slot": []})
    with pytest.raises(ValidationError):
        bad = copy.deepcopy(payload)
        bad["comparison"][0]["score"] = "not a number"
        model.model_validate(bad)
    with pytest.raises(ValidationError):
        over = copy.deepcopy(payload)
        over["key_findings"] = ["x"] * (LIST_CAP + 1)
        model.model_validate(over)


def test_slot_docs_mention_every_slot() -> None:
    slots = collect_output_slots(_surface()).slots_for("run")
    docs = slot_docs(slots)
    for slot in slots:
        assert slot in docs
    assert "item (string): Item" in docs


def test_schema_is_json_serializable() -> None:
    slots = collect_output_slots(_surface()).slots_for("run")
    json.dumps(build_output_model(slots).model_json_schema())


# ---------------------------------------------------------------------------
# resolve_binding_for_run
# ---------------------------------------------------------------------------


def test_resolve_binding_explicit_sole_and_ambiguous() -> None:
    collected = collect_output_slots(_surface())

    explicit = resolve_binding_for_run(collected, "run")
    assert explicit is not None and explicit.action == "run"
    assert explicit.ambiguous is False

    sole = resolve_binding_for_run(collected, None)
    assert sole is not None and sole.action == "run"

    assert resolve_binding_for_run(collected, "missing") is None

    # Two slotted bindings without an explicit action → first + ambiguous.
    surface = _surface()
    surface["bindings"].append(
        {"action": "second", "kind": "run_agent",
         "inputs": {"query": {"path": "/query"}}, "options": {},
         "output": {"target": "/results/second", "mode": "report"},
         "concurrency": "replace"}
    )
    surface["components"].append(
        {"id": "lst2", "component": "KeyFindings",
         "props": {"source": {"path": "/results/second/data/notes"}},
         "children": []}
    )
    surface["components"][0]["children"].append("lst2")
    two = collect_output_slots(surface)
    picked = resolve_binding_for_run(two, None)
    assert picked is not None and picked.ambiguous is True

    empty = collect_output_slots(
        {**_surface(), "components": [
            {"id": "root", "component": "Column", "props": {}, "children": []}
        ]}
    )
    assert resolve_binding_for_run(empty, None) is None


# ---------------------------------------------------------------------------
# build_slot_wire_model (v2 per-slot wires)
# ---------------------------------------------------------------------------


def _run_slots() -> dict[str, Any]:
    return dict(collect_output_slots(_surface()).slots_for("run"))


def test_wire_model_single_top_level_slot_field() -> None:
    slots = _run_slots()
    model = build_slot_wire_model("comparison", slots["comparison"])
    assert set(model.model_fields.keys()) == {"comparison"}


def test_wire_table_schema_closed_capped_serializable() -> None:
    slots = _run_slots()
    model = build_slot_wire_model("comparison", slots["comparison"])
    schema = model.model_json_schema()
    json.dumps(schema)  # must be serializable

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for value in node:
                _walk(value)

    _walk(schema)
    assert schema["properties"]["comparison"]["maxItems"] == ROWS_CAP

    row_schema = next(iter(schema.get("$defs", {}).values()))
    assert set(row_schema["properties"]) == {"item", "score", "as_of", "source_refs"}
    assert "source_refs" not in row_schema.get("required", [])
    assert row_schema["properties"]["source_refs"]["maxItems"] == SOURCE_REFS_CAP


def test_wire_model_tolerant_coercions_active() -> None:
    slots = _run_slots()
    model = build_slot_wire_model("comparison", slots["comparison"])
    obj = model.model_validate(
        {"comparison": [{
            "item": {"name": "A", "kind": "x"},
            "score": 1.5,
            "as_of": "2026-01-01",
            "source_refs": [1, 3],
        }]}
    )
    row = obj.comparison[0]  # type: ignore[attr-defined]
    assert row.item == "name: A; kind: x"
    assert row.source_refs == ["1", "3"]


def test_wire_model_unwraps_transport_envelope_before_coercion() -> None:
    slots = _run_slots()
    model = build_slot_wire_model("comparison", slots["comparison"])
    wrapped = {"$PARAMETER_VALUE": {"comparison": [{
        "item": "A", "score": 2.0, "as_of": "2026-01-01", "source_refs": [7],
    }]}}
    obj = model.model_validate(wrapped)
    assert obj.comparison[0].source_refs == ["7"]  # type: ignore[attr-defined]


def test_wire_metrics_model_shape() -> None:
    slots = _run_slots()
    model = build_slot_wire_model("headline_metrics", slots["headline_metrics"])
    obj = model.model_validate(
        {"headline_metrics": [{"label": "TAM", "value": "18.4", "source_refs": ["2"]}]}
    )
    metric = obj.headline_metrics[0]  # type: ignore[attr-defined]
    assert metric.unit is None
    assert metric.source_refs == ["2"]


def test_wire_strings_model_items_are_objects() -> None:
    slots = _run_slots()
    model = build_slot_wire_model("key_findings", slots["key_findings"])
    obj = model.model_validate(
        {"key_findings": [{"text": "finding", "source_refs": ["1"]}]}
    )
    assert obj.key_findings[0].text == "finding"  # type: ignore[attr-defined]
    with pytest.raises(ValidationError):
        model.model_validate({"key_findings": [{"source_refs": ["1"]}]})


def test_wire_slot_docs_per_kind() -> None:
    slots = _run_slots()
    table_doc = wire_slot_docs("comparison", slots["comparison"])
    assert "MUST include source_refs" in table_doc
    assert "item (string)" in table_doc
    metrics_doc = wire_slot_docs("headline_metrics", slots["headline_metrics"])
    assert "whenever" in metrics_doc
    strings_doc = wire_slot_docs("key_findings", slots["key_findings"])
    assert "'text'" in strings_doc


def test_build_slot_wire_model_invalid_name_raises() -> None:
    slots = _run_slots()
    with pytest.raises(ValueError, match="invalid slot name"):
        build_slot_wire_model("bad-name", slots["comparison"])


def test_wire_row_column_named_source_refs_wins() -> None:
    spec = SlotSpec(
        slot="s",
        kind="table",
        columns=(ColumnSpec(key="source_refs", label="Refs", type="string"),),
    )
    model = build_slot_wire_model("s", spec)
    obj = model.model_validate({"s": [{"source_refs": "raw text"}]})
    assert obj.s[0].source_refs == "raw text"  # type: ignore[attr-defined]
