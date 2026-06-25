"""Tests for grouping + deref decoration of the registry's node-type schemas."""

from __future__ import annotations

from deep_research.agent_designer.field_groups import HIDDEN_FIELDS
from deep_research.agent_designer.registry import node_types_payload
from deep_research.agent_designer.schema_deref import assert_no_refs


def _agent_schema() -> dict:
    payload = {n["type"]: n for n in node_types_payload()}
    return payload["agent"]["config_schema"]


def test_all_node_schemas_are_ref_free() -> None:
    for node in node_types_payload():
        assert_no_refs(node["config_schema"])


def test_agent_schema_props_are_grouped() -> None:
    props = _agent_schema()["properties"]
    assert props, "agent schema must expose properties"
    for name, prop in props.items():
        assert prop.get("x-group"), f"{name} has no x-group"
        assert "x-order" in prop, f"{name} has no x-order"


def test_hidden_fields_dropped() -> None:
    props = _agent_schema()["properties"]
    for hidden in HIDDEN_FIELDS:
        assert hidden not in props, f"{hidden} should be hidden from the config form"


def test_nested_model_renders_as_object() -> None:
    budget = _agent_schema()["properties"]["tool_output_budget"]
    assert budget.get("type") == "object"
    assert isinstance(budget.get("properties"), dict) and budget["properties"]


def test_dict_field_tagged_json_widget() -> None:
    props = _agent_schema()["properties"]
    assert props["spawnable_subagents"].get("x-widget") == "json"
    assert props["per_tool_limits"].get("x-widget") == "json"


def test_existing_hints_preserved() -> None:
    props = _agent_schema()["properties"]
    assert props["system_prompt"].get("x-widget") == "prompt"
    assert isinstance(props["subtype"].get("enum"), list) and props["subtype"]["enum"]
    assert isinstance(props["model_tier"].get("enum"), list) and props["model_tier"]["enum"]


def test_advanced_groups_flagged() -> None:
    props = _agent_schema()["properties"]
    # A Context & Memory field is advanced; a Basics field is not.
    assert props["tool_output_offload"].get("x-advanced") is True
    assert props["subtype"].get("x-advanced") is False


def test_properties_ordered_basics_first() -> None:
    props = list(_agent_schema()["properties"])
    assert props[0] == "subtype", f"expected subtype first, got {props[:3]}"
