"""Tests for the JSON-schema dereferencer used by the Designer inspector."""

from __future__ import annotations

from databricks_deep_research.agents.config import AgentNodeConfig

from deep_research.agent_designer.schema_deref import assert_no_refs, deref_schema


def test_real_agent_schema_has_no_refs_after_deref() -> None:
    """The full AgentNodeConfig schema must be $ref/$defs-free after deref."""
    schema = deref_schema(AgentNodeConfig.model_json_schema())
    assert_no_refs(schema)  # raises if any $ref/$defs remain
    assert "$defs" not in schema


def test_nested_model_field_becomes_inline_object() -> None:
    """tool_output_budget ($ref) renders as an object with its properties inline."""
    schema = deref_schema(AgentNodeConfig.model_json_schema())
    budget = schema["properties"]["tool_output_budget"]
    assert budget.get("type") == "object"
    assert isinstance(budget.get("properties"), dict) and budget["properties"]


def test_optional_enum_field_inlines_enum() -> None:
    """tone (Optional[Tone]) collapses to the enum so the select widget fires."""
    schema = deref_schema(AgentNodeConfig.model_json_schema())
    tone = schema["properties"]["tone"]
    assert isinstance(tone.get("enum"), list) and tone["enum"], tone
    assert tone.get("x-nullable") is True  # Optional => nullable marker


def test_bare_ref_inlined() -> None:
    src = {
        "$defs": {"Inner": {"type": "object", "properties": {"a": {"type": "integer"}}}},
        "type": "object",
        "properties": {"nested": {"$ref": "#/$defs/Inner"}},
    }
    out = deref_schema(src)
    assert out["properties"]["nested"]["type"] == "object"
    assert out["properties"]["nested"]["properties"]["a"]["type"] == "integer"
    assert_no_refs(out)


def test_allof_single_ref_with_siblings_merges() -> None:
    src = {
        "$defs": {"Inner": {"type": "object", "properties": {"a": {"type": "string"}}}},
        "type": "object",
        "properties": {
            "nested": {"allOf": [{"$ref": "#/$defs/Inner"}], "title": "My Nested", "default": {}},
        },
    }
    out = deref_schema(src)
    nested = out["properties"]["nested"]
    assert nested["type"] == "object"
    assert nested["title"] == "My Nested"  # sibling preserved
    assert nested["default"] == {}
    assert_no_refs(out)


def test_anyof_ref_plus_null_picks_non_null() -> None:
    src = {
        "$defs": {"Inner": {"enum": ["x", "y"]}},
        "type": "object",
        "properties": {
            "opt": {"anyOf": [{"$ref": "#/$defs/Inner"}, {"type": "null"}], "default": None},
        },
    }
    out = deref_schema(src)
    opt = out["properties"]["opt"]
    assert opt["enum"] == ["x", "y"]
    assert opt["x-nullable"] is True
    assert_no_refs(out)


def test_cycle_is_broken_safely() -> None:
    """A self-referential $defs model must not recurse forever."""
    src = {
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {
                    "child": {"anyOf": [{"$ref": "#/$defs/Node"}, {"type": "null"}]},
                    "name": {"type": "string"},
                },
            }
        },
        "type": "object",
        "properties": {"root": {"$ref": "#/$defs/Node"}},
    }
    out = deref_schema(src)
    assert_no_refs(out)  # terminates + no dangling ref
    # The first level resolves; the self-reference collapses to a bare object.
    assert out["properties"]["root"]["type"] == "object"


def test_scalars_and_lists_pass_through() -> None:
    src = {"type": "array", "items": {"type": "string"}, "enum": ["a", "b"]}
    out = deref_schema(src)
    assert out == {"type": "array", "items": {"type": "string"}, "enum": ["a", "b"]}
