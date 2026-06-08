"""``$ref`` resolution tests for the MCP schema inliner."""

from __future__ import annotations

from databricks_deep_research.tools.mcp import _inline_refs


def test_inline_top_level_ref() -> None:
    schema = {
        "type": "object",
        "properties": {"x": {"$ref": "#/$defs/X"}},
        "$defs": {"X": {"type": "string"}},
    }
    out = _inline_refs(dict(schema))
    assert out["properties"]["x"]["type"] == "string"
    assert "$defs" not in out


def test_inline_nested_ref() -> None:
    schema = {
        "type": "object",
        "properties": {
            "outer": {
                "type": "object",
                "properties": {"inner": {"$ref": "#/$defs/Inner"}},
            }
        },
        "$defs": {"Inner": {"type": "integer"}},
    }
    out = _inline_refs(dict(schema))
    assert out["properties"]["outer"]["properties"]["inner"]["type"] == "integer"


def test_inline_array_items_ref() -> None:
    schema = {
        "type": "object",
        "properties": {
            "items": {"type": "array", "items": {"$ref": "#/$defs/Item"}}
        },
        "$defs": {"Item": {"type": "string"}},
    }
    out = _inline_refs(dict(schema))
    assert out["properties"]["items"]["items"]["type"] == "string"


def test_inline_definitions_alias() -> None:
    """Some MCP servers use the legacy ``definitions`` key."""
    schema = {
        "type": "object",
        "properties": {"x": {"$ref": "#/definitions/X"}},
        "definitions": {"X": {"type": "string"}},
    }
    out = _inline_refs(dict(schema))
    assert out["properties"]["x"]["type"] == "string"


def test_unresolved_ref_left_alone() -> None:
    schema = {
        "type": "object",
        "properties": {"x": {"$ref": "#/$defs/Missing"}},
        "$defs": {},
    }
    out = _inline_refs(dict(schema))
    assert out["properties"]["x"] == {"$ref": "#/$defs/Missing"}


def test_inline_ref_with_siblings_left_alone() -> None:
    """A ref node with sibling keys is intentionally not collapsed."""
    schema = {
        "type": "object",
        "properties": {"x": {"$ref": "#/$defs/X", "description": "hi"}},
        "$defs": {"X": {"type": "string"}},
    }
    out = _inline_refs(dict(schema))
    # Nodes with extra keys remain as-is.
    assert "$ref" in out["properties"]["x"]
