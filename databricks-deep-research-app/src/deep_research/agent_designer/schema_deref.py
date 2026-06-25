"""Inline Pydantic ``$ref``/``$defs`` into a self-contained JSON schema.

The Designer's ``SchemaField`` renderer (frontend) does not resolve JSON-schema
references: a property like ``{"$ref": "#/$defs/ToolOutputBudgetConfig"}`` has no
``type``/``properties``, so it falls through to a plain text input (a broken box
for a nested object) — and frontend ajv throws on a dangling ``$ref`` at compile.

:func:`deref_schema` produces an equivalent schema with **zero** ``$ref``/``$defs``
by inlining the targets, so the existing object/array/enum widgets render nested
Pydantic models, ``Optional[...]`` fields, and enums-in-``$defs`` correctly.

Handled shapes (everything Pydantic v2 emits for our config models):
- bare ``{"$ref": "#/$defs/X"}``
- ``{"allOf": [{"$ref": ...}], "default": …, "title": …}`` (annotated nested model)
- ``{"anyOf": [{"$ref": ...}, {"type": "null"}], …}`` (``Optional[Model]`` / ``Optional[Enum]``)
- enums living in ``$defs`` (inlined so the select widget fires)

It is cycle-safe (a ``$defs`` model referencing itself collapses to a bare
``{"type": "object"}`` once revisited) and depth-bounded.
"""

from __future__ import annotations

import copy
from typing import Any

# Defensive bound; real config schemas nest only a few levels.
_MAX_DEPTH = 60

_COMBINERS = ("allOf", "anyOf", "oneOf")


def deref_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of *schema* with all ``$ref``/``$defs`` inlined.

    The returned schema contains no ``$ref`` or ``$defs`` keys (asserted by the
    accompanying tests); pass a Pydantic ``model_json_schema()`` result.
    """
    defs = schema.get("$defs") or {}
    resolved = _resolve(schema, defs, (), 0)
    if isinstance(resolved, dict):
        resolved.pop("$defs", None)
        return resolved
    return {}


def _resolve(node: Any, defs: dict[str, Any], stack: tuple[str, ...], depth: int) -> Any:
    if depth > _MAX_DEPTH:
        return {"type": "object"}
    if isinstance(node, list):
        return [_resolve(item, defs, stack, depth + 1) for item in node]
    if not isinstance(node, dict):
        return node

    ref = node.get("$ref")
    if isinstance(ref, str):
        return _resolve_ref(ref, node, defs, stack, depth)

    for combiner in _COMBINERS:
        branches = node.get(combiner)
        if isinstance(branches, list):
            return _resolve_combiner(branches, node, defs, stack, depth)

    out: dict[str, Any] = {}
    for key, value in node.items():
        if key == "$defs":
            continue
        out[key] = _resolve(value, defs, stack, depth + 1)
    return out


def _resolve_ref(
    ref: str,
    node: dict[str, Any],
    defs: dict[str, Any],
    stack: tuple[str, ...],
    depth: int,
) -> dict[str, Any]:
    name = ref.rsplit("/", 1)[-1]
    if name in stack:  # cycle — break with a permissive object
        base: dict[str, Any] = {"type": "object", "title": name}
    else:
        target = defs.get(name)
        base = (
            _resolve(copy.deepcopy(target), defs, stack + (name,), depth + 1)
            if isinstance(target, dict)
            else {}
        )
    if not isinstance(base, dict):
        return {}
    # Field-level siblings (title/default/description) fill in around the target,
    # never overriding the resolved structure (type/enum/properties).
    merged = dict(base)
    for key, value in node.items():
        if key == "$ref":
            continue
        merged.setdefault(key, _resolve(value, defs, stack, depth + 1))
    return merged


def _resolve_combiner(
    branches: list[Any],
    node: dict[str, Any],
    defs: dict[str, Any],
    stack: tuple[str, ...],
    depth: int,
) -> dict[str, Any]:
    resolved = [_resolve(branch, defs, stack, depth + 1) for branch in branches]
    non_null = [
        b for b in resolved if not (isinstance(b, dict) and b.get("type") == "null")
    ]
    # Optional[X] => one non-null branch. True unions => best-effort first branch
    # (SchemaField has no union widget); both are far better than a dangling ref.
    chosen = non_null[0] if non_null else (resolved[0] if resolved else {})
    if not isinstance(chosen, dict):
        chosen = {}
    merged = dict(chosen)
    for key, value in node.items():
        if key in _COMBINERS:
            continue
        merged.setdefault(key, _resolve(value, defs, stack, depth + 1))
    if len(non_null) < len(resolved):
        merged.setdefault("x-nullable", True)
    return merged


def assert_no_refs(schema: Any) -> None:
    """Raise ``AssertionError`` if any ``$ref``/``$defs`` remains (test helper)."""
    if isinstance(schema, dict):
        assert "$ref" not in schema, f"dangling $ref: {schema.get('$ref')}"
        assert "$defs" not in schema, "dangling $defs"
        for value in schema.values():
            assert_no_refs(value)
    elif isinstance(schema, list):
        for item in schema:
            assert_no_refs(item)
