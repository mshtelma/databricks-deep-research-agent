"""The fixed component catalog — the surface's trust boundary.

Agents (and the Designer's LLM) can only use components declared here; the
renderer only renders components declared here. No URL/image/HTML props and no
event-handler props exist in v1, so a hostile surface can at worst render odd
text. The TS renderer catalog mirrors this table 1:1; the enum-parity unit
test pins the two via :func:`catalog_reference`.

Props are described with a small typed DSL (:class:`PropSpec`) instead of raw
JSON Schema so validation stays dependency-free and mypy-strict-friendly;
:func:`component_props_json_schema` derives a JSON Schema view for UI/docs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Value kinds a prop may accept (before considering `dynamic`).
# "pathref"      → {"path": "/json/pointer"} data-model reference (required form)
# "options"      → static Select option list [{"label", "value"}, ...]
# "string_list"  → non-empty list[str]
# "object_list"  → non-empty list of objects validated against `item_shape`
PropKind = str  # "string" | "number" | "boolean" | "options" | "pathref" | "string_list" | "object_list"


@dataclass(frozen=True)
class PropSpec:
    """Shape of one component prop.

    ``dynamic=True`` additionally allows a ``{"path": "/json/pointer"}``
    reference resolved against the surface data model at render time.
    ``kind == "options"`` is the static option list of a ``Select``:
    ``[{"label": str, "value": str}, ...]``. ``kind == "object_list"``
    validates each item against ``item_shape`` (a nested prop map; unknown
    item keys are rejected).
    """

    kind: PropKind
    required: bool = False
    dynamic: bool = False
    enum: tuple[Any, ...] | None = None
    item_shape: dict[str, PropSpec] | None = None
    doc: str = ""


@dataclass(frozen=True)
class ComponentSpec:
    """One catalog entry: allowed props + whether it may hold children."""

    name: str
    container: bool = False
    props: dict[str, PropSpec] = field(default_factory=dict)
    doc: str = ""


def _pathref_only(doc: str, *, required: bool = True) -> PropSpec:
    """A prop that MUST be a data-model reference (two-way bound inputs)."""
    return PropSpec(kind="pathref", required=required, dynamic=True, doc=doc)


_COMPONENTS: tuple[ComponentSpec, ...] = (
    # --- layout -----------------------------------------------------------
    ComponentSpec(
        "Column",
        container=True,
        props={"gap": PropSpec(kind="string", enum=("sm", "md", "lg"))},
        doc="Vertical stack of children.",
    ),
    ComponentSpec(
        "Row",
        container=True,
        props={"gap": PropSpec(kind="string", enum=("sm", "md", "lg"))},
        doc="Horizontal row of children.",
    ),
    ComponentSpec(
        "Card",
        container=True,
        props={"title": PropSpec(kind="string")},
        doc="Bordered grouping card with an optional title.",
    ),
    ComponentSpec("Divider", doc="Horizontal separator."),
    ComponentSpec(
        "Tabs",
        container=True,
        doc="Tabbed container; every child must be a TabPane.",
    ),
    ComponentSpec(
        "TabPane",
        container=True,
        props={"label": PropSpec(kind="string", required=True)},
        doc="One labeled pane inside a Tabs container.",
    ),
    # --- static content ---------------------------------------------------
    ComponentSpec(
        "Heading",
        props={
            "text": PropSpec(kind="string", required=True, dynamic=True),
            "level": PropSpec(kind="number", enum=(1, 2, 3)),
        },
        doc="Section heading.",
    ),
    ComponentSpec(
        "Text",
        props={"text": PropSpec(kind="string", required=True, dynamic=True)},
        doc="Plain text line (rendered as text, never HTML).",
    ),
    ComponentSpec(
        "Markdown",
        props={"content": PropSpec(kind="string", required=True, dynamic=True)},
        doc="Markdown block rendered through the app's sanitized renderer.",
    ),
    # --- inputs (two-way bound to the data model) --------------------------
    ComponentSpec(
        "TextField",
        props={
            "label": PropSpec(kind="string"),
            "placeholder": PropSpec(kind="string"),
            "value": _pathref_only("Data-model path the field reads/writes."),
        },
        doc="Single-line text input.",
    ),
    ComponentSpec(
        "TextArea",
        props={
            "label": PropSpec(kind="string"),
            "placeholder": PropSpec(kind="string"),
            "rows": PropSpec(kind="number"),
            "value": _pathref_only("Data-model path the field reads/writes."),
        },
        doc="Multi-line text input.",
    ),
    ComponentSpec(
        "Select",
        props={
            "label": PropSpec(kind="string"),
            "options": PropSpec(kind="options", required=True),
            "value": _pathref_only("Data-model path holding the selected value."),
        },
        doc="Dropdown over a static option list.",
    ),
    ComponentSpec(
        "Checkbox",
        props={
            "label": PropSpec(kind="string"),
            "value": _pathref_only("Data-model path holding the boolean."),
        },
        doc="Boolean toggle.",
    ),
    # --- actions ------------------------------------------------------------
    ComponentSpec(
        "Button",
        props={
            "label": PropSpec(kind="string", required=True),
            "action": PropSpec(
                kind="string",
                required=True,
                doc="Name of the ActionBinding this button triggers.",
            ),
            "variant": PropSpec(kind="string", enum=("primary", "secondary")),
        },
        doc="Triggers the named action binding.",
    ),
    # --- results ------------------------------------------------------------
    ComponentSpec(
        "ReportRegion",
        props={
            "source": _pathref_only(
                "Data-model path holding the run reference "
                "({status, session_id, message_id})."
            ),
            "empty_text": PropSpec(kind="string"),
        },
        doc="Renders a run's report (with citations) resolved by reference.",
    ),
    ComponentSpec(
        "StatusBadge",
        props={
            "source": _pathref_only("Data-model path holding the run reference."),
            "label": PropSpec(kind="string"),
        },
        doc="Compact status indicator for a run reference.",
    ),
    # --- structured output (slots the model fills after each run) -----------
    # Each of these binds a SLOT: a pointer exactly one segment under a
    # binding's output target, `<target>/data/<slot>`. The structuring pass
    # compiles the slots into a JSON schema, the model fills them from the
    # verified report, and the renderer reads the payload at the pointer.
    ComponentSpec(
        "Table",
        props={
            "source": _pathref_only(
                "Slot path `<binding output target>/data/<slot>`; filled with "
                "an array of row objects matching `columns`."
            ),
            "columns": PropSpec(
                kind="object_list",
                required=True,
                item_shape={
                    "key": PropSpec(kind="string", required=True),
                    "label": PropSpec(kind="string", required=True),
                    "type": PropSpec(
                        kind="string",
                        required=True,
                        enum=("string", "number", "date"),
                    ),
                },
                doc="Typed columns; `key` must be a stable identifier.",
            ),
            "empty_text": PropSpec(kind="string"),
        },
        doc="Structured table the model fills with typed rows.",
    ),
    ComponentSpec(
        "MetricGrid",
        props={
            "source": _pathref_only(
                "Slot path filled with metric cards "
                "[{label, value, unit?, delta?}, ...]."
            ),
            "empty_text": PropSpec(kind="string"),
        },
        doc="Grid of headline metric cards the model fills.",
    ),
    ComponentSpec(
        "KeyFindings",
        props={
            "source": _pathref_only(
                "Slot path filled with finding strings (may carry [Key] "
                "citation markers)."
            ),
            "max_items": PropSpec(kind="number"),
            "empty_text": PropSpec(kind="string"),
        },
        doc="Bulleted key findings the model fills.",
    ),
    ComponentSpec(
        "Chart",
        props={
            "source": _pathref_only(
                "Slot path with table-shaped rows; may SHARE a Table's slot."
            ),
            "kind": PropSpec(kind="string", required=True, enum=("bar", "line")),
            "x_key": PropSpec(
                kind="string", required=True, doc="Row key for the x axis."
            ),
            "y_keys": PropSpec(
                kind="string_list",
                required=True,
                doc="Row keys plotted as numeric series.",
            ),
            "height": PropSpec(kind="number", doc="Chart height in px."),
            "empty_text": PropSpec(kind="string"),
        },
        doc="Bar/line chart over table-shaped structured rows.",
    ),
    ComponentSpec(
        "List",
        props={
            "items": _pathref_only(
                "Data-model path holding an array of strings (a filled slot "
                "or a static list)."
            ),
            "ordered": PropSpec(kind="boolean"),
            "empty_text": PropSpec(kind="string"),
        },
        doc="Plain list over an array of strings.",
    ),
)

CATALOG: dict[str, ComponentSpec] = {spec.name: spec for spec in _COMPONENTS}

CONTAINER_COMPONENTS: frozenset[str] = frozenset(
    spec.name for spec in _COMPONENTS if spec.container
)

# Inputs whose `value` prop two-way binds to the data model.
INPUT_COMPONENTS: frozenset[str] = frozenset(
    {"TextField", "TextArea", "Select", "Checkbox"}
)

# Components whose pointer prop names a structured-output SLOT the model
# fills after a run (List may also read a static array — it participates in
# slot collection only when its pointer sits under a binding output target).
OUTPUT_COMPONENTS: frozenset[str] = frozenset(
    {"Table", "MetricGrid", "KeyFindings", "Chart", "List"}
)

# Pointer-bearing prop per output component (slot collection reads these).
OUTPUT_POINTER_PROPS: dict[str, str] = {
    "Table": "source",
    "MetricGrid": "source",
    "KeyFindings": "source",
    "Chart": "source",
    "List": "items",
}


def component_names() -> list[str]:
    """Stable, ordered catalog component names."""
    return [spec.name for spec in _COMPONENTS]


def component_spec(name: str) -> ComponentSpec | None:
    """Catalog entry for *name*, or None when the component is unknown."""
    return CATALOG.get(name)


def _prop_json_schema(spec: PropSpec) -> dict[str, Any]:
    base: dict[str, Any]
    if spec.kind == "pathref":
        base = {"$ref": "#/$defs/pathRef"}
    elif spec.kind == "options":
        base = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["label", "value"],
                "additionalProperties": False,
            },
        }
    elif spec.kind == "string_list":
        base = {"type": "array", "minItems": 1, "items": {"type": "string"}}
    elif spec.kind == "object_list":
        shape = spec.item_shape or {}
        base = {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {k: _prop_json_schema(p) for k, p in shape.items()},
                "required": [k for k, p in shape.items() if p.required],
                "additionalProperties": False,
            },
        }
    else:
        base = {"type": spec.kind}
    if spec.enum is not None:
        base = {**base, "enum": list(spec.enum)}
    if spec.dynamic and spec.kind != "pathref":
        base = {"oneOf": [base, {"$ref": "#/$defs/pathRef"}]}
    if spec.doc:
        base["description"] = spec.doc
    return base


def component_props_json_schema(name: str) -> dict[str, Any] | None:
    """JSON Schema view of a component's props (for UI forms and docs)."""
    spec = CATALOG.get(name)
    if spec is None:
        return None
    return {
        "type": "object",
        "properties": {k: _prop_json_schema(p) for k, p in spec.props.items()},
        "required": [k for k, p in spec.props.items() if p.required],
        "additionalProperties": False,
    }


def catalog_reference() -> dict[str, Any]:
    """Serializable snapshot of the whole catalog.

    Consumed by the TS↔Python enum-parity test and by the Designer prompt's
    catalog documentation so all three surfaces stay in lockstep.
    """
    def _prop_ref(prop: PropSpec) -> dict[str, Any]:
        ref: dict[str, Any] = {
            "kind": prop.kind,
            "required": prop.required,
            "dynamic": prop.dynamic,
            "enum": list(prop.enum) if prop.enum is not None else None,
        }
        if prop.item_shape is not None:
            ref["item_shape"] = {
                k: _prop_ref(p) for k, p in prop.item_shape.items()
            }
        return ref

    return {
        name: {
            "container": spec.container,
            "doc": spec.doc,
            "props": {key: _prop_ref(prop) for key, prop in spec.props.items()},
        }
        for name, spec in CATALOG.items()
    }


def surface_catalog_cheatsheet() -> str:
    """Compact, generated catalog documentation for the Designer LLM.

    One line per component (kind, doc, prop signatures) derived from the
    specs so the prompt vocabulary can never drift from the validator.
    """

    def _sig(key: str, prop: PropSpec) -> str:
        sig = f"{key}{'*' if prop.required else ''}:{prop.kind}"
        if prop.enum is not None:
            sig += "(" + "|".join(str(v) for v in prop.enum) + ")"
        if prop.item_shape is not None:
            inner = ", ".join(
                _sig(k, p) for k, p in prop.item_shape.items()
            )
            sig += "{" + inner + "}"
        return sig

    lines: list[str] = []
    for spec in _COMPONENTS:
        role = "container" if spec.container else "leaf"
        props = ", ".join(_sig(k, p) for k, p in spec.props.items()) or "—"
        lines.append(f"- {spec.name} ({role}): {spec.doc} Props: {props}")
    return "\n".join(lines)


__all__ = [
    "CATALOG",
    "CONTAINER_COMPONENTS",
    "INPUT_COMPONENTS",
    "OUTPUT_COMPONENTS",
    "OUTPUT_POINTER_PROPS",
    "ComponentSpec",
    "PropSpec",
    "catalog_reference",
    "component_names",
    "component_props_json_schema",
    "component_spec",
    "surface_catalog_cheatsheet",
]
