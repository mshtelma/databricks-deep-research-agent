"""Compile a surface's structured-output slots into a Pydantic schema.

Output components (Table, MetricGrid, KeyFindings, Chart, List) bind SLOTS:
JSON pointers exactly one segment under a binding's output target,
``<target>/data/<slot>``. This module is the single source of truth for the
slot grammar, shared by design-time validation (surface/validation.py) and
the run-time structuring pass (agent/structured_surface.py):

* :func:`collect_output_slots` walks a surface and merges component
  declarations into per-binding :class:`SlotSpec` maps (recording
  contract conflicts as :class:`SlotIssue` instead of raising).
* :func:`build_output_model` turns one binding's slots into a dynamic
  Pydantic model (``extra="forbid"``, size caps expressed IN the schema) fit
  for ``LLMClient.complete(structured_output=Model)``.
* :func:`slot_docs` renders the generic prompt documentation for the slots.
* :func:`resolve_binding_for_run` picks the binding a run should fill.

The slot pointer is exactly ONE segment under ``/data/`` because the surface
pointer grammar has no array indices and the TS ``getAtPointer`` never
traverses arrays — arrays are legal FINAL values only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from databricks_deep_research.surface.pointers import OUTPUT_POINTER_PROPS
from databricks_deep_research.surface.schema import Surface, is_valid_identifier
from databricks_deep_research.surface.tolerant import (
    TolerantWireBase,
    unwrap_placeholder_envelope,
)

# Size caps expressed inside the compiled schema (the model sees them) and
# enforced again by Pydantic validation of the response.
ROWS_CAP = 50
LIST_CAP = 12
METRICS_CAP = 12
CELL_MAX = 500
ITEM_MAX = 400
METRIC_LABEL_MAX = 80
METRIC_VALUE_MAX = 40
METRIC_UNIT_MAX = 16
METRIC_DELTA_MAX = 24
DATE_MAX = 64

SlotKind = Literal["table", "metrics", "strings"]

_CITATION_HINT = "May include citation markers like [Key] from the allowed keys."


@dataclass(frozen=True)
class ColumnSpec:
    """One typed Table column."""

    key: str
    label: str
    type: str  # "string" | "number" | "date"


@dataclass
class SlotSpec:
    """Merged contract for one slot within one binding."""

    slot: str
    kind: SlotKind
    columns: tuple[ColumnSpec, ...] = ()
    component_ids: tuple[str, ...] = ()
    # Keys charts plot from this slot's rows (x_key + y_keys), used by
    # validation to warn when they are absent from a shared Table's columns.
    chart_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class SlotIssue:
    """A slot-contract problem discovered during collection."""

    message: str
    component_id: str
    severity: str = "blocking"  # "blocking" | "warning"


@dataclass
class CollectedSlots:
    """All slots grouped per binding action, plus contract issues."""

    by_action: dict[str, dict[str, SlotSpec]] = field(default_factory=dict)
    issues: list[SlotIssue] = field(default_factory=list)

    def slots_for(self, action: str) -> dict[str, SlotSpec]:
        return self.by_action.get(action, {})


def _pointer_of(value: Any) -> str | None:
    if isinstance(value, dict):
        raw = value.get("path")
        return raw if isinstance(raw, str) else None
    path = getattr(value, "path", None)
    return path if isinstance(path, str) else None


def split_slot_pointer(
    pointer: str, targets: dict[str, str]
) -> tuple[str, str] | None:
    """``(action, slot)`` when *pointer* is ``<target>/data/<slot>``.

    Returns None when the pointer is not exactly one segment under any
    binding target's ``/data/`` subtree.
    """
    for action, target in targets.items():
        prefix = f"{target}/data/"
        if pointer.startswith(prefix):
            rest = pointer[len(prefix) :]
            if rest and "/" not in rest:
                return (action, rest)
    return None


def pointer_under_target(pointer: str, targets: dict[str, str]) -> str | None:
    """The binding action whose output target prefixes *pointer* (or None)."""
    for action, target in targets.items():
        if pointer == target or pointer.startswith(target + "/"):
            return action
    return None


def _columns_from_props(comp_props: dict[str, Any]) -> tuple[ColumnSpec, ...]:
    raw = comp_props.get("columns")
    if not isinstance(raw, list):
        return ()
    columns: list[ColumnSpec] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        label = item.get("label")
        col_type = item.get("type")
        if (
            isinstance(key, str)
            and isinstance(label, str)
            and isinstance(col_type, str)
        ):
            columns.append(ColumnSpec(key=key, label=label, type=col_type))
    return tuple(columns)


def _chart_keys_from_props(comp_props: dict[str, Any]) -> tuple[str, ...]:
    keys: list[str] = []
    x_key = comp_props.get("x_key")
    if isinstance(x_key, str):
        keys.append(x_key)
    y_keys = comp_props.get("y_keys")
    if isinstance(y_keys, list):
        keys.extend(k for k in y_keys if isinstance(k, str))
    return tuple(keys)


def _wanted_kind(component: str) -> SlotKind:
    if component in ("Table", "Chart"):
        return "table"
    if component == "MetricGrid":
        return "metrics"
    return "strings"  # KeyFindings, List


def _new_slot_spec(
    slot: str, comp_id: str, component: str, props: dict[str, Any]
) -> SlotSpec:
    return SlotSpec(
        slot=slot,
        kind=_wanted_kind(component),
        columns=_columns_from_props(props) if component == "Table" else (),
        component_ids=(comp_id,),
        chart_keys=(
            _chart_keys_from_props(props) if component == "Chart" else ()
        ),
    )


def _merge_component_into_slot(
    spec: SlotSpec, comp_id: str, component: str, props: dict[str, Any]
) -> SlotIssue | None:
    """Merge one component's contract into an existing slot spec."""
    spec.component_ids = (*spec.component_ids, comp_id)
    wanted = _wanted_kind(component)

    if spec.kind != wanted:
        return SlotIssue(
            message=(
                f"slot '{spec.slot}' is used both as '{spec.kind}' and as "
                f"'{wanted}' — give each dataset its own slot"
            ),
            component_id=comp_id,
        )

    if component == "Table":
        columns = _columns_from_props(props)
        if spec.columns and columns != spec.columns:
            return SlotIssue(
                message=(
                    f"components sharing slot '{spec.slot}' declare "
                    "conflicting Table columns"
                ),
                component_id=comp_id,
            )
        spec.columns = columns
    elif component == "Chart":
        chart_keys = _chart_keys_from_props(props)
        spec.chart_keys = tuple(dict.fromkeys((*spec.chart_keys, *chart_keys)))
    return None


def collect_output_slots(surface: Surface | dict[str, Any]) -> CollectedSlots:
    """Walk *surface* and group output-component slots per binding action.

    Accepts a parsed :class:`Surface` or a raw dict (runtime path); a dict
    that fails schema validation yields an empty result (fail-soft — the
    save gate already guarantees validity for persisted agents).
    """
    if not isinstance(surface, Surface):
        try:
            surface = Surface.model_validate(surface)
        except Exception:  # noqa: BLE001 — fail-soft by contract
            return CollectedSlots()

    targets = {b.action: b.output.target for b in surface.bindings}
    collected = CollectedSlots()

    for comp in surface.components:
        pointer_prop = OUTPUT_POINTER_PROPS.get(comp.component)
        if pointer_prop is None:
            continue
        pointer = _pointer_of(comp.props.get(pointer_prop))
        if pointer is None:
            continue
        split = split_slot_pointer(pointer, targets)
        if split is None:
            # Not a slot (List over static data, or malformed — validation
            # decides severity; collection just skips it).
            continue
        action, slot = split
        slots = collected.by_action.setdefault(action, {})
        spec = slots.get(slot)
        if spec is None:
            slots[slot] = _new_slot_spec(slot, comp.id, comp.component, comp.props)
            continue
        issue = _merge_component_into_slot(
            spec, comp.id, comp.component, comp.props
        )
        if issue is not None:
            collected.issues.append(issue)

    # A Chart-only table slot synthesizes columns from its keys so the
    # compiled schema still has a concrete row shape.
    for slots in collected.by_action.values():
        for spec in slots.values():
            if spec.kind == "table" and not spec.columns and spec.chart_keys:
                x_key = spec.chart_keys[0]
                spec.columns = tuple(
                    ColumnSpec(
                        key=key,
                        label=key,
                        type="string" if key == x_key else "number",
                    )
                    for key in spec.chart_keys
                )
    return collected


# ---------------------------------------------------------------------------
# Schema build
# ---------------------------------------------------------------------------


def _row_model(slot: str, columns: tuple[ColumnSpec, ...]) -> type[BaseModel]:
    fields: dict[str, Any] = {}
    for col in columns:
        if col.type == "number":
            fields[col.key] = (
                float,
                Field(description=f"{col.label} (numeric — no citation markers)"),
            )
        elif col.type == "date":
            fields[col.key] = (
                str,
                Field(
                    max_length=DATE_MAX,
                    description=f"{col.label} — ISO-8601 date (YYYY-MM-DD)",
                ),
            )
        else:
            fields[col.key] = (
                str,
                Field(
                    max_length=CELL_MAX,
                    description=f"{col.label}. {_CITATION_HINT}",
                ),
            )
    model: type[BaseModel] = create_model(
        f"Row_{slot}",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )
    return model


def _metric_model(slot: str) -> type[BaseModel]:
    model: type[BaseModel] = create_model(
        f"Metric_{slot}",
        __config__=ConfigDict(extra="forbid"),
        label=(str, Field(max_length=METRIC_LABEL_MAX)),
        value=(
            str,
            Field(
                max_length=METRIC_VALUE_MAX,
                description="The metric value as text (no citation markers).",
            ),
        ),
        unit=(str | None, Field(default=None, max_length=METRIC_UNIT_MAX)),
        delta=(
            str | None,
            Field(
                default=None,
                max_length=METRIC_DELTA_MAX,
                description="Change vs a stated baseline, e.g. '+4 vs prior'.",
            ),
        ),
    )
    return model


def build_output_model(
    slots: dict[str, SlotSpec], name: str = "SurfaceRunOutput"
) -> type[BaseModel]:
    """Dynamic Pydantic model over *slots* (all slots required, closed)."""
    fields: dict[str, Any] = {}
    for slot_name, spec in slots.items():
        if not is_valid_identifier(slot_name):
            # Validation blocks these at save; runtime skips defensively.
            continue
        if spec.kind == "table":
            row = _row_model(slot_name, spec.columns)
            fields[slot_name] = (
                list[row],  # type: ignore[valid-type]
                Field(
                    max_length=ROWS_CAP,
                    description=f"Rows for slot '{slot_name}' (≤{ROWS_CAP}).",
                ),
            )
        elif spec.kind == "metrics":
            metric = _metric_model(slot_name)
            fields[slot_name] = (
                list[metric],  # type: ignore[valid-type]
                Field(
                    max_length=METRICS_CAP,
                    description=f"Metric cards for slot '{slot_name}' "
                    f"(≤{METRICS_CAP}).",
                ),
            )
        else:
            fields[slot_name] = (
                list[str],
                Field(
                    max_length=LIST_CAP,
                    description=(
                        f"Items for slot '{slot_name}' (≤{LIST_CAP}, each "
                        f"≤{ITEM_MAX} chars). {_CITATION_HINT}"
                    ),
                ),
            )
    model: type[BaseModel] = create_model(
        name,
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )
    return model


# ---------------------------------------------------------------------------
# Per-slot wire models (v2 native structuring pass)
# ---------------------------------------------------------------------------
# One small model per slot beats one big schema on the Databricks Claude
# structured-output path (json_schema = forced tool call with a hard key
# budget, and a single big call is the timeout mode observed live). Wire
# models are built on TolerantWireBase (create_model cannot combine
# __config__ with __base__, so extra="forbid" and the loose-shape coercion
# live on the base). The v1 builders above stay untouched until the
# single-shot pass is removed.

SOURCE_REFS_CAP = 8

_SOURCE_REFS_DESC = (
    "Index strings of the evidence sources supporting this item, "
    'e.g. ["1", "3"].'
)


def _source_refs_field() -> tuple[Any, Any]:
    return (
        list[str],
        Field(
            default_factory=list,
            max_length=SOURCE_REFS_CAP,
            description=_SOURCE_REFS_DESC,
        ),
    )


def _wire_row_model(slot: str, columns: tuple[ColumnSpec, ...]) -> type[BaseModel]:
    fields: dict[str, Any] = {}
    for col in columns:
        if col.type == "number":
            fields[col.key] = (
                float,
                Field(description=f"{col.label} (numeric — no citation markers)"),
            )
        elif col.type == "date":
            fields[col.key] = (
                str,
                Field(
                    max_length=DATE_MAX,
                    description=f"{col.label} — ISO-8601 date (YYYY-MM-DD)",
                ),
            )
        else:
            fields[col.key] = (
                str,
                Field(
                    max_length=CELL_MAX,
                    description=f"{col.label}. {_CITATION_HINT}",
                ),
            )
    if "source_refs" not in fields:
        fields["source_refs"] = _source_refs_field()
    model: type[BaseModel] = create_model(
        f"WireRow_{slot}",
        __base__=TolerantWireBase,
        **fields,
    )
    return model


def _wire_metric_model(slot: str) -> type[BaseModel]:
    model: type[BaseModel] = create_model(
        f"WireMetric_{slot}",
        __base__=TolerantWireBase,
        label=(str, Field(max_length=METRIC_LABEL_MAX)),
        value=(
            str,
            Field(
                max_length=METRIC_VALUE_MAX,
                description="The metric value as text (no citation markers).",
            ),
        ),
        unit=(str | None, Field(default=None, max_length=METRIC_UNIT_MAX)),
        delta=(
            str | None,
            Field(
                default=None,
                max_length=METRIC_DELTA_MAX,
                description="Change vs a stated baseline, e.g. '+4 vs prior'.",
            ),
        ),
        source_refs=_source_refs_field(),
    )
    return model


def _wire_item_model(slot: str) -> type[BaseModel]:
    model: type[BaseModel] = create_model(
        f"WireItem_{slot}",
        __base__=TolerantWireBase,
        text=(
            str,
            Field(
                max_length=ITEM_MAX,
                description=f"A self-contained statement. {_CITATION_HINT}",
            ),
        ),
        source_refs=_source_refs_field(),
    )
    return model


class WireModelBase(TolerantWireBase):
    """Base for top-level wire models: unwraps transport envelopes first.

    Subclass before-validators run before inherited ones, so the payload is
    unwrapped before ``TolerantWireBase`` normalizes the loose field shapes
    (verified by the wire-model unit tests).
    """

    @model_validator(mode="before")
    @classmethod
    def _unwrap_transport(cls, data: Any) -> Any:
        return unwrap_placeholder_envelope(cls, data)


def build_slot_wire_model(slot_name: str, spec: SlotSpec) -> type[BaseModel]:
    """Wire model for ONE slot: a single top-level field named after it."""
    if not is_valid_identifier(slot_name):
        raise ValueError(f"invalid slot name: {slot_name!r}")
    if spec.kind == "table":
        item: type[BaseModel] = _wire_row_model(slot_name, spec.columns)
        cap = ROWS_CAP
        desc = f"Rows for slot '{slot_name}' (≤{ROWS_CAP})."
    elif spec.kind == "metrics":
        item = _wire_metric_model(slot_name)
        cap = METRICS_CAP
        desc = f"Metric cards for slot '{slot_name}' (≤{METRICS_CAP})."
    else:
        item = _wire_item_model(slot_name)
        cap = LIST_CAP
        desc = f"Items for slot '{slot_name}' (≤{LIST_CAP})."
    fields: dict[str, Any] = {
        slot_name: (
            list[item],  # type: ignore[valid-type]
            Field(max_length=cap, description=desc),
        )
    }
    model: type[BaseModel] = create_model(
        f"Wire_{slot_name}",
        __base__=WireModelBase,
        **fields,
    )
    return model


def wire_slot_docs(slot_name: str, spec: SlotSpec) -> str:
    """Single-slot contract text for one wire prompt."""
    if spec.kind == "table":
        cols = "; ".join(f"{c.key} ({c.type}): {c.label}" for c in spec.columns)
        return (
            f"Slot '{slot_name}' — table rows (≤{ROWS_CAP}). Columns: {cols}\n"
            "Every row MUST include source_refs — the index strings of the "
            'evidence sources supporting it (e.g. ["1", "3"]). OMIT rows you '
            "cannot support with at least one source."
        )
    if spec.kind == "metrics":
        return (
            f"Slot '{slot_name}' — headline metric cards (≤{METRICS_CAP}): "
            "label, value, optional unit and delta.\n"
            "Include source_refs (evidence index strings) whenever the "
            "metric comes from a source."
        )
    return (
        f"Slot '{slot_name}' — concise text items (≤{LIST_CAP}), each an "
        f"object with 'text' (a self-contained statement, ≤{ITEM_MAX} chars) "
        "and 'source_refs'.\n"
        "Every item MUST include source_refs — the index strings of the "
        'evidence sources supporting it (e.g. ["1", "3"]). OMIT items you '
        "cannot support with at least one source."
    )


def slot_docs(slots: dict[str, SlotSpec]) -> str:
    """Generic per-slot documentation for the structuring prompt."""
    lines: list[str] = []
    for slot_name, spec in slots.items():
        if spec.kind == "table":
            cols = "; ".join(
                f"{c.key} ({c.type}): {c.label}" for c in spec.columns
            )
            lines.append(
                f"- {slot_name} — table rows (≤{ROWS_CAP}). Columns: {cols}"
            )
        elif spec.kind == "metrics":
            lines.append(
                f"- {slot_name} — headline metric cards "
                f"(≤{METRICS_CAP}): label, value, optional unit and delta"
            )
        else:
            lines.append(
                f"- {slot_name} — concise text items (≤{LIST_CAP}), each a "
                "self-contained statement"
            )
    return "\n".join(lines)


@dataclass(frozen=True)
class ResolvedBinding:
    """The binding a run fills, plus whether the choice was ambiguous."""

    action: str
    slots: dict[str, SlotSpec]
    ambiguous: bool = False


def resolve_binding_for_run(
    collected: CollectedSlots, surface_action: str | None
) -> ResolvedBinding | None:
    """Pick the binding whose slots this run should fill.

    Explicit ``surface_action`` wins; otherwise the sole slotted binding;
    with several, the first (insertion order) is used and flagged ambiguous.
    """
    slotted = {a: s for a, s in collected.by_action.items() if s}
    if not slotted:
        return None
    if surface_action is not None:
        slots = slotted.get(surface_action)
        if slots is None:
            return None
        return ResolvedBinding(action=surface_action, slots=slots)
    if len(slotted) == 1:
        action, slots = next(iter(slotted.items()))
        return ResolvedBinding(action=action, slots=slots)
    action, slots = next(iter(slotted.items()))
    return ResolvedBinding(action=action, slots=slots, ambiguous=True)


__all__ = [
    "CELL_MAX",
    "ITEM_MAX",
    "LIST_CAP",
    "METRICS_CAP",
    "ROWS_CAP",
    "SOURCE_REFS_CAP",
    "CollectedSlots",
    "ColumnSpec",
    "ResolvedBinding",
    "SlotIssue",
    "SlotSpec",
    "WireModelBase",
    "build_output_model",
    "build_slot_wire_model",
    "collect_output_slots",
    "pointer_under_target",
    "resolve_binding_for_run",
    "slot_docs",
    "split_slot_pointer",
    "wire_slot_docs",
]
