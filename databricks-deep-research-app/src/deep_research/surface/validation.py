"""Deterministic surface validation — pure checks, no LLM, never silent.

``validate_surface(definition)`` is the single validator every writer goes
through: the agents_v2 save gate (schemas/agent_v2.py), the Designer's
``set_surface`` tool (errors return to the LLM as the tool result so it can
self-correct), and the YAML-import metadata carry (invalid → dropped with a
structured warning). Blocking errors mean the surface would misrender or
mis-run; warnings are advisory and never block a save.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from deep_research.surface.catalog import (
    CATALOG,
    CONTAINER_COMPONENTS,
    INPUT_COMPONENTS,
    OUTPUT_COMPONENTS,
    OUTPUT_POINTER_PROPS,
    PropSpec,
)
from deep_research.surface.output_schema import (
    build_output_model,
    collect_output_slots,
    pointer_under_target,
    split_slot_pointer,
)
from deep_research.surface.schema import (
    PathRef,
    RunOptions,
    Surface,
    SurfaceComponent,
    is_valid_identifier,
    is_valid_pointer,
    resolve_pointer,
)

MAX_COMPONENTS = 100
MAX_SURFACE_BYTES = 64 * 1024

# Binding input keys that collide with pipeline-owned state/template variables.
# "query" is deliberately NOT here: it is special-cased into the job's query
# field and never seeded as workflow state. Sources (all verified):
# - forced template vars: agents/harness.py template_vars {"query", "tool_catalog"}
# - selector-shadowed keys: workflow/runtime_core/selectors.py SELECTORS table
#   (resolve_input_key wins over state.get, so a seeded key would be invisible)
# - orchestrator-seeded keys: framework_orchestrator init-state seeding
# - harness auto-injected context keys
RESERVED_INPUT_KEYS: frozenset[str] = frozenset(
    {
        "tool_catalog",
        # selector-shadowed
        "background_summary",
        "data_landscape",
        "discovered_sources",
        "plan",
        "findings",
        "observation",
        "all_observations",
        "sources_count",
        "current_step",
        "step_title",
        "claims",
        "verification_summary",
        "analysis_summary",
        "verification_details",
        # orchestrator-seeded
        "conversation_history",
        "existing_sources",
        "prior_sources_for_seed",
        "seed_prior_sources",
        # harness auto-injected
        "current_date",
        "current_iso_datetime",
        "current_timezone",
        "compute_namespace",
        "revision_block_md",
        "source_quality",
        "chat_memory_appendix",
    }
)

# `{/json/pointer}` placeholders inside a string binding input (query
# templates). Only brace groups starting with `/` are substitution sites;
# plain `{word}` braces are left alone so they cannot collide with
# prompt-style placeholders.
_TEMPLATE_POINTER_RE = re.compile(r"\{(/[^}]*)\}")

_ENUM_RUN_OPTIONS: dict[str, frozenset[str]] = {
    "research_depth": frozenset({"auto", "light", "medium", "extended"}),
    "query_mode": frozenset({"simple", "web_search", "deep_research"}),
    "source_scope": frozenset({"enterprise_only", "web_only", "all"}),
    "turn_intent": frozenset({"auto", "chat", "research"}),
}

_BOOLEAN_RUN_OPTIONS: frozenset[str] = frozenset(
    {
        "verify_sources",
        "enable_plan_review",
        "enable_cross_session_memory",
        "allow_live_search",
    }
)

_STRING_RUN_OPTIONS: frozenset[str] = frozenset({"tone", "output_language"})


@dataclass(frozen=True)
class SurfaceValidationError:
    """One surface violation. Mirrors ``SemanticValidationError``'s shape so
    API adapters and banner rendering treat both uniformly."""

    message: str
    path: str | None = None
    severity: str = "blocking"  # "blocking" | "warning"


def _err(message: str, path: str | None = None) -> SurfaceValidationError:
    return SurfaceValidationError(message=message, path=path)


def _warn(message: str, path: str | None = None) -> SurfaceValidationError:
    return SurfaceValidationError(message=message, path=path, severity="warning")


def has_blocking(errors: Iterable[SurfaceValidationError]) -> bool:
    """True when any error in *errors* is blocking."""
    return any(e.severity == "blocking" for e in errors)


def _is_pathref_dict(value: Any) -> bool:
    if not isinstance(value, PathRef) and not isinstance(value, dict):
        return False
    if isinstance(value, PathRef):
        return True
    return set(value.keys()) == {"path"} and isinstance(value.get("path"), str)


def _pathref_pointer(value: Any) -> str | None:
    if isinstance(value, PathRef):
        return value.path
    if isinstance(value, dict):
        raw = value.get("path")
        return raw if isinstance(raw, str) else None
    return None


def _format_allowed(values: frozenset[str]) -> str:
    return ", ".join(f"'{value}'" for value in sorted(values))


def _validate_run_options(
    binding_index: int,
    options: RunOptions,
) -> list[SurfaceValidationError]:
    errors: list[SurfaceValidationError] = []
    for option_name, allowed_values in _ENUM_RUN_OPTIONS.items():
        value = getattr(options, option_name)
        if value is None or isinstance(value, PathRef):
            continue
        option_path = f"surface.bindings[{binding_index}].options.{option_name}"
        if not isinstance(value, str) or value not in allowed_values:
            errors.append(
                _err(
                    f"option '{option_name}' must be one of "
                    f"{_format_allowed(allowed_values)} or a PathRef",
                    option_path,
                )
            )

    for option_name in _BOOLEAN_RUN_OPTIONS:
        value = getattr(options, option_name)
        if value is None or isinstance(value, PathRef):
            continue
        if not isinstance(value, bool):
            errors.append(
                _err(
                    f"option '{option_name}' must be a boolean or a PathRef",
                    f"surface.bindings[{binding_index}].options.{option_name}",
                )
            )

    for option_name in _STRING_RUN_OPTIONS:
        value = getattr(options, option_name)
        if value is None or isinstance(value, PathRef):
            continue
        if not isinstance(value, str):
            errors.append(
                _err(
                    f"option '{option_name}' must be a string or a PathRef",
                    f"surface.bindings[{binding_index}].options.{option_name}",
                )
            )
    return errors


def _prop_matches_kind(value: Any, spec: PropSpec) -> bool:
    if spec.kind == "pathref":
        pointer = _pathref_pointer(value)
        return (
            _is_pathref_dict(value)
            and pointer is not None
            and is_valid_pointer(pointer)
        )
    if spec.kind == "string":
        return isinstance(value, str)
    if spec.kind == "boolean":
        return isinstance(value, bool)
    if spec.kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if spec.kind == "options":
        return isinstance(value, list) and all(
            isinstance(item, dict)
            and set(item.keys()) == {"label", "value"}
            and isinstance(item.get("label"), str)
            and isinstance(item.get("value"), str)
            for item in value
        )
    if spec.kind == "string_list":
        return (
            isinstance(value, list)
            and len(value) > 0
            and all(isinstance(item, str) for item in value)
        )
    if spec.kind == "object_list":
        if not isinstance(value, list) or len(value) == 0:
            return False
        shape = spec.item_shape or {}
        for item in value:
            if not isinstance(item, dict):
                return False
            if set(item.keys()) - set(shape.keys()):
                return False  # unknown item keys
            for key, item_spec in shape.items():
                if key not in item:
                    if item_spec.required:
                        return False
                    continue
                if not _prop_matches_kind(item[key], item_spec):
                    return False
                if item_spec.enum is not None and item[key] not in item_spec.enum:
                    return False
        return True
    return False


def _validate_props(
    comp: SurfaceComponent, path: str
) -> list[SurfaceValidationError]:
    spec = CATALOG[comp.component]
    errors: list[SurfaceValidationError] = []
    for key in comp.props:
        if key not in spec.props:
            errors.append(
                _err(
                    f"unknown prop '{key}' for component '{comp.component}' "
                    f"(allowed: {', '.join(sorted(spec.props)) or 'none'})",
                    f"{path}.props.{key}",
                )
            )
    for key, prop_spec in spec.props.items():
        if key not in comp.props:
            if prop_spec.required:
                errors.append(
                    _err(
                        f"component '{comp.component}' requires prop '{key}'",
                        f"{path}.props",
                    )
                )
            continue
        value = comp.props[key]
        prop_path = f"{path}.props.{key}"
        if prop_spec.dynamic and _is_pathref_dict(value):
            pointer = _pathref_pointer(value)
            if pointer is None or not is_valid_pointer(pointer):
                errors.append(
                    _err(f"invalid JSON pointer in prop '{key}'", prop_path)
                )
            continue
        if not _prop_matches_kind(value, prop_spec):
            expected = prop_spec.kind + (" or {path}" if prop_spec.dynamic else "")
            errors.append(
                _err(f"prop '{key}' must be {expected}", prop_path)
            )
            continue
        if prop_spec.enum is not None and value not in prop_spec.enum:
            allowed = ", ".join(repr(v) for v in prop_spec.enum)
            errors.append(
                _err(f"prop '{key}' must be one of: {allowed}", prop_path)
            )
    return errors


def _validate_components(
    surface: Surface,
) -> tuple[list[SurfaceValidationError], dict[str, SurfaceComponent]]:
    errors: list[SurfaceValidationError] = []
    by_id: dict[str, SurfaceComponent] = {}

    if len(surface.components) > MAX_COMPONENTS:
        errors.append(
            _err(
                f"surface has {len(surface.components)} components; "
                f"the cap is {MAX_COMPONENTS}",
                "surface.components",
            )
        )

    for index, comp in enumerate(surface.components):
        path = f"surface.components[{index}]"
        if comp.id in by_id:
            errors.append(_err(f"duplicate component id '{comp.id}'", path))
            continue
        by_id[comp.id] = comp
        if comp.component not in CATALOG:
            errors.append(
                _err(
                    f"unknown component '{comp.component}' "
                    f"(catalog: {', '.join(sorted(CATALOG))})",
                    path,
                )
            )
            continue
        if comp.children and comp.component not in CONTAINER_COMPONENTS:
            errors.append(
                _err(
                    f"component '{comp.component}' cannot have children "
                    f"(containers: {', '.join(sorted(CONTAINER_COMPONENTS))})",
                    f"{path}.children",
                )
            )
        errors.extend(_validate_props(comp, path))

    if "root" not in by_id:
        errors.append(_err("surface must contain a component with id 'root'"))
        return (errors, by_id)

    # Children must exist; detect cycles + unreachable components from root.
    for comp in surface.components:
        for child_id in comp.children:
            if child_id not in by_id:
                errors.append(
                    _err(
                        f"component '{comp.id}' references unknown child "
                        f"'{child_id}'",
                        f"surface.components[{comp.id}].children",
                    )
                )

    visited: set[str] = set()
    in_stack: set[str] = set()
    cycle_found = False

    def _walk(node_id: str) -> None:
        nonlocal cycle_found
        if node_id in in_stack:
            cycle_found = True
            return
        if node_id in visited or node_id not in by_id:
            return
        visited.add(node_id)
        in_stack.add(node_id)
        for child_id in by_id[node_id].children:
            _walk(child_id)
        in_stack.discard(node_id)

    _walk("root")
    if cycle_found:
        errors.append(_err("component tree contains a cycle", "surface.components"))
    orphans = sorted(set(by_id) - visited)
    if orphans:
        errors.append(
            _warn(
                "components not reachable from 'root' will never render: "
                + ", ".join(orphans),
                "surface.components",
            )
        )
    return (errors, by_id)


def _validate_bindings(
    surface: Surface,
    required_inputs: list[str],
) -> list[SurfaceValidationError]:
    errors: list[SurfaceValidationError] = []

    button_actions: set[str] = set()
    for comp in surface.components:
        if comp.component == "Button":
            action = comp.props.get("action")
            if isinstance(action, str):
                button_actions.add(action)

    # value-pointer → component type, to flag a query bound to a non-free-text
    # input (a research query should be free-text; see the query check below).
    pointer_component: dict[str, str] = {}
    for comp in surface.components:
        if comp.component in INPUT_COMPONENTS:
            ptr = _pathref_pointer(comp.props.get("value"))
            if ptr is not None:
                pointer_component[ptr] = comp.component

    seen_actions: set[str] = set()
    targets: list[tuple[str, str]] = []
    for index, binding in enumerate(surface.bindings):
        path = f"surface.bindings[{index}]"
        if binding.action in seen_actions:
            errors.append(_err(f"duplicate binding action '{binding.action}'", path))
        seen_actions.add(binding.action)
        errors.extend(_validate_run_options(index, binding.options))

        for key, value in binding.inputs.items():
            key_path = f"{path}.inputs.{key}"
            if key != "query":
                if not is_valid_identifier(key):
                    errors.append(
                        _err(
                            f"input key '{key}' is not a valid identifier "
                            "([A-Za-z_][A-Za-z0-9_]*)",
                            key_path,
                        )
                    )
                elif key in RESERVED_INPUT_KEYS:
                    errors.append(
                        _err(
                            f"input key '{key}' collides with pipeline-owned "
                            "state and is reserved",
                            key_path,
                        )
                    )
            if isinstance(value, str):
                for match in _TEMPLATE_POINTER_RE.finditer(value):
                    if not is_valid_pointer(match.group(1)):
                        errors.append(
                            _err(
                                f"malformed pointer placeholder "
                                f"'{{{match.group(1)}}}' in input '{key}'",
                                key_path,
                            )
                        )
            pointer = _pathref_pointer(value)
            if pointer is not None and is_valid_pointer(pointer):
                found, _ = resolve_pointer(surface.data_model, pointer)
                if not found:
                    errors.append(
                        _warn(
                            f"input '{key}' reads '{pointer}' which is not "
                            "initialized in data_model",
                            key_path,
                        )
                    )
            # A research query should come from free text. Binding it to a
            # Select/Checkbox means the run has no query unless an option is
            # picked (and a sibling free-text field, if any, is ignored) — the
            # "pick-or-custom" anti-pattern. Non-blocking (authoring hint).
            if key == "query" and pointer is not None:
                q_comp = pointer_component.get(pointer)
                if q_comp in ("Select", "Checkbox"):
                    errors.append(
                        _warn(
                            f"binding '{binding.action}' binds its query to a "
                            f"{q_comp} input; a research query should come from a "
                            "free-text field (TextField/TextArea) so a run isn't "
                            "blocked when no option is selected",
                            key_path,
                        )
                    )

        missing = [k for k in required_inputs if k not in binding.inputs]
        if missing:
            errors.append(
                _err(
                    f"binding '{binding.action}' does not provide required "
                    f"workflow input(s): {', '.join(missing)}",
                    f"{path}.inputs",
                )
            )

        targets.append((binding.action, binding.output.target))
        if binding.action not in button_actions:
            errors.append(
                _warn(
                    f"binding '{binding.action}' has no Button that triggers it",
                    path,
                )
            )

    for action in sorted(button_actions - seen_actions):
        errors.append(
            _err(f"Button references action '{action}' but no binding defines it")
        )

    # Output targets must be pairwise disjoint (no equal / prefix-nested paths)
    # so concurrent runs can never write into each other's regions.
    for i, (action_a, target_a) in enumerate(targets):
        for action_b, target_b in targets[i + 1 :]:
            seg_a = target_a.strip("/").split("/")
            seg_b = target_b.strip("/").split("/")
            shorter, longer = sorted((seg_a, seg_b), key=len)
            if longer[: len(shorter)] == shorter:
                errors.append(
                    _err(
                        f"output targets of bindings '{action_a}' and "
                        f"'{action_b}' overlap ('{target_a}' vs '{target_b}')"
                    )
                )

    # Result components should read a binding output (or at least a valid,
    # initialized pointer) — advisory only.
    binding_targets = {t for _, t in targets}
    for comp in surface.components:
        if comp.component in {"ReportRegion", "StatusBadge"}:
            pointer = _pathref_pointer(comp.props.get("source"))
            if pointer is not None and pointer not in binding_targets:
                errors.append(
                    _warn(
                        f"'{comp.id}' reads '{pointer}' which is not any "
                        "binding's output target",
                        f"surface.components[{comp.id}].props.source",
                    )
                )

    # Two-way inputs should start from an initialized data-model slot.
    for comp in surface.components:
        if comp.component in INPUT_COMPONENTS:
            pointer = _pathref_pointer(comp.props.get("value"))
            if pointer is not None and is_valid_pointer(pointer):
                found, _ = resolve_pointer(surface.data_model, pointer)
                if not found:
                    errors.append(
                        _warn(
                            f"input '{comp.id}' binds '{pointer}' which is not "
                            "initialized in data_model",
                            f"surface.components[{comp.id}].props.value",
                        )
                    )
    return errors


def _validate_output_components(
    surface: Surface,
) -> list[SurfaceValidationError]:
    """Rules for structured-output components (slots, Tabs, Table columns).

    Slots are pointers exactly one segment under a binding output target
    (``<target>/data/<slot>``) — required because the pointer grammar has no
    array indices and the renderer never traverses arrays, so the payload
    must sit whole at the slot pointer.
    """
    errors: list[SurfaceValidationError] = []
    targets = {b.action: b.output.target for b in surface.bindings}
    by_id = {c.id: c for c in surface.components}
    parent_of: dict[str, str] = {}
    for comp in surface.components:
        for child_id in comp.children:
            parent_of[child_id] = comp.id

    for comp in surface.components:
        path = f"surface.components[{comp.id}]"

        # --- Tabs / TabPane structure ------------------------------------
        if comp.component == "Tabs":
            if not comp.children:
                errors.append(
                    _err(
                        "Tabs must contain at least one TabPane",
                        f"{path}.children",
                    )
                )
            for child_id in comp.children:
                child = by_id.get(child_id)
                if child is not None and child.component != "TabPane":
                    errors.append(
                        _err(
                            f"Tabs child '{child_id}' must be a TabPane "
                            f"(got '{child.component}')",
                            f"{path}.children",
                        )
                    )
        if comp.component == "TabPane":
            parent_id = parent_of.get(comp.id)
            parent = by_id.get(parent_id) if parent_id is not None else None
            if parent is None or parent.component != "Tabs":
                errors.append(
                    _err(
                        f"TabPane '{comp.id}' must be a direct child of a "
                        "Tabs container",
                        path,
                    )
                )

        # --- Table column details -----------------------------------------
        if comp.component == "Table":
            raw_cols = comp.props.get("columns")
            if isinstance(raw_cols, list):
                seen_keys: set[str] = set()
                for col in raw_cols:
                    key = col.get("key") if isinstance(col, dict) else None
                    if not isinstance(key, str):
                        continue
                    col_path = f"{path}.props.columns"
                    if not is_valid_identifier(key):
                        errors.append(
                            _err(
                                f"Table '{comp.id}' column key '{key}' must "
                                "be a valid identifier",
                                col_path,
                            )
                        )
                    if key in seen_keys:
                        errors.append(
                            _err(
                                f"Table '{comp.id}' has duplicate column "
                                f"key '{key}'",
                                col_path,
                            )
                        )
                    seen_keys.add(key)

        # --- Chart key grammar ---------------------------------------------
        if comp.component == "Chart":
            keys: list[str] = []
            x_key = comp.props.get("x_key")
            if isinstance(x_key, str):
                keys.append(x_key)
            y_keys = comp.props.get("y_keys")
            if isinstance(y_keys, list):
                keys.extend(k for k in y_keys if isinstance(k, str))
            for key in keys:
                if not is_valid_identifier(key):
                    errors.append(
                        _err(
                            f"Chart '{comp.id}' key '{key}' must be a valid "
                            "identifier",
                            f"{path}.props",
                        )
                    )

        # --- Slot pointer grammar -------------------------------------------
        pointer_prop = OUTPUT_POINTER_PROPS.get(comp.component)
        if pointer_prop is None:
            continue
        pointer = _pathref_pointer(comp.props.get(pointer_prop))
        if pointer is None or not is_valid_pointer(pointer):
            continue  # shape errors are already reported by prop validation
        prop_path = f"{path}.props.{pointer_prop}"
        split = split_slot_pointer(pointer, targets)
        if split is not None:
            _action, slot = split
            if not is_valid_identifier(slot):
                errors.append(
                    _err(
                        f"slot name '{slot}' must be a valid identifier "
                        "([A-Za-z_][A-Za-z0-9_]*)",
                        prop_path,
                    )
                )
            continue
        under_action = pointer_under_target(pointer, targets)
        if under_action is not None:
            errors.append(
                _err(
                    f"'{comp.id}' binds '{pointer}' inside binding "
                    f"'{under_action}'s output target but not as a slot — "
                    "use '<target>/data/<slot>' with exactly one segment "
                    "after /data/",
                    prop_path,
                )
            )
        elif comp.component != "List":
            # List may read static arrays; the model-filled components
            # outside any output target will simply never be filled.
            errors.append(
                _warn(
                    f"'{comp.id}' binds '{pointer}' which is not under any "
                    "binding's output target; the model will never fill it",
                    prop_path,
                )
            )

    # --- Slot contract conflicts + compiled-schema sanity -------------------
    collected = collect_output_slots(surface)
    for issue in collected.issues:
        entry = (
            _err(issue.message, f"surface.components[{issue.component_id}]")
            if issue.severity == "blocking"
            else _warn(issue.message, f"surface.components[{issue.component_id}]")
        )
        errors.append(entry)

    for action, slots in collected.by_action.items():
        for spec in slots.values():
            if spec.kind == "table" and spec.chart_keys and spec.columns:
                column_keys = {c.key for c in spec.columns}
                missing = [k for k in spec.chart_keys if k not in column_keys]
                if missing:
                    errors.append(
                        _warn(
                            f"chart keys {missing} on slot '{spec.slot}' are "
                            "not columns of the Table sharing that slot",
                            "surface.components",
                        )
                    )
                # y series should be numeric columns (x may be any type).
                non_numeric = [
                    c.key
                    for c in spec.columns
                    if c.key in spec.chart_keys[1:] and c.type != "number"
                ]
                if non_numeric:
                    errors.append(
                        _warn(
                            f"chart y keys {non_numeric} on slot "
                            f"'{spec.slot}' reference non-number columns; "
                            "their points will be dropped at render",
                            "surface.components",
                        )
                    )
        if not slots:
            continue
        try:
            schema = build_output_model(
                slots, f"SurfaceOutput_{action}"
            ).model_json_schema()
        except Exception:  # noqa: BLE001 — surfaced as a finding, never raised
            errors.append(
                _err(
                    f"could not compile the structured-output schema for "
                    f"binding '{action}'",
                    "surface",
                )
            )
            continue
        if len(json.dumps(schema)) > 16 * 1024:
            errors.append(
                _warn(
                    f"compiled structured-output schema for binding "
                    f"'{action}' exceeds 16KB; consider fewer slots/columns",
                    "surface",
                )
            )
    return errors


def _validate_layout(
    surface: Surface,
    by_id: dict[str, SurfaceComponent],
) -> list[SurfaceValidationError]:
    """Validate optional host-frame layout metadata."""
    errors: list[SurfaceValidationError] = []
    layout = surface.layout
    if layout is None or layout.sections is None:
        return errors

    # An inputs/results section with no children falls back to host inference of its
    # contents from the component tree. Warn (non-blocking) only when the surface
    # actually has matching-category content to place, so the author can list
    # children to control placement (or omit the section) — never a false positive
    # on a genuinely contentless role.
    input_present = any(c.component in INPUT_COMPONENTS for c in surface.components)
    result_present = any(
        c.component in OUTPUT_COMPONENTS or c.component in {"StatusBadge", "ReportRegion"}
        for c in surface.components
    )

    seen_sections: set[str] = set()
    for index, section in enumerate(layout.sections):
        path = f"surface.layout.sections[{index}]"
        if section.id in seen_sections:
            errors.append(_err(f"duplicate layout section id '{section.id}'", path))
        seen_sections.add(section.id)
        if not section.children and section.role in ("inputs", "results"):
            has_content = input_present if section.role == "inputs" else result_present
            if has_content:
                errors.append(
                    _warn(
                        f"layout section '{section.id}' (role '{section.role}') lists "
                        "no children; the host will infer its contents from the "
                        "component tree — set children to control placement",
                        path,
                    )
                )
        for child_id in section.children:
            if child_id not in by_id:
                errors.append(
                    _err(
                        f"layout section '{section.id}' references unknown child "
                        f"'{child_id}'",
                        f"{path}.children",
                    )
                )
    return errors


def validate_surface(definition: dict[str, Any]) -> list[SurfaceValidationError]:
    """Validate ``definition["surface"]`` against schema, catalog, and bindings.

    Absent surface → no errors (surfaces are optional). Returns ALL findings
    (blocking + warning); callers gate on :func:`has_blocking`.
    """
    raw = definition.get("surface")
    if raw is None:
        return []
    if not isinstance(raw, dict):
        return [_err("surface must be an object", "surface")]

    try:
        serialized = json.dumps(raw, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return [_err("surface is not JSON-serializable", "surface")]
    if len(serialized.encode("utf-8")) > MAX_SURFACE_BYTES:
        return [
            _err(
                f"surface exceeds the {MAX_SURFACE_BYTES // 1024}KB size cap",
                "surface",
            )
        ]

    try:
        surface = Surface.model_validate(raw)
    except ValidationError as exc:
        return [
            _err(
                error.get("msg", "invalid value"),
                "surface." + ".".join(str(loc) for loc in error.get("loc", ())),
            )
            for error in exc.errors()
        ]

    raw_required = definition.get("required_inputs")
    if isinstance(raw_required, list):
        required_inputs = [k for k in raw_required if isinstance(k, str) and k]
    else:
        required_inputs = []
    if not required_inputs:
        required_inputs = ["query"]

    errors, by_id = _validate_components(surface)
    errors.extend(_validate_bindings(surface, required_inputs))
    errors.extend(_validate_output_components(surface))
    errors.extend(_validate_layout(surface, by_id))
    return errors


__all__ = [
    "MAX_COMPONENTS",
    "MAX_SURFACE_BYTES",
    "RESERVED_INPUT_KEYS",
    "SurfaceValidationError",
    "has_blocking",
    "validate_surface",
]
