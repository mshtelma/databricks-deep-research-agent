"""Tolerant validation for structured-output wire models.

The per-slot wire calls (``agent/structured_surface.py``) validate LLM JSON
against dynamic Pydantic models. Verbose LLMs reliably emit a handful of
loose shapes that a strict schema hard-rejects — and one rejection would
lose a whole slot. This module absorbs those shapes instead:

* :class:`TolerantWireBase` — base class for every wire model (passed as
  ``__base__`` to ``create_model``; pydantic v2 forbids combining
  ``__config__`` with ``__base__``, so ``extra="forbid"`` lives here). One
  ``mode="before"`` validator normalizes every declared field: a ``str``
  field handed a dict/list is stringified then soft-truncated to the
  field's ``max_length``; a ``list[str]`` field handed a scalar, prose
  string, or non-str elements (the classic integer citation refs
  ``[1, 3]``) is coerced to ``list[str]``.
* :func:`validate_lenient` — validation ladder (≤6 passes): coerce the
  offending leaf in place (int→str, truncate over-long — content
  preserved), else drop the dict-field leaf so its schema default applies
  (also absorbs ``extra_forbidden``), then retry. Raises
  :class:`WireValidationError` only when it still cannot validate.
* :func:`unwrap_placeholder_envelope` — strips the two Databricks-Claude
  structured-output transport wrappers observed in practice
  (``$PARAMETER_VALUE`` placeholder nesting; tool-use XML leaked as a
  single JSON-shaped key).
* :func:`coerce_citation_ref` — ``"S23"`` / ``"src 23"`` / ``"#23"`` /
  ``23`` → ``"23"`` so hallucination guards and legends agree on what
  resolves.

Ported from the sapresalesbot project's proven synthesis stack
(``output_types/_tolerant.py``, ``output_types/_guards.py``,
``synthesis/wire_assembly.py``).
"""

from __future__ import annotations

import copy
import logging
import re
import types as _types
import typing as _typing
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

logger = logging.getLogger(__name__)


class WireValidationError(ValueError):
    """A wire payload failed validation even after lenient repair."""


# ---------------------------------------------------------------------------
# Loose-shape coercion helpers
# ---------------------------------------------------------------------------


def _stringify_structured(value: Any) -> Any:
    """dict/list -> readable string (content preserved); str/None pass through."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "; ".join(
            f"{k}: {_stringify_structured(v) if isinstance(v, dict | list) else v}"
            for k, v in value.items()
        )
    if isinstance(value, list):
        return "\n".join(
            _stringify_structured(i) if isinstance(i, dict | list) else str(i)
            for i in value
        )
    return str(value)


def _as_str_list(value: Any) -> Any:
    """scalar/prose str -> list[str]; list -> stringified items; None passes."""
    if isinstance(value, list):
        return [i if isinstance(i, str) else _stringify_structured(i) for i in value]
    if value is None:
        return value
    if isinstance(value, str):
        for sep in ("\n", " + ", ";"):
            if sep in value:
                return [p.strip() for p in value.split(sep) if p.strip()]
        return [value] if value.strip() else []
    return [str(value)]


def _coerce_str(value: Any, limit: int | None) -> Any:
    """Stringify dict/list then soft-truncate to ``limit`` (if any)."""
    s = _stringify_structured(value)
    if limit and isinstance(s, str) and len(s) > limit:
        return s[: limit - 1] + "…"
    return s


def _core_type(annotation: Any) -> Any:
    """Unwrap ``Optional`` / ``X | None`` to the single non-None member."""
    origin = _typing.get_origin(annotation)
    if origin is _typing.Union or origin is getattr(_types, "UnionType", None):
        non_none = [a for a in _typing.get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            return _core_type(non_none[0])
    return annotation


def _is_list_of_str(core: Any) -> bool:
    """True for ``list[str]`` (the citation-ref / prose-list shape)."""
    return _typing.get_origin(core) is list and _typing.get_args(core) == (str,)


def _field_max_length(field_info: Any) -> int | None:
    """Read ``max_length`` from a field's annotated-types metadata."""
    for meta in getattr(field_info, "metadata", ()) or ():
        ml = getattr(meta, "max_length", None)
        if isinstance(ml, int):
            return ml
    return None


class TolerantWireBase(BaseModel):
    """Base for wire models: absorbs the loose shapes verbose LLMs emit.

    Carries ``extra="forbid"`` for every subclass (``create_model`` cannot
    combine ``__config__`` with ``__base__``). The ``mode="before"``
    validator normalizes ``str`` and ``list[str]`` fields before per-field
    validation so the schema never hard-rejects dict-for-str, over-length
    prose, or integer citation refs. Idempotent, so it composes with any
    field-level validators, which run after it on the normalized value.
    """

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _coerce_loose_shapes(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        coerced = dict(data)
        for name, field_info in cls.model_fields.items():
            if name not in coerced:
                continue
            value = coerced[name]
            if value is None:
                continue
            core = _core_type(field_info.annotation)
            if core is str:
                coerced[name] = _coerce_str(value, _field_max_length(field_info))
            elif _is_list_of_str(core):
                coerced[name] = _as_str_list(value)
        return coerced


# ---------------------------------------------------------------------------
# Citation-ref canonicalization
# ---------------------------------------------------------------------------

# Citation refs may arrive prefixed (``S23``, ``src 7``, ``#5``); the
# canonical id is the bare integer. Strip a KNOWN prefix only — never invent
# digits, so a truly-unresolvable ref stays unresolvable to the guard.
_CITATION_REF_RE = re.compile(r"^(?:s|src|source)?\s*#?\s*(\d+)$", re.IGNORECASE)


def coerce_citation_ref(ref: Any) -> str:
    """Return a source ref's canonical bare-numeric id.

    ``"S23"`` / ``"src 23"`` / ``"#23"`` / ``23`` -> ``"23"``. Refs with no
    resolvable integer are returned stripped-but-unchanged so they remain
    correctly unresolvable.
    """
    text = str(ref).strip()
    match = _CITATION_REF_RE.match(text)
    return match.group(1) if match else text


# ---------------------------------------------------------------------------
# Lenient validation ladder
# ---------------------------------------------------------------------------

_MISSING = object()


def _walk_to_parent(container: Any, loc: tuple[Any, ...]) -> tuple[Any, Any]:
    """Navigate ``loc[:-1]`` through dicts AND lists; ``(parent, last_key)``.

    Returns ``(None, None)`` if the path is unreachable. List indices are
    followed, so leaves INSIDE a list (e.g. ``('rows', 0, 'source_refs',
    0)``) are reachable for coercion.
    """
    if not loc:
        return None, None
    parent = container
    for part in loc[:-1]:
        if isinstance(parent, dict):
            parent = parent.get(part)
        elif isinstance(parent, list) and isinstance(part, int) and 0 <= part < len(parent):
            parent = parent[part]
        else:
            return None, None
        if parent is None:
            return None, None
    return parent, loc[-1]


def _get_leaf(container: Any, loc: tuple[Any, ...]) -> Any:
    """The value at ``loc`` (dict key or list index), else ``_MISSING``."""
    parent, last = _walk_to_parent(container, loc)
    if isinstance(parent, dict) and last in parent:
        return parent[last]
    if isinstance(parent, list) and isinstance(last, int) and 0 <= last < len(parent):
        return parent[last]
    return _MISSING


def _set_leaf(container: Any, loc: tuple[Any, ...], value: Any) -> bool:
    """Set the value at ``loc`` (dict key or list index)."""
    parent, last = _walk_to_parent(container, loc)
    if isinstance(parent, dict):
        parent[last] = value
        return True
    if isinstance(parent, list) and isinstance(last, int) and 0 <= last < len(parent):
        parent[last] = value
        return True
    return False


def _drop_leaf(container: Any, loc: tuple[Any, ...]) -> bool:
    """Remove the dict-keyed leaf at ``loc`` so its schema default applies.

    Only object-field leaves are dropped; a list-element ``loc`` returns
    False (lists have no per-element default — those are coerced by
    :func:`_coerce_leaf`).
    """
    parent, last = _walk_to_parent(container, loc)
    if isinstance(parent, dict) and last in parent:
        del parent[last]
        return True
    return False


def _coerce_leaf(working: Any, err: Mapping[str, Any]) -> bool:
    """Coerce the offending leaf in place, PRESERVING content.

    - ``string_type`` with an int/float/bool value → ``str(value)`` (the
      integer citation-ref case, reachable inside lists);
    - ``string_too_long`` → truncate to the error's ``ctx.max_length``.
    """
    loc = tuple(err.get("loc") or ())
    if not loc:
        return False
    etype = err.get("type")
    if etype == "string_type":
        val = _get_leaf(working, loc)
        if isinstance(val, bool | int | float):
            return _set_leaf(working, loc, str(val))
        return False
    if etype == "string_too_long":
        val = _get_leaf(working, loc)
        if isinstance(val, str):
            max_len = (err.get("ctx") or {}).get("max_length")
            if isinstance(max_len, int) and max_len > 0:
                return _set_leaf(working, loc, val[: max_len - 1] + "…")
    return False


def validate_lenient(
    model_cls: type[BaseModel],
    payload: dict[str, Any],
    *,
    max_passes: int = 6,
) -> tuple[BaseModel, list[str]]:
    """Validate ``payload``; coerce-before-drop on errors, retry ≤ passes.

    On ``ValidationError`` first COERCE the offending leaf (int→str,
    truncate over-long — content preserved), else drop the dict-field leaf
    (→ its schema default; this also absorbs ``extra_forbidden`` keys), and
    retry. Returns ``(obj, dropped_paths)``; coerced paths are warn-logged.

    Raises :class:`WireValidationError` (clean, not a raw pydantic error)
    only if it still cannot validate — e.g. a required field with no
    default and no value.
    """
    working = copy.deepcopy(payload)
    dropped: list[str] = []
    coerced: list[str] = []
    last_exc: Exception | None = None
    for _ in range(max_passes):
        try:
            obj = model_cls.model_validate(working)
        except ValidationError as exc:
            last_exc = exc
            changed = False
            for err in exc.errors():
                loc_str = ".".join(str(p) for p in (err.get("loc") or ()))
                if _coerce_leaf(working, err):
                    coerced.append(loc_str)
                    changed = True
                elif _drop_leaf(working, tuple(err.get("loc") or ())):
                    dropped.append(loc_str)
                    changed = True
            if not changed:
                break
            continue
        if coerced:
            logger.warning(
                "WIRE_VALIDATION_COERCED_LEAF count=%d paths=%s",
                len(coerced),
                coerced[:20],
            )
        return obj, dropped
    raise WireValidationError(
        f"{getattr(model_cls, '__name__', 'wire')} failed validation after "
        f"coercing {len(coerced)} + dropping {len(dropped)} field(s) "
        f"({dropped[:15]}): {last_exc}"
    )


# ---------------------------------------------------------------------------
# Transport-envelope unwrapping
# ---------------------------------------------------------------------------


def json_repair_structured(text: str) -> dict[str, Any] | list[Any] | None:
    """Tolerant parse for fenced / lightly-malformed JSON strings.

    Returns a non-empty ``dict``/``list`` when ``json_repair`` recovers
    structured content, else ``None``. ``json_repair`` coerces genuine
    non-JSON to ``''``/scalars; those are rejected so the caller still
    surfaces a parse error for true garbage.
    """
    try:
        import json_repair
    except ImportError:  # pragma: no cover — json-repair is a declared dep
        return None
    try:
        repaired = json_repair.loads(text)
    except Exception:  # pragma: no cover — defensive; rarely raises
        return None
    if isinstance(repaired, dict | list) and repaired:
        return repaired
    return None


def unwrap_placeholder_envelope(cls: type[BaseModel], data: Any) -> Any:
    """Strip Databricks Claude structured-output transport wrappers.

    Two failure modes observed in practice:

    1. **Placeholder wrapper** — the proxy fails to substitute its template
       variable, leaving the real payload nested under a literal
       ``$PARAMETER_NAME`` / ``$PARAMETER_VALUE`` key.
       Shape: ``{"<placeholder>": {<real fields>}}``.
    2. **Tool-use XML leak** — the model emits the JSON inside an
       Anthropic-style tool-call wrapper; the OpenAI-compat proxy then
       encodes the whole blob as a single string *key* with an
       empty-string value. Shape: ``{"{...real json...": ""}``.

    Both shapes are unambiguous (single-key dict, key not in the schema,
    content recoverable as a dict whose keys overlap the schema). Anything
    else passes through untouched so genuine validation errors surface.
    """
    if not isinstance(data, dict) or len(data) != 1:
        return data
    outer_key = next(iter(data))
    inner = data[outer_key]
    fields = set(cls.model_fields.keys())

    # Mode 1 — placeholder wrapper around a clean dict payload.
    if isinstance(inner, dict):
        if outer_key in fields:
            return data
        if not (set(inner.keys()) & fields):
            return data
        return inner

    # Mode 2 — tool-use XML leak with empty value and JSON-shaped key.
    if isinstance(outer_key, str) and outer_key.lstrip().startswith("{") and inner in ("", None):
        recovered = json_repair_structured(outer_key)
        if isinstance(recovered, dict) and (set(recovered.keys()) & fields):
            return recovered

    return data


__all__ = [
    "TolerantWireBase",
    "WireValidationError",
    "coerce_citation_ref",
    "json_repair_structured",
    "unwrap_placeholder_envelope",
    "validate_lenient",
]
