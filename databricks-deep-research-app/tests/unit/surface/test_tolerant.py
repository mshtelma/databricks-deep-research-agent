"""Tests for the tolerant wire-validation stack (surface/tolerant.py).

Pins the sapresalesbot-ported behaviors: TolerantWireBase loose-shape
coercion, the validate_lenient coerce-before-drop ladder, transport-envelope
unwrapping, and citation-ref canonicalization.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from deep_research.surface.tolerant import (
    TolerantWireBase,
    WireValidationError,
    coerce_citation_ref,
    unwrap_placeholder_envelope,
    validate_lenient,
)

pytestmark = pytest.mark.unit


class _Item(TolerantWireBase):
    name: str = Field(max_length=20)
    refs: list[str] = Field(default_factory=list)
    note: str | None = None


# ---------------------------------------------------------------------------
# TolerantWireBase coercions
# ---------------------------------------------------------------------------


def test_tolerant_stringifies_dict_for_str_field() -> None:
    obj = _Item.model_validate({"name": {"a": 1, "b": 2}})
    assert obj.name == "a: 1; b: 2"


def test_tolerant_stringifies_list_for_str_field() -> None:
    obj = _Item.model_validate({"name": ["x", "y"]})
    assert obj.name == "x\ny"


def test_tolerant_soft_truncates_to_max_length() -> None:
    obj = _Item.model_validate({"name": "x" * 50})
    assert len(obj.name) == 20
    assert obj.name.endswith("…")


def test_tolerant_coerces_integer_refs() -> None:
    obj = _Item.model_validate({"name": "a", "refs": [1, 3]})
    assert obj.refs == ["1", "3"]


def test_tolerant_splits_prose_into_list() -> None:
    obj = _Item.model_validate({"name": "a", "refs": "one; two; three"})
    assert obj.refs == ["one", "two", "three"]


def test_tolerant_wraps_scalar_string_in_list() -> None:
    obj = _Item.model_validate({"name": "a", "refs": "solo"})
    assert obj.refs == ["solo"]


def test_tolerant_handles_optional_str_field() -> None:
    obj = _Item.model_validate({"name": "a", "note": {"k": "v"}})
    assert obj.note == "k: v"


def test_tolerant_base_forbids_extra_keys() -> None:
    with pytest.raises(ValidationError):
        _Item.model_validate({"name": "a", "unexpected": 1})


def test_tolerant_passes_non_dict_through() -> None:
    with pytest.raises(ValidationError):
        _Item.model_validate("not a dict")


# ---------------------------------------------------------------------------
# coerce_citation_ref
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("S23", "23"),
        ("src 23", "23"),
        ("source 7", "7"),
        ("#23", "23"),
        (23, "23"),
        (" 12 ", "12"),
        ("3", "3"),
        ("gartner.com", "gartner.com"),  # unresolvable stays unresolvable
        ("S23b", "S23b"),  # trailing garbage — no digit invention
    ],
)
def test_coerce_citation_ref(raw: Any, expected: str) -> None:
    assert coerce_citation_ref(raw) == expected


# ---------------------------------------------------------------------------
# validate_lenient ladder (exercised on a STRICT model, no tolerant base)
# ---------------------------------------------------------------------------


class _StrictRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(max_length=10)
    refs: list[str] = Field(default_factory=list)
    optional_note: str | None = None


class _StrictWire(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[_StrictRow] = Field(default_factory=list)


def test_validate_lenient_coerces_int_inside_list() -> None:
    obj, dropped = validate_lenient(
        _StrictWire, {"rows": [{"label": "a", "refs": [127, 143]}]}
    )
    assert dropped == []
    assert obj.rows[0].refs == ["127", "143"]  # type: ignore[attr-defined]


def test_validate_lenient_truncates_over_long_string() -> None:
    obj, _ = validate_lenient(
        _StrictWire, {"rows": [{"label": "x" * 40, "refs": []}]}
    )
    assert len(obj.rows[0].label) == 10  # type: ignore[attr-defined]


def test_validate_lenient_drops_extra_forbidden_key() -> None:
    obj, dropped = validate_lenient(
        _StrictWire, {"rows": [{"label": "a", "bogus": "x"}]}
    )
    assert "rows.0.bogus" in dropped
    assert obj.rows[0].label == "a"  # type: ignore[attr-defined]


def test_validate_lenient_drops_bad_optional_field() -> None:
    # A dict where a str|None is expected and coercion doesn't apply
    # (bool/int/float only) — the leaf is dropped to its default.
    obj, dropped = validate_lenient(
        _StrictWire, {"rows": [{"label": "a", "optional_note": {"deep": {"x": 1}}}]}
    )
    assert dropped == ["rows.0.optional_note"]
    assert obj.rows[0].optional_note is None  # type: ignore[attr-defined]


def test_validate_lenient_raises_clean_error_when_unrepairable() -> None:
    # 'label' is required with no default: missing → no coercion, no drop.
    with pytest.raises(WireValidationError):
        validate_lenient(_StrictWire, {"rows": [{"refs": ["1"]}]})


# ---------------------------------------------------------------------------
# unwrap_placeholder_envelope
# ---------------------------------------------------------------------------


def test_unwrap_placeholder_wrapper() -> None:
    payload = {"rows": [{"label": "a"}]}
    assert unwrap_placeholder_envelope(
        _StrictWire, {"$PARAMETER_VALUE": payload}
    ) == payload


def test_unwrap_leaves_schema_keyed_dict_alone() -> None:
    data = {"rows": [{"label": "a"}]}
    assert unwrap_placeholder_envelope(_StrictWire, data) == data


def test_unwrap_leaves_non_overlapping_wrapper_alone() -> None:
    data = {"outer": {"unrelated": 1}}
    assert unwrap_placeholder_envelope(_StrictWire, data) == data


def test_unwrap_recovers_tool_xml_leak() -> None:
    key = '{"rows": [{"label": "a", "refs": ["1"]}]}'
    recovered = unwrap_placeholder_envelope(_StrictWire, {key: ""})
    assert isinstance(recovered, dict)
    assert recovered["rows"][0]["label"] == "a"


@pytest.mark.parametrize(
    "data",
    [
        "not a dict",
        {"a": 1, "b": 2},  # multi-key
        [],
    ],
)
def test_unwrap_passes_other_shapes_through(data: Any) -> None:
    assert unwrap_placeholder_envelope(_StrictWire, data) == data
