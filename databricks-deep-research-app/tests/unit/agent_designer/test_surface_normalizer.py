"""Pillar 4 — write-path surface normalization (`normalize_surface_in_definition`).

Mirrors the frontend read-time normalizer (frontend/src/lib/surfaceSchema.ts): the save
gate (`schemas/agent_v2.py` `_enforce_surface_validation`) *validates* the surface but does
not *default* it, so a component persisted without a ``children`` key would reach the
renderer as ``undefined`` and crash on ``component.children.length`` (the `[App]` blank
screen). This normalizer fills the invariant at write time — additive, non-lossy,
idempotent, and tolerant of malformed input.
"""

from __future__ import annotations

import copy

from deep_research.agent_designer.ast_normalizer import (
    normalize_surface_in_definition,
)


def test_fills_missing_children_and_props() -> None:
    definition = {
        "surface": {
            "version": 1,
            "components": [
                {"id": "root", "component": "Column"},  # no children/props
                {"id": "f", "component": "TextField", "children": ["x"]},
            ],
            "data_model": {},
            "bindings": [],
        }
    }
    normalize_surface_in_definition(definition)
    comps = definition["surface"]["components"]
    assert comps[0]["children"] == []
    assert comps[0]["props"] == {}
    assert comps[1]["children"] == ["x"]  # preserved


def test_defaults_section_children() -> None:
    definition = {
        "surface": {
            "version": 1,
            "components": [],
            "layout": {"sections": [{"id": "s", "title": "S", "role": "results"}]},
        }
    }
    normalize_surface_in_definition(definition)
    assert definition["surface"]["layout"]["sections"][0]["children"] == []


def test_noop_without_surface() -> None:
    definition = {"root": {"children": []}}
    before = copy.deepcopy(definition)
    normalize_surface_in_definition(definition)
    assert definition == before


def test_idempotent_and_non_lossy() -> None:
    definition = {
        "surface": {
            "version": 1,
            "components": [{"id": "root", "component": "Column", "_x": 7}],
            "data_model": {"foo": "bar"},
            "bindings": [],
            "_future": "keep",
        }
    }
    normalize_surface_in_definition(definition)
    once = copy.deepcopy(definition)
    normalize_surface_in_definition(definition)
    assert definition == once  # idempotent
    assert definition["surface"]["_future"] == "keep"  # non-lossy top-level
    assert definition["surface"]["components"][0]["_x"] == 7  # non-lossy per-component


def test_tolerates_garbage_without_raising() -> None:
    for bad in [
        {"surface": "nope"},
        {"surface": {"version": 1, "components": "nope"}},
        {"surface": {"version": 1, "components": ["garbage", None, 5]}},
        {"surface": {"version": 1, "components": [], "layout": "nope"}},
    ]:
        normalize_surface_in_definition(bad)  # must not raise
