"""Tests for TableBindingRegistry."""

from __future__ import annotations

import pytest

from databricks_deep_research.tools.builtins.text_table.binding import (
    BindingInfo,
    BindingSource,
    RoleMap,
)
from databricks_deep_research.tools.builtins.text_table.error_codes import (
    ErrorCode,
    ToolErrorException,
)
from databricks_deep_research.tools.builtins.text_table.registry import (
    TableBindingRegistry,
)


def _bound(name: str, fqn: str | None = None) -> BindingInfo:
    return BindingInfo(
        name=name,
        fqn=fqn or f"c.s.{name}",
        source=BindingSource.BOUND,
        roles=RoleMap(id_column="id", content_column="content"),
    )


def _discovered(name: str, fqn: str | None = None) -> BindingInfo:
    return BindingInfo(
        name=name,
        fqn=fqn or f"d.s.{name}",
        source=BindingSource.DISCOVERED,
    )


def test_register_bound_and_lookup() -> None:
    reg = TableBindingRegistry()
    b = _bound("treasury")
    reg.register_bound(b)
    assert reg.get("treasury") is b


def test_register_bound_with_wrong_source_raises_value_error() -> None:
    reg = TableBindingRegistry()
    with pytest.raises(ValueError):
        reg.register_bound(_discovered("treasury"))


def test_register_discovered_with_wrong_source_raises_value_error() -> None:
    reg = TableBindingRegistry()
    with pytest.raises(ValueError):
        reg.register_discovered(_bound("treasury"))


def test_duplicate_bound_name_raises_invalid_binding() -> None:
    reg = TableBindingRegistry()
    reg.register_bound(_bound("treasury"))
    with pytest.raises(ToolErrorException) as exc:
        reg.register_bound(_bound("treasury"))
    assert exc.value.error.error_code is ErrorCode.INVALID_BINDING


def test_register_discovered_no_collision() -> None:
    reg = TableBindingRegistry()
    info = _discovered("foo")
    name, warning = reg.register_discovered(info)
    assert name == "foo"
    assert warning is None
    assert reg.get("foo") is info


def test_register_discovered_collision_with_bound_namespaced() -> None:
    reg = TableBindingRegistry()
    bound = _bound("treasury")
    reg.register_bound(bound)
    discovered = _discovered("treasury")
    name, warning = reg.register_discovered(discovered)
    assert name == "discovered.treasury"
    assert warning is not None
    assert warning.error_code is ErrorCode.DUPLICATE_BINDING
    # Original BOUND is preserved.
    assert reg.get("treasury") is bound
    # DISCOVERED stored under namespaced key.
    assert reg.get("discovered.treasury") is discovered


def test_get_unknown_raises_invalid_binding() -> None:
    reg = TableBindingRegistry()
    with pytest.raises(ToolErrorException) as exc:
        reg.get("missing")
    assert exc.value.error.error_code is ErrorCode.INVALID_BINDING


def test_contains_and_iter() -> None:
    reg = TableBindingRegistry()
    reg.register_bound(_bound("a"))
    reg.register_bound(_bound("b"))
    assert "a" in reg
    assert "b" in reg
    assert "c" not in reg
    assert sorted(iter(reg)) == ["a", "b"]
    assert sorted(reg.names()) == ["a", "b"]


def test_metadata_snapshot_is_frozen() -> None:
    reg = TableBindingRegistry()
    reg.register_bound(_bound("a"))
    snap = reg.metadata_snapshot()
    reg.register_bound(_bound("b"))
    # Snapshot does not reflect later mutations.
    assert "a" in snap
    assert "b" not in snap


def test_metadata_view_reflects_post_capture_mutations() -> None:
    reg = TableBindingRegistry()
    reg.register_bound(_bound("a"))
    view = reg.metadata_view()
    assert "a" in view
    assert "b" not in view
    reg.register_bound(_bound("b"))
    # Live view sees the new entry without re-fetching.
    assert "b" in view
