"""``ToolContext.extras`` field — frozen dataclass with mutable extras dict."""

from __future__ import annotations

import dataclasses

import pytest

from databricks_deep_research.tools.protocol import ToolContext


def test_extras_default_is_empty_dict() -> None:
    ctx = ToolContext()
    assert ctx.extras == {}


def test_extras_dict_is_mutable() -> None:
    ctx = ToolContext()
    ctx.extras["_framework_thread_id"] = "t1"
    ctx.extras["custom_key"] = 42
    assert ctx.extras["_framework_thread_id"] == "t1"
    assert ctx.extras["custom_key"] == 42


def test_extras_reference_is_frozen() -> None:
    ctx = ToolContext()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.extras = {}  # type: ignore[misc]


def test_extras_independent_per_instance() -> None:
    a = ToolContext()
    b = ToolContext()
    a.extras["k"] = 1
    assert "k" not in b.extras


def test_extras_initial_value_via_constructor() -> None:
    ctx = ToolContext(extras={"_framework_vfs": "vfs_obj"})
    assert ctx.extras["_framework_vfs"] == "vfs_obj"


def test_other_fields_unchanged() -> None:
    ctx = ToolContext(query="hello")
    assert ctx.query == "hello"
    assert ctx.url_registry is None
    assert ctx.recent_observations == []
