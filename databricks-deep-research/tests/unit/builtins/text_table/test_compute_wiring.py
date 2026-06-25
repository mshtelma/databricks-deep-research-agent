"""Unit tests for ``inject_table_callables``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from databricks_deep_research.tools.builtins.text_table import (
    BindingInfo,
    BindingSource,
    RoleMap,
    Schema,
    SchemaColumn,
    TableAggregateTool,
    TableBindingRegistry,
    TableDiscoveryTool,
    TableLoadTool,
    TableNeighborsTool,
    TableReadTool,
    TableSearchTool,
    inject_table_callables,
)


def _docs_binding() -> BindingInfo:
    return BindingInfo(
        name="docs",
        fqn="cat.sch.docs",
        source=BindingSource.BOUND,
        roles=RoleMap(
            id_column="id",
            content_column="text",
            partition_column="doc",
            order_column="seq",
        ),
        numeric_columns=("seq",),
    )


def _schema() -> Schema:
    return Schema(
        fqn="cat.sch.docs",
        columns=(
            SchemaColumn(name="id", data_type="string", nullable=False),
            SchemaColumn(name="text", data_type="string"),
            SchemaColumn(name="doc", data_type="string"),
            SchemaColumn(name="seq", data_type="bigint"),
        ),
    )


class _FakeSchemaCache:
    def __init__(self, schema: Schema) -> None:
        self._schema = schema

    def get(self, fqn: str, user_token: str) -> Schema:
        return self._schema


class _StubCompute:
    """Mimics the public surface of PythonComputeTool that wiring uses."""

    def __init__(self) -> None:
        self.namespace: dict[str, Any] = {}

    def inject_variable(self, name: str, value: Any) -> None:
        self.namespace[name] = value


def _make_all_tools(
    registry: TableBindingRegistry, schema_cache: _FakeSchemaCache
) -> list[Any]:
    def _exec(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return []

    return [
        TableDiscoveryTool(provider=None, registry=registry),
        TableSearchTool(
            registry=registry, schema_cache=schema_cache, sql_executor=_exec
        ),
        TableReadTool(
            registry=registry, schema_cache=schema_cache, sql_executor=_exec
        ),
        TableNeighborsTool(
            registry=registry, schema_cache=schema_cache, sql_executor=_exec
        ),
        TableLoadTool(
            registry=registry, schema_cache=schema_cache, sql_executor=_exec
        ),
        TableAggregateTool(
            registry=registry, schema_cache=schema_cache, sql_executor=_exec
        ),
    ]


@pytest.mark.unit
def test_inject_registers_all_six_callables() -> None:
    registry = TableBindingRegistry()
    registry.register_bound(_docs_binding())
    schema_cache = _FakeSchemaCache(_schema())
    compute = _StubCompute()
    tools = _make_all_tools(registry, schema_cache)

    injected = inject_table_callables(
        compute=compute, providers=tools, registry=registry
    )

    assert set(injected) == {
        "table_discovery",
        "table_search",
        "table_read",
        "table_neighbors",
        "table_load",
        "table_aggregate",
    }
    for name in injected:
        assert callable(compute.namespace[name])


@pytest.mark.unit
def test_inject_exposes_bindings_snapshot_and_live_view() -> None:
    registry = TableBindingRegistry()
    registry.register_bound(_docs_binding())
    schema_cache = _FakeSchemaCache(_schema())
    compute = _StubCompute()
    tools = _make_all_tools(registry, schema_cache)

    inject_table_callables(
        compute=compute, providers=tools, registry=registry
    )

    snapshot = compute.namespace["bindings"]
    live = compute.namespace["bindings_live"]
    assert "docs" in snapshot
    assert "docs" in live

    # Mutate the registry post-injection. The snapshot does NOT reflect this;
    # the live view does.
    registry.register_discovered(
        BindingInfo(
            name="late",
            fqn="cat.sch.late",
            source=BindingSource.DISCOVERED,
        )
    )
    assert "late" not in snapshot
    assert "late" in live


@pytest.mark.unit
def test_inject_exposes_vector_indexes_snapshot_when_provided() -> None:
    registry = TableBindingRegistry()
    registry.register_bound(_docs_binding())
    schema_cache = _FakeSchemaCache(_schema())
    compute = _StubCompute()
    tools = _make_all_tools(registry, schema_cache)
    indexes = {"docs_vs": {"index_name": "cat.sch.docs_vs"}}

    inject_table_callables(
        compute=compute,
        providers=tools,
        registry=registry,
        vector_indexes=indexes,
    )

    assert compute.namespace["vector_indexes"] == indexes
    indexes["late"] = {"index_name": "cat.sch.late_vs"}
    assert "late" not in compute.namespace["vector_indexes"]


@pytest.mark.unit
def test_inject_skips_bindings_when_disabled() -> None:
    registry = TableBindingRegistry()
    registry.register_bound(_docs_binding())
    schema_cache = _FakeSchemaCache(_schema())
    compute = _StubCompute()
    tools = _make_all_tools(registry, schema_cache)

    inject_table_callables(
        compute=compute,
        providers=tools,
        registry=registry,
        expose_bindings=False,
    )

    assert "bindings" not in compute.namespace
    assert "bindings_live" not in compute.namespace


@pytest.mark.unit
def test_inject_skips_bindings_when_registry_omitted() -> None:
    registry = TableBindingRegistry()
    registry.register_bound(_docs_binding())
    schema_cache = _FakeSchemaCache(_schema())
    compute = _StubCompute()
    tools = _make_all_tools(registry, schema_cache)

    inject_table_callables(compute=compute, providers=tools)

    assert "bindings" not in compute.namespace
    assert "bindings_live" not in compute.namespace


@pytest.mark.unit
def test_inject_rejects_compute_without_inject_variable() -> None:
    class _Bad:
        pass

    with pytest.raises(TypeError):
        inject_table_callables(compute=_Bad(), providers=[])


@pytest.mark.unit
def test_inject_is_idempotent() -> None:
    registry = TableBindingRegistry()
    registry.register_bound(_docs_binding())
    schema_cache = _FakeSchemaCache(_schema())
    compute = _StubCompute()
    tools = _make_all_tools(registry, schema_cache)

    inject_table_callables(
        compute=compute, providers=tools, registry=registry
    )
    first_search: Callable[..., Any] = compute.namespace["table_search"]
    inject_table_callables(
        compute=compute, providers=tools, registry=registry
    )
    second_search: Callable[..., Any] = compute.namespace["table_search"]
    # The wiring rebuilds the callable on each invocation; both must be
    # callable and registered under the same name.
    assert callable(first_search)
    assert callable(second_search)
    # And every expected callable is still present.
    for name in (
        "table_discovery",
        "table_search",
        "table_read",
        "table_neighbors",
        "table_load",
        "table_aggregate",
    ):
        assert callable(compute.namespace[name])
