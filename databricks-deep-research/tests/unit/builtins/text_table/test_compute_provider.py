"""Unit tests for the ComputeCallableProvider integration on table_* tools.

Each tool exposes ``compute_name`` (str) and a
``to_compute_callable(*, compute) -> Callable`` factory. The callable bypasses
the JSON envelope: errors raise ``ToolErrorException`` directly, and successful
calls return native Python data. ``table_load`` returns ``Table`` object(s)
and mutates the compute namespace when available.
"""

from __future__ import annotations

from typing import Any

import pytest

from databricks_deep_research.tools.builtins.text_table import (
    BindingInfo,
    BindingSource,
    ComputeCallableProvider,
    RoleMap,
    Schema,
    SchemaColumn,
    Table,
    TableAggregateTool,
    TableBindingRegistry,
    TableDiscoveryTool,
    TableLoadTool,
    TableNeighborsTool,
    TableReadTool,
    TableSearchTool,
    ToolErrorException,
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


def _docs_schema() -> Schema:
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


def _registry(*infos: BindingInfo) -> TableBindingRegistry:
    r = TableBindingRegistry()
    for info in infos:
        if info.source is BindingSource.BOUND:
            r.register_bound(info)
        else:
            r.register_discovered(info)
    return r


# ---------------------------------------------------------------------------
# Protocol conformance — every tool must satisfy ComputeCallableProvider.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_all_tools_satisfy_compute_callable_provider() -> None:
    registry = _registry(_docs_binding())
    schema_cache = _FakeSchemaCache(_docs_schema())

    def _exec(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return []

    tools: list[ComputeCallableProvider] = [
        TableDiscoveryTool(provider=None, registry=registry),
        TableSearchTool(
            registry=registry,
            schema_cache=schema_cache,
            sql_executor=_exec,
        ),
        TableReadTool(
            registry=registry,
            schema_cache=schema_cache,
            sql_executor=_exec,
        ),
        TableNeighborsTool(
            registry=registry,
            schema_cache=schema_cache,
            sql_executor=_exec,
        ),
        TableLoadTool(
            registry=registry,
            schema_cache=schema_cache,
            sql_executor=_exec,
        ),
        TableAggregateTool(
            registry=registry,
            schema_cache=schema_cache,
            sql_executor=_exec,
        ),
    ]
    expected_names = {
        "table_discovery",
        "table_search",
        "table_read",
        "table_neighbors",
        "table_load",
        "table_aggregate",
    }
    actual_names = set()
    for tool in tools:
        assert isinstance(tool, ComputeCallableProvider)
        assert isinstance(tool.compute_name, str)
        assert tool.compute_name
        actual_names.add(tool.compute_name)
        callable_obj = tool.to_compute_callable(compute=None)
        assert callable(callable_obj)
    assert actual_names == expected_names


# ---------------------------------------------------------------------------
# table_search compute callable
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_search_callable_returns_native_list_and_filters_substring() -> None:
    registry = _registry(_docs_binding())
    schema_cache = _FakeSchemaCache(_docs_schema())
    rows = [
        {"id": "1", "text": "alpha bravo", "doc": "A", "seq": 1},
        {"id": "2", "text": "charlie delta", "doc": "A", "seq": 2},
        {"id": "3", "text": "echo bravo foxtrot", "doc": "B", "seq": 3},
    ]

    def _exec(_sql: str, _params: Any, _tok: str) -> list[dict[str, Any]]:
        return rows

    tool = TableSearchTool(
        registry=registry, schema_cache=schema_cache, sql_executor=_exec
    )
    call = tool.to_compute_callable(compute=None)
    out = call(binding="docs", query="bravo", user_token="t", limit=10)
    assert isinstance(out, list)
    assert {r["id"] for r in out} == {"1", "3"}
    assert all("snippet" in r and "score" in r for r in out)


@pytest.mark.unit
def test_search_callable_raises_on_invalid_binding() -> None:
    registry = _registry(_docs_binding())
    schema_cache = _FakeSchemaCache(_docs_schema())
    tool = TableSearchTool(
        registry=registry,
        schema_cache=schema_cache,
        sql_executor=lambda *_args, **_kwargs: [],
    )
    call = tool.to_compute_callable(compute=None)
    with pytest.raises(ToolErrorException):
        call(binding="nonexistent", query="x", user_token="t")


# ---------------------------------------------------------------------------
# table_read compute callable
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_read_callable_returns_rows() -> None:
    registry = _registry(_docs_binding())
    schema_cache = _FakeSchemaCache(_docs_schema())
    rows = [{"id": "1", "text": "x"}, {"id": "2", "text": "y"}]

    def _exec(_sql: str, _params: Any, _tok: str) -> list[dict[str, Any]]:
        return rows

    tool = TableReadTool(
        registry=registry, schema_cache=schema_cache, sql_executor=_exec
    )
    call = tool.to_compute_callable(compute=None)
    out = call(binding="docs", user_token="t", columns=["id", "text"], limit=5)
    assert out == rows


# ---------------------------------------------------------------------------
# table_neighbors compute callable
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_neighbors_callable_window_arithmetic() -> None:
    registry = _registry(_docs_binding())
    schema_cache = _FakeSchemaCache(_docs_schema())
    anchor_row = {"id": "5", "doc": "A", "seq": 5}
    sibling_rows = [
        {"id": "4", "doc": "A", "seq": 4, "text": "before"},
        {"id": "5", "doc": "A", "seq": 5, "text": "anchor"},
        {"id": "6", "doc": "A", "seq": 6, "text": "after"},
    ]
    call_count = {"n": 0}

    def _exec(_sql: str, _params: Any, _tok: str) -> list[dict[str, Any]]:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return [anchor_row]
        return sibling_rows

    tool = TableNeighborsTool(
        registry=registry, schema_cache=schema_cache, sql_executor=_exec
    )
    call = tool.to_compute_callable(compute=None)
    out = call(binding="docs", id="5", user_token="t", before=1, after=1)
    assert call_count["n"] == 2
    assert out == sibling_rows


@pytest.mark.unit
def test_neighbors_callable_raises_when_anchor_missing() -> None:
    registry = _registry(_docs_binding())
    schema_cache = _FakeSchemaCache(_docs_schema())
    tool = TableNeighborsTool(
        registry=registry,
        schema_cache=schema_cache,
        sql_executor=lambda *_args, **_kwargs: [],
    )
    call = tool.to_compute_callable(compute=None)
    with pytest.raises(ToolErrorException):
        call(binding="docs", id="999", user_token="t")


# ---------------------------------------------------------------------------
# table_load compute callable mutates compute namespace via inject_variable
# ---------------------------------------------------------------------------


class _StubCompute:
    def __init__(self) -> None:
        self.injected: dict[str, Any] = {}

    def inject_variable(self, name: str, value: Any) -> None:
        self.injected[name] = value

    def get_variable(self, name: str, default: Any = None) -> Any:
        return self.injected.get(name, default)


@pytest.mark.unit
def test_load_callable_mutates_compute_namespace() -> None:
    registry = _registry(_docs_binding())
    schema_cache = _FakeSchemaCache(_docs_schema())
    rows = [{"id": "1", "text": "hello"}]

    def _exec(_sql: str, _params: Any, _tok: str) -> list[dict[str, Any]]:
        return rows

    tool = TableLoadTool(
        registry=registry, schema_cache=schema_cache, sql_executor=_exec
    )
    compute = _StubCompute()
    call = tool.to_compute_callable(compute=compute)
    out = call(binding="docs", id="1", user_token="t", as_var="my_table")
    assert isinstance(out, Table)
    assert out.row_count == 1
    # As-var slot + last_table + tables list.
    assert "my_table" in compute.injected
    assert "last_table" in compute.injected
    assert "tables" in compute.injected
    assert isinstance(compute.injected["tables"], list)
    assert len(compute.injected["tables"]) == 1


@pytest.mark.unit
def test_load_callable_appends_tables_namespace() -> None:
    registry = _registry(_docs_binding())
    schema_cache = _FakeSchemaCache(_docs_schema())
    rows = [{"id": "1", "text": "hello"}]

    tool = TableLoadTool(
        registry=registry,
        schema_cache=schema_cache,
        sql_executor=lambda *_args: rows,
    )
    compute = _StubCompute()
    call = tool.to_compute_callable(compute=compute)

    call(binding="docs", id="1", user_token="t")
    call(binding="docs", id="1", user_token="t")

    assert len(compute.injected["tables"]) == 2
    assert compute.injected["last_table"] is compute.injected["tables"][-1]


@pytest.mark.unit
def test_load_callable_without_compute_returns_table_object() -> None:
    registry = _registry(_docs_binding())
    schema_cache = _FakeSchemaCache(_docs_schema())
    rows = [{"id": "1", "text": "x"}]
    tool = TableLoadTool(
        registry=registry,
        schema_cache=schema_cache,
        sql_executor=lambda *_args, **_kwargs: rows,
    )
    call = tool.to_compute_callable(compute=None)
    out = call(binding="docs", id="1", user_token="t")
    assert isinstance(out, Table)
    assert out.row_count == 1


# ---------------------------------------------------------------------------
# table_aggregate compute callable
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_aggregate_callable_count() -> None:
    registry = _registry(_docs_binding())
    schema_cache = _FakeSchemaCache(_docs_schema())
    rows = [{"__agg": 7}]

    def _exec(_sql: str, _params: Any, _tok: str) -> list[dict[str, Any]]:
        return rows

    tool = TableAggregateTool(
        registry=registry, schema_cache=schema_cache, sql_executor=_exec
    )
    call = tool.to_compute_callable(compute=None)
    out = call(binding="docs", op="count", user_token="t")
    assert out == rows


@pytest.mark.unit
def test_aggregate_callable_rejects_non_numeric_sum_column() -> None:
    registry = _registry(_docs_binding())
    schema_cache = _FakeSchemaCache(_docs_schema())
    tool = TableAggregateTool(
        registry=registry,
        schema_cache=schema_cache,
        sql_executor=lambda *_args, **_kwargs: [],
    )
    call = tool.to_compute_callable(compute=None)
    # 'text' is not in numeric_columns of the binding
    with pytest.raises(ToolErrorException):
        call(binding="docs", op="sum", column="text", user_token="t")


# ---------------------------------------------------------------------------
# table_discovery compute callable — only error path (provider=None)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_discovery_callable_raises_when_provider_unset() -> None:
    tool = TableDiscoveryTool(provider=None, registry=_registry())
    call = tool.to_compute_callable(compute=None)
    with pytest.raises(ToolErrorException):
        call(user_token="t")
