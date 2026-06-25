"""Unit tests for TableAggregateTool."""

from __future__ import annotations

import json
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
)
from databricks_deep_research.tools.protocol import ToolContext


def _binding() -> BindingInfo:
    return BindingInfo(
        name="sales",
        fqn="cat.sch.sales",
        source=BindingSource.BOUND,
        roles=RoleMap(id_column="id", content_column="text"),
        numeric_columns=("amount", "qty"),
    )


class FakeSchemaCache:
    def __init__(self, schema: Schema) -> None:
        self._schema = schema

    def get(self, fqn: str, user_token: str) -> Schema:
        return self._schema


def _schema() -> Schema:
    return Schema(
        fqn="cat.sch.sales",
        columns=(
            SchemaColumn(name="id", data_type="string", nullable=False),
            SchemaColumn(name="text", data_type="string"),
            SchemaColumn(name="region", data_type="string"),
            SchemaColumn(name="amount", data_type="double"),
            SchemaColumn(name="qty", data_type="int"),
        ),
    )


def _registry() -> TableBindingRegistry:
    r = TableBindingRegistry()
    r.register_bound(_binding())
    return r


def _ctx() -> ToolContext:
    return ToolContext(extras={"user_token": "tok"})


@pytest.mark.unit
def test_definition() -> None:
    tool = TableAggregateTool(
        registry=_registry(),
        schema_cache=FakeSchemaCache(_schema()),
        sql_executor=lambda *_args, **_kwargs: [],
    )
    d = tool.definition
    assert d.name == "table_aggregate"
    assert "binding" in d.parameters["properties"]
    assert d.parameters["properties"]["op"]["enum"] == [
        "count", "sum", "avg", "min", "max"
    ]


@pytest.mark.unit
def test_validate_rejects_unknown_op() -> None:
    tool = TableAggregateTool(
        registry=_registry(),
        schema_cache=FakeSchemaCache(_schema()),
        sql_executor=lambda *_args, **_kwargs: [],
    )
    with pytest.raises(ValueError):
        tool.validate_arguments({"binding": "sales", "op": "median"})


@pytest.mark.unit
def test_validate_requires_column_for_sum() -> None:
    tool = TableAggregateTool(
        registry=_registry(),
        schema_cache=FakeSchemaCache(_schema()),
        sql_executor=lambda *_args, **_kwargs: [],
    )
    with pytest.raises(ValueError):
        tool.validate_arguments({"binding": "sales", "op": "sum"})


@pytest.mark.unit
@pytest.mark.asyncio
async def test_count_no_column() -> None:
    captured: list[str] = []

    def exec_sql(sql: str, params: list[Any], token: str) -> list[dict[str, Any]]:
        captured.append(sql)
        return [{"__agg": 100}]

    tool = TableAggregateTool(
        registry=_registry(),
        schema_cache=FakeSchemaCache(_schema()),
        sql_executor=exec_sql,
    )
    args = tool.validate_arguments({"binding": "sales", "op": "count"})
    res = await tool.execute(args, _ctx())
    assert res.success is True
    payload = json.loads(res.content)
    assert payload["op"] == "count"
    assert payload["rows"] == [{"__agg": 100}]
    assert "COUNT(*)" in captured[0]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sum_with_group_by() -> None:
    captured: list[str] = []

    def exec_sql(sql: str, params: list[Any], token: str) -> list[dict[str, Any]]:
        captured.append(sql)
        return [
            {"region": "NA", "__agg": 1000},
            {"region": "EU", "__agg": 500},
        ]

    tool = TableAggregateTool(
        registry=_registry(),
        schema_cache=FakeSchemaCache(_schema()),
        sql_executor=exec_sql,
    )
    args = tool.validate_arguments(
        {
            "binding": "sales",
            "op": "sum",
            "column": "amount",
            "group_by": ["region"],
        }
    )
    res = await tool.execute(args, _ctx())
    assert res.success is True
    payload = json.loads(res.content)
    assert payload["group_by"] == ["region"]
    assert "GROUP BY" in captured[0]
    assert "SUM(`amount`)" in captured[0]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sum_non_numeric_column_rejected() -> None:
    tool = TableAggregateTool(
        registry=_registry(),
        schema_cache=FakeSchemaCache(_schema()),
        sql_executor=lambda *_args, **_kwargs: [],
    )
    args = tool.validate_arguments(
        {"binding": "sales", "op": "sum", "column": "region"}
    )
    res = await tool.execute(args, _ctx())
    assert res.success is False
    payload = json.loads(res.content)
    assert payload["error"]["error_code"] == "invalid_column"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unknown_column_rejected() -> None:
    tool = TableAggregateTool(
        registry=_registry(),
        schema_cache=FakeSchemaCache(_schema()),
        sql_executor=lambda *_args, **_kwargs: [],
    )
    args = tool.validate_arguments(
        {"binding": "sales", "op": "sum", "column": "ghost"}
    )
    res = await tool.execute(args, _ctx())
    assert res.success is False
    payload = json.loads(res.content)
    assert payload["error"]["error_code"] == "invalid_column"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_having_not_supported_returns_invalid_filter() -> None:
    tool = TableAggregateTool(
        registry=_registry(),
        schema_cache=FakeSchemaCache(_schema()),
        sql_executor=lambda *_args, **_kwargs: [],
    )
    args = tool.validate_arguments(
        {
            "binding": "sales",
            "op": "count",
            "having": {"gt": {"__agg": 1}},
        }
    )
    res = await tool.execute(args, _ctx())
    assert res.success is False
    payload = json.loads(res.content)
    assert payload["error"]["error_code"] == "invalid_filter"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_group_cardinality_exceeded() -> None:
    # Return PER_STMT_LIMIT_GROUPS+1 groups
    from databricks_deep_research.tools.builtins.text_table import (
        PER_STMT_LIMIT_GROUPS,
    )

    big = [
        {"region": f"R-{i}", "__agg": i}
        for i in range(PER_STMT_LIMIT_GROUPS + 1)
    ]

    tool = TableAggregateTool(
        registry=_registry(),
        schema_cache=FakeSchemaCache(_schema()),
        sql_executor=lambda *_args, **_kwargs: big,
    )
    args = tool.validate_arguments(
        {
            "binding": "sales",
            "op": "count",
            "group_by": ["region"],
            "limit": PER_STMT_LIMIT_GROUPS + 5,
        }
    )
    res = await tool.execute(args, _ctx())
    assert res.success is False
    payload = json.loads(res.content)
    assert payload["error"]["error_code"] == "group_cardinality_exceeded"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unknown_binding_returns_error() -> None:
    tool = TableAggregateTool(
        registry=TableBindingRegistry(),
        schema_cache=FakeSchemaCache(_schema()),
        sql_executor=lambda *_args, **_kwargs: [],
    )
    args = tool.validate_arguments({"binding": "ghost", "op": "count"})
    res = await tool.execute(args, _ctx())
    assert res.success is False
    payload = json.loads(res.content)
    assert payload["error"]["error_code"] == "invalid_binding"
