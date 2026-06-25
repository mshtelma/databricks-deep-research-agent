"""Unit tests for TableReadTool."""

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
    TableBindingRegistry,
    TableReadTool,
)
from databricks_deep_research.tools.protocol import ToolContext


def _binding() -> BindingInfo:
    return BindingInfo(
        name="docs",
        fqn="cat.sch.docs",
        source=BindingSource.BOUND,
        roles=RoleMap(id_column="id", content_column="text"),
    )


class FakeSchemaCache:
    def __init__(self, schema: Schema) -> None:
        self._schema = schema

    def get(self, fqn: str, user_token: str) -> Schema:
        return self._schema


def _schema() -> Schema:
    return Schema(
        fqn="cat.sch.docs",
        columns=(
            SchemaColumn(name="id", data_type="string", nullable=False),
            SchemaColumn(name="text", data_type="string"),
            SchemaColumn(name="kind", data_type="string"),
            SchemaColumn(name="rank", data_type="int"),
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
    tool = TableReadTool(
        registry=_registry(),
        schema_cache=FakeSchemaCache(_schema()),
        sql_executor=lambda *_args, **_kwargs: [],
    )
    d = tool.definition
    assert d.name == "table_read"
    assert d.parameters["required"] == ["binding"]


@pytest.mark.unit
def test_validate_rejects_bad_columns() -> None:
    tool = TableReadTool(
        registry=_registry(),
        schema_cache=FakeSchemaCache(_schema()),
        sql_executor=lambda *_args, **_kwargs: [],
    )
    with pytest.raises(ValueError):
        tool.validate_arguments({"binding": "docs", "columns": [1, 2]})


@pytest.mark.unit
def test_validate_rejects_negative_offset() -> None:
    tool = TableReadTool(
        registry=_registry(),
        schema_cache=FakeSchemaCache(_schema()),
        sql_executor=lambda *_args, **_kwargs: [],
    )
    with pytest.raises(ValueError):
        tool.validate_arguments({"binding": "docs", "offset": -1})


@pytest.mark.unit
def test_validate_clamps_limit_to_per_stmt_max() -> None:
    tool = TableReadTool(
        registry=_registry(),
        schema_cache=FakeSchemaCache(_schema()),
        sql_executor=lambda *_args, **_kwargs: [],
    )
    out = tool.validate_arguments({"binding": "docs", "limit": 10**6})
    assert out["limit"] <= 5000


@pytest.mark.unit
@pytest.mark.asyncio
async def test_read_returns_rows() -> None:
    rows = [{"id": "1", "text": "foo"}, {"id": "2", "text": "bar"}]
    tool = TableReadTool(
        registry=_registry(),
        schema_cache=FakeSchemaCache(_schema()),
        sql_executor=lambda *_args, **_kwargs: rows,
    )
    args = tool.validate_arguments({"binding": "docs"})
    res = await tool.execute(args, _ctx())
    assert res.success is True
    payload = json.loads(res.content)
    assert payload["count"] == 2
    assert payload["rows"] == rows


@pytest.mark.unit
@pytest.mark.asyncio
async def test_read_invalid_column_returns_error() -> None:
    tool = TableReadTool(
        registry=_registry(),
        schema_cache=FakeSchemaCache(_schema()),
        sql_executor=lambda *_args, **_kwargs: [],
    )
    args = tool.validate_arguments(
        {"binding": "docs", "columns": ["not_real"]}
    )
    res = await tool.execute(args, _ctx())
    assert res.success is False
    payload = json.loads(res.content)
    assert payload["error"]["error_code"] == "invalid_column"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_read_unknown_binding_returns_error() -> None:
    tool = TableReadTool(
        registry=_registry(),
        schema_cache=FakeSchemaCache(_schema()),
        sql_executor=lambda *_args, **_kwargs: [],
    )
    args = tool.validate_arguments({"binding": "ghost"})
    res = await tool.execute(args, _ctx())
    assert res.success is False
    payload = json.loads(res.content)
    assert payload["error"]["error_code"] == "invalid_binding"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_read_passes_where_filter_through_sql() -> None:
    captured_sql: list[str] = []

    def exec_sql(sql: str, params: list[Any], token: str) -> list[dict[str, Any]]:
        captured_sql.append(sql)
        return [{"id": "1", "text": "match"}]

    tool = TableReadTool(
        registry=_registry(),
        schema_cache=FakeSchemaCache(_schema()),
        sql_executor=exec_sql,
    )
    args = tool.validate_arguments(
        {"binding": "docs", "where": {"eq": {"id": "1"}}}
    )
    res = await tool.execute(args, _ctx())
    assert res.success is True
    assert len(captured_sql) == 1
    assert "WHERE" in captured_sql[0]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_read_with_order_by_and_columns() -> None:
    rows = [{"id": "1", "text": "a"}]
    tool = TableReadTool(
        registry=_registry(),
        schema_cache=FakeSchemaCache(_schema()),
        sql_executor=lambda *_args, **_kwargs: rows,
    )
    args = tool.validate_arguments(
        {
            "binding": "docs",
            "columns": ["id", "text"],
            "order_by": ["-rank"],
            "limit": 10,
        }
    )
    res = await tool.execute(args, _ctx())
    assert res.success is True
    payload = json.loads(res.content)
    assert payload["count"] == 1
