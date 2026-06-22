"""Unit tests for TableNeighborsTool."""

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
    TableNeighborsTool,
)
from databricks_deep_research.tools.protocol import ToolContext


def _binding_full_roles() -> BindingInfo:
    return BindingInfo(
        name="chunks",
        fqn="cat.sch.chunks",
        source=BindingSource.BOUND,
        roles=RoleMap(
            id_column="id",
            content_column="text",
            order_column="ord",
            partition_column="doc_id",
        ),
    )


def _binding_partial_roles() -> BindingInfo:
    # Missing partition_column / order_column.
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
        fqn="cat.sch.chunks",
        columns=(
            SchemaColumn(name="id", data_type="string", nullable=False),
            SchemaColumn(name="text", data_type="string"),
            SchemaColumn(name="ord", data_type="int"),
            SchemaColumn(name="doc_id", data_type="string"),
        ),
    )


def _registry(info: BindingInfo) -> TableBindingRegistry:
    r = TableBindingRegistry()
    r.register_bound(info)
    return r


def _ctx() -> ToolContext:
    return ToolContext(extras={"user_token": "tok"})


@pytest.mark.unit
def test_definition() -> None:
    tool = TableNeighborsTool(
        registry=_registry(_binding_full_roles()),
        schema_cache=FakeSchemaCache(_schema()),
        sql_executor=lambda *_args, **_kwargs: [],
    )
    d = tool.definition
    assert d.name == "table_neighbors"
    assert d.parameters["required"] == ["binding", "id"]


@pytest.mark.unit
def test_validate_rejects_negative_before() -> None:
    tool = TableNeighborsTool(
        registry=_registry(_binding_full_roles()),
        schema_cache=FakeSchemaCache(_schema()),
        sql_executor=lambda *_args, **_kwargs: [],
    )
    with pytest.raises(ValueError):
        tool.validate_arguments({"binding": "chunks", "id": "1", "before": -1})


@pytest.mark.unit
def test_validate_rejects_missing_id() -> None:
    tool = TableNeighborsTool(
        registry=_registry(_binding_full_roles()),
        schema_cache=FakeSchemaCache(_schema()),
        sql_executor=lambda *_args, **_kwargs: [],
    )
    with pytest.raises(ValueError):
        tool.validate_arguments({"binding": "chunks"})


@pytest.mark.unit
@pytest.mark.asyncio
async def test_neighbors_happy_path_two_step_fetch() -> None:
    calls: list[str] = []

    def exec_sql(sql: str, params: list[Any], token: str) -> list[dict[str, Any]]:
        calls.append(sql)
        if len(calls) == 1:
            # Anchor lookup.
            return [{"id": "anchor", "doc_id": "D1", "ord": 5}]
        # Window query.
        return [
            {"id": "n4", "doc_id": "D1", "ord": 4, "text": "before"},
            {"id": "anchor", "doc_id": "D1", "ord": 5, "text": "self"},
            {"id": "n6", "doc_id": "D1", "ord": 6, "text": "after"},
        ]

    tool = TableNeighborsTool(
        registry=_registry(_binding_full_roles()),
        schema_cache=FakeSchemaCache(_schema()),
        sql_executor=exec_sql,
    )
    args = tool.validate_arguments(
        {"binding": "chunks", "id": "anchor", "before": 1, "after": 1}
    )
    res = await tool.execute(args, _ctx())
    assert res.success is True
    payload = json.loads(res.content)
    assert payload["anchor_id"] == "anchor"
    assert payload["window"] == {"lower": 4, "upper": 6}
    assert len(payload["rows"]) == 3
    assert len(calls) == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_neighbors_anchor_not_found() -> None:
    tool = TableNeighborsTool(
        registry=_registry(_binding_full_roles()),
        schema_cache=FakeSchemaCache(_schema()),
        sql_executor=lambda *_args, **_kwargs: [],
    )
    args = tool.validate_arguments({"binding": "chunks", "id": "nope"})
    res = await tool.execute(args, _ctx())
    assert res.success is False
    payload = json.loads(res.content)
    assert payload["error"]["error_code"] == "invalid_binding"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_neighbors_missing_role_columns() -> None:
    tool = TableNeighborsTool(
        registry=_registry(_binding_partial_roles()),
        schema_cache=FakeSchemaCache(_schema()),
        sql_executor=lambda *_args, **_kwargs: [],
    )
    args = tool.validate_arguments({"binding": "docs", "id": "x"})
    res = await tool.execute(args, _ctx())
    assert res.success is False
    payload = json.loads(res.content)
    assert payload["error"]["error_code"] == "neighbor_config_missing"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_neighbors_non_integer_order_value() -> None:
    def exec_sql(sql: str, params: list[Any], token: str) -> list[dict[str, Any]]:
        return [{"id": "anchor", "doc_id": "D1", "ord": "not-an-int"}]

    tool = TableNeighborsTool(
        registry=_registry(_binding_full_roles()),
        schema_cache=FakeSchemaCache(_schema()),
        sql_executor=exec_sql,
    )
    args = tool.validate_arguments({"binding": "chunks", "id": "anchor"})
    res = await tool.execute(args, _ctx())
    assert res.success is False
    payload = json.loads(res.content)
    assert payload["error"]["error_code"] == "neighbor_config_missing"
