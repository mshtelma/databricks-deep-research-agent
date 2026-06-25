"""Unit tests for TableLoadTool."""

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
    Table,
    TableBindingRegistry,
    TableLoadTool,
)
from databricks_deep_research.tools.protocol import ToolContext


def _binding() -> BindingInfo:
    return BindingInfo(
        name="docs",
        fqn="cat.sch.docs",
        source=BindingSource.BOUND,
        roles=RoleMap(id_column="id", content_column="text"),
    )


def _binding_no_roles() -> BindingInfo:
    return BindingInfo(
        name="raw",
        fqn="cat.sch.raw",
        source=BindingSource.DISCOVERED,
        roles=None,
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
        ),
    )


def _registry(*infos: BindingInfo) -> TableBindingRegistry:
    r = TableBindingRegistry()
    for info in infos:
        if info.source is BindingSource.BOUND:
            r.register_bound(info)
        else:
            r.register_discovered(info)
    return r


def _ctx() -> ToolContext:
    return ToolContext(extras={"user_token": "tok"})


@pytest.mark.unit
def test_definition() -> None:
    tool = TableLoadTool(
        registry=_registry(_binding()),
        schema_cache=FakeSchemaCache(_schema()),
        sql_executor=lambda *_args, **_kwargs: [],
    )
    d = tool.definition
    assert d.name == "table_load"
    assert d.parameters["required"] == ["binding", "id"]


@pytest.mark.unit
def test_validate_rejects_invalid_as_var() -> None:
    tool = TableLoadTool(
        registry=_registry(_binding()),
        schema_cache=FakeSchemaCache(_schema()),
        sql_executor=lambda *_args, **_kwargs: [],
    )
    with pytest.raises(ValueError):
        tool.validate_arguments(
            {"binding": "docs", "id": "1", "as_var": "not a name"}
        )


@pytest.mark.unit
def test_validate_rejects_empty_id_list() -> None:
    tool = TableLoadTool(
        registry=_registry(_binding()),
        schema_cache=FakeSchemaCache(_schema()),
        sql_executor=lambda *_args, **_kwargs: [],
    )
    with pytest.raises(ValueError):
        tool.validate_arguments({"binding": "docs", "id": []})


@pytest.mark.unit
@pytest.mark.asyncio
async def test_load_single_id_returns_row() -> None:
    rows = [{"id": "1", "text": "hello"}]
    tool = TableLoadTool(
        registry=_registry(_binding()),
        schema_cache=FakeSchemaCache(_schema()),
        sql_executor=lambda *_args, **_kwargs: rows,
    )
    args = tool.validate_arguments({"binding": "docs", "id": "1"})
    res = await tool.execute(args, _ctx())
    assert res.success is True
    payload = json.loads(res.content)
    assert payload["loaded"] == 1
    assert payload["rows"] == rows


@pytest.mark.unit
@pytest.mark.asyncio
async def test_load_multi_id_uses_or_branch() -> None:
    captured_sql: list[str] = []

    def exec_sql(sql: str, params: list[Any], token: str) -> list[dict[str, Any]]:
        captured_sql.append(sql)
        return [{"id": "1", "text": "a"}, {"id": "2", "text": "b"}]

    tool = TableLoadTool(
        registry=_registry(_binding()),
        schema_cache=FakeSchemaCache(_schema()),
        sql_executor=exec_sql,
    )
    args = tool.validate_arguments({"binding": "docs", "id": ["1", "2"]})
    res = await tool.execute(args, _ctx())
    assert res.success is True
    payload = json.loads(res.content)
    assert payload["loaded"] == 2
    # The compiled SQL should use OR over equality predicates.
    assert " OR " in captured_sql[0]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_load_namespace_setter_invoked() -> None:
    rows = [{"id": "1", "text": "hello"}]
    captured: dict[str, Any] = {}

    def setter(name: str, value: Any) -> None:
        captured[name] = value

    tool = TableLoadTool(
        registry=_registry(_binding()),
        schema_cache=FakeSchemaCache(_schema()),
        sql_executor=lambda *_args, **_kwargs: rows,
        compute_namespace_setter=setter,
    )
    args = tool.validate_arguments(
        {"binding": "docs", "id": "1", "as_var": "anchor"}
    )
    await tool.execute(args, _ctx())
    assert "anchor" in captured
    assert isinstance(captured["anchor"], Table)
    assert "last_table" in captured
    assert "tables" in captured
    assert isinstance(captured["tables"], list)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_load_namespace_setter_optional() -> None:
    rows = [{"id": "1", "text": "x"}]
    tool = TableLoadTool(
        registry=_registry(_binding()),
        schema_cache=FakeSchemaCache(_schema()),
        sql_executor=lambda *_args, **_kwargs: rows,
        compute_namespace_setter=None,
    )
    args = tool.validate_arguments({"binding": "docs", "id": "1"})
    res = await tool.execute(args, _ctx())
    # Without a setter, just JSON returned and no crash.
    assert res.success is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_load_discovered_binding_infers_roles() -> None:
    r = _registry(_binding_no_roles())
    calls: list[str] = []

    def exec_sql(sql: str, params: list[Any], token: str) -> list[dict[str, Any]]:
        calls.append(sql)
        if len(calls) == 1:
            return [
                {
                    "id": "1",
                    "text": "sample text for discovered binding inference. " * 20,
                    "kind": "paragraph",
                }
            ]
        return [{"id": "1", "text": "x"}]

    tool = TableLoadTool(
        registry=r,
        schema_cache=FakeSchemaCache(_schema()),
        sql_executor=exec_sql,
    )
    args = tool.validate_arguments({"binding": "raw", "id": "1"})
    res = await tool.execute(args, _ctx())
    assert res.success is True
    payload = json.loads(res.content)
    assert payload["loaded"] == 1
    inferred = r.get("raw").roles
    assert inferred is not None
    assert inferred.id_column == "id"
    assert inferred.content_column == "text"
    assert len(calls) == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_load_with_explicit_columns_includes_id() -> None:
    rows = [{"id": "1", "text": "x"}]
    tool = TableLoadTool(
        registry=_registry(_binding()),
        schema_cache=FakeSchemaCache(_schema()),
        sql_executor=lambda *_args, **_kwargs: rows,
    )
    args = tool.validate_arguments(
        {"binding": "docs", "id": "1", "columns": ["text"]}
    )
    res = await tool.execute(args, _ctx())
    assert res.success is True
