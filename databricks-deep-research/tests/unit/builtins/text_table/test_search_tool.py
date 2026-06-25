"""Unit tests for TableSearchTool."""

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
    TableSearchTool,
)
from databricks_deep_research.tools.protocol import ToolContext


def _binding_with_roles() -> BindingInfo:
    return BindingInfo(
        name="docs",
        fqn="cat.sch.docs",
        source=BindingSource.BOUND,
        description="docs corpus",
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
    tool = TableSearchTool(
        registry=_registry(),
        schema_cache=FakeSchemaCache(_schema()),
        sql_executor=lambda *_args, **_kwargs: [],
    )
    d = tool.definition
    assert d.name == "table_search"
    assert "binding" in d.parameters["properties"]
    assert "query" in d.parameters["properties"]
    assert d.parameters["required"] == ["binding", "query"]


@pytest.mark.unit
def test_validate_rejects_empty_query() -> None:
    tool = TableSearchTool(
        registry=_registry(),
        schema_cache=FakeSchemaCache(_schema()),
        sql_executor=lambda *_args, **_kwargs: [],
    )
    with pytest.raises(ValueError):
        tool.validate_arguments({"binding": "docs", "query": ""})


@pytest.mark.unit
def test_validate_clamps_limit() -> None:
    tool = TableSearchTool(
        registry=_registry(),
        schema_cache=FakeSchemaCache(_schema()),
        sql_executor=lambda *_args, **_kwargs: [],
    )
    out = tool.validate_arguments(
        {"binding": "docs", "query": "x", "limit": 10**6}
    )
    assert out["limit"] <= 5000


@pytest.mark.unit
@pytest.mark.asyncio
async def test_search_query_is_parameterized_sql_like() -> None:
    captured: list[tuple[str, list[Any]]] = []

    def exec_sql(sql: str, params: list[Any], token: str) -> list[dict[str, Any]]:
        captured.append((sql, params))
        return [
            {"id": "1", "text": "Apple%_ Inc reported strong earnings"},
            {"id": "3", "text": "apple%_ farm prices"},
        ]

    tool = TableSearchTool(
        registry=_registry(_binding_with_roles()),
        schema_cache=FakeSchemaCache(_schema()),
        sql_executor=exec_sql,
    )
    args = tool.validate_arguments({"binding": "docs", "query": "apple%_"})
    res = await tool.execute(args, _ctx())
    assert res.success is True
    payload = json.loads(res.content)
    assert payload["total_matched"] == 2
    assert {r["id"] for r in payload["results"]} == {"1", "3"}
    assert all(r["score"] == 1.0 for r in payload["results"])
    sql, params = captured[0]
    assert "LOWER(`text`) LIKE LOWER(:p_text_search_1)" in sql
    assert "apple%_" not in sql
    assert params[-1].value == "%apple\\%\\_%"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_search_paginates_matches_in_sql() -> None:
    captured_sql: list[str] = []

    def exec_sql(sql: str, params: list[Any], token: str) -> list[dict[str, Any]]:
        captured_sql.append(sql)
        return [{"id": str(i), "text": f"apple-{i}"} for i in range(10, 15)]

    tool = TableSearchTool(
        registry=_registry(_binding_with_roles()),
        schema_cache=FakeSchemaCache(_schema()),
        sql_executor=exec_sql,
    )
    args = tool.validate_arguments(
        {"binding": "docs", "query": "apple", "limit": 5, "offset": 10}
    )
    res = await tool.execute(args, _ctx())
    payload = json.loads(res.content)
    ids = [r["id"] for r in payload["results"]]
    assert ids == ["10", "11", "12", "13", "14"]
    assert payload["total_matched"] == 5
    assert "LIMIT 5" in captured_sql[0]
    assert "OFFSET 10" in captured_sql[0]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_search_no_matches_returns_empty_results() -> None:
    tool = TableSearchTool(
        registry=_registry(_binding_with_roles()),
        schema_cache=FakeSchemaCache(_schema()),
        sql_executor=lambda *_args, **_kwargs: [],
    )
    args = tool.validate_arguments({"binding": "docs", "query": "xyz"})
    res = await tool.execute(args, _ctx())
    payload = json.loads(res.content)
    assert payload["results"] == []
    assert res.data["total_matched"] == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_search_discovered_binding_infers_roles_once() -> None:
    r = _registry(_binding_no_roles())
    calls: list[str] = []

    def exec_sql(sql: str, params: list[Any], token: str) -> list[dict[str, Any]]:
        calls.append(sql)
        if len(calls) == 1:
            return [
                {
                    "id": "1",
                    "text": "A long text passage about revenue. " * 20,
                    "kind": "p",
                }
            ]
        return [{"id": "1", "text": "A long text passage about revenue"}]

    tool = TableSearchTool(
        registry=r,
        schema_cache=FakeSchemaCache(_schema()),
        sql_executor=exec_sql,
    )
    args = tool.validate_arguments({"binding": "raw", "query": "revenue"})
    res = await tool.execute(args, _ctx())
    assert res.success is True
    inferred = r.get("raw").roles
    assert inferred is not None
    assert inferred.id_column == "id"
    assert inferred.content_column == "text"
    assert len(calls) == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_search_discovered_binding_accepts_explicit_roles() -> None:
    schema = Schema(
        fqn="cat.sch.raw",
        columns=(
            SchemaColumn(name="uuid", data_type="string", nullable=False),
            SchemaColumn(name="body", data_type="string"),
        ),
    )
    r = _registry(_binding_no_roles())
    calls: list[str] = []

    def exec_sql(sql: str, params: list[Any], token: str) -> list[dict[str, Any]]:
        calls.append(sql)
        return [{"uuid": "u1", "body": "needle in a haystack"}]

    tool = TableSearchTool(
        registry=r,
        schema_cache=FakeSchemaCache(schema),
        sql_executor=exec_sql,
    )
    args = tool.validate_arguments(
        {
            "binding": "raw",
            "query": "needle",
            "roles": {"id": "uuid", "content": "body"},
        }
    )
    res = await tool.execute(args, _ctx())
    assert res.success is True
    payload = json.loads(res.content)
    assert payload["results"][0]["id"] == "u1"
    assert r.get("raw").roles == RoleMap(id_column="uuid", content_column="body")
    assert len(calls) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_search_unknown_binding_returns_invalid_binding() -> None:
    tool = TableSearchTool(
        registry=_registry(),
        schema_cache=FakeSchemaCache(_schema()),
        sql_executor=lambda *_args, **_kwargs: [],
    )
    args = tool.validate_arguments({"binding": "ghost", "query": "x"})
    res = await tool.execute(args, _ctx())
    assert res.success is False
    payload = json.loads(res.content)
    assert payload["error"]["error_code"] == "invalid_binding"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_search_truncates_snippet_to_512_chars() -> None:
    long = "apple " + ("X" * 1000)
    tool = TableSearchTool(
        registry=_registry(_binding_with_roles()),
        schema_cache=FakeSchemaCache(_schema()),
        sql_executor=lambda *_args, **_kwargs: [{"id": "1", "text": long}],
    )
    args = tool.validate_arguments({"binding": "docs", "query": "apple"})
    res = await tool.execute(args, _ctx())
    payload = json.loads(res.content)
    assert len(payload["results"][0]["snippet"]) == 512
