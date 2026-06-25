"""Unit tests for TableDiscoveryTool."""

from __future__ import annotations

import json
from typing import Any

import pytest

from databricks_deep_research.tools.builtins.text_table import (
    BindingInfo,
    BindingSource,
    RoleMap,
    Schema,
    SchemaCache,
    SchemaColumn,
    TableBindingRegistry,
    TableDiscoveryTool,
)
from databricks_deep_research.tools.protocol import ToolContext


class FakeProvider:
    def __init__(self, items: list[BindingInfo]) -> None:
        self._items = items
        self.calls: list[dict[str, Any]] = []

    async def list_tables(
        self,
        *,
        user_token: str,
        name_pattern: str | None = None,
    ) -> list[BindingInfo]:
        self.calls.append({"user_token": user_token, "name_pattern": name_pattern})
        if name_pattern:
            return [b for b in self._items if name_pattern.lower() in b.name.lower()]
        return list(self._items)


class FailingProvider:
    async def list_tables(
        self,
        *,
        user_token: str,
        name_pattern: str | None = None,
    ) -> list[BindingInfo]:
        raise RuntimeError("upstream fail")


def _ctx(user_token: str = "tok-x") -> ToolContext:
    return ToolContext(extras={"user_token": user_token})


def _binding(name: str, fqn: str, source: BindingSource = BindingSource.DISCOVERED) -> BindingInfo:
    return BindingInfo(
        name=name,
        fqn=fqn,
        source=source,
        description=f"desc-{name}",
        roles=RoleMap(id_column="id", content_column="text"),
    )


def _schema_for(fqn: str) -> Schema:
    return Schema(
        fqn=fqn,
        columns=(
            SchemaColumn(name="id", data_type="string", nullable=False),
            SchemaColumn(name="text", data_type="string"),
        ),
    )


@pytest.mark.unit
def test_definition_has_required_fields() -> None:
    registry = TableBindingRegistry()
    tool = TableDiscoveryTool(provider=None, registry=registry)
    d = tool.definition
    assert d.name == "table_discovery"
    assert d.parameters["type"] == "object"
    assert "name_pattern" in d.parameters["properties"]
    assert "detail" in d.parameters["properties"]


@pytest.mark.unit
def test_validate_arguments_defaults() -> None:
    tool = TableDiscoveryTool(provider=None, registry=TableBindingRegistry())
    out = tool.validate_arguments({})
    assert out == {"name_pattern": None, "detail": "basic"}


@pytest.mark.unit
def test_validate_arguments_rejects_invalid_detail() -> None:
    tool = TableDiscoveryTool(provider=None, registry=TableBindingRegistry())
    with pytest.raises(ValueError):
        tool.validate_arguments({"detail": "fancy"})


@pytest.mark.unit
def test_validate_arguments_rejects_non_string_pattern() -> None:
    tool = TableDiscoveryTool(provider=None, registry=TableBindingRegistry())
    with pytest.raises(ValueError):
        tool.validate_arguments({"name_pattern": 5})


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_provider_returns_discovery_unavailable() -> None:
    registry = TableBindingRegistry()
    tool = TableDiscoveryTool(provider=None, registry=registry)
    res = await tool.execute({}, _ctx())
    assert res.success is False
    payload = json.loads(res.content)
    assert payload["error"]["error_code"] == "discovery_unavailable"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_basic_lists_and_registers_discovered() -> None:
    items = [_binding("t1", "c.s.t1"), _binding("t2", "c.s.t2")]
    provider = FakeProvider(items)
    registry = TableBindingRegistry()
    tool = TableDiscoveryTool(provider=provider, registry=registry)

    res = await tool.execute({"detail": "basic"}, _ctx())
    assert res.success is True
    payload = json.loads(res.content)
    names = {row["name"] for row in payload["tables"]}
    assert names == {"t1", "t2"}
    # Each registered into registry
    assert "t1" in registry
    assert "t2" in registry
    # No 'schema' key for basic
    for row in payload["tables"]:
        assert "schema" not in row


@pytest.mark.unit
@pytest.mark.asyncio
async def test_schema_detail_requires_schema_cache() -> None:
    provider = FakeProvider([_binding("t1", "c.s.t1")])
    tool = TableDiscoveryTool(
        provider=provider, registry=TableBindingRegistry(), schema_cache=None
    )
    res = await tool.execute({"detail": "schema"}, _ctx())
    assert res.success is False
    payload = json.loads(res.content)
    assert payload["error"]["error_code"] == "discovery_unavailable"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_schema_detail_returns_columns() -> None:
    items = [_binding("t1", "c.s.t1")]
    provider = FakeProvider(items)
    cache = SchemaCache(fetcher=lambda fqn, _t: _schema_for(fqn))
    tool = TableDiscoveryTool(
        provider=provider, registry=TableBindingRegistry(), schema_cache=cache
    )
    res = await tool.execute({"detail": "schema"}, _ctx())
    assert res.success is True
    payload = json.loads(res.content)
    row = payload["tables"][0]
    assert row["name"] == "t1"
    assert any(c["name"] == "id" for c in row["schema"])
    assert any(c["name"] == "text" for c in row["schema"])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_full_detail_includes_redacted_sample() -> None:
    items = [_binding("t1", "c.s.t1")]
    provider = FakeProvider(items)
    cache = SchemaCache(fetcher=lambda fqn, _t: _schema_for(fqn))
    captured_sql: list[str] = []

    def exec_sql(sql: str, params: list[Any], token: str) -> list[dict[str, Any]]:
        captured_sql.append(sql)
        return [
            {
                "id": "1",
                "text": "Contact jane@example.com or 555-123-4567",
            }
        ]

    tool = TableDiscoveryTool(
        provider=provider,
        registry=TableBindingRegistry(),
        schema_cache=cache,
        sql_executor=exec_sql,
    )
    res = await tool.execute({"detail": "full"}, _ctx())
    assert res.success is True
    payload = json.loads(res.content)
    assert payload["tables"][0]["sample"] == [
        {
            "id": "1",
            "text": "Contact [redacted-email] or [redacted-phone]",
        }
    ]
    assert "LIMIT 1" in captured_sql[0]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_provider_exception_becomes_error_result() -> None:
    tool = TableDiscoveryTool(
        provider=FailingProvider(), registry=TableBindingRegistry()
    )
    res = await tool.execute({}, _ctx())
    assert res.success is False
    payload = json.loads(res.content)
    assert payload["error"]["error_code"] == "discovery_unavailable"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bound_provider_records_coerced_to_discovered() -> None:
    items = [_binding("t1", "c.s.t1", source=BindingSource.BOUND)]
    provider = FakeProvider(items)
    registry = TableBindingRegistry()
    tool = TableDiscoveryTool(provider=provider, registry=registry)
    res = await tool.execute({}, _ctx())
    assert res.success is True
    info = registry.get("t1")
    assert info.source is BindingSource.DISCOVERED


@pytest.mark.unit
@pytest.mark.asyncio
async def test_collision_warning_emitted_when_name_clashes_with_bound() -> None:
    registry = TableBindingRegistry()
    bound = BindingInfo(
        name="t1", fqn="c.s.bound1", source=BindingSource.BOUND,
        roles=RoleMap(id_column="id", content_column="text"),
    )
    registry.register_bound(bound)

    provider = FakeProvider([_binding("t1", "c.s.disc1")])
    tool = TableDiscoveryTool(provider=provider, registry=registry)
    res = await tool.execute({}, _ctx())
    payload = json.loads(res.content)
    # The discovered entry was namespaced under discovered.t1
    assert any(t["name"] == "discovered.t1" for t in payload["tables"])
    assert "warnings" in payload
    assert payload["warnings"][0]["error_code"] == "duplicate_binding"
