"""Unit tests for the UC-browse extensions to DesignerDiscoveryAdapter.

Monkeypatches ``_build_sql_executor`` to inject a fake SqlExecutor so the browse
dispatch + signature path are exercised without a warehouse. Also asserts that a
uc-only request does NOT trigger a full discovery sweep (discover_all is never
called — the dummy discovery service has no such method).
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from deep_research.agent_designer.discovery import (
    DesignerDiscoveryAdapter,
    _DiscoveryServiceProto,
)


def _executor(responses: list[tuple[str, list[dict[str, Any]]]]):
    def _exec(sql: str, params: list[Any], token: str) -> list[dict[str, Any]]:
        for pattern, rows in responses:
            if pattern in sql:
                return rows
        return []

    return _exec


def _adapter() -> DesignerDiscoveryAdapter:
    # object() has no discover_all: a uc-only request must never reach it.
    return DesignerDiscoveryAdapter(cast(_DiscoveryServiceProto, object()))


def test_list_uc_resources_catalogs(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter()
    fake = _executor([("SHOW CATALOGS", [{"catalog": "main"}, {"catalog": "sales"}])])
    monkeypatch.setattr(adapter, "_build_sql_executor", lambda _ut: fake)
    res = adapter._list_uc_resources(["uc_catalog"], None, "", None)
    assert [r.name for r in res] == ["main", "sales"]
    assert res[0].kind == "uc_catalog"
    assert res[0].full_name == "main"


def test_list_uc_resources_schemas_need_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter()
    fake = _executor([("information_schema.schemata", [{"schema_name": "finance"}])])
    monkeypatch.setattr(adapter, "_build_sql_executor", lambda _ut: fake)
    # No parent => no schemas queried.
    assert adapter._list_uc_resources(["uc_schema"], None, "", None) == []
    res = adapter._list_uc_resources(["uc_schema"], "main", "", None)
    assert [r.full_name for r in res] == ["main.finance"]


def test_list_uc_resources_functions_split_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter()
    fake = _executor(
        [("information_schema.routines", [{"routine_name": "get_price", "data_type": "DOUBLE"}])]
    )
    monkeypatch.setattr(adapter, "_build_sql_executor", lambda _ut: fake)
    res = adapter._list_uc_resources(["uc_function"], "main.finance", "", None)
    assert res[0].full_name == "main.finance.get_price"
    assert res[0].description == "returns DOUBLE"


def test_list_uc_resources_no_warehouse_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter()
    monkeypatch.setattr(adapter, "_build_sql_executor", lambda _ut: None)
    assert adapter._list_uc_resources(["uc_catalog"], None, "", None) == []


def test_get_function_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter()
    rows = [
        {
            "specific_name": "get_price",
            "parameter_name": "ticker",
            "data_type": "STRING",
            "full_data_type": "STRING",
            "ordinal_position": 1,
            "parameter_default": None,
        }
    ]
    fake = _executor([("information_schema.parameters", rows)])
    monkeypatch.setattr(adapter, "_build_sql_executor", lambda _ut: fake)
    sig = adapter.get_function_signature(None, "main.finance.get_price")
    assert sig["scalar"] is True
    assert sig["params"][0]["name"] == "ticker"


def test_get_function_signature_no_warehouse_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter()
    monkeypatch.setattr(adapter, "_build_sql_executor", lambda _ut: None)
    sig = adapter.get_function_signature(None, "main.finance.get_price")
    assert sig["scalar"] is True
    assert sig["params"] == []
    assert "warning" in sig


@pytest.mark.asyncio
async def test_list_for_user_uc_only_skips_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter()
    fake = _executor([("SHOW CATALOGS", [{"catalog": "main"}])])
    monkeypatch.setattr(adapter, "_build_sql_executor", lambda _ut: fake)
    # If discover_all were called, object() would AttributeError.
    res = await adapter.list_for_user(user_token="", kinds=["uc_catalog"], user_id="u1")
    assert [r.name for r in res] == ["main"]
