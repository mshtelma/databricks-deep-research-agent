"""Unit tests for the UC-browse extensions to DesignerDiscoveryAdapter.

Monkeypatches ``_build_sql_executor`` to inject a fake SqlExecutor so the browse
dispatch + signature + run-readiness paths are exercised without a warehouse.
Browse uses SHOW/DESCRIBE (BROWSE-sufficient); a pure uc-only request never
triggers a full discovery sweep (discover_all is never called — the dummy
discovery service has no such method).
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from deep_research.agent_designer.discovery import (
    DesignerDiscoveryAdapter,
    UcBrowseError,
    _DiscoveryServiceProto,
)


def _executor(responses: list[tuple[str, list[dict[str, Any]]]]):
    def _exec(sql: str, params: list[Any], token: str) -> list[dict[str, Any]]:
        for pattern, rows in responses:
            if pattern in sql:
                return rows
        return []

    return _exec


def _describe(lines: list[str]) -> list[dict[str, Any]]:
    return [{"function_desc": line} for line in lines]


def _adapter() -> DesignerDiscoveryAdapter:
    # object() has no discover_all: a uc-only request must never reach it.
    return DesignerDiscoveryAdapter(cast(_DiscoveryServiceProto, object()))


def _patch_executor(monkeypatch: pytest.MonkeyPatch, adapter: DesignerDiscoveryAdapter, fake: Any) -> None:
    # _build_sql_executor now takes optional catalog/schema kwargs (function ctx).
    monkeypatch.setattr(adapter, "_build_sql_executor", lambda _ut, **_kw: fake)


def test_list_uc_resources_catalogs(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter()
    fake = _executor([("SHOW CATALOGS", [{"catalog": "main"}, {"catalog": "sales"}])])
    _patch_executor(monkeypatch, adapter, fake)
    res = adapter._list_uc_resources(["uc_catalog"], None, "", None)
    assert [r.name for r in res] == ["main", "sales"]
    assert res[0].kind == "uc_catalog"
    assert res[0].full_name == "main"


def test_list_uc_resources_schemas_need_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter()
    fake = _executor([("SHOW SCHEMAS", [{"databaseName": "finance"}])])
    _patch_executor(monkeypatch, adapter, fake)
    # No parent => no schemas queried.
    assert adapter._list_uc_resources(["uc_schema"], None, "", None) == []
    res = adapter._list_uc_resources(["uc_schema"], "main", "", None)
    assert [r.full_name for r in res] == ["main.finance"]


def test_list_uc_resources_functions_split_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter()
    fake = _executor([("SHOW USER FUNCTIONS", [{"function": "main.finance.get_price"}])])
    _patch_executor(monkeypatch, adapter, fake)
    res = adapter._list_uc_resources(["uc_function"], "main.finance", "", None)
    assert res[0].full_name == "main.finance.get_price"
    assert res[0].name == "get_price"


def test_list_uc_resources_no_warehouse_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter()
    monkeypatch.setattr(adapter, "_build_sql_executor", lambda _ut, **_kw: None)
    assert adapter._list_uc_resources(["uc_catalog"], None, "", None) == []


def test_get_function_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter()
    fake = _executor(
        [
            (
                "DESCRIBE FUNCTION",
                _describe(
                    [
                        "Function: main.finance.get_price",
                        "Type: SCALAR",
                        "Input: ticker STRING 'the ticker'",
                        "Returns: DOUBLE",
                    ]
                ),
            )
            # information_schema.schemata probe returns [] => run_ready True
        ]
    )
    _patch_executor(monkeypatch, adapter, fake)
    sig = adapter.get_function_signature(None, "main.finance.get_price")
    assert sig["scalar"] is True
    assert sig["returns_table"] is False
    assert sig["run_ready"] is True
    assert sig["params"][0]["name"] == "ticker"


def test_get_function_signature_table_and_not_run_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter()

    def fake(sql: str, params: list[Any], token: str) -> list[dict[str, Any]]:
        if "DESCRIBE FUNCTION" in sql:
            return _describe(
                [
                    "Function: mcp.default.get_orders",
                    "Type: TABLE",
                    "Input: input_customer_id STRING 'id'",
                    "Returns: sale_id STRING",
                ]
            )
        if "information_schema.schemata" in sql:
            raise RuntimeError(
                "[INSUFFICIENT_PERMISSIONS] User does not have USE CATALOG on "
                "Catalog 'mcp'. SQLSTATE: 42501"
            )
        return []

    _patch_executor(monkeypatch, adapter, fake)
    sig = adapter.get_function_signature(None, "mcp.default.get_orders")
    assert sig["returns_table"] is True
    assert sig["run_ready"] is False
    assert sig["warning"] and "USE CATALOG" in sig["warning"]
    assert "table-valued" in sig["warning"]


def test_get_function_signature_no_warehouse_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter()
    monkeypatch.setattr(adapter, "_build_sql_executor", lambda _ut, **_kw: None)
    sig = adapter.get_function_signature(None, "main.finance.get_price")
    assert sig["scalar"] is True
    assert sig["params"] == []
    assert sig["run_ready"] is False
    assert "warning" in sig


@pytest.mark.asyncio
async def test_list_for_user_uc_only_skips_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter()
    fake = _executor([("SHOW CATALOGS", [{"catalog": "main"}])])
    _patch_executor(monkeypatch, adapter, fake)
    # If discover_all were called, object() would AttributeError.
    res = await adapter.list_for_user(user_token="", kinds=["uc_catalog"], user_id="u1")
    assert [r.name for r in res] == ["main"]


@pytest.mark.asyncio
async def test_list_for_user_pure_uc_browse_error_is_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter()

    def fake(sql: str, params: list[Any], token: str) -> list[dict[str, Any]]:
        raise RuntimeError(
            "[INSUFFICIENT_PERMISSIONS] User does not have USE CATALOG on "
            "Catalog 'x'. SQLSTATE: 42501"
        )

    _patch_executor(monkeypatch, adapter, fake)
    with pytest.raises(UcBrowseError) as exc_info:
        await adapter.list_for_user(
            user_token="", kinds=["uc_schema"], user_id="u1", parent="x"
        )
    assert exc_info.value.code == "permission"


# ---------------------------------------------------------------------------
# Catalog-scoped function search (parent = bare catalog)
# ---------------------------------------------------------------------------

_PERM_ERROR = (
    "[INSUFFICIENT_PERMISSIONS] User does not have USE CATALOG on "
    "Catalog 'main'. SQLSTATE: 42501"
)


@pytest.fixture(autouse=True)
def _clear_uc_search_cache():
    from deep_research.agent_designer import discovery

    discovery._UC_SEARCH_CACHE.clear()
    yield
    discovery._UC_SEARCH_CACHE.clear()


def test_catalog_search_uses_information_schema_when_use_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter()
    fake = _executor(
        [
            ("information_schema.schemata", []),  # USE CATALOG probe succeeds
            (
                "information_schema.routines",
                [
                    {"routine_schema": "metrics", "routine_name": "pct_change"},
                    {"routine_schema": "sales", "routine_name": "pct_margin"},
                ],
            ),
        ]
    )
    _patch_executor(monkeypatch, adapter, fake)
    res = adapter._list_uc_resources(["uc_function"], "main", "pct", None, "u1")
    assert [r.full_name for r in res] == [
        "main.metrics.pct_change",
        "main.sales.pct_margin",
    ]
    assert adapter.uc_search_warning is None


def test_catalog_search_falls_back_to_show_fanout_without_use_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter()

    def fake(sql: str, params: list[Any], token: str) -> list[dict[str, Any]]:
        if "information_schema.schemata" in sql:
            raise RuntimeError(_PERM_ERROR)
        if "SHOW SCHEMAS" in sql:
            return [{"databaseName": "metrics"}, {"databaseName": "sales"}]
        if "SHOW USER FUNCTIONS" in sql:
            # The scoped executor is shared in tests; list_functions keeps only
            # rows qualified by its own catalog.schema, so both appear once.
            return [
                {"function": "main.metrics.pct_change"},
                {"function": "main.sales.pct_margin"},
                {"function": "main.sales.other_fn"},
            ]
        return []

    _patch_executor(monkeypatch, adapter, fake)
    res = adapter._list_uc_resources(["uc_function"], "main", "pct", None, "u1")
    assert [r.full_name for r in res] == [
        "main.metrics.pct_change",
        "main.sales.pct_margin",
    ]
    assert adapter.uc_search_warning is None


def test_catalog_search_truncates_schema_fanout_with_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deep_research.agent_designer import discovery

    adapter = _adapter()

    def fake(sql: str, params: list[Any], token: str) -> list[dict[str, Any]]:
        if "information_schema.schemata" in sql:
            raise RuntimeError(_PERM_ERROR)
        if "SHOW SCHEMAS" in sql:
            return [{"databaseName": f"s{i:02d}"} for i in range(30)]
        return []

    _patch_executor(monkeypatch, adapter, fake)
    res = adapter._list_uc_resources(["uc_function"], "main", "", None, "u1")
    assert res == []
    warning = adapter.uc_search_warning
    assert warning is not None
    assert f"first {discovery._UC_SEARCH_MAX_SCHEMAS} of 30" in warning


def test_catalog_search_results_are_cached_per_user_and_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter()
    calls = {"n": 0}

    def fake(sql: str, params: list[Any], token: str) -> list[dict[str, Any]]:
        calls["n"] += 1
        if "information_schema.schemata" in sql:
            return []
        if "information_schema.routines" in sql:
            return [{"routine_schema": "metrics", "routine_name": "pct_change"}]
        return []

    _patch_executor(monkeypatch, adapter, fake)
    first = adapter._list_uc_resources(["uc_function"], "main", "pct", None, "u1")
    count_after_first = calls["n"]
    second = adapter._list_uc_resources(["uc_function"], "main", "pct", None, "u1")
    assert calls["n"] == count_after_first  # served from cache
    assert [r.full_name for r in first] == [r.full_name for r in second]
    # A different prefix misses the cache.
    adapter._list_uc_resources(["uc_function"], "main", "pct_ch", None, "u1")
    assert calls["n"] > count_after_first


def test_schema_scoped_listing_still_works_with_dotted_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter()
    fake = _executor(
        [("SHOW USER FUNCTIONS", [{"function": "main.finance.get_price"}])]
    )
    _patch_executor(monkeypatch, adapter, fake)
    res = adapter._list_uc_resources(["uc_function"], "main.finance", "", None, "u1")
    assert [r.full_name for r in res] == ["main.finance.get_price"]
    assert adapter.uc_search_warning is None
