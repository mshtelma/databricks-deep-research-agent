"""Tests for :class:`DesignerTableDiscoveryProvider`.

Covers:
- Static bindings are surfaced and coerced to ``source=DISCOVERED``.
- UC scope enumeration uses the SDK ``tables.list`` shape and builds FQNs.
- Substring filter is case-insensitive over name AND fqn.
- Token plaintext is not logged when the client factory succeeds OR fails.
- Factory failures are isolated to the affected scope; other scopes still run.
- ``role`` inference is NOT performed at discovery time —
  ``BindingInfo.roles`` is ``None`` for every returned record.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest
from databricks_deep_research.tools.builtins.text_table import (
    BindingInfo,
    BindingSource,
    RoleMap,
)

from deep_research.agent.adapters.table_discovery_adapter import (
    DesignerTableDiscoveryProvider,
    workspace_client_factory_from,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _table_info(name: str, *, catalog: str, schema: str, comment: str | None = None) -> Any:
    full_name = f"{catalog}.{schema}.{name}"
    return SimpleNamespace(name=name, full_name=full_name, comment=comment)


class _FakeTablesAPI:
    def __init__(self, mapping: dict[tuple[str, str], list[Any]]) -> None:
        self._mapping = mapping
        self.calls: list[dict[str, str]] = []
        self.fail_for: set[tuple[str, str]] = set()

    def list(self, *, catalog_name: str, schema_name: str) -> list[Any]:
        self.calls.append({"catalog": catalog_name, "schema": schema_name})
        if (catalog_name, schema_name) in self.fail_for:
            raise RuntimeError("simulated SDK failure")
        return self._mapping.get((catalog_name, schema_name), [])


class _FakeWorkspaceClient:
    def __init__(self, tables_api: _FakeTablesAPI) -> None:
        self.tables = tables_api


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_static_bindings_surface_with_source_discovered() -> None:
    bound = BindingInfo(
        name="alpha",
        fqn="cat.sch.alpha",
        source=BindingSource.BOUND,
        roles=RoleMap(id_column="id", content_column="body"),
    )
    provider = DesignerTableDiscoveryProvider(static_bindings=[bound])

    out = await provider.list_tables(user_token="t")

    assert len(out) == 1
    info = out[0]
    assert info.name == "alpha"
    assert info.fqn == "cat.sch.alpha"
    # Coerced to DISCOVERED, roles cleared.
    assert info.source is BindingSource.DISCOVERED
    assert info.roles is None


@pytest.mark.asyncio
async def test_uc_scope_enumeration_builds_fqn_and_description() -> None:
    api = _FakeTablesAPI(
        {
            ("main", "sales"): [
                _table_info("orders", catalog="main", schema="sales", comment="Order ledger"),
                _table_info("customers", catalog="main", schema="sales", comment=None),
            ],
        }
    )
    fake_client = _FakeWorkspaceClient(api)

    provider = DesignerTableDiscoveryProvider.from_pairs(
        client_factory=lambda *, user_token: fake_client,
        scopes=[("main", "sales")],
    )

    out = await provider.list_tables(user_token="x")
    by_name = {info.name: info for info in out}

    assert set(by_name) == {"orders", "customers"}
    assert by_name["orders"].source is BindingSource.DISCOVERED
    assert by_name["orders"].fqn == "main.sales.orders"
    assert by_name["orders"].description == "Order ledger"
    assert by_name["customers"].description is None
    assert all(info.roles is None for info in out)
    assert api.calls == [{"catalog": "main", "schema": "sales"}]


@pytest.mark.asyncio
async def test_static_bindings_take_precedence_over_uc() -> None:
    static = BindingInfo(
        name="orders",
        fqn="static.fqn.orders",
        source=BindingSource.DISCOVERED,
        description="static-supplied",
    )
    api = _FakeTablesAPI(
        {
            ("main", "sales"): [
                _table_info("orders", catalog="main", schema="sales"),
            ],
        }
    )
    provider = DesignerTableDiscoveryProvider.from_pairs(
        client_factory=lambda *, user_token: _FakeWorkspaceClient(api),
        scopes=[("main", "sales")],
        static_bindings=[static],
    )

    out = await provider.list_tables(user_token="x")
    assert len(out) == 1
    assert out[0].fqn == "static.fqn.orders"
    assert out[0].description == "static-supplied"


@pytest.mark.asyncio
async def test_substring_filter_is_case_insensitive_on_name_and_fqn() -> None:
    api = _FakeTablesAPI(
        {
            ("Cat", "Sch"): [
                _table_info("OrderHeader", catalog="Cat", schema="Sch"),
                _table_info("ProductCatalog", catalog="Cat", schema="Sch"),
                _table_info("inventory", catalog="Cat", schema="Sch"),
            ],
        }
    )
    provider = DesignerTableDiscoveryProvider.from_pairs(
        client_factory=lambda *, user_token: _FakeWorkspaceClient(api),
        scopes=[("Cat", "Sch")],
    )

    by_name = await provider.list_tables(user_token="x", name_pattern="order")
    assert {info.name for info in by_name} == {"OrderHeader"}

    by_fqn = await provider.list_tables(user_token="x", name_pattern="cat.sch")
    # All entries' fqn contains "Cat.Sch" — case-insensitive match.
    assert {info.name for info in by_fqn} == {
        "OrderHeader",
        "ProductCatalog",
        "inventory",
    }


@pytest.mark.asyncio
async def test_failed_scope_does_not_break_remaining_scopes() -> None:
    api = _FakeTablesAPI(
        {
            ("main", "sales"): [
                _table_info("orders", catalog="main", schema="sales"),
            ],
            ("main", "ops"): [
                _table_info("events", catalog="main", schema="ops"),
            ],
        }
    )
    api.fail_for.add(("main", "sales"))

    provider = DesignerTableDiscoveryProvider.from_pairs(
        client_factory=lambda *, user_token: _FakeWorkspaceClient(api),
        scopes=[("main", "sales"), ("main", "ops")],
    )

    out = await provider.list_tables(user_token="x")
    assert {info.name for info in out} == {"events"}


@pytest.mark.asyncio
async def test_client_factory_failure_falls_back_to_static() -> None:
    static = BindingInfo(
        name="alpha",
        fqn="cat.sch.alpha",
        source=BindingSource.DISCOVERED,
    )

    def failing_factory(*, user_token: str) -> Any:
        raise RuntimeError("auth failure")

    provider = DesignerTableDiscoveryProvider.from_pairs(
        client_factory=failing_factory,
        scopes=[("main", "sales")],
        static_bindings=[static],
    )

    out = await provider.list_tables(user_token="secret-token")
    assert {info.name for info in out} == {"alpha"}


@pytest.mark.asyncio
async def test_token_is_not_logged_on_factory_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sensitive = "dapi-supersecret-1234567890"

    def failing_factory(*, user_token: str) -> Any:
        raise RuntimeError("auth failure")

    provider = DesignerTableDiscoveryProvider.from_pairs(
        client_factory=failing_factory,
        scopes=[("main", "sales")],
    )

    with caplog.at_level(logging.WARNING):
        await provider.list_tables(user_token=sensitive)

    full_log = "\n".join(record.getMessage() for record in caplog.records)
    assert sensitive not in full_log


@pytest.mark.asyncio
async def test_token_is_not_logged_on_scope_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sensitive = "dapi-supersecret-9876543210"
    api = _FakeTablesAPI({})
    api.fail_for.add(("main", "sales"))

    provider = DesignerTableDiscoveryProvider.from_pairs(
        client_factory=lambda *, user_token: _FakeWorkspaceClient(api),
        scopes=[("main", "sales")],
    )

    with caplog.at_level(logging.WARNING):
        await provider.list_tables(user_token=sensitive)

    full_log = "\n".join(record.getMessage() for record in caplog.records)
    assert sensitive not in full_log


@pytest.mark.asyncio
async def test_no_factory_no_scopes_returns_static_only() -> None:
    static = BindingInfo(
        name="solo",
        fqn="x.y.solo",
        source=BindingSource.DISCOVERED,
    )
    provider = DesignerTableDiscoveryProvider(static_bindings=[static])

    out = await provider.list_tables(user_token="")
    assert len(out) == 1
    assert out[0].name == "solo"


@pytest.mark.asyncio
async def test_workspace_client_factory_from_returns_supplied_client() -> None:
    api = _FakeTablesAPI(
        {
            ("c", "s"): [_table_info("t", catalog="c", schema="s")],
        }
    )
    client = _FakeWorkspaceClient(api)
    factory = workspace_client_factory_from(client)

    assert factory(user_token="ignored-1") is client
    assert factory(user_token="ignored-2") is client


@pytest.mark.asyncio
async def test_invalid_scope_strings_are_ignored() -> None:
    """``from_pairs`` filters empty strings; explicit scope tuples are skipped."""
    api = _FakeTablesAPI({})
    provider = DesignerTableDiscoveryProvider.from_pairs(
        client_factory=lambda *, user_token: _FakeWorkspaceClient(api),
        scopes=[("", "sales"), ("main", ""), ("main", "valid")],
    )
    api._mapping[("main", "valid")] = [
        _table_info("t", catalog="main", schema="valid"),
    ]

    out = await provider.list_tables(user_token="x")
    assert {info.name for info in out} == {"t"}
    # Only the valid scope was queried.
    assert api.calls == [{"catalog": "main", "schema": "valid"}]


@pytest.mark.asyncio
async def test_results_are_sorted_by_fqn_for_determinism() -> None:
    api = _FakeTablesAPI(
        {
            ("main", "sales"): [
                _table_info("zeta", catalog="main", schema="sales"),
                _table_info("alpha", catalog="main", schema="sales"),
                _table_info("mid", catalog="main", schema="sales"),
            ],
        }
    )
    provider = DesignerTableDiscoveryProvider.from_pairs(
        client_factory=lambda *, user_token: _FakeWorkspaceClient(api),
        scopes=[("main", "sales")],
    )
    out = await provider.list_tables(user_token="x")
    fqns = [info.fqn for info in out]
    assert fqns == sorted(fqns)
