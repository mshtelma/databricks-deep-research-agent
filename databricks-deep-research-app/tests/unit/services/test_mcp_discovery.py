"""Tests for MCP server discovery (C1)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from deep_research.services.mcp_discovery import (
    discover_mcp_connections,
    is_mcp_connection,
    managed_mcp_catalog,
)


def _conn(
    name: str,
    *,
    ctype: str = "HTTP",
    options: dict | None = None,
    properties: dict | None = None,
    url: str = "",
    comment: str = "",
) -> Any:
    return SimpleNamespace(
        name=name,
        connection_type=SimpleNamespace(value=ctype),
        options=options or {},
        properties=properties or {},
        url=url,
        comment=comment,
    )


# ---------------------------------------------------------------------------
# is_mcp_connection
# ---------------------------------------------------------------------------


def test_http_with_mcp_option_key_is_mcp() -> None:
    assert is_mcp_connection(_conn("c", options={"mcp_server_url": "https://x"})) is True


def test_http_with_mcp_in_url_is_mcp() -> None:
    assert (
        is_mcp_connection(_conn("c", url="https://w/api/2.0/mcp/external/c")) is True
    )


def test_http_without_marker_is_not_mcp() -> None:
    assert is_mcp_connection(_conn("c", options={"host": "https://x"})) is False


def test_non_http_is_not_mcp() -> None:
    assert is_mcp_connection(_conn("c", ctype="MYSQL", url="mcp://x")) is False


# ---------------------------------------------------------------------------
# discover_mcp_connections
# ---------------------------------------------------------------------------


class _FakeConnectionsApi:
    def __init__(self, conns: list[Any], *, raises: bool = False) -> None:
        self._conns = conns
        self._raises = raises

    def list(self) -> list[Any]:
        if self._raises:
            raise RuntimeError("listing failed")
        return self._conns


class _FakeClient:
    def __init__(self, conns: list[Any], *, raises: bool = False) -> None:
        self.connections = _FakeConnectionsApi(conns, raises=raises)


def test_discover_returns_only_mcp_connections() -> None:
    client = _FakeClient(
        [
            _conn("weather", options={"mcp_server_url": "https://x"}, comment="c"),
            _conn("plain_db", ctype="POSTGRESQL"),
            _conn("nameless", options={"mcp": "1"}),  # has marker; keep
        ]
    )
    servers = discover_mcp_connections(client)
    names = {s.name for s in servers}
    assert "weather" in names
    assert "plain_db" not in names
    weather = next(s for s in servers if s.name == "weather")
    assert weather.client_kind == "databricks"
    assert weather.connection_name == "weather"
    assert weather.description == "c"


def test_discover_none_client_returns_empty() -> None:
    assert discover_mcp_connections(None) == []


def test_discover_failsoft_on_listing_error() -> None:
    assert discover_mcp_connections(_FakeClient([], raises=True)) == []


def test_managed_catalog_lists_known_kinds() -> None:
    kinds = {entry["kind"] for entry in managed_mcp_catalog()}
    assert kinds == {"functions", "vector-search", "genie"}
