"""Tests for the Designer discovery adapter's MCP + skill branches (C1)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from deep_research.agent_designer.discovery import DesignerDiscoveryAdapter


class _FakeDiscoveryResponse:
    def __init__(self, sources: list[Any]) -> None:
        self.sources = sources


class _FakeDiscoveryService:
    async def discover_all(
        self, user_id: str, user_token: str | None = None, **_: Any
    ) -> _FakeDiscoveryResponse:
        return _FakeDiscoveryResponse([])  # no Databricks resources for these tests


class _FakeConnectionsApi:
    def __init__(self, conns: list[Any]) -> None:
        self._conns = conns

    def list(self) -> list[Any]:
        return self._conns


class _FakeWsClient:
    def __init__(self, conns: list[Any]) -> None:
        self.connections = _FakeConnectionsApi(conns)


def _mcp_conn(name: str) -> Any:
    return SimpleNamespace(
        name=name,
        connection_type=SimpleNamespace(value="HTTP"),
        options={"mcp_server_url": "https://x"},
        properties={},
        url="",
        comment="a server",
    )


async def test_list_for_user_returns_mcp_servers() -> None:
    adapter = DesignerDiscoveryAdapter(
        _FakeDiscoveryService(),
        workspace_client_factory=lambda _t: _FakeWsClient([_mcp_conn("weather")]),
    )
    out = await adapter.list_for_user(user_token="t", user_id="u", kinds=["mcp_server"])
    assert len(out) == 1
    assert out[0].kind == "mcp_server"
    assert out[0].name == "weather"
    assert out[0].metadata["client_kind"] == "databricks"
    assert out[0].metadata["connection_name"] == "weather"


async def test_list_for_user_returns_skills(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_list_runtime_skills(**_: Any) -> list[Any]:
        return [
            SimpleNamespace(name="market-research", description="how to research"),
            SimpleNamespace(name="finance", description="finance methodology"),
        ]

    monkeypatch.setattr(
        "deep_research.services.skill_runtime.list_runtime_skills",
        _fake_list_runtime_skills,
    )
    adapter = DesignerDiscoveryAdapter(
        _FakeDiscoveryService(),
        workspace_client_factory=lambda _t: SimpleNamespace(),
    )
    out = await adapter.list_for_user(user_token="", user_id="u", kinds=["skill"])
    assert [r.name for r in out] == ["market-research", "finance"]
    assert all(r.kind == "skill" for r in out)


async def test_mcp_discovery_failsoft_when_only_kind() -> None:
    """A failing MCP listing for an explicit kind re-raises (caller sees it)."""

    def _boom(_t: str | None) -> Any:
        raise RuntimeError("connections api down")

    adapter = DesignerDiscoveryAdapter(
        _FakeDiscoveryService(), workspace_client_factory=_boom
    )
    with pytest.raises(RuntimeError):
        await adapter.list_for_user(user_token="t", user_id="u", kinds=["mcp_server"])
