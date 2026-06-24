"""Tests for the Databricks MCP adapter branch (B1).

Covers URL derivation (allowlisted, host-derived — never the persisted url),
OBO fail-closed, and the databricks client branch wiring.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from databricks_deep_research import MCPServerConfig

from deep_research.agent.adapters import mcp_adapter
from deep_research.agent.adapters.mcp_adapter import (
    MCPConfigError,
    _derive_databricks_mcp_url,
    _parse_secret_ref,
    build_mcp_toolsets,
)

_HOST = "https://w.cloud.databricks.com"


# ---------------------------------------------------------------------------
# URL derivation
# ---------------------------------------------------------------------------


def test_derive_external_connection_url() -> None:
    server = MCPServerConfig(
        name="ext", client_kind="databricks", connection_name="my_conn"
    )
    assert (
        _derive_databricks_mcp_url(_HOST, server)
        == f"{_HOST}/api/2.0/mcp/external/my_conn"
    )


@pytest.mark.parametrize(
    "target",
    ["functions/main/default", "vector-search/cat/sch", "genie/abc123"],
)
def test_derive_managed_targets(target: str) -> None:
    server = MCPServerConfig(
        name="mgd", client_kind="databricks", managed_target=target
    )
    assert _derive_databricks_mcp_url(_HOST, server) == f"{_HOST}/api/2.0/mcp/{target}"


def test_derive_ignores_persisted_url() -> None:
    # Even if a persisted url were set, derivation uses ONLY the trusted host.
    server = MCPServerConfig(
        name="ext",
        client_kind="databricks",
        connection_name="conn",
        url="https://evil.example.com/mcp",
    )
    assert _derive_databricks_mcp_url(_HOST, server).startswith(_HOST)
    assert "evil.example.com" not in _derive_databricks_mcp_url(_HOST, server)


@pytest.mark.parametrize(
    ("kwargs", "host"),
    [
        ({"connection_name": "bad-name!"}, _HOST),  # invalid identifier
        ({"managed_target": "functions/only_two"}, _HOST),  # wrong arity
        ({"managed_target": "secrets/list"}, _HOST),  # off-namespace
        ({"connection_name": "conn"}, ""),  # no host
    ],
)
def test_derive_rejects_invalid(kwargs: dict, host: str) -> None:
    server = MCPServerConfig(name="s", client_kind="databricks", **kwargs)
    with pytest.raises(MCPConfigError):
        _derive_databricks_mcp_url(host, server)


# ---------------------------------------------------------------------------
# build_mcp_toolsets — fail-closed + databricks branch
# ---------------------------------------------------------------------------


def test_managed_databricks_mcp_fails_closed_without_token() -> None:
    # Managed MCP (functions/vector-search/genie) runs strictly as the OBO user
    # (per-user UC data; valid Apps scopes), so it stays fail-closed when no OBO
    # token is available.
    server = MCPServerConfig(
        name="mgd", client_kind="databricks", managed_target="genie/abc123"
    )
    with pytest.raises(MCPConfigError):
        build_mcp_toolsets([server], sp_client=object(), user_token=None)


class _FakeToolset:
    def __len__(self) -> int:
        return 0


def _identity_aware_resolve(sp_sentinel: object, obo_sentinel: object):
    """A ``resolve_workspace_client`` fake returning a DISTINCT client per
    identity: the OBO sentinel when a token is present, else the SP sentinel.
    Lets a test assert which identity a server was built with."""

    def _resolve(*, sp_client: object, user_token: str | None) -> object:
        return obo_sentinel if user_token else sp_sentinel

    return _resolve


def test_external_databricks_built_as_service_principal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # EXTERNAL MCP must be built with the SP client EVEN WHEN an OBO token is
    # present: Databricks authorizes the external proxy via the SP's
    # USE_CONNECTION grant, and no user scope authorizes the OBO invoke.
    server = MCPServerConfig(
        name="ext", client_kind="databricks", connection_name="conn"
    )
    sp = SimpleNamespace(config=SimpleNamespace(host=_HOST), kind="sp")
    obo = SimpleNamespace(config=SimpleNamespace(host=_HOST), kind="obo")
    monkeypatch.setattr(
        mcp_adapter, "resolve_workspace_client", _identity_aware_resolve(sp, obo)
    )

    captured: dict[str, object] = {}

    def _fake_client(url: str, ws: object) -> str:
        captured["url"] = url
        captured["ws"] = ws
        return "FAKE_MCP_CLIENT"

    def _fake_build(cfg: MCPServerConfig, **kwargs: object) -> _FakeToolset:
        captured["client"] = kwargs.get("client")
        return _FakeToolset()

    monkeypatch.setattr(mcp_adapter, "_build_databricks_mcp_client", _fake_client)
    monkeypatch.setattr(mcp_adapter, "build_mcp_toolset", _fake_build)

    toolsets = build_mcp_toolsets([server], sp_client=sp, user_token="tok")

    assert len(toolsets) == 1
    assert captured["url"] == f"{_HOST}/api/2.0/mcp/external/conn"
    assert captured["client"] == "FAKE_MCP_CLIENT"
    assert captured["ws"] is sp  # SP identity, NOT obo


def test_managed_databricks_built_as_obo_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # MANAGED MCP must be built with the OBO user client.
    server = MCPServerConfig(
        name="mgd", client_kind="databricks", managed_target="genie/abc123"
    )
    sp = SimpleNamespace(config=SimpleNamespace(host=_HOST), kind="sp")
    obo = SimpleNamespace(config=SimpleNamespace(host=_HOST), kind="obo")
    monkeypatch.setattr(
        mcp_adapter, "resolve_workspace_client", _identity_aware_resolve(sp, obo)
    )

    captured: dict[str, object] = {}

    def _fake_client(url: str, ws: object) -> str:
        captured["url"] = url
        captured["ws"] = ws
        return "FAKE_MCP_CLIENT"

    monkeypatch.setattr(mcp_adapter, "_build_databricks_mcp_client", _fake_client)
    monkeypatch.setattr(
        mcp_adapter, "build_mcp_toolset", lambda cfg, **kw: _FakeToolset()
    )

    toolsets = build_mcp_toolsets([server], sp_client=sp, user_token="tok")

    assert len(toolsets) == 1
    assert captured["url"] == f"{_HOST}/api/2.0/mcp/genie/abc123"
    assert captured["ws"] is obo  # OBO user identity


def test_external_databricks_allowed_without_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # External (UC-connection) MCP is invoked as the app SP, so it does NOT
    # require an OBO token — it must build even when user_token is None.
    server = MCPServerConfig(
        name="ext", client_kind="databricks", connection_name="conn"
    )
    sp = SimpleNamespace(config=SimpleNamespace(host=_HOST))
    monkeypatch.setattr(mcp_adapter, "resolve_workspace_client", lambda **_: sp)
    monkeypatch.setattr(
        mcp_adapter, "_build_databricks_mcp_client", lambda url, ws: "C"
    )
    monkeypatch.setattr(
        mcp_adapter, "build_mcp_toolset", lambda cfg, **kw: _FakeToolset()
    )

    toolsets = build_mcp_toolsets([server], sp_client=sp, user_token=None)

    assert len(toolsets) == 1


def test_empty_servers_noop() -> None:
    assert build_mcp_toolsets([], sp_client=object(), user_token="tok") == []


# ---------------------------------------------------------------------------
# Secret-ref validation (security review HIGH-1)
# ---------------------------------------------------------------------------


def test_parse_secret_ref_accepts_valid() -> None:
    assert _parse_secret_ref("my-scope/my_key.v2") == ("my-scope", "my_key.v2")
    assert _parse_secret_ref("{{secrets/sc/k}}") == ("sc", "k")


@pytest.mark.parametrize(
    "ref",
    [
        "scope/key\nX-Inject: 1",  # newline header injection
        "sc/..%2Fetc",  # URL-encoded traversal
        "sc/key with space",  # whitespace
        "sc/k@y",  # disallowed char
        "../scope/key",  # traversal-ish (3 parts)
        "scope/" + "k" * 200,  # over length
    ],
)
def test_parse_secret_ref_rejects_malicious(ref: str) -> None:
    with pytest.raises(ValueError):
        _parse_secret_ref(ref)
