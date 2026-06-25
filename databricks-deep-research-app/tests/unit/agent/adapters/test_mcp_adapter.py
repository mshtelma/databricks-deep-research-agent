"""App-side MCP toolset injection adapter (spec §4.3).

Covers the runtime-override wiring that mirrors the per-agent ``domain_filter``
precedent:

* ``build_mcp_toolsets`` builds one toolset per server (with the OBO client),
  using a STUB client so there is NO network and the SSRF validator is skipped.
* An SSRF-rejected server is SKIPPED (graceful), not fatal to the request.
* Secrets are read via the workspace client's ``secrets.get_secret`` (base64
  decoded) and NEVER logged.
* The default path (no ``mcp_servers``) returns an empty list (byte-identical).

The framework ``MCPToolset`` itself is built with ``client=`` injected, so these
tests do not hit ``mcp`` SDK code paths.
"""
from __future__ import annotations

import base64
import logging
from typing import Any

import pytest
from databricks_deep_research import MCPServerConfig

from deep_research.agent.adapters.mcp_adapter import (
    _make_secret_resolver,
    _parse_secret_ref,
    build_mcp_toolsets,
)

# ---------------------------------------------------------------------------
# Fakes (no network, no real WorkspaceClient)
# ---------------------------------------------------------------------------


class _FakeMCPClient:
    def list_tools(self) -> list[Any]:
        return [
            type("T", (), {
                "name": "ask",
                "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}}},
                "description": "Ask.",
            })()
        ]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:  # pragma: no cover
        return type("R", (), {"content": [], "isError": False})()


class _FakeSecretsAPI:
    def __init__(self, value_b64: str) -> None:
        self._value_b64 = value_b64
        self.calls: list[tuple[str, str]] = []

    def get_secret(self, scope: str, key: str) -> Any:
        self.calls.append((scope, key))
        return type("Resp", (), {"key": key, "value": self._value_b64})()


class _FakeWorkspaceClient:
    """Minimal stand-in. ``config.host`` lets resolve_workspace_client run."""

    def __init__(self, secret_value: str = "tok-123") -> None:
        self.secrets = _FakeSecretsAPI(base64.b64encode(secret_value.encode()).decode())
        self.config = type("C", (), {"host": "https://example.databricks.com"})()


# ---------------------------------------------------------------------------
# secret_ref parsing + resolution
# ---------------------------------------------------------------------------


def test_parse_secret_ref_scope_key() -> None:
    assert _parse_secret_ref("my_scope/my_key") == ("my_scope", "my_key")


def test_parse_secret_ref_templated() -> None:
    assert _parse_secret_ref("{{secrets/my_scope/my_key}}") == ("my_scope", "my_key")


def test_parse_secret_ref_malformed() -> None:
    with pytest.raises(ValueError, match="scope/key"):
        _parse_secret_ref("just_a_key")


def test_secret_resolver_decodes_base64() -> None:
    wc = _FakeWorkspaceClient(secret_value="super-secret")
    resolve = _make_secret_resolver(wc)
    assert resolve("scope/key") == "super-secret"
    assert wc.secrets.calls == [("scope", "key")]


def test_secret_resolver_without_client_raises() -> None:
    resolve = _make_secret_resolver(None)
    with pytest.raises(ValueError, match="no Databricks workspace client"):
        resolve("scope/key")


# ---------------------------------------------------------------------------
# build_mcp_toolsets
# ---------------------------------------------------------------------------


def test_empty_servers_returns_empty() -> None:
    """Default path: no mcp_servers => no toolsets, no client touched."""
    assert build_mcp_toolsets([], sp_client=None, user_token=None) == []


def test_builds_toolset_with_obo_and_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """A server is built into a toolset using an injected stub client (no net)."""
    # Force build_mcp_toolset to use our stub client (skips SSRF + the mcp SDK).
    import deep_research.agent.adapters.mcp_adapter as mod

    captured: dict[str, Any] = {}
    real_build = mod.build_mcp_toolset

    def _fake_build(config: MCPServerConfig, *, secret_resolver: Any = None) -> Any:
        captured["config_name"] = config.name
        captured["has_resolver"] = secret_resolver is not None
        return real_build(config, secret_resolver=secret_resolver, client=_FakeMCPClient())

    monkeypatch.setattr(mod, "build_mcp_toolset", _fake_build)

    servers = [MCPServerConfig(name="corp", url="https://mcp.example.com/sse", name_prefix="corp_")]
    toolsets = build_mcp_toolsets(
        servers,
        sp_client=_FakeWorkspaceClient(),
        user_token="user-obo-token",
    )
    assert len(toolsets) == 1
    assert toolsets[0].tools[0].definition.name == "corp_ask"
    assert captured["config_name"] == "corp"
    assert captured["has_resolver"] is True


def test_ssrf_rejected_server_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A server whose URL fails SSRF validation is skipped, not fatal."""
    import deep_research.agent.adapters.mcp_adapter as mod

    real_build = mod.build_mcp_toolset

    def _fake_build(config: MCPServerConfig, *, secret_resolver: Any = None) -> Any:
        if config.name == "good":
            return real_build(config, secret_resolver=secret_resolver, client=_FakeMCPClient())
        # 'bad' server: no client => real SSRF validation runs on a loopback URL.
        return real_build(config, secret_resolver=secret_resolver)

    monkeypatch.setattr(mod, "build_mcp_toolset", _fake_build)

    servers = [
        MCPServerConfig(name="bad", url="http://127.0.0.1/sse"),
        MCPServerConfig(name="good", url="https://mcp.example.com/sse", name_prefix="g_"),
    ]
    toolsets = build_mcp_toolsets(servers, sp_client=_FakeWorkspaceClient(), user_token=None)
    # Only the good server survived.
    assert len(toolsets) == 1
    assert toolsets[0].tools[0].definition.name == "g_ask"


def test_secret_never_logged(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """An auth secret resolved during build must never appear in logs."""
    import deep_research.agent.adapters.mcp_adapter as mod

    secret = "ultra-secret-bearer-xyz"
    real_build = mod.build_mcp_toolset

    def _fake_build(config: MCPServerConfig, *, secret_resolver: Any = None) -> Any:
        return real_build(config, secret_resolver=secret_resolver, client=_FakeMCPClient())

    monkeypatch.setattr(mod, "build_mcp_toolset", _fake_build)
    # OBO resolution would build a REAL WorkspaceClient from the token; pin it
    # to our fake so the secret read hits the in-test secrets API (no network).
    fake_wc = _FakeWorkspaceClient(secret_value=secret)
    monkeypatch.setattr(mod, "resolve_workspace_client", lambda **_kw: fake_wc)

    servers = [
        MCPServerConfig(
            name="corp",
            url="https://mcp.example.com/sse",
            auth_type="bearer",
            secret_ref="scope/key",
            name_prefix="corp_",
        )
    ]
    with caplog.at_level(logging.DEBUG):
        toolsets = build_mcp_toolsets(
            servers,
            sp_client=fake_wc,
            user_token="obo",
        )
    assert len(toolsets) == 1
    combined = " ".join(rec.getMessage() for rec in caplog.records)
    assert secret not in combined
    assert "Bearer" not in combined
