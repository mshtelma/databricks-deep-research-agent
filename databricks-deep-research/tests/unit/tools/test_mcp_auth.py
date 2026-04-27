"""MCPAuth strategies — header threading + no credential leakage."""

from __future__ import annotations

import pytest

from databricks_deep_research.api import (
    ApiKey,
    BearerToken,
    CustomHeaders,
    MCPAuth,
)
from databricks_deep_research.tools.mcp import MCPToolset
from databricks_deep_research.tools.protocol import ToolContext


class _CapturingClient:
    """Fake client that records the tools/calls it sees but never reveals headers."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def list_tools(self):  # type: ignore[no-untyped-def]
        return [type("T", (), {"name": "t1", "inputSchema": {"type": "object", "properties": {}}, "description": ""})()]

    def call_tool(self, name, arguments):  # type: ignore[no-untyped-def]
        self.calls.append((name, dict(arguments)))
        return type("R", (), {"content": [], "isError": False})()


def test_bearer_token_yields_authorization_header() -> None:
    auth = BearerToken(token="secret-xyz")
    headers = auth.headers()
    assert headers == {"Authorization": "Bearer secret-xyz"}


def test_apikey_yields_named_header() -> None:
    auth = ApiKey(header="X-API-Key", value="abc123")
    headers = auth.headers()
    assert headers == {"X-API-Key": "abc123"}


def test_custom_headers_pass_through() -> None:
    auth = CustomHeaders(headers_dict={"X-Tenant": "acme", "X-Region": "us-east"})
    headers = auth.headers()
    assert headers["X-Tenant"] == "acme"
    assert headers["X-Region"] == "us-east"


def test_mcpauth_base_is_abstract() -> None:
    with pytest.raises(NotImplementedError):
        MCPAuth().headers()


@pytest.mark.asyncio
async def test_credentials_never_leak_into_tool_context() -> None:
    """Tool execution receives a ToolContext that has no auth in extras."""
    fake = _CapturingClient()
    ts = MCPToolset(client=fake, auth=BearerToken(token="leak-canary-xyz"))
    tool = ts.tools[0]
    ctx = ToolContext()
    await tool.execute({"q": "hello"}, ctx)
    assert "leak-canary-xyz" not in str(ctx.extras)
    for v in ctx.extras.values():
        assert "leak-canary-xyz" not in str(v)


def test_separate_bearer_tokens_dont_share_state() -> None:
    a = BearerToken(token="A")
    b = BearerToken(token="B")
    assert a.headers() != b.headers()
    a.token = "X"
    assert b.headers()["Authorization"] == "Bearer B"
