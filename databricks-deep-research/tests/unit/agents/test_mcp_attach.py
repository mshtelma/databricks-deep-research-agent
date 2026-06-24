"""Tests for MCP tool auto-attach (executor wiring helper, Feature 4.3).

Mirrors ``tests/unit/skills/test_skill_attach.py``. These guard the wire that
makes an MCP server's discovered tools callable by the agents that bind it via
``config.mcp_servers`` — the step whose absence left injected MCP tools
orphaned in the resolver (servers attached, ``tools=0`` reaching every agent).
"""

from __future__ import annotations

from types import SimpleNamespace

from databricks_deep_research.agents.mcp_attach import maybe_attach_mcp


def _tool(name: str) -> SimpleNamespace:
    """Minimal duck-typed ResearchTool: only ``.definition.name`` is read."""
    return SimpleNamespace(definition=SimpleNamespace(name=name))


def _ctx(by_server: dict[str, list]) -> SimpleNamespace:
    return SimpleNamespace(extras={"_mcp_tools_by_server": by_server})


def test_attaches_bound_server_tools() -> None:
    tools: list = []
    n = maybe_attach_mcp(
        tools, ["tavily_mcp"], _ctx({"tavily_mcp": [_tool("tavily_search")]})
    )
    assert n == 1
    assert [t.definition.name for t in tools] == ["tavily_search"]


def test_only_bound_servers_attached() -> None:
    tools: list = []
    ctx = _ctx(
        {"tavily_mcp": [_tool("tavily_search")], "other": [_tool("other_tool")]}
    )
    n = maybe_attach_mcp(tools, ["tavily_mcp"], ctx)
    assert n == 1
    assert [t.definition.name for t in tools] == ["tavily_search"]


def test_noop_without_servers() -> None:
    tools: list = []
    assert maybe_attach_mcp(tools, [], _ctx({"tavily_mcp": [_tool("x")]})) == 0
    assert tools == []


def test_noop_without_map() -> None:
    # No map wired (e.g. mcp/databricks-mcp not installed) or no factory context.
    tools: list = []
    assert maybe_attach_mcp(tools, ["tavily_mcp"], SimpleNamespace(extras={})) == 0
    assert maybe_attach_mcp(tools, ["tavily_mcp"], None) == 0
    assert tools == []


def test_dedup_existing_tool() -> None:
    tools: list = [_tool("tavily_search")]
    n = maybe_attach_mcp(
        tools, ["tavily_mcp"], _ctx({"tavily_mcp": [_tool("tavily_search")]})
    )
    assert n == 0
    assert len(tools) == 1


def test_multiple_servers_and_tools() -> None:
    tools: list = []
    ctx = _ctx({"s1": [_tool("a"), _tool("b")], "s2": [_tool("c")]})
    n = maybe_attach_mcp(tools, ["s1", "s2"], ctx)
    assert n == 3
    assert {t.definition.name for t in tools} == {"a", "b", "c"}


def test_unknown_server_is_skipped() -> None:
    tools: list = []
    # Agent binds a server that produced no toolset (e.g. it failed to build).
    n = maybe_attach_mcp(tools, ["missing"], _ctx({"tavily_mcp": [_tool("x")]}))
    assert n == 0
    assert tools == []
