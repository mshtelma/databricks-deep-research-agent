"""MCP toolset discovery + schema validation tests using a fake client."""

from __future__ import annotations

import pytest

from databricks_deep_research.tools.mcp import MCPSchemaError, MCPToolset


class _FakeMCPTool:
    def __init__(self, name: str, schema: dict, description: str = "") -> None:
        self.name = name
        self.inputSchema = schema  # noqa: N815 — match MCP spec field name
        self.description = description


class _FakeListResult:
    def __init__(self, tools: list) -> None:
        self.tools = tools


class _FakeClient:
    def __init__(self, tools: list) -> None:
        self._tools = tools

    def list_tools(self) -> _FakeListResult:
        return _FakeListResult(self._tools)

    def call_tool(self, name, arguments) -> dict:
        return {"name": name, "arguments": arguments}


def test_discovery_exposes_tools_by_name() -> None:
    fake = _FakeClient([
        _FakeMCPTool("search", {"type": "object", "properties": {"q": {"type": "string"}}}, "Search."),
        _FakeMCPTool("fetch", {"type": "object", "properties": {"url": {"type": "string"}}}, "Fetch."),
    ])
    ts = MCPToolset(client=fake)
    assert len(ts) == 2
    names = [t.definition.name for t in ts.tools]
    assert "search" in names
    assert "fetch" in names


def test_discovery_applies_allow_filter() -> None:
    fake = _FakeClient([
        _FakeMCPTool("a", {"type": "object", "properties": {}}),
        _FakeMCPTool("b", {"type": "object", "properties": {}}),
    ])
    ts = MCPToolset(client=fake, allow=["a"])
    names = [t.definition.name for t in ts.tools]
    assert names == ["a"]


def test_discovery_applies_deny_filter() -> None:
    fake = _FakeClient([
        _FakeMCPTool("a", {"type": "object", "properties": {}}),
        _FakeMCPTool("b", {"type": "object", "properties": {}}),
    ])
    ts = MCPToolset(client=fake, deny=["b"])
    names = [t.definition.name for t in ts.tools]
    assert names == ["a"]


def test_discovery_applies_name_prefix() -> None:
    fake = _FakeClient([_FakeMCPTool("search", {"type": "object", "properties": {}})])
    ts = MCPToolset(client=fake, name_prefix="brave_")
    assert ts.tools[0].definition.name == "brave_search"


def test_discovery_skips_invalid_oneof_root() -> None:
    fake = _FakeClient([
        _FakeMCPTool("invalid", {"oneOf": [{"type": "string"}, {"type": "number"}]}),
        _FakeMCPTool("valid", {"type": "object", "properties": {}}),
    ])
    ts = MCPToolset(client=fake)
    names = [t.definition.name for t in ts.tools]
    assert names == ["valid"]


def test_discovery_inlines_refs() -> None:
    fake = _FakeClient([
        _FakeMCPTool(
            "with_refs",
            {
                "type": "object",
                "properties": {"x": {"$ref": "#/$defs/X"}},
                "$defs": {"X": {"type": "string"}},
            },
        ),
    ])
    ts = MCPToolset(client=fake)
    schema = ts.tools[0].definition.parameters
    assert schema["properties"]["x"]["type"] == "string"
    assert "$defs" not in schema


def test_iteration_yields_tools() -> None:
    fake = _FakeClient([
        _FakeMCPTool("a", {"type": "object", "properties": {}}),
        _FakeMCPTool("b", {"type": "object", "properties": {}}),
    ])
    ts = MCPToolset(client=fake)
    names = [t.definition.name for t in ts]
    assert names == ["a", "b"]


def test_missing_url_and_client_raises() -> None:
    with pytest.raises(ValueError, match="url= or client="):
        MCPToolset()
