"""MCP image / resource content normalization tests."""

from __future__ import annotations

import pytest

from databricks_deep_research.tools.mcp import MCPToolset
from databricks_deep_research.tools.protocol import ToolContext


class _Part:
    def __init__(self, **kwargs: object) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class _Result:
    def __init__(self, content: list[_Part]) -> None:
        self.content = content
        self.isError = False  # noqa: N815


class _ClientWithMixedContent:
    def __init__(self) -> None:
        pass

    def list_tools(self):  # type: ignore[no-untyped-def]
        return [type("T", (), {
            "name": "draw",
            "inputSchema": {"type": "object", "properties": {}},
            "description": "Draw something.",
        })()]

    def call_tool(self, name, arguments):  # type: ignore[no-untyped-def]
        return _Result([
            _Part(type="text", text="here is your result"),
            _Part(type="image", mimeType="image/png", data=b"\x89PNG\r\n" * 100),
            _Part(type="resource", uri="resource://x", resource={"key": "value"}),
        ])


@pytest.mark.asyncio
async def test_image_content_creates_attachment_marker() -> None:
    ts = MCPToolset(client=_ClientWithMixedContent())
    tool = ts.tools[0]
    result = await tool.execute({}, ToolContext())
    assert "[image attached: image/png," in result.content
    assert "here is your result" in result.content


@pytest.mark.asyncio
async def test_image_data_preserved_in_attachments() -> None:
    ts = MCPToolset(client=_ClientWithMixedContent())
    tool = ts.tools[0]
    result = await tool.execute({}, ToolContext())
    attachments = result.data.get("mcp_attachments", [])
    images = [a for a in attachments if a.get("kind") == "image"]
    assert len(images) == 1
    assert images[0]["mime_type"] == "image/png"
    assert images[0]["bytes"] is not None


@pytest.mark.asyncio
async def test_resource_attachment_preserved() -> None:
    ts = MCPToolset(client=_ClientWithMixedContent())
    tool = ts.tools[0]
    result = await tool.execute({}, ToolContext())
    attachments = result.data.get("mcp_attachments", [])
    resources = [a for a in attachments if a.get("kind") == "resource"]
    assert len(resources) == 1
    assert resources[0]["uri"] == "resource://x"
    assert resources[0]["data"] == {"key": "value"}


@pytest.mark.asyncio
async def test_text_only_response_no_attachments() -> None:
    class _TextOnly:
        def list_tools(self):  # type: ignore[no-untyped-def]
            return [type("T", (), {"name": "t", "inputSchema": {"type": "object", "properties": {}}, "description": ""})()]
        def call_tool(self, name, arguments):  # type: ignore[no-untyped-def]
            return _Result([_Part(type="text", text="just text")])

    ts = MCPToolset(client=_TextOnly())
    tool = ts.tools[0]
    result = await tool.execute({}, ToolContext())
    assert result.content == "just text"
    assert "mcp_attachments" not in result.data
