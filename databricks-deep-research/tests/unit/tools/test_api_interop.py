"""Interop tests: ``@tool`` callables satisfy the ResearchTool protocol."""

from __future__ import annotations

import asyncio

from databricks_deep_research.tools.api import _DecoratedTool, tool
from databricks_deep_research.tools.factories.decorated import DecoratedToolFactory
from databricks_deep_research.tools.factory import ToolFactoryContext
from databricks_deep_research.tools.protocol import (
    ResearchTool,
    ToolContext,
    ToolDefinition,
    ToolResult,
)
from databricks_deep_research.workflow.definition import ToolDeclaration


def test_decorated_isinstance_research_tool() -> None:
    @tool
    def f(x: str) -> str:
        """X"""
        return x

    assert isinstance(f, ResearchTool)
    assert isinstance(f, _DecoratedTool)
    assert isinstance(f.definition, ToolDefinition)


def test_decorated_returns_tool_result() -> None:
    @tool
    def f(x: str) -> str:
        """X"""
        return x

    result = asyncio.run(f.execute({"x": "hello"}, ToolContext()))
    assert isinstance(result, ToolResult)
    assert result.content == "hello"


def test_decorated_validates_arguments() -> None:
    @tool
    def f(x: int, y: int = 0) -> int:
        """X"""
        return x + y

    validated = f.validate_arguments({"x": 5})
    assert validated["x"] == 5
    assert validated["y"] == 0


def test_decorated_factory_resolves_callable() -> None:
    factory = DecoratedToolFactory()
    assert factory.supports("decorated") is True
    assert factory.supports("web_search") is False

    decl = ToolDeclaration(
        name="my_tool",
        kind="decorated",
        config={"import": "databricks_deep_research.tools.api:tool"},
    )
    # ``tool`` is a function — DecoratedToolFactory wraps it via tool()
    ctx = ToolFactoryContext()
    resolved = asyncio.run(factory.create(decl, ctx))
    assert isinstance(resolved, _DecoratedTool)
    assert resolved.definition.name == "my_tool"


def test_decorated_factory_rejects_non_callable() -> None:
    import databricks_deep_research.tools.api as api_mod

    api_mod.test_not_callable_attr = "string"
    factory = DecoratedToolFactory()
    decl = ToolDeclaration(
        name="bad",
        kind="decorated",
        config={"import": "databricks_deep_research.tools.api:test_not_callable_attr"},
    )
    ctx = ToolFactoryContext()
    try:
        asyncio.run(factory.create(decl, ctx))
        raise AssertionError("Should have raised")
    except ValueError:
        pass


def test_decorated_factory_missing_import() -> None:
    factory = DecoratedToolFactory()
    decl = ToolDeclaration(name="bad", kind="decorated", config={})
    try:
        asyncio.run(factory.create(decl, ToolFactoryContext()))
        raise AssertionError("Should have raised")
    except ValueError:
        pass


def test_decorated_supports_async_callables() -> None:
    @tool
    async def f(msg: str) -> str:
        """A"""
        return msg.upper()

    result = asyncio.run(f.execute({"msg": "hi"}, ToolContext()))
    assert result.content == "HI"
