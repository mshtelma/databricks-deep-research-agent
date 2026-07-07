"""RegisteredToolFactory (catalog lookup) + DecoratedToolFactory import gate."""

from __future__ import annotations

from typing import Any

import pytest

from databricks_deep_research.tools.factories.decorated import DecoratedToolFactory
from databricks_deep_research.tools.factories.registered import RegisteredToolFactory
from databricks_deep_research.tools.factory import ToolFactoryContext
from databricks_deep_research.tools.protocol import (
    ToolContext,
    ToolDefinition,
    ToolResult,
)
from databricks_deep_research.workflow.definition import ToolDeclaration


class _StubTool:
    def __init__(self, name: str = "stub") -> None:
        self._name = name

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name, description="stub", parameters={"type": "object"}
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return dict(arguments)

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        return ToolResult(content="stub-ran", data=dict(arguments))


def sample_registered(x: int) -> int:
    """Doubles (import target for decorated gate tests)."""
    return x * 2


class TestRegisteredFactory:
    async def test_lookup_returns_catalog_instance(self) -> None:
        stub = _StubTool(name="forecaster")
        factory = RegisteredToolFactory({"acme.forecast": stub})
        decl = ToolDeclaration(
            name="forecaster", kind="registered", config={"key": "acme.forecast"}
        )
        assert await factory.create(decl, ToolFactoryContext()) is stub

    async def test_rename_wraps_without_mutating_shared_instance(self) -> None:
        stub = _StubTool(name="forecaster")
        factory = RegisteredToolFactory({"acme.forecast": stub})
        decl = ToolDeclaration(
            name="my_forecaster", kind="registered", config={"key": "acme.forecast"}
        )
        wrapped = await factory.create(decl, ToolFactoryContext())
        assert wrapped is not stub
        assert wrapped.definition.name == "my_forecaster"
        assert stub.definition.name == "forecaster"
        result = await wrapped.execute({"a": 1}, ToolContext())
        assert result.content == "stub-ran" and result.data == {"a": 1}

    async def test_unknown_key_lists_available(self) -> None:
        factory = RegisteredToolFactory({"acme.forecast": _StubTool()})
        decl = ToolDeclaration(name="x", kind="registered", config={"key": "ghost"})
        with pytest.raises(ValueError, match=r"unknown key 'ghost'.*acme\.forecast"):
            await factory.create(decl, ToolFactoryContext())

    async def test_missing_key_rejected(self) -> None:
        factory = RegisteredToolFactory({})
        decl = ToolDeclaration(name="x", kind="registered", config={})
        with pytest.raises(ValueError, match="requires config.key"):
            await factory.create(decl, ToolFactoryContext())


class TestDecoratedImportGate:
    def _decl(self) -> ToolDeclaration:
        return ToolDeclaration(
            name="doubler",
            kind="decorated",
            config={"import": "tests.test_registered_factory:sample_registered"},
        )

    async def test_deny_all_by_default_sequence(self) -> None:
        factory = DecoratedToolFactory(allowed_import_prefixes=())
        with pytest.raises(ValueError, match="not allowed on this host"):
            await factory.create(self._decl(), ToolFactoryContext())

    async def test_prefix_allows_package_subtree(self) -> None:
        factory = DecoratedToolFactory(allowed_import_prefixes=("tests",))
        tool = await factory.create(self._decl(), ToolFactoryContext())
        assert tool.definition.name == "doubler"

    async def test_exact_module_prefix_allows(self) -> None:
        factory = DecoratedToolFactory(
            allowed_import_prefixes=("tests.test_registered_factory",)
        )
        tool = await factory.create(self._decl(), ToolFactoryContext())
        assert tool.definition.name == "doubler"

    async def test_non_matching_prefix_denied(self) -> None:
        factory = DecoratedToolFactory(allowed_import_prefixes=("myapp",))
        with pytest.raises(ValueError, match="not allowed on this host"):
            await factory.create(self._decl(), ToolFactoryContext())

    async def test_prefix_is_not_a_string_prefix_match(self) -> None:
        # "tests" allowlisted must NOT allow "tests_evil" (dot-boundary match).
        factory = DecoratedToolFactory(allowed_import_prefixes=("test",))
        with pytest.raises(ValueError, match="not allowed on this host"):
            await factory.create(self._decl(), ToolFactoryContext())

    async def test_none_preserves_import_time_allow_all(self) -> None:
        factory = DecoratedToolFactory(allowed_import_prefixes=None)
        tool = await factory.create(self._decl(), ToolFactoryContext())
        assert tool.definition.name == "doubler"
