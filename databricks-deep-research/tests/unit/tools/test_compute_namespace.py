"""Unit tests for ComputeNamespaceListTool."""

from __future__ import annotations

import types
from typing import Any

import pytest

from databricks_deep_research.tools.builtins.compute import PythonComputeTool
from databricks_deep_research.tools.builtins.compute_namespace import (
    ComputeNamespaceListTool,
)
from databricks_deep_research.tools.protocol import ToolContext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx() -> ToolContext:
    return ToolContext(query="test")


def _make_compute(**kwargs: Any) -> PythonComputeTool:
    return PythonComputeTool(**kwargs)


def _make_namespace_tool(
    compute_tool: PythonComputeTool | None = None,
    **kwargs: Any,
) -> ComputeNamespaceListTool:
    """Build a ComputeNamespaceListTool backed by *compute_tool* (or None)."""
    resolver = (lambda: compute_tool) if compute_tool is not None else (lambda: None)
    return ComputeNamespaceListTool(compute_resolver=resolver, **kwargs)


async def _store(tool: PythonComputeTool, code: str) -> None:
    """Execute code on *tool* to persist variables in its namespace."""
    args = tool.validate_arguments({"code": code})
    await tool.execute(args, _ctx())


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------


class TestToolDefinition:
    def test_tool_definition(self) -> None:
        """Verify name, parameters schema, and budget_free metadata."""
        tool = _make_namespace_tool()
        defn = tool.definition
        assert defn.name == "compute_namespace_list"
        assert "prefix" in defn.parameters["properties"]
        assert "names" in defn.parameters["properties"]
        assert "max_items" in defn.parameters["properties"]
        assert defn.metadata["budget_free"] is True

    def test_budget_free_metadata(self) -> None:
        """definition.metadata['budget_free'] is True."""
        tool = _make_namespace_tool()
        assert tool.definition.metadata["budget_free"] is True


# ---------------------------------------------------------------------------
# Empty / missing compute
# ---------------------------------------------------------------------------


class TestEmptyAndMissing:
    @pytest.mark.asyncio
    async def test_empty_namespace(self) -> None:
        """Returns 'Namespace is empty' (success, not error) when no variables stored."""
        compute = _make_compute()
        tool = _make_namespace_tool(compute)
        result = await tool.execute({}, _ctx())
        assert result.success is True
        assert "empty" in result.content.lower()

    @pytest.mark.asyncio
    async def test_no_compute_tool(self) -> None:
        """Resolver returns None -> 'No shared compute namespace available'."""
        tool = _make_namespace_tool(None)
        result = await tool.execute({}, _ctx())
        assert result.success is True
        assert "no shared compute namespace" in result.content.lower()


# ---------------------------------------------------------------------------
# Listing variables
# ---------------------------------------------------------------------------


class TestListVariables:
    @pytest.mark.asyncio
    async def test_lists_stored_variables(self) -> None:
        """Shared compute instance -> tool sees variables."""
        compute = _make_compute()
        await _store(compute, "op1 = 42")
        await _store(compute, "op2 = 99")

        tool = _make_namespace_tool(compute)
        result = await tool.execute({}, _ctx())

        assert result.success is True
        assert "op1" in result.content
        assert "op2" in result.content
        assert "42" in result.content
        assert "99" in result.content


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


class TestFiltering:
    @pytest.mark.asyncio
    async def test_prefix_filter(self) -> None:
        """prefix='op' filters correctly."""
        compute = _make_compute()
        await _store(compute, "op1 = 10")
        await _store(compute, "op2 = 20")
        await _store(compute, "total = 30")

        tool = _make_namespace_tool(compute)
        result = await tool.execute({"prefix": "op"}, _ctx())

        assert result.success is True
        assert "op1" in result.content
        assert "op2" in result.content
        assert "total" not in result.content

    @pytest.mark.asyncio
    async def test_names_filter(self) -> None:
        """names=['op1', 'op2'] returns exact matches."""
        compute = _make_compute()
        await _store(compute, "op1 = 10")
        await _store(compute, "op2 = 20")
        await _store(compute, "op3 = 30")

        tool = _make_namespace_tool(compute)
        result = await tool.execute({"names": ["op1", "op2"]}, _ctx())

        assert result.success is True
        assert "op1" in result.content
        assert "op2" in result.content
        assert "op3" not in result.content


# ---------------------------------------------------------------------------
# Unsafe type exclusion
# ---------------------------------------------------------------------------


class TestUnsafeTypes:
    @pytest.mark.asyncio
    async def test_excludes_unsafe_types(self) -> None:
        """Callables and modules are excluded from the namespace listing."""
        compute = _make_compute()
        await _store(compute, "safe_val = 42")

        # Inject a callable and a module directly into the namespace
        compute._namespace["my_func"] = lambda x: x
        compute._namespace["my_module"] = types

        tool = _make_namespace_tool(compute)
        result = await tool.execute({}, _ctx())

        assert result.success is True
        assert "safe_val" in result.content
        assert "my_func" not in result.content
        assert "my_module" not in result.content


# ---------------------------------------------------------------------------
# Shared instance semantics
# ---------------------------------------------------------------------------


class TestSharedInstance:
    @pytest.mark.asyncio
    async def test_shared_instance_semantics(self) -> None:
        """Same PythonComputeTool instance used by compute and namespace tool."""
        compute = _make_compute()
        await _store(compute, "x = 123")

        # Both tools share the same compute instance
        ns_tool = ComputeNamespaceListTool(compute_resolver=lambda: compute)
        result = await ns_tool.execute({}, _ctx())

        assert "x" in result.content
        assert "123" in result.content

        # Store another variable via the same compute tool
        await _store(compute, "y = 456")
        result2 = await ns_tool.execute({}, _ctx())

        assert "y" in result2.content
        assert "456" in result2.content


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_validate_arguments_passthrough(self) -> None:
        """validate_arguments returns args as-is."""
        tool = _make_namespace_tool()
        args = {"prefix": "op", "names": ["a"], "max_items": 10}
        assert tool.validate_arguments(args) == args
