"""Regression tests: concurrent ToolResolvers must not share _resolver_cache.

Reproduces the bug where multiple ToolResolver instances sharing a single
ToolFactoryContext would overwrite each other's ``_resolver_cache`` in
``extras``, causing ``DeltaTableReadTool`` to inject variables into the
wrong ``PythonComputeTool`` instance.

See: resolver.py — ``dataclasses.replace(base_ctx, extras=...)`` fix.
"""

from __future__ import annotations

import pytest

from databricks_deep_research.tools.builtins.compute import PythonComputeTool
from databricks_deep_research.tools.builtins.compute_namespace import (
    ComputeNamespaceListTool,
)
from databricks_deep_research.tools.factory import ToolFactoryContext
from databricks_deep_research.tools.factories.builtin import BuiltinToolFactory
from databricks_deep_research.tools.protocol import ToolContext
from databricks_deep_research.tools.resolver import ToolResolver
from databricks_deep_research.workflow.definition import ToolDeclaration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx() -> ToolContext:
    return ToolContext(query="test")


def _make_resolver(
    shared_ctx: ToolFactoryContext,
    decls: list[ToolDeclaration] | None = None,
) -> ToolResolver:
    """Build a ToolResolver with the BuiltinToolFactory."""
    if decls is None:
        decls = [ToolDeclaration(name="compute", kind="compute", config={})]
    return ToolResolver(
        declarations=decls,
        factories=[BuiltinToolFactory()],
        factory_context=shared_ctx,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestResolverCacheIsolation:
    """Each ToolResolver must get its own _resolver_cache, even with shared context."""

    @pytest.mark.asyncio
    async def test_two_resolvers_have_independent_caches(self) -> None:
        """Two resolvers from same ToolFactoryContext must not share _resolver_cache."""
        shared_ctx = ToolFactoryContext()

        resolver_a = _make_resolver(shared_ctx)
        resolver_b = _make_resolver(shared_ctx)

        tool_a = await resolver_a.resolve("compute")
        tool_b = await resolver_b.resolve("compute")

        # Different tool instances
        assert tool_a is not tool_b

        # Different cache dicts
        cache_a = resolver_a._context.extras["_resolver_cache"]
        cache_b = resolver_b._context.extras["_resolver_cache"]
        assert cache_a is not cache_b

        # Each cache references the correct tool
        assert cache_a.get("compute") is tool_a
        assert cache_b.get("compute") is tool_b

    @pytest.mark.asyncio
    async def test_shared_context_extras_not_mutated(self) -> None:
        """The original shared context's extras dict must not be mutated."""
        shared_ctx = ToolFactoryContext(extras={"app_key": "value"})
        original_extras = shared_ctx.extras
        original_id = id(original_extras)

        _ = _make_resolver(shared_ctx)

        # Original dict object unchanged
        assert id(shared_ctx.extras) == original_id
        # _resolver_cache NOT injected into the original
        assert "_resolver_cache" not in original_extras
        # App-provided key survives
        assert original_extras["app_key"] == "value"

    @pytest.mark.asyncio
    async def test_inject_variable_isolated_across_resolvers(self) -> None:
        """inject_variable on resolver A's compute must not leak to resolver B."""
        shared_ctx = ToolFactoryContext()
        decls = [
            ToolDeclaration(name="compute", kind="compute", config={}),
            ToolDeclaration(
                name="ns",
                kind="compute_namespace",
                config={"compute_tool_name": "compute"},
            ),
        ]

        resolver_a = _make_resolver(shared_ctx, decls)
        resolver_b = _make_resolver(shared_ctx, decls)

        # Resolve compute + namespace in both
        compute_a = await resolver_a.resolve("compute")
        compute_b = await resolver_b.resolve("compute")
        ns_a = await resolver_a.resolve("ns")
        ns_b = await resolver_b.resolve("ns")

        assert isinstance(compute_a, PythonComputeTool)
        assert isinstance(compute_b, PythonComputeTool)
        assert isinstance(ns_a, ComputeNamespaceListTool)
        assert isinstance(ns_b, ComputeNamespaceListTool)

        # Inject variable ONLY into resolver A's compute tool
        compute_a.inject_variable("table", {"col": [1, 2, 3]})

        # Resolver A's namespace tool should see it
        result_a = await ns_a.execute({}, _ctx())
        assert "table" in result_a.content

        # Resolver B's namespace tool must NOT see it
        result_b = await ns_b.execute({}, _ctx())
        assert "table" not in result_b.content

    @pytest.mark.asyncio
    async def test_sequential_resolvers_dont_interfere(self) -> None:
        """Sequential resolver creation (concurrency=1) must still work correctly."""
        shared_ctx = ToolFactoryContext()
        decls = [
            ToolDeclaration(name="compute", kind="compute", config={}),
            ToolDeclaration(
                name="ns",
                kind="compute_namespace",
                config={"compute_tool_name": "compute"},
            ),
        ]

        # Create resolver A, use it, then discard
        resolver_a = _make_resolver(shared_ctx, decls)
        compute_a = await resolver_a.resolve("compute")
        ns_a = await resolver_a.resolve("ns")
        assert isinstance(compute_a, PythonComputeTool)
        compute_a.inject_variable("x", 42)
        result_a = await ns_a.execute({}, _ctx())
        assert "x" in result_a.content

        # Create resolver B with the same shared context
        resolver_b = _make_resolver(shared_ctx, decls)
        compute_b = await resolver_b.resolve("compute")
        ns_b = await resolver_b.resolve("ns")
        assert isinstance(compute_b, PythonComputeTool)

        # Resolver B should have a clean namespace
        result_b = await ns_b.execute({}, _ctx())
        assert "x" not in result_b.content

        # Resolver A should still see its variable
        result_a2 = await ns_a.execute({}, _ctx())
        assert "x" in result_a2.content
