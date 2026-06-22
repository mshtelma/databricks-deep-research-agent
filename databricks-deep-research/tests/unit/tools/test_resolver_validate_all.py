"""Layer-3 pre-execution guard tests for ``ToolResolver.validate_all``.

These tests pin the contract that ``validate_all`` is **stricter** than
``initialize`` — it raises a single ``ValueError`` listing every tool whose
factory cannot construct it, instead of swallowing failures with a log line.

Anti-regression for the silent missing-config defect (``STORAGE_WAREHOUSE_ID``
unset → ``schema_cache`` is None → ``table_search`` factory raises mid-stream
as a misleading ``WorkflowError: missing declared tools``).
"""

from __future__ import annotations

import pytest

from databricks_deep_research.tools.factories.builtin import BuiltinToolFactory
from databricks_deep_research.tools.factory import ToolFactoryContext
from databricks_deep_research.tools.resolver import ToolResolver
from databricks_deep_research.workflow.definition import ToolDeclaration


@pytest.mark.asyncio
async def test_validate_all_raises_listing_each_unsatisfiable_tool() -> None:
    """Empty ToolFactoryContext + table_search declaration → single raise."""
    resolver = ToolResolver(
        declarations=[
            ToolDeclaration(name="t_search", kind="table_search", config={}),
            ToolDeclaration(name="t_read", kind="table_read", config={}),
        ],
        factories=[BuiltinToolFactory()],
        factory_context=ToolFactoryContext(),
    )

    with pytest.raises(ValueError) as excinfo:
        await resolver.validate_all()

    msg = str(excinfo.value)
    # Both unsatisfiable tools surface in a single error, not just the first one.
    assert "t_search" in msg
    assert "t_read" in msg
    assert "table_search" in msg
    assert "table_read" in msg


@pytest.mark.asyncio
async def test_validate_all_passes_when_no_declarations() -> None:
    """An empty resolver is trivially valid — no tools, no failures."""
    resolver = ToolResolver(
        declarations=[],
        factories=[BuiltinToolFactory()],
        factory_context=ToolFactoryContext(),
    )
    await resolver.validate_all()  # must not raise


@pytest.mark.asyncio
async def test_validate_all_skips_overrides() -> None:
    """Overridden tools bypass factory construction — must not be re-validated."""
    resolver = ToolResolver(
        declarations=[
            ToolDeclaration(name="t_search", kind="table_search", config={}),
        ],
        factories=[BuiltinToolFactory()],
        factory_context=ToolFactoryContext(),
    )

    class _StubTool:
        @property
        def definition(self):  # type: ignore[no-untyped-def]
            raise NotImplementedError

        def validate_arguments(self, arguments):  # type: ignore[no-untyped-def]
            return arguments

        async def execute(self, arguments, context):  # type: ignore[no-untyped-def]
            raise NotImplementedError

    resolver.override("t_search", _StubTool())  # type: ignore[arg-type]
    await resolver.validate_all()  # must not raise — override wins


@pytest.mark.asyncio
async def test_validate_all_message_contains_factory_error_text() -> None:
    """The raised message should embed the factory's own error so operators
    see *why* the tool could not be built (e.g. ``schema_cache required``)."""
    resolver = ToolResolver(
        declarations=[
            ToolDeclaration(name="t_search", kind="table_search", config={}),
        ],
        factories=[BuiltinToolFactory()],
        factory_context=ToolFactoryContext(),
    )

    with pytest.raises(ValueError) as excinfo:
        await resolver.validate_all()

    msg = str(excinfo.value)
    # Factory inline guard text or one of the required ctx field names must appear.
    assert any(
        token in msg
        for token in ("schema_cache", "sql_executor", "table_registry")
    )
