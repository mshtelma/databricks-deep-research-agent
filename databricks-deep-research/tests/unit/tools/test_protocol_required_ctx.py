"""Tests for the per-tool-kind required-ctx metadata table.

Anti-regression: the inline ``if ctx.X is None: raise`` blocks inside
``BuiltinToolFactory.create()`` (databricks_deep_research/tools/factories/builtin.py)
are the runtime source of truth. This metadata table mirrors them so app-level
deploy/boot validators can introspect requirements WITHOUT importing the
factory module. If a new ToolKind adds a required ctx field, BOTH the
factory ``if`` block AND this table must be updated; the test in
``test_required_ctx_matches_factory`` enforces the invariant for the
text-table family (the kinds whose missing context caused the original
incident this design was written to prevent).
"""

from __future__ import annotations

import dataclasses

from databricks_deep_research import required_ctx_fields_for_kind
from databricks_deep_research.tools.factory import ToolFactoryContext
from databricks_deep_research.tools.protocol import ToolKind


def test_table_kinds_require_three_text_table_fields() -> None:
    table_kinds = (
        ToolKind.table_search,
        ToolKind.table_read,
        ToolKind.table_neighbors,
        ToolKind.table_load,
        ToolKind.table_aggregate,
    )
    expected = frozenset({"table_registry", "schema_cache", "sql_executor"})
    for kind in table_kinds:
        assert required_ctx_fields_for_kind(kind) == expected, kind


def test_table_discovery_requires_registry_and_provider() -> None:
    assert required_ctx_fields_for_kind(ToolKind.table_discovery) == frozenset(
        {"table_registry", "table_discovery_provider"}
    )


def test_web_search_requires_search_client() -> None:
    assert required_ctx_fields_for_kind(ToolKind.web_search) == frozenset(
        {"search_client"}
    )
    assert required_ctx_fields_for_kind(ToolKind.web_research) == frozenset(
        {"search_client"}
    )


def test_databricks_kinds_require_workspace_client() -> None:
    for kind in (
        ToolKind.vector_search,
        ToolKind.genie,
        ToolKind.knowledge_assistant,
    ):
        assert required_ctx_fields_for_kind(kind) == frozenset(
            {"workspace_client"}
        ), kind


def test_kinds_without_required_ctx_return_empty_frozenset() -> None:
    for kind in (
        ToolKind.web_crawl,
        ToolKind.file_search,
        ToolKind.compute,
        ToolKind.compute_namespace,
        ToolKind.custom,
    ):
        assert required_ctx_fields_for_kind(kind) == frozenset(), kind


def test_unknown_kind_returns_empty_frozenset() -> None:
    assert required_ctx_fields_for_kind("not_a_real_kind") == frozenset()


def test_required_field_names_exist_on_factory_context() -> None:
    """Every field name in the table must be a real attribute on ToolFactoryContext.

    Catches typos like ``schema_caches`` (plural) or ``sqlexecutor`` (no
    underscore) that would silently make the validators always pass.
    """
    ctx_field_names = {f.name for f in dataclasses.fields(ToolFactoryContext)}
    for kind in ToolKind:
        for name in required_ctx_fields_for_kind(kind):
            assert name in ctx_field_names, (
                f"required_ctx_fields_for_kind({kind!r}) lists {name!r}, "
                f"which is not a field on ToolFactoryContext "
                f"(known fields: {sorted(ctx_field_names)})"
            )


def test_empty_context_blocks_text_table_kinds() -> None:
    """Realistic: an empty ToolFactoryContext can satisfy zero required fields."""
    ctx = ToolFactoryContext()
    for kind in (
        ToolKind.table_search,
        ToolKind.table_read,
        ToolKind.table_neighbors,
        ToolKind.table_load,
        ToolKind.table_aggregate,
    ):
        unsatisfied = {
            name
            for name in required_ctx_fields_for_kind(kind)
            if getattr(ctx, name) is None
        }
        assert unsatisfied == required_ctx_fields_for_kind(kind), kind
