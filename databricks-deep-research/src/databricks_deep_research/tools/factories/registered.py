"""Registered tool factory — dict-lookup resolution of host-curated tools.

``kind: registered`` is the SAFE way for stored (database-sourced) workflow
definitions to use host Python tools: the declaration carries only a ``key``
into a catalog the HOST built at startup from operator-trusted sources
(config entries, plugin providers). Resolution is a dict lookup — NEVER
importlib/getattr/eval on declaration data (the ``BUILTIN_FACTORIES``
discipline; contrast with ``kind: decorated``, whose import path executes
code and is prefix-gated).

::

    tools:
      - name: forecaster
        kind: registered
        config: {key: acme_plugins.prophet_forecast}
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from databricks_deep_research.tools.catalog_types import CatalogCard, SafeProbe
from databricks_deep_research.tools.factory import ToolFactoryContext
from databricks_deep_research.tools.protocol import (
    ResearchTool,
    ToolContext,
    ToolDefinition,
    ToolResult,
)
from databricks_deep_research.workflow.definition import ToolDeclaration

__all__ = ["RegisteredToolFactory"]


class _RenamedTool:
    """Delegating wrapper exposing a catalog tool under the declaration name.

    The resolver caches tools by DECLARATION name while the LLM sees
    ``definition.name`` — they must match, so a decl that renames a catalog
    entry gets this thin wrapper instead of a mutated shared instance.
    """

    def __init__(self, inner: ResearchTool, name: str) -> None:
        self._inner = inner
        self._name = name

    @property
    def definition(self) -> ToolDefinition:
        inner_def = self._inner.definition
        return ToolDefinition(
            name=self._name,
            description=inner_def.description,
            parameters=inner_def.parameters,
            source_type=inner_def.source_type,
            source_kind=inner_def.source_kind,
            metadata=inner_def.metadata,
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._inner.validate_arguments(arguments)

    async def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        return await self._inner.execute(arguments, context)


class RegisteredToolFactory:
    """Creates tools for ``kind: registered`` declarations by catalog lookup."""

    SUPPORTED_KIND = "registered"
    catalog_cards: ClassVar[Mapping[str, CatalogCard]] = {}
    safe_probes: ClassVar[Mapping[str, SafeProbe | None]] = {}

    def __init__(self, registry: Mapping[str, ResearchTool]) -> None:
        self._registry: dict[str, ResearchTool] = dict(registry)

    def supports(self, kind: str) -> bool:
        return kind == self.SUPPORTED_KIND

    async def create(
        self,
        decl: ToolDeclaration,
        ctx: ToolFactoryContext,  # noqa: ARG002 — required by protocol
    ) -> ResearchTool:
        key = decl.config.get("key")
        if not isinstance(key, str) or not key:
            raise ValueError(
                f"Registered tool {decl.name!r} requires config.key naming a "
                f"catalog entry. Available: {sorted(self._registry)}"
            )
        tool = self._registry.get(key)
        if tool is None:
            raise ValueError(
                f"Registered tool {decl.name!r}: unknown key {key!r}. "
                f"Available: {sorted(self._registry)}"
            )
        if decl.name and decl.name != tool.definition.name:
            return _RenamedTool(tool, decl.name)
        return tool
