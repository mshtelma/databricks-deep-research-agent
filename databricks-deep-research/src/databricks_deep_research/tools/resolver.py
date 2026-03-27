"""Tool resolver — resolves tool references to ResearchTool instances.

The resolver is the single entry point for tool resolution in the executor.
It handles both new-style name strings (from YAML ``tools:`` declarations)
and legacy ``{type, name}`` dicts for backward compatibility.

Resolution order:
    1. **Overrides** — app-injected tools (highest priority).
    2. **Cache** — previously resolved declarations.
    3. **Declarations** — created via factory chain from YAML ``tools:`` section.
    4. **Legacy fallback** — :class:`ToolRegistry` for old ``{type, name}`` dicts.
"""

from __future__ import annotations

import logging
from typing import Any

from databricks_deep_research.tools.factory import ToolFactory, ToolFactoryContext
from databricks_deep_research.tools.protocol import ResearchTool
from databricks_deep_research.tools.registry import ToolRegistry
from databricks_deep_research.workflow.definition import ToolDeclaration

logger = logging.getLogger(__name__)


class ToolResolver:
    """Resolves tool references (name strings or legacy dicts) to ResearchTool instances."""

    def __init__(
        self,
        declarations: list[ToolDeclaration] | None = None,
        factories: list[ToolFactory] | None = None,
        factory_context: ToolFactoryContext | None = None,
        legacy_registry: ToolRegistry | None = None,
    ) -> None:
        self._declarations: dict[str, ToolDeclaration] = {
            d.name: d for d in (declarations or [])
        }
        self._factories: list[ToolFactory] = list(factories or [])
        self._context: ToolFactoryContext = factory_context or ToolFactoryContext()
        self._legacy: ToolRegistry | None = legacy_registry
        self._overrides: dict[str, ResearchTool] = {}
        self._cache: dict[str, ResearchTool] = {}
        # Share cache ref so sibling-resolution closures (e.g. compute_namespace)
        # can look up tools created by other factories.
        self._context.extras["_resolver_cache"] = self._cache

    def override(self, name: str, tool: ResearchTool) -> None:
        """Register a runtime override (highest priority)."""
        self._overrides[name] = tool

    async def resolve(self, ref: str | dict[str, Any]) -> ResearchTool:
        """Resolve a tool name or legacy dict to a ResearchTool instance.

        Raises:
            ValueError: If the tool cannot be resolved.
        """
        name = ref if isinstance(ref, str) else ref.get("name", "")
        if not name:
            raise ValueError(f"Cannot resolve tool with empty name: {ref!r}")

        # 1. Override
        if name in self._overrides:
            return self._overrides[name]

        # 2. Cache (from previous declaration resolution)
        if name in self._cache:
            return self._cache[name]

        # 3. Declaration → factory chain
        if name in self._declarations:
            decl = self._declarations[name]
            factory_errors: list[str] = []
            for factory in self._factories:
                if factory.supports(decl.kind):
                    try:
                        tool = await factory.create(decl, self._context)
                    except Exception as exc:
                        factory_errors.append(
                            f"{type(factory).__name__}: {exc}"
                        )
                        continue
                    self._cache[name] = tool
                    return tool

            if self._legacy is not None and self._legacy.has(name):
                from databricks_deep_research.tools.protocol import ToolRef

                logger.info(
                    "TOOL_RESOLVER_FACTORY_FALLBACK tool=%s kind=%s errors=%s",
                    name,
                    decl.kind,
                    factory_errors,
                )
                return self._legacy.resolve(ToolRef(type="builtin", name=name))

            if factory_errors:
                raise ValueError(
                    f"Failed to create declared tool {name!r} (kind={decl.kind!r}). "
                    f"Factory errors: {factory_errors}"
                )

            raise ValueError(
                f"No factory supports kind={decl.kind!r} for tool {name!r}. "
                f"Registered factories: {[type(f).__name__ for f in self._factories]}"
            )

        # 4. Legacy ToolRegistry fallback (for {type, name} dicts)
        if self._legacy is not None:
            if isinstance(ref, dict):
                from databricks_deep_research.tools.protocol import ToolRef

                tr = ToolRef(
                    type=ref.get("type", "builtin"), name=name
                )
                return self._legacy.resolve(tr)
            elif self._legacy.has(name):
                from databricks_deep_research.tools.protocol import ToolRef

                return self._legacy.resolve(ToolRef(type="builtin", name=name))

        raise ValueError(
            f"Tool not found: {name!r}. "
            f"Declared: {sorted(self._declarations)}, "
            f"Overrides: {sorted(self._overrides)}"
        )

    async def resolve_many(
        self, refs: list[str | dict[str, Any]]
    ) -> list[ResearchTool]:
        """Resolve multiple refs.  Collects errors instead of failing on first."""
        tools: list[ResearchTool] = []
        errors: list[str] = []
        for ref in refs:
            try:
                tools.append(await self.resolve(ref))
            except ValueError as e:
                errors.append(str(e))
        if errors:
            logger.warning("TOOL_RESOLVE_ERRORS errors=%s", errors)
        return tools

    async def initialize(self) -> None:
        """Eagerly create all declared tools (optional pre-warming)."""
        for name, decl in self._declarations.items():
            if name not in self._cache and name not in self._overrides:
                try:
                    await self.resolve(name)
                except ValueError:
                    logger.warning(
                        "TOOL_INIT_FAILED name=%s kind=%s", name, decl.kind
                    )

    def list_available(self) -> list[str]:
        """Return all resolvable tool names."""
        names: set[str] = set(self._overrides) | set(self._declarations) | set(self._cache)
        if self._legacy:
            names |= set(self._legacy.get_all_builtins())
        return sorted(names)
