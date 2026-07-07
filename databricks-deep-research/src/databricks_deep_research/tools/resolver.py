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

import dataclasses
import logging
from typing import Any

from databricks_deep_research.tools.factory import ToolFactory, ToolFactoryContext
from databricks_deep_research.tools.protocol import ResearchTool, ToolRef
from databricks_deep_research.tools.registry import ToolRegistry
from databricks_deep_research.workflow.definition import ToolDeclaration

logger = logging.getLogger(__name__)

_EXTERNAL_DECLARATION_FIELDS: dict[str, str] = {
    "uc_function": "function_name",
    "uc_tool": "tool_name",
    "enterprise": "tool_name",
}


def _external_ref_for_declaration(decl: ToolDeclaration) -> ToolRef | None:
    """Map a declaration-backed external tool to its runtime registry ref."""
    field = _EXTERNAL_DECLARATION_FIELDS.get(decl.kind)
    if field is None:
        return None

    raw_name = decl.config.get(field)
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise ValueError(
            f"External tool declaration {decl.name!r} (kind={decl.kind!r}) requires config.{field}"
        )
    return ToolRef(type=decl.kind, name=raw_name.strip())


class ToolResolver:
    """Resolves tool references (name strings or legacy dicts) to ResearchTool instances."""

    def __init__(
        self,
        declarations: list[ToolDeclaration] | None = None,
        factories: list[ToolFactory] | None = None,
        factory_context: ToolFactoryContext | None = None,
        legacy_registry: ToolRegistry | None = None,
    ) -> None:
        self._declarations: dict[str, ToolDeclaration] = {d.name: d for d in (declarations or [])}
        self._factories: list[ToolFactory] = list(factories or [])
        # Shallow-copy the context so each resolver gets its own extras dict.
        # Without this, concurrent resolvers sharing a ToolFactoryContext
        # overwrite each other's _resolver_cache reference, causing
        # cross-question namespace leaks (e.g. TableLoadTool injects
        # variables into the wrong PythonComputeTool instance).
        base_ctx = factory_context or ToolFactoryContext()
        self._context: ToolFactoryContext = dataclasses.replace(
            base_ctx,
            extras={**base_ctx.extras},
        )
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
            external_ref = _external_ref_for_declaration(decl)
            if external_ref is not None:
                if self._legacy is None:
                    raise ValueError(
                        f"External tool declaration {name!r} (kind={decl.kind!r}) "
                        "requires a legacy/external ToolRegistry"
                    )
                tool = self._legacy.resolve(external_ref)
                self._cache[name] = tool
                return tool

            factory_errors: list[str] = []
            for factory in self._factories:
                if factory.supports(decl.kind):
                    try:
                        tool = await factory.create(decl, self._context)
                    except Exception as exc:
                        factory_errors.append(f"{type(factory).__name__}: {exc}")
                        continue
                    self._cache[name] = tool
                    return tool

            if self._legacy is not None and self._legacy.has(name):
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
                tr = ToolRef(type=ref.get("type", "builtin"), name=name)
                return self._legacy.resolve(tr)
            elif self._legacy.has(name):
                return self._legacy.resolve(ToolRef(type="builtin", name=name))

        raise ValueError(
            f"Tool not found: {name!r}. "
            f"Declared: {sorted(self._declarations)}, "
            f"Overrides: {sorted(self._overrides)}"
        )

    async def resolve_many(self, refs: list[str | dict[str, Any]]) -> list[ResearchTool]:
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
                    logger.warning("TOOL_INIT_FAILED name=%s kind=%s", name, decl.kind)

    async def validate_all(self) -> None:
        """Eagerly resolve every declared tool, raising on the first failure batch.

        Stricter sibling of :meth:`initialize`. Whereas ``initialize`` swallows
        per-tool resolution failures and emits a warning, ``validate_all``
        collects them and raises a single ``ValueError`` listing every
        unsatisfiable declaration. Intended as a pre-execution guard so a
        misconfigured workflow fails before any LLM tokens are spent on
        planning.

        Tools already satisfied by an override or by a previous resolution
        (cache hit) are skipped — they are known-good.
        """
        failures: list[str] = []
        for name, decl in self._declarations.items():
            if name in self._overrides or name in self._cache:
                continue
            try:
                await self.resolve(name)
            except ValueError as exc:
                failures.append(f"{name} (kind={decl.kind}): {exc}")
        if failures:
            raise ValueError(
                "Workflow declares tools that cannot be constructed:\n  " + "\n  ".join(failures)
            )

    def list_available(self) -> list[str]:
        """Return all resolvable tool names."""
        names: set[str] = set(self._overrides) | set(self._declarations) | set(self._cache)
        if self._legacy:
            names |= set(self._legacy.get_all_builtins())
        return sorted(names)

    def get_declaration(self, name: str) -> ToolDeclaration | None:
        """Return the original YAML declaration for a named tool, if any."""
        return self._declarations.get(name)

    @property
    def factory_context(self) -> ToolFactoryContext:
        """Return the resolver-local factory context.

        The resolver owns a shallow copy of the caller-provided context so
        factory extras, including the per-resolver cache, cannot leak across
        concurrent workflow runs.  Read-only consumers such as the workflow
        executor use this accessor to wire runtime integrations after all tools
        for a node have been resolved.
        """
        return self._context
