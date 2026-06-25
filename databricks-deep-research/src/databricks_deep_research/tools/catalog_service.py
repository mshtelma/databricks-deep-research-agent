"""Catalog service — aggregates per-factory metadata into a single render.

This module sits between the catalog-aware factory layer (which owns
:class:`CatalogCard` / :class:`SafeProbe` mappings) and the pure renderer
(which turns declarations + cards into a system-prompt block). Two paths
call into it:

* **Save-time materialization** (Designer pipeline) — the workflow builder
  asks the service to render a catalog for a researcher/planner agent and
  stashes the result inside ``AgentNodeConfig.extras`` so the persisted
  configuration carries the prompt block verbatim.
* **Runtime fallback** (harness) — when an agent runs without a persisted
  catalog (legacy YAML, registry version drift), the harness asks the
  service to render fresh from the current factory metadata.

Both paths call :func:`render_tool_catalog` so the persisted and live
prompt blocks remain byte-identical when the registry version matches.

Security
--------
The default factories list is built from
:data:`databricks_deep_research.tools.factories.BUILTIN_FACTORIES`. That
mapping is the framework's only authoritative factory allow-list — never
extend the catalog service from a dynamic source, and never resolve
factory metadata via ``importlib`` / ``getattr`` against user-supplied
strings. The :class:`CatalogProvider` Protocol is structural so test
fixtures can pass lightweight stubs without subclassing the production
factories.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable, Mapping
from typing import Any, Final

from databricks_deep_research.agents.config import AgentNodeConfig
from databricks_deep_research.tools.catalog_renderer import (
    REGISTRY_VERSION,
    CatalogConfig,
    CatalogRenderResult,
    render_tool_catalog,
)
from databricks_deep_research.tools.catalog_types import (
    CatalogCard,
    CatalogProvider,
    ProbeSample,
)
from databricks_deep_research.tools.protocol import ResearchTool
from databricks_deep_research.workflow.definition import ToolDeclaration

logger = logging.getLogger(__name__)

CATALOG_TEXT_EXTRA: Final[str] = "_framework_tool_catalog"
CATALOG_REGISTRY_VERSION_EXTRA: Final[str] = "_framework_tool_catalog_registry_version"
CATALOG_KINDS_EXTRA: Final[str] = "_framework_tool_catalog_kinds"
CATALOG_DECLS_HASH_EXTRA: Final[str] = "_framework_tool_catalog_decls_hash"
CATALOG_DECLARATIONS_EXTRA: Final[str] = "_framework_tool_catalog_declarations"
CATALOG_INJECTION_ENABLED_EXTRA: Final[str] = "_framework_tool_catalog_injection_enabled"
CATALOG_RENDER_ERROR_EXTRA: Final[str] = "_framework_tool_catalog_render_error"
CATALOG_USER_EDITED_EXTRA: Final[str] = "_framework_tool_catalog_user_edited"


def _declaration_to_jsonable(decl: ToolDeclaration) -> dict[str, Any]:
    return decl.model_dump(mode="json", exclude_none=True)


def stable_declarations_hash(declarations: Iterable[ToolDeclaration]) -> str:
    """Stable JSON hash key for declaration-sensitive catalog invalidation."""
    import hashlib

    payload = [_declaration_to_jsonable(decl) for decl in declarations]
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def declarations_to_jsonable(declarations: Iterable[ToolDeclaration]) -> list[dict[str, Any]]:
    return [_declaration_to_jsonable(decl) for decl in declarations]


def declarations_from_jsonable(raw: Any) -> list[ToolDeclaration]:
    if not isinstance(raw, list):
        return []
    declarations: list[ToolDeclaration] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            declarations.append(ToolDeclaration(**item))
        except Exception:
            logger.warning("TOOL_CATALOG_DECLARATION_RESTORE_FAILED raw=%r", item)
    return declarations


def declarations_from_tools(tools: Iterable[ResearchTool]) -> list[ToolDeclaration]:
    """Best-effort fallback for non-Designer runtime paths.

    Runtime tool objects do not reliably expose their original YAML
    declaration. New save-time materialization stamps declarations into
    ``AgentNodeConfig.extras``; this fallback is only for direct framework
    callers whose tools include ``definition.metadata['tool_kind']``.
    """
    declarations: list[ToolDeclaration] = []
    for tool in tools:
        definition = tool.definition
        kind = definition.metadata.get("tool_kind")
        if not isinstance(kind, str) or not kind:
            continue
        declarations.append(
            ToolDeclaration(
                name=definition.name,
                kind=kind,
                config={},
                description=definition.description,
            )
        )
    return declarations


class CatalogService:
    """Aggregates per-factory ``catalog_cards`` and renders catalog blocks.

    The service is constructed from an iterable of :class:`CatalogProvider`
    instances. Each provider contributes per-kind cards into a single
    ``Mapping[str, CatalogCard]`` keyed by tool kind. When the same kind
    appears in multiple providers, the first-seen card wins — providers
    earlier in the iterable take precedence, matching the order in which
    :data:`BUILTIN_FACTORIES` lists them.

    Attributes
    ----------
    catalog_cards:
        Aggregated per-kind cards. Read-only; rebuild the service to add
        a new factory.
    registry_version:
        The renderer registry version this service emits. Persisted into
        ``AgentNodeConfig.extras`` alongside the rendered text so the
        runtime can detect drift between save-time and run-time and
        re-render the catalog when the constant has been bumped.
    """

    REGISTRY_VERSION: Final[str] = REGISTRY_VERSION

    def __init__(self, providers: Iterable[CatalogProvider]) -> None:
        merged: dict[str, CatalogCard] = {}
        for provider in providers:
            for kind, card in provider.catalog_cards.items():
                if kind in merged:
                    # First provider wins — deterministic precedence.
                    continue
                merged[kind] = card
        self._catalog_cards: Mapping[str, CatalogCard] = merged
        self._registry_version: str = REGISTRY_VERSION

    @classmethod
    def from_default_factories(cls) -> CatalogService:
        """Build the canonical service backed by :data:`BUILTIN_FACTORIES`.

        Iterates the *unique* factory classes registered in the allow-list
        and instantiates each one with no arguments. Both
        :class:`BuiltinToolFactory` and :class:`DatabricksToolFactory`
        carry their cards as :class:`ClassVar` mappings, so instantiation
        is cheap and does not require runtime-only collaborators.
        """
        from databricks_deep_research.tools.factories import BUILTIN_FACTORIES

        seen: set[type] = set()
        providers: list[CatalogProvider] = []
        for factory_cls in BUILTIN_FACTORIES.values():
            if factory_cls in seen:
                continue
            seen.add(factory_cls)
            providers.append(factory_cls())
        return cls(providers)

    @property
    def catalog_cards(self) -> Mapping[str, CatalogCard]:
        return self._catalog_cards

    @property
    def registry_version(self) -> str:
        return self._registry_version

    def materialize_for_workflow(
        self,
        declarations: Iterable[ToolDeclaration],
        *,
        config: CatalogConfig | None = None,
        probe_samples_by_name: Mapping[str, ProbeSample] | None = None,
    ) -> CatalogRenderResult:
        """Render the catalog block for a single agent's tool list.

        Pure delegation to :func:`render_tool_catalog`; kept as a method so
        callers do not have to import the renderer or know the cards
        mapping shape.
        """
        return render_tool_catalog(
            declarations,
            self._catalog_cards,
            config=config,
            probe_samples_by_name=probe_samples_by_name,
        )

    def materialize_extras(
        self,
        declarations: Iterable[ToolDeclaration],
        *,
        existing_extras: Mapping[str, Any] | None = None,
        force_regen: bool = False,
        config: CatalogConfig | None = None,
    ) -> dict[str, Any]:
        """Return extras with a save-time materialized catalog stamp.

        This is the framework-side primitive used by app save paths. It
        preserves user-edited prose unless ``force_regen`` is explicit.
        """
        decls = list(declarations)
        extras = dict(existing_extras or {})
        extras[CATALOG_INJECTION_ENABLED_EXTRA] = True
        extras[CATALOG_DECLARATIONS_EXTRA] = declarations_to_jsonable(decls)
        decls_hash = stable_declarations_hash(decls)

        if force_regen:
            extras[CATALOG_USER_EDITED_EXTRA] = False

        if extras.get(CATALOG_USER_EDITED_EXTRA) and not force_regen:
            return extras

        if (
            extras.get(CATALOG_TEXT_EXTRA)
            and extras.get(CATALOG_DECLS_HASH_EXTRA) == decls_hash
            and extras.get(CATALOG_REGISTRY_VERSION_EXTRA) == REGISTRY_VERSION
            and not force_regen
        ):
            return extras

        t0 = time.monotonic()
        try:
            result = self.materialize_for_workflow(decls, config=config)
        except Exception as exc:
            extras[CATALOG_RENDER_ERROR_EXTRA] = str(exc)
            logger.warning(
                "TOOL_CATALOG_MATERIALIZE_FAILED error=%s", exc, exc_info=True
            )
            return extras

        extras[CATALOG_TEXT_EXTRA] = result.text
        extras[CATALOG_DECLS_HASH_EXTRA] = decls_hash
        extras[CATALOG_REGISTRY_VERSION_EXTRA] = result.registry_version
        extras[CATALOG_KINDS_EXTRA] = sorted({decl.kind for decl in decls})
        extras[CATALOG_RENDER_ERROR_EXTRA] = None
        logger.info(
            "TOOL_CATALOG_MATERIALIZED tools=%d decls_hash=%s registry_version=%s elapsed_ms=%.1f",
            len(decls),
            decls_hash,
            result.registry_version,
            (time.monotonic() - t0) * 1000,
        )
        return extras

    def resolve_for_runtime(
        self,
        node_config: AgentNodeConfig,
        tools: Iterable[ResearchTool],
        *,
        config: CatalogConfig | None = None,
        node_id: str | None = None,
    ) -> str:
        """Return the catalog prose for runtime prompt injection.

        Uses persisted save-time prose when the declaration hash and registry
        version still match. Falls back to the same renderer when prose is
        absent or stale. Missing/false injection flag preserves legacy prompt
        behavior for revisions saved before this feature.
        """
        extras = node_config.extras or {}
        if extras.get(CATALOG_INJECTION_ENABLED_EXTRA) is not True:
            return ""

        declarations = declarations_from_jsonable(extras.get(CATALOG_DECLARATIONS_EXTRA))
        if not declarations:
            declarations = declarations_from_tools(tools)
        if not declarations:
            return str(extras.get(CATALOG_TEXT_EXTRA) or "")

        decls_hash = stable_declarations_hash(declarations)
        stored_text = str(extras.get(CATALOG_TEXT_EXTRA) or "")
        stored_hash = extras.get(CATALOG_DECLS_HASH_EXTRA)
        stored_version = extras.get(CATALOG_REGISTRY_VERSION_EXTRA)
        if (
            stored_text
            and stored_hash == decls_hash
            and stored_version == REGISTRY_VERSION
        ):
            return stored_text

        reason = "absent"
        if stored_text and stored_hash != decls_hash:
            reason = "hash_mismatch"
        if stored_text and stored_version != REGISTRY_VERSION:
            reason = "version_drift"
            if extras.get(CATALOG_USER_EDITED_EXTRA):
                logger.warning(
                    "TOOL_CATALOG_USER_EDIT_STALE node=%s registry_version=%s",
                    node_id or "<unknown>",
                    stored_version,
                )

        result = self.materialize_for_workflow(declarations, config=config)
        logger.info(
            "TOOL_CATALOG_RUNTIME_FRESH_RENDERED node=%s reason=%s tools=%d registry_version=%s",
            node_id or "<unknown>",
            reason,
            len(declarations),
            result.registry_version,
        )
        return result.text


__all__ = [
    "CATALOG_DECLARATIONS_EXTRA",
    "CATALOG_DECLS_HASH_EXTRA",
    "CATALOG_INJECTION_ENABLED_EXTRA",
    "CATALOG_KINDS_EXTRA",
    "CATALOG_REGISTRY_VERSION_EXTRA",
    "CATALOG_RENDER_ERROR_EXTRA",
    "CATALOG_TEXT_EXTRA",
    "CATALOG_USER_EDITED_EXTRA",
    "CatalogService",
    "declarations_from_jsonable",
    "declarations_to_jsonable",
    "stable_declarations_hash",
]
