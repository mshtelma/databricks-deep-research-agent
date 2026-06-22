"""Save-time Tool Catalog materialization for Agent Designer workflows.

The framework owns catalog metadata and rendering; this app service owns the
Designer integration point. It walks a WorkflowDefinition AST, finds agent
configs with bound tools, and writes the framework-reserved catalog extras
that runtime prompt injection consumes.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Iterable
from typing import Any

from databricks_deep_research.tools.catalog_renderer import CatalogConfig
from databricks_deep_research.tools.catalog_service import (
    CATALOG_DECLARATIONS_EXTRA,
    CATALOG_DECLS_HASH_EXTRA,
    CATALOG_INJECTION_ENABLED_EXTRA,
    CATALOG_KINDS_EXTRA,
    CATALOG_REGISTRY_VERSION_EXTRA,
    CATALOG_RENDER_ERROR_EXTRA,
    CATALOG_TEXT_EXTRA,
    CATALOG_USER_EDITED_EXTRA,
)
from databricks_deep_research.tools.catalog_service import (
    CatalogService as FrameworkCatalogService,
)
from databricks_deep_research.workflow.definition import ToolDeclaration

from deep_research.core.app_config import get_app_config

logger = logging.getLogger(__name__)

_CATALOG_AGENT_SUBTYPES: frozenset[str] = frozenset({"researcher", "planner"})


def _catalog_config_from_app() -> CatalogConfig:
    cfg = get_app_config().agent_designer.tool_catalog
    return CatalogConfig(
        max_chars=cfg.max_chars,
        summary_only_above_n_tools=cfg.summary_only_above_n_tools,
        include_probes=cfg.include_probes,
    )


class CatalogService:
    """Materialize framework tool catalogs into Designer workflow ASTs."""

    def __init__(
        self,
        framework_service: FrameworkCatalogService | None = None,
        *,
        config: CatalogConfig | None = None,
    ) -> None:
        self._framework = framework_service or FrameworkCatalogService.from_default_factories()
        self._config = config or _catalog_config_from_app()

    def materialize_for_save(
        self,
        definition: dict[str, Any],
        *,
        previous: dict[str, Any] | None = None,
        force_regen: bool = False,
    ) -> dict[str, Any]:
        """Return a copy of ``definition`` with tool catalog extras stamped.

        ``previous`` is accepted for the public contract; the current AST
        carries the authoritative extras state, so no cross-definition merge
        is needed.
        """
        _ = previous
        next_definition = copy.deepcopy(definition)
        next_definition.setdefault("schema_version", 1)
        self.materialize_inplace(next_definition, force_regen=force_regen)
        return next_definition

    def materialize_inplace(
        self,
        workflow: dict[str, Any],
        *,
        force_regen: bool = False,
    ) -> None:
        name_to_decl = _coerce_tool_declarations(workflow.get("tools") or [])
        if not name_to_decl:
            return

        def maybe_materialize(agent_config: dict[str, Any], scope: str) -> None:
            subtype = str(agent_config.get("subtype", ""))
            if subtype not in _CATALOG_AGENT_SUBTYPES:
                return
            tool_names = agent_config.get("tools") or []
            if not isinstance(tool_names, list) or not tool_names:
                return

            declarations: list[ToolDeclaration] = []
            seen_names: set[str] = set()
            for raw_name in tool_names:
                if not isinstance(raw_name, str) or raw_name in seen_names:
                    continue
                decl = name_to_decl.get(raw_name)
                if decl is None:
                    continue
                declarations.append(decl)
                seen_names.add(raw_name)
            if not declarations:
                return

            extras = agent_config.get("extras")
            if not isinstance(extras, dict):
                extras = {}
                agent_config["extras"] = extras
            try:
                agent_config["extras"] = self._framework.materialize_extras(
                    declarations,
                    existing_extras=extras,
                    force_regen=force_regen,
                    config=self._config,
                )
            except Exception as exc:  # noqa: BLE001 - save is best-effort
                extras[CATALOG_RENDER_ERROR_EXTRA] = str(exc)
                extras[CATALOG_INJECTION_ENABLED_EXTRA] = True
                logger.warning(
                    "TOOL_CATALOG_MATERIALIZE_FAILED node=%s error=%s",
                    scope,
                    exc,
                    exc_info=True,
                )

        def walk(node: Any, scope: str) -> None:
            if not isinstance(node, dict):
                return
            node_type = str(node.get("type", ""))
            node_id = str(node.get("id", "<unknown>"))
            raw_config = node.get("config")
            config = raw_config if isinstance(raw_config, dict) else None
            if node_type == "agent" and config is not None:
                maybe_materialize(config, f"{scope}/{node_id}")
            elif node_type == "plan_and_execute" and config is not None:
                planner = config.get("planner")
                if isinstance(planner, dict):
                    maybe_materialize(planner, f"{scope}/{node_id}.planner")
                evaluator = config.get("evaluator")
                if isinstance(evaluator, dict):
                    maybe_materialize(evaluator, f"{scope}/{node_id}.evaluator")
                body = config.get("body")
                if isinstance(body, dict):
                    walk(body, f"{scope}/{node_id}.body")
            for child in node.get("children") or []:
                walk(child, f"{scope}/{node_id}")

        root = workflow.get("root")
        if isinstance(root, dict):
            walk(root, "root")


def _coerce_tool_declarations(raw_decls: Iterable[Any]) -> dict[str, ToolDeclaration]:
    name_to_decl: dict[str, ToolDeclaration] = {}
    for raw in raw_decls:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name", "")).strip()
        kind = str(raw.get("kind", "")).strip()
        if not name or not kind:
            continue
        try:
            name_to_decl[name] = ToolDeclaration(
                name=name,
                kind=kind,
                config=dict(raw.get("config") or {}),
                description=str(raw.get("description", "")),
                probe=raw.get("probe"),
            )
        except Exception:  # noqa: BLE001 - isolated bad declaration
            logger.warning(
                "TOOL_CATALOG_DECL_COERCE_FAILED name=%r kind=%r",
                name,
                kind,
                exc_info=True,
            )
    return name_to_decl


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
]
