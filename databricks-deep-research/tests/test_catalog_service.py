from __future__ import annotations

from databricks_deep_research.agents.config import AgentNodeConfig
from databricks_deep_research.tools.catalog_renderer import REGISTRY_VERSION
from databricks_deep_research.tools.catalog_service import (
    CATALOG_INJECTION_ENABLED_EXTRA,
    CATALOG_REGISTRY_VERSION_EXTRA,
    CATALOG_TEXT_EXTRA,
    CatalogService,
)
from databricks_deep_research.workflow.definition import ToolDeclaration


def _node(extras: dict[str, object]) -> AgentNodeConfig:
    return AgentNodeConfig(
        subtype="researcher",
        system_prompt="{tool_catalog}",
        user_prompt_template="{query}",
        extras=extras,
    )


def test_runtime_uses_persisted_catalog_when_hash_and_version_match() -> None:
    decl = ToolDeclaration(name="docs", kind="vector_search", config={})
    service = CatalogService.from_default_factories()
    extras = service.materialize_extras([decl])
    extras[CATALOG_TEXT_EXTRA] = "persisted catalog"

    rendered = service.resolve_for_runtime(_node(extras), [], node_id="n1")

    assert rendered == "persisted catalog"


def test_runtime_fresh_renders_on_registry_version_drift() -> None:
    decl = ToolDeclaration(name="docs", kind="vector_search", config={})
    service = CatalogService.from_default_factories()
    extras = service.materialize_extras([decl])
    extras[CATALOG_TEXT_EXTRA] = "stale catalog"
    extras[CATALOG_REGISTRY_VERSION_EXTRA] = "0"

    rendered = service.resolve_for_runtime(_node(extras), [], node_id="n1")

    assert rendered != "stale catalog"
    assert "vector_search" in rendered
    assert REGISTRY_VERSION in rendered


def test_runtime_injection_disabled_preserves_legacy_path() -> None:
    service = CatalogService.from_default_factories()
    rendered = service.resolve_for_runtime(
        _node({CATALOG_INJECTION_ENABLED_EXTRA: False}),
        [],
        node_id="legacy",
    )

    assert rendered == ""
