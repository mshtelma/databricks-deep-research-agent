"""ResourceResolver — walk WorkflowDefinition AST → MLflow resource list.

Plan reference: agent-designer-deployment.md Section D.3.

The MLflow resources list passed to ``mlflow.pyfunc.log_model(resources=[...])``
drives **automatic auth-passthrough** at deploy time: Databricks issues an
OAuth scope grant to the agent's service principal for each resource, so the
deployed agent can call those tools / endpoints / indexes without manual
permission wiring.

Constitution principle 4: ``isinstance`` is used here only on **external
MLflow SDK Resource types** (which form a closed union without a built-in
discriminator). Internal types are typed via Pydantic / dataclasses; this
file does not introspect any internal type for safety.
"""
from __future__ import annotations

from typing import Any

from mlflow.models.resources import (
    DatabricksFunction,
    DatabricksGenieSpace,
    DatabricksServingEndpoint,
    DatabricksSQLWarehouse,
    DatabricksVectorSearchIndex,
)

Resource = (
    DatabricksServingEndpoint
    | DatabricksVectorSearchIndex
    | DatabricksGenieSpace
    | DatabricksFunction
    | DatabricksSQLWarehouse
)


# Tool kinds that carry no Databricks resource (web tooling + standalone files).
_NO_RESOURCE_KINDS: frozenset[str] = frozenset(
    {"web_search", "web_crawl", "file_search"}
)

# Tool kinds that resolve to a SQL warehouse (the kind config carries
# ``warehouse_id``). SQL-analytics paths (delta_read, table_read, compute*).
_SQL_WAREHOUSE_KINDS: frozenset[str] = frozenset(
    {
        "delta_read",
        "delta_grep",
        "delta_table_read",
        "table_read",
        "compute",
        "compute_namespace",
    }
)


def _resource_key(r: Resource) -> str:
    """Stable identity key per resource type (used for deduplication).

    All MLflow Resource subclasses expose ``.name`` as the identifying
    attribute. The class type is part of the key to keep e.g. an endpoint
    and an index with the same string name distinct.
    """
    return f"{type(r).__name__}:{r.name}"


class ResourceResolver:
    """Resolves a workflow definition (JSONB dict) into MLflow resources."""

    def resolve(self, definition: dict[str, Any]) -> list[Resource]:
        """Walk the AST and return a deduplicated list of resources."""
        resources: dict[str, Resource] = {}

        # 1. Walk the top-level tool declarations.
        for tool in definition.get("tools", []) or []:
            for r in self._resolve_tool(tool):
                resources[_resource_key(r)] = r

        # 2. Walk model-tier endpoint references.
        for tier_config in (definition.get("models", {}) or {}).values():
            if not isinstance(tier_config, dict):
                continue
            for ep in tier_config.get("endpoints", []) or []:
                ep_name = ep if isinstance(ep, str) else (
                    ep.get("name") if isinstance(ep, dict) else None
                )
                if ep_name:
                    r = DatabricksServingEndpoint(endpoint_name=ep_name)
                    resources[_resource_key(r)] = r

        # 3. Walk agent nodes recursively for endpoint config refs.
        root = definition.get("root")
        if isinstance(root, dict):
            self._walk_node(root, resources)

        return list(resources.values())

    def _resolve_tool(self, tool: dict[str, Any]) -> list[Resource]:
        """Map one tool declaration to zero or more resources."""
        if not isinstance(tool, dict):
            return []
        kind = tool.get("kind", "")
        config = tool.get("config", {}) or {}

        if kind == "vector_search":
            index_name = config.get("index_name", "")
            if index_name:
                return [DatabricksVectorSearchIndex(index_name=index_name)]
            return []

        if kind == "genie":
            space_id = config.get("genie_space_id", "")
            if space_id:
                return [DatabricksGenieSpace(genie_space_id=space_id)]
            return []

        if kind == "knowledge_assistant":
            ep_name = config.get("endpoint_name", "")
            if ep_name:
                return [DatabricksServingEndpoint(endpoint_name=ep_name)]
            return []

        if kind in _SQL_WAREHOUSE_KINDS:
            warehouse_id = config.get("warehouse_id", "")
            if warehouse_id:
                return [DatabricksSQLWarehouse(warehouse_id=warehouse_id)]
            return []

        if kind == "custom":
            # Custom tools are out of scope for ResourceResolver -- their
            # plugins emit their own resources separately. Documented at
            # plan Section M per-mode feature handling table.
            return []

        if kind in _NO_RESOURCE_KINDS:
            return []

        # Unknown tool kind -- emit no resource. This is safe; the deployed
        # agent will fail at runtime with a clear "tool not registered" if it
        # tries to use one. Mode 3 explicitly rejects unsupported tool kinds
        # in MlflowAgentTranslator.validate() before ever reaching deploy.
        return []

    def _walk_node(
        self,
        node: dict[str, Any],
        resources: dict[str, Resource],
    ) -> None:
        """Recursively walk node tree; emit resources for endpoint refs.

        Agent nodes carry an optional ``config.endpoint`` referencing a
        named serving endpoint by string. Composite nodes (sequence,
        parallel, loop, conditional, plan_and_execute) walk children.

        ``plan_and_execute`` stores its nested agents OUTSIDE the
        ``children`` array — the planner and evaluator are materialized
        agent configs at ``config.planner`` / ``config.evaluator``, and
        the executed body is a node tree at ``config.body``. The base
        ``children`` walk does not reach those, so this method handles
        them explicitly (W11 of the fix plan — codex flagged this gap
        because endpoints referenced by nested agents otherwise never get
        a permission grant at deploy time, causing 403s at runtime).
        """
        node_type = node.get("type")

        if node_type == "agent":
            config = node.get("config", {}) or {}
            ep = config.get("endpoint", "")
            if isinstance(ep, str) and ep:
                r = DatabricksServingEndpoint(endpoint_name=ep)
                resources[_resource_key(r)] = r

        if node_type == "plan_and_execute":
            config = node.get("config", {}) or {}
            # Planner + evaluator are stored as materialized agent configs
            # (dicts with an ``endpoint`` field), not wrapped in a node.
            for nested_key in ("planner", "evaluator"):
                nested = config.get(nested_key)
                if isinstance(nested, dict):
                    ep = nested.get("endpoint", "")
                    if isinstance(ep, str) and ep:
                        r = DatabricksServingEndpoint(endpoint_name=ep)
                        resources[_resource_key(r)] = r
            # Body is a node tree — recurse so any agent inside it is
            # discovered.
            body = config.get("body")
            if isinstance(body, dict):
                self._walk_node(body, resources)

        for child in node.get("children", []) or []:
            if isinstance(child, dict):
                self._walk_node(child, resources)
