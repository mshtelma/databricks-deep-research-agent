"""Reusable builder for a Databricks-backed ``WorkflowRunner`` (OBO or SP).

Both host applications build a fresh ``WorkflowRunner`` per request. This is
the single place that:

* resolves the per-request ``WorkspaceClient`` — OBO when a user token is
  present, else the service principal — via :func:`resolve_workspace_client`,
  and
* wires the text-table SQL tools (plus any injected table-discovery provider)
  into a fresh ``ToolFactoryContext``.

Because the resolved client is baked into the context, EVERY Databricks tool
(``table_*``, ``vector_search``, ``genie``, ``knowledge_assistant``,
``table_discovery``) runs under the same identity: the calling user under
OBO, the service principal otherwise. Hosts no longer hand-roll this wiring.
"""

from __future__ import annotations

from typing import Any

from databricks_deep_research.core.databricks_auth import resolve_workspace_client
from databricks_deep_research.llm.client import FrameworkLLMClient
from databricks_deep_research.runner import WorkflowRunner
from databricks_deep_research.tools.builtins.text_table import (
    wire_statement_execution_text_table_context,
)
from databricks_deep_research.tools.factory import ToolFactoryContext
from databricks_deep_research.tools.protocol import DATABRICKS_BOUND_TOOL_KINDS
from databricks_deep_research.workflow.definition import WorkflowDefinition


def build_databricks_workflow_runner(
    *,
    llm_client: FrameworkLLMClient,
    sp_workspace_client: Any | None,
    user_token: str | None,
    warehouse_id: str | None = None,
    table_discovery_provider: Any | None = None,
    brave_api_key: str | None = None,
    extras: dict[str, Any] | None = None,
) -> WorkflowRunner:
    """Build a per-request runner whose Databricks tools authenticate as
    the OBO user (when ``user_token`` is set) or the service principal.

    Parameters
    ----------
    sp_workspace_client:
        The host's service-principal / default ``WorkspaceClient``. Used
        directly when no ``user_token`` is present, and as the host source
        for OBO clients otherwise (the token's host is derived from it).
    user_token:
        The caller's OBO access token (``x-forwarded-access-token`` in
        Databricks Apps). When present, every Databricks tool runs as the
        user.
    warehouse_id:
        SQL warehouse for the text-table tools; falls back to
        ``TABLE_TOOLS_WAREHOUSE_ID`` / ``STORAGE_WAREHOUSE_ID`` env when None.
    table_discovery_provider:
        Optional host-built provider for the ``table_discovery`` tool (e.g.
        the app's designer-asset provider). Left to the host so the framework
        stays application-agnostic.
    """
    workspace_client = resolve_workspace_client(
        sp_client=sp_workspace_client, user_token=user_token
    )
    ctx = ToolFactoryContext.from_defaults(
        workspace_client=workspace_client,
        user_token=user_token,
        brave_api_key=brave_api_key,
        extras=extras,
    )
    wire_statement_execution_text_table_context(
        ctx,
        warehouse_id=warehouse_id,
        table_discovery_provider=table_discovery_provider,
    )
    return WorkflowRunner(llm_client=llm_client, factory_context=ctx)


def workflow_requires_databricks(definition: WorkflowDefinition) -> bool:
    """Return ``True`` iff the workflow needs the caller's Databricks identity.

    Hosts use this to fail closed: when a workflow needs UC-gated resources but
    no OBO user token was forwarded, running as the service principal would
    silently produce permission errors / empty results, so the host should
    reject the request instead. This is True when either:

    * a top-level declared tool is Databricks-bound
      (``DATABRICKS_BOUND_TOOL_KINDS``), or
    * any configured MCP server is a Databricks managed / UC-connection server
      (``client_kind='databricks'``), which is reached strictly under OBO (B1).
    """
    if any(tool.kind in DATABRICKS_BOUND_TOOL_KINDS for tool in definition.tools):
        return True
    return any(
        getattr(server, "client_kind", "http") == "databricks"
        for server in (getattr(definition, "mcp_servers", None) or [])
    )
