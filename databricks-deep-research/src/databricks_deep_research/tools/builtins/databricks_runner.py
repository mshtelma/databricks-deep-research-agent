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

import logging
import os
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

logger = logging.getLogger(__name__)

# Default Databricks built-in web-search serving endpoint for provider-inheriting
# web tools when the host wires no explicit one. Overridable via the
# ``DATABRICKS_WEB_SEARCH_ENDPOINT`` env var (the shell-app exporter sets it from
# the app's ``search.databricks.endpoint``). Gemini-flash-lite is the documented
# default (native generateContent grounding, single fast call) and is family-safe
# for ``model_family=None`` auto-detection.
_DEFAULT_DATABRICKS_WEB_SEARCH_ENDPOINT = "databricks-gemini-3-1-flash-lite"


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

    # Default web-search backend for provider-inheriting web tools (web_search /
    # web_research / web_crawl declared with no explicit ``config.provider``).
    # Mirrors the main app's workflow_runner_factory._apply_default_search_client +
    # _apply_serving_client_provider so a framework-only host that builds its
    # context HERE (the shell app — NOT via build_app_workflow_runner) gets the
    # SAME databricks built-in search the main app uses, with NO Brave key
    # required. Without this, an inheriting web tool fails at factory build with
    # "search_client required …", which surfaces as a confusing "missing declared
    # tools" WorkflowError.
    #
    # Built-in web search is a public-web model-serving call, so it authenticates
    # as the app service principal (``llm_client.openai_client``) — never the OBO
    # user, whose token need not carry the model-serving passthrough scope.
    ctx.serving_client_provider = lambda: llm_client.openai_client
    # Only seed a default when the host wired none (``from_defaults`` sets it from
    # a Brave key) and we have an SP to call as — so Brave-pinned agents and
    # explicit per-tool providers (which resolve via ``_resolve_search_provider``)
    # are untouched.
    if ctx.search_client is None and ctx.workspace_client is not None:
        endpoint = (
            os.environ.get("DATABRICKS_WEB_SEARCH_ENDPOINT")
            or _DEFAULT_DATABRICKS_WEB_SEARCH_ENDPOINT
        )
        try:
            from databricks_deep_research.tools.builtins.databricks_web_search import (
                build_databricks_web_search_adapter,
            )

            ctx.search_client = build_databricks_web_search_adapter(
                client_provider=ctx.serving_client_provider,
                model=endpoint,
                max_results=10,
            )
            logger.info("FWK_DBX_DEFAULT_SEARCH endpoint=%s", endpoint)
        except Exception as exc:  # noqa: BLE001 — degrade gracefully, never crash build
            logger.warning(
                "FWK_DBX_DEFAULT_SEARCH_FAILED endpoint=%s exc=%s — provider-inheriting "
                "web tools will fail at factory build with a clear error",
                endpoint,
                str(exc)[:200],
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
