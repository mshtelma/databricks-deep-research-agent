"""Single production entry point for constructing the framework WorkflowRunner.

All app code that executes (or streams) an agentic workflow MUST go through
:func:`build_app_workflow_runner`. Direct construction of
:class:`WorkflowExecutor` or :class:`ToolFactoryContext` in app code is
forbidden — enforced by the anti-regression test at
``tests/unit/agent/test_workflow_runner_factory.py``.

Rationale
---------

Previously the app had two competing construction paths
(``framework_orchestrator.py``, ``agent_designer/orchestrator.py``). Both
silently dropped ``BRAVE_API_KEY`` because they bypassed
:meth:`ToolFactoryContext.from_defaults`. This factory makes the correct
construction the only construction:

1. Reads ``BRAVE_API_KEY`` and ``JINA_API_KEY`` from the process env via
   :meth:`ToolFactoryContext.from_defaults`.
2. Builds a :class:`BraveSearchAdapter` when a key is present so
   ``BuiltinToolFactory._resolve_search_provider`` can construct
   ``web_research`` / ``web_search`` / ``brave_search`` tools at runtime.
3. Emits a structured ``TOOL_FACTORY_CONTEXT_BUILT`` log line so deployment
   misconfigurations (missing secret, revoked key, missing workspace
   client) surface at the first workflow run instead of being buried in
   mid-run synthesizer output.
4. Wires the text-table stack: a fresh :class:`TableBindingRegistry` per
   runner and a :class:`DesignerTableDiscoveryProvider` so the framework's
   ``table_discovery`` / ``table_search`` / ``table_read`` /
   ``table_neighbors`` / ``table_load`` / ``table_aggregate`` tools have
   their dependencies resolved at factory time. Designer-supplied static
   bindings and Unity Catalog ``(catalog, schema)`` scopes can be passed
   in via ``table_static_bindings`` / ``table_uc_scopes``.

Why ``WorkflowRunner`` and not ``WorkflowExecutor`` directly
-----------------------------------------------------------

``WorkflowRunner`` is the framework's documented public API. It handles
workflow loading from YAML/dict/definition, model-tier derivation, state
seeding (including ``conversation_history``), and per-call executor
construction. The four kwargs the app needs (``tool_resolver``,
``tool_registry``, ``context``, ``strict_tool_resolution``) are exposed on
``WorkflowRunner.run`` / ``.stream`` so app callers never have to drop
down to ``WorkflowExecutor``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from databricks_deep_research import (
    WorkflowRunner,
    required_ctx_fields_for_kind,
    resolve_workspace_client,
)
from databricks_deep_research.skills import SkillStore
from databricks_deep_research.tools.builtins.text_table import (
    BindingInfo,
    wire_statement_execution_text_table_context,
)
from databricks_deep_research.tools.factory import ToolFactoryContext

from deep_research.agent.adapters.table_discovery_adapter import (
    DesignerTableDiscoveryProvider,
    workspace_client_factory_from,
)

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient
    from databricks_deep_research.llm.client import FrameworkLLMClient

logger = logging.getLogger(__name__)


def _resolve_table_warehouse_id() -> str | None:
    env_value = os.environ.get("TABLE_TOOLS_WAREHOUSE_ID") or os.environ.get(
        "STORAGE_WAREHOUSE_ID"
    )
    if env_value:
        return env_value
    try:
        from deep_research.core.config import get_settings

        return get_settings().storage_warehouse_id
    except Exception:  # noqa: BLE001 - config may be incomplete in unit/local runs
        logger.debug("TEXT_TABLE_WAREHOUSE_SETTINGS_UNAVAILABLE", exc_info=True)
        return None


def assert_runtime_can_satisfy_workflows(
    ctx: ToolFactoryContext,
    declared_kinds: Iterable[str],
) -> None:
    """Raise ``RuntimeError`` if any declared tool kind cannot be constructed
    from ``ctx`` because a required field is ``None``.

    This is the **boot-time** sibling of :class:`ToolResolver.validate_all`
    (Layer 3). Layer 3 catches per-request unsatisfiable tools after the
    workflow YAML has been parsed; this helper catches them at process
    startup against the **app-wide catalog** of tool kinds declared by all
    statically known workflows. The two layers are complementary — Layer 3
    cannot run without a parsed workflow, but Layer 2 cannot enumerate
    dynamically-supplied designer workflows.

    The check walks each kind's required-ctx-field set (sourced from
    :data:`databricks_deep_research.tools.protocol._TOOL_KIND_REQUIRED_CTX`)
    and reports every missing field grouped by the kinds that need it.

    Parameters
    ----------
    ctx:
        The :class:`ToolFactoryContext` that
        :func:`build_app_workflow_runner` would build at request time.
    declared_kinds:
        The union of tool kinds declared across every workflow the app may
        execute. Typically the keys of the agent_designer registry.

    Raises
    ------
    RuntimeError
        Single message listing every missing field paired with every kind
        that would need it, plus a remediation hint per blocker.
    """
    # Map missing field name → set of kinds blocked by it.
    missing_to_kinds: dict[str, set[str]] = {}
    for kind in declared_kinds:
        for field in required_ctx_fields_for_kind(kind):
            if getattr(ctx, field, None) is None:
                missing_to_kinds.setdefault(field, set()).add(kind)
    if not missing_to_kinds:
        return

    lines = ["APP_BOOT_TOOL_DEPS_MISSING:"]
    for field in sorted(missing_to_kinds):
        kinds = sorted(missing_to_kinds[field])
        lines.append(f"  {field}=None blocks tool kinds: {', '.join(kinds)}")
    # Field-specific remediation hints — keep generic but actionable so the
    # operator does not have to grep the source for what env var fixes what.
    hints: dict[str, str] = {
        "schema_cache": (
            "Set STORAGE_WAREHOUSE_ID (or TABLE_TOOLS_WAREHOUSE_ID, or "
            "settings.storage_warehouse_id) and redeploy. See "
            "preflight.resolve_warehouse_id_or_fail."
        ),
        "sql_executor": (
            "Set STORAGE_WAREHOUSE_ID (or TABLE_TOOLS_WAREHOUSE_ID, or "
            "settings.storage_warehouse_id) and redeploy. See "
            "preflight.resolve_warehouse_id_or_fail."
        ),
        "table_registry": (
            "table_registry should be auto-wired by build_app_workflow_runner. "
            "If this fires, the factory is being bypassed."
        ),
        "table_discovery_provider": (
            "table_discovery_provider should be auto-wired by "
            "build_app_workflow_runner. If this fires, the factory is being "
            "bypassed."
        ),
        "workspace_client": (
            "WorkspaceClient could not be auto-detected. Pass an "
            "OBO-authenticated WorkspaceClient via build_app_workflow_runner."
        ),
        "search_client": (
            "Set BRAVE_API_KEY in the app env (via secret scope binding) so "
            "BraveSearchAdapter can be constructed."
        ),
    }
    seen_hints: set[str] = set()
    for field in sorted(missing_to_kinds):
        hint = hints.get(field)
        if hint and hint not in seen_hints:
            lines.append(f"  → {hint}")
            seen_hints.add(hint)
    raise RuntimeError("\n".join(lines))


def _apply_default_search_client(
    ctx: ToolFactoryContext, llm_client: FrameworkLLMClient | None
) -> None:
    """Point ``ctx.search_client`` at the workspace-default search backend.

    Tools that omit ``config.provider`` inherit ``ctx.search_client`` in the
    builtin tool factory (its no-provider branch). ``from_defaults`` seeds that
    field from ``BRAVE_API_KEY`` (or ``None``). When the workspace
    ``search.provider`` is ``databricks`` we replace it with the model-serving
    built-in-search adapter (reusing the central :func:`_build_web_search_client`)
    so inherited per-agent web tools use Databricks built-in search with **no**
    Brave key assumed. Any other provider — or an un-buildable adapter (e.g. no
    ``llm_client``) — leaves the ``from_defaults`` value untouched. Explicit
    per-tool providers are unaffected: they build via ``_resolve_search_provider``,
    not ``ctx.search_client``.
    """
    from deep_research.agent.adapters.tool_adapter import _build_web_search_client
    from deep_research.core.app_config import get_app_config

    try:
        search_cfg = get_app_config().search
    except Exception:  # pragma: no cover - defensive: config unreadable
        return
    if getattr(search_cfg, "provider", None) != "databricks":
        return
    client = _build_web_search_client(
        search_provider="databricks",
        brave_client=None,
        domain_filter_config=None,
        llm_client=llm_client,
        databricks_search_cfg=search_cfg.databricks,
    )
    if client is not None:
        ctx.search_client = client
        logger.info(
            "FWK_CTX_SEARCH_CLIENT provider=databricks endpoint=%s",
            getattr(search_cfg.databricks, "endpoint", "?"),
        )


def _apply_serving_client_provider(
    ctx: ToolFactoryContext, llm_client: FrameworkLLMClient | None
) -> None:
    """Point ``ctx.serving_client_provider`` at the app's SP serving client.

    Databricks built-in web search is a model-serving call over the PUBLIC web,
    so it authenticates as the app / service principal — the SAME identity every
    LLM call already uses — never the OBO user (``ctx.user_token``), which is
    reserved for user-scoped data tools and does not carry the ``model-serving``
    passthrough scope. Set provider-agnostically (not gated on the global
    ``search.provider``) so an explicit per-tool ``provider: databricks`` is
    covered even when the workspace default is brave/jina. The framework's
    databricks search factory consumes ``serving_client_provider``; the inherited
    default path already builds from this same SP client.
    """
    if llm_client is None:  # pragma: no cover - defensive; the runner always has one
        return
    ctx.serving_client_provider = lambda: llm_client.openai_client
    logger.info("FWK_CTX_SERVING_CLIENT identity=sp source=llm_client")


def build_app_workflow_runner(
    *,
    llm_client: FrameworkLLMClient,
    workspace_client: WorkspaceClient | None,
    user_token: str | None,
    table_static_bindings: Iterable[BindingInfo] | None = None,
    table_uc_scopes: Sequence[tuple[str, str]] | None = None,
    skill_store: SkillStore | None = None,
    skill_scripts_enabled: bool = False,
    skill_scanner: object | None = None,
) -> WorkflowRunner:
    """Build a :class:`WorkflowRunner` with the app's standard tool context.

    Parameters
    ----------
    llm_client:
        The framework LLM client. The factory does not own its lifecycle —
        connection pooling and auth live upstream in the caller.
    workspace_client:
        OBO-authenticated workspace client to thread into Databricks-bound
        tool factories (Genie, vector_search, table_*, etc.). When
        ``None``, :meth:`ToolFactoryContext.from_defaults` attempts to
        auto-detect via ``WorkspaceClient()``; auto-detection failure is
        non-fatal — the resulting tools just won't have workspace access.
    user_token:
        OBO token forwarded to tools that need user-scoped Databricks
        access (Genie, vector_search, etc.). Independent of
        ``BRAVE_API_KEY`` lifecycle.
    table_static_bindings:
        Optional pre-curated ``BindingInfo`` records to surface from the
        ``table_discovery`` tool without round-tripping to UC. Typically
        sourced from the Designer's user-selected ``DesignerAsset``
        payload.
    table_uc_scopes:
        Optional ``(catalog, schema)`` tuples the discovery adapter will
        enumerate via ``WorkspaceClient.tables.list``. Empty / missing
        values are dropped silently.

    Returns
    -------
    :class:`WorkflowRunner`
        A fresh runner with ``factory_context`` populated from env.
        Callers should construct a new runner per request — the framework
        runner is documented as not thread-safe.
    """
    # Resolve OBO-vs-SP ONCE, here, via the framework's shared helper, so
    # EVERY Databricks tool (table_*, vector_search, genie,
    # knowledge_assistant, table_discovery) and the discovery client_factory
    # below run under one identity: the calling user when an OBO token is
    # present, the service principal otherwise. Previously the caller passed
    # the SP client and the OBO token was dropped, so UC-gated tools silently
    # ran as the SP and hit permission errors on user-owned data.
    workspace_client = resolve_workspace_client(
        sp_client=workspace_client, user_token=user_token
    )
    ctx = ToolFactoryContext.from_defaults(
        workspace_client=workspace_client,
        user_token=user_token,
    )
    # Default web-search backend for tools that DON'T pin an explicit provider:
    # follow the workspace ``search.provider`` (Databricks built-in search by
    # default) instead of the Brave adapter ``from_defaults`` seeds. Runs before
    # the context-built log below so it reflects the final ``search_client``.
    _apply_default_search_client(ctx, llm_client)
    _apply_serving_client_provider(ctx, llm_client)

    # Wire the runtime skill store (Feature 2.2): the read_skill / run_skill_script
    # builtins resolve their store from ``ctx.extras["_skill_store"]``. The caller
    # composes it from request context (workspace-FS + Lakebase + bundled seeds).
    # Absent => no skills at runtime (read_skill simply isn't built) — graceful.
    if skill_store is not None:
        ctx.extras["_skill_store"] = skill_store
    # Global half of the skill-script gate (A2). The framework auto-attaches
    # ``run_skill_script`` only when this is True AND the per-agent
    # ``allow_skill_scripts`` is set; default False keeps scripts off everywhere.
    ctx.extras["_skill_scripts_enabled"] = bool(skill_scripts_enabled)
    if skill_scanner is not None:
        ctx.extras["_skill_scanner"] = skill_scanner

    # The discovery adapter pairs static designer bindings with optional
    # UC catalog scopes. ``ctx.workspace_client`` now carries the resolved
    # (OBO when a token was supplied) client, so we wrap it as a stable
    # factory; the adapter ignores the per-call ``user_token`` it receives
    # because auth is already baked into the client. When no client is
    # available the adapter falls back to static bindings only.
    client_factory = (
        workspace_client_factory_from(ctx.workspace_client)
        if ctx.workspace_client is not None
        else None
    )
    table_discovery_provider = DesignerTableDiscoveryProvider.from_pairs(
        client_factory=client_factory,
        scopes=list(table_uc_scopes or ()),
        static_bindings=list(table_static_bindings or ()),
    )

    warehouse_id = _resolve_table_warehouse_id()
    wire_statement_execution_text_table_context(
        ctx,
        warehouse_id=warehouse_id,
        table_discovery_provider=table_discovery_provider,
    )

    logger.info(
        "TOOL_FACTORY_CONTEXT_BUILT workspace_client=%s auth=%s search_client=%s "
        "brave_key=%s jina_key=%s user_token=%s table_registry=present "
        "table_discovery_provider=present table_static_bindings=%d "
        "table_uc_scopes=%d table_sql_executor=%s table_schema_cache=%s",
        "present" if ctx.workspace_client else "MISSING",
        "obo" if user_token else "sp",
        "present" if ctx.search_client else "MISSING",
        "present" if ctx.api_keys.get("brave") else "MISSING",
        "present" if ctx.api_keys.get("jina") else "MISSING",
        "present" if ctx.user_token else "MISSING",
        len(list(table_static_bindings or ())),
        len(list(table_uc_scopes or ())),
        "present" if ctx.sql_executor else "MISSING",
        "present" if ctx.schema_cache else "MISSING",
    )
    return WorkflowRunner(llm_client=llm_client, factory_context=ctx)
