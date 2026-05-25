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
from typing import TYPE_CHECKING

from databricks_deep_research import WorkflowRunner
from databricks_deep_research.tools.factory import ToolFactoryContext

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient
    from databricks_deep_research.llm.client import FrameworkLLMClient

logger = logging.getLogger(__name__)


def build_app_workflow_runner(
    *,
    llm_client: FrameworkLLMClient,
    workspace_client: WorkspaceClient | None,
    user_token: str | None,
) -> WorkflowRunner:
    """Build a :class:`WorkflowRunner` with the app's standard tool context.

    Parameters
    ----------
    llm_client:
        The framework LLM client. The factory does not own its lifecycle —
        connection pooling and auth live upstream in the caller.
    workspace_client:
        OBO-authenticated workspace client to thread into Databricks-bound
        tool factories (Genie, vector_search, delta_read, etc.). When
        ``None``, :meth:`ToolFactoryContext.from_defaults` attempts to
        auto-detect via ``WorkspaceClient()``; auto-detection failure is
        non-fatal — the resulting tools just won't have workspace access.
    user_token:
        OBO token forwarded to tools that need user-scoped Databricks
        access (Genie, vector_search, etc.). Independent of
        ``BRAVE_API_KEY`` lifecycle.

    Returns
    -------
    :class:`WorkflowRunner`
        A fresh runner with ``factory_context`` populated from env.
        Callers should construct a new runner per request — the framework
        runner is documented as not thread-safe.
    """
    ctx = ToolFactoryContext.from_defaults(
        workspace_client=workspace_client,
        user_token=user_token,
    )
    logger.info(
        "TOOL_FACTORY_CONTEXT_BUILT workspace_client=%s search_client=%s "
        "brave_key=%s jina_key=%s user_token=%s",
        "present" if ctx.workspace_client else "MISSING",
        "present" if ctx.search_client else "MISSING",
        "present" if ctx.api_keys.get("brave") else "MISSING",
        "present" if ctx.api_keys.get("jina") else "MISSING",
        "present" if ctx.user_token else "MISSING",
    )
    return WorkflowRunner(llm_client=llm_client, factory_context=ctx)
