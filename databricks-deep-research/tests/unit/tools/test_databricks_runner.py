"""Unit tests for the reusable Databricks workflow-runner builder."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from databricks_deep_research.core import databricks_auth
from databricks_deep_research.tools.builtins.databricks_runner import (
    build_databricks_workflow_runner,
    workflow_requires_databricks,
)
from databricks_deep_research.workflow.definition import (
    NodeType,
    ToolDeclaration,
    WorkflowDefinition,
    WorkflowNode,
)


def _sp(host: str = "https://wsp.example.databricks.com") -> MagicMock:
    sp = MagicMock(name="sp_client")
    sp.config.host = host
    return sp


def test_runner_bakes_obo_client_into_every_tool_when_token_present() -> None:
    sp = _sp()
    obo = MagicMock(name="obo_client")
    with patch.object(databricks_auth, "WorkspaceClient", return_value=obo):
        runner = build_databricks_workflow_runner(
            llm_client=MagicMock(),
            sp_workspace_client=sp,
            user_token="user-tok",
            warehouse_id="wh-1",
        )
    ctx = runner.factory_context
    assert ctx.workspace_client is obo
    assert ctx.user_token == "user-tok"
    # The text-table SQL executor runs statements AS THE USER (the bug fix).
    assert ctx.sql_executor is not None
    assert ctx.sql_executor._workspace_client is obo
    assert ctx.schema_cache is not None
    assert ctx.table_registry is not None


def test_runner_uses_sp_client_when_no_token() -> None:
    sp = _sp()
    with patch.object(databricks_auth, "WorkspaceClient") as wc:
        runner = build_databricks_workflow_runner(
            llm_client=MagicMock(),
            sp_workspace_client=sp,
            user_token=None,
            warehouse_id="wh-1",
        )
    wc.assert_not_called()
    ctx = runner.factory_context
    assert ctx.workspace_client is sp
    assert ctx.sql_executor._workspace_client is sp


def _wf_with_tool_kinds(*kinds: str) -> WorkflowDefinition:
    return WorkflowDefinition(
        id="t",
        name="t",
        root=WorkflowNode(
            id="root",
            type=NodeType.agent,
            label="Root",
            config={"subtype": "researcher", "output_key": "findings"},
        ),
        tools=[
            ToolDeclaration(name=f"tool_{i}", kind=kind)
            for i, kind in enumerate(kinds)
        ],
    )


def test_workflow_requires_databricks_true_for_vector_search() -> None:
    assert workflow_requires_databricks(_wf_with_tool_kinds("vector_search")) is True


def test_workflow_requires_databricks_true_for_table_read() -> None:
    assert (
        workflow_requires_databricks(_wf_with_tool_kinds("web_search", "table_read"))
        is True
    )


def test_workflow_requires_databricks_false_for_web_only() -> None:
    assert (
        workflow_requires_databricks(_wf_with_tool_kinds("web_search", "web_crawl"))
        is False
    )


def test_workflow_requires_databricks_false_for_no_tools() -> None:
    assert workflow_requires_databricks(_wf_with_tool_kinds()) is False


# ---------------------------------------------------------------------------
# Default databricks web-search backend for provider-inheriting web tools.
# A framework-only host (the shell app) builds its context here, NOT via the
# app's build_app_workflow_runner, so the runner must seed the same databricks
# built-in search the main app wires — with no Brave key.
# ---------------------------------------------------------------------------

_DBX_SEARCH = (
    "databricks_deep_research.tools.builtins.databricks_web_search."
    "build_databricks_web_search_adapter"
)


def test_runner_wires_databricks_default_search_when_no_brave(monkeypatch) -> None:
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.delenv("DATABRICKS_WEB_SEARCH_ENDPOINT", raising=False)
    llm = MagicMock()
    fake_adapter = MagicMock(name="dbx_search_adapter")
    with patch(_DBX_SEARCH, return_value=fake_adapter) as mk:
        runner = build_databricks_workflow_runner(
            llm_client=llm, sp_workspace_client=_sp(), user_token=None, warehouse_id="wh-1"
        )
    ctx = runner.factory_context
    assert ctx.search_client is fake_adapter
    assert ctx.serving_client_provider is not None
    # Built-in web search runs as the app SP via the LLM client's serving client.
    assert ctx.serving_client_provider() is llm.openai_client
    # Default endpoint when env is unset.
    assert mk.call_args.kwargs["model"] == "databricks-gemini-3-1-flash-lite"


def test_runner_default_search_honors_env_endpoint(monkeypatch) -> None:
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.setenv("DATABRICKS_WEB_SEARCH_ENDPOINT", "databricks-gpt-5")
    with patch(_DBX_SEARCH, return_value=MagicMock()) as mk:
        build_databricks_workflow_runner(
            llm_client=MagicMock(), sp_workspace_client=_sp(), user_token=None
        )
    assert mk.call_args.kwargs["model"] == "databricks-gpt-5"


def test_runner_does_not_overwrite_brave_search_client(monkeypatch) -> None:
    monkeypatch.delenv("DATABRICKS_WEB_SEARCH_ENDPOINT", raising=False)
    with patch(_DBX_SEARCH) as mk:
        runner = build_databricks_workflow_runner(
            llm_client=MagicMock(),
            sp_workspace_client=_sp(),
            user_token=None,
            brave_api_key="brave-key",
        )
    mk.assert_not_called()  # brave-pinned host keeps its Brave search client
    assert runner.factory_context.search_client is not None


def test_runner_default_search_degrades_gracefully_on_adapter_error(monkeypatch) -> None:
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    with patch(_DBX_SEARCH, side_effect=RuntimeError("endpoint unclassifiable")):
        runner = build_databricks_workflow_runner(
            llm_client=MagicMock(), sp_workspace_client=_sp(), user_token=None, warehouse_id="wh-1"
        )
    # Build never crashes; the web tool fails later with a clear message instead.
    assert runner.factory_context.search_client is None
    assert runner.factory_context.serving_client_provider is not None
