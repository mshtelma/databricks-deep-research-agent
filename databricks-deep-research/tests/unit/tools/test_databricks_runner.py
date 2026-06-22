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
