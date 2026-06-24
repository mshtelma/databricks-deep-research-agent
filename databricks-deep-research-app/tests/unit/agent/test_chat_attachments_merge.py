"""Tests for chat-attached skills + MCP servers merge (E1)."""

from __future__ import annotations

from databricks_deep_research.workflow.definition import (
    NodeType,
    WorkflowDefinition,
    WorkflowNode,
)

from deep_research.agent.framework_orchestrator import _merge_chat_attachments


def _agent_def() -> WorkflowDefinition:
    root = WorkflowNode(
        id="a1",
        type=NodeType.agent,
        label="A",
        config={"subtype": "researcher", "system_prompt": "x"},
    )
    return WorkflowDefinition(id="w", name="w", root=root)


def test_merges_skills_into_agent_config() -> None:
    defn = _agent_def()
    _merge_chat_attachments(defn, ["market-research", "finance"], None)
    assert defn.root.config["skills"] == ["market-research", "finance"]


def test_skills_merge_dedups_with_existing() -> None:
    defn = _agent_def()
    defn.root.config["skills"] = ["finance"]
    _merge_chat_attachments(defn, ["finance", "market-research"], None)
    assert defn.root.config["skills"] == ["finance", "market-research"]


def test_merges_mcp_servers_as_databricks() -> None:
    defn = _agent_def()
    _merge_chat_attachments(defn, None, ["weather", "sales"])
    names = [s.name for s in defn.mcp_servers]
    assert names == ["weather", "sales"]
    server = defn.mcp_servers[0]
    assert server.client_kind == "databricks"
    assert server.connection_name == "weather"


def test_mcp_merge_dedups_with_existing() -> None:
    defn = _agent_def()
    _merge_chat_attachments(defn, None, ["weather"])
    _merge_chat_attachments(defn, None, ["weather", "sales"])
    assert [s.name for s in defn.mcp_servers] == ["weather", "sales"]


def test_noop_when_both_empty() -> None:
    defn = _agent_def()
    _merge_chat_attachments(defn, None, None)
    assert "skills" not in defn.root.config
    assert defn.mcp_servers == []


def test_merges_into_plan_and_execute_nested_agents() -> None:
    body = {"subtype": "researcher", "system_prompt": "y"}
    planner = {"subtype": "planner", "system_prompt": "p"}
    root = WorkflowNode(
        id="pae",
        type=NodeType.plan_and_execute,
        label="PAE",
        config={"planner": planner, "body": body},
    )
    defn = WorkflowDefinition(id="w", name="w", root=root)
    _merge_chat_attachments(defn, ["finance"], None)
    assert defn.root.config["planner"]["skills"] == ["finance"]
    assert defn.root.config["body"]["skills"] == ["finance"]


# -- MCP -> researcher agent binding (so executor.maybe_attach_mcp fires) -----


def test_mcp_servers_bound_to_researcher_agent() -> None:
    defn = _agent_def()  # root subtype == "researcher"
    _merge_chat_attachments(defn, None, ["tavily_mcp"])
    # Workflow-level config (unchanged behavior) + the new per-agent binding.
    assert [s.name for s in defn.mcp_servers] == ["tavily_mcp"]
    assert defn.root.config["mcp_servers"] == ["tavily_mcp"]


def test_mcp_servers_not_bound_to_non_researcher() -> None:
    root = WorkflowNode(
        id="s1",
        type=NodeType.agent,
        label="S",
        config={"subtype": "synthesizer", "system_prompt": "x"},
    )
    defn = WorkflowDefinition(id="w", name="w", root=root)
    _merge_chat_attachments(defn, None, ["tavily_mcp"])
    # Server still registered at workflow level, but the synthesizer is NOT bound.
    assert [s.name for s in defn.mcp_servers] == ["tavily_mcp"]
    assert "mcp_servers" not in root.config


def test_mcp_binds_plan_and_execute_researcher_body_only() -> None:
    body = {"subtype": "researcher", "system_prompt": "y"}
    planner = {"subtype": "planner", "system_prompt": "p"}
    root = WorkflowNode(
        id="pae",
        type=NodeType.plan_and_execute,
        label="PAE",
        config={"planner": planner, "body": body},
    )
    defn = WorkflowDefinition(id="w", name="w", root=root)
    _merge_chat_attachments(defn, None, ["tavily_mcp"])
    assert defn.root.config["body"]["mcp_servers"] == ["tavily_mcp"]
    assert "mcp_servers" not in defn.root.config["planner"]


def test_mcp_agent_binding_dedups_with_existing() -> None:
    defn = _agent_def()
    defn.root.config["mcp_servers"] = ["tavily_mcp"]
    _merge_chat_attachments(defn, None, ["tavily_mcp", "sales"])
    assert defn.root.config["mcp_servers"] == ["tavily_mcp", "sales"]
