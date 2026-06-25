"""Tests for MCP save-lift + agent binding validation (B2)."""

from __future__ import annotations

from typing import Any

from deep_research.agent_designer.ast_normalizer import normalize_ast
from deep_research.agent_designer.semantic_validation import semantic_validation_errors


def _fix_kinds(fixes: list[Any]) -> set[str]:
    return {f.kind for f in fixes}


def _ast_with_tools(tools: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "root": {
            "id": "agent_1",
            "type": "agent",
            "config": {"subtype": "researcher", "system_prompt": "x"},
        },
        "tools": tools,
    }


# ---------------------------------------------------------------------------
# Save-lift (normalize_ast)
# ---------------------------------------------------------------------------


def test_lift_moves_mcp_card_to_mcp_servers() -> None:
    ast = _ast_with_tools(
        [
            {
                "kind": "mcp",
                "name": "weather",
                "config": {"name": "weather", "url": "https://w/mcp"},
            },
            {"kind": "web_search", "name": "web"},
        ]
    )
    new, fixes = normalize_ast(ast)
    assert "mcp_server_lift" in _fix_kinds(fixes)
    # The mcp card is removed from tools, lifted to mcp_servers.
    assert [t.get("kind") for t in new["tools"]] == ["web_search"]
    assert [s["name"] for s in new["mcp_servers"]] == ["weather"]
    assert new["mcp_servers"][0]["url"] == "https://w/mcp"


def test_lift_dedups_by_name() -> None:
    ast = _ast_with_tools(
        [
            {"kind": "mcp", "name": "x", "config": {"name": "x", "url": "https://a"}},
            {"kind": "mcp", "name": "x", "config": {"name": "x", "url": "https://b"}},
        ]
    )
    new, fixes = normalize_ast(ast)
    assert len(new["mcp_servers"]) == 1
    assert "mcp_server_dedup" in _fix_kinds(fixes)


def test_lift_drops_nameless_card() -> None:
    ast = _ast_with_tools([{"kind": "mcp", "config": {"url": "https://a"}}])
    new, fixes = normalize_ast(ast)
    assert new.get("mcp_servers", []) == []
    assert "mcp_server_dropped" in _fix_kinds(fixes)


def test_lift_noop_without_mcp_cards() -> None:
    ast = _ast_with_tools([{"kind": "web_search", "name": "web"}])
    new, fixes = normalize_ast(ast)
    assert "mcp_server_lift" not in _fix_kinds(fixes)
    # tools unchanged (still the single web_search)
    assert [t.get("kind") for t in new["tools"]] == ["web_search"]


def test_lift_preserves_databricks_fields() -> None:
    ast = _ast_with_tools(
        [
            {
                "kind": "mcp",
                "name": "uc",
                "config": {
                    "name": "uc",
                    "client_kind": "databricks",
                    "connection_name": "my_conn",
                },
            }
        ]
    )
    new, _ = normalize_ast(ast)
    server = new["mcp_servers"][0]
    assert server["client_kind"] == "databricks"
    assert server["connection_name"] == "my_conn"


# ---------------------------------------------------------------------------
# Agent binding validation
# ---------------------------------------------------------------------------


def _definition_with_agent_mcp(
    refs: list[str], *, mcp_servers: list[dict] | None = None, tools: list | None = None
) -> dict[str, Any]:
    return {
        "root": {
            "id": "agent_1",
            "type": "agent",
            "config": {
                "subtype": "researcher",
                "system_prompt": "x",
                "mcp_servers": refs,
            },
        },
        "tools": tools or [],
        "mcp_servers": mcp_servers or [],
    }


def test_validation_accepts_declared_server() -> None:
    defn = _definition_with_agent_mcp(["weather"], mcp_servers=[{"name": "weather"}])
    errors = semantic_validation_errors(defn)
    assert not [e for e in errors if "MCP server" in e.message]


def test_validation_rejects_undeclared_server() -> None:
    defn = _definition_with_agent_mcp(["ghost"], mcp_servers=[{"name": "weather"}])
    errors = semantic_validation_errors(defn)
    assert any("MCP server 'ghost'" in e.message for e in errors)


def test_validation_accepts_prelift_mcp_card() -> None:
    # Before the lift runs, the server is still a kind=="mcp" tool card.
    defn = _definition_with_agent_mcp(
        ["weather"],
        tools=[{"kind": "mcp", "name": "weather", "config": {"name": "weather"}}],
    )
    errors = semantic_validation_errors(defn)
    assert not [e for e in errors if "MCP server" in e.message]
