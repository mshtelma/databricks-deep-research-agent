"""The ``mcp`` tool kind surfaced by the designer registry (spec §4.3).

``mcp`` must appear in the tool-kinds payload (so the Designer palette,
``list_tool_kinds``, and ``semantic_validation`` all know about it) with a
config schema describing the MCP server fields — secret-ref-only credentials,
fast/deep strategy, citeable toggle. ``SchemaField`` renders these with no
React change.
"""
from __future__ import annotations

from typing import Any

from databricks_deep_research.tools.protocol import ToolKind

from deep_research.agent_designer.registry import tool_kinds_payload


def _schemas() -> dict[str, Any]:
    return {t["kind"]: t["config_schema"] for t in tool_kinds_payload()}


def test_mcp_kind_in_payload() -> None:
    kinds = {t["kind"] for t in tool_kinds_payload()}
    assert "mcp" in kinds


def test_mcp_kind_in_tool_enum() -> None:
    # The structural gate + list_tool_kinds derive their allowlist from ToolKind.
    assert "mcp" in {k.value for k in ToolKind}


def test_mcp_schema_has_server_fields() -> None:
    props = _schemas()["mcp"]["properties"]
    for key in ("name", "url", "transport", "auth_type", "secret_ref", "strategy", "citeable"):
        assert key in props, f"mcp config schema missing {key}"


def test_mcp_schema_requires_name_only() -> None:
    # ``url`` is conditionally required (http needs it; databricks derives it),
    # so it is NOT a structural JSON-schema requirement (B1). Only ``name`` is.
    required = _schemas()["mcp"].get("required", [])
    assert "name" in required
    assert "url" not in required


def test_mcp_schema_exposes_databricks_fields() -> None:
    props = _schemas()["mcp"]["properties"]
    assert props["client_kind"]["enum"] == ["http", "databricks"]
    assert "connection_name" in props
    assert "managed_target" in props


def test_mcp_auth_and_strategy_enums() -> None:
    props = _schemas()["mcp"]["properties"]
    assert props["auth_type"]["enum"] == ["none", "bearer", "api_key"]
    assert props["strategy"]["enum"] == ["fast", "deep"]
    assert props["transport"]["enum"] == ["http", "sse"]


def test_mcp_payload_entry_shape() -> None:
    entry = next(t for t in tool_kinds_payload() if t["kind"] == "mcp")
    assert entry["layer"] == "B"
    assert isinstance(entry["config_schema"], dict)
