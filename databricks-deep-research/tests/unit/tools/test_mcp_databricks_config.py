"""Tests for Databricks MCP config + the databricks-aware preflight (B1)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from databricks_deep_research.tools.builtins.databricks_runner import (
    workflow_requires_databricks,
)
from databricks_deep_research.tools.mcp import MCPServerConfig


def test_http_server_requires_url() -> None:
    MCPServerConfig(name="ok", url="https://x/mcp")  # no error
    with pytest.raises(ValueError):
        MCPServerConfig(name="bad", client_kind="http", url="")


def test_databricks_external_needs_connection_name_only() -> None:
    cfg = MCPServerConfig(
        name="ext", client_kind="databricks", connection_name="my_conn"
    )
    assert cfg.url == ""  # derived at runtime, not persisted
    assert cfg.connection_name == "my_conn"


def test_databricks_managed_needs_target_only() -> None:
    cfg = MCPServerConfig(
        name="mgd", client_kind="databricks", managed_target="functions/main/default"
    )
    assert cfg.managed_target == "functions/main/default"


def test_databricks_requires_exactly_one_target() -> None:
    with pytest.raises(ValueError):  # neither
        MCPServerConfig(name="bad", client_kind="databricks")
    with pytest.raises(ValueError):  # both
        MCPServerConfig(
            name="bad",
            client_kind="databricks",
            connection_name="c",
            managed_target="sql",
        )


def test_preflight_counts_databricks_mcp() -> None:
    db_server = MCPServerConfig(
        name="ext", client_kind="databricks", connection_name="conn"
    )
    http_server = MCPServerConfig(name="h", url="https://x/mcp")

    assert (
        workflow_requires_databricks(
            SimpleNamespace(tools=[], mcp_servers=[db_server])
        )
        is True
    )
    # An http-only MCP workflow does NOT require the Databricks identity.
    assert (
        workflow_requires_databricks(
            SimpleNamespace(tools=[], mcp_servers=[http_server])
        )
        is False
    )
    assert (
        workflow_requires_databricks(SimpleNamespace(tools=[], mcp_servers=[]))
        is False
    )
