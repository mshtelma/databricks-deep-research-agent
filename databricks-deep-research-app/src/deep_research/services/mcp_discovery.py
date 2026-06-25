"""Discover Databricks MCP servers for the UI (Feature 4.3 / C1).

Two MCP shapes are surfaced:

* **External / UC-connection MCP** — a Unity Catalog HTTP connection that backs a
  Databricks-provisioned proxy at ``/api/2.0/mcp/external/{connection_name}``.
  Enumerated via ``WorkspaceClient.connections.list()`` and filtered by
  :func:`is_mcp_connection`.
* **Managed MCP** — functions / vector-search / genie servers. These are not
  "discovered" (there is nothing to list); the user names a target
  (``functions/{cat}/{schema}`` etc.) which the runtime validates against an
  allowlist. :func:`managed_mcp_catalog` returns the known managed kinds so the UI
  can present them.

OBO scoping is the caller's responsibility — pass a workspace client built with
the user's identity. All listing is fail-soft.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

__all__ = [
    "DiscoveredMcpServer",
    "discover_mcp_connections",
    "is_mcp_connection",
    "managed_mcp_catalog",
]


class DiscoveredMcpServer(BaseModel):
    """A discovered MCP server candidate (external UC-connection or managed)."""

    name: str
    client_kind: str = "databricks"
    # For external: the UC connection name. For managed: empty.
    connection_name: str = ""
    # For managed: the path suffix (functions/<cat>/<schema>, genie/<id>, ...).
    managed_target: str = ""
    description: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


def _enum_value(value: Any) -> str:
    """Return a comparable string for an SDK enum / string / None."""
    if value is None:
        return ""
    return str(getattr(value, "value", value))


def is_mcp_connection(conn: Any) -> bool:
    """Best-effort predicate: is *conn* a UC connection backing an MCP server?

    A Databricks MCP external server is an HTTP-type Unity Catalog connection
    whose options/properties mark it as MCP (the proxy convention). There is no
    dedicated ``ConnectionType.MCP`` enum (verified on SDK 0.118), so we match
    ``connection_type == HTTP`` plus an ``mcp`` marker in the connection's
    options/properties/url. Conservative by design; the exact marker is
    live-verified on AIS in the security/e2e gate (F1) and may be tuned there.
    """
    if _enum_value(getattr(conn, "connection_type", None)).upper() != "HTTP":
        return False
    haystack: list[str] = []
    for attr in ("options", "properties"):
        val = getattr(conn, attr, None)
        if isinstance(val, dict):
            haystack.extend(str(k).lower() for k in val)
            haystack.extend(str(v).lower() for v in val.values())
    url = _enum_value(getattr(conn, "url", None)).lower()
    if url:
        haystack.append(url)
    return any("mcp" in item for item in haystack)


def discover_mcp_connections(workspace_client: Any) -> list[DiscoveredMcpServer]:
    """List external UC-connection MCP servers visible to *workspace_client*.

    Fail-soft: returns ``[]`` (with a warning) if the workspace client is absent
    or the listing fails, so a discovery surface never hard-errors on MCP.
    """
    if workspace_client is None:
        return []
    try:
        connections = list(workspace_client.connections.list())
    except Exception:  # noqa: BLE001 — discovery is best-effort
        logger.warning("MCP_CONNECTION_DISCOVERY_FAILED", exc_info=True)
        return []

    servers: list[DiscoveredMcpServer] = []
    for conn in connections:
        if not is_mcp_connection(conn):
            continue
        name = _enum_value(getattr(conn, "name", None))
        if not name:
            continue
        servers.append(
            DiscoveredMcpServer(
                name=name,
                client_kind="databricks",
                connection_name=name,
                description=_enum_value(getattr(conn, "comment", None)),
                metadata={"connection_type": "HTTP"},
            )
        )
    servers.sort(key=lambda s: s.name.lower())
    logger.info("MCP_CONNECTION_DISCOVERY count=%d", len(servers))
    return servers


def managed_mcp_catalog() -> list[dict[str, str]]:
    """Return the known managed-MCP target kinds for the UI to offer.

    These are not discovered — the user supplies the concrete target; the runtime
    validates it against the allowlist in ``mcp_adapter._derive_databricks_mcp_url``.
    """
    return [
        {
            "kind": "functions",
            "template": "functions/{catalog}/{schema}",
            "label": "Unity Catalog functions",
        },
        {
            "kind": "vector-search",
            "template": "vector-search/{catalog}/{schema}",
            "label": "Vector Search indexes",
        },
        {
            "kind": "genie",
            "template": "genie/{space_id}",
            "label": "Genie space",
        },
    ]
