"""Authentication strategies for :class:`MCPToolset`.

Each strategy yields a ``headers()`` dict threaded into the MCP transport.
Credentials live ONLY here — never in :class:`ToolContext.extras` — so a
malicious tool cannot exfiltrate them via state inspection.
"""

from __future__ import annotations

from dataclasses import dataclass


class MCPAuth:
    """Abstract base — subclasses implement :meth:`headers`."""

    def headers(self) -> dict[str, str]:  # pragma: no cover — abstract
        raise NotImplementedError


@dataclass
class BearerToken(MCPAuth):
    """Send ``Authorization: Bearer <token>`` on every MCP call."""

    token: str

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}


@dataclass
class ApiKey(MCPAuth):
    """Send a fixed ``<header>: <value>`` pair on every MCP call."""

    header: str
    value: str

    def headers(self) -> dict[str, str]:
        return {self.header: self.value}


@dataclass
class CustomHeaders(MCPAuth):
    """Send arbitrary headers (use sparingly — single source of truth wins)."""

    headers_dict: dict[str, str]

    def headers(self) -> dict[str, str]:
        return dict(self.headers_dict)


__all__ = ["ApiKey", "BearerToken", "CustomHeaders", "MCPAuth"]
