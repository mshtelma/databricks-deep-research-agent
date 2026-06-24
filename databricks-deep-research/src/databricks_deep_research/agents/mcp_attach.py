"""Auto-attach discovered MCP tools to agents that bind an MCP server.

An agent declares the MCP *servers* it uses via ``config.mcp_servers`` (server
NAMES, not tool names — discovered MCP tool names are not statically known at
author time, so they never enter ``config.tools`` or the declared-tools
validator). The host discovers each server's tools once per request and stashes
a ``{server_name: [tools]}`` map in the factory context under
``extras["_mcp_tools_by_server"]``. The executor calls
:func:`maybe_attach_mcp` after resolving declared tools (and after the skill
auto-attach) so those tools become callable by the agent's ReAct loop, which
builds its name->tool map from this same resolved ``tools`` list.

This mirrors :mod:`databricks_deep_research.agents.skill_attach`. It is a no-op
(and never raises) when the agent binds no servers, no map is wired (e.g. the
optional ``mcp`` / ``databricks-mcp`` packages are not installed, so no toolset
was built), or a bound server contributed no tools.
"""

from __future__ import annotations

import logging
from typing import Any

from databricks_deep_research.tools.protocol import ResearchTool

logger = logging.getLogger(__name__)

__all__ = ["maybe_attach_mcp"]


def maybe_attach_mcp(
    tools: list[ResearchTool],
    mcp_servers: list[str],
    factory_context: Any | None,
) -> int:
    """Append the tools of each bound MCP server to *tools* (in place).

    Args:
        tools: the agent's resolved tool list (mutated in place when attaching).
        mcp_servers: ``config.mcp_servers`` — the server NAMES this agent binds.
        factory_context: the resolver's :class:`ToolFactoryContext`; its
            ``extras["_mcp_tools_by_server"]`` supplies the discovered tools.

    Returns:
        The number of tools appended (``0`` on any no-op path).
    """
    if not mcp_servers:
        return 0

    extras = getattr(factory_context, "extras", None) or {}
    by_server = extras.get("_mcp_tools_by_server")
    if not by_server:
        # No toolsets were built — e.g. the optional mcp/databricks-mcp packages
        # are absent, or every server failed to build. The host logs the cause
        # (MCP_SERVER_BUILD_FAILED / FWK_MCP_INJECTED); here we just stay a no-op.
        logger.warning(
            "MCP_AUTOATTACH_NO_TOOLS servers=%d — no discovered tools wired in "
            "the factory context (mcp/databricks-mcp not installed, or all "
            "servers failed to build)",
            len(mcp_servers),
        )
        return 0

    existing = {tool.definition.name for tool in tools}
    attached = 0
    for server_name in mcp_servers:
        for tool in by_server.get(server_name, []):
            if tool.definition.name in existing:
                continue
            tools.append(tool)
            existing.add(tool.definition.name)
            attached += 1

    if attached:
        logger.info(
            "MCP_TOOLS_AUTOATTACHED servers=%d tools=%d", len(mcp_servers), attached
        )
    return attached
