"""Shared helpers for MCP client modules.

Provides async wrappers around ``DatabricksMCPClient`` to avoid the
``asyncio.run()`` / ``run_in_executor`` conflict, and a utility to unwrap
Python 3.11+ ``ExceptionGroup``s so error messages are actionable.
"""

from __future__ import annotations

import logging
from typing import Any

from databricks_mcp import DatabricksMCPClient

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Async wrappers
# ------------------------------------------------------------------


async def mcp_list_tools(client: DatabricksMCPClient):
    """Async version of ``client.list_tools()``."""
    return await client._get_tools_async()


async def mcp_call_tool(
    client: DatabricksMCPClient,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
):
    """Async version of ``client.call_tool(tool_name, arguments)``."""
    return await client._call_tools_async(tool_name, arguments)


# ------------------------------------------------------------------
# ExceptionGroup unwrapper
# ------------------------------------------------------------------

def unwrap_exception(exc: BaseException) -> str:
    """Return a human-readable message from *exc*.

    If *exc* is a Python 3.11+ ``ExceptionGroup`` (or ``BaseExceptionGroup``),
    recursively extracts the first sub-exception so the real error is visible.
    """
    if hasattr(exc, "exceptions"):
        sub_exceptions = exc.exceptions  # type: ignore[union-attr]
        messages: list[str] = []
        for sub in sub_exceptions:
            messages.append(unwrap_exception(sub))
        return "; ".join(messages) if messages else str(exc)
    return str(exc)
