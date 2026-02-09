"""External MCP client for AIditor integration."""

import json
import logging
from typing import Optional

from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient

from ..models import ExternalConnectionInfo, ExternalQueryResponse, ExternalSearchResult
from ._helpers import mcp_call_tool, mcp_list_tools, unwrap_exception

logger = logging.getLogger(__name__)


class ExternalClient:
    """Client for external MCP integrations via UC connections."""

    def __init__(self, workspace_client: Optional[WorkspaceClient] = None):
        self.client = workspace_client

    async def list_connections(self) -> list[ExternalConnectionInfo]:
        """List available external connections from UC."""
        connections: list[ExternalConnectionInfo] = []
        if not self.client:
            return connections
        try:
            for conn in self.client.connections.list():
                conn_name = getattr(conn, "name", None)
                conn_type = getattr(conn, "connection_type", None)
                conn_comment = getattr(conn, "comment", None) or ""
                conn_type_str = conn_type.value if hasattr(conn_type, "value") else str(conn_type) if conn_type else "unknown"
                if conn_name:
                    connections.append(
                        ExternalConnectionInfo(
                            name=conn_name, type=conn_type_str,
                            status=f"active — {conn_comment}" if conn_comment else "active",
                        )
                    )
        except Exception as exc:
            logger.error("Failed to list UC connections: %s", exc)
        return connections

    async def search(self, connection_name: str, query: str, max_results: int = 3) -> ExternalQueryResponse:
        """Execute a search query via an external MCP server."""
        if not self.client:
            return ExternalQueryResponse(
                results=[], markdown_summary="",
                error=f"No workspace client available to query connection '{connection_name}'",
            )
        try:
            mcp_client = self._get_mcp_client(connection_name)
            tool_name = self._infer_search_tool_name(connection_name)
            arguments = {"query": query, "max_results": max_results, "search_depth": "basic"}
            result = await mcp_call_tool(mcp_client, tool_name, arguments)
            return self._parse_mcp_result(query, result, max_results)
        except Exception as exc:
            real_msg = unwrap_exception(exc)
            logger.error("MCP call_tool failed for '%s': %s", connection_name, real_msg)
            return ExternalQueryResponse(
                results=[], markdown_summary="",
                error=f"MCP call_tool failed for connection '{connection_name}': {real_msg}",
            )

    def _get_mcp_client(self, connection_name: str):
        host = self.client.config.host.rstrip("/")
        server_url = f"{host}/api/2.0/mcp/external/{connection_name}"
        return DatabricksMCPClient(server_url=server_url, workspace_client=self.client)

    @staticmethod
    def _infer_search_tool_name(connection_name: str) -> str:
        lower = connection_name.lower()
        if "tavily" in lower:
            return "tavily_search"
        return "search"

    @staticmethod
    def _parse_mcp_result(query: str, result, max_results: int) -> ExternalQueryResponse:
        results: list[ExternalSearchResult] = []
        content_items = getattr(result, "content", []) or []
        for item in content_items:
            text = getattr(item, "text", None) or ""
            if not text:
                continue
            try:
                data = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                results.append(ExternalSearchResult(title="Search Result", url="", snippet=text[:500]))
                continue

            if isinstance(data, dict) and "results" in data:
                for r in data["results"][:max_results]:
                    results.append(
                        ExternalSearchResult(
                            title=r.get("title", ""), url=r.get("url", ""),
                            snippet=(r.get("content", "") or "")[:200] + "...",
                        )
                    )
            elif isinstance(data, list):
                for r in data[:max_results]:
                    if isinstance(r, dict):
                        results.append(
                            ExternalSearchResult(
                                title=r.get("title", ""), url=r.get("url", ""),
                                snippet=(r.get("content", "") or r.get("snippet", "") or str(r))[:200] + "...",
                            )
                        )
            else:
                results.append(ExternalSearchResult(title="Search Result", url="", snippet=json.dumps(data)[:500]))

        md_lines = [f'> **Web Search Results** for "{query}":', ">"]
        for i, r in enumerate(results, 1):
            md_lines.append(f"> {i}. **{r.title}**")
            md_lines.append(f">    {r.snippet}")
            if r.url:
                md_lines.append(f">    Source: [{r.url}]({r.url})")
            md_lines.append(">")
        if not results:
            md_lines.append("> No results found.")

        return ExternalQueryResponse(results=results, markdown_summary="\n".join(md_lines))
