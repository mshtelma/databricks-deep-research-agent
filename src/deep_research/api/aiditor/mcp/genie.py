"""Genie Space client for AIditor.

Discovers Genie Spaces via the Databricks SDK and queries them through the
managed MCP server at ``/api/2.0/mcp/genie/{space_id}`` using
``DatabricksMCPClient``.
"""

import asyncio
import json
import logging
from typing import Any, Optional

from databricks_mcp import DatabricksMCPClient

from ..models import GenieQueryResponse, GenieSpaceInfo
from ._helpers import mcp_call_tool, mcp_list_tools, unwrap_exception

logger = logging.getLogger(__name__)

# Polling constants for async Genie queries
_POLL_INTERVAL_SECS = 2.0
_MAX_POLLS = 30  # 60 seconds max wait


class GenieClient:
    """Client for interacting with Databricks Genie Spaces via MCP."""

    def __init__(self, workspace_client: Optional[Any] = None):
        self.client = workspace_client

    async def list_spaces(self) -> list[GenieSpaceInfo]:
        """List available Genie Spaces."""
        if self.client is None:
            return []
        try:
            response = self.client.genie.list_spaces()
            spaces = response.spaces or []
            return [
                GenieSpaceInfo(
                    id=space.space_id, name=space.title, tables=[], status="active"
                )
                for space in spaces
            ]
        except Exception as exc:
            logger.error("Failed to list Genie Spaces: %s", exc)
            return []

    async def query(self, space_id: str, query: str) -> GenieQueryResponse:
        """Execute a natural-language query against a Genie Space via MCP."""
        if self.client is None:
            return GenieQueryResponse(
                status="ERROR", sql=None, columns=[], data=[],
                markdown_table="", error="No Databricks client configured",
            )

        try:
            host = self.client.config.host.rstrip("/")
            server_url = f"{host}/api/2.0/mcp/genie/{space_id}"
            mcp_client = DatabricksMCPClient(
                server_url=server_url, workspace_client=self.client,
            )

            tools = await mcp_list_tools(mcp_client)
            tool_names = [getattr(t, "name", str(t)) for t in tools]
            logger.info("Genie MCP tools for space %s: %s", space_id, tool_names)

            ask_tool = None
            for name in tool_names:
                if any(kw in name.lower() for kw in ("ask", "query", "genie", "execute")):
                    ask_tool = name
                    break
            if ask_tool is None and tool_names:
                ask_tool = tool_names[0]

            if ask_tool is None:
                return GenieQueryResponse(
                    status="ERROR", sql=None, columns=[], data=[],
                    markdown_table="",
                    error=f"No tools found on Genie MCP server for space '{space_id}'",
                )

            param_name = "content"
            for tool in tools:
                if getattr(tool, "name", None) == ask_tool:
                    schema = getattr(tool, "inputSchema", None) or {}
                    props = schema.get("properties", {}) if isinstance(schema, dict) else {}
                    if props:
                        required = schema.get("required", [])
                        param_name = required[0] if required else next(iter(props))
                    break

            result = await mcp_call_tool(mcp_client, ask_tool, {param_name: query})

            parsed_initial = self._extract_mcp_text(result)
            status = parsed_initial.get("status", "")
            conversation_id = parsed_initial.get("conversationId", "")
            message_id = parsed_initial.get("messageId", "")

            if (
                status
                and status not in ("COMPLETED", "FAILED", "CANCELLED")
                and conversation_id
                and message_id
            ):
                poll_tool = None
                for name in tool_names:
                    if "poll" in name.lower():
                        poll_tool = name
                        break

                if poll_tool:
                    for attempt in range(1, _MAX_POLLS + 1):
                        await asyncio.sleep(_POLL_INTERVAL_SECS)
                        poll_result = await mcp_call_tool(
                            mcp_client, poll_tool,
                            {"conversation_id": conversation_id, "message_id": message_id},
                        )
                        parsed_poll = self._extract_mcp_text(poll_result)
                        poll_status = parsed_poll.get("status", "")
                        if poll_status in ("COMPLETED", "FAILED", "CANCELLED"):
                            return self._parse_structured_result(parsed_poll)
                    return GenieQueryResponse(
                        status="ERROR", sql=None, columns=[], data=[],
                        markdown_table="", error="Genie query timed out after polling",
                    )

            return self._parse_structured_result(parsed_initial)

        except Exception as exc:
            real_msg = unwrap_exception(exc)
            logger.error("Genie MCP query failed for space %s: %s", space_id, real_msg)
            return GenieQueryResponse(
                status="ERROR", sql=None, columns=[], data=[],
                markdown_table="", error=f"Genie query failed: {real_msg}",
            )

    @staticmethod
    def _extract_mcp_text(result) -> dict:
        content_items = getattr(result, "content", []) or []
        text_parts: list[str] = []
        for item in content_items:
            text = getattr(item, "text", None) or ""
            if text:
                text_parts.append(text)
        combined_text = "\n".join(text_parts)
        try:
            parsed = json.loads(combined_text)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        return {"_raw_text": combined_text}

    @staticmethod
    def _parse_structured_result(parsed: dict) -> GenieQueryResponse:
        status = parsed.get("status", "COMPLETED")
        content = parsed.get("content", {})

        if status == "FAILED":
            error_text = ""
            text_attachments = content.get("textAttachments", [])
            if text_attachments:
                error_text = "\n".join(str(t) for t in text_attachments)
            return GenieQueryResponse(
                status="ERROR", sql=None, columns=[], data=[],
                markdown_table="", error=error_text or "Genie query failed",
            )

        sql = None
        columns: list[str] = []
        data: list[list[Any]] = []
        markdown_parts: list[str] = []

        query_attachments = content.get("queryAttachments", [])
        for qa in query_attachments:
            if isinstance(qa, dict):
                sql = qa.get("query") or qa.get("sql") or sql
                qa_columns = qa.get("columns", [])
                if qa_columns:
                    columns = [
                        c.get("name", str(c)) if isinstance(c, dict) else str(c)
                        for c in qa_columns
                    ]
                qa_data = qa.get("data", [])
                if qa_data:
                    data = qa_data

        text_attachments = content.get("textAttachments", [])
        for ta in text_attachments:
            if isinstance(ta, str):
                markdown_parts.append(ta)
            elif isinstance(ta, dict):
                markdown_parts.append(ta.get("text", str(ta)))

        if columns and data:
            md_lines = ["| " + " | ".join(columns) + " |"]
            md_lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
            for row in data:
                if isinstance(row, list):
                    md_lines.append("| " + " | ".join(str(v) for v in row) + " |")
                elif isinstance(row, dict):
                    md_lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
            markdown_parts.insert(0, "\n".join(md_lines))

        if sql:
            markdown_parts.append(f"\n```sql\n{sql}\n```")

        if not markdown_parts:
            raw = parsed.get("_raw_text", "")
            if raw:
                markdown_parts.append(raw)

        return GenieQueryResponse(
            status="COMPLETED", sql=sql, columns=columns, data=data,
            markdown_table="\n\n".join(markdown_parts) or "_No results_",
        )
