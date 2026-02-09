"""Vector Search client for AIditor integration."""

import json
import logging
from typing import Any, Optional

from databricks_mcp import DatabricksMCPClient

from ..models import VectorIndexInfo, VectorSearchResponse, VectorSearchResult
from ._helpers import mcp_call_tool, mcp_list_tools, unwrap_exception

logger = logging.getLogger(__name__)


class VectorSearchClient:
    """Client for interacting with Databricks Vector Search via MCP."""

    def __init__(self, workspace_client: Optional[Any] = None):
        self.client = workspace_client

    async def list_indexes(self) -> list[VectorIndexInfo]:
        """List available Vector Search indexes."""
        if self.client is None:
            return []
        try:
            indexes: list[VectorIndexInfo] = []
            for endpoint in self.client.vector_search_endpoints.list_endpoints():
                endpoint_name = getattr(endpoint, "name", None)
                if not endpoint_name:
                    continue
                try:
                    for idx in self.client.vector_search_indexes.list_indexes(endpoint_name):
                        num_docs = 0
                        status = getattr(idx, "status", None)
                        if status and getattr(status, "indexed_row_count", None) is not None:
                            num_docs = int(status.indexed_row_count)
                        indexes.append(
                            VectorIndexInfo(name=idx.name or "", endpoint=endpoint_name, num_docs=num_docs)
                        )
                except Exception as exc:
                    logger.warning("Failed to list indexes for endpoint %s: %s", endpoint_name, exc)
            return indexes
        except Exception as exc:
            logger.error("Failed to list Vector Search endpoints: %s", exc)
            return []

    async def search(self, index_name: str, query: str, num_results: int = 5) -> VectorSearchResponse:
        """Execute a semantic search query via the managed MCP server."""
        if self.client is None:
            return VectorSearchResponse(results=[], markdown_list="", error="No Databricks client configured")

        try:
            host = self.client.config.host.rstrip("/")
            parts = index_name.split(".")
            if len(parts) < 2:
                return VectorSearchResponse(
                    results=[], markdown_list="",
                    error=f"Index name must be at least catalog.schema, got '{index_name}'",
                )

            url_path = "/".join(parts)
            server_url = f"{host}/api/2.0/mcp/vector-search/{url_path}"
            mcp_client = DatabricksMCPClient(server_url=server_url, workspace_client=self.client)

            tools = await mcp_list_tools(mcp_client)
            tool_names = [getattr(t, "name", str(t)) for t in tools]

            search_tool = None
            for name in tool_names:
                if any(kw in name.lower() for kw in ("search", "query", "similarity")):
                    search_tool = name
                    break
            if search_tool is None and tool_names:
                search_tool = tool_names[0]

            if search_tool is None:
                return VectorSearchResponse(
                    results=[], markdown_list="",
                    error=f"No tools found on Vector Search MCP server for '{index_name}'",
                )

            result = await mcp_call_tool(mcp_client, search_tool, {"query": query, "num_results": num_results})
            return self._parse_mcp_result(query, result, num_results)

        except Exception as exc:
            real_msg = unwrap_exception(exc)
            logger.error("Vector Search MCP query failed for %s: %s", index_name, real_msg)
            return VectorSearchResponse(results=[], markdown_list="", error=f"Vector search failed: {real_msg}")

    @staticmethod
    def _parse_mcp_result(query: str, result, num_results: int) -> VectorSearchResponse:
        results: list[VectorSearchResult] = []
        content_items = getattr(result, "content", []) or []
        for item in content_items:
            text = getattr(item, "text", None) or ""
            if not text:
                continue
            try:
                data = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                results.append(VectorSearchResult(text=text[:500], source=None, score=0.0))
                continue

            if isinstance(data, dict):
                rows = data.get("results", data.get("matches", []))
                if isinstance(rows, list):
                    for row in rows[:num_results]:
                        if isinstance(row, dict):
                            doc_text = row.get("text", "") or row.get("content", "") or row.get("page_content", "") or str(row)
                            score = float(row.get("score", 0.0))
                            source = row.get("source") or row.get("metadata", {}).get("source")
                            results.append(VectorSearchResult(text=doc_text[:500], source=source, score=min(max(score, 0.0), 1.0)))
                elif not rows:
                    results.append(VectorSearchResult(text=json.dumps(data)[:500], source=None, score=0.0))
            elif isinstance(data, list):
                for row in data[:num_results]:
                    if isinstance(row, dict):
                        doc_text = row.get("text", "") or row.get("content", "") or str(row)
                        score = float(row.get("score", 0.0))
                        results.append(VectorSearchResult(text=doc_text[:500], source=row.get("source"), score=min(max(score, 0.0), 1.0)))

        md_lines = [f'> **Related Documents** (based on: "{query}"):', ">"]
        for r in results[:num_results]:
            score_str = f"{r.score:.0%} relevance" if r.score > 0 else ""
            suffix = f" ({score_str})" if score_str else ""
            md_lines.append(f"> - {r.text[:120]}...{suffix}")
        if not results:
            md_lines.append("> No results found.")

        return VectorSearchResponse(results=results[:num_results], markdown_list="\n".join(md_lines))
