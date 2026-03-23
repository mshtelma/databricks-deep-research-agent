"""Synthesis tools for ReAct-based report generation.

Provides tool definitions and execution for the REACT synthesis mode.
The LLM calls ``search_evidence`` and ``read_snippet`` to retrieve
evidence on demand, keeping the prompt at ~5-10K tokens regardless
of pool size.

Search delegates to ``PoolRegistry`` -- zero duplicated BM25/embedding code.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from databricks_deep_research.citation.types import RankedEvidence
from databricks_deep_research.llm.client import FrameworkLLMClient, LLMResponse
from databricks_deep_research.pools.pool_registry import PoolRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------

EVIDENCE_SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_evidence",
        "description": (
            "Search the evidence pool for spans relevant to a query. "
            "Returns previews with indices. Use read_snippet to get full text."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query describing the evidence you need.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (1-10, default 5).",
                },
            },
            "required": ["query"],
        },
    },
}

EVIDENCE_READ_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_snippet",
        "description": (
            "Read the full text of an evidence span by its index. "
            "Returns the quote, source title, and citation key."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "index": {
                    "type": "integer",
                    "description": "Evidence index from search_evidence results.",
                },
            },
            "required": ["index"],
        },
    },
}

SYNTHESIS_TOOLS: list[dict[str, Any]] = [EVIDENCE_SEARCH_TOOL, EVIDENCE_READ_TOOL]


# ---------------------------------------------------------------------------
# Evidence search index (thin wrapper around PoolRegistry)
# ---------------------------------------------------------------------------

class EvidenceSearchIndex:
    """Thin wrapper: populates a PoolState, delegates search to PoolRegistry.

    Reuses the framework's full hybrid search stack:
    1. Hybrid (BM25 + vector) -- bm25s + embedding model configured
    2. BM25 only -- bm25s installed, no embeddings
    3. Keyword fallback -- word overlap
    4. Chronological -- get_recent()
    """

    _POOL_NAME = "_synthesis_evidence"

    def __init__(self, registry: PoolRegistry) -> None:
        self._registry = registry

    @classmethod
    def create(
        cls,
        evidence_pool: list[RankedEvidence],
        llm_client: FrameworkLLMClient | None = None,
    ) -> EvidenceSearchIndex:
        """Build search index by populating a PoolState with evidence items."""
        registry = PoolRegistry(llm_client=llm_client)
        pool = registry.get_or_create(cls._POOL_NAME)

        for i, e in enumerate(evidence_pool):
            pool.add({
                "_idx": i,
                "title": e.source_title or "",
                "text": e.quote_text,
                "section": e.section_heading or "",
            })

        logger.info(
            "EVIDENCE_SEARCH_INDEX_CREATED items=%d",
            len(evidence_pool),
        )
        return cls(registry)

    async def search(self, query: str, limit: int = 5) -> list[int]:
        """Return evidence indices sorted by relevance."""
        results = await self._registry.search(self._POOL_NAME, query, top_k=limit)
        return [
            item["_idx"]
            for item in results
            if isinstance(item, dict) and "_idx" in item
        ]


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

class SynthesisToolExecutor:
    """Execute synthesis tool calls against the evidence pool."""

    def __init__(
        self,
        evidence_pool: list[RankedEvidence],
        key_map: dict[int, str],
        search_index: EvidenceSearchIndex,
    ) -> None:
        self._pool = evidence_pool
        self._keys = key_map
        self._search_index = search_index
        self.read_indices: set[int] = set()

    async def execute(self, tool_name: str, raw_arguments: str) -> str:
        """Execute a synthesis tool. *raw_arguments* is a JSON string."""
        try:
            args = json.loads(raw_arguments)
        except json.JSONDecodeError:
            return f"Invalid JSON arguments: {raw_arguments[:100]}"

        if tool_name == "search_evidence":
            return await self._search(args.get("query", ""), args.get("limit", 5))
        if tool_name == "read_snippet":
            return self._read(args.get("index", -1))
        return f"Unknown tool: {tool_name}"

    async def _search(self, query: str, limit: int) -> str:
        indices = await self._search_index.search(query, limit=min(limit, 10))
        if not indices:
            return "No matching evidence found. Try a broader or different query."
        lines = ["Found relevant evidence:"]
        for idx in indices:
            if not (0 <= idx < len(self._pool)):
                continue
            e = self._pool[idx]
            numeric = " [NUMERIC]" if e.has_numeric_content else ""
            lines.append(
                f"[{idx}] {e.source_title or 'Unknown'}{numeric}\n"
                f"    Preview: {e.quote_text[:200]}..."
            )
        lines.append("\nUse read_snippet with an index to see the full quote.")
        return "\n".join(lines)

    def _read(self, index: int) -> str:
        if not (0 <= index < len(self._pool)):
            return f"Invalid index {index}. Valid range: 0-{len(self._pool) - 1}"
        e = self._pool[index]
        self.read_indices.add(index)
        key = self._keys.get(index, f"Source-{index}")
        section = f" (Section: {e.section_heading})" if e.section_heading else ""
        return (
            f"Evidence [{index}] from {e.source_title or 'Unknown'}{section}:\n"
            f'"{e.quote_text}"\n\n'
            f"Citation key: [{key}]\n"
            f'Use: <cite key="{key}">Your claim based on this evidence.</cite>'
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_assistant_message(response: LLMResponse) -> dict[str, Any]:
    """Convert LLMResponse to OpenAI assistant message format."""
    msg: dict[str, Any] = {"role": "assistant"}
    if response.content:
        msg["content"] = response.content
    if response.tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function_name,
                    "arguments": tc.arguments,
                },
            }
            for tc in response.tool_calls
        ]
    return msg


def build_evidence_source_index(
    evidence_pool: list[RankedEvidence],
    key_map: dict[int, str],
) -> str:
    """Build a compact evidence source index for the LLM prompt.

    Groups evidence by source, showing citation keys, span counts,
    numeric flags, and top section headings. Gives the LLM a map
    of what's available without including any actual evidence text.
    """
    source_info: dict[str, dict[str, Any]] = {}
    for idx, e in enumerate(evidence_pool):
        title = e.source_title or "Unknown"
        if title not in source_info:
            source_info[title] = {
                "key": key_map.get(idx, "Source"),
                "count": 0,
                "numeric": 0,
                "sections": set(),
            }
        info = source_info[title]
        info["count"] += 1
        if e.has_numeric_content:
            info["numeric"] += 1
        if e.section_heading:
            info["sections"].add(e.section_heading)

    lines = [
        f"Evidence map ({len(evidence_pool)} spans from {len(source_info)} sources):"
    ]
    for title, info in source_info.items():
        numeric = f", {info['numeric']} numeric" if info["numeric"] else ""
        sections = ""
        if info["sections"]:
            section_list = sorted(info["sections"])[:3]
            sections = f" — {', '.join(section_list)}"
        lines.append(
            f"  [{info['key']}] {title}: {info['count']} spans{numeric}{sections}"
        )

    return "\n".join(lines)
