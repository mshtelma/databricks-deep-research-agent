"""search_chat_memory tool — recall prior verified findings before web search.

Reads the hydrated ChatMemoryService projection (built by ``hydrate``); the
keyword stub becomes hybrid in Phase 3c with no signature change. Exposed to
research agents whenever the chat has accumulated memory.

Citation contract (Codex §7): findings are prior CONCLUSIONS for orientation,
not citable evidence. The tool surfaces no citable sources and tells the model
to re-ground against a source before citing — so memory recall never becomes
paraphrase-without-grounding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from deep_research.agent.tools.base import ResearchContext, ToolDefinition, ToolResult
from deep_research.core.logging_utils import get_logger

if TYPE_CHECKING:
    from deep_research.services.chat_memory_service import ChatMemoryService

logger = get_logger(__name__)


class SearchChatMemoryTool:
    """Search this chat's durable memory (verified findings from prior turns)."""

    def __init__(self, memory: ChatMemoryService) -> None:
        self._memory = memory
        self._definition = ToolDefinition(
            name="search_chat_memory",
            description=(
                "Search what you already learned earlier in THIS conversation "
                "(verified findings from prior research turns). Call this BEFORE "
                "web search to decide what still needs researching. Returns prior "
                "findings with confidence labels. IMPORTANT: these are prior "
                "CONCLUSIONS for orientation — do NOT cite them directly; if you "
                "use a fact, re-ground it against a source (web_search / the "
                "seeded source pool) before citing."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to recall."},
                    "k": {"type": "integer", "description": "Max results (default 5)."},
                },
                "required": ["query"],
            },
            source_type="file_search",
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(
        self,
        _arguments: dict[str, Any],
        _context: ResearchContext,
    ) -> ToolResult:
        query = str(_arguments.get("query") or "").strip()
        k = int(_arguments.get("k") or 5)
        if not query:
            return ToolResult(content="Provide a query to search memory.", success=True)
        findings = await self._memory.search_findings(query, k=k)
        if not findings:
            return ToolResult(
                content="No prior findings in this conversation match that query.",
                success=True,
                sources=[],
                data={"findings": []},
            )
        # KnowledgeFinding is a typed projection (.content/.confidence) — no
        # getattr/hasattr introspection (Constitution / Codex §8).
        lines = [
            "Prior findings from this conversation "
            "(orientation only — re-ground against a source before citing; "
            "do not cite these directly):"
        ]
        for f in findings:
            lines.append(f"- [{f.confidence}] {f.content}")
        return ToolResult(
            content="\n".join(lines),
            success=True,
            sources=[],  # no citable sources surfaced by design
            data={
                "findings": [
                    {"content": f.content, "confidence": f.confidence} for f in findings
                ]
            },
        )
