"""list_attached_files tool — reads the hydrated ChatMemoryService snapshot.

Exposed to every agent when the chat has uploaded files. Zero I/O at
call time: reads from the pre-built in-memory projection built by
``ChatMemoryService.hydrate``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from deep_research.agent.tools.base import ResearchContext, ToolDefinition, ToolResult
from deep_research.core.logging_utils import get_logger

if TYPE_CHECKING:
    from deep_research.services.chat_memory_service import ChatMemoryService

logger = get_logger(__name__)


class ListAttachedFilesTool:
    """Return the list of files attached to the current chat with one-line summaries."""

    def __init__(self, memory: ChatMemoryService) -> None:
        self._memory = memory
        self._definition = ToolDefinition(
            name="list_attached_files",
            description=(
                "List all files the user has attached to this conversation, "
                "with filename, type, and a one-line summary. Call this when "
                "you want to know what documents are available before "
                "searching or reading their contents."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            source_type="file_search",
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ResearchContext,
    ) -> ToolResult:
        snapshot = self._memory.snapshot()
        if not snapshot.files:
            return ToolResult(
                content="No files are attached to this conversation.",
                success=True,
                sources=[],
                data={"files": []},
            )

        lines = ["Attached files:"]
        for f in snapshot.files:
            lines.append(
                f"- id={f.id} filename={f.filename} type={f.file_type} "
                f"size={f.size} chunks={f.chunk_count} summary={f.one_line_summary!r}"
            )
        return ToolResult(
            content="\n".join(lines),
            success=True,
            sources=[],
            data={"files": [f.model_dump(mode="json") for f in snapshot.files]},
        )
