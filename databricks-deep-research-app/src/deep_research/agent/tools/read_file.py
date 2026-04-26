"""read_attached_file tool — paginated verbatim read of file chunks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from deep_research.agent.tools.base import ResearchContext, ToolDefinition, ToolResult
from deep_research.core.logging_utils import get_logger

if TYPE_CHECKING:
    from deep_research.services._protocols import IFileUploadService
    from deep_research.services.chat_memory_service import ChatMemoryService

logger = get_logger(__name__)


class ReadAttachedFileTool:
    """Return chunks of an attached file starting at ``offset`` (character index)."""

    def __init__(
        self,
        memory: "ChatMemoryService",
        file_service: "IFileUploadService",
    ) -> None:
        self._memory = memory
        self._file_service = file_service
        self._definition = ToolDefinition(
            name="read_attached_file",
            description=(
                "Read verbatim content from an attached file. Use this when "
                "you need exact quotes or values from a user-uploaded "
                "document. Supports offset/limit pagination. Call "
                "list_attached_files first to get the file_id."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "file_id": {
                        "type": "string",
                        "description": "UUID of the file (from list_attached_files).",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Character offset to start reading from (default 0).",
                        "default": 0,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum characters to return (default 4000, max 10000).",
                        "default": 4000,
                    },
                },
                "required": ["file_id"],
            },
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
        raw_id = arguments.get("file_id", "")
        offset = max(0, int(arguments.get("offset", 0) or 0))
        limit = min(10_000, max(1, int(arguments.get("limit", 4000) or 4000)))

        try:
            file_id = UUID(str(raw_id))
        except (TypeError, ValueError):
            return ToolResult(
                content=f"Invalid file_id: {raw_id!r}. Call list_attached_files first.",
                success=False,
                error="invalid_file_id",
            )

        snapshot = self._memory.snapshot()
        known_ids = {f.id for f in snapshot.files}
        if file_id not in known_ids:
            return ToolResult(
                content=(
                    f"file_id {file_id} is not attached to this conversation. "
                    f"Call list_attached_files to see available files."
                ),
                success=False,
                error="file_not_attached",
            )

        try:
            chunks = await self._file_service.get_file_chunks(file_id)
        except Exception as e:
            logger.warning(
                "READ_ATTACHED_FILE_FAILED file_id=%s error=%s",
                file_id, str(e)[:200],
            )
            return ToolResult(
                content=(
                    f"Could not read file {file_id}: {type(e).__name__}. "
                    "The file may have expired; use the file summary instead."
                ),
                success=False,
                error="file_read_failed",
            )

        # Stitch chunks in order, then slice by offset/limit.
        full_text = "".join(c.content or "" for c in chunks)
        total_len = len(full_text)
        slice_ = full_text[offset : offset + limit]

        return ToolResult(
            content=slice_ or "(no content at the requested offset)",
            success=True,
            sources=[
                {
                    "type": "file_search",
                    "file_id": str(file_id),
                    "title": f"Attached file {file_id}",
                    "offset": offset,
                    "limit": limit,
                }
            ],
            data={
                "file_id": str(file_id),
                "offset": offset,
                "limit": limit,
                "returned_chars": len(slice_),
                "total_chars": total_len,
                "has_more": offset + limit < total_len,
            },
        )
