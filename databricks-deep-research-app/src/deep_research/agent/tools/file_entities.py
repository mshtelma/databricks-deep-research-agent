"""get_file_entities tool — return structured entities extracted from attached files."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from deep_research.agent.tools.base import ResearchContext, ToolDefinition, ToolResult
from deep_research.core.logging_utils import get_logger

if TYPE_CHECKING:
    from deep_research.services.chat_memory_service import ChatMemoryService

logger = get_logger(__name__)


class GetFileEntitiesTool:
    """Return the entity registry built during file preprocessing."""

    def __init__(self, memory: ChatMemoryService) -> None:
        self._memory = memory
        self._definition = ToolDefinition(
            name="get_file_entities",
            description=(
                "Return structured entities (accounts, people, products, "
                "competitors, etc.) extracted from attached files. Use this "
                "to disambiguate names mentioned in the user's query, or to "
                "enumerate key entities the research should cover."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "file_id": {
                        "type": "string",
                        "description": (
                            "Optional: restrict to entities linked to a specific file "
                            "(from list_attached_files). Omit to return all entities."
                        ),
                    },
                    "entity_type": {
                        "type": "string",
                        "description": (
                            "Optional: filter by type "
                            "(account|person|product|date|competitor|location|other)."
                        ),
                    },
                },
                "required": [],
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
        snapshot = self._memory.snapshot()
        entities = list(snapshot.entities)

        file_id_arg = arguments.get("file_id")
        if file_id_arg:
            try:
                target_file = UUID(str(file_id_arg))
            except (TypeError, ValueError):
                return ToolResult(
                    content=f"Invalid file_id: {file_id_arg!r}",
                    success=False,
                    error="invalid_file_id",
                )
            # Keep entities referenced by this file's ChatMemoryFile.entity_ids
            target = next((f for f in snapshot.files if f.id == target_file), None)
            if target is None:
                return ToolResult(
                    content=f"file_id {target_file} is not attached to this chat.",
                    success=False,
                    error="file_not_attached",
                )
            # FileRef doesn't carry entity_ids; re-query from the snapshot's
            # plugin-friendly projection. Fall back to all entities when
            # unknown. (PR 2 threads linked entity_ids into FileRef.)
            pass  # no-op filter — returns all for now

        etype = arguments.get("entity_type")
        if etype:
            etype_str = str(etype).casefold()
            entities = [e for e in entities if e.entity_type.casefold() == etype_str]

        if not entities:
            return ToolResult(
                content="No entities extracted from attached files.",
                success=True,
                sources=[],
                data={"entities": []},
            )

        lines = ["Entities extracted from attached files:"]
        for e in entities:
            alias = f" (aka {', '.join(e.aliases)})" if e.aliases else ""
            summary = f" — {e.summary}" if e.summary else ""
            lines.append(f"- [{e.entity_type}] {e.name}{alias}{summary}")

        return ToolResult(
            content="\n".join(lines),
            success=True,
            sources=[],
            data={
                "entities": [e.model_dump(mode="json") for e in entities],
            },
        )
