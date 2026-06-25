"""``read_skill`` tool — progressive disclosure of a skill's full body.

The prompt injector lists attached skills' METADATA only (name + description).
When the agent decides to use one, it calls ``read_skill`` with the skill name
to load the full Markdown body into context.  Scripts are NOT returned — they
are executed in the compute scratchpad, never read into the LLM context.

Backend-agnostic: depends only on the :class:`SkillStore` protocol, so the same
tool works over the bundled :class:`FilesystemSkillStore` or the app's
Lakebase-backed store.
"""

from __future__ import annotations

from typing import Any

from databricks_deep_research.skills.store import SkillStore
from databricks_deep_research.tools.protocol import (
    SourceKind,
    ToolContext,
    ToolDefinition,
    ToolResult,
)

__all__ = ["ReadSkillTool"]

_DEFAULT_DESCRIPTION = (
    "Load the full instructions for an attached skill by name. The available "
    "skills are listed (name + description) in your system prompt; call this "
    "tool with a skill's exact name to read its complete methodology before "
    "applying it. Returns the skill body as markdown."
)


class ReadSkillTool:
    """Returns a named skill's full body from a :class:`SkillStore`."""

    def __init__(
        self,
        skill_store: SkillStore,
        *,
        name: str = "read_skill",
        description: str = "",
    ) -> None:
        self._store = skill_store
        self._name = name
        self._description = description or _DEFAULT_DESCRIPTION

    @property
    def store(self) -> SkillStore:
        """The backing :class:`SkillStore`.

        Exposed so the agent harness can resolve attached-skill metadata for the
        prompt's skills section from the SAME store the tool reads bodies from
        (avoids threading a second store reference through the harness).
        """
        return self._store

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description=self._description,
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Exact name of the skill to load.",
                    },
                },
                "required": ["name"],
                "additionalProperties": False,
            },
            source_type="builtin",
            source_kind=SourceKind.builtin,
            metadata={"budget_free": True},
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raw = arguments.get("name")
        name = raw.strip() if isinstance(raw, str) else ""
        if not name:
            raise ValueError("read_skill requires a non-empty 'name' argument")
        return {"name": name}

    async def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        del context  # required by the ResearchTool protocol; unused here
        name = arguments["name"]
        skill = await self._store.get_skill(name)
        if skill is None:
            # Graceful miss — surface what is available rather than failing hard,
            # so the agent can self-correct on a typo'd name.
            available = ", ".join(meta.name for meta in await self._store.list_skills())
            hint = f" Available skills: {available}." if available else ""
            return ToolResult(
                content=f"No skill named {name!r} was found.{hint}",
                success=True,
                data={"source_kind": SourceKind.builtin, "skill_found": False},
            )

        return ToolResult(
            content=skill.body,
            success=True,
            data={
                "source_kind": SourceKind.builtin,
                "skill_found": True,
                "skill_name": skill.name,
            },
        )
