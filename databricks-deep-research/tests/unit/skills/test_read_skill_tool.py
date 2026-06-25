"""Unit tests for the read_skill ResearchTool (progressive disclosure)."""

from __future__ import annotations

import pytest

from databricks_deep_research.skills.models import Skill, SkillMeta
from databricks_deep_research.tools.builtins.read_skill import ReadSkillTool
from databricks_deep_research.tools.protocol import SourceKind, ToolContext


class _StubStore:
    """Minimal SkillStore stub returning a fixed set of skills."""

    def __init__(self, skills: list[Skill]) -> None:
        self._by_name = {s.name: s for s in skills}

    async def list_skills(self) -> list[SkillMeta]:
        return [s.meta for s in self._by_name.values()]

    async def get_skill(self, name: str) -> Skill | None:
        return self._by_name.get(name)

    async def put_skill(self, skill, *, scan):  # type: ignore[no-untyped-def]
        raise NotImplementedError


def _store() -> _StubStore:
    return _StubStore(
        [
            Skill(
                name="deep-research",
                description="Research methodology.",
                body="# Deep Research\n\nThe full body content.",
            )
        ]
    )


def _ctx() -> ToolContext:
    return ToolContext(query="")


class TestDefinition:
    def test_definition_shape(self) -> None:
        tool = ReadSkillTool(_store())
        defn = tool.definition
        assert defn.name == "read_skill"
        assert defn.source_kind == SourceKind.builtin
        assert "name" in defn.parameters["properties"]
        assert defn.parameters["required"] == ["name"]


class TestValidateArguments:
    def test_strips_name(self) -> None:
        tool = ReadSkillTool(_store())
        assert tool.validate_arguments({"name": "  chart  "}) == {"name": "chart"}

    def test_rejects_empty_name(self) -> None:
        tool = ReadSkillTool(_store())
        with pytest.raises(ValueError, match="non-empty"):
            tool.validate_arguments({"name": "   "})

    def test_rejects_missing_name(self) -> None:
        tool = ReadSkillTool(_store())
        with pytest.raises(ValueError, match="non-empty"):
            tool.validate_arguments({})


class TestExecute:
    async def test_returns_body_for_valid_name(self) -> None:
        tool = ReadSkillTool(_store())
        args = tool.validate_arguments({"name": "deep-research"})
        result = await tool.execute(args, _ctx())
        assert result.success is True
        assert "The full body content." in result.content
        assert result.data["skill_found"] is True
        assert result.data["skill_name"] == "deep-research"

    async def test_graceful_miss_for_unknown_name(self) -> None:
        tool = ReadSkillTool(_store())
        args = tool.validate_arguments({"name": "nonexistent"})
        result = await tool.execute(args, _ctx())
        # Graceful — success True, found False, and lists what is available.
        assert result.success is True
        assert result.data["skill_found"] is False
        assert "nonexistent" in result.content
        assert "deep-research" in result.content
