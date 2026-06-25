"""Harness skills-section injection (Feature 2.2 runtime wiring).

Covers the framework half of A1: ``_render_attached_skills_section`` resolves
attached-skill metadata from the wired ``read_skill`` tool's store and renders
the metadata-only prompt section, and degrades gracefully when no store/tool is
attached.
"""

from __future__ import annotations

from databricks_deep_research.agents.harness import _render_attached_skills_section
from databricks_deep_research.skills.models import Skill, SkillMeta
from databricks_deep_research.tools.builtins.read_skill import ReadSkillTool


class _FakeStore:
    """Minimal SkillStore: only get_skill/list_skills are exercised here."""

    def __init__(self, skills: list[Skill]) -> None:
        self._skills = {s.name: s for s in skills}

    async def list_skills(self) -> list[SkillMeta]:
        return [s.meta for s in self._skills.values()]

    async def get_skill(self, name: str) -> Skill | None:
        return self._skills.get(name)

    async def put_skill(self, skill: Skill, *, scan: object) -> None:  # noqa: ARG002
        raise NotImplementedError


def _skill(name: str, desc: str) -> Skill:
    return Skill(name=name, description=desc, body="# body\nmethodology")


async def test_render_lists_only_attached_skills() -> None:
    store = _FakeStore(
        [
            _skill("pricing_playbook", "How to analyze pricing"),
            _skill("unused_skill", "Should not appear"),
        ]
    )
    tool = ReadSkillTool(skill_store=store)

    section = await _render_attached_skills_section(["pricing_playbook"], [tool])

    assert "## Available Skills" in section
    assert "pricing_playbook" in section
    assert "How to analyze pricing" in section
    # Only the attached name is rendered, not every skill in the store.
    assert "unused_skill" not in section


async def test_render_returns_empty_without_read_skill_tool() -> None:
    # No read_skill tool attached (store not wired) => graceful empty section.
    section = await _render_attached_skills_section(["pricing_playbook"], [])
    assert section == ""


async def test_render_returns_empty_when_names_do_not_resolve() -> None:
    tool = ReadSkillTool(skill_store=_FakeStore([]))
    section = await _render_attached_skills_section(["missing"], [tool])
    # render_skills_section("") on an empty meta list => "".
    assert section == ""
