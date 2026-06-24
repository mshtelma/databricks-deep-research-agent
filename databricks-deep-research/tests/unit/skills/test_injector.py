"""Unit tests for the metadata-only skills prompt injector."""

from __future__ import annotations

from databricks_deep_research.skills.injector import render_skills_section
from databricks_deep_research.skills.models import Skill, SkillMeta


def _meta(name: str, desc: str) -> SkillMeta:
    return SkillMeta(name=name, description=desc)


class TestRenderSkillsSection:
    def test_empty_returns_empty_string(self) -> None:
        assert render_skills_section([]) == ""

    def test_lists_name_and_description(self) -> None:
        section = render_skills_section(
            [_meta("deep-research", "Multi-angle research method.")]
        )
        assert "**deep-research**" in section
        assert "Multi-angle research method." in section

    def test_mentions_read_skill_tool(self) -> None:
        section = render_skills_section([_meta("x", "y")])
        assert "read_skill" in section

    def test_renders_metadata_only_not_body(self) -> None:
        skill = Skill(
            name="secret",
            description="A one-line description.",
            body="SENSITIVE BODY CONTENT that must not leak into the prompt.",
            scripts={"run": "print('do not leak')"},
        )
        section = render_skills_section([skill])
        assert "secret" in section
        assert "A one-line description." in section
        # The body and scripts must NOT appear — metadata only.
        assert "SENSITIVE BODY CONTENT" not in section
        assert "do not leak" not in section

    def test_accepts_mixed_meta_and_skill(self) -> None:
        skill = Skill(name="a", description="da", body="body a")
        meta = _meta("b", "db")
        section = render_skills_section([skill, meta])
        assert "**a**" in section
        assert "**b**" in section
        assert "body a" not in section

    def test_lists_all_in_order(self) -> None:
        section = render_skills_section(
            [_meta("one", "d1"), _meta("two", "d2"), _meta("three", "d3")]
        )
        assert section.index("one") < section.index("two") < section.index("three")
