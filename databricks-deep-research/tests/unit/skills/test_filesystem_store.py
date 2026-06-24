"""Unit tests for FilesystemSkillStore over the bundled seed skills."""

from __future__ import annotations

import pytest

from databricks_deep_research.skills.store import (
    FilesystemSkillStore,
    SkillStoreError,
)

_EXPECTED_SEEDS = {"deep-research", "data-analysis", "chart"}


class _DummyScanner:
    async def scan(self, skill):  # type: ignore[no-untyped-def]
        from databricks_deep_research.skills.store import SkillScanResult  # noqa: F401

        class _R:
            safe = True
            reason = ""

        return _R()


class TestListSkills:
    async def test_lists_three_seed_metadatas(self) -> None:
        store = FilesystemSkillStore()
        metas = await store.list_skills()
        assert {m.name for m in metas} == _EXPECTED_SEEDS

    async def test_metadata_has_name_and_description(self) -> None:
        store = FilesystemSkillStore()
        metas = await store.list_skills()
        for meta in metas:
            assert meta.name
            assert meta.description
            # metadata only — SkillMeta has no body field at all
            assert not hasattr(meta, "body")

    async def test_sorted_by_name(self) -> None:
        store = FilesystemSkillStore()
        names = [m.name for m in await store.list_skills()]
        assert names == sorted(names)


class TestGetSkill:
    async def test_fetches_body_by_name(self) -> None:
        store = FilesystemSkillStore()
        skill = await store.get_skill("deep-research")
        assert skill is not None
        assert skill.name == "deep-research"
        # body carries the ported DeerFlow methodology content
        assert "Diversity" in skill.body
        assert "Temporal awareness" in skill.body

    async def test_unknown_name_returns_none(self) -> None:
        store = FilesystemSkillStore()
        assert await store.get_skill("does-not-exist") is None

    async def test_each_seed_parses_and_has_body(self) -> None:
        store = FilesystemSkillStore()
        for name in _EXPECTED_SEEDS:
            skill = await store.get_skill(name)
            assert skill is not None
            assert skill.body.strip()


class TestReadOnly:
    async def test_put_skill_raises_read_only(self) -> None:
        store = FilesystemSkillStore()
        skill = await store.get_skill("chart")
        assert skill is not None
        with pytest.raises(SkillStoreError, match="read-only"):
            await store.put_skill(skill, scan=_DummyScanner())
