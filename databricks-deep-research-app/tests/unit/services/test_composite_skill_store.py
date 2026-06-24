"""Tests for CompositeSkillStore (A1 skills runtime wiring — read aggregator)."""

from __future__ import annotations

import pytest
from databricks_deep_research.skills import Skill, SkillMeta, SkillStoreError

from deep_research.services.composite_skill_store import CompositeSkillStore


def _skill(name: str, desc: str, body: str = "# body") -> Skill:
    return Skill(name=name, description=desc, body=body)


class _FakeStore:
    def __init__(self, skills: list[Skill]) -> None:
        self._skills = {s.name: s for s in skills}

    async def list_skills(self) -> list[SkillMeta]:
        return [s.meta for s in self._skills.values()]

    async def get_skill(self, name: str) -> Skill | None:
        return self._skills.get(name)

    async def put_skill(self, skill: Skill, *, scan: object) -> None:  # noqa: ARG002
        raise SkillStoreError("read-only")


class _BrokenStore:
    """Raises on every read — exercises the fail-soft path."""

    async def list_skills(self) -> list[SkillMeta]:
        raise RuntimeError("backend down")

    async def get_skill(self, name: str) -> Skill | None:  # noqa: ARG002
        raise RuntimeError("backend down")

    async def put_skill(self, skill: Skill, *, scan: object) -> None:  # noqa: ARG002
        raise SkillStoreError("read-only")


@pytest.mark.asyncio
async def test_precedence_earlier_store_wins() -> None:
    high = _FakeStore([_skill("dup", "from high", body="HIGH")])
    low = _FakeStore([_skill("dup", "from low", body="LOW"), _skill("only_low", "x")])
    composite = CompositeSkillStore([high, low])

    # get_skill resolves from the highest-precedence store that has the name.
    got = await composite.get_skill("dup")
    assert got is not None
    assert got.description == "from high"
    assert got.body == "HIGH"

    # A name only in the lower store still resolves.
    only = await composite.get_skill("only_low")
    assert only is not None and only.description == "x"

    # Unknown name => None.
    assert await composite.get_skill("missing") is None


@pytest.mark.asyncio
async def test_list_merges_and_dedups_by_name() -> None:
    high = _FakeStore([_skill("dup", "from high")])
    low = _FakeStore([_skill("dup", "from low"), _skill("only_low", "x")])
    composite = CompositeSkillStore([high, low])

    metas = await composite.list_skills()
    names = sorted(m.name for m in metas)
    assert names == ["dup", "only_low"]  # dup listed once
    dup = next(m for m in metas if m.name == "dup")
    assert dup.description == "from high"  # precedence preserved in listing


@pytest.mark.asyncio
async def test_failsoft_skips_broken_backend() -> None:
    composite = CompositeSkillStore([_BrokenStore(), _FakeStore([_skill("ok", "fine")])])
    # Broken backend is skipped; the healthy one still resolves.
    metas = await composite.list_skills()
    assert [m.name for m in metas] == ["ok"]
    got = await composite.get_skill("ok")
    assert got is not None and got.description == "fine"


@pytest.mark.asyncio
async def test_put_skill_is_read_only() -> None:
    composite = CompositeSkillStore([_FakeStore([])])
    with pytest.raises(SkillStoreError, match="read-only"):
        await composite.put_skill(_skill("x", "y"), scan=object())
