"""Tests for WorkspaceFsSkillStore (A1 — per-user workspace-FS skills)."""

from __future__ import annotations

import io

import pytest
from databricks_deep_research.skills import Skill, SkillStoreError

from deep_research.services.workspace_fs_skill_store import (
    WorkspaceFsSkillStore,
    default_skill_roots,
)

USER = "me@example.com"
SKILLS_ROOT, ASSISTANT_ROOT = default_skill_roots(USER)


def _md(name: str, desc: str) -> str:
    return f"---\nname: {name}\ndescription: {desc}\n---\n# {name}\nmethodology body\n"


class _ObjType:
    def __init__(self, value: str) -> None:
        self.value = value


class _Obj:
    def __init__(self, path: str, *, is_dir: bool = False) -> None:
        self.path = path
        self.object_type = _ObjType("DIRECTORY" if is_dir else "FILE")


class _FakeWorkspace:
    def __init__(self, tree: dict[str, list[_Obj]], files: dict[str, str]) -> None:
        self._tree = tree
        self._files = files

    def list(self, path: str) -> list[_Obj]:
        if path not in self._tree:
            raise FileNotFoundError(path)  # missing folder => fail-soft
        return list(self._tree[path])

    def download(self, path: str) -> io.BytesIO:
        return io.BytesIO(self._files[path].encode("utf-8"))


class _FakeWS:
    def __init__(self, workspace: _FakeWorkspace) -> None:
        self.workspace = workspace


class _Verdict:
    def __init__(self, safe: bool) -> None:
        self.safe = safe
        self.reason = "" if safe else "blocked"


class _Scanner:
    def __init__(self, safe: bool) -> None:
        self._safe = safe
        self.calls = 0

    async def scan(self, skill: Skill) -> _Verdict:  # noqa: ARG002
        self.calls += 1
        return _Verdict(self._safe)


def _ws_with_skills() -> _FakeWS:
    flat = f"{SKILLS_ROOT}/flat.md"
    nested_dir = f"{SKILLS_ROOT}/nested"
    nested_md = f"{nested_dir}/SKILL.md"
    tree = {
        SKILLS_ROOT: [_Obj(flat), _Obj(nested_dir, is_dir=True)],
        nested_dir: [_Obj(nested_md)],
        # ASSISTANT_ROOT intentionally absent => list() raises => fail-soft.
    }
    files = {
        flat: _md("flat_skill", "flat one"),
        nested_md: _md("nested_skill", "nested one"),
    }
    return _FakeWS(_FakeWorkspace(tree, files))


@pytest.mark.asyncio
async def test_lists_flat_and_nested_and_skips_missing_root() -> None:
    store = WorkspaceFsSkillStore(_ws_with_skills(), user_name=USER)
    metas = await store.list_skills()
    names = sorted(m.name for m in metas)
    assert names == ["flat_skill", "nested_skill"]  # both layouts; missing root skipped


@pytest.mark.asyncio
async def test_get_skill_returns_body() -> None:
    store = WorkspaceFsSkillStore(_ws_with_skills(), user_name=USER)
    skill = await store.get_skill("nested_skill")
    assert skill is not None
    assert skill.description == "nested one"
    assert "methodology body" in skill.body


@pytest.mark.asyncio
async def test_unsafe_scan_blocks_body_failclosed() -> None:
    scanner = _Scanner(safe=False)
    store = WorkspaceFsSkillStore(_ws_with_skills(), user_name=USER, scanner=scanner)
    # Listing is unscanned (metadata only)...
    assert len(await store.list_skills()) == 2
    # ...but fetching a body runs the scanner and blocks on unsafe.
    assert await store.get_skill("flat_skill") is None
    assert scanner.calls == 1


@pytest.mark.asyncio
async def test_safe_scan_allows_and_caches_verdict() -> None:
    scanner = _Scanner(safe=True)
    store = WorkspaceFsSkillStore(_ws_with_skills(), user_name=USER, scanner=scanner)
    assert await store.get_skill("flat_skill") is not None
    assert await store.get_skill("flat_skill") is not None
    assert scanner.calls == 1  # verdict cached by content hash


@pytest.mark.asyncio
async def test_put_skill_read_only() -> None:
    store = WorkspaceFsSkillStore(_ws_with_skills(), user_name=USER)
    with pytest.raises(SkillStoreError, match="read-only"):
        await store.put_skill(
            Skill(name="x", description="y", body="z"), scan=_Scanner(True)
        )
