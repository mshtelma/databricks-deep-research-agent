"""Tests for build_runtime_skill_store (A1 — runtime skill-store composition)."""

from __future__ import annotations

import io

import pytest

from deep_research.services.composite_skill_store import CompositeSkillStore
from deep_research.services.skill_runtime import build_runtime_skill_store
from deep_research.services.workspace_fs_skill_store import default_skill_roots

USER = "me@example.com"
SKILLS_ROOT, _ = default_skill_roots(USER)


def _md(name: str, desc: str) -> str:
    return f"---\nname: {name}\ndescription: {desc}\n---\n# {name}\nbody\n"


class _ObjType:
    def __init__(self, value: str) -> None:
        self.value = value


class _Obj:
    def __init__(self, path: str) -> None:
        self.path = path
        self.object_type = _ObjType("FILE")


class _Workspace:
    def __init__(self, files: dict[str, str]) -> None:
        self._files = files

    def list(self, path: str) -> list[_Obj]:
        if path != SKILLS_ROOT:
            raise FileNotFoundError(path)  # other roots absent => fail-soft
        return [_Obj(p) for p in self._files]

    def download(self, path: str) -> io.BytesIO:
        return io.BytesIO(self._files[path].encode("utf-8"))


class _Me:
    user_name = USER


class _CurrentUser:
    def me(self) -> _Me:
        return _Me()


class _WS:
    def __init__(self, files: dict[str, str]) -> None:
        self.workspace = _Workspace(files)
        self.current_user = _CurrentUser()


@pytest.mark.asyncio
async def test_composes_workspace_source_plus_seeds() -> None:
    # user_token=None => resolve_workspace_client returns the passed client as-is
    # (the OBO-vs-SP choice itself is framework-tested); here the fake client is
    # the resolved identity and the workspace source must be composed from it.
    ws = _WS({f"{SKILLS_ROOT}/s.md": _md("ws_skill", "from workspace")})
    store = build_runtime_skill_store(
        llm_client=None, workspace_client=ws, user_token=None
    )
    assert isinstance(store, CompositeSkillStore)
    names = {m.name for m in await store.list_skills()}
    # Workspace source is wired (the user's skill is visible) ...
    assert "ws_skill" in names
    # ... and fetching its body resolves through the composite.
    got = await store.get_skill("ws_skill")
    assert got is not None and got.description == "from workspace"


@pytest.mark.asyncio
async def test_no_identity_still_returns_seed_store() -> None:
    # No workspace client / name => workspace source omitted, seeds still present.
    store = build_runtime_skill_store(
        llm_client=None, workspace_client=None, user_token=None
    )
    assert isinstance(store, CompositeSkillStore)
    # list_skills must not raise even with only the bundled seed store.
    assert isinstance(await store.list_skills(), list)
