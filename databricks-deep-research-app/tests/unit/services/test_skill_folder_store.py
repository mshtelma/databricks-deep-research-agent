"""Tests for the user skill-folder store + path validation (A3)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from deep_research.schemas.skill_folder import (
    AddSkillFolderRequest,
    normalize_skill_folder_path,
)
from deep_research.services.skill_folder_store import (
    SkillFolderStore,
    load_user_skill_roots,
)

# ---------------------------------------------------------------------------
# Path validation (pure)
# ---------------------------------------------------------------------------


def test_normalize_accepts_workspace_and_volume() -> None:
    assert (
        normalize_skill_folder_path("/Workspace/Users/x/.skills/", "workspace")
        == "/Workspace/Users/x/.skills"
    )
    assert (
        normalize_skill_folder_path("/Volumes/cat/sch/vol/skills", "volume")
        == "/Volumes/cat/sch/vol/skills"
    )


@pytest.mark.parametrize(
    ("path", "kind"),
    [
        ("", "workspace"),
        ("   ", "workspace"),
        ("relative/path", "workspace"),
        ("/has/../traversal", "workspace"),
        ("/Workspace/not/a/volume", "volume"),  # volume must be /Volumes/
    ],
)
def test_normalize_rejects_invalid(path: str, kind: str) -> None:
    with pytest.raises(ValueError):
        normalize_skill_folder_path(path, kind)  # type: ignore[arg-type]


def test_request_validator_requires_absolute() -> None:
    with pytest.raises(ValueError):
        AddSkillFolderRequest(path="relative", kind="workspace")
    ok = AddSkillFolderRequest(path="/Workspace/x", kind="workspace")
    assert ok.path == "/Workspace/x"


# ---------------------------------------------------------------------------
# Store (mock session)
# ---------------------------------------------------------------------------


def _result_with(*, scalars: list | None = None, one=..., rowcount: int | None = None):
    result = MagicMock()
    if scalars is not None:
        result.scalars.return_value.all.return_value = scalars
    if one is not ...:
        result.scalar_one_or_none.return_value = one
    if rowcount is not None:
        result.rowcount = rowcount
    return result


async def test_list_for_user_returns_rows() -> None:
    rows = [MagicMock(path="/a"), MagicMock(path="/b")]
    session = MagicMock()
    session.execute = AsyncMock(return_value=_result_with(scalars=rows))
    store = SkillFolderStore(session)
    out = await store.list_for_user("u1")
    assert out == rows


async def test_add_is_idempotent_when_existing() -> None:
    existing = MagicMock(path="/Workspace/x")
    session = MagicMock()
    session.execute = AsyncMock(return_value=_result_with(one=existing))
    session.add = MagicMock()
    session.flush = AsyncMock()
    store = SkillFolderStore(session)
    out = await store.add("u1", "/Workspace/x", "workspace")
    assert out is existing
    session.add.assert_not_called()  # no insert when it already exists


async def test_add_inserts_when_absent() -> None:
    session = MagicMock()
    session.execute = AsyncMock(return_value=_result_with(one=None))
    session.add = MagicMock()
    session.flush = AsyncMock()
    store = SkillFolderStore(session)
    out = await store.add("u1", "/Workspace/new", "workspace")
    session.add.assert_called_once()
    session.flush.assert_awaited_once()
    assert out.path == "/Workspace/new"


async def test_add_rejects_invalid_path() -> None:
    session = MagicMock()
    store = SkillFolderStore(session)
    with pytest.raises(ValueError):
        await store.add("u1", "relative", "workspace")


async def test_remove_reports_rowcount() -> None:
    session = MagicMock()
    session.execute = AsyncMock(return_value=_result_with(rowcount=1))
    session.flush = AsyncMock()
    store = SkillFolderStore(session)
    assert await store.remove("u1", uuid4()) is True

    session.execute = AsyncMock(return_value=_result_with(rowcount=0))
    assert await store.remove("u1", uuid4()) is False


# ---------------------------------------------------------------------------
# load_user_skill_roots (fail-soft)
# ---------------------------------------------------------------------------


async def test_roots_empty_for_no_user() -> None:
    assert await load_user_skill_roots(MagicMock(), "") == []


async def test_roots_returns_paths() -> None:
    rows = [MagicMock(path="/Workspace/a"), MagicMock(path="/Volumes/b")]
    session = MagicMock()
    session.execute = AsyncMock(return_value=_result_with(scalars=rows))
    assert await load_user_skill_roots(session, "u1") == ["/Workspace/a", "/Volumes/b"]


async def test_roots_fail_soft_on_error() -> None:
    session = MagicMock()
    session.execute = AsyncMock(side_effect=RuntimeError("db down"))
    assert await load_user_skill_roots(session, "u1") == []
