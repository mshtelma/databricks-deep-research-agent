"""Pydantic schemas for user-registered skill folders (Feature 2.2 / A3)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

SkillFolderKind = Literal["workspace", "volume"]

_MAX_PATH_LEN = 1024


def normalize_skill_folder_path(path: str, kind: SkillFolderKind) -> str:
    """Validate and normalize a skill-folder path for *kind*.

    Raises :class:`ValueError` for an empty / non-absolute / traversal / wrong-
    prefix path. The path is read with the requesting user's OBO identity, so
    this is hygiene (not an authorization boundary): it rejects obvious mistakes
    and keeps stored paths well-formed.
    """
    cleaned = (path or "").strip().rstrip("/")
    if not cleaned:
        raise ValueError("skill folder path must not be empty")
    if len(cleaned) > _MAX_PATH_LEN:
        raise ValueError(f"skill folder path exceeds {_MAX_PATH_LEN} characters")
    if not cleaned.startswith("/"):
        raise ValueError("skill folder path must be absolute (start with '/')")
    if ".." in cleaned.split("/"):
        raise ValueError("skill folder path must not contain '..'")
    if kind == "volume" and not cleaned.startswith("/Volumes/"):
        raise ValueError("a volume skill folder path must start with '/Volumes/'")
    return cleaned


class AddSkillFolderRequest(BaseModel):
    """Request body to register a skill folder for the current user."""

    path: str = Field(..., min_length=1, max_length=_MAX_PATH_LEN)
    kind: SkillFolderKind = "workspace"

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str, info: object) -> str:  # noqa: ARG003
        # ``kind`` may not be populated yet during field validation, so the
        # kind-specific prefix check is re-applied in the store/endpoint via
        # ``normalize_skill_folder_path``; here we only enforce absoluteness.
        cleaned = value.strip().rstrip("/")
        if not cleaned.startswith("/"):
            raise ValueError("skill folder path must be absolute (start with '/')")
        return cleaned


class SkillFolderResponse(BaseModel):
    """A registered skill folder."""

    id: UUID
    path: str
    kind: SkillFolderKind
    created_at: datetime


class SkillFolderListResponse(BaseModel):
    """A user's registered skill folders."""

    folders: list[SkillFolderResponse] = Field(default_factory=list)
