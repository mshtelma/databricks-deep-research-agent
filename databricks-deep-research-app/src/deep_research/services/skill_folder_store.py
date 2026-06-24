"""CRUD store for user-registered skill folders (Feature 2.2 / A3).

The store persists per-user skill-folder roots and exposes the read helper
:func:`load_user_skill_roots`, which the runtime uses to extend the
``WorkspaceFsSkillStore`` ``extra_roots``. All access is scoped by ``user_id`` so
a user only ever sees / mutates their own folders.
"""

from __future__ import annotations

import logging
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from deep_research.models.user_skill_folder import UserSkillFolder
from deep_research.schemas.skill_folder import (
    SkillFolderKind,
    normalize_skill_folder_path,
)

logger = logging.getLogger(__name__)

__all__ = ["SkillFolderStore", "load_user_skill_roots"]


class SkillFolderStore:
    """Per-user CRUD over the ``user_skill_folders`` table."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_for_user(self, user_id: str) -> list[UserSkillFolder]:
        """Return the user's registered folders, newest first."""
        result = await self._session.execute(
            select(UserSkillFolder)
            .where(UserSkillFolder.user_id == user_id)
            .order_by(UserSkillFolder.created_at.desc())
        )
        return list(result.scalars().all())

    async def add(
        self, user_id: str, path: str, kind: SkillFolderKind = "workspace"
    ) -> UserSkillFolder:
        """Register a folder for *user_id* (idempotent on (user_id, path)).

        Raises :class:`ValueError` for an invalid path/kind.
        """
        normalized = normalize_skill_folder_path(path, kind)
        existing = await self._session.execute(
            select(UserSkillFolder).where(
                UserSkillFolder.user_id == user_id,
                UserSkillFolder.path == normalized,
            )
        )
        row = existing.scalar_one_or_none()
        if row is not None:
            return row

        row = UserSkillFolder(user_id=user_id, path=normalized, kind=kind)
        self._session.add(row)
        try:
            await self._session.flush()
        except IntegrityError:
            # Concurrent insert of the same (user_id, path) — fetch the winner.
            await self._session.rollback()
            existing = await self._session.execute(
                select(UserSkillFolder).where(
                    UserSkillFolder.user_id == user_id,
                    UserSkillFolder.path == normalized,
                )
            )
            won = existing.scalar_one_or_none()
            if won is None:  # pragma: no cover - defensive
                raise
            return won
        return row

    async def remove(self, user_id: str, folder_id: UUID) -> bool:
        """Delete one of the user's folders. Returns True if a row was removed."""
        result = await self._session.execute(
            delete(UserSkillFolder).where(
                UserSkillFolder.user_id == user_id,
                UserSkillFolder.id == folder_id,
            )
        )
        await self._session.flush()
        # DELETE yields a CursorResult; ``rowcount`` isn't on the base Result type.
        return bool(getattr(result, "rowcount", 0))


async def load_user_skill_roots(session: AsyncSession, user_id: str) -> list[str]:
    """Return the user's registered skill-folder paths (fail-soft, empty on error).

    Used by the runtime to extend the workspace-FS skill store roots. Never
    raises — a folder-store failure must not break a research run that merely
    declared skills.
    """
    if not user_id:
        return []
    try:
        store = SkillFolderStore(session)
        folders = await store.list_for_user(user_id)
        return [f.path for f in folders]
    except Exception:  # noqa: BLE001 — fail-soft; skills still work without extras
        logger.warning("SKILL_FOLDER_ROOTS_LOAD_FAILED user=%s", user_id, exc_info=True)
        return []
