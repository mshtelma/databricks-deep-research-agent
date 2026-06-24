"""SQLAlchemy ORM model for user-registered skill folders (Feature 2.2 / A3).

A user may register extra skill ROOTS — workspace-FS directories or UC Volume
paths — that the runtime ``WorkspaceFsSkillStore`` scans (under the user's OBO
identity) on top of the built-in roots. One row per (user, path); the path is
read with the user's own credentials, so registering a folder grants no access
the user does not already have.
"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import Index, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from deep_research.db.base import BaseModel


class UserSkillFolder(BaseModel):
    """A user-registered skill-folder root (migration 033).

    ``kind`` is ``"workspace"`` (a ``/Workspace`` path) or ``"volume"`` (a
    ``/Volumes`` UC path). ``(user_id, path)`` is unique so re-adding the same
    folder is idempotent.
    """

    __tablename__ = "user_skill_folders"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    path: Mapped[str] = mapped_column(String(1024), nullable=False)
    kind: Mapped[str] = mapped_column(String(20), nullable=False, default="workspace")
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]

    __table_args__ = (
        UniqueConstraint("user_id", "path", name="uq_user_skill_folders_user_path"),
        Index("idx_user_skill_folders_user", "user_id"),
    )
