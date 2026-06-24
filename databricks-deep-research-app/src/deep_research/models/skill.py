"""SQLAlchemy ORM models for governed skills (Feature 2.2).

A *skill* is governed Markdown (frontmatter + body + optional scripts) carrying
reusable methodology. The ``skills`` table is the queryable store backing the
framework's ``SkillStore`` protocol: cheap metadata listing (``name`` +
``description``) with body fetched on demand.

``SkillRevision`` is an append-only audit log (sibling of ``agent_revisions``)
recording every successful upsert together with the security-scan verdict — the
governance trail shared with the revision/audit story (5.6).
"""

import uuid
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from deep_research.db.base import Base, BaseModel


class Skill(BaseModel):
    """A governed skill row (migration 030).

    ``name`` is unique (skills are addressed by name). ``body`` holds the full
    Markdown methodology; ``scripts`` is a JSON map of name->code executed in the
    compute scratchpad (never injected into LLM context). ``security_verdict``
    records the scan outcome that permitted persistence.
    """

    __tablename__ = "skills"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    description: Mapped[str] = mapped_column(String(1024), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    scripts: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    author: Mapped[str | None] = mapped_column(String(255), nullable=True)
    security_verdict: Mapped[str | None] = mapped_column(String(50), nullable=True)
    is_seed: Mapped[int] = mapped_column(
        # stored as a small int (1 = bundled seed, 0 = authored) for parity
        # with the schema's other boolean-as-int columns
        Integer,
        nullable=False,
        default=0,
    )
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]

    __table_args__ = (Index("idx_skills_name", "name"),)


class SkillRevision(Base):
    """Append-only snapshot of a skill after each successful upsert.

    Written best-effort after each create/update so the audit trail captures the
    body + scan verdict at the time of the write. Never blocks the primary
    upsert on a revision-write failure.
    """

    __tablename__ = "skill_revisions"

    rev_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    skill_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("skills.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    description: Mapped[str] = mapped_column(String(1024), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    scripts: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict
    )
    security_verdict: Mapped[str | None] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )
    created_by: Mapped[str] = mapped_column(String(255), nullable=False)

    __table_args__ = (
        Index("idx_skill_revisions_skill_created", "skill_id", "created_at"),
    )
