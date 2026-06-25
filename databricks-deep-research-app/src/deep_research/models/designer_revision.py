"""Append-only audit log for Designer-authored changes (spec §5.6 / §4.4).

A :class:`DesignerRevision` is ONE immutable row per Designer-authored change to
a prompt / skill / agent definition. It records the ``prev`` and ``new``
snapshots, the fail-closed security-scan verdict that GATED the persist, and
who/when. Rows are NEVER mutated in place — the table is the governance trail
behind the Designer's self-evolution sub-patterns.

This sits alongside the existing ``agent_revisions`` (migration 025) and
``skill_revisions`` (migration 030) tables, but is intentionally subject-
agnostic: a Designer-authored change may target a prompt on a node, a whole
agent definition, or a skill body, so the subject is identified by a
``(subject_type, subject_ref)`` pair rather than a typed foreign key.
"""

import uuid
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import DateTime, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from deep_research.db.base import Base


class DesignerRevision(Base):
    """Append-only snapshot of one Designer-authored change (migration 031).

    Written only after the fail-closed security scan returns an explicit SAFE
    verdict for the authored content. ``security_verdict`` records the scan
    rationale (or the literal verdict token) that permitted the persist.

    Append-only: there is no update path. Each authored change is a new row.
    """

    __tablename__ = "designer_revisions"

    rev_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    # Subject identity — agnostic to the kind of thing being evolved.
    subject_type: Mapped[str] = mapped_column(String(32), nullable=False)
    subject_ref: Mapped[str] = mapped_column(String(255), nullable=False)
    # prev/new snapshots of the authored content (JSON; ``prev`` is None on the
    # first authored change for a subject).
    prev_snapshot: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )
    new_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    # The fail-closed scan verdict that gated this persist. Reason text from the
    # scanner, or the literal SAFE token — never an UNSAFE row (those are denied
    # before any write).
    security_verdict: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    created_by: Mapped[str] = mapped_column(String(255), nullable=False)

    __table_args__ = (
        Index(
            "idx_designer_revisions_subject_created",
            "subject_type",
            "subject_ref",
            "created_at",
        ),
    )
