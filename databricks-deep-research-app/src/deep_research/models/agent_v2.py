"""AgentV2 SQLAlchemy ORM model for Agent Designer V1.

Mirrors the agents_v2 table created by migration 023 (greenfield, parallel
to legacy custom_agents). The full agent definition (workflow AST) is stored
as a JSONB blob; etag enables optimistic locking.

AgentRevision tracks every successful create/update as an immutable snapshot
for audit-grade revision history (migration 025).
"""
import uuid
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from deep_research.db.base import Base, BaseModel
from deep_research.models.visibility import AgentVisibility


class AgentV2(BaseModel):
    """Custom research agent (V1 schema).

    The `definition` column holds a validated `WorkflowDefinition` AST as JSON.
    The `etag` column is sha1(sorted_json(definition) + updated_at_iso) for
    optimistic-locking PATCH semantics.
    """

    __tablename__ = "agents_v2"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    owner_id: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    avatar_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    visibility: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=AgentVisibility.PRIVATE.value,
    )
    definition: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    etag: Mapped[str] = mapped_column(String(40), nullable=False)
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]

    __table_args__ = (
        CheckConstraint(
            "visibility IN ('private', 'workspace', 'system')",
            name="agents_v2_visibility_check",
        ),
        Index("idx_agents_v2_owner_visibility", "owner_id", "visibility"),
    )

    @property
    def is_system(self) -> bool:
        return self.visibility == AgentVisibility.SYSTEM.value

    @property
    def is_workspace_visible(self) -> bool:
        return self.visibility == AgentVisibility.WORKSPACE.value


class CustomToolDef(BaseModel):
    """User-authored tool definition (migration 026).

    The ``factory_ref`` column references a key in BUILTIN_FACTORIES — a static
    allow-list that is validated at the API layer.  The factory is NEVER
    resolved via importlib.import_module or any other dynamic import.
    """

    __tablename__ = "custom_tool_defs"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    owner_id: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    kind: Mapped[int] = mapped_column(Integer, nullable=False, default=15)  # 15 = custom
    config_schema: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    factory_ref: Mapped[str] = mapped_column(String(255), nullable=False)
    etag: Mapped[str] = mapped_column(String(40), nullable=False)
    visibility: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="private",
    )
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]

    __table_args__ = (
        Index("idx_custom_tool_defs_owner_visibility", "owner_id", "visibility"),
    )


class AgentRevision(Base):
    """Immutable snapshot of an AgentV2 definition after each create/update.

    Written best-effort after every successful CRUD mutation so that the
    primary operation is never blocked by a revision-write failure.
    """

    __tablename__ = "agent_revisions"

    rev_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    agent_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("agents_v2.id", ondelete="CASCADE"),
        nullable=False,
    )
    etag: Mapped[str] = mapped_column(String(40), nullable=False)
    definition: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )
    created_by: Mapped[str] = mapped_column(String(255), nullable=False)
