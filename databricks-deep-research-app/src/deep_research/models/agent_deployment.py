"""AgentDeployment SQLAlchemy ORM model for deployment lifecycle tracking.

Each row represents one deployment attempt of an AgentRevision to a target
mode. Rows are NEVER reused for rollback -- rollback creates a NEW row
pointing at a previous revision_id.

Migration 027 created the underlying ``agent_deployments`` table with
``ON DELETE RESTRICT`` on ``agent_id`` -- the parent agent cannot be deleted
while any deployment row exists. The cleanup lifecycle (plan Section N.2)
physically deletes terminal-status rows before the agent row is deleted.
"""
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from deep_research.db.base import BaseModel


class DeploymentMode(StrEnum):
    """The four deployment targets supported by the Agent Designer."""

    IN_APP = "in_app"
    SHELL_APP = "shell_app"
    MLFLOW_AGENT = "mlflow_agent"
    BATCH = "batch"


class DeploymentStatus(StrEnum):
    """Lifecycle states for an AgentDeployment row.

    Terminal statuses (DEACTIVATED, CLEANUP_FAILED) allow the parent agent
    to be deleted only after the deployment row itself is physically removed
    -- see plan Section N.2.
    """

    PENDING = "pending"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    FAILED = "failed"
    DEACTIVATED = "deactivated"
    CLEANUP_FAILED = "cleanup_failed"


# Statuses that count as "active" for the deletion guard. PENDING and DEPLOYING
# are mid-lifecycle; ACTIVE means the external resource is live.
# Public so service-layer SQL ``IN()`` clauses share the same source of truth
# as the ``is_active`` / ``is_terminal`` model properties.
ACTIVE_STATUSES: frozenset[str] = frozenset(
    {
        DeploymentStatus.PENDING.value,
        DeploymentStatus.DEPLOYING.value,
        DeploymentStatus.ACTIVE.value,
    }
)
# Statuses that stop UI polling. FAILED is included so the frontend status-poll
# hook does not keep refetching a row that will never change. Forensic value
# is preserved by DELETABLE_STATUSES (below): failed rows are not in the
# deletable set, so force-delete of the parent agent cannot silently erase
# them — the user must opt in via explicit deactivate.
TERMINAL_STATUSES: frozenset[str] = frozenset(
    {
        DeploymentStatus.DEACTIVATED.value,
        DeploymentStatus.CLEANUP_FAILED.value,
        DeploymentStatus.FAILED.value,
    }
)
# Statuses that may be physically deleted as part of force-deleting the
# parent agent. FAILED is intentionally excluded so the audit trail of why
# a deployment failed is preserved — the user must explicitly deactivate
# the failed row first (which sets DEACTIVATED, in this set). See W2 of the
# fix plan and the architect review for the rationale.
DELETABLE_STATUSES: frozenset[str] = frozenset(
    {
        DeploymentStatus.DEACTIVATED.value,
        DeploymentStatus.CLEANUP_FAILED.value,
    }
)

# Max attempts the cleanup orchestrator will try before transitioning the
# row to ``cleanup_failed``. Lives next to the status sets so the contract
# is co-located with the state model — tests import this constant rather
# than hardcoding "3". Each user-initiated DELETE counts as one attempt;
# there is no retry inside a single request (W4 of the fix plan).
MAX_CLEANUP_ATTEMPTS: int = 3


class AgentDeployment(BaseModel):
    """One deployment attempt of an AgentRevision to a deployment target.

    The ``config`` JSONB column holds a mode-specific Pydantic model (validated
    at the API layer via the discriminated-union ``DeploymentConfig``). The
    ``external_resource_ids`` JSONB column is the **authoritative** record of
    Databricks resources created by this deployment -- the orphan-detection
    cron uses it (not naming conventions) to identify leaked resources.
    """

    __tablename__ = "agent_deployments"

    agent_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        # Block parent delete while any deployment row exists.
        # Plan Section N.2: terminal rows must be physically deleted first.
        ForeignKey("agents_v2.id", ondelete="RESTRICT"),
        nullable=False,
    )
    revision_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("agent_revisions.rev_id", ondelete="RESTRICT"),
        nullable=False,
    )
    mode: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=DeploymentStatus.PENDING.value,
    )
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    endpoint_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    external_resource_ids: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    cleanup_attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    deployed_by: Mapped[str] = mapped_column(String(255), nullable=False)
    deactivated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # W12 — async DeploymentJobRunner state (migration 028).
    # Owned by an in-flight worker; null when the row is terminal or has
    # never been claimed. ``last_heartbeat`` / ``heartbeat_timeout_at``
    # are written by the runner's heartbeat loop. The janitor scans for
    # rows where heartbeat_timeout_at < now() AND status in active set
    # to detect zombies. ``cancel_requested`` is set by DELETE on a
    # pending/deploying row; the worker polls it and aborts gracefully.
    worker_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    last_heartbeat: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    heartbeat_timeout_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    cancel_requested: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
    )

    __table_args__ = (
        CheckConstraint(
            "mode IN ('in_app', 'shell_app', 'mlflow_agent', 'batch')",
            name="agent_deployments_mode_check",
        ),
        CheckConstraint(
            "status IN ('pending', 'deploying', 'active', 'failed', "
            "'deactivated', 'cleanup_failed')",
            name="agent_deployments_status_check",
        ),
        Index("idx_agent_deployments_agent_status", "agent_id", "status"),
        Index("idx_agent_deployments_mode_status", "mode", "status"),
        Index("idx_agent_deployments_deployed_by", "deployed_by"),
        # W12: partial index used by DeploymentJobRunner._sweep_zombies.
        # Mirrors migration 028 — declared here so Alembic autogenerate
        # does not propose a redundant drop on the next migration cycle.
        Index(
            "ix_agent_deployments_zombie_sweep",
            "status",
            "heartbeat_timeout_at",
            postgresql_where=text("status IN ('pending', 'deploying')"),
        ),
    )

    @property
    def is_active(self) -> bool:
        """True when the deployment is mid-lifecycle or live.

        Used by the deletion guard (plan Section N.1) to count "blocking"
        deployments.
        """
        return self.status in ACTIVE_STATUSES

    @property
    def is_terminal(self) -> bool:
        """True when the deployment has reached an end state.

        Terminal rows must be physically deleted before the parent agent
        can be deleted (plan Section N.2 step 4).
        """
        return self.status in TERMINAL_STATUSES
