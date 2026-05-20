"""Add runtime columns to ``agent_deployments`` for the async DeploymentJobRunner.

W12 (Phase 3 of the deployment-pipeline fix plan) replaces the previous
inline-synchronous ``POST /api/v1/deployments`` flow with a real async job
model: writes a PENDING row, spawns a background task, transitions the row
through DEPLOYING -> ACTIVE | FAILED while the worker emits heartbeats. A
janitor task sweeps zombie rows (heartbeat past timeout) and marks them
failed + dispatches translator cleanup so external resources are not
silently leaked.

Columns added (all nullable / default-false so existing rows are
compatible without backfill):

- ``worker_id`` — opaque identifier of the asyncio task / process owning
  this in-flight deployment. Cleared on transition to a terminal status.
- ``last_heartbeat`` — TIMESTAMPTZ of the most recent heartbeat write.
  Used by the janitor to detect zombies.
- ``heartbeat_timeout_at`` — pre-computed (last_heartbeat + interval)
  so the janitor can range-scan with a single index lookup. Cheaper than
  a function-based scan.
- ``cancel_requested`` — set by DELETE on a PENDING/DEPLOYING row; the
  worker polls this each heartbeat and aborts at the next safe point.

Partial index on (status, heartbeat_timeout_at) restricted to in-flight
statuses keeps the janitor's sweep query cheap as the deployments table
grows.

Revision ID: 028_deployment_runtime_columns
Revises: 027_create_agent_deployments
Create Date: 2026-05-11
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "028_deployment_runtime_columns"
down_revision: str | None = "027_create_agent_deployments"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_JANITOR_INDEX = "ix_agent_deployments_zombie_sweep"


def upgrade() -> None:
    op.add_column(
        "agent_deployments",
        sa.Column("worker_id", sa.String(64), nullable=True),
    )
    op.add_column(
        "agent_deployments",
        sa.Column(
            "last_heartbeat",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )
    op.add_column(
        "agent_deployments",
        sa.Column(
            "heartbeat_timeout_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )
    op.add_column(
        "agent_deployments",
        sa.Column(
            "cancel_requested",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    # Partial index: janitor only sweeps in-flight rows. Most rows in a
    # production deployment will be terminal at any time, so a partial
    # index keeps the scan cost bounded.
    op.create_index(
        _JANITOR_INDEX,
        "agent_deployments",
        ["status", "heartbeat_timeout_at"],
        postgresql_where=sa.text("status IN ('pending', 'deploying')"),
    )


def downgrade() -> None:
    op.drop_index(_JANITOR_INDEX, table_name="agent_deployments")
    op.drop_column("agent_deployments", "cancel_requested")
    op.drop_column("agent_deployments", "heartbeat_timeout_at")
    op.drop_column("agent_deployments", "last_heartbeat")
    op.drop_column("agent_deployments", "worker_id")
