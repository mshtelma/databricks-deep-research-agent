"""Create agent_deployments table for deployment-feature Phase 1.

Tracks every deployment attempt (in_app, shell_app, mlflow_agent, batch) of a
specific AgentRevision to a target. Rollback = create a NEW row pointing at a
previous revision_id; existing rows are never reused.

The FK on agent_id is **ON DELETE RESTRICT** (only this new FK; the existing
agent_revisions.agent_id FK from migration 025 stays CASCADE). This is the
deletion-guard mechanism specified in plan Section N.1.

Revision ID: 027_create_agent_deployments
Revises: 026_create_custom_tool_defs
Create Date: 2026-05-09
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "027_create_agent_deployments"
down_revision: str | None = "026_create_custom_tool_defs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_DEPLOYMENT_MODES = ("in_app", "shell_app", "mlflow_agent", "batch")
_DEPLOYMENT_STATUSES = (
    "pending",
    "deploying",
    "active",
    "failed",
    "deactivated",
    "cleanup_failed",
)


def upgrade() -> None:
    """Create agent_deployments table with FK RESTRICT on agent_id."""
    op.create_table(
        "agent_deployments",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("agent_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("revision_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("mode", sa.String(20), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column(
            "config",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("endpoint_name", sa.String(255), nullable=True),
        sa.Column("model_name", sa.String(255), nullable=True),
        sa.Column(
            "external_resource_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "cleanup_attempts",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("deployed_by", sa.String(255), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("deactivated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_agent_deployments"),
        # Block parent agent deletion while any deployment row exists.
        # Plan Section N.2: terminal-status rows must be physically deleted
        # before the parent agent row can be deleted; PG RESTRICT checks
        # row existence, not status.
        sa.ForeignKeyConstraint(
            ["agent_id"],
            ["agents_v2.id"],
            ondelete="RESTRICT",
            name="fk_agent_deployments_agent_id_agents_v2",
        ),
        sa.ForeignKeyConstraint(
            ["revision_id"],
            ["agent_revisions.rev_id"],
            ondelete="RESTRICT",
            name="fk_agent_deployments_revision_id_agent_revisions",
        ),
        sa.CheckConstraint(
            f"mode IN ({', '.join(repr(m) for m in _DEPLOYMENT_MODES)})",
            name="agent_deployments_mode_check",
        ),
        sa.CheckConstraint(
            f"status IN ({', '.join(repr(s) for s in _DEPLOYMENT_STATUSES)})",
            name="agent_deployments_status_check",
        ),
    )
    op.create_index(
        "idx_agent_deployments_agent_status",
        "agent_deployments",
        ["agent_id", "status"],
    )
    op.create_index(
        "idx_agent_deployments_mode_status",
        "agent_deployments",
        ["mode", "status"],
    )
    op.create_index(
        "idx_agent_deployments_deployed_by",
        "agent_deployments",
        ["deployed_by"],
    )


def downgrade() -> None:
    """Drop agent_deployments table and its indexes."""
    op.drop_index(
        "idx_agent_deployments_deployed_by",
        table_name="agent_deployments",
    )
    op.drop_index(
        "idx_agent_deployments_mode_status",
        table_name="agent_deployments",
    )
    op.drop_index(
        "idx_agent_deployments_agent_status",
        table_name="agent_deployments",
    )
    op.drop_table("agent_deployments")
