"""Add workflow-validation persistence + content-addressed verdict cache.

Revision ID: 034_add_workflow_validation
Revises: 033_create_user_skill_folders
Create Date: 2026-06-27

Unifies the Designer validation gate (see
``.omc/plans/unify-designer-validation-gate.md``):

- ``agents_v2`` gains the latest validation verdict + its content hash so the
  authoritative state lives in the DB (never in spoofable AST metadata).
- ``agent_revisions`` gets a per-snapshot ``validation`` blob for audit — each
  revision retains the verdict it had when saved.
- ``workflow_validation_cache`` is the reusable, content-addressed cache keyed by
  ``(validator_version, intent_hash, semantic_hash)`` so an unchanged workflow
  (already validated during the build loop) reuses its verdict at save with ZERO
  LLM calls. Bumping ``VALIDATOR_VERSION`` transparently invalidates stale rows.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "034_add_workflow_validation"
down_revision: str | None = "033_create_user_skill_folders"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add validation columns + create the validation cache table."""
    op.add_column(
        "agents_v2",
        sa.Column(
            "last_validation",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )
    op.add_column(
        "agents_v2",
        sa.Column("last_validation_verdict", sa.String(20), nullable=True),
    )
    op.add_column(
        "agents_v2",
        sa.Column("last_validation_hash", sa.String(40), nullable=True),
    )
    op.add_column(
        "agent_revisions",
        sa.Column(
            "validation",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )
    op.create_table(
        "workflow_validation_cache",
        sa.Column("validator_version", sa.String(40), nullable=False),
        sa.Column("intent_hash", sa.String(40), nullable=False),
        sa.Column("semantic_hash", sa.String(40), nullable=False),
        sa.Column(
            "result",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint(
            "validator_version",
            "intent_hash",
            "semantic_hash",
            name="pk_workflow_validation_cache",
        ),
    )


def downgrade() -> None:
    """Drop the cache table and the validation columns."""
    op.drop_table("workflow_validation_cache")
    op.drop_column("agent_revisions", "validation")
    op.drop_column("agents_v2", "last_validation_hash")
    op.drop_column("agents_v2", "last_validation_verdict")
    op.drop_column("agents_v2", "last_validation")
