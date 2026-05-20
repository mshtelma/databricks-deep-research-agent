"""Create agent_revisions table for V1.5 audit-grade revision history.

Revision ID: 025_create_agent_revisions
Revises: 024_drop_custom_agents
Create Date: 2026-04-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "025_create_agent_revisions"
down_revision: str | None = "024_drop_custom_agents"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create agent_revisions table with B-tree index for perf at 1k+ revisions."""
    op.create_table(
        "agent_revisions",
        sa.Column(
            "rev_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("agent_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("etag", sa.String(40), nullable=False),
        sa.Column(
            "definition",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("created_by", sa.String(255), nullable=False),
        sa.PrimaryKeyConstraint("rev_id", name="pk_agent_revisions"),
        sa.ForeignKeyConstraint(
            ["agent_id"],
            ["agents_v2.id"],
            ondelete="CASCADE",
            name="fk_agent_revisions_agent_id_agents_v2",
        ),
    )
    # B-tree index: fast per-agent listing ordered by created_at DESC
    op.create_index(
        "idx_agent_revisions_agent_created",
        "agent_revisions",
        ["agent_id", sa.text("created_at DESC")],
    )


def downgrade() -> None:
    """Drop agent_revisions table and its indexes."""
    op.drop_index("idx_agent_revisions_agent_created", table_name="agent_revisions")
    op.drop_table("agent_revisions")
