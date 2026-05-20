"""Create agents_v2 table for Agent Designer V1 (greenfield, parallel to legacy custom_agents).

Per agent-designer-final.md §3 Data Model. Non-destructive: legacy custom_agents
and agent_preset_steps tables remain untouched. Final rename of agents_v2 →
custom_agents is a Phase 5 cutover concern (production pre-deploy data check).

Revision ID: 023_create_agents_v2
Revises: 022_create_chat_memory_tables
Create Date: 2026-04-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "023_create_agents_v2"
down_revision: str | None = "022_create_chat_memory_tables"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create agents_v2 table with indexes."""
    op.create_table(
        "agents_v2",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("owner_id", sa.String(255), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("avatar_url", sa.String(512), nullable=True),
        sa.Column(
            "visibility",
            sa.String(20),
            nullable=False,
            server_default="private",
        ),
        sa.Column(
            "definition",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column(
            "schema_version",
            sa.Integer(),
            nullable=False,
            server_default="1",
        ),
        sa.Column("etag", sa.String(40), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_agents_v2"),
        sa.CheckConstraint(
            "visibility IN ('private', 'workspace', 'system')",
            name="agents_v2_visibility_check",
        ),
    )
    op.create_index(
        "idx_agents_v2_owner_visibility",
        "agents_v2",
        ["owner_id", "visibility"],
    )
    op.create_index(
        "idx_agents_v2_visibility",
        "agents_v2",
        ["visibility"],
        postgresql_where=sa.text("visibility != 'private'"),
    )


def downgrade() -> None:
    """Drop agents_v2 table and its indexes."""
    op.drop_index("idx_agents_v2_visibility", table_name="agents_v2")
    op.drop_index("idx_agents_v2_owner_visibility", table_name="agents_v2")
    op.drop_table("agents_v2")
