"""Drop legacy custom_agents and agent_preset_steps tables.

Revision ID: 024_drop_custom_agents
Revises: 023_create_agents_v2
Create Date: 2026-04-27

These tables were created by migration 016_create_custom_agents (US6 / T076)
and superseded by agents_v2 (migration 023_create_agents_v2, Agent Designer V1).
All production data should be verified empty or migrated before applying this
migration — see scripts/preflight_v1_data_check.py for the pre-deploy data
check that gates this cutover.

Drop order: agent_preset_steps first (child via FK → custom_agents), then
custom_agents (parent).  Downgrade recreates both tables in reverse order so
rollback is clean.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "024_drop_custom_agents"
down_revision: str | None = "023_create_agents_v2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Drop agent_preset_steps (child) then custom_agents (parent).

    Idempotent: these tables may already have been removed out-of-band by
    scripts/cleanup_legacy_tables.sql (legacy public.* decommission), which keeps
    alembic_version stamped < 024, so this migration can be replayed against a DB
    where the tables are already gone. DROP TABLE ... CASCADE removes the
    dependent indexes/constraints, matching the cleanup script's IF EXISTS idiom.
    """
    # Child first (FK -> custom_agents), then parent. CASCADE drops the
    # dependent indexes, so explicit DROP INDEX statements are unnecessary.
    op.execute("DROP TABLE IF EXISTS agent_preset_steps CASCADE")
    op.execute("DROP TABLE IF EXISTS custom_agents CASCADE")


def downgrade() -> None:
    """Recreate custom_agents and agent_preset_steps (rollback path)."""
    # Recreate custom_agents table
    op.create_table(
        "custom_agents",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("owner_id", sa.String(255), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("avatar_url", sa.String(500), nullable=True),
        sa.Column(
            "system_prompt_template_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("prompt_templates.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "synthesis_template_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("prompt_templates.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "source_scope",
            sa.String(50),
            nullable=False,
            server_default="all",
        ),
        sa.Column(
            "enabled_sources",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "disabled_sources",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="[]",
        ),
        sa.Column(
            "use_planner",
            sa.Boolean(),
            nullable=False,
            server_default="true",
        ),
        sa.Column(
            "default_depth",
            sa.String(20),
            nullable=False,
            server_default="medium",
        ),
        sa.Column(
            "default_mode",
            sa.String(20),
            nullable=False,
            server_default="planner",
        ),
        sa.Column(
            "enable_clarification",
            sa.Boolean(),
            nullable=False,
            server_default="true",
        ),
        sa.Column(
            "output_format",
            sa.String(20),
            nullable=False,
            server_default="markdown",
        ),
        sa.Column(
            "output_schema",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "visibility",
            sa.String(20),
            nullable=False,
            server_default="private",
        ),
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
        sa.PrimaryKeyConstraint("id", name="pk_custom_agents"),
    )

    # Recreate custom_agents indexes
    op.create_index(
        "idx_custom_agents_owner",
        "custom_agents",
        ["owner_id"],
    )
    op.create_index(
        "idx_custom_agents_visibility",
        "custom_agents",
        ["visibility"],
    )
    op.create_index(
        "idx_custom_agents_owner_visibility",
        "custom_agents",
        ["owner_id", "visibility"],
    )
    op.create_index(
        "uq_custom_agents_owner_name",
        "custom_agents",
        ["owner_id", "name"],
        unique=True,
    )

    # Recreate agent_preset_steps table
    op.create_table(
        "agent_preset_steps",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "agent_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("custom_agents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("order", sa.Integer(), nullable=False, server_default="1"),
        sa.Column(
            "is_required",
            sa.Boolean(),
            nullable=False,
            server_default="true",
        ),
        sa.Column(
            "source_hints",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("source_scope", sa.String(50), nullable=True),
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
        sa.PrimaryKeyConstraint("id", name="pk_agent_preset_steps"),
    )

    # Recreate agent_preset_steps indexes
    op.create_index(
        "idx_agent_preset_steps_agent",
        "agent_preset_steps",
        ["agent_id"],
    )
    op.create_index(
        "idx_agent_preset_steps_agent_order",
        "agent_preset_steps",
        ["agent_id", "order"],
    )
