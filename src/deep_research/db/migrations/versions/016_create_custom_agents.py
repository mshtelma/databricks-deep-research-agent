"""Add custom_agents and agent_preset_steps tables for custom agent configuration.

Revision ID: 016_create_custom_agents
Revises: 015_create_prompt_templates
Create Date: 2026-02-04

This migration adds tables for custom agent configuration:
- custom_agents: User-defined research agents with custom settings
- agent_preset_steps: Preset research steps for manual/hybrid workflows

Part of US6 - Custom Agent Configurations (T076).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "016_create_custom_agents"
down_revision: str | None = "015_create_prompt_templates"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create custom_agents and agent_preset_steps tables."""
    # Create custom_agents table
    op.create_table(
        "custom_agents",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        # Owner identification
        sa.Column("owner_id", sa.String(255), nullable=False),
        # Agent name
        sa.Column("name", sa.String(255), nullable=False),
        # Description
        sa.Column("description", sa.Text(), nullable=True),
        # Avatar URL
        sa.Column("avatar_url", sa.String(500), nullable=True),
        # Template references (nullable FKs)
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
        # Source scope configuration
        sa.Column(
            "source_scope",
            sa.String(50),
            nullable=False,
            server_default="all",
        ),
        # Explicit source lists (JSONB arrays)
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
        # Workflow configuration
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
        # Output configuration
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
        # Visibility
        sa.Column(
            "visibility",
            sa.String(20),
            nullable=False,
            server_default="private",
        ),
        # Timestamps
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

    # Create indexes for custom_agents
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
    # Unique constraint: name must be unique per owner
    op.create_index(
        "uq_custom_agents_owner_name",
        "custom_agents",
        ["owner_id", "name"],
        unique=True,
    )

    # Create agent_preset_steps table
    op.create_table(
        "agent_preset_steps",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        # Parent agent reference (cascade delete)
        sa.Column(
            "agent_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("custom_agents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        # Step title
        sa.Column("title", sa.String(255), nullable=False),
        # Step description
        sa.Column("description", sa.Text(), nullable=True),
        # Execution order (1-based)
        sa.Column("order", sa.Integer(), nullable=False, server_default="1"),
        # Required flag
        sa.Column(
            "is_required",
            sa.Boolean(),
            nullable=False,
            server_default="true",
        ),
        # Source hints (JSONB)
        sa.Column(
            "source_hints",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        # Optional source scope override
        sa.Column("source_scope", sa.String(50), nullable=True),
        # Timestamps
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

    # Create indexes for agent_preset_steps
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


def downgrade() -> None:
    """Drop agent_preset_steps and custom_agents tables."""
    # Drop agent_preset_steps indexes
    op.drop_index("idx_agent_preset_steps_agent_order", table_name="agent_preset_steps")
    op.drop_index("idx_agent_preset_steps_agent", table_name="agent_preset_steps")

    # Drop agent_preset_steps table
    op.drop_table("agent_preset_steps")

    # Drop custom_agents indexes
    op.drop_index("uq_custom_agents_owner_name", table_name="custom_agents")
    op.drop_index("idx_custom_agents_owner_visibility", table_name="custom_agents")
    op.drop_index("idx_custom_agents_visibility", table_name="custom_agents")
    op.drop_index("idx_custom_agents_owner", table_name="custom_agents")

    # Drop custom_agents table
    op.drop_table("custom_agents")
