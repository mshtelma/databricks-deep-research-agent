"""Add prompt_templates table for custom prompt template library.

Revision ID: 015_create_prompt_templates
Revises: 014_user_data_sources
Create Date: 2026-02-04

This migration adds the prompt_templates table for users to create and manage
custom prompt templates for various agent components (system prompts, step
prompts, synthesis prompts, queries).

Part of US5 - Custom Prompt Template Library (T066).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "015_create_prompt_templates"
down_revision: str | None = "014_user_data_sources"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create prompt_templates table and indexes."""
    # Create the table
    op.create_table(
        "prompt_templates",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        # Owner identification
        sa.Column("owner_id", sa.String(255), nullable=False),
        # Template name
        sa.Column("name", sa.String(255), nullable=False),
        # Template type (system, step, synthesis, query)
        sa.Column("type", sa.String(50), nullable=False),
        # Template content with {{variable}} placeholders
        sa.Column("content", sa.Text(), nullable=False),
        # Variable metadata (JSONB array)
        sa.Column(
            "variables",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="[]",
        ),
        # Tags for filtering (JSONB array)
        sa.Column(
            "tags",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="[]",
        ),
        # Visibility (private, workspace)
        sa.Column(
            "visibility",
            sa.String(20),
            nullable=False,
            server_default="private",
        ),
        # Default template flag
        sa.Column(
            "is_default",
            sa.Boolean(),
            nullable=False,
            server_default="false",
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
        sa.PrimaryKeyConstraint("id", name="pk_prompt_templates"),
    )

    # Create indexes for efficient queries
    op.create_index(
        "idx_prompt_templates_owner",
        "prompt_templates",
        ["owner_id"],
    )
    op.create_index(
        "idx_prompt_templates_type",
        "prompt_templates",
        ["type"],
    )
    op.create_index(
        "idx_prompt_templates_visibility",
        "prompt_templates",
        ["visibility"],
    )
    op.create_index(
        "idx_prompt_templates_owner_visibility",
        "prompt_templates",
        ["owner_id", "visibility"],
    )
    op.create_index(
        "idx_prompt_templates_type_default",
        "prompt_templates",
        ["type", "is_default"],
    )
    # Unique constraint: name must be unique per owner
    op.create_index(
        "uq_prompt_templates_owner_name",
        "prompt_templates",
        ["owner_id", "name"],
        unique=True,
    )


def downgrade() -> None:
    """Drop prompt_templates table and indexes."""
    # Drop indexes
    op.drop_index("uq_prompt_templates_owner_name", table_name="prompt_templates")
    op.drop_index("idx_prompt_templates_type_default", table_name="prompt_templates")
    op.drop_index("idx_prompt_templates_owner_visibility", table_name="prompt_templates")
    op.drop_index("idx_prompt_templates_visibility", table_name="prompt_templates")
    op.drop_index("idx_prompt_templates_type", table_name="prompt_templates")
    op.drop_index("idx_prompt_templates_owner", table_name="prompt_templates")

    # Drop table
    op.drop_table("prompt_templates")
