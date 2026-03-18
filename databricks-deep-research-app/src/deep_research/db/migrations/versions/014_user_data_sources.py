"""Add user_data_sources table for enterprise data source integration.

Revision ID: 014_user_data_sources
Revises: 013_incognito_support
Create Date: 2026-02-04

This migration adds the user_data_sources table for users to configure
their own enterprise data sources (Vector Search indexes, Genie spaces,
Knowledge Assistants) for use in research.

Part of 007-enterprise-data-sources feature (T011).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "014_user_data_sources"
down_revision: str | None = "013_incognito_support"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create user_data_sources table and indexes."""
    # Create the table
    op.create_table(
        "user_data_sources",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        # Owner identification
        sa.Column("owner_id", sa.String(255), nullable=False),
        # Source type (vector_search, genie, knowledge_assistant, etc.)
        sa.Column("type", sa.String(50), nullable=False),
        # Display name
        sa.Column("name", sa.String(255), nullable=False),
        # Description
        sa.Column("description", sa.Text(), nullable=True),
        # Primary identifier (index name, space ID, endpoint name)
        sa.Column("endpoint_identifier", sa.String(500), nullable=False),
        # Type-specific configuration (JSONB)
        sa.Column(
            "config",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        # Visibility (private, workspace)
        sa.Column(
            "visibility",
            sa.String(20),
            nullable=False,
            server_default="private",
        ),
        # Validation status (pending, valid, invalid, expired)
        sa.Column(
            "validation_status",
            sa.String(20),
            nullable=False,
            server_default="pending",
        ),
        # Last validation timestamp
        sa.Column(
            "last_validated_at",
            sa.DateTime(timezone=True),
            nullable=True,
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
        sa.PrimaryKeyConstraint("id", name="pk_user_data_sources"),
    )

    # Create indexes for efficient queries
    op.create_index(
        "idx_user_data_sources_owner",
        "user_data_sources",
        ["owner_id"],
    )
    op.create_index(
        "idx_user_data_sources_type",
        "user_data_sources",
        ["type"],
    )
    op.create_index(
        "idx_user_data_sources_visibility",
        "user_data_sources",
        ["visibility"],
    )
    op.create_index(
        "idx_user_data_sources_owner_visibility",
        "user_data_sources",
        ["owner_id", "visibility"],
    )
    # Unique constraint: name must be unique per owner
    op.create_index(
        "uq_user_data_sources_owner_name",
        "user_data_sources",
        ["owner_id", "name"],
        unique=True,
    )


def downgrade() -> None:
    """Drop user_data_sources table and indexes."""
    # Drop indexes
    op.drop_index("uq_user_data_sources_owner_name", table_name="user_data_sources")
    op.drop_index("idx_user_data_sources_owner_visibility", table_name="user_data_sources")
    op.drop_index("idx_user_data_sources_visibility", table_name="user_data_sources")
    op.drop_index("idx_user_data_sources_type", table_name="user_data_sources")
    op.drop_index("idx_user_data_sources_owner", table_name="user_data_sources")

    # Drop table
    op.drop_table("user_data_sources")
