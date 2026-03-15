"""Add source_type field for multi-source citations.

Revision ID: 010_source_type_field
Revises: 009_background_research_jobs
Create Date: 2026-01-17 00:00:00.000000

This migration adds columns to sources table for multi-source citation attribution (US7):
1. source_type - Type of source (web, vector_search, knowledge_assistant, custom)
2. source_metadata - JSONB field for source-specific metadata (index name, score, etc.)

These changes enable:
- Visual distinction between sources from different retrieval tools
- Source-specific metadata for citation attribution
- Plugin extensibility via custom source types
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = "010_source_type_field"
down_revision: str | None = "009_background_research_jobs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add source_type and source_metadata columns."""
    # Add source_type with default 'web' for existing rows
    op.add_column(
        "sources",
        sa.Column(
            "source_type",
            sa.String(50),
            nullable=False,
            server_default="web",
        ),
    )

    # Add source_metadata JSONB column (nullable)
    op.add_column(
        "sources",
        sa.Column(
            "source_metadata",
            JSONB,
            nullable=True,
        ),
    )

    # Add index on source_type for filtering queries
    op.create_index(
        "ix_sources_source_type",
        "sources",
        ["source_type"],
    )


def downgrade() -> None:
    """Remove source_type and source_metadata columns."""
    op.drop_index("ix_sources_source_type", table_name="sources")
    op.drop_column("sources", "source_metadata")
    op.drop_column("sources", "source_type")
