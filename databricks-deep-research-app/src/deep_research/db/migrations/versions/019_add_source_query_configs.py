"""Add source_query_configs column to custom_agents table.

Revision ID: 019_add_source_query_configs
Revises: 018_add_template_description
Create Date: 2026-02-11

Adds a JSONB column for per-source Vector Search query configuration.
Keys are source names, values are query config overrides (columns,
reranking, query type, strategy).

Part of 009-custom-agent-config (M5).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "019_add_source_query_configs"
down_revision: str | None = "018_add_template_description"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add source_query_configs column to custom_agents."""
    op.add_column(
        "custom_agents",
        sa.Column("source_query_configs", postgresql.JSONB(), nullable=True),
    )


def downgrade() -> None:
    """Remove source_query_configs column from custom_agents."""
    op.drop_column("custom_agents", "source_query_configs")
