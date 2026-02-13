"""Add model_overrides and domain filter columns to custom_agents table.

Revision ID: 017_custom_agent_model_overrides
Revises: 016_create_uploaded_files
Create Date: 2026-02-10

Adds columns for per-agent model tier overrides and domain filtering:
- model_overrides: JSONB mapping tier names to endpoint identifiers
- domain_filter_mode: include/exclude/both filter mode
- include_domains: JSONB array of domain whitelist patterns
- exclude_domains: JSONB array of domain blacklist patterns

Part of 009-custom-agent-config (T001).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "017_custom_agent_model_overrides"
down_revision: str | None = "016_create_uploaded_files"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add model_overrides and domain filter columns to custom_agents."""
    op.add_column(
        "custom_agents",
        sa.Column("model_overrides", postgresql.JSONB(), nullable=True),
    )
    op.add_column(
        "custom_agents",
        sa.Column("domain_filter_mode", sa.String(20), nullable=True),
    )
    op.add_column(
        "custom_agents",
        sa.Column("include_domains", postgresql.JSONB(), nullable=True),
    )
    op.add_column(
        "custom_agents",
        sa.Column("exclude_domains", postgresql.JSONB(), nullable=True),
    )


def downgrade() -> None:
    """Remove model_overrides and domain filter columns from custom_agents."""
    op.drop_column("custom_agents", "exclude_domains")
    op.drop_column("custom_agents", "include_domains")
    op.drop_column("custom_agents", "domain_filter_mode")
    op.drop_column("custom_agents", "model_overrides")
