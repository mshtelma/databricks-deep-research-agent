"""Add workflow_ref to custom_agents.

Revision ID: 020_add_workflow_ref
Revises: 019_add_source_query_configs
Create Date: 2026-03-15

Allows custom agents to reference a named workflow provided by a plugin
instead of using the default config_translator pipeline.

Part of 012-workflow-provider.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "020_add_workflow_ref"
down_revision: str | None = "019_add_source_query_configs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add workflow_ref column to custom_agents."""
    op.add_column(
        "custom_agents",
        sa.Column("workflow_ref", sa.String(255), nullable=True),
    )


def downgrade() -> None:
    """Remove workflow_ref column from custom_agents."""
    op.drop_column("custom_agents", "workflow_ref")
