"""Add description column to prompt_templates table.

Revision ID: 018_add_template_description
Revises: 017_custom_agent_model_overrides
Create Date: 2026-02-10

Adds an optional description column for prompt templates.

Part of 009-custom-agent-config post-cleanup.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "018_add_template_description"
down_revision: str | None = "017_custom_agent_model_overrides"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add description column to prompt_templates."""
    op.add_column(
        "prompt_templates",
        sa.Column("description", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    """Remove description column from prompt_templates."""
    op.drop_column("prompt_templates", "description")
