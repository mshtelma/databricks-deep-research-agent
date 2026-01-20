"""Add citation_keys array column to claims table.

Revision ID: 004_add_citation_keys_array
Revises: 003_add_citation_key
Create Date: 2025-12-30 00:00:00.000000

This migration adds the citation_keys array column to support sentences with
multiple citation markers. For example, a sentence like:
  "Tax rate is 20% [Arxiv][Arxiv-2][Github]"
would have citation_keys = ["Arxiv", "Arxiv-2", "Github"].

This allows the frontend to resolve ALL markers to the same claim, not just
the first one (which is stored in citation_key for backward compatibility).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ARRAY

# revision identifiers, used by Alembic.
revision: str = "004_add_citation_keys_array"
down_revision: str | None = "003_add_citation_key"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add citation_keys array column to claims table."""
    op.add_column(
        "claims",
        sa.Column(
            "citation_keys",
            ARRAY(sa.String(50)),
            nullable=True,
            comment="All citation keys in this claim for multi-marker sentences",
        ),
    )


def downgrade() -> None:
    """Remove citation_keys array column from claims table."""
    op.drop_column("claims", "citation_keys")
