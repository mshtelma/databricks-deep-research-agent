"""Add citation_key column to claims table.

Revision ID: 003_add_citation_key
Revises: 002_claim_level_citations
Create Date: 2025-12-28 00:00:00.000000

This migration adds the citation_key column to the claims table for
human-readable citation markers (e.g., [Arxiv], [Zhipu] instead of [0], [1]).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "003_add_citation_key"
down_revision: str | None = "002_claim_level_citations"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add citation_key column to claims table."""
    op.add_column(
        "claims",
        sa.Column(
            "citation_key",
            sa.String(50),
            nullable=True,
            comment="Human-readable citation key like 'Arxiv' or 'Zhipu'",
        ),
    )


def downgrade() -> None:
    """Remove citation_key column from claims table."""
    op.drop_column("claims", "citation_key")
