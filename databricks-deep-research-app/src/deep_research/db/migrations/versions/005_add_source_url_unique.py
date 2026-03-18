"""Add unique constraint on sources(research_session_id, url).

Revision ID: 005_add_source_url_unique
Revises: 004_add_citation_keys_array
Create Date: 2025-01-05 00:00:00.000000

This migration adds a unique constraint to prevent duplicate URLs within
the same research session. This enables atomic upsert operations using
ON CONFLICT and eliminates race conditions in source persistence.

Steps:
1. Remove any existing duplicates (keep newest by fetched_at)
2. Add unique constraint
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "005_add_source_url_unique"
down_revision: str | None = "004_add_citation_keys_array"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add unique constraint on sources(research_session_id, url)."""
    # Step 1: Remove any existing duplicates (keep newest by fetched_at)
    # This uses a self-join to find and delete older duplicates
    op.execute("""
        DELETE FROM sources s1
        USING sources s2
        WHERE s1.research_session_id = s2.research_session_id
          AND s1.url = s2.url
          AND s1.id != s2.id
          AND s1.fetched_at < s2.fetched_at
    """)

    # Step 2: Add unique constraint
    op.create_unique_constraint(
        "uq_sources_session_url",
        "sources",
        ["research_session_id", "url"],
    )


def downgrade() -> None:
    """Remove unique constraint on sources(research_session_id, url)."""
    op.drop_constraint("uq_sources_session_url", "sources", type_="unique")
