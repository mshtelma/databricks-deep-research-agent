"""Add chat_id column to sources for chat-level source pool.

Revision ID: 006_add_chat_id_to_sources
Revises: 005_add_source_url_unique
Create Date: 2025-01-05 00:00:01.000000

This migration adds a direct chat_id reference to sources for efficient
chat-level source pool queries. Previously, sources were only linked via
research_session_id → message_id → chat_id (4-table join).

The chat_id column enables:
- O(1) lookup of all sources in a chat
- Chat-level URL deduplication via UNIQUE(chat_id, url)
- Hybrid search across all accumulated sources in a conversation
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "006_add_chat_id_to_sources"
down_revision: str | None = "005_add_source_url_unique"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add chat_id column and unique constraint to sources."""
    # Step 1: Add chat_id column (nullable for backward compatibility)
    op.add_column(
        "sources",
        sa.Column(
            "chat_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("chats.id", ondelete="CASCADE"),
            nullable=True,
            comment="Direct chat reference for O(1) chat-level source pool queries",
        ),
    )

    # Step 2: Create index on chat_id for fast lookups
    op.create_index(
        "ix_sources_chat_id",
        "sources",
        ["chat_id"],
    )

    # Step 3: Backfill chat_id from research_session → message → chat
    # This populates chat_id for existing sources
    op.execute("""
        UPDATE sources s
        SET chat_id = m.chat_id
        FROM research_sessions rs
        JOIN messages m ON rs.message_id = m.id
        WHERE s.research_session_id = rs.id
          AND s.chat_id IS NULL
    """)

    # Step 4: Create unique index for chat-level URL deduplication
    # This enables ON CONFLICT (chat_id, url) DO UPDATE in persistence.py
    # Note: Must be non-partial index for PostgreSQL ON CONFLICT inference
    # NULLs are distinct in unique indexes, so (NULL, url) won't conflict
    op.create_index(
        "uq_sources_chat_url",
        "sources",
        ["chat_id", "url"],
        unique=True,
    )


def downgrade() -> None:
    """Remove chat_id column and constraints from sources."""
    # Drop unique index first (was created with create_index, not create_constraint)
    op.drop_index("uq_sources_chat_url", "sources")

    # Drop regular index
    op.drop_index("ix_sources_chat_id", "sources")

    # Drop column
    op.drop_column("sources", "chat_id")
