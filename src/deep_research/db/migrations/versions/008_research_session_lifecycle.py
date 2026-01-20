"""Add research session lifecycle support for crash resilience.

Revision ID: 008_research_session_lifecycle
Revises: 007_tiered_query_modes
Create Date: 2026-01-09 00:00:00.000000

This migration adds:
1. sequence_number column to research_events for guaranteed event ordering
2. Makes message.content nullable (agent placeholder at start has NULL content)

These changes enable:
- Research session created at START with IN_PROGRESS status
- Events persisted during streaming (FK to session satisfied)
- Agent message updated with final content at END
- Frontend can reconnect to in-progress research
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "008_research_session_lifecycle"
down_revision: str | None = "007_tiered_query_modes"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add research session lifecycle support."""
    # ============================================================
    # 1. Add sequence_number to research_events
    # ============================================================
    # Sequence numbers guarantee event ordering even if timestamps collide
    # (microsecond precision may not be enough for rapid event emission)
    op.add_column(
        "research_events",
        sa.Column(
            "sequence_number",
            sa.Integer(),
            nullable=True,  # Nullable for existing rows
            comment="Monotonically increasing sequence number per session",
        ),
    )

    # Composite index for efficient event retrieval by sequence
    # This is critical for reconnection: "get all events where sequence > X"
    op.create_index(
        "ix_research_events_session_sequence",
        "research_events",
        ["research_session_id", "sequence_number"],
    )

    # ============================================================
    # 2. Make message.content nullable
    # ============================================================
    # Agent message is created at START with NULL content (placeholder)
    # Content is updated to final_report at END when synthesis completes
    op.alter_column(
        "messages",
        "content",
        existing_type=sa.Text(),
        nullable=True,
    )


def downgrade() -> None:
    """Remove research session lifecycle support."""
    # Make message.content NOT NULL again
    # First, set any NULL values to empty string to avoid constraint violation
    op.execute("UPDATE messages SET content = '' WHERE content IS NULL")
    op.alter_column(
        "messages",
        "content",
        existing_type=sa.Text(),
        nullable=False,
    )

    # Drop sequence index and column
    op.drop_index("ix_research_events_session_sequence", "research_events")
    op.drop_column("research_events", "sequence_number")
