"""Add tiered query modes support.

Revision ID: 007_tiered_query_modes
Revises: 006_add_chat_id_to_sources
Create Date: 2026-01-05 00:00:00.000000

This migration adds:
1. query_mode column to research_sessions (simple/web_search/deep_research)
2. default_query_mode column to user_preferences
3. Source tracking columns: is_cited, step_index, step_title, crawl_status, error_reason
4. New research_events table for activity event persistence
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "007_tiered_query_modes"
down_revision: str | None = "006_add_chat_id_to_sources"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add tiered query modes support."""
    # ============================================================
    # 1. Add query_mode to research_sessions
    # ============================================================
    op.add_column(
        "research_sessions",
        sa.Column(
            "query_mode",
            sa.String(20),
            nullable=False,
            server_default="deep_research",
            comment="Query mode: simple, web_search, or deep_research",
        ),
    )

    # ============================================================
    # 2. Add default_query_mode to user_preferences
    # ============================================================
    op.add_column(
        "user_preferences",
        sa.Column(
            "default_query_mode",
            sa.String(20),
            nullable=False,
            server_default="simple",
            comment="User's default query mode",
        ),
    )

    # ============================================================
    # 3. Add source tracking columns
    # ============================================================
    # is_cited: Whether source is cited in final report
    op.add_column(
        "sources",
        sa.Column(
            "is_cited",
            sa.Boolean(),
            nullable=False,
            server_default="false",
            comment="Whether source is cited in final report",
        ),
    )

    # step_index: Research step that visited this source
    op.add_column(
        "sources",
        sa.Column(
            "step_index",
            sa.Integer(),
            nullable=True,
            comment="Research step index that visited this source",
        ),
    )

    # step_title: Title of research step
    op.add_column(
        "sources",
        sa.Column(
            "step_title",
            sa.String(255),
            nullable=True,
            comment="Title of research step that visited this source",
        ),
    )

    # crawl_status: Status of crawl attempt
    op.add_column(
        "sources",
        sa.Column(
            "crawl_status",
            sa.String(20),
            nullable=False,
            server_default="success",
            comment="Crawl status: success, failed, timeout, blocked",
        ),
    )

    # error_reason: Error message if crawl failed
    op.add_column(
        "sources",
        sa.Column(
            "error_reason",
            sa.Text(),
            nullable=True,
            comment="Error message if crawl failed",
        ),
    )

    # Indexes for source tracking queries
    op.create_index(
        "ix_sources_is_cited",
        "sources",
        ["research_session_id", "is_cited"],
    )
    op.create_index(
        "ix_sources_step_index",
        "sources",
        ["research_session_id", "step_index"],
    )

    # ============================================================
    # 4. Create research_events table
    # ============================================================
    op.create_table(
        "research_events",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "research_session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("research_sessions.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "event_type",
            sa.String(50),
            nullable=False,
            comment="Event type (e.g., claim_verified, tool_call, step_started)",
        ),
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
            comment="When event occurred",
        ),
        sa.Column(
            "payload",
            postgresql.JSONB(),
            nullable=False,
            server_default="{}",
            comment="Event-specific data",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )

    # Composite index for efficient event retrieval
    op.create_index(
        "ix_research_events_session_timestamp",
        "research_events",
        ["research_session_id", "timestamp"],
    )

    # ============================================================
    # 5. Backfill existing data
    # ============================================================
    # Set all existing sources as cited (they were before this feature)
    op.execute("UPDATE sources SET is_cited = true WHERE is_cited = false")


def downgrade() -> None:
    """Remove tiered query modes support."""
    # Drop research_events table
    op.drop_index("ix_research_events_session_timestamp", "research_events")
    op.drop_table("research_events")

    # Drop source tracking indexes
    op.drop_index("ix_sources_step_index", "sources")
    op.drop_index("ix_sources_is_cited", "sources")

    # Drop source tracking columns
    op.drop_column("sources", "error_reason")
    op.drop_column("sources", "crawl_status")
    op.drop_column("sources", "step_title")
    op.drop_column("sources", "step_index")
    op.drop_column("sources", "is_cited")

    # Drop user_preferences column
    op.drop_column("user_preferences", "default_query_mode")

    # Drop research_sessions column
    op.drop_column("research_sessions", "query_mode")
