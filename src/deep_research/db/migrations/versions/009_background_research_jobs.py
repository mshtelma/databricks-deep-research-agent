"""Add background job support to research_sessions.

Revision ID: 009_background_research_jobs
Revises: 008_research_session_lifecycle
Create Date: 2026-01-11 00:00:00.000000

This migration adds columns to research_sessions for background job management:
1. user_id - Direct lookup without joining through message→chat
2. chat_id - Direct lookup for "active research in this chat"
3. execution_state - Serialized ResearchState.to_dict() for resume
4. worker_id - Which app instance owns this job (for multi-instance)
5. last_heartbeat - Zombie detection (stale if >30s without update)
6. verify_sources - Capture request parameter for resume

These changes enable:
- Research execution decoupled from HTTP request lifecycle
- Background job tracking with concurrency limits per user
- Heartbeat mechanism for zombie job detection
- State checkpointing for future job resumption
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID

# revision identifiers, used by Alembic.
revision: str = "009_background_research_jobs"
down_revision: str | None = "008_research_session_lifecycle"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add background job support columns to research_sessions."""
    # ============================================================
    # 1. Add user_id column (for direct job queries by user)
    # ============================================================
    op.add_column(
        "research_sessions",
        sa.Column(
            "user_id",
            sa.String(100),
            nullable=True,  # Initially nullable for backfill
            comment="User ID for direct job queries",
        ),
    )

    # ============================================================
    # 2. Add chat_id column (for direct job queries by chat)
    # ============================================================
    op.add_column(
        "research_sessions",
        sa.Column(
            "chat_id",
            PGUUID(as_uuid=True),
            nullable=True,  # Initially nullable for backfill
            comment="Chat ID for direct job queries",
        ),
    )

    # ============================================================
    # 3. Add execution_state column (for job resumption)
    # ============================================================
    op.add_column(
        "research_sessions",
        sa.Column(
            "execution_state",
            JSONB,
            nullable=True,
            comment="Serialized ResearchState for job resumption",
        ),
    )

    # ============================================================
    # 4. Add worker_id column (for multi-instance tracking)
    # ============================================================
    op.add_column(
        "research_sessions",
        sa.Column(
            "worker_id",
            sa.String(100),
            nullable=True,
            comment="Worker instance ID that owns this job",
        ),
    )

    # ============================================================
    # 5. Add last_heartbeat column (for zombie detection)
    # ============================================================
    op.add_column(
        "research_sessions",
        sa.Column(
            "last_heartbeat",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="Last heartbeat timestamp for zombie detection",
        ),
    )

    # ============================================================
    # 6. Add verify_sources column (for parameter capture)
    # ============================================================
    op.add_column(
        "research_sessions",
        sa.Column(
            "verify_sources",
            sa.Boolean(),
            nullable=False,
            server_default="true",
            comment="Whether citation verification is enabled",
        ),
    )

    # ============================================================
    # 7. Backfill user_id and chat_id from existing data
    # ============================================================
    # Existing sessions get user_id/chat_id from message→chat
    op.execute(
        """
        UPDATE research_sessions rs
        SET user_id = c.user_id, chat_id = m.chat_id
        FROM messages m
        JOIN chats c ON m.chat_id = c.id
        WHERE rs.message_id = m.id
          AND rs.user_id IS NULL
        """
    )

    # For any remaining rows (shouldn't exist, but be safe)
    # Set placeholder values before NOT NULL constraint
    op.execute(
        """
        UPDATE research_sessions
        SET user_id = 'unknown', chat_id = (SELECT id FROM chats LIMIT 1)
        WHERE user_id IS NULL AND (SELECT COUNT(*) FROM chats) > 0
        """
    )

    # ============================================================
    # 8. Make user_id and chat_id NOT NULL (if data exists)
    # ============================================================
    # Only alter if table has data to avoid constraint issues
    op.execute(
        """
        DO $$
        BEGIN
            -- Check if there are any NULL values remaining
            IF NOT EXISTS (SELECT 1 FROM research_sessions WHERE user_id IS NULL LIMIT 1) THEN
                ALTER TABLE research_sessions ALTER COLUMN user_id SET NOT NULL;
            END IF;
            IF NOT EXISTS (SELECT 1 FROM research_sessions WHERE chat_id IS NULL LIMIT 1) THEN
                ALTER TABLE research_sessions ALTER COLUMN chat_id SET NOT NULL;
            END IF;
        END $$;
        """
    )

    # ============================================================
    # 9. Add foreign key constraint for chat_id
    # ============================================================
    op.create_foreign_key(
        "fk_research_sessions_chat",
        "research_sessions",
        "chats",
        ["chat_id"],
        ["id"],
        ondelete="CASCADE",
    )

    # ============================================================
    # 10. Add indexes for efficient job queries
    # ============================================================
    # Index for user's jobs by status (most common query)
    op.create_index(
        "idx_research_sessions_user_status",
        "research_sessions",
        ["user_id", "status"],
    )

    # Index for chat's active jobs
    op.create_index(
        "idx_research_sessions_chat_status",
        "research_sessions",
        ["chat_id", "status"],
    )

    # Partial index for zombie detection (only in_progress jobs)
    op.create_index(
        "idx_research_sessions_heartbeat",
        "research_sessions",
        ["last_heartbeat", "status"],
        postgresql_where=sa.text("status = 'in_progress'"),
    )


def downgrade() -> None:
    """Remove background job support columns from research_sessions."""
    # Drop indexes first
    op.drop_index("idx_research_sessions_heartbeat", "research_sessions")
    op.drop_index("idx_research_sessions_chat_status", "research_sessions")
    op.drop_index("idx_research_sessions_user_status", "research_sessions")

    # Drop foreign key
    op.drop_constraint("fk_research_sessions_chat", "research_sessions", type_="foreignkey")

    # Drop columns
    op.drop_column("research_sessions", "verify_sources")
    op.drop_column("research_sessions", "last_heartbeat")
    op.drop_column("research_sessions", "worker_id")
    op.drop_column("research_sessions", "execution_state")
    op.drop_column("research_sessions", "chat_id")
    op.drop_column("research_sessions", "user_id")
