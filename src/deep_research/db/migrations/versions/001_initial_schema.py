"""Initial schema with all tables.

Revision ID: 001_initial_schema
Revises:
Create Date: 2024-01-01 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create all tables."""
    # Create chats table
    op.create_table(
        "chats",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("title", sa.String(255), nullable=True),
        sa.Column(
            "status",
            sa.String(20),
            nullable=False,
            server_default="active",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metadata", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_chats_user_id", "chats", ["user_id"])
    op.create_index("ix_chats_status", "chats", ["status"])
    op.create_index("ix_chats_created_at", "chats", ["created_at"])

    # Create messages table
    op.create_table(
        "messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chat_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("research_session_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("is_edited", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("metadata", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["chat_id"], ["chats.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_messages_chat_id", "messages", ["chat_id"])
    op.create_index("ix_messages_created_at", "messages", ["created_at"])

    # Full-text search index on message content
    op.execute(
        """
        CREATE INDEX ix_messages_content_fts
        ON messages USING gin(to_tsvector('english', content))
        """
    )

    # Create research_sessions table
    op.create_table(
        "research_sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("message_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("query", sa.Text, nullable=False),
        sa.Column(
            "status",
            sa.String(20),
            nullable=False,
            server_default="in_progress",
        ),
        sa.Column("plan", postgresql.JSONB, nullable=True),
        sa.Column("observations", postgresql.JSONB, nullable=True),
        sa.Column("reasoning_steps", postgresql.JSONB, nullable=False, server_default="[]"),
        sa.Column("research_depth", sa.String(20), nullable=False, server_default="auto"),
        sa.Column("query_classification", postgresql.JSONB, nullable=True),
        # Execution state columns
        sa.Column("current_step_index", sa.Integer, nullable=True),
        sa.Column("plan_iterations", sa.Integer, nullable=False, server_default="1"),
        sa.Column("current_agent", sa.String(50), nullable=True),
        # Timestamp columns (from TimestampMixin in BaseModel)
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["message_id"], ["messages.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_research_sessions_message_id", "research_sessions", ["message_id"])
    op.create_index("ix_research_sessions_status", "research_sessions", ["status"])

    # Create sources table
    op.create_table(
        "sources",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("research_session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("url", sa.Text, nullable=False),
        sa.Column("title", sa.String(500), nullable=True),
        sa.Column("snippet", sa.Text, nullable=True),
        sa.Column("content", sa.Text, nullable=True),
        sa.Column("relevance_score", sa.Float, nullable=True),
        sa.Column(
            "fetched_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["research_session_id"],
            ["research_sessions.id"],
            ondelete="CASCADE",
        ),
    )
    op.create_index("ix_sources_research_session_id", "sources", ["research_session_id"])
    op.create_index("ix_sources_url", "sources", ["url"])

    # Create user_preferences table
    op.create_table(
        "user_preferences",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", sa.String(255), nullable=False, unique=True),
        sa.Column(
            "default_research_depth",
            sa.String(20),
            nullable=False,
            server_default="auto",
        ),
        sa.Column("system_instructions", sa.Text, nullable=True),
        sa.Column("theme", sa.String(20), nullable=False, server_default="system"),
        sa.Column("notifications_enabled", sa.Boolean, nullable=False, server_default="true"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_user_preferences_user_id", "user_preferences", ["user_id"], unique=True)

    # Create message_feedback table
    op.create_table(
        "message_feedback",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("message_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("rating", sa.String(20), nullable=False),  # 'positive', 'negative'
        sa.Column("feedback_text", sa.Text, nullable=True),
        sa.Column("feedback_category", sa.String(50), nullable=True),
        sa.Column("trace_id", sa.String(100), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["message_id"], ["messages.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_message_feedback_message_id", "message_feedback", ["message_id"])
    op.create_index("ix_message_feedback_user_id", "message_feedback", ["user_id"])
    op.create_index("ix_message_feedback_rating", "message_feedback", ["rating"])

    # Create audit_logs table
    op.create_table(
        "audit_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("action", sa.String(50), nullable=False),
        sa.Column("resource_type", sa.String(50), nullable=False),
        sa.Column("resource_id", sa.String(100), nullable=True),
        sa.Column("details", postgresql.JSONB, nullable=True),
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.Column("user_agent", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_audit_logs_user_id", "audit_logs", ["user_id"])
    op.create_index("ix_audit_logs_action", "audit_logs", ["action"])
    op.create_index("ix_audit_logs_resource_type", "audit_logs", ["resource_type"])
    op.create_index("ix_audit_logs_created_at", "audit_logs", ["created_at"])

    # Add foreign key from messages to research_sessions (after research_sessions is created)
    op.create_foreign_key(
        "fk_messages_research_session",
        "messages",
        "research_sessions",
        ["research_session_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    """Drop all tables."""
    op.drop_constraint("fk_messages_research_session", "messages", type_="foreignkey")
    op.drop_table("audit_logs")
    op.drop_table("message_feedback")
    op.drop_table("user_preferences")
    op.drop_table("sources")
    op.drop_table("research_sessions")
    op.drop_table("messages")
    op.drop_table("chats")
