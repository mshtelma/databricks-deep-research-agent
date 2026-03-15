"""Add incognito chat support.

Revision ID: 013_incognito_support
Revises: 012_drop_citation_tables
Create Date: 2026-01-25

This migration adds support for incognito (ephemeral) chats:
1. Creates the `chattype` enum for distinguishing regular vs incognito chats
2. Creates the `incognito_sessions` table for server-side session tracking
3. Adds `chat_type` and `incognito_session_id` columns to the `chats` table
4. Creates necessary indexes for efficient queries

Incognito chats are:
- Stored server-side (survives page refresh)
- Associated with a browser session via httpOnly cookie
- Automatically deleted when session expires (1-hour idle timeout)
- Limited to 5 concurrent chats per session
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "013_incognito_support"
down_revision: str | None = "012_drop_citation_tables"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add incognito chat support."""
    # 1. Create ChatType enum
    chattype_enum = postgresql.ENUM("regular", "incognito", name="chattype")
    chattype_enum.create(op.get_bind(), checkfirst=True)

    # 2. Create incognito_sessions table
    op.create_table(
        "incognito_sessions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("session_token", sa.String(64), nullable=False),
        sa.Column("last_activity", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
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
        sa.PrimaryKeyConstraint("id", name="pk_incognito_sessions"),
    )

    # Create indexes for incognito_sessions
    op.create_index(
        "idx_incognito_sessions_token",
        "incognito_sessions",
        ["session_token"],
        unique=True,
    )
    op.create_index(
        "idx_incognito_sessions_expires",
        "incognito_sessions",
        ["expires_at"],
    )
    op.create_index(
        "idx_incognito_sessions_user",
        "incognito_sessions",
        ["user_id"],
    )

    # 3. Add chat_type column to chats table
    op.add_column(
        "chats",
        sa.Column(
            "chat_type",
            postgresql.ENUM("regular", "incognito", name="chattype", create_type=False),
            nullable=False,
            server_default="regular",
        ),
    )

    # 4. Add incognito_session_id FK column to chats table
    op.add_column(
        "chats",
        sa.Column(
            "incognito_session_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
    )

    # 5. Add foreign key constraint
    op.create_foreign_key(
        "fk_chats_incognito_session_id_incognito_sessions",
        "chats",
        "incognito_sessions",
        ["incognito_session_id"],
        ["id"],
        ondelete="CASCADE",
    )

    # 6. Create indexes for chat filtering
    op.create_index("idx_chats_type", "chats", ["chat_type"])
    op.create_index("idx_chats_incognito_session", "chats", ["incognito_session_id"])


def downgrade() -> None:
    """Remove incognito chat support."""
    # Drop indexes
    op.drop_index("idx_chats_incognito_session", table_name="chats")
    op.drop_index("idx_chats_type", table_name="chats")

    # Drop foreign key
    op.drop_constraint(
        "fk_chats_incognito_session_id_incognito_sessions",
        "chats",
        type_="foreignkey",
    )

    # Drop columns from chats
    op.drop_column("chats", "incognito_session_id")
    op.drop_column("chats", "chat_type")

    # Drop incognito_sessions table
    op.drop_index("idx_incognito_sessions_user", table_name="incognito_sessions")
    op.drop_index("idx_incognito_sessions_expires", table_name="incognito_sessions")
    op.drop_index("idx_incognito_sessions_token", table_name="incognito_sessions")
    op.drop_table("incognito_sessions")

    # Drop enum type
    postgresql.ENUM(name="chattype").drop(op.get_bind(), checkfirst=True)
