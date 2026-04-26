"""Create users table with FK constraints from all user_id/owner_id columns.

Revision ID: 021_create_users_table
Revises: 020_add_workflow_ref
Create Date: 2026-04-14

The migration backfills the new ``users`` table from distinct user_id /
owner_id values across every referencing table BEFORE adding the FK
constraints, so a rolling migration on a non-empty database does not
violate FKs. An ``'anonymous'`` sentinel row is inserted up-front to
satisfy the dev-mode anonymous fallback in middleware/auth.py.

``incognito_sessions`` is intentionally NOT included in the FK list:
incognito sessions are by-design ephemeral and not tied to a long-lived
user identity (a hard FK would couple "delete user" cascades to privacy-
sensitive incognito state). The ``user_id`` column on incognito_sessions
remains as a soft pointer.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "021_create_users_table"
down_revision: str | None = "020_add_workflow_ref"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Tables referencing users via user_id column
_USER_ID_TABLES = [
    "chats",
    "research_sessions",
    "audit_logs",
    "user_preferences",
    "message_feedback",
]

# Tables referencing users via owner_id column
_OWNER_ID_TABLES = [
    "user_data_sources",
    "custom_agents",
    "prompt_templates",
    "uploaded_files",
]


def upgrade() -> None:
    """Create users table, backfill from existing rows, then add FK constraints."""
    # 1. Create users table
    op.create_table(
        "users",
        sa.Column("user_id", sa.String(255), primary_key=True),
        sa.Column("email", sa.String(320), nullable=True),
        sa.Column("display_name", sa.String(255), nullable=True),
        sa.Column(
            "first_seen_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "last_seen_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_users_email", "users", ["email"])
    op.create_index("ix_users_last_seen_at", "users", ["last_seen_at"])

    # 2. Insert anonymous sentinel BEFORE backfill so anon FK references resolve.
    #    Used by middleware/auth.py when in development mode without Databricks
    #    credentials (UserIdentity.anonymous() returns user_id='anonymous').
    op.execute(
        "INSERT INTO users (user_id, email, display_name, first_seen_at, last_seen_at) "
        "VALUES ('anonymous', NULL, 'Anonymous (dev)', now(), now()) "
        "ON CONFLICT (user_id) DO NOTHING"
    )

    # 3. Backfill users from distinct user_id/owner_id across every referencing
    #    table so the FK ADDs in step 4 don't violate. Wrap in a single
    #    statement so the DISTINCT happens once.
    op.execute(
        """
        INSERT INTO users (user_id, first_seen_at, last_seen_at)
        SELECT DISTINCT u.user_id, now(), now()
        FROM (
            SELECT user_id FROM chats WHERE user_id IS NOT NULL
            UNION SELECT user_id FROM research_sessions WHERE user_id IS NOT NULL
            UNION SELECT user_id FROM audit_logs WHERE user_id IS NOT NULL
            UNION SELECT user_id FROM user_preferences WHERE user_id IS NOT NULL
            UNION SELECT user_id FROM incognito_sessions WHERE user_id IS NOT NULL
            UNION SELECT user_id FROM message_feedback WHERE user_id IS NOT NULL
            UNION SELECT owner_id FROM user_data_sources WHERE owner_id IS NOT NULL
            UNION SELECT owner_id FROM custom_agents WHERE owner_id IS NOT NULL
            UNION SELECT owner_id FROM prompt_templates WHERE owner_id IS NOT NULL
            UNION SELECT owner_id FROM uploaded_files WHERE owner_id IS NOT NULL
        ) u
        WHERE u.user_id IS NOT NULL
        ON CONFLICT (user_id) DO NOTHING
        """
    )

    # 4. Add FK constraints from user_id columns
    for table in _USER_ID_TABLES:
        op.create_foreign_key(
            f"fk_{table}_user_id_users",
            table,
            "users",
            ["user_id"],
            ["user_id"],
        )

    # 5. Add FK constraints from owner_id columns
    for table in _OWNER_ID_TABLES:
        op.create_foreign_key(
            f"fk_{table}_owner_id_users",
            table,
            "users",
            ["owner_id"],
            ["user_id"],
        )


def downgrade() -> None:
    """Drop FK constraints and users table."""
    # Drop FKs first (reverse order)
    for table in _OWNER_ID_TABLES:
        op.drop_constraint(
            f"fk_{table}_owner_id_users", table, type_="foreignkey"
        )
    for table in _USER_ID_TABLES:
        op.drop_constraint(
            f"fk_{table}_user_id_users", table, type_="foreignkey"
        )

    op.drop_index("ix_users_last_seen_at", table_name="users")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
