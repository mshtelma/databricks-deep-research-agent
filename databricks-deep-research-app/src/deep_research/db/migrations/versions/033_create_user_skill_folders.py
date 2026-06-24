"""Create user_skill_folders table for user-addable skill roots (Feature 2.2 / A3).

Revision ID: 033_create_user_skill_folders
Revises: 032_add_promotion_trace
Create Date: 2026-06-23

A *skill folder* is a user-registered workspace-FS or UC-Volume path that the
runtime ``WorkspaceFsSkillStore`` scans (under the user's OBO identity) in
addition to the built-in roots (``~/.skills``, ``~/.assistant/skills``). Each row
is owned by one user; ``UNIQUE(user_id, path)`` keeps re-adding idempotent.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "033_create_user_skill_folders"
down_revision: str | None = "032_add_promotion_trace"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the user_skill_folders table with a per-user uniqueness index."""
    op.create_table(
        "user_skill_folders",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("path", sa.String(1024), nullable=False),
        # 'workspace' (a /Workspace path) or 'volume' (a /Volumes UC path).
        sa.Column(
            "kind",
            sa.String(20),
            nullable=False,
            server_default="workspace",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_user_skill_folders"),
        sa.UniqueConstraint(
            "user_id", "path", name="uq_user_skill_folders_user_path"
        ),
    )
    op.create_index(
        "idx_user_skill_folders_user", "user_skill_folders", ["user_id"]
    )


def downgrade() -> None:
    """Drop the user_skill_folders table and its index."""
    op.drop_index("idx_user_skill_folders_user", table_name="user_skill_folders")
    op.drop_table("user_skill_folders")
