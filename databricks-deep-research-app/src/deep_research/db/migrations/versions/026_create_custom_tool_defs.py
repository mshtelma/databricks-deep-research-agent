"""Create custom_tool_defs table for user-defined tool definitions.

Revision ID: 026_create_custom_tool_defs
Revises: 025_create_agent_revisions
Create Date: 2026-04-27

Each row represents a user-authored tool definition that references a
factory_ref from the BUILTIN_FACTORIES allow-list.  The factory_ref column
is validated at the API layer against a static dict — never resolved via
importlib.import_module.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "026_create_custom_tool_defs"
down_revision: str | None = "025_create_agent_revisions"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create custom_tool_defs table with indexes."""
    op.create_table(
        "custom_tool_defs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("owner_id", sa.String(255), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column(
            "kind",
            sa.Integer(),
            nullable=False,
            server_default="15",  # 15 = custom ToolKind
        ),
        sa.Column(
            "config_schema",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("factory_ref", sa.String(255), nullable=False),
        sa.Column("etag", sa.String(40), nullable=False),
        sa.Column(
            "visibility",
            sa.String(50),
            nullable=False,
            server_default="private",
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
        sa.PrimaryKeyConstraint("id", name="pk_custom_tool_defs"),
        sa.UniqueConstraint(
            "owner_id",
            "name",
            name="uq_custom_tool_defs_owner_name",
        ),
    )
    op.create_index(
        "idx_custom_tool_defs_owner",
        "custom_tool_defs",
        ["owner_id"],
    )
    op.create_index(
        "idx_custom_tool_defs_owner_visibility",
        "custom_tool_defs",
        ["owner_id", "visibility"],
    )
    op.create_index(
        "idx_custom_tool_defs_visibility",
        "custom_tool_defs",
        ["visibility"],
        postgresql_where=sa.text("visibility != 'private'"),
    )


def downgrade() -> None:
    """Drop custom_tool_defs table and its indexes."""
    op.drop_index("idx_custom_tool_defs_visibility", table_name="custom_tool_defs")
    op.drop_index("idx_custom_tool_defs_owner_visibility", table_name="custom_tool_defs")
    op.drop_index("idx_custom_tool_defs_owner", table_name="custom_tool_defs")
    op.drop_table("custom_tool_defs")
