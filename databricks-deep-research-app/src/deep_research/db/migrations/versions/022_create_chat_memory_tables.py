"""Add chat_memory_* tables for chat-scoped durable research memory.

Revision ID: 022_create_chat_memory_tables
Revises: 021_create_users_table
Create Date: 2026-04-18

Adds five tables implementing the chat-scoped memory layer (parallel to
ChatSourcePool): chat_memory_findings, chat_memory_entities,
chat_memory_coverage, chat_memory_files, chat_memory_plugin_ext.

Memory is hydrated per turn, spans all turns in a chat, and unifies
file-derived knowledge (source_step=0) with research-derived findings
(source_step>=1). Agents read a projection of these tables; pools remain
the per-run event-stream layer and stay unchanged.
"""

from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "022_create_chat_memory_tables"
down_revision: str | None = "021_create_users_table"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _timestamp_columns() -> list[sa.Column[Any]]:
    return [
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
    ]


def upgrade() -> None:
    """Create all five chat_memory_* tables with indexes."""
    # chat_memory_findings
    op.create_table(
        "chat_memory_findings",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("chat_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("research_session_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("source_step", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("origin", sa.String(20), nullable=False, server_default="web"),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("confidence", sa.String(10), nullable=False, server_default="medium"),
        sa.Column(
            "entity_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="[]",
        ),
        sa.Column("supersedes_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("content_hash", sa.String(64), nullable=False),
        *_timestamp_columns(),
        sa.PrimaryKeyConstraint("id", name="pk_chat_memory_findings"),
        sa.ForeignKeyConstraint(
            ["chat_id"],
            ["chats.id"],
            name="fk_chat_memory_findings_chat_id_chats",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["research_session_id"],
            ["research_sessions.id"],
            name="fk_chat_memory_findings_research_session_id_research_sessions",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["supersedes_id"],
            ["chat_memory_findings.id"],
            name="fk_chat_memory_findings_supersedes_id_chat_memory_findings",
            ondelete="SET NULL",
        ),
    )
    op.create_index("idx_cmf_chat", "chat_memory_findings", ["chat_id"])
    op.create_index(
        "idx_cmf_chat_hash",
        "chat_memory_findings",
        ["chat_id", "content_hash"],
        unique=True,
    )
    op.create_index(
        "idx_cmf_chat_step",
        "chat_memory_findings",
        ["chat_id", "source_step"],
    )

    # chat_memory_entities
    op.create_table(
        "chat_memory_entities",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("chat_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("entity_type", sa.String(20), nullable=False, server_default="other"),
        sa.Column("summary", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "aliases",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="[]",
        ),
        sa.Column(
            "supporting_finding_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="[]",
        ),
        *_timestamp_columns(),
        sa.PrimaryKeyConstraint("id", name="pk_chat_memory_entities"),
        sa.ForeignKeyConstraint(
            ["chat_id"],
            ["chats.id"],
            name="fk_chat_memory_entities_chat_id_chats",
            ondelete="CASCADE",
        ),
    )
    op.create_index("idx_cme_chat", "chat_memory_entities", ["chat_id"])
    op.create_index(
        "idx_cme_chat_name",
        "chat_memory_entities",
        ["chat_id", "name"],
        unique=True,
    )
    op.create_index(
        "idx_cme_chat_type",
        "chat_memory_entities",
        ["chat_id", "entity_type"],
    )

    # chat_memory_coverage
    op.create_table(
        "chat_memory_coverage",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("chat_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("topic", sa.Text(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="gap"),
        sa.Column("depth", sa.String(16), nullable=False, server_default="surface"),
        *_timestamp_columns(),
        sa.PrimaryKeyConstraint("id", name="pk_chat_memory_coverage"),
        sa.ForeignKeyConstraint(
            ["chat_id"],
            ["chats.id"],
            name="fk_chat_memory_coverage_chat_id_chats",
            ondelete="CASCADE",
        ),
    )
    op.create_index("idx_cmc_chat", "chat_memory_coverage", ["chat_id"])
    op.create_index(
        "idx_cmc_chat_topic",
        "chat_memory_coverage",
        ["chat_id", "topic"],
        unique=True,
    )

    # chat_memory_files
    op.create_table(
        "chat_memory_files",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("chat_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("file_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("one_line_summary", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "entity_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="[]",
        ),
        sa.Column(
            "preprocessed_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
        *_timestamp_columns(),
        sa.PrimaryKeyConstraint("id", name="pk_chat_memory_files"),
        sa.ForeignKeyConstraint(
            ["chat_id"],
            ["chats.id"],
            name="fk_chat_memory_files_chat_id_chats",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["file_id"],
            ["uploaded_files.id"],
            name="fk_chat_memory_files_file_id_uploaded_files",
            ondelete="CASCADE",
        ),
    )
    op.create_index("idx_cmfi_chat", "chat_memory_files", ["chat_id"])
    op.create_index(
        "idx_cmfi_chat_file",
        "chat_memory_files",
        ["chat_id", "file_id"],
        unique=True,
    )

    # chat_memory_plugin_ext
    op.create_table(
        "chat_memory_plugin_ext",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("chat_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("plugin_name", sa.String(128), nullable=False),
        sa.Column(
            "payload_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        *_timestamp_columns(),
        sa.PrimaryKeyConstraint("id", name="pk_chat_memory_plugin_ext"),
        sa.ForeignKeyConstraint(
            ["chat_id"],
            ["chats.id"],
            name="fk_chat_memory_plugin_ext_chat_id_chats",
            ondelete="CASCADE",
        ),
    )
    op.create_index("idx_cmpe_chat", "chat_memory_plugin_ext", ["chat_id"])
    op.create_index(
        "idx_cmpe_chat_plugin",
        "chat_memory_plugin_ext",
        ["chat_id", "plugin_name"],
        unique=True,
    )


def downgrade() -> None:
    """Drop all chat_memory_* tables in reverse dependency order."""
    # plugin_ext
    op.drop_index("idx_cmpe_chat_plugin", table_name="chat_memory_plugin_ext")
    op.drop_index("idx_cmpe_chat", table_name="chat_memory_plugin_ext")
    op.drop_table("chat_memory_plugin_ext")

    # files
    op.drop_index("idx_cmfi_chat_file", table_name="chat_memory_files")
    op.drop_index("idx_cmfi_chat", table_name="chat_memory_files")
    op.drop_table("chat_memory_files")

    # coverage
    op.drop_index("idx_cmc_chat_topic", table_name="chat_memory_coverage")
    op.drop_index("idx_cmc_chat", table_name="chat_memory_coverage")
    op.drop_table("chat_memory_coverage")

    # entities
    op.drop_index("idx_cme_chat_type", table_name="chat_memory_entities")
    op.drop_index("idx_cme_chat_name", table_name="chat_memory_entities")
    op.drop_index("idx_cme_chat", table_name="chat_memory_entities")
    op.drop_table("chat_memory_entities")

    # findings — dropped last because plugin_ext/files/coverage/entities
    # don't reference it; findings self-references via supersedes_id, handled
    # by CASCADE on the FK.
    op.drop_index("idx_cmf_chat_step", table_name="chat_memory_findings")
    op.drop_index("idx_cmf_chat_hash", table_name="chat_memory_findings")
    op.drop_index("idx_cmf_chat", table_name="chat_memory_findings")
    op.drop_table("chat_memory_findings")
