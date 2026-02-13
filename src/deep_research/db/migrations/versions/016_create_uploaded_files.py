"""Add uploaded_files and file_chunks tables for user file uploads.

Revision ID: 016_create_uploaded_files
Revises: 016_create_custom_agents
Create Date: 2026-02-04

This migration adds tables for user file upload functionality (US7):
- uploaded_files: File metadata and processing status
- file_chunks: Chunked content for search

Part of 007-enterprise-data-sources feature (T087).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "016_create_uploaded_files"
down_revision: str | None = "016_create_custom_agents"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create uploaded_files and file_chunks tables with indexes."""
    # Create uploaded_files table
    op.create_table(
        "uploaded_files",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        # Owner identification
        sa.Column("owner_id", sa.String(255), nullable=False),
        # Session scope (nullable for permanent files)
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
        # File metadata
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("file_type", sa.String(20), nullable=False),
        sa.Column("file_size", sa.Integer(), nullable=False),
        sa.Column("storage_path", sa.String(1024), nullable=False),
        # Processing status
        sa.Column(
            "processing_status",
            sa.String(20),
            nullable=False,
            server_default="pending",
        ),
        sa.Column(
            "chunk_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        # Optional expiration for session-scoped files
        sa.Column(
            "expires_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        # Additional metadata (JSONB)
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        # Timestamps
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
        sa.PrimaryKeyConstraint("id", name="pk_uploaded_files"),
    )

    # Create indexes for uploaded_files
    op.create_index(
        "idx_uploaded_files_owner",
        "uploaded_files",
        ["owner_id"],
    )
    op.create_index(
        "idx_uploaded_files_session",
        "uploaded_files",
        ["session_id"],
    )
    op.create_index(
        "idx_uploaded_files_owner_session",
        "uploaded_files",
        ["owner_id", "session_id"],
    )
    op.create_index(
        "idx_uploaded_files_expires_at",
        "uploaded_files",
        ["expires_at"],
    )
    op.create_index(
        "idx_uploaded_files_status",
        "uploaded_files",
        ["processing_status"],
    )

    # Create file_chunks table
    op.create_table(
        "file_chunks",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        # Foreign key to file
        sa.Column(
            "file_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        # Chunk position
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        # Content
        sa.Column("content", sa.Text(), nullable=False),
        # Location metadata (JSONB)
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        # Timestamps
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
        sa.PrimaryKeyConstraint("id", name="pk_file_chunks"),
        sa.ForeignKeyConstraint(
            ["file_id"],
            ["uploaded_files.id"],
            name="fk_file_chunks_file_id_uploaded_files",
            ondelete="CASCADE",
        ),
    )

    # Create indexes for file_chunks
    op.create_index(
        "idx_file_chunks_file",
        "file_chunks",
        ["file_id"],
    )
    op.create_index(
        "idx_file_chunks_file_index",
        "file_chunks",
        ["file_id", "chunk_index"],
    )


def downgrade() -> None:
    """Drop file_chunks and uploaded_files tables with indexes."""
    # Drop file_chunks indexes
    op.drop_index("idx_file_chunks_file_index", table_name="file_chunks")
    op.drop_index("idx_file_chunks_file", table_name="file_chunks")

    # Drop file_chunks table
    op.drop_table("file_chunks")

    # Drop uploaded_files indexes
    op.drop_index("idx_uploaded_files_status", table_name="uploaded_files")
    op.drop_index("idx_uploaded_files_expires_at", table_name="uploaded_files")
    op.drop_index("idx_uploaded_files_owner_session", table_name="uploaded_files")
    op.drop_index("idx_uploaded_files_session", table_name="uploaded_files")
    op.drop_index("idx_uploaded_files_owner", table_name="uploaded_files")

    # Drop uploaded_files table
    op.drop_table("uploaded_files")
