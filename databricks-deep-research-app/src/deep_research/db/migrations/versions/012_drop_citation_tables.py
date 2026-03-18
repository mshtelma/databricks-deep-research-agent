"""Drop normalized citation tables (replaced by JSONB).

Revision ID: 012_drop_citation_tables
Revises: 011_jsonb_verification_data
Create Date: 2025-01-21

This migration drops the normalized citation tables that have been replaced
by the verification_data JSONB column on research_sessions (Migration 011).

Tables dropped:
- verification_summaries
- citation_corrections
- numeric_claims
- citations
- evidence_spans
- claims

The data is now stored in the verification_data JSONB column which reduces
write queries from 45-200+ to 3-5 queries per research completion.

IMPORTANT: This migration is NOT reversible in terms of data.
The downgrade creates empty tables but does not migrate JSONB data back.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "012_drop_citation_tables"
down_revision: str | None = "011_jsonb_verification_data"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Drop normalized citation tables (replaced by JSONB)."""
    # Drop in reverse FK order to avoid constraint violations
    op.drop_table("verification_summaries")
    op.drop_table("citation_corrections")
    op.drop_table("numeric_claims")
    op.drop_table("citations")
    op.drop_table("evidence_spans")
    op.drop_table("claims")


def downgrade() -> None:
    """Recreate citation tables for rollback.

    WARNING: This creates empty tables. Data stored in verification_data JSONB
    will NOT be migrated back to these tables. Only use this for emergency
    rollback if the JSONB migration needs to be reverted.
    """
    # Create claims table
    op.create_table(
        "claims",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("message_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("claim_text", sa.Text, nullable=False),
        sa.Column("claim_type", sa.String(20), nullable=False),
        sa.Column("confidence_level", sa.String(20), nullable=True),
        sa.Column("position_start", sa.Integer, nullable=False),
        sa.Column("position_end", sa.Integer, nullable=False),
        sa.Column("verification_verdict", sa.String(20), nullable=True),
        sa.Column("verification_reasoning", sa.Text, nullable=True),
        sa.Column("abstained", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("citation_key", sa.String(100), nullable=True),
        sa.Column("citation_keys", postgresql.ARRAY(sa.String(100)), nullable=True),
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
        sa.ForeignKeyConstraint(["message_id"], ["messages.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_claims_message_id", "claims", ["message_id"])
    op.create_index("idx_claims_verdict", "claims", ["verification_verdict"])

    # Create evidence_spans table
    op.create_table(
        "evidence_spans",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("quote_text", sa.Text, nullable=False),
        sa.Column("start_offset", sa.Integer, nullable=True),
        sa.Column("end_offset", sa.Integer, nullable=True),
        sa.Column("section_heading", sa.String(500), nullable=True),
        sa.Column("relevance_score", sa.Float, nullable=True),
        sa.Column("has_numeric_content", sa.Boolean, nullable=False, server_default="false"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["source_id"], ["sources.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_evidence_spans_source_id", "evidence_spans", ["source_id"])

    # Create citations table
    op.create_table(
        "citations",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("claim_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("evidence_span_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("confidence_score", sa.Float, nullable=True),
        sa.Column("is_primary", sa.Boolean, nullable=False, server_default="true"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["claim_id"], ["claims.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["evidence_span_id"], ["evidence_spans.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("claim_id", "evidence_span_id", name="uq_claim_evidence"),
    )
    op.create_index("ix_citations_claim_id", "citations", ["claim_id"])
    op.create_index("ix_citations_evidence_span_id", "citations", ["evidence_span_id"])

    # Create numeric_claims table
    op.create_table(
        "numeric_claims",
        sa.Column("claim_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("raw_value", sa.Text, nullable=False),
        sa.Column("normalized_value", sa.Numeric, nullable=True),
        sa.Column("unit", sa.String(50), nullable=True),
        sa.Column("entity_reference", sa.Text, nullable=True),
        sa.Column("derivation_type", sa.String(20), nullable=False),
        sa.Column("computation_details", postgresql.JSONB, nullable=True),
        sa.Column("assumptions", postgresql.JSONB, nullable=True),
        sa.Column("qa_verification", postgresql.JSONB, nullable=True),
        sa.PrimaryKeyConstraint("claim_id"),
        sa.ForeignKeyConstraint(["claim_id"], ["claims.id"], ondelete="CASCADE"),
    )

    # Create citation_corrections table
    op.create_table(
        "citation_corrections",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("claim_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("original_evidence_span_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("corrected_evidence_span_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("correction_type", sa.String(20), nullable=False),
        sa.Column("reasoning", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["claim_id"], ["claims.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["original_evidence_span_id"],
            ["evidence_spans.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["corrected_evidence_span_id"],
            ["evidence_spans.id"],
            ondelete="SET NULL",
        ),
    )
    op.create_index("ix_citation_corrections_claim_id", "citation_corrections", ["claim_id"])

    # Create verification_summaries table
    op.create_table(
        "verification_summaries",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("message_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("total_claims", sa.Integer, nullable=False, server_default="0"),
        sa.Column("supported_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("partial_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("unsupported_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("contradicted_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("abstained_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("unsupported_rate", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("contradicted_rate", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("warning", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("citation_corrections", sa.Integer, nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["message_id"], ["messages.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("message_id", name="uq_verification_summaries_message"),
    )
    op.create_index("ix_verification_summaries_message_id", "verification_summaries", ["message_id"])
