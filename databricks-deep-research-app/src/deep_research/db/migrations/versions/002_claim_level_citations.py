"""Add claim-level citation tables.

Revision ID: 002_claim_level_citations
Revises: 001_initial_schema
Create Date: 2025-12-25 00:00:00.000000

This migration adds tables for the 6-stage citation verification pipeline:
- claims: Atomic factual assertions extracted from messages
- evidence_spans: Citable quotes from source documents
- citations: Links between claims and evidence
- numeric_claims: Extended metadata for numeric claims
- citation_corrections: Tracks citation fixes during post-processing
- verification_summaries: Cached verification statistics per message
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "002_claim_level_citations"
down_revision: str | None = "001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create claim-level citation tables."""
    # Add source location columns to sources table (T012)
    op.add_column("sources", sa.Column("total_pages", sa.Integer, nullable=True))
    op.add_column("sources", sa.Column("detected_sections", sa.Text, nullable=True))
    op.add_column("sources", sa.Column("content_type", sa.String(50), nullable=True))

    # Create claims table (T006)
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
        sa.CheckConstraint(
            "claim_type IN ('general', 'numeric')",
            name="ck_claims_type",
        ),
        sa.CheckConstraint(
            "verification_verdict IS NULL OR "
            "verification_verdict IN ('supported', 'partial', 'unsupported', 'contradicted')",
            name="ck_claims_verdict",
        ),
        sa.CheckConstraint(
            "confidence_level IS NULL OR "
            "confidence_level IN ('high', 'medium', 'low')",
            name="ck_claims_confidence",
        ),
        sa.CheckConstraint(
            "position_start < position_end",
            name="ck_claims_position_order",
        ),
    )
    op.create_index("ix_claims_message_id", "claims", ["message_id"])
    op.create_index("idx_claims_verdict", "claims", ["verification_verdict"])
    op.create_index("idx_claims_confidence", "claims", ["confidence_level"])

    # Create evidence_spans table (T007)
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
    op.create_index("idx_evidence_spans_relevance", "evidence_spans", ["relevance_score"])

    # Create citations table (T008)
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
        sa.CheckConstraint(
            "confidence_score IS NULL OR (confidence_score >= 0.0 AND confidence_score <= 1.0)",
            name="ck_citations_confidence_range",
        ),
    )
    op.create_index("ix_citations_claim_id", "citations", ["claim_id"])
    op.create_index("ix_citations_evidence_span_id", "citations", ["evidence_span_id"])

    # Create numeric_claims table (T009)
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
        sa.CheckConstraint(
            "derivation_type IN ('direct', 'computed')",
            name="ck_numeric_claims_derivation",
        ),
    )

    # Create citation_corrections table (T010)
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
        sa.CheckConstraint(
            "correction_type IN ('keep', 'replace', 'remove', 'add_alternate')",
            name="ck_citation_corrections_type",
        ),
    )
    op.create_index("ix_citation_corrections_claim_id", "citation_corrections", ["claim_id"])
    op.create_index("idx_citation_corrections_type", "citation_corrections", ["correction_type"])

    # Create verification_summaries table (T011)
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


def downgrade() -> None:
    """Drop claim-level citation tables."""
    op.drop_table("verification_summaries")
    op.drop_table("citation_corrections")
    op.drop_table("numeric_claims")
    op.drop_table("citations")
    op.drop_table("evidence_spans")
    op.drop_table("claims")

    # Remove source location columns
    op.drop_column("sources", "content_type")
    op.drop_column("sources", "detected_sections")
    op.drop_column("sources", "total_pages")
