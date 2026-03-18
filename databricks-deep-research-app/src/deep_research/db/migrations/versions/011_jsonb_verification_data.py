"""Add verification_data JSONB column to research_sessions.

This migration adds a JSONB column to store claims and verification summary
in a denormalized format, reducing write queries from 45-200+ to 3-5.

The JSONB structure:
{
    "claims": [
        {
            "claim_text": str,
            "claim_type": "general" | "numeric",
            "position_start": int,
            "position_end": int,
            "confidence_level": "high" | "medium" | "low" | null,
            "verification_verdict": "supported" | "partial" | "unsupported" | "contradicted" | null,
            "verification_reasoning": str | null,
            "abstained": bool,
            "citation_key": str | null,
            "citation_keys": list[str] | null,
            "from_free_block": bool,
            "evidence": {
                "source_url": str,
                "source_title": str | null,
                "quote_text": str,
                "start_offset": int | null,
                "end_offset": int | null,
                "section_heading": str | null,
                "relevance_score": float | null,
                "has_numeric_content": bool
            } | null
        }
    ],
    "summary": {
        "total_claims": int,
        "supported_count": int,
        "partial_count": int,
        "unsupported_count": int,
        "contradicted_count": int,
        "abstained_count": int,
        "unsupported_rate": float,
        "contradicted_rate": float,
        "warning": bool,
        "citation_corrections": int
    }
}

Revision ID: 011_jsonb_verification_data
Revises: 010_source_type_field
Create Date: 2025-01-21
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "011_jsonb_verification_data"
down_revision = "010_source_type_field"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add verification_data JSONB column to research_sessions."""
    op.add_column(
        "research_sessions",
        sa.Column("verification_data", postgresql.JSONB, nullable=True),
    )


def downgrade() -> None:
    """Remove verification_data column from research_sessions."""
    op.drop_column("research_sessions", "verification_data")
