"""Add research_sessions.promotion_trace (spec Wave 5 / feature 6.1).

Revision ID: 032_add_promotion_trace
Revises: 031_create_designer_revisions
Create Date: 2026-06-23

A nullable JSONB column holding the value-free ``PromotionTrace`` captured at run
completion (ordered step kinds + tool argument *shapes*, never raw values). NULL
means "not promotable" (the default for every pre-existing row and for simple
runs). Additive and backward-compatible; downgrade simply drops the column.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "032_add_promotion_trace"
down_revision: str | None = "031_create_designer_revisions"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _research_sessions_columns() -> set[str] | None:
    """Return ``research_sessions`` column names, or ``None`` if the table is absent.

    On event-sourced deployments the legacy ``public.research_sessions`` table has
    been dropped (see ``scripts/cleanup_legacy_tables.sql``); promotion traces there
    live in the storage stack (schema ``deep_research_state``), so altering the legacy
    table is a no-op. Guarding keeps this revision applying on BOTH lineages — legacy
    ORM DBs (table present) and cleaned event-sourced DBs (table absent).
    """
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table("research_sessions"):
        return None
    return {col["name"] for col in inspector.get_columns("research_sessions")}


def upgrade() -> None:
    """Add the nullable promotion_trace JSONB column (legacy ORM table only)."""
    columns = _research_sessions_columns()
    if columns is None or "promotion_trace" in columns:
        return  # legacy table absent (event-sourced) or column already present
    op.add_column(
        "research_sessions",
        sa.Column(
            "promotion_trace",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )


def downgrade() -> None:
    """Drop the promotion_trace column if present."""
    columns = _research_sessions_columns()
    if columns is None or "promotion_trace" not in columns:
        return
    op.drop_column("research_sessions", "promotion_trace")
