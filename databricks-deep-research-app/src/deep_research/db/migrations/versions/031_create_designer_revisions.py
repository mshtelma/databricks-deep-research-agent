"""Create designer_revisions append-only audit log (spec §5.6 / §4.4).

Revision ID: 031_create_designer_revisions
Revises: 030_create_skills
Create Date: 2026-06-23

``designer_revisions`` is the append-only governance trail for Designer-authored
changes: one immutable row per change = {prev, new, security-scan verdict,
timestamp, actor}. Subject-agnostic — identified by a (subject_type,
subject_ref) pair so a prompt, an agent definition, or a skill body can all be
recorded uniformly. Sibling of agent_revisions (025) and skill_revisions (030);
no typed FK because the subject kind is heterogeneous.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "031_create_designer_revisions"
down_revision: str | None = "030_create_skills"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the append-only designer_revisions table with a subject index."""
    op.create_table(
        "designer_revisions",
        sa.Column(
            "rev_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("subject_type", sa.String(32), nullable=False),
        sa.Column("subject_ref", sa.String(255), nullable=False),
        sa.Column(
            "prev_snapshot",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "new_snapshot",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("security_verdict", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("created_by", sa.String(255), nullable=False),
        sa.PrimaryKeyConstraint("rev_id", name="pk_designer_revisions"),
    )
    op.create_index(
        "idx_designer_revisions_subject_created",
        "designer_revisions",
        ["subject_type", "subject_ref", "created_at"],
    )


def downgrade() -> None:
    """Drop the designer_revisions table and its index."""
    op.drop_index(
        "idx_designer_revisions_subject_created",
        table_name="designer_revisions",
    )
    op.drop_table("designer_revisions")
