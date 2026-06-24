"""Create skills + skill_revisions tables for governed skills (Feature 2.2).

Revision ID: 030_create_skills
Revises: 029_backfill_in_app_deployments
Create Date: 2026-06-22

``skills`` is the queryable store backing the framework's SkillStore protocol
(cheap metadata listing, body fetched on demand). ``skill_revisions`` is an
append-only audit log (sibling of agent_revisions) recording each upsert with
the security-scan verdict.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "030_create_skills"
down_revision: str | None = "029_backfill_in_app_deployments"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create skills and skill_revisions tables with indexes."""
    op.create_table(
        "skills",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("description", sa.String(1024), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column(
            "scripts",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("author", sa.String(255), nullable=True),
        sa.Column("security_verdict", sa.String(50), nullable=True),
        sa.Column("is_seed", sa.Integer(), nullable=False, server_default="0"),
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
        sa.PrimaryKeyConstraint("id", name="pk_skills"),
        sa.UniqueConstraint("name", name="uq_skills_name"),
    )
    op.create_index("idx_skills_name", "skills", ["name"])

    op.create_table(
        "skill_revisions",
        sa.Column(
            "rev_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("skill_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("description", sa.String(1024), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column(
            "scripts",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("security_verdict", sa.String(50), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("created_by", sa.String(255), nullable=False),
        sa.PrimaryKeyConstraint("rev_id", name="pk_skill_revisions"),
        sa.ForeignKeyConstraint(
            ["skill_id"],
            ["skills.id"],
            ondelete="CASCADE",
            name="fk_skill_revisions_skill_id_skills",
        ),
    )
    op.create_index(
        "idx_skill_revisions_skill_created",
        "skill_revisions",
        ["skill_id", "created_at"],
    )


def downgrade() -> None:
    """Drop skill_revisions and skills tables and their indexes."""
    op.drop_index(
        "idx_skill_revisions_skill_created", table_name="skill_revisions"
    )
    op.drop_table("skill_revisions")
    op.drop_index("idx_skills_name", table_name="skills")
    op.drop_table("skills")
