"""Backfill in_app deployments for legacy visibility='workspace' agents.

Phase 2 of the v3 rewire: the chat picker now reads agent_deployments
(mode=in_app, status=active) instead of agents_v2.visibility. Existing
agents that were marked 'workspace' but never had an explicit in_app
deployment get a synthetic ACTIVE row so they remain visible in chat.

Revision ID: 029_backfill_in_app_deployments
Revises: 028_deployment_runtime_columns
Create Date: 2026-05-12
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "029_backfill_in_app_deployments"
down_revision: str | None = "028_deployment_runtime_columns"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """For each agents_v2 row with visibility='workspace' AND no active
    in_app deployment, insert a synthetic agent_deployments row that mirrors
    the picker's "this agent is selectable" contract.

    Uses the latest revision per agent (the visibility flag was tied to the
    head revision implicitly). If an agent has zero revisions, it is skipped —
    there is nothing to surface anyway.
    """
    op.execute(
        sa.text(
            """
            INSERT INTO agent_deployments (
                id, agent_id, revision_id, mode, status, config,
                deployed_by, external_resource_ids, created_at, updated_at
            )
            SELECT
                gen_random_uuid(),
                a.id,
                (
                    SELECT r.rev_id
                    FROM agent_revisions r
                    WHERE r.agent_id = a.id
                    ORDER BY r.created_at DESC
                    LIMIT 1
                ) AS revision_id,
                'in_app',
                'active',
                '{"mode":"in_app"}'::jsonb,
                a.owner_id,
                '{}'::jsonb,
                now(),
                now()
            FROM agents_v2 a
            WHERE a.visibility = 'workspace'
              AND EXISTS (SELECT 1 FROM agent_revisions r WHERE r.agent_id = a.id)
              AND NOT EXISTS (
                  SELECT 1 FROM agent_deployments d
                  WHERE d.agent_id = a.id
                    AND d.mode = 'in_app'
                    AND d.status = 'active'
              );
            """
        )
    )


def downgrade() -> None:
    """Reverse the backfill: delete synthetic rows whose external_resource_ids
    is an empty JSONB object (the marker used at insert time).

    Note: this also catches any rows that genuinely have empty
    external_resource_ids — that is acceptable since the in_app mode never
    populates them (in_app is a no-op deploy by design).
    """
    op.execute(
        sa.text(
            """
            DELETE FROM agent_deployments
            WHERE mode = 'in_app'
              AND status = 'active'
              AND external_resource_ids = '{}'::jsonb
              AND config = '{"mode":"in_app"}'::jsonb;
            """
        )
    )
