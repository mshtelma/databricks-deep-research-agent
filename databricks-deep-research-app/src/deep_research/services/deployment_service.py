"""DeploymentService -- CRUD + lifecycle for the ``agent_deployments`` table.

Mirrors plan Section B.3. Cursor-based pagination is opaque base64-encoded
``(updated_at_iso, id_hex)`` so list endpoints can stream stable results
across writes.

The service is intentionally thin: it owns the SQL surface, not the
deployment lifecycle. Translators (``services/deployment/``) own the
external resource lifecycle, and the API layer dispatches between them.
"""
from __future__ import annotations

import base64
import logging
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from deep_research.models.agent_deployment import (
    ACTIVE_STATUSES,
    DELETABLE_STATUSES,
    AgentDeployment,
    DeploymentMode,
    DeploymentStatus,
)
from deep_research.models.agent_v2 import AgentV2
from deep_research.services.base import BaseRepository

logger = logging.getLogger(__name__)


_DEFAULT_LIMIT = 50
_MAX_LIMIT = 200


def _encode_cursor(updated_at: datetime, deployment_id: UUID) -> str:
    """Encode ``(updated_at, id)`` into an opaque base64 token."""
    raw = f"{updated_at.isoformat()}|{deployment_id.hex}".encode()
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _decode_cursor(cursor: str) -> tuple[datetime, UUID]:
    """Inverse of ``_encode_cursor``. Raises on malformed input."""
    raw = base64.urlsafe_b64decode(cursor.encode("ascii")).decode("utf-8")
    iso, id_hex = raw.split("|", 1)
    return datetime.fromisoformat(iso), UUID(hex=id_hex)


class DeploymentService(BaseRepository[AgentDeployment]):
    """CRUD + lifecycle helpers for ``agent_deployments``."""

    model = AgentDeployment

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    async def create(
        self,
        *,
        agent_id: UUID,
        revision_id: UUID,
        mode: DeploymentMode,
        config: dict[str, Any],
        deployed_by: str,
    ) -> AgentDeployment:
        """Insert a new deployment row in ``PENDING`` status."""
        deployment = AgentDeployment(
            agent_id=agent_id,
            revision_id=revision_id,
            mode=mode.value,
            status=DeploymentStatus.PENDING.value,
            config=config,
            deployed_by=deployed_by,
        )
        return await self.add(deployment)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def get(self, deployment_id: UUID) -> AgentDeployment | None:
        """Return the deployment row or ``None``."""
        return await self._session.get(AgentDeployment, deployment_id)

    async def get_for_agent(
        self,
        agent_id: UUID,
        *,
        mode: DeploymentMode | None = None,
        status: DeploymentStatus | None = None,
    ) -> list[AgentDeployment]:
        """List all deployments for one agent, optionally filtered."""
        stmt = select(AgentDeployment).where(AgentDeployment.agent_id == agent_id)
        if mode is not None:
            stmt = stmt.where(AgentDeployment.mode == mode.value)
        if status is not None:
            stmt = stmt.where(AgentDeployment.status == status.value)
        stmt = stmt.order_by(AgentDeployment.created_at.desc())
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def list_for_user(
        self,
        user_id: str,
        *,
        mode: DeploymentMode | None = None,
        status: DeploymentStatus | None = None,
        agent_id: UUID | None = None,
        cursor: str | None = None,
        limit: int = _DEFAULT_LIMIT,
    ) -> tuple[list[AgentDeployment], str | None]:
        """Cursor-paginated list of deployments the user may manage.

        W9: returns deployments where the user is **either** the original
        deployer OR the owner of the parent agent. Previously the filter
        was strict ``deployed_by == user_id``, which left agent owners
        with no surface to manage deployments others made against their
        shared (workspace-visible) agent.

        Returns ``(items, next_cursor)``. ``next_cursor`` is ``None`` when
        the page is the last one.
        """
        bounded_limit = max(1, min(limit, _MAX_LIMIT))
        stmt = (
            select(AgentDeployment)
            .join(AgentV2, AgentV2.id == AgentDeployment.agent_id)
            .where(
                or_(
                    AgentDeployment.deployed_by == user_id,
                    AgentV2.owner_id == user_id,
                ),
            )
        )
        if agent_id is not None:
            stmt = stmt.where(AgentDeployment.agent_id == agent_id)
        if mode is not None:
            stmt = stmt.where(AgentDeployment.mode == mode.value)
        if status is not None:
            stmt = stmt.where(AgentDeployment.status == status.value)
        if cursor is not None:
            cur_updated_at, cur_id = _decode_cursor(cursor)
            stmt = stmt.where(
                (AgentDeployment.updated_at < cur_updated_at)
                | (
                    (AgentDeployment.updated_at == cur_updated_at)
                    & (AgentDeployment.id < cur_id)
                )
            )
        stmt = stmt.order_by(
            AgentDeployment.updated_at.desc(),
            AgentDeployment.id.desc(),
        ).limit(bounded_limit + 1)

        result = await self._session.execute(stmt)
        rows = list(result.scalars().all())

        if len(rows) > bounded_limit:
            tail = rows[-1]
            next_cursor = _encode_cursor(tail.updated_at, tail.id)
            rows = rows[:bounded_limit]
        else:
            next_cursor = None
        return rows, next_cursor

    async def count_active_for_agent(self, agent_id: UUID) -> int:
        """Count of deployments in PENDING/DEPLOYING/ACTIVE for an agent."""
        stmt = (
            select(func.count())
            .select_from(AgentDeployment)
            .where(AgentDeployment.agent_id == agent_id)
            .where(AgentDeployment.status.in_(ACTIVE_STATUSES))
        )
        result = await self._session.execute(stmt)
        count = result.scalar_one()
        return int(count)

    async def list_active_for_agent(
        self,
        agent_id: UUID,
    ) -> list[AgentDeployment]:
        """Return all PENDING/DEPLOYING/ACTIVE deployments for an agent."""
        stmt = (
            select(AgentDeployment)
            .where(AgentDeployment.agent_id == agent_id)
            .where(AgentDeployment.status.in_(ACTIVE_STATUSES))
            .order_by(AgentDeployment.created_at.desc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def list_terminal_for_agent(
        self,
        agent_id: UUID,
    ) -> list[AgentDeployment]:
        """Return deployments that may be physically deleted with the agent.

        Returns DEACTIVATED + CLEANUP_FAILED rows only (the
        ``DELETABLE_STATUSES`` set). FAILED rows are *terminal* for polling
        purposes but are kept until the user explicitly deactivates them so
        forensic data isn't silently erased on force-delete (W2 of the fix
        plan — preserves audit trail).
        """
        stmt = (
            select(AgentDeployment)
            .where(AgentDeployment.agent_id == agent_id)
            .where(AgentDeployment.status.in_(DELETABLE_STATUSES))
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def list_failed_for_agent(
        self,
        agent_id: UUID,
    ) -> list[AgentDeployment]:
        """Return FAILED deployments for an agent.

        FAILED rows are produced by:
          - the recovery sweep at app startup (``error_message="server_shutdown"``)
          - the zombie janitor (``error_message="worker_zombie"``)
          - genuine deploy failures from translator.deploy()

        They sit between ACTIVE_STATUSES and DELETABLE_STATUSES, blocking
        the FK on ``agent_deployments.agent_id`` (``ON DELETE RESTRICT``).
        Force-delete uses this list to flip them to DEACTIVATED before
        ``delete_terminal_rows_for_agent`` clears them.
        """
        stmt = (
            select(AgentDeployment)
            .where(AgentDeployment.agent_id == agent_id)
            .where(AgentDeployment.status == DeploymentStatus.FAILED.value)
            .order_by(AgentDeployment.created_at.desc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    async def update_status(
        self,
        deployment_id: UUID,
        status: DeploymentStatus,
        *,
        error_message: str | None = None,
        endpoint_name: str | None = None,
        model_name: str | None = None,
        external_resource_ids: dict[str, Any] | None = None,
    ) -> AgentDeployment:
        """Transition the deployment to ``status`` and persist updates.

        The ``updated_at`` timestamp is bumped automatically by the
        ``TimestampMixin`` ``onupdate`` hook.
        """
        deployment = await self._session.get(AgentDeployment, deployment_id)
        if deployment is None:
            raise ValueError(f"Deployment {deployment_id} not found")
        deployment.status = status.value
        if error_message is not None:
            deployment.error_message = error_message
        if endpoint_name is not None:
            deployment.endpoint_name = endpoint_name
        if model_name is not None:
            deployment.model_name = model_name
        if external_resource_ids is not None:
            deployment.external_resource_ids = external_resource_ids
        await self._session.flush()
        await self._session.refresh(deployment)
        return deployment

    async def deactivate(self, deployment_id: UUID) -> AgentDeployment:
        """Set status=DEACTIVATED + deactivated_at=now()."""
        deployment = await self._session.get(AgentDeployment, deployment_id)
        if deployment is None:
            raise ValueError(f"Deployment {deployment_id} not found")
        deployment.status = DeploymentStatus.DEACTIVATED.value
        deployment.deactivated_at = datetime.now(UTC)
        await self._session.flush()
        await self._session.refresh(deployment)
        return deployment

    async def mark_cleanup_failed(
        self,
        deployment_id: UUID,
        error_message: str,
    ) -> AgentDeployment:
        """Terminal-state escape after retry exhaustion (plan Section N.2)."""
        deployment = await self._session.get(AgentDeployment, deployment_id)
        if deployment is None:
            raise ValueError(f"Deployment {deployment_id} not found")
        deployment.status = DeploymentStatus.CLEANUP_FAILED.value
        deployment.error_message = error_message
        deployment.deactivated_at = datetime.now(UTC)
        await self._session.flush()
        await self._session.refresh(deployment)
        return deployment

    async def increment_cleanup_attempts(
        self,
        deployment_id: UUID,
    ) -> AgentDeployment:
        """Bump the retry counter; called by the cleanup-task on each retry."""
        deployment = await self._session.get(AgentDeployment, deployment_id)
        if deployment is None:
            raise ValueError(f"Deployment {deployment_id} not found")
        deployment.cleanup_attempts = deployment.cleanup_attempts + 1
        await self._session.flush()
        await self._session.refresh(deployment)
        return deployment

    async def reset_cleanup_attempts(
        self,
        deployment_id: UUID,
    ) -> AgentDeployment:
        """Reset cleanup_attempts to 0.

        Called when a user retries cleanup on a CLEANUP_FAILED row so the
        next translator failure starts a fresh MAX_CLEANUP_ATTEMPTS budget
        rather than immediately re-escalating.
        """
        deployment = await self._session.get(AgentDeployment, deployment_id)
        if deployment is None:
            raise ValueError(f"Deployment {deployment_id} not found")
        deployment.cleanup_attempts = 0
        await self._session.flush()
        await self._session.refresh(deployment)
        return deployment

    async def delete_terminal_rows_for_agent(
        self,
        agent_id: UUID,
        *,
        include_failed: bool = False,
    ) -> int:
        """Physically remove deletable rows for an agent.

        Plan Section N.2 step 4: PostgreSQL ``ON DELETE RESTRICT`` blocks
        the parent agent delete based on row existence, not status. Terminal
        rows must be physically removed before the agent row can be deleted.

        Default (``include_failed=False``): removes DEACTIVATED + CLEANUP_FAILED
        only, preserving FAILED rows for forensics on non-forced paths.

        ``include_failed=True``: callers from force-delete cascades have
        already flipped FAILED rows to DEACTIVATED before calling this — the
        flag widens the SELECT defensively in case any rows slipped through
        the cascade (e.g., concurrent INSERT during cascade processing).

        Returns the number of rows deleted.
        """
        rows = await self.list_terminal_for_agent(agent_id)
        if include_failed:
            rows.extend(await self.list_failed_for_agent(agent_id))
        for row in rows:
            await self._session.delete(row)
        await self._session.flush()
        return len(rows)
