"""AgentV2Service — CRUD with etag-based optimistic locking.

Etag computation:  sha1((sorted_json(definition) + updated_at_iso).encode()).hexdigest()
- Recomputed on every successful write
- Returned on responses; clients pass it back via If-Match on PATCH
- update() with stale If-Match raises EtagConflictError

Every successful create/update also writes an AgentRevision snapshot
best-effort (failure is logged + metered but does NOT propagate).
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import exists, or_, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from deep_research.agent_designer.ast_normalizer import (
    apply_web_search_provider_defaults,
)
from deep_research.agent_designer.catalog_service import CatalogService
from deep_research.agent_designer.workflow_validation import WorkflowValidationResult
from deep_research.models.agent_deployment import (
    MAX_CLEANUP_ATTEMPTS,
    AgentDeployment,
    DeploymentMode,
    DeploymentStatus,
)
from deep_research.models.agent_v2 import AgentRevision, AgentV2
from deep_research.models.visibility import AgentVisibility
from deep_research.observability.agent_designer_metrics import record_revision_write_failed
from deep_research.schemas.agent_v2 import (
    AgentV2Summary,
    CreateAgentV2Request,
    UpdateAgentV2Request,
)
from deep_research.services.deployment import (
    DeploymentCleanupError,
    DeploymentCleanupExhaustedError,
    translator_for,
)
from deep_research.services.deployment.auth import WorkspaceClientResolver
from deep_research.services.deployment_service import DeploymentService

logger = logging.getLogger(__name__)


class EtagConflictError(Exception):
    """Raised by AgentV2Service.update() when the supplied If-Match etag
    does not match the agent's current etag (stale client state)."""

    def __init__(self, expected: str, actual: str) -> None:
        super().__init__(f"Etag conflict: client sent {expected!r}, current is {actual!r}")
        self.expected = expected
        self.actual = actual


class ActiveDeploymentsError(Exception):
    """Raised by AgentV2Service.delete() when active deployments block delete.

    Plan Section N.1: the API layer translates this into HTTP 409 with a
    structured body listing the blocking deployments. Force-delete (``?force=true``)
    bypasses by transitioning + physically removing the deployment rows.
    """

    def __init__(
        self,
        active_count: int,
        deployments: list[dict[str, Any]],
    ) -> None:
        super().__init__(
            f"Cannot delete agent: {active_count} active deployment(s) exist"
        )
        self.active_count = active_count
        self.deployments = deployments


def _compute_etag(definition: dict[str, Any], updated_at: datetime) -> str:
    payload = json.dumps(definition, sort_keys=True, default=str) + updated_at.isoformat()
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _node_count(node: dict[str, Any]) -> int:
    """Recursively count nodes in a node subtree (the node itself + all descendants)."""
    count = 1
    for child in node.get("children", []) or []:
        count += _node_count(child)
    config = node.get("config", {})
    if isinstance(config, dict):
        body = config.get("body")
        if isinstance(body, dict):
            count += _node_count(body)
        evaluator = config.get("evaluator")
        if isinstance(evaluator, dict):
            count += _node_count(evaluator)
    return count


def _find_synth_grounding_mode(node: dict[str, Any]) -> str | None:
    """Return the grounding_mode of the first synthesizer node in the subtree.

    Walks the same node shape as ``_node_count`` (children + ``config.body`` /
    ``config.evaluator``). ``subtype`` and ``grounding_mode`` live inside
    ``node["config"]``. Returns ``None`` if no synthesizer is present; returns
    ``"reclaim"`` for a synthesizer that omits ``grounding_mode`` (the designer's
    baked default / safe floor).
    """
    config = node.get("config", {})
    if isinstance(config, dict):
        if config.get("subtype") == "synthesizer":
            grounding = config.get("grounding_mode")
            return grounding if isinstance(grounding, str) else "reclaim"
        body = config.get("body")
        if isinstance(body, dict):
            found = _find_synth_grounding_mode(body)
            if found is not None:
                return found
        evaluator = config.get("evaluator")
        if isinstance(evaluator, dict):
            found = _find_synth_grounding_mode(evaluator)
            if found is not None:
                return found
    for child in node.get("children", []) or []:
        found = _find_synth_grounding_mode(child)
        if found is not None:
            return found
    return None


def _default_verify_sources(definition: dict[str, Any]) -> bool:
    """Derive an agent's default 'verify sources' from its saved synthesizer.

    ``reclaim`` -> ``True`` (cite + NLI verify); ``classical_lite``/``none`` ->
    ``False`` (cite-only). No synthesizer found -> ``True`` (the reclaim safe
    floor). The chat composer seeds the verify toggle from this; the toggle then
    fully overrides per run (see ``framework_orchestrator``).
    """
    grounding = _find_synth_grounding_mode(definition.get("root", {}))
    if grounding is None:
        return True
    return grounding == "reclaim"


class AgentV2Service:
    """CRUD service for AgentV2 with optimistic-locking semantics."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def _write_revision_best_effort(
        self,
        agent: AgentV2,
        etag: str,
        created_by: str,
        *,
        validation: WorkflowValidationResult | None = None,
    ) -> None:
        """Write an AgentRevision snapshot after a successful primary write.

        Best-effort: any SQLAlchemyError is caught, logged, and metered but
        NOT propagated — the primary create/update has already committed. The
        ``validation`` snapshot (if any) is stored so each revision retains the
        verdict it had when saved.
        """
        try:
            rev = AgentRevision(
                rev_id=uuid4(),
                agent_id=agent.id,
                etag=etag,
                definition=agent.definition,
                created_by=created_by,
                validation=(
                    validation.model_dump(mode="json")
                    if validation is not None
                    else None
                ),
            )
            self._session.add(rev)
            await self._session.commit()
        except SQLAlchemyError as exc:
            logger.warning(
                "agent_designer.revision_write_failed agent_id=%s error=%s",
                str(agent.id),
                str(exc),
                extra={"agent_id": str(agent.id), "error": str(exc)},
            )
            record_revision_write_failed()
            # Do NOT propagate — primary write already committed.

    async def create(self, owner_id: str, request: CreateAgentV2Request) -> AgentV2:
        now = datetime.now(UTC)
        definition = CatalogService().materialize_for_save(request.definition)
        # Fill databricks web-search defaults (e.g. the serving endpoint) so an agent
        # saved via the UI inspector — which bypasses the designer normalizer —
        # persists a self-describing tool, not a provider:databricks tool missing
        # `model` that fails at tool construction.
        apply_web_search_provider_defaults(definition)
        etag = _compute_etag(definition, now)
        agent = AgentV2(
            id=uuid4(),
            owner_id=owner_id,
            name=request.name,
            description=request.description,
            avatar_url=request.avatar_url,
            visibility=request.visibility,
            definition=definition,
            schema_version=1,
            etag=etag,
            created_at=now,
            updated_at=now,
        )
        self._session.add(agent)
        await self._session.flush()
        return agent

    async def get_for_user(self, agent_id: UUID, user_id: str) -> AgentV2 | None:
        """Returns the agent only if visible to the user (owner | workspace | system)."""
        stmt = select(AgentV2).where(AgentV2.id == agent_id)
        result = await self._session.execute(stmt)
        agent = result.scalar_one_or_none()
        if agent is None:
            return None
        # Visibility check
        if agent.owner_id == user_id:
            return agent
        if agent.visibility in (AgentVisibility.WORKSPACE.value, AgentVisibility.SYSTEM.value):
            return agent
        return None

    async def get_owned(self, agent_id: UUID, user_id: str) -> AgentV2 | None:
        """Returns the agent only if owned by the user (for mutations)."""
        stmt = select(AgentV2).where(AgentV2.id == agent_id, AgentV2.owner_id == user_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def update(
        self,
        agent_id: UUID,
        user_id: str,
        request: UpdateAgentV2Request,
        if_match_etag: str,
    ) -> AgentV2 | None:
        agent = await self.get_owned(agent_id, user_id)
        if agent is None:
            return None
        if agent.etag != if_match_etag:
            raise EtagConflictError(expected=if_match_etag, actual=agent.etag)

        if request.name is not None:
            agent.name = request.name
        if request.description is not None:
            agent.description = request.description
        if request.avatar_url is not None:
            agent.avatar_url = request.avatar_url
        if request.visibility is not None:
            agent.visibility = request.visibility
        if request.definition is not None:
            definition = CatalogService().materialize_for_save(
                request.definition,
                previous=agent.definition,
            )
            # Fill databricks web-search defaults so a UI save (which bypasses the
            # designer normalizer) persists a self-describing tool; reassign after the
            # in-place fill so the JSON column is marked dirty for the flush.
            apply_web_search_provider_defaults(definition)
            agent.definition = definition

        agent.updated_at = datetime.now(UTC)
        agent.etag = _compute_etag(agent.definition, agent.updated_at)
        await self._session.flush()
        return agent

    async def delete(
        self,
        agent_id: UUID,
        user_id: str,
        *,
        force: bool = False,
        client_resolver: WorkspaceClientResolver | None = None,
    ) -> bool:
        """Delete an owned agent. Plan Section N (deletion guard + lifecycle).

        Default (``force=False``): raises ``ActiveDeploymentsError`` if any
        deployment in PENDING/DEPLOYING/ACTIVE references this agent. The
        API layer translates this into HTTP 409.

        Force (``force=True``): synchronously marks all active deployments
        DEACTIVATED, then physically deletes terminal-status deployment rows
        (PostgreSQL ``ON DELETE RESTRICT`` on the ``agent_deployments.agent_id``
        FK requires this -- the constraint checks row existence, not status),
        then deletes the agent (revision history cascades per migration 025).

        W9 contract change: ``force=True`` now cascades through the
        per-mode translator before flipping the DB row to DEACTIVATED.
        Previously it skipped translator cleanup for Modes 2/3/4 and left
        external resources (UC models, Apps, serving endpoints) leaking.
        If any translator raises ``DeploymentCleanupError``, the cascade
        either retries (under ``MAX_CLEANUP_ATTEMPTS``) or marks the row
        ``cleanup_failed`` and re-raises so the API layer surfaces a 409.

        ``client_resolver`` lets user-initiated paths (the agent DELETE
        endpoint) thread an OBO-scoped ``WorkspaceClient`` into the
        translator. Required for Shell-App deactivate because the App
        was created by the user via OBO at deploy time; the SP doesn't
        own user-created Apps. ``None`` falls back to SP — used by
        background callers / tests with no request context.
        """
        agent = await self.get_owned(agent_id, user_id)
        if agent is None:
            return False

        deployment_service = DeploymentService(self._session)
        active = await deployment_service.list_active_for_agent(agent_id)
        if active and not force:
            raise ActiveDeploymentsError(
                active_count=len(active),
                deployments=[
                    {
                        "id": str(d.id),
                        "mode": d.mode,
                        "status": d.status,
                        "endpoint_name": d.endpoint_name,
                    }
                    for d in active
                ],
            )

        for deployment in active:
            # Translator-driven cleanup first (W9). 404/NotFound from the
            # upstream SDK is handled inside the translator as idempotent
            # success; only real failures escape as DeploymentCleanupError.
            try:
                translator = translator_for(DeploymentMode(deployment.mode))
                await translator.deactivate(
                    deployment, client_resolver=client_resolver
                )
            except DeploymentCleanupExhaustedError as exc:
                # Deterministic failure (e.g., dual-identity PermissionDenied
                # on workspace.delete). Bumping cleanup_attempts and asking
                # the user to click 3 times would just burn 3 identical
                # failures. Mark cleanup_failed once and continue the loop —
                # the row stays in DELETABLE_STATUSES so the FK-clearing
                # pass below physically removes it, and the parent agent
                # delete proceeds. The structured ERROR log line from the
                # translator + the persisted error_message form the audit
                # trail.
                logger.error(
                    "AGENT_DELETE_CLEANUP_EXHAUSTED "
                    "agent_id=%s deployment_id=%s exc=%s",
                    agent_id,
                    deployment.id,
                    exc,
                )
                await deployment_service.mark_cleanup_failed(
                    deployment.id, error_message=str(exc)
                )
                continue
            except DeploymentCleanupError as exc:
                # Transient/retryable failure: 3-strike pattern.
                # Bump attempts; if we hit the threshold, mark
                # cleanup_failed (terminal) and re-raise so the API can
                # 409. Otherwise also re-raise — force-delete is atomic
                # from the user's perspective; partial success is
                # confusing. Subsequent DELETE calls retry.
                attempts = deployment.cleanup_attempts + 1
                if attempts >= MAX_CLEANUP_ATTEMPTS:
                    await deployment_service.mark_cleanup_failed(
                        deployment.id, error_message=str(exc)
                    )
                else:
                    await deployment_service.increment_cleanup_attempts(
                        deployment.id
                    )
                raise
            # Translator success → mark DB row as deactivated.
            await deployment_service.deactivate(deployment.id)

        # Cascade through FAILED rows so they don't block the FK below.
        # FAILED rows are produced by the recovery sweep (app restart,
        # error_message="server_shutdown"), the zombie janitor
        # (error_message="worker_zombie"), or a genuine translator.deploy()
        # failure. We do NOT call translator.deactivate on them: either no
        # external resources were ever provisioned (DEPLOYING was
        # interrupted before deploy ran), or the translator already failed
        # its pass once and re-running it would just regress the 3-strike
        # attempt counter. Mirror of api/v1/deployments.py:859-860.
        #
        # Gated on ``force=True`` so the forensics-on-default semantics are
        # preserved: a non-forced delete with leftover FAILED rows still
        # raises IntegrityError (translated to 409 by the API handler),
        # surfacing the rows to the user instead of silently erasing them.
        if force:
            failed = await deployment_service.list_failed_for_agent(agent_id)
            for deployment in failed:
                await deployment_service.deactivate(deployment.id)

        # PG ON DELETE RESTRICT checks row existence, not status. Deletable
        # rows (DEACTIVATED + CLEANUP_FAILED) and any residual FAILED rows
        # (defensive — should already be DEACTIVATED above under force=True)
        # are physically removed before self._session.delete(agent).
        await deployment_service.delete_terminal_rows_for_agent(
            agent_id, include_failed=force
        )

        await self._session.delete(agent)
        await self._session.flush()
        return True

    async def update_visibility(
        self,
        agent_id: UUID,
        visibility: str,
    ) -> None:
        """Set agent.visibility in-place and flush (no etag bump, no revision).

        Used exclusively by the D2 visibility shim in ``api/v1/deployments.py``
        to keep the chat picker working while the proper D2 rewire (reading
        from ``agent_deployments`` directly) is pending.  The caller is
        responsible for committing the session after all related changes.
        """
        agent = await self._session.get(AgentV2, agent_id)
        if agent is None:
            return
        agent.visibility = visibility
        await self._session.flush()

    async def list_for_user(self, user_id: str) -> list[AgentV2Summary]:
        """Returns owner's agents + workspace-visible + system.

        The ``in_app_active`` field is computed via an EXISTS subquery against
        ``agent_deployments`` so the chat picker can read deployment state
        directly instead of relying on ``visibility='workspace'``.
        """
        in_app_active_subq = (
            select(1).where(
                AgentDeployment.agent_id == AgentV2.id,
                AgentDeployment.mode == DeploymentMode.IN_APP.value,
                AgentDeployment.status == DeploymentStatus.ACTIVE.value,
            )
        )
        stmt = (
            select(AgentV2, exists(in_app_active_subq).label("in_app_active"))
            .where(
                or_(
                    AgentV2.owner_id == user_id,
                    AgentV2.visibility == AgentVisibility.WORKSPACE.value,
                    AgentV2.visibility == AgentVisibility.SYSTEM.value,
                ),
            )
            .order_by(AgentV2.updated_at.desc())
        )
        rows = (await self._session.execute(stmt)).all()
        return [
            AgentV2Summary(
                id=agent.id,
                name=agent.name,
                description=agent.description,
                visibility=AgentVisibility(agent.visibility),
                owner_id=agent.owner_id,
                updated_at=agent.updated_at,
                node_count=_node_count(agent.definition.get("root", {})),
                in_app_active=bool(in_app_active),
                default_verify_sources=_default_verify_sources(agent.definition),
            )
            for agent, in_app_active in rows
        ]
