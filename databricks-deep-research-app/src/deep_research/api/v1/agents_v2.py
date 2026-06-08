"""Agent Designer V1 — agents_v2 CRUD endpoints with etag optimistic locking."""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response, status
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from deep_research.agent_designer.mermaid_export import serialize_to_mermaid
from deep_research.agent_designer.workflow_critic import (
    CritiqueResult,
    critique_workflow_against_intent,
)
from deep_research.agent_designer.yaml_export import serialize_to_yaml
from deep_research.core.databricks_auth import get_user_workspace_client
from deep_research.db.session import get_db
from deep_research.middleware.auth import CurrentUser
from deep_research.models.agent_deployment import MAX_CLEANUP_ATTEMPTS
from deep_research.models.agent_v2 import AgentRevision, AgentV2
from deep_research.observability.agent_designer_metrics import (
    record_etag_conflict,
    record_mermaid_export_ms,
    record_save_latency,
    record_yaml_export_ms,
)
from deep_research.schemas.agent_v2 import (
    AgentV2ListResponse,
    AgentV2Response,
    CreateAgentV2Request,
    UpdateAgentV2Request,
)
from deep_research.services.agent_v2_service import (
    ActiveDeploymentsError,
    AgentV2Service,
    EtagConflictError,
)
from deep_research.services.deployment import DeploymentCleanupError
from deep_research.services.deployment.auth import WorkspaceClientResolver
from deep_research.services.deployment_service import DeploymentService

logger = logging.getLogger(__name__)

router = APIRouter()


# ----- Save-time LLM-as-judge critic gate ------------------------------------


def _extract_intent_from_definition(definition: dict[str, Any]) -> str:
    """Pull the user's original intent out of a generated workflow definition.

    Lookup order:
    1. Any ``config.synthesis_metadata.designer_goal`` found while walking
       the AST. ``plan_and_execute`` workflows carry this on their P&E node.
    2. The top-level ``definition.description`` field, which the Designer
       builders populate with the intent for every topology. This is the
       fallback for ``parallel_lanes`` and ``single_agent`` workflows that
       cannot carry ``synthesis_metadata`` (``AgentNodeConfig`` rejects it).

    Returns an empty string when neither source has content (legacy agents
    that pre-date both mechanisms).
    """
    if not isinstance(definition, dict):
        return ""

    def walk(node: Any) -> str:
        if not isinstance(node, dict):
            return ""
        config = node.get("config") or {}
        if isinstance(config, dict):
            meta = config.get("synthesis_metadata") or {}
            if isinstance(meta, dict):
                goal = meta.get("designer_goal")
                if isinstance(goal, str) and goal.strip():
                    return goal.strip()
            body = config.get("body")
            if isinstance(body, dict):
                hit = walk(body)
                if hit:
                    return hit
        for child in node.get("children", []) or []:
            hit = walk(child)
            if hit:
                return hit
        return ""

    found = walk(definition.get("root"))
    if found:
        return found
    # Fallback for parallel_lanes / single_agent topologies — the user's
    # intent is on the workflow's top-level description.
    description = definition.get("description")
    if isinstance(description, str) and description.strip():
        return description.strip()
    return ""


async def _run_save_critic_gate(
    definition: dict[str, Any],
    llm_client: Any,
    *,
    force: bool,
) -> tuple[CritiqueResult | None, list[str]]:
    """Run the workflow critic at save time.

    Returns (critique, warnings). When ``critique.verdict == "fail"`` AND
    ``force is False``, the caller raises HTTP 422 to block the save.
    ``needs_revision`` is surfaced as a warning header but does not block.
    Returns ``(None, [])`` when the gate is skipped — e.g., the definition
    carries no intent (legacy agent), no LLM client is wired, or the import
    fails — so legacy/headless paths continue to work unchanged.
    """
    intent = _extract_intent_from_definition(definition)
    if not intent:
        # No intent to judge against — legacy agents and bare configs pass.
        return None, []
    if llm_client is None:
        logger.warning("save-time critic skipped: no llm_client available")
        return None, []
    try:
        from deep_research.agent.adapters.llm_adapter import AppLLMAdapter

        adapter = AppLLMAdapter(llm_client)
    except Exception as exc:  # noqa: BLE001 — fail open, never block save
        logger.warning("save-time critic adapter init failed: %s", exc, exc_info=True)
        return None, []
    try:
        critique = await critique_workflow_against_intent(
            definition=definition,
            intent=intent,
            required_outputs=None,
            llm=adapter,
        )
    except Exception as exc:  # noqa: BLE001 — fail open, never block save
        logger.warning("save-time critic call failed: %s", exc, exc_info=True)
        return None, []

    warnings: list[str] = []
    if critique.verdict == "needs_revision":
        warnings.append(
            f"critic verdict=needs_revision: {critique.summary}"
        )
    elif critique.verdict == "fail" and force:
        warnings.append(
            f"critic verdict=fail (overridden by force=true): {critique.summary}"
        )
    return critique, warnings


def _raise_if_critic_blocks(
    critique: CritiqueResult | None,
    *,
    force: bool,
) -> None:
    """Raise HTTP 422 when ``verdict == "fail"`` and ``force`` is not set."""
    if critique is None or force:
        return
    if critique.verdict == "fail":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": (
                    "Workflow critic verdict=fail — the workflow does not "
                    "answer the user's request. Pass ?force=true to save "
                    "anyway, or revise the workflow first."
                ),
                "critique": critique.model_dump(),
            },
        )


class RevisionResponse(BaseModel):
    """Single revision snapshot — definition payload omitted for list view."""

    rev_id: UUID
    etag: str
    created_at: datetime
    created_by: str

    model_config = {"from_attributes": True}


class RevisionListResponse(BaseModel):
    """Paginated list of agent revisions."""

    items: list[RevisionResponse]
    total: int


def _to_response(agent: AgentV2) -> AgentV2Response:
    return AgentV2Response.model_validate(agent)


@router.post("", response_model=AgentV2Response, status_code=status.HTTP_201_CREATED)
async def create_agent(
    request: CreateAgentV2Request,
    response: Response,
    user: CurrentUser,
    fastapi_request: Request,
    force: bool = Query(False, description="Bypass the LLM critic verdict=fail gate."),
    session: AsyncSession = Depends(get_db),
) -> AgentV2Response:
    # Save-time critic gate: block on verdict=fail unless force=true.
    # Reads the AST definition from the request payload; uses the app's LLM
    # client. Critique failures are logged and treated as "skip gate" so
    # save reliability is never coupled to LLM availability.
    llm_client = getattr(fastapi_request.app.state, "llm_client", None)
    critique, critic_warnings = await _run_save_critic_gate(
        request.definition or {},
        llm_client,
        force=force,
    )
    _raise_if_critic_blocks(critique, force=force)

    service = AgentV2Service(session)
    _t0 = time.monotonic()
    agent = await service.create(owner_id=user.user_id, request=request)
    await session.commit()
    record_save_latency("create", (time.monotonic() - _t0) * 1000)
    response.headers["ETag"] = agent.etag
    if critic_warnings:
        # Surface needs_revision / force-override warnings to the UI so the
        # user knows the agent saved but the critic flagged issues.
        response.headers["X-Critic-Warning"] = "; ".join(critic_warnings)
    await service._write_revision_best_effort(agent, agent.etag, user.user_id)
    return _to_response(agent)


@router.get("/{agent_id}", response_model=AgentV2Response)
async def get_agent(
    agent_id: UUID,
    response: Response,
    user: CurrentUser,
    session: AsyncSession = Depends(get_db),
) -> AgentV2Response:
    service = AgentV2Service(session)
    agent = await service.get_for_user(agent_id, user.user_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    response.headers["ETag"] = agent.etag
    return _to_response(agent)


@router.patch("/{agent_id}", response_model=AgentV2Response)
async def update_agent(
    agent_id: UUID,
    request: UpdateAgentV2Request,
    response: Response,
    user: CurrentUser,
    fastapi_request: Request,
    if_match: str | None = Header(default=None, alias="If-Match"),
    force: bool = Query(False, description="Bypass the LLM critic verdict=fail gate."),
    session: AsyncSession = Depends(get_db),
) -> AgentV2Response:
    if not if_match:
        raise HTTPException(
            status_code=428,
            detail="If-Match header required for PATCH",
        )
    # Save-time critic gate: only runs when the PATCH carries a definition
    # payload (UpdateAgentV2Request allows partial updates; name/description
    # changes without a new definition skip the gate).
    critique = None
    critic_warnings: list[str] = []
    if request.definition is not None:
        llm_client = getattr(fastapi_request.app.state, "llm_client", None)
        critique, critic_warnings = await _run_save_critic_gate(
            request.definition,
            llm_client,
            force=force,
        )
        _raise_if_critic_blocks(critique, force=force)

    service = AgentV2Service(session)
    _t0 = time.monotonic()
    try:
        agent = await service.update(agent_id, user.user_id, request, if_match)
    except EtagConflictError as exc:
        record_etag_conflict()
        raise HTTPException(
            status_code=409,
            detail={"message": "Etag conflict", "current_etag": exc.actual},
        ) from exc
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    await session.commit()
    record_save_latency("update", (time.monotonic() - _t0) * 1000)
    response.headers["ETag"] = agent.etag
    if critic_warnings:
        response.headers["X-Critic-Warning"] = "; ".join(critic_warnings)
    await service._write_revision_best_effort(agent, agent.etag, user.user_id)
    return _to_response(agent)


@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(
    agent_id: UUID,
    fastapi_request: Request,
    user: CurrentUser,
    force: bool = False,
    session: AsyncSession = Depends(get_db),
) -> Response:
    """Delete an owned agent (plan Section N).

    Default returns HTTP 409 when active deployments exist; ``?force=true``
    triggers synchronous deactivation + terminal-row cleanup before delete.

    When the caller carries an OBO token (``x-forwarded-access-token``),
    builds a user-scoped ``WorkspaceClient`` and threads it into the
    deactivate cascade via ``WorkspaceClientResolver`` — so Shell-App /
    MLflow deactivate calls run as the identity that originally created
    those resources. Falls back to the parent-app SP when no token is
    present (local dev / curl without OBO).
    """
    service = AgentV2Service(session)
    # Build the resolver from the request's OBO token when present so the
    # deactivate cascade runs as the user identity that originally created
    # the external resources. ``get_user_workspace_client`` raises 401 in
    # Databricks Apps if the header is absent — we only invoke it when the
    # header is present, so local-dev / curl-without-OBO continue to work
    # (resolver falls back to SP internally).
    obo_client = (
        get_user_workspace_client(fastapi_request)
        if fastapi_request.headers.get("X-Forwarded-Access-Token")
        else None
    )
    resolver = WorkspaceClientResolver(obo_client=obo_client)
    try:
        deleted = await service.delete(
            agent_id,
            user.user_id,
            force=force,
            client_resolver=resolver,
        )
    except ActiveDeploymentsError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "error_kind": "active_deployments_exist",
                "active_count": exc.active_count,
                "deployments": exc.deployments,
                "message": (
                    "Deactivate all deployments before deleting, "
                    "or use ?force=true"
                ),
            },
        ) from exc
    except DeploymentCleanupError as exc:
        # W9: translator cleanup failed during force-delete. Commit the
        # partial state (attempts counter / cleanup_failed transition the
        # service applied before re-raising) so the next retry sees the
        # bumped count.
        await session.commit()
        raise HTTPException(
            status_code=409,
            detail={
                "error_kind": "deployment_cleanup_failed",
                "message": (
                    "Translator-driven cleanup failed during force-delete. "
                    f"Detail: {exc}. Retry the request; after "
                    f"{MAX_CLEANUP_ATTEMPTS} attempts the deployment "
                    "transitions to 'cleanup_failed' (terminal) and you "
                    "can re-issue force=true to skip the remaining "
                    "external cleanup."
                ),
                "max_attempts": MAX_CLEANUP_ATTEMPTS,
            },
        ) from exc
    except IntegrityError as exc:
        # Defense in depth: the force-delete cascade should clear every
        # blocking row, but a future status set could slip through. Surface
        # the surviving rows instead of letting FastAPI emit an opaque 500.
        await session.rollback()
        deployment_service = DeploymentService(session)
        blockers = await deployment_service.get_for_agent(agent_id)
        logger.warning(
            "AGENT_DELETE_FK_BLOCKED agent_id=%s blocking_count=%s exc=%s",
            agent_id,
            len(blockers),
            exc.orig,
        )
        raise HTTPException(
            status_code=409,
            detail={
                "error_kind": "deployment_rows_block_delete",
                "message": (
                    "Cannot delete agent: residual deployment rows block "
                    "the foreign-key constraint. Resolve each row via "
                    "DELETE /deployments/{id} before retrying."
                ),
                "blocking_deployments": [
                    {
                        "id": str(d.id),
                        "mode": d.mode,
                        "status": d.status,
                        "endpoint_name": d.endpoint_name,
                    }
                    for d in blockers
                ],
            },
        ) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Agent not found")
    await session.commit()
    return Response(status_code=204)


@router.get("", response_model=AgentV2ListResponse)
async def list_agents(
    user: CurrentUser,
    session: AsyncSession = Depends(get_db),
) -> AgentV2ListResponse:
    service = AgentV2Service(session)
    items = await service.list_for_user(user.user_id)
    return AgentV2ListResponse(items=items, total=len(items))


@router.get("/{agent_id}/mermaid", response_class=PlainTextResponse)
async def export_mermaid(
    agent_id: UUID,
    user: CurrentUser,
    session: AsyncSession = Depends(get_db),
) -> PlainTextResponse:
    """Export agent WorkflowDefinition AST as a Mermaid flowchart document.

    Returns Content-Type: text/plain.  Loops are projected acyclically with a
    ``↻ repeat`` annotation; conditional branches converge to a synthetic merge
    node.  The caller must own or have visibility of the agent — otherwise 404
    is returned (same scoping as GET /{agent_id}).
    """
    service = AgentV2Service(session)
    agent = await service.get_for_user(agent_id, user.user_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="agent not found")
    _t0 = time.monotonic()
    try:
        body = serialize_to_mermaid(
            agent.definition,
            agent_name=agent.name,
            agent_id=str(agent.id),
        )
    finally:
        record_mermaid_export_ms((time.monotonic() - _t0) * 1000)
    return PlainTextResponse(content=body, media_type="text/plain")


@router.get("/{agent_id}/yaml", response_class=PlainTextResponse)
async def export_yaml(
    agent_id: UUID,
    user: CurrentUser,
    session: AsyncSession = Depends(get_db),
) -> PlainTextResponse:
    """Export agent WorkflowDefinition AST as a deterministic YAML document.

    Returns Content-Type: text/yaml with ``registry_version`` pinned at the
    top of the document.  The caller must own or have visibility of the agent
    — otherwise 404 is returned (same scoping as GET /{agent_id}).
    """
    service = AgentV2Service(session)
    agent = await service.get_for_user(agent_id, user.user_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="agent not found")
    _t0 = time.monotonic()
    try:
        body = serialize_to_yaml(agent.definition)
    finally:
        record_yaml_export_ms((time.monotonic() - _t0) * 1000)
    return PlainTextResponse(content=body, media_type="text/yaml")


@router.get("/{agent_id}/revisions", response_model=RevisionListResponse)
async def list_revisions(
    agent_id: UUID,
    user: CurrentUser,
    limit: int = 20,
    offset: int = 0,
    session: AsyncSession = Depends(get_db),
) -> RevisionListResponse:
    """List revision snapshots for an agent, newest first.

    Returns the ``rev_id``, ``etag``, ``created_at``, and ``created_by`` for
    each revision.  The full ``definition`` payload is omitted from list view
    — use GET /{agent_id}/revisions/{rev_id} to retrieve it.

    Owner check: same visibility rules as GET /{agent_id}.
    """
    service = AgentV2Service(session)
    agent = await service.get_for_user(agent_id, user.user_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="agent not found")
    result = await session.execute(
        select(AgentRevision)
        .where(AgentRevision.agent_id == agent_id)
        .order_by(AgentRevision.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    items = [
        RevisionResponse(
            rev_id=r.rev_id,
            etag=r.etag,
            created_at=r.created_at,
            created_by=r.created_by,
        )
        for r in result.scalars()
    ]
    count_result = await session.execute(
        select(func.count()).select_from(AgentRevision).where(AgentRevision.agent_id == agent_id)
    )
    total = count_result.scalar_one()
    return RevisionListResponse(items=items, total=total)


@router.get("/{agent_id}/revisions/{rev_id}")
async def get_revision(
    agent_id: UUID,
    rev_id: UUID,
    user: CurrentUser,
    session: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    """Return a single revision snapshot including the full ``definition`` AST.

    The caller must own or have visibility of the parent agent (same rules as
    GET /{agent_id}).  Returns 404 if the agent or revision is not found, or
    if the revision does not belong to the specified agent.
    """
    service = AgentV2Service(session)
    agent = await service.get_for_user(agent_id, user.user_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="agent not found")
    result = await session.execute(
        select(AgentRevision).where(
            AgentRevision.rev_id == rev_id,
            AgentRevision.agent_id == agent_id,
        )
    )
    revision = result.scalar_one_or_none()
    if revision is None:
        raise HTTPException(status_code=404, detail="revision not found")
    return {
        "rev_id": revision.rev_id,
        "etag": revision.etag,
        "definition": revision.definition,
        "created_at": revision.created_at,
        "created_by": revision.created_by,
    }
