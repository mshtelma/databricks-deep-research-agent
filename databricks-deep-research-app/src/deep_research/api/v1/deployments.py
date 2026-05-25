"""REST API for the Agent Designer Deployment Feature (Phase 1).

Endpoints:
  POST   /api/v1/deployments                                    create
  GET    /api/v1/deployments                                    list (cursor paginated)
  GET    /api/v1/deployments/{deployment_id}                    detail
  DELETE /api/v1/deployments/{deployment_id}                    deactivate
  GET    /api/v1/deployments/{deployment_id}/status             lightweight status poll
  POST   /api/v1/deployments/{deployment_id}/actions/deploy-here  start OBO deploy
  GET    /api/v1/deployments/can-run/fast/{agent_id}            visibility-only probe
  GET    /api/v1/deployments/can-run/slow/{agent_id}            UC probe (Phase-3 stub)

The SSE stream endpoint and async cleanup lifecycle are explicit follow-ups
(plan Section B.8). Phase 1 ships synchronous deactivate via DELETE.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Annotated, cast
from uuid import UUID, uuid4

from databricks.sdk import WorkspaceClient
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    Response,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from deep_research.agent_designer.deployability import (
    classify_revision_deployability,
    default_revision_not_deployable_detail,
)
from deep_research.core.databricks_auth import get_user_workspace_client
from deep_research.db.session import get_db
from deep_research.middleware.auth import CurrentUser
from deep_research.models.agent_deployment import (
    MAX_CLEANUP_ATTEMPTS,
    AgentDeployment,
    DeploymentMode,
    DeploymentStatus,
)
from deep_research.models.agent_v2 import AgentRevision
from deep_research.schemas.deployment import (
    CanDeployHereResponse,
    CanRunFastResponse,
    CanRunSlowResponse,
    CreateDeploymentRequest,
    DeploymentListResponse,
    DeploymentResponse,
    DeploymentStatusResponse,
)
from deep_research.services.agent_v2_service import AgentV2Service
from deep_research.services.deployment.auth import WorkspaceClientResolver
from deep_research.services.deployment.batch import BatchTranslator
from deep_research.services.deployment.capability_probe import (
    _classify_probe_error,
    get_default_cache,
)
from deep_research.services.deployment.in_app import InAppTranslator
from deep_research.services.deployment.job_runner import (
    HEARTBEAT_INTERVAL_SECONDS,
    HEARTBEAT_TIMEOUT_SECONDS,
    DeploymentBudgetExceededError,
    DeploymentJobRunner,
)
from deep_research.services.deployment.mlflow_deploy import MlflowAgentTranslator
from deep_research.services.deployment.shell_app import ShellAppExporter
from deep_research.services.deployment.translator import (
    Artifact,
    DeploymentCleanupError,
    DeploymentTranslator,
    InlineDeploymentTranslator,
)
from deep_research.services.deployment_service import DeploymentService

logger = logging.getLogger(__name__)

router = APIRouter()


def _probe_deploy_here_permissions(workspace_client: WorkspaceClient) -> None:
    """Verify the OBO token can use the APIs required by Deploy Here."""
    next(iter(workspace_client.apps.list(page_size=1)), None)


def _clip_log_value(value: object | None, *, limit: int = 1000) -> str | None:
    """Return a single-line, bounded string for diagnostic log fields."""
    if value is None:
        return None
    text = str(value).replace("\n", "\\n").replace("\r", "\\r")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...<truncated>"


def _workflow_log_summary(definition: dict[str, object]) -> dict[str, object | None]:
    """Return safe workflow fields for deployment diagnostics."""
    root = definition.get("root")
    root_children: list[str] = []
    root_type: object | None = None
    if isinstance(root, dict):
        root_type = root.get("type")
        children = root.get("children")
        if isinstance(children, list):
            for child in children:
                if not isinstance(child, dict):
                    continue
                child_id = child.get("id") or "<unnamed>"
                child_type = child.get("type") or "<unknown>"
                child_label = child.get("label") or ""
                root_children.append(f"{child_id}:{child_type}:{child_label}")
    return {
        "workflow_name": _clip_log_value(definition.get("name"), limit=200),
        "workflow_description": _clip_log_value(
            definition.get("description"),
            limit=300,
        ),
        "root_type": _clip_log_value(root_type, limit=80),
        "root_children": _clip_log_value(root_children, limit=500),
        "output_keys": _clip_log_value(definition.get("output_keys"), limit=200),
    }


def _first_attr(obj: object, names: tuple[str, ...]) -> object | None:
    """Return the first present non-None attribute from ``obj``."""
    for name in names:
        value: object | None = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _probe_exception_debug(exc: BaseException) -> dict[str, object | None]:
    """Extract useful SDK/HTTP fields without depending on SDK internals."""
    response = getattr(exc, "response", None)
    response_headers = getattr(response, "headers", None)
    response_body = (
        _first_attr(response, ("text", "content"))
        if response is not None
        else _first_attr(exc, ("body", "response_body"))
    )
    request_id = None
    headers_get = getattr(response_headers, "get", None)
    if callable(headers_get):
        request_id = (
            headers_get("x-databricks-request-id")
            or headers_get("x-request-id")
            or headers_get("x-databricks-org-id")
        )
    request_id = request_id or _first_attr(exc, ("request_id", "request_id_"))
    return {
        "status_code": _first_attr(exc, ("status_code", "status"))
        or _first_attr(response, ("status_code", "status")),
        "error_code": _first_attr(exc, ("error_code", "error_code_")),
        "request_id": request_id,
        "response_body": _clip_log_value(response_body),
    }


async def _can_manage_deployment(
    session: AsyncSession,
    user_id: str,
    deployment: AgentDeployment,
) -> bool:
    """W9: predicate for who may list/get/deactivate a deployment.

    Owners of the parent agent OR the user who originally deployed it
    can manage the deployment. Anyone else — even if they can *see* the
    parent agent via workspace visibility — sees a 403. This closes the
    GET/DELETE asymmetry codex flagged: pre-W9, a shared-agent owner
    could see a teammate's deployment but couldn't deactivate it, and
    the only escape hatch was a force-delete that bypassed translator
    cleanup and leaked external resources.
    """
    if deployment.deployed_by == user_id:
        return True
    agent_service = AgentV2Service(session)
    agent = await agent_service.get_owned(deployment.agent_id, user_id)
    return agent is not None


# ---------------------------------------------------------------------------
# Translator dispatch
# ---------------------------------------------------------------------------

# All four deployment modes wired:
#   Phase 1   -> InAppTranslator        (in-app picker)
#   Phase 2-A -> BatchTranslator        (Lakeflow + SQL via ai_query)
#   Phase 2-B -> ShellAppExporter       (standalone Databricks App zip)
#   Phase 3   -> MlflowAgentTranslator  (Mosaic AI agent serving)
# The _translator_for() 501 fallback now serves only as a defensive
# catch-all for future modes that may be added without registration.
_TRANSLATORS: dict[DeploymentMode, DeploymentTranslator] = {
    DeploymentMode.IN_APP: InAppTranslator(),
    DeploymentMode.BATCH: BatchTranslator(),
    DeploymentMode.SHELL_APP: ShellAppExporter(),
    DeploymentMode.MLFLOW_AGENT: MlflowAgentTranslator(),
}


def _translator_for(mode: DeploymentMode) -> DeploymentTranslator:
    """Return the registered translator for ``mode`` or raise 501."""
    translator = _TRANSLATORS.get(mode)
    if translator is None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=(
                f"Deployment mode {mode.value!r} is not yet implemented; "
                "see plan Section L for phase sequencing."
            ),
        )
    return translator


def _to_response(deployment: AgentDeployment) -> DeploymentResponse:
    return DeploymentResponse.model_validate(deployment)


def _track_deploy_here_task(request: Request, task: asyncio.Task[None]) -> None:
    """Keep fire-and-forget deploy-here tasks reachable until they finish."""
    tasks: set[asyncio.Task[None]] | None = getattr(
        request.app.state,
        "deploy_here_tasks",
        None,
    )
    if tasks is None:
        tasks = set()
        request.app.state.deploy_here_tasks = tasks
    tasks.add(task)
    task.add_done_callback(tasks.discard)


def _schedule_deploy_here_background(
    request: Request,
    *,
    deployment_id: UUID,
    artifact: Artifact,
    workspace_client: WorkspaceClient,
    worker_id: str,
) -> None:
    """Create and track a deploy-here background task."""
    task = asyncio.create_task(
        _run_deploy_here_background(
            deployment_id=deployment_id,
            artifact=artifact,
            workspace_client=workspace_client,
            worker_id=worker_id,
        ),
        name=f"deploy-here-{deployment_id}",
    )
    _track_deploy_here_task(request, task)


def _touch_deploy_here_heartbeat(
    deployment: AgentDeployment,
    *,
    worker_id: str,
) -> None:
    """Write heartbeat fields used by the shared deployment zombie janitor."""
    now = datetime.now(UTC)
    deployment.worker_id = worker_id
    deployment.last_heartbeat = now
    deployment.heartbeat_timeout_at = now + timedelta(
        seconds=HEARTBEAT_TIMEOUT_SECONDS,
    )


async def _deploy_here_heartbeat_loop(
    *,
    deployment_id: UUID,
    worker_id: str,
    stop: asyncio.Event,
) -> None:
    """Refresh heartbeat fields while an OBO deploy-here task is running."""
    from deep_research.db.session import get_session_maker  # noqa: PLC0415

    session_maker = get_session_maker()
    while not stop.is_set():
        try:
            await asyncio.wait_for(
                stop.wait(),
                timeout=HEARTBEAT_INTERVAL_SECONDS,
            )
            return
        except TimeoutError:
            pass

        try:
            async with session_maker() as heartbeat_session:
                deployment = await heartbeat_session.get(
                    AgentDeployment,
                    deployment_id,
                )
                if (
                    deployment is None
                    or deployment.status != DeploymentStatus.DEPLOYING.value
                    or deployment.worker_id != worker_id
                ):
                    return
                _touch_deploy_here_heartbeat(deployment, worker_id=worker_id)
                await heartbeat_session.commit()
                logger.info(
                    "DEPLOY_HERE_HEARTBEAT deployment=%s worker_id=%s",
                    deployment_id,
                    worker_id,
                )
        except Exception:  # noqa: BLE001
            logger.exception(
                "DEPLOY_HERE_HEARTBEAT_FAILED deployment=%s worker_id=%s",
                deployment_id,
                worker_id,
            )


async def _run_deploy_here_background(
    *,
    deployment_id: UUID,
    artifact: Artifact,
    workspace_client: WorkspaceClient,
    worker_id: str,
) -> None:
    """Finish a deploy-here action after the HTTP response has returned.

    The caller constructs ``workspace_client`` from the request's OBO token
    before scheduling this task. Keeping that client in the task preserves the
    user-on-behalf-of semantics while avoiding a multi-minute open HTTP
    request through the Databricks App proxy.
    """
    from deep_research.db.session import get_session_maker  # noqa: PLC0415

    start_ts = time.monotonic()
    logger.info(
        "DEPLOY_HERE_BACKGROUND_START deployment=%s",
        deployment_id,
    )

    heartbeat_stop = asyncio.Event()
    heartbeat_task = asyncio.create_task(
        _deploy_here_heartbeat_loop(
            deployment_id=deployment_id,
            worker_id=worker_id,
            stop=heartbeat_stop,
        ),
        name=f"deploy-here-heartbeat-{deployment_id}",
    )

    try:
        session_maker = get_session_maker()
        async with session_maker() as bg_session:
            service = DeploymentService(bg_session)
            deployment = await service.get(deployment_id)
            if deployment is None:
                logger.warning(
                    "DEPLOY_HERE_BACKGROUND_ROW_MISSING deployment=%s",
                    deployment_id,
                )
                return
            if deployment.worker_id != worker_id:
                logger.warning(
                    "DEPLOY_HERE_BACKGROUND_WORKER_MISMATCH deployment=%s "
                    "expected_worker_id=%s actual_worker_id=%s",
                    deployment_id,
                    worker_id,
                    deployment.worker_id,
                )
                return

            translator = _translator_for(DeploymentMode(deployment.mode))
            inline_translator = cast(InlineDeploymentTranslator, translator)
            result = await inline_translator.deploy_inline(
                artifact,
                deployment.config,
                deployment,
                workspace_client,
            )

            duration_ms = int((time.monotonic() - start_ts) * 1000)
            if result.success:
                deployment = await service.update_status(
                    deployment.id,
                    DeploymentStatus.ACTIVE,
                    endpoint_name=result.endpoint_name,
                    model_name=result.model_name,
                    external_resource_ids=result.external_resource_ids,
                )
                logger.info(
                    "DEPLOY_HERE_BACKGROUND_DONE deployment=%s status=%s duration_ms=%s",
                    deployment.id,
                    DeploymentStatus.ACTIVE.value,
                    duration_ms,
                )
            else:
                ext = dict(result.external_resource_ids or {})
                if result.error_message:
                    ext.setdefault("error_kind", "deploy_failed")
                deployment = await service.update_status(
                    deployment.id,
                    DeploymentStatus.FAILED,
                    error_message=result.error_message,
                    external_resource_ids=ext if ext else None,
                )
                logger.info(
                    "DEPLOY_HERE_BACKGROUND_DONE deployment=%s "
                    "status=%s duration_ms=%s error_message=%s error_kind=%s",
                    deployment.id,
                    DeploymentStatus.FAILED.value,
                    duration_ms,
                    result.error_message,
                    ext.get("error_kind"),
                )
            await bg_session.commit()
    except asyncio.CancelledError:
        logger.warning(
            "DEPLOY_HERE_BACKGROUND_CANCELLED deployment=%s worker_id=%s",
            deployment_id,
            worker_id,
        )
        try:
            session_maker = get_session_maker()
            async with session_maker() as bg_session:
                service = DeploymentService(bg_session)
                await service.update_status(
                    deployment_id,
                    DeploymentStatus.FAILED,
                    error_message="server_shutdown",
                    external_resource_ids={"error_kind": "deploy_failed"},
                )
                await bg_session.commit()
        finally:
            raise
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "DEPLOY_HERE_BACKGROUND_FAILED deployment=%s exc_class=%s exc_msg=%s",
            deployment_id,
            type(exc).__name__,
            str(exc),
        )
        try:
            session_maker = get_session_maker()
            async with session_maker() as bg_session:
                service = DeploymentService(bg_session)
                await service.update_status(
                    deployment_id,
                    DeploymentStatus.FAILED,
                    error_message=f"{type(exc).__name__}: {exc}",
                    external_resource_ids={"error_kind": "deploy_failed"},
                )
                await bg_session.commit()
        except Exception:  # noqa: BLE001
            logger.exception(
                "DEPLOY_HERE_BACKGROUND_MARK_FAILED_FAILED deployment=%s",
                deployment_id,
            )
    finally:
        heartbeat_stop.set()
        await heartbeat_task


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=DeploymentResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_deployment(
    request: CreateDeploymentRequest,
    fastapi_request: Request,
    user: CurrentUser,
    session: Annotated[AsyncSession, Depends(get_db)],
    run_async: bool = Query(
        default=True,
        description=(
            "When false, create only the deployment row. Used by inline "
            "same-workspace deploy flows that call a follow-up action."
        ),
    ),
) -> DeploymentResponse:
    """Submit a deployment for async execution (W12).

    Pre-W12 this endpoint ran the entire translator chain inline, leaving
    the request blocked for the multi-minute MLflow + serving-endpoint
    create + Databricks-Apps spin-up. Now we:

      1. Run the fast translator.validate() inline so the wizard sees a
         422 immediately on bad input.
      2. Write a PENDING row, commit.
      3. Submit to the ``DeploymentJobRunner`` unless ``run_async=false``.
         Inline deploy-here flows create the row first, then run through the
         OBO-scoped action endpoint on the same row.
      4. Return 202 with the PENDING row. The frontend's status-poll
         hook (``useDeploymentStatusPoll``) observes the transitions.

    A per-user concurrency budget guards against accidental floods —
    ``DeploymentBudgetExceededError`` becomes a 429 with a hint.
    """
    agent_service = AgentV2Service(session)
    agent = await agent_service.get_for_user(request.agent_id, user.user_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")

    revision = await session.get(AgentRevision, request.revision_id)
    if revision is None or revision.agent_id != agent.id:
        raise HTTPException(
            status_code=404,
            detail="Revision not found for this agent",
        )

    definition = revision.definition if isinstance(revision.definition, dict) else {}
    deployability = classify_revision_deployability(definition)
    if not deployability.deployable:
        logger.warning(
            "DEPLOYMENT_CREATE_BLOCKED_DEFAULT_REVISION user=%s agent_id=%s "
            "revision_id=%s workflow_name=%s root_children=%s",
            user.user_id,
            request.agent_id,
            request.revision_id,
            deployability.workflow_name,
            deployability.root_child_summary,
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=default_revision_not_deployable_detail(
                agent_id=request.agent_id,
                revision_id=request.revision_id,
                deployability=deployability,
            ),
        )

    config_dict = request.config.model_dump()
    mode = request.config.mode
    translator = _translator_for(mode)
    workflow_summary = _workflow_log_summary(definition)
    logger.info(
        "DEPLOYMENT_CREATE_REQUEST user=%s agent_id=%s revision_id=%s mode=%s "
        "app_name=%s target=%s framework_git_tag=%s workflow_name=%s "
        "workflow_description=%s root_type=%s root_children=%s output_keys=%s",
        user.user_id,
        request.agent_id,
        request.revision_id,
        mode.value,
        config_dict.get("app_name"),
        config_dict.get("target"),
        config_dict.get("framework_git_tag"),
        workflow_summary["workflow_name"],
        workflow_summary["workflow_description"],
        workflow_summary["root_type"],
        workflow_summary["root_children"],
        workflow_summary["output_keys"],
    )

    validation = await translator.validate(agent, revision, config_dict)
    if not validation.valid:
        raise HTTPException(
            status_code=422,
            detail={
                "error_kind": "validation_failed",
                "errors": [e.message for e in validation.errors],
                "warnings": [w.message for w in validation.warnings],
            },
        )

    service = DeploymentService(session)
    deployment = await service.create(
        agent_id=agent.id,
        revision_id=revision.rev_id,
        mode=mode,
        config=config_dict,
        deployed_by=user.user_id,
    )
    await session.commit()

    if not run_async:
        return _to_response(deployment)

    runner: DeploymentJobRunner | None = getattr(
        fastapi_request.app.state, "deployment_runner", None
    )
    if runner is None:
        # Should never happen — lifespan kicks the runner before the app
        # accepts traffic. Surface as a 503 so on-call can spot the gap.
        raise HTTPException(
            status_code=503,
            detail="DeploymentJobRunner not initialized",
        )
    try:
        runner.submit(deployment.id, user.user_id)
    except DeploymentBudgetExceededError as exc:
        raise HTTPException(
            status_code=429,
            detail={
                "error_kind": "deployment_budget_exceeded",
                "in_flight": exc.current,
                "limit": exc.limit,
            },
            headers={"Retry-After": "60"},
        ) from exc

    return _to_response(deployment)


@router.get("", response_model=DeploymentListResponse)
async def list_deployments(
    user: CurrentUser,
    session: Annotated[AsyncSession, Depends(get_db)],
    mode: DeploymentMode | None = Query(default=None),
    deployment_status: DeploymentStatus | None = Query(default=None, alias="status"),
    agent_id: UUID | None = Query(
        default=None,
        description="Filter to deployments of a single agent (W9 authz still applies)",
    ),
    cursor: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
) -> DeploymentListResponse:
    """List deployments owned by the current user with cursor pagination."""
    service = DeploymentService(session)
    items, next_cursor = await service.list_for_user(
        user.user_id,
        mode=mode,
        status=deployment_status,
        agent_id=agent_id,
        cursor=cursor,
        limit=limit,
    )
    return DeploymentListResponse(
        items=[_to_response(d) for d in items],
        next_cursor=next_cursor,
    )


# ---------------------------------------------------------------------------
# Deploy-here capability probe endpoints (Section S2)
# ---------------------------------------------------------------------------
# IMPORTANT: these routes must be registered BEFORE /{deployment_id} so that
# FastAPI does not greedily match "can-deploy-here" as a UUID path param.


def _request_has_obo(request: Request) -> str:
    """Return ``"obo"`` when the request carries a forwarded user token.

    In Databricks Apps the runtime injects ``X-Forwarded-Access-Token`` with
    the user's OBO token.  In local dev this header is absent so we fall back
    to the service-principal client.  The ``actor`` field in the response lets
    the frontend surface a contextual hint about ownership semantics.
    """
    return "obo" if request.headers.get("X-Forwarded-Access-Token") else "sp_fallback"


@router.get("/can-deploy-here", response_model=CanDeployHereResponse)
async def can_deploy_here(
    fastapi_request: Request,
    user: CurrentUser,
) -> CanDeployHereResponse:
    """Eager workspace-capability probe for the Deploy Here feature (S2).

    Calls the Apps API with the caller's OBO-scoped client to verify they can
    create/update apps. Results are TTL-cached per ``(user_id, host)`` so
    repeated wizard opens do not hammer Databricks control-plane APIs.

    Only confirmed permission failures and successes are cached; transient
    network errors are NOT cached so the next call retries.

    Diagnostic logging: every branch emits a structured log line tagged
    ``CAN_DEPLOY_HERE_*`` so the path of any single request can be traced
    end-to-end via ``make logs TARGET=ais SEARCH="--search CAN_DEPLOY_HERE"``.
    """
    logger.info(
        "CAN_DEPLOY_HERE_REQUEST user=%s url=%s has_obo=%s",
        user.user_id,
        str(fastapi_request.url),
        bool(fastapi_request.headers.get("X-Forwarded-Access-Token")),
    )

    cache = get_default_cache()
    try:
        workspace_client = get_user_workspace_client(fastapi_request)
    except Exception as exc:  # noqa: BLE001
        # `get_user_workspace_client` may raise HTTPException(401) for missing
        # OBO header in Apps runtime — surface as can_deploy=false rather than
        # bubbling a 401 to the wizard (which would render a generic toast).
        logger.warning(
            "CAN_DEPLOY_HERE_WC_BUILD_FAILED user=%s exc_class=%s exc_msg=%s",
            user.user_id,
            type(exc).__name__,
            str(exc),
        )
        return CanDeployHereResponse(
            can_deploy=False,
            reason="missing_obo_token",
            probe_status="denied",
            actor="sp_fallback",
        )

    actor = _request_has_obo(fastapi_request)
    host: str = getattr(getattr(workspace_client, "config", None), "host", "") or ""

    cached = cache.get(user.user_id, host)
    if cached is not None:
        logger.info(
            "CAN_DEPLOY_HERE_CACHE_HIT user=%s host=%s ok=%s reason=%s",
            user.user_id,
            host,
            cached.ok,
            cached.reason,
        )
        return CanDeployHereResponse(
            can_deploy=cached.ok,
            reason=cached.reason,
            probe_status="ok" if cached.ok else "denied",
            actor=actor,
        )

    logger.info(
        "CAN_DEPLOY_HERE_PROBE_START user=%s host=%s actor=%s",
        user.user_id,
        host,
        actor,
    )

    try:
        await asyncio.to_thread(_probe_deploy_here_permissions, workspace_client)
        cache.set(user.user_id, host, ok=True)
        logger.info(
            "CAN_DEPLOY_HERE_PROBE_OK user=%s host=%s probe_status=ok can_deploy=True",
            user.user_id,
            host,
        )
        return CanDeployHereResponse(
            can_deploy=True,
            reason=None,
            probe_status="ok",
            actor=actor,
        )
    except Exception as exc:  # noqa: BLE001
        _ok, reason = _classify_probe_error(exc)
        if reason == "missing_workspace_permission":
            cache.set(user.user_id, host, ok=False, reason=reason)
        probe_status = "denied" if reason == "missing_workspace_permission" else "unknown"
        can_deploy = probe_status != "denied"
        error_debug = _probe_exception_debug(exc)
        logger.warning(
            "CAN_DEPLOY_HERE_PROBE_FAILED user=%s host=%s "
            "probe_status=%s can_deploy=%s exc_class=%s exc_msg=%s "
            "classified_reason=%s exc_status=%s exc_error_code=%s "
            "exc_request_id=%s exc_body=%s",
            user.user_id,
            host,
            probe_status,
            can_deploy,
            type(exc).__name__,
            _clip_log_value(str(exc)),
            reason,
            error_debug["status_code"],
            error_debug["error_code"],
            error_debug["request_id"],
            error_debug["response_body"],
        )
        # "probe_error" is an internal ProbeReason not in DeployHereErrorKind;
        # surface it as probe_status=unknown. The list probe is advisory only:
        # the real deploy path will call apps.create/apps.deploy and return the
        # authoritative permission failure if the user actually lacks access.
        deploy_reason = reason if reason == "missing_workspace_permission" else None
        return CanDeployHereResponse(
            can_deploy=can_deploy,
            reason=deploy_reason,
            probe_status=probe_status,
            actor=actor,
        )


@router.post("/can-deploy-here/refresh", response_model=CanDeployHereResponse)
async def refresh_can_deploy_here(
    fastapi_request: Request,
    user: CurrentUser,
) -> CanDeployHereResponse:
    """Invalidate the cached capability probe and re-probe immediately.

    Called by the UI's "Re-check permissions" button after an admin has
    granted the user App permissions mid-session.
    """
    logger.info(
        "CAN_DEPLOY_HERE_REFRESH_REQUEST user=%s url=%s has_obo=%s origin=%s",
        user.user_id,
        str(fastapi_request.url),
        bool(fastapi_request.headers.get("X-Forwarded-Access-Token")),
        fastapi_request.headers.get("origin"),
    )
    workspace_client = get_user_workspace_client(fastapi_request)
    host: str = getattr(getattr(workspace_client, "config", None), "host", "") or ""
    get_default_cache().invalidate(user.user_id, host)
    logger.info(
        "CAN_DEPLOY_HERE_REFRESH_INVALIDATE user=%s host=%s",
        user.user_id,
        host,
    )
    return await can_deploy_here(fastapi_request=fastapi_request, user=user)


@router.get("/{deployment_id}", response_model=DeploymentResponse)
async def get_deployment(
    deployment_id: UUID,
    user: CurrentUser,
    session: Annotated[AsyncSession, Depends(get_db)],
) -> DeploymentResponse:
    """Return a deployment by id. Caller must be the deployer or own the
    parent agent (W9 authorization model — previously a workspace-visible
    non-owner could GET deployments but not manage them; that asymmetry
    is now closed in both directions: see ``DELETE`` below).
    """
    service = DeploymentService(session)
    deployment = await service.get(deployment_id)
    if deployment is None:
        raise HTTPException(status_code=404, detail="Deployment not found")
    if not await _can_manage_deployment(session, user.user_id, deployment):
        # Use 404 (not 403) to avoid leaking the existence of a deployment
        # the caller is not permitted to manage.
        raise HTTPException(status_code=404, detail="Deployment not found")
    return _to_response(deployment)


@router.delete("/{deployment_id}", status_code=status.HTTP_200_OK)
async def deactivate_deployment(
    deployment_id: UUID,
    fastapi_request: Request,
    user: CurrentUser,
    session: Annotated[AsyncSession, Depends(get_db)],
) -> DeploymentResponse:
    """Deactivate (or cancel) the deployment.

    Three paths:

    - ACTIVE → translator-driven deactivate (existing W4 contract). On
      ``DeploymentCleanupError`` the row transitions through
      ``cleanup_failed`` after ``MAX_CLEANUP_ATTEMPTS`` (one DELETE call
      = one attempt; no retry inside a single request).
    - PENDING / DEPLOYING (W12 new) → sets ``cancel_requested`` via the
      ``DeploymentJobRunner``. The background worker observes the flag
      on its next heartbeat tick and lands the row in FAILED with
      ``error_message="cancelled"``. Returns 202 with the current row
      shape so the frontend can keep polling.
    - Terminal → short-circuit 200 with the row as-is.
    """
    service = DeploymentService(session)
    deployment = await service.get(deployment_id)
    if deployment is None:
        raise HTTPException(status_code=404, detail="Deployment not found")
    if not await _can_manage_deployment(session, user.user_id, deployment):
        # W9: parent-agent owners are now permitted alongside the
        # original deployer (closes the shared-agent "ownership
        # dead-end" — codex finding).
        raise HTTPException(status_code=404, detail="Deployment not found")
    status_before = deployment.status
    deployment_mode = deployment.mode
    if deployment.status == DeploymentStatus.FAILED.value:
        deployment = await service.deactivate(deployment.id)
        await session.commit()
        logger.info(
            "DEPLOYMENT_UNDEPLOY_OK deployment=%s mode=%s "
            "status_before=%s status_after=%s actor=%s",
            deployment_id,
            deployment_mode,
            status_before,
            deployment.status,
            user.user_id,
        )
        return _to_response(deployment)

    # Narrowed short-circuit: only DEACTIVATED is truly terminal here.
    # CLEANUP_FAILED must drop through to re-enter the translator path so
    # users can retry cleanup; without this guard a "Retry cleanup" click
    # would be a no-op. See plan imperative-wishing-lynx.md §Step 2b.
    if deployment.status == DeploymentStatus.DEACTIVATED.value:
        return _to_response(deployment)

    # W12 cancellation path: PENDING / DEPLOYING rows are owned by the
    # async DeploymentJobRunner. The DELETE here is a cancel-request, not
    # a synchronous teardown — the worker resolves it on its next
    # heartbeat tick.
    if deployment.status in (
        DeploymentStatus.PENDING.value,
        DeploymentStatus.DEPLOYING.value,
    ):
        runner: DeploymentJobRunner | None = getattr(
            fastapi_request.app.state, "deployment_runner", None
        )
        if runner is not None:
            await runner.cancel(deployment_id)
        # Re-read to surface the (already-mutated) cancel_requested flag.
        deployment = await service.get(deployment_id)
        if deployment is None:
            raise HTTPException(status_code=404, detail="Deployment not found")
        await session.commit()
        logger.info(
            "DEPLOYMENT_UNDEPLOY_OK deployment=%s mode=%s "
            "status_before=%s status_after=%s actor=%s cancel_requested=%s",
            deployment_id,
            deployment_mode,
            status_before,
            deployment.status,
            user.user_id,
            deployment.cancel_requested,
        )
        return _to_response(deployment)

    # When the user retries cleanup on a CLEANUP_FAILED row, give them a
    # fresh budget. Otherwise the next failure would immediately
    # re-escalate (attempts already at MAX_CLEANUP_ATTEMPTS).
    if deployment.status == DeploymentStatus.CLEANUP_FAILED.value:
        deployment = await service.reset_cleanup_attempts(deployment.id)

    translator = _translator_for(DeploymentMode(deployment.mode))
    # Prefer the user's OBO-scoped client so shell_app / mlflow deactivate
    # runs as the identity that originally created the external resources.
    # Falls back to SP inside the resolver when the OBO header is absent
    # (local dev / curl) — same behaviour as before this change.
    obo_client = (
        get_user_workspace_client(fastapi_request)
        if fastapi_request.headers.get("X-Forwarded-Access-Token")
        else None
    )
    resolver = WorkspaceClientResolver(obo_client=obo_client)
    try:
        await translator.deactivate(deployment, client_resolver=resolver)
    except DeploymentCleanupError as exc:
        # Real upstream cleanup failure (not 404). Bump attempts and decide
        # whether to escalate to CLEANUP_FAILED.
        logger.warning(
            "DEPLOYMENT_CLEANUP_FAILED deployment=%s resource=%s upstream=%s",
            deployment_id,
            exc.resource,
            exc.upstream_error_type,
        )
        attempts = deployment.cleanup_attempts + 1
        if attempts >= MAX_CLEANUP_ATTEMPTS:
            deployment = await service.mark_cleanup_failed(
                deployment.id,
                error_message=str(exc),
            )
        else:
            await service.increment_cleanup_attempts(deployment.id)
        await session.commit()
        raise HTTPException(
            status_code=409,
            detail={
                "error_kind": "deployment_cleanup_failed",
                "attempts": attempts,
                "max_attempts": MAX_CLEANUP_ATTEMPTS,
                "message": str(exc),
            },
        ) from exc
    except Exception as exc:  # noqa: BLE001 -- surface as CLEANUP_FAILED + 500
        # Non-DeploymentCleanupError translator crash (programmer bug, DB
        # error, etc.). Previously marked the row FAILED — but FAILED has
        # a no-op deactivate branch at the top of this handler, so any
        # later "Clean up" click would mark the row DEACTIVATED without
        # actually re-running the translator. That stranded external
        # resources as invisible orphans. Marking CLEANUP_FAILED routes
        # the next DELETE through the narrowed-terminal carve-out above,
        # so the translator can re-run.
        logger.exception("Translator deactivate failed for %s", deployment_id)
        deployment = await service.mark_cleanup_failed(
            deployment.id,
            error_message=str(exc),
        )
        await session.commit()
        raise HTTPException(
            status_code=500,
            detail="Deactivate failed; deployment marked CLEANUP_FAILED",
        ) from exc

    deployment = await service.deactivate(deployment.id)

    # TEMPORARY SHIM — D2 visibility flip-back for IN_APP mode.
    # Mirror of the ACTIVE-transition shim in job_runner.py: when the last
    # active IN_APP deployment for an agent is deactivated, flip
    # agent.visibility back to 'private' so the agent disappears from the
    # chat-picker without any frontend changes.
    # See .omc/plans/we-don-t-need-legacy-composed-wren.md §D2 for the real fix.
    if deployment.mode == DeploymentMode.IN_APP.value:
        remaining_active = await service.list_active_for_agent(deployment.agent_id)
        in_app_active = [d for d in remaining_active if d.mode == DeploymentMode.IN_APP.value]
        if not in_app_active:
            agent_service = AgentV2Service(session)
            await agent_service.update_visibility(deployment.agent_id, "private")

    await session.commit()
    logger.info(
        "DEPLOYMENT_UNDEPLOY_OK deployment=%s mode=%s "
        "status_before=%s status_after=%s actor=%s",
        deployment_id,
        deployment_mode,
        status_before,
        deployment.status,
        user.user_id,
    )
    return _to_response(deployment)


@router.get("/{deployment_id}/export-zip")
async def export_shell_app_zip(
    deployment_id: UUID,
    user: CurrentUser,
    session: Annotated[AsyncSession, Depends(get_db)],
) -> Response:
    """Re-render and download the shell-app zip for a SHELL_APP deployment.

    The zip is regenerated on demand from the persisted ``revision_id`` +
    ``config`` (we do not store the zip bytes -- the deployment row's
    ``external_resource_ids.shell_app_zip_sha256`` is the integrity check
    against subsequent re-renders).
    """
    service = DeploymentService(session)
    deployment = await service.get(deployment_id)
    if deployment is None:
        raise HTTPException(status_code=404, detail="Deployment not found")
    if not await _can_manage_deployment(session, user.user_id, deployment):
        # W9: same authorization model as GET/DELETE — owners + deployer.
        raise HTTPException(status_code=404, detail="Deployment not found")
    if deployment.mode != DeploymentMode.SHELL_APP.value:
        raise HTTPException(
            status_code=400,
            detail="Only shell_app deployments support zip export",
        )

    agent_service = AgentV2Service(session)
    agent = await agent_service.get_for_user(deployment.agent_id, user.user_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    revision = await session.get(AgentRevision, deployment.revision_id)
    if revision is None:
        raise HTTPException(status_code=404, detail="Revision not found")

    translator = _translator_for(DeploymentMode.SHELL_APP)
    artifact = await translator.translate(agent, revision, deployment.config)
    if not isinstance(artifact.payload, bytes):
        raise HTTPException(
            status_code=500,
            detail="Shell-app artifact payload is not bytes",
        )

    # W7: integrity check — recompute SHA256 over the regenerated zip and
    # compare against the digest captured at deploy time. A mismatch means
    # the deployment row's revision/config has drifted from what was
    # originally generated (or a non-deterministic input snuck in). Surfacing
    # this prevents serving an artifact that no longer matches the audit
    # trail.
    stored_resources = deployment.external_resource_ids or {}
    stored_digest = stored_resources.get("shell_app_zip_sha256")
    actual_digest = hashlib.sha256(artifact.payload).hexdigest()
    if stored_digest and stored_digest != actual_digest:
        logger.warning(
            "SHELL_APP_ZIP_INTEGRITY_MISMATCH",
            extra={
                "deployment_id": str(deployment.id),
                "stored_digest": stored_digest,
                "actual_digest": actual_digest,
            },
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error_kind": "shell_app_integrity_mismatch",
                "message": (
                    "Regenerated shell-app zip does not match the digest "
                    "stored at deploy time. The deployment may have drifted."
                ),
            },
        )

    app_name = deployment.config.get("app_name", f"shell-app-{deployment_id}")
    return Response(
        content=artifact.payload,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{app_name}.zip"',
        },
    )


@router.get("/{deployment_id}/export-sql")
async def export_batch_sql(
    deployment_id: UUID,
    user: CurrentUser,
    session: Annotated[AsyncSession, Depends(get_db)],
) -> Response:
    """W16: re-render and download the Lakeflow batch SQL for a BATCH deployment.

    Mirrors ``/export-zip`` for Mode 2: regenerate the SQL from the persisted
    revision + config, verify the SHA256 against the digest captured at
    deploy time, and stream the bytes as ``text/plain``. Authorization
    follows the W9 model (owner OR deployer).
    """
    service = DeploymentService(session)
    deployment = await service.get(deployment_id)
    if deployment is None:
        raise HTTPException(status_code=404, detail="Deployment not found")
    if not await _can_manage_deployment(session, user.user_id, deployment):
        raise HTTPException(status_code=404, detail="Deployment not found")
    if deployment.mode != DeploymentMode.BATCH.value:
        raise HTTPException(
            status_code=400,
            detail="Only batch deployments support SQL export",
        )

    agent_service = AgentV2Service(session)
    agent = await agent_service.get_for_user(deployment.agent_id, user.user_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    revision = await session.get(AgentRevision, deployment.revision_id)
    if revision is None:
        raise HTTPException(status_code=404, detail="Revision not found")

    translator = _translator_for(DeploymentMode.BATCH)
    artifact = await translator.translate(agent, revision, deployment.config)
    if not isinstance(artifact.payload, bytes):
        raise HTTPException(
            status_code=500,
            detail="Batch artifact payload is not bytes",
        )

    stored_resources = deployment.external_resource_ids or {}
    stored_digest = stored_resources.get("sql_artifact_sha256")
    actual_digest = hashlib.sha256(artifact.payload).hexdigest()
    if stored_digest and stored_digest != actual_digest:
        logger.warning(
            "BATCH_SQL_INTEGRITY_MISMATCH",
            extra={
                "deployment_id": str(deployment_id),
                "stored_digest": stored_digest,
                "actual_digest": actual_digest,
            },
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error_kind": "batch_sql_integrity_mismatch",
                "message": (
                    "Regenerated batch SQL does not match the digest stored "
                    "at deploy time. The deployment may have drifted."
                ),
            },
        )

    filename = f"agent-batch-{deployment_id}.sql"
    return Response(
        content=artifact.payload,
        media_type="text/plain; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


@router.post(
    "/{deployment_id}/actions/deploy-here",
    response_model=DeploymentResponse,
    status_code=status.HTTP_200_OK,
)
async def deploy_here_action(
    deployment_id: UUID,
    fastapi_request: Request,
    user: CurrentUser,
    session: Annotated[AsyncSession, Depends(get_db)],
    confirm_redeploy: int = Query(default=0, ge=0, le=1),
) -> DeploymentResponse:
    """Start an OBO-scoped deploy-here action and return the DEPLOYING row.

    The full Databricks Apps create/deploy/reachability lifecycle can take
    minutes and must not hold the Databricks App proxy request open. This
    endpoint captures the caller's OBO-scoped ``WorkspaceClient``, moves the
    row to DEPLOYING, schedules the actual deploy in-process, then lets the
    frontend poll ``/{deployment_id}/status`` for ACTIVE/FAILED.

    The caller must own the parent agent or be the original deployer (W9
    authorization model).

    Query params:
        confirm_redeploy (int, 0|1): Required when the deployment is already
            ACTIVE — prevents accidental re-deploys without explicit opt-in.
    """
    from sqlalchemy import text as sql_text  # noqa: PLC0415

    service = DeploymentService(session)
    deployment = await service.get(deployment_id)
    if deployment is None:
        raise HTTPException(status_code=404, detail="Deployment not found")

    if not await _can_manage_deployment(session, user.user_id, deployment):
        raise HTTPException(status_code=404, detail="Deployment not found")

    translator = _translator_for(DeploymentMode(deployment.mode))
    if getattr(translator, "deploy_inline", None) is None:
        raise HTTPException(
            status_code=400,
            detail={"error_kind": "mode_does_not_support_inline_deploy"},
        )

    # Redeploy guard: require explicit opt-in when already ACTIVE.
    if deployment.status == DeploymentStatus.ACTIVE.value and confirm_redeploy != 1:
        raise HTTPException(
            status_code=409,
            detail={"error_kind": "redeploy_requires_confirmation"},
        )
    if deployment.status == DeploymentStatus.DEPLOYING.value:
        logger.info(
            "DEPLOY_HERE_ALREADY_DEPLOYING deployment=%s user=%s",
            deployment_id,
            user.user_id,
        )
        return _to_response(deployment)

    # OBO-scoped workspace client for the deploy call (module-level import
    # so tests can patch the symbol at this module — see test_deployments_deploy_here.py).
    workspace_client = get_user_workspace_client(fastapi_request)

    # Single-flight advisory lock keyed on the deployment id string.
    lock_result = await session.execute(
        sql_text("SELECT pg_try_advisory_xact_lock(hashtext(:id))"),
        {"id": str(deployment_id)},
    )
    if not lock_result.scalar():
        raise HTTPException(
            status_code=409,
            detail={"error_kind": "deploy_already_in_progress"},
        )

    # Advisory capability-probe state: do not block Deploy Here on apps.list().
    # Databricks Apps OBO may reject list-all-apps probes even when the caller
    # can create/deploy the target app. The real apps.create/apps.deploy calls
    # below are the authoritative permission checks.
    _probe_host: str = getattr(getattr(workspace_client, "config", None), "host", "") or ""
    _cached_probe = get_default_cache().get(user.user_id, _probe_host)
    logger.info(
        "DEPLOY_HERE_CAPABILITY_PROBE_ADVISORY deployment=%s user=%s host=%s "
        "cached_probe_status=%s cached_reason=%s note=real_apps_create_deploy_is_authoritative",
        deployment_id,
        user.user_id,
        _probe_host,
        None if _cached_probe is None else ("ok" if _cached_probe.ok else "denied"),
        None if _cached_probe is None else _cached_probe.reason,
    )

    # Look up the agent and revision that back this deployment.
    agent_service = AgentV2Service(session)
    agent = await agent_service.get_for_user(deployment.agent_id, user.user_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    revision = await session.get(AgentRevision, deployment.revision_id)
    if revision is None:
        raise HTTPException(status_code=404, detail="Revision not found")

    # Re-render the artifact from the persisted revision + config.
    workflow_summary = _workflow_log_summary(revision.definition or {})
    logger.info(
        "DEPLOY_HERE_RENDER_ARTIFACT deployment=%s user=%s agent_id=%s "
        "revision_id=%s mode=%s app_name=%s workflow_name=%s "
        "workflow_description=%s root_type=%s root_children=%s output_keys=%s",
        deployment.id,
        user.user_id,
        deployment.agent_id,
        deployment.revision_id,
        deployment.mode,
        (deployment.config or {}).get("app_name"),
        workflow_summary["workflow_name"],
        workflow_summary["workflow_description"],
        workflow_summary["root_type"],
        workflow_summary["root_children"],
        workflow_summary["output_keys"],
    )
    artifact = await translator.translate(agent, revision, deployment.config)

    # Transition to DEPLOYING.
    prev_status = deployment.status
    deployment = await service.update_status(deployment.id, DeploymentStatus.DEPLOYING)
    deploy_here_worker_id = f"deploy-here:{uuid4().hex[:12]}"
    _touch_deploy_here_heartbeat(
        deployment,
        worker_id=deploy_here_worker_id,
    )
    await session.commit()
    await session.refresh(deployment)
    logger.info(
        "DEPLOYMENT_TRANSITION",
        extra={
            "deployment_id": str(deployment.id),
            "mode": deployment.mode,
            "from": prev_status,
            "to": DeploymentStatus.DEPLOYING.value,
        },
    )

    _schedule_deploy_here_background(
        fastapi_request,
        deployment_id=deployment.id,
        artifact=artifact,
        workspace_client=workspace_client,
        worker_id=deploy_here_worker_id,
    )
    logger.info(
        "DEPLOY_HERE_REQUEST_RETURNING deployment=%s status=%s",
        deployment.id,
        deployment.status,
    )

    return _to_response(deployment)


@router.get("/{deployment_id}/status", response_model=DeploymentStatusResponse)
async def get_deployment_status(
    deployment_id: UUID,
    user: CurrentUser,
    session: Annotated[AsyncSession, Depends(get_db)],
) -> DeploymentStatusResponse:
    """Lightweight status poll -- mirrors plan Section B.8.

    Used by the (future) frontend ``StatusPanel`` as a polling fallback when
    the SSE stream is unavailable.
    """
    service = DeploymentService(session)
    deployment = await service.get(deployment_id)
    if deployment is None:
        raise HTTPException(status_code=404, detail="Deployment not found")
    if deployment.deployed_by != user.user_id:
        agent_service = AgentV2Service(session)
        agent = await agent_service.get_for_user(deployment.agent_id, user.user_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Deployment not found")
    return DeploymentStatusResponse(
        status=DeploymentStatus(deployment.status),
        updated_at=deployment.updated_at,
        error_message=deployment.error_message,
        external_resource_ids=deployment.external_resource_ids,
    )


# ---------------------------------------------------------------------------
# Capability probes
# ---------------------------------------------------------------------------


@router.get(
    "/can-run/fast/{agent_id}",
    response_model=CanRunFastResponse,
)
async def can_run_fast(
    agent_id: UUID,
    user: CurrentUser,
    session: Annotated[AsyncSession, Depends(get_db)],
) -> CanRunFastResponse:
    """Visibility-only probe (plan Section B.7).

    Target latency <100 ms. The frontend renders the agent picker from this
    response; UC permission probing happens via ``/can-run/slow``.
    """
    agent_service = AgentV2Service(session)
    agent = await agent_service.get_for_user(agent_id, user.user_id)
    if agent is None:
        return CanRunFastResponse(can_run=False, reasons=["not_visible"])
    return CanRunFastResponse(can_run=True)


@router.get(
    "/can-run/slow/{agent_id}",
    response_model=CanRunSlowResponse,
)
async def can_run_slow(
    agent_id: UUID,
    user: CurrentUser,
    session: Annotated[AsyncSession, Depends(get_db)],
) -> CanRunSlowResponse:
    """UC permission probe (plan Section B.7).

    Phase 1 returns the same shape as the fast probe -- the UC catalog probe
    lands with Mode 3 (Phase 3). Cached field is False until that lands.
    """
    agent_service = AgentV2Service(session)
    agent = await agent_service.get_for_user(agent_id, user.user_id)
    if agent is None:
        return CanRunSlowResponse(can_run=False, reasons=["not_visible"])
    return CanRunSlowResponse(can_run=True, cached=False)
