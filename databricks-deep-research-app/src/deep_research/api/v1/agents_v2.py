"""Agent Designer V1 — agents_v2 CRUD endpoints with etag optimistic locking."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    status,
)
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ValidationError
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from deep_research.agent_designer.mermaid_export import serialize_to_mermaid
from deep_research.agent_designer.validation_cache import DbValidationCache
from deep_research.agent_designer.workflow_validation import (
    VALIDATOR_VERSION,
    ValidationSource,
    WorkflowValidationResult,
    compute_semantic_hash,
    validate_workflow,
)
from deep_research.agent_designer.yaml_export import serialize_to_yaml
from deep_research.core.databricks_auth import get_user_workspace_client
from deep_research.db.session import get_db, get_session_maker
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


def _build_critic_llm(llm_client: Any) -> Any:
    """Build the LLM adapter for the workflow validator, or ``None`` to skip.

    Fail-open: any missing client or import/init failure returns ``None`` so save
    reliability is never coupled to LLM availability — the validator then returns
    ``verdict="skipped"`` and the save proceeds.
    """
    if llm_client is None:
        logger.warning("save-time validation skipped: no llm_client available")
        return None
    try:
        from deep_research.agent_designer.llm_adapter import AppLLMAdapter

        return AppLLMAdapter(llm_client)
    except Exception as exc:  # noqa: BLE001 — fail open, never block save
        logger.warning(
            "save-time validation adapter init failed: %s", exc, exc_info=True
        )
        return None


async def _run_save_validation(
    definition: dict[str, Any],
    llm_client: Any,
    session: AsyncSession,
) -> WorkflowValidationResult:
    """Validate ``definition`` via the single validator service + DB cache.

    Never raises and never blocks (the caller decides via
    :func:`_raise_if_validation_blocks`). Logs every non-pass verdict so the
    outcome is never silent in the logs. An unchanged workflow (already validated
    during the build loop) is served from the cache with no LLM call.
    """
    result = await validate_workflow(
        definition=definition,
        intent=_extract_intent_from_definition(definition),
        required_outputs=None,
        llm=_build_critic_llm(llm_client),
        cache=DbValidationCache(session),
    )
    if result.verdict in ("needs_revision", "fail"):
        logger.warning(
            "save-time validation verdict=%s source=%s summary=%s",
            result.verdict,
            result.source.value,
            result.summary,
        )
    return result


def _raise_if_validation_blocks(
    result: WorkflowValidationResult,
    *,
    force: bool,
    strict: bool,
) -> None:
    """Raise HTTP 422 on ``verdict == "fail"`` — ONLY in strict mode and when
    ``force`` is not set.

    Advisory mode (the default) never hard-blocks a save on the stochastic LLM
    verdict: structural + deterministic-semantic validation already ran at
    request parse (``schemas.agent_v2``), and the verdict is surfaced to the UI
    via the response body + warning header instead. Strict mode
    (``?validation_mode=strict``) restores the hard gate for callers that want
    it (CI / admin / import); ``?force=true`` overrides only that gate.
    """
    if force or not strict:
        return
    if result.verdict == "fail":
        payload = result.model_dump(mode="json")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": (
                    "Workflow validation verdict=fail — the workflow does not "
                    "answer the user's request. Pass ?force=true to save "
                    "anyway, or revise the workflow first."
                ),
                # Surfaced under BOTH keys: "validation" is the canonical
                # WorkflowValidationResult contract; "critique" is kept so the
                # current frontend AgentCriticError parser (which keys on
                # "critique") still gets the typed path. Safe because a strict
                # block only fires on verdict=="fail" (a valid CritiqueResult
                # verdict). Drop "critique" once the FE reads "validation".
                "validation": payload,
                "critique": payload,
            },
        )


def _raise_if_coverage_blocks(definition: dict[str, Any], *, force: bool) -> None:
    """Deterministic save-gate: block the save when the report producer does not
    cover every requested topic, UNLESS ``force`` (the frontend 'Save draft anyway').

    Fast (no LLM call) so it never times out, and force-overridable so a deliberate
    work-in-progress draft can still be persisted. Structural errors already hard-block
    at request parse (``schemas.agent_v2``); the stochastic LLM critic stays advisory
    (background) — this gate is the deterministic, specific, never-silent quality bar."""
    if force:
        return
    from deep_research.agent_designer.semantic_validation import (
        prompt_term_coverage_errors,
    )

    errors = prompt_term_coverage_errors(definition)
    if errors:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": (
                    "This workflow can't be saved yet — its report doesn't cover every "
                    "requested topic. Resolve these (or pass ?force=true to save a draft):"
                ),
                "coverage_errors": [
                    {"message": e.message, "path": e.path} for e in errors
                ],
            },
        )


def _validation_warning_header(result: WorkflowValidationResult) -> str | None:
    """Latin-1-safe ``X-Critic-Warning`` value for a non-pass verdict, else None."""
    if result.verdict in ("needs_revision", "fail"):
        return _critic_warning_header_value(
            [f"validation verdict={result.verdict}: {result.summary}"]
        )
    return None


def _stamp_validation(agent: AgentV2, result: WorkflowValidationResult) -> None:
    """Persist the latest verdict + content hash on the agent row.

    Authoritative DB state — never trust AST-embedded metadata (spoofable)."""
    agent.last_validation = result.model_dump(mode="json")
    agent.last_validation_verdict = result.verdict
    agent.last_validation_hash = result.semantic_hash


def _critic_warning_header_value(warnings: list[str]) -> str:
    """Return a Starlette-safe HTTP header value for critic warnings.

    Starlette encodes response headers as latin-1. Critic summaries come from
    model text and may include Unicode punctuation or newlines, so normalize
    this advisory header before assigning it to ``response.headers``.
    """
    value = "; ".join(warnings).replace("\r", " ").replace("\n", " ")
    return value.encode("latin-1", errors="replace").decode("latin-1")


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


def _response_with_validation(
    agent: AgentV2,
    validation: WorkflowValidationResult | None,
    *,
    pending: bool = False,
) -> AgentV2Response:
    """``_to_response`` plus the advisory validation result + pending flag in the
    body so the UI can surface the verdict (never silent), regardless of mode.

    ``pending=True`` means a background validation for the current definition is
    in flight; ``validation`` is then None and the UI should poll GET until the
    verdict lands."""
    resp = AgentV2Response.model_validate(agent)
    resp.validation = validation
    resp.validation_pending = pending
    return resp


# ----- Decoupled advisory validation (off the request path) ------------------
#
# The advisory save (the only path the UI uses) commits the write durably and
# returns immediately; the slow LLM critic runs in a FastAPI background task so a
# save can NEVER hit the frontend's 30s request timeout. A cache hit still
# returns the verdict inline (no background needed). Strict mode keeps the old
# inline-blocking behavior. See `.omc/plans/decouple-designer-save-validation.md`.

# Hard cap on the background validator: a wedged critic stream can never run
# longer than this. The request has already returned — this only bounds the task
# (and lets the frontend's pending-poll resolve to a fallback instead of hanging).
BG_VALIDATION_TIMEOUT_S = 120.0


async def _advisory_save_probe(
    agent: AgentV2, session: AsyncSession
) -> tuple[WorkflowValidationResult | None, bool]:
    """Fast, LLM-free advisory check at save time. Returns ``(validation, needs_bg)``:

    * cache HIT  -> ``(cached authoritative verdict, False)``  [instant; caller stamps]
    * no intent  -> ``(None, False)``                          [nothing to judge]
    * cache MISS -> ``(None, True)``  [caller schedules background validation]

    Calls the ONE validator with ``llm=None`` so it only consults the content-
    addressed cache and never blocks on the critic — the whole point of the
    decoupling.
    """
    intent = _extract_intent_from_definition(agent.definition)
    if not intent.strip():
        return None, False
    probe = await validate_workflow(
        definition=agent.definition,
        intent=intent,
        required_outputs=None,
        llm=None,
        cache=DbValidationCache(session),
    )
    if probe.source == ValidationSource.CACHE:
        return probe, False
    return None, True


def _bg_fallback_result(
    definition: dict[str, Any], intent: str, reason: str
) -> WorkflowValidationResult:
    """A non-authoritative ``skipped`` result for when the background validator
    cannot finish (timeout). Carries the CURRENT semantic hash so the stale-race
    guard still stamps it — clearing the frontend's pending state — but it is
    never cached (``cacheable=False``)."""
    return WorkflowValidationResult(
        verdict="skipped",
        summary=(
            f"Background validation did not complete ({reason}). The agent is "
            "saved; ask in the designer chat to re-run validation."
        ),
        semantic_hash=compute_semantic_hash(definition, intent),
        intent_hash=hashlib.sha1(intent.strip().encode("utf-8")).hexdigest(),
        validator_version=VALIDATOR_VERSION,
        source=ValidationSource.FALLBACK,
        cacheable=False,
    )


async def _validate_in_background(
    *,
    agent_id: UUID,
    owner_id: str,
    definition: dict[str, Any],
    llm_client: Any,
) -> None:
    """Run the advisory validator OFF the request path (the save already
    committed), warm the cache, and stamp the verdict on the agent row.

    Opens its OWN session (the request session is closed by now). Stamps ONLY if
    the agent's CURRENT definition still matches what we validated, so a newer
    save is never clobbered by a slower older validation. Never raises: a
    background failure must not affect the (already successful) save.
    """
    intent = _extract_intent_from_definition(definition)
    try:
        session_maker = get_session_maker()
        async with session_maker() as session:
            try:
                result = await asyncio.wait_for(
                    validate_workflow(
                        definition=definition,
                        intent=intent,
                        required_outputs=None,
                        llm=_build_critic_llm(llm_client),
                        cache=DbValidationCache(session),
                    ),
                    timeout=BG_VALIDATION_TIMEOUT_S,
                )
            except TimeoutError:
                logger.warning(
                    "bg-validation timed out after %.0fs for agent %s",
                    BG_VALIDATION_TIMEOUT_S,
                    agent_id,
                )
                result = _bg_fallback_result(
                    definition, intent, f"timed out after {BG_VALIDATION_TIMEOUT_S:.0f}s"
                )

            agent = await AgentV2Service(session).get_for_user(agent_id, owner_id)
            if agent is None:
                return
            current_hash = compute_semantic_hash(
                agent.definition, _extract_intent_from_definition(agent.definition)
            )
            if current_hash != result.semantic_hash:
                logger.info(
                    "bg-validation result is stale (agent %s changed since save); "
                    "not stamping",
                    agent_id,
                )
                return
            _stamp_validation(agent, result)
            await session.commit()
            if result.verdict in ("needs_revision", "fail"):
                logger.warning(
                    "bg-validation verdict=%s source=%s summary=%s",
                    result.verdict,
                    result.source.value,
                    result.summary,
                )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 — a background task must never crash
        logger.warning(
            "bg-validation failed for agent %s: %s", agent_id, exc, exc_info=True
        )


def _hydrate_get_validation(
    agent: AgentV2,
) -> tuple[WorkflowValidationResult | None, bool]:
    """For GET: surface the stamped verdict only when it matches the agent's
    CURRENT definition. Returns ``(validation, pending)``.

    ``pending=True`` means the workflow has an intent to judge but no current
    verdict is stamped yet (an advisory background validation is/should be in
    flight). The frontend only acts on this flag while polling an in-flight save,
    so a never-validated legacy agent reading ``pending=True`` is harmless.
    """
    intent = _extract_intent_from_definition(agent.definition)
    if not intent.strip():
        return None, False
    stamped = agent.last_validation
    if isinstance(stamped, dict) and agent.last_validation_hash == compute_semantic_hash(
        agent.definition, intent
    ):
        try:
            return WorkflowValidationResult.model_validate(stamped), False
        except ValidationError:
            return None, True
    return None, True


async def _maybe_introspect_uc_params(
    definition: dict[str, Any] | None, fastapi_request: Request
) -> None:
    """Best-effort: fill uc_function ``config.params`` from information_schema
    under the caller's OBO identity, before persist.

    Fail-soft by contract — never raises, never blocks the save. Off the
    synchronous service path so a cold warehouse cannot exceed the client's 30s
    timeout: a short-capped ``asyncio.to_thread`` query, and the whole thing is
    skipped when there is no OBO token or no SQL warehouse.
    """
    if not isinstance(definition, dict):
        return
    tools = definition.get("tools")
    if not isinstance(tools, list) or not any(
        isinstance(t, dict) and t.get("kind") == "uc_function" for t in tools
    ):
        return
    # get_user_workspace_client 401s in Databricks Apps without this header;
    # local dev / curl-without-OBO simply skip introspection.
    if not fastapi_request.headers.get("X-Forwarded-Access-Token"):
        return
    try:
        from databricks_deep_research.tools.builtins.text_table.runtime_wiring import (
            StatementExecutionTableSQL,
        )

        from deep_research.agent.workflow_runner_factory import (
            _resolve_table_warehouse_id,
        )
        from deep_research.agent_designer.uc_function_introspect import (
            introspect_and_fill_uc_params,
        )

        warehouse_id = _resolve_table_warehouse_id()
        if not warehouse_id:
            logger.info("UC_FUNCTION_INTROSPECT_SKIP reason=no_warehouse")
            return
        executor = StatementExecutionTableSQL(
            workspace_client=get_user_workspace_client(fastapi_request),
            warehouse_id=warehouse_id,
            timeout_sec=5.0,
        )
        warnings = await introspect_and_fill_uc_params(
            definition, executor, timeout_seconds=8.0
        )
        for warning in warnings:
            logger.info("UC_FUNCTION_INTROSPECT_WARN %s", warning)
    except Exception as exc:  # noqa: BLE001 - introspection must never block save
        logger.warning("UC_FUNCTION_INTROSPECT_SKIP error=%s", str(exc)[:200])


@router.post("", response_model=AgentV2Response, status_code=status.HTTP_201_CREATED)
async def create_agent(
    request: CreateAgentV2Request,
    response: Response,
    user: CurrentUser,
    fastapi_request: Request,
    background_tasks: BackgroundTasks,
    force: bool = Query(False, description="Bypass the strict LLM critic verdict=fail gate."),
    validation_mode: str = Query(
        "advisory",
        description=(
            "'advisory' (default): save proceeds even on critic verdict=fail; "
            "the verdict is returned in the response body. 'strict': block "
            "with 422 on verdict=fail unless force=true."
        ),
    ),
    session: AsyncSession = Depends(get_db),
) -> AgentV2Response:
    # ADVISORY (default, the only path the UI uses): commit the write durably,
    # then validate OFF the request path so the save can never hit the client's
    # 30s timeout. A cache hit returns the verdict inline; a cache miss runs the
    # critic in a background task that stamps the verdict + warms the cache.
    # STRICT mode (?validation_mode=strict) validates inline and 422s on
    # verdict=fail (rolling back the create). Structural/semantic validation
    # already blocked at request parse.
    service = AgentV2Service(session)
    _t0 = time.monotonic()
    await _maybe_introspect_uc_params(request.definition, fastapi_request)
    agent = await service.create(owner_id=user.user_id, request=request)
    llm_client = getattr(fastapi_request.app.state, "llm_client", None)
    needs_bg = False
    validation: WorkflowValidationResult | None
    if validation_mode == "strict":
        validation = await _run_save_validation(agent.definition, llm_client, session)
        _raise_if_validation_blocks(validation, force=force, strict=True)
        _stamp_validation(agent, validation)
    else:
        validation, needs_bg = await _advisory_save_probe(agent, session)
        if validation is not None:
            _stamp_validation(agent, validation)
    # Deterministic coverage gate (force-overridable). Structural errors already
    # blocked at request parse; the LLM critic stays advisory/background.
    _raise_if_coverage_blocks(agent.definition, force=force)
    await session.commit()
    # Reload the DB-generated timestamps inside the async session: created_at
    # (server_default) / updated_at (onupdate=now()) are expired after the write
    # regardless of expire_on_commit, so the synchronous
    # AgentV2Response.model_validate(agent) below would otherwise lazy-load them
    # outside the greenlet → MissingGreenlet (a 500 on an otherwise-durable save).
    await session.refresh(agent, attribute_names=["created_at", "updated_at"])
    if needs_bg:
        background_tasks.add_task(
            _validate_in_background,
            agent_id=agent.id,
            owner_id=user.user_id,
            definition=agent.definition,
            llm_client=llm_client,
        )
    record_save_latency("create", (time.monotonic() - _t0) * 1000)
    response.headers["ETag"] = agent.etag
    header = _validation_warning_header(validation) if validation is not None else None
    if header is not None:
        response.headers["X-Critic-Warning"] = header
    await service._write_revision_best_effort(
        agent, agent.etag, user.user_id, validation=validation
    )
    return _response_with_validation(agent, validation, pending=needs_bg)


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
    # Surface the stamped advisory verdict (gated to the current definition) so
    # the UI can poll this endpoint after an advisory save until the background
    # validation lands (validation_pending flips False).
    validation, pending = _hydrate_get_validation(agent)
    return _response_with_validation(agent, validation, pending=pending)


@router.patch("/{agent_id}", response_model=AgentV2Response)
async def update_agent(
    agent_id: UUID,
    request: UpdateAgentV2Request,
    response: Response,
    user: CurrentUser,
    fastapi_request: Request,
    background_tasks: BackgroundTasks,
    if_match: str | None = Header(default=None, alias="If-Match"),
    force: bool = Query(False, description="Bypass the strict LLM critic verdict=fail gate."),
    validation_mode: str = Query(
        "advisory",
        description=(
            "'advisory' (default): save proceeds even on critic verdict=fail; "
            "the verdict is returned in the response body. 'strict': block "
            "with 422 on verdict=fail unless force=true."
        ),
    ),
    session: AsyncSession = Depends(get_db),
) -> AgentV2Response:
    if not if_match:
        raise HTTPException(
            status_code=428,
            detail="If-Match header required for PATCH",
        )
    # ETag is checked inside service.update FIRST, so a conflicting save 409s
    # before any LLM call. Validation then runs on the persisted (materialized)
    # definition — only when the PATCH carried one (partial name/description
    # updates skip it) — via the ONE validator + cache; advisory by default,
    # strict mode 422s on verdict=fail (which rolls back the update).
    service = AgentV2Service(session)
    _t0 = time.monotonic()
    await _maybe_introspect_uc_params(request.definition, fastapi_request)
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
    validation: WorkflowValidationResult | None = None
    llm_client = getattr(fastapi_request.app.state, "llm_client", None)
    needs_bg = False
    if request.definition is not None:
        if validation_mode == "strict":
            validation = await _run_save_validation(agent.definition, llm_client, session)
            _raise_if_validation_blocks(validation, force=force, strict=True)
            _stamp_validation(agent, validation)
        else:
            validation, needs_bg = await _advisory_save_probe(agent, session)
            if validation is not None:
                _stamp_validation(agent, validation)
        # Deterministic coverage gate (force-overridable) — only on a definition change.
        _raise_if_coverage_blocks(agent.definition, force=force)
    await session.commit()
    # Reload the DB-generated timestamps inside the async session: created_at
    # (server_default) / updated_at (onupdate=now()) are expired after the write
    # regardless of expire_on_commit, so the synchronous
    # AgentV2Response.model_validate(agent) below would otherwise lazy-load them
    # outside the greenlet → MissingGreenlet (a 500 on an otherwise-durable save).
    await session.refresh(agent, attribute_names=["created_at", "updated_at"])
    if needs_bg:
        background_tasks.add_task(
            _validate_in_background,
            agent_id=agent.id,
            owner_id=user.user_id,
            definition=agent.definition,
            llm_client=llm_client,
        )
    record_save_latency("update", (time.monotonic() - _t0) * 1000)
    response.headers["ETag"] = agent.etag
    header = (
        _validation_warning_header(validation) if validation is not None else None
    )
    if header is not None:
        response.headers["X-Critic-Warning"] = header
    await service._write_revision_best_effort(
        agent, agent.etag, user.user_id, validation=validation
    )
    return _response_with_validation(agent, validation, pending=needs_bg)


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
