"""Agent Designer V1 API — validate, registry, chat, and custom-tools endpoints.

POST   /api/v1/agent-designer/validate
GET    /api/v1/agent-designer/registry
POST   /api/v1/agent-designer/chat
POST   /api/v1/agent-designer/custom-tools
GET    /api/v1/agent-designer/custom-tools
GET    /api/v1/agent-designer/custom-tools/{tool_id}
PATCH  /api/v1/agent-designer/custom-tools/{tool_id}
DELETE /api/v1/agent-designer/custom-tools/{tool_id}
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any, Literal, cast
from uuid import UUID

from databricks_deep_research.tools.catalog_types import ProbeSample
from databricks_deep_research.tools.factories import BUILTIN_FACTORIES
from databricks_deep_research.tools.protocol import ToolContext
from databricks_deep_research.workflow.definition import ToolDeclaration
from databricks_deep_research.workflow.loader import load_workflow_from_dict
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from deep_research.agent_designer.assets import DesignerAsset
from deep_research.agent_designer.catalog_service import CatalogService
from deep_research.agent_designer.discovery import (
    DesignerDiscoveryAdapter,
    DiscoveredResource,
    SourceKind,
    _DiscoveryServiceProto,
)
from deep_research.agent_designer.registry import (
    AGENT_SUBTYPES,
    REGISTRY_VERSION,
    model_tiers_payload,
    node_types_payload,
    query_modes_payload,
    research_depths_payload,
    source_kinds_payload,
    tool_kinds_payload_with_custom,
)
from deep_research.agent_designer.semantic_validation import (
    semantic_validation_errors,
)
from deep_research.agent_designer.tool_probe import ProbeConfig, ProbeOrchestrator
from deep_research.agent_designer.yaml_import import YamlImportError, parse_and_validate_yaml
from deep_research.core.app_config import get_app_config
from deep_research.db.session import get_db
from deep_research.middleware.auth import CurrentUser
from deep_research.models.agent_v2 import CustomToolDef
from deep_research.observability.agent_designer_metrics import (
    record_registry_fetch,
    record_validation_error,
    record_yaml_import_outcome,
)
from deep_research.services.agent_v2_service import AgentV2Service
from deep_research.services.discovery_service import DiscoveryService

router = APIRouter()
logger = logging.getLogger(__name__)


async def _verify_agent_edit_permission(
    agent_id: UUID | None,
    *,
    user: CurrentUser,
    session: AsyncSession,
) -> None:
    if agent_id is None:
        return
    agent = await AgentV2Service(session).get_owned(agent_id, user.user_id)
    if agent is None:
        raise HTTPException(status_code=403, detail="Agent edit permission required")


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------


def _node_count(node: dict[str, Any]) -> int:
    """Recursively count nodes in a workflow subtree (the node itself + all descendants)."""
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


# ---------------------------------------------------------------------------
# Etag helper
# ---------------------------------------------------------------------------

def _compute_tool_etag(config_schema: dict[str, Any], updated_at: datetime) -> str:
    payload = json.dumps(config_schema, sort_keys=True, default=str) + updated_at.isoformat()
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# /custom-tools  CRUD
# ---------------------------------------------------------------------------


class CreateCustomToolRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    config_schema: dict[str, Any]
    factory_ref: str
    visibility: Literal["private", "workspace"] = "private"


class UpdateCustomToolRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str | None = None
    config_schema: dict[str, Any] | None = None
    factory_ref: str | None = None
    visibility: Literal["private", "workspace"] | None = None


class CustomToolResponse(BaseModel):
    id: UUID
    name: str
    kind: int
    config_schema: dict[str, Any]
    factory_ref: str
    visibility: str
    etag: str
    owner_id: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class CustomToolListResponse(BaseModel):
    items: list[CustomToolResponse]
    total: int


def _tool_response(tool: CustomToolDef) -> CustomToolResponse:
    return CustomToolResponse.model_validate(tool)


@router.post("/custom-tools", response_model=CustomToolResponse, status_code=201)
async def create_custom_tool(
    req: CreateCustomToolRequest,
    user: CurrentUser,
    session: AsyncSession = Depends(get_db),
) -> CustomToolResponse:
    # SECURITY: validate factory_ref against allow-list BEFORE any DB write.
    # Never resolve via importlib.import_module or any dynamic import.
    if req.factory_ref not in BUILTIN_FACTORIES:
        raise HTTPException(
            status_code=400,
            detail={
                "error_kind": "factory_ref_not_in_allowlist",
                "received": req.factory_ref,
            },
        )
    now = datetime.now(UTC)
    tool = CustomToolDef(
        id=uuid.uuid4(),
        owner_id=user.user_id,
        name=req.name,
        kind=15,
        config_schema=req.config_schema,
        factory_ref=req.factory_ref,
        etag=_compute_tool_etag(req.config_schema, now),
        visibility=req.visibility,
        created_at=now,
        updated_at=now,
    )
    session.add(tool)
    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=409,
            detail={"error_kind": "duplicate_name", "name": req.name},
        ) from exc
    return _tool_response(tool)


@router.get("/custom-tools", response_model=CustomToolListResponse)
async def list_custom_tools(
    user: CurrentUser,
    session: AsyncSession = Depends(get_db),
) -> CustomToolListResponse:
    stmt = select(CustomToolDef).where(
        or_(
            CustomToolDef.owner_id == user.user_id,
            CustomToolDef.visibility == "workspace",
        )
    )
    result = await session.execute(stmt)
    tools = list(result.scalars().all())
    return CustomToolListResponse(items=[_tool_response(tool) for tool in tools], total=len(tools))


@router.get("/custom-tools/{tool_id}", response_model=CustomToolResponse)
async def get_custom_tool(
    tool_id: UUID,
    response: Response,
    user: CurrentUser,
    session: AsyncSession = Depends(get_db),
) -> CustomToolResponse:
    stmt = select(CustomToolDef).where(CustomToolDef.id == tool_id)
    result = await session.execute(stmt)
    tool = result.scalar_one_or_none()
    if tool is None:
        raise HTTPException(status_code=404, detail="Custom tool not found")
    if tool.owner_id != user.user_id and tool.visibility != "workspace":
        raise HTTPException(status_code=404, detail="Custom tool not found")
    response.headers["ETag"] = tool.etag
    return _tool_response(tool)


@router.patch("/custom-tools/{tool_id}", response_model=CustomToolResponse)
async def update_custom_tool(
    tool_id: UUID,
    req: UpdateCustomToolRequest,
    response: Response,
    user: CurrentUser,
    if_match: str | None = Header(default=None, alias="If-Match"),
    session: AsyncSession = Depends(get_db),
) -> CustomToolResponse:
    if not if_match:
        raise HTTPException(status_code=428, detail="If-Match header required for PATCH")
    # SECURITY: validate factory_ref against allow-list before any write.
    if req.factory_ref is not None and req.factory_ref not in BUILTIN_FACTORIES:
        raise HTTPException(
            status_code=400,
            detail={
                "error_kind": "factory_ref_not_in_allowlist",
                "received": req.factory_ref,
            },
        )
    stmt = select(CustomToolDef).where(
        CustomToolDef.id == tool_id, CustomToolDef.owner_id == user.user_id
    )
    result = await session.execute(stmt)
    tool = result.scalar_one_or_none()
    if tool is None:
        raise HTTPException(status_code=404, detail="Custom tool not found")
    if tool.etag != if_match:
        raise HTTPException(
            status_code=409,
            detail={"error_kind": "etag_conflict", "current_etag": tool.etag},
        )
    if req.name is not None:
        tool.name = req.name
    if req.config_schema is not None:
        tool.config_schema = req.config_schema
    if req.factory_ref is not None:
        tool.factory_ref = req.factory_ref
    if req.visibility is not None:
        tool.visibility = req.visibility
    tool.updated_at = datetime.now(UTC)
    tool.etag = _compute_tool_etag(tool.config_schema, tool.updated_at)
    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=409,
            detail={"error_kind": "duplicate_name", "name": req.name},
        ) from exc
    response.headers["ETag"] = tool.etag
    return _tool_response(tool)


@router.delete("/custom-tools/{tool_id}", status_code=204)
async def delete_custom_tool(
    tool_id: UUID,
    user: CurrentUser,
    session: AsyncSession = Depends(get_db),
) -> Response:
    stmt = select(CustomToolDef).where(
        CustomToolDef.id == tool_id, CustomToolDef.owner_id == user.user_id
    )
    result = await session.execute(stmt)
    tool = result.scalar_one_or_none()
    if tool is None:
        raise HTTPException(status_code=404, detail="Custom tool not found")
    await session.delete(tool)
    await session.commit()
    return Response(status_code=204)


# ---------- /validate ----------

class ValidateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    definition: dict[str, Any]


class ValidationErrorItem(BaseModel):
    message: str
    path: str | None = None
    line: int | None = None
    kind: Literal["syntax", "schema", "validation"]


class WorkflowSummary(BaseModel):
    node_count: int
    tool_count: int
    source_count: int


class ValidateResponse(BaseModel):
    valid: bool
    errors: list[ValidationErrorItem]
    workflow_summary: WorkflowSummary | None


def _semantic_validation_errors(definition: dict[str, Any]) -> list[ValidationErrorItem]:
    """Adapter that runs the shared ``semantic_validation_errors`` checker
    and converts its frozen-dataclass output into Pydantic
    ``ValidationErrorItem`` for the API response shape.

    The checker itself moved to ``agent_designer.semantic_validation`` in
    W10 so that the ``agents-v2`` CRUD endpoints can run the same
    validation at write time (closing the "ASTs persist with semantic
    errors, then fail at runtime" gap).
    """
    raw = semantic_validation_errors(definition)
    items: list[ValidationErrorItem] = []
    for e in raw:
        # The dataclass kind is a str; ValidationErrorItem wants the
        # Literal["syntax","schema","validation"]. Narrow defensively
        # (the checker only ever emits "validation" today).
        kind: Literal["syntax", "schema", "validation"] = (
            cast(Literal["syntax", "schema", "validation"], e.kind)
            if e.kind in ("syntax", "schema", "validation")
            else "validation"
        )
        items.append(
            ValidationErrorItem(
                message=e.message,
                path=e.path,
                line=e.line,
                kind=kind,
            )
        )
    return items


@router.post("/validate", response_model=ValidateResponse)
async def validate_workflow(req: ValidateRequest) -> ValidateResponse:
    semantic_errors = _semantic_validation_errors(req.definition)
    if semantic_errors:
        record_validation_error("validation")
        return ValidateResponse(
            valid=False,
            errors=semantic_errors,
            workflow_summary=None,
        )
    try:
        load_workflow_from_dict(req.definition)
    except Exception as exc:
        logger.warning("AGENT_DESIGNER_VALIDATE_FAILED error=%s", str(exc)[:2000])
        record_validation_error("validation")
        return ValidateResponse(
            valid=False,
            errors=[ValidationErrorItem(message=str(exc), path=None, line=None, kind="validation")],
            workflow_summary=None,
        )
    summary = WorkflowSummary(
        node_count=_node_count(req.definition.get("root", {})),
        tool_count=len(req.definition.get("tools", []) or []),
        source_count=len(req.definition.get("sources", []) or []),
    )
    return ValidateResponse(valid=True, errors=[], workflow_summary=summary)


# ---------- /registry ----------


class RegistryResponse(BaseModel):
    node_types: list[dict[str, Any]]
    agent_subtypes: list[dict[str, Any]]
    tool_kinds: list[dict[str, Any]]
    model_tiers: list[str]
    query_modes: list[str]
    research_depths: list[str]
    source_kinds: list[dict[str, str]]
    version: str


class ResourcesResponse(BaseModel):
    resources: list[DiscoveredResource]
    total: int


class RefreshCatalogRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    definition: dict[str, Any]
    agent_id: UUID | None = None
    force_regen: bool = True


class RefreshCatalogResponse(BaseModel):
    definition: dict[str, Any]


class ProbeToolsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    definition: dict[str, Any]
    agent_id: UUID | None = None
    tool_names: list[str] | None = None
    user_query: str | None = None
    persist: bool = False


class ProbeToolsResponse(BaseModel):
    samples: list[ProbeSample]
    definition: dict[str, Any]
    persist: bool


_VALID_RESOURCE_KINDS: frozenset[str] = frozenset(
    {
        "vector_index",
        "genie_space",
        "knowledge_assistant",
        "serving_endpoint",
        "delta_table",
        "sql_warehouse",
    }
)


def _obo_token_from_request(fastapi_request: Request) -> str:
    return str(
        getattr(fastapi_request.state, "obo_token", None)
        or fastapi_request.headers.get("x-forwarded-access-token", "")
    )


def _parse_resource_kinds(raw_kinds: list[str] | None) -> list[SourceKind] | None:
    if not raw_kinds:
        return None
    parsed: list[SourceKind] = []
    for raw in raw_kinds:
        for item in raw.split(","):
            kind = item.strip()
            if not kind:
                continue
            if kind not in _VALID_RESOURCE_KINDS:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error_kind": "invalid_source_kind",
                        "kind": kind,
                        "valid_kinds": sorted(_VALID_RESOURCE_KINDS),
                    },
                )
            parsed.append(cast(SourceKind, kind))
    return parsed or None


@router.get("/resources", response_model=ResourcesResponse)
async def list_resources(
    user: CurrentUser,
    fastapi_request: Request,
    kinds: list[str] | None = Query(default=None),
) -> ResourcesResponse:
    """List Databricks resources available for Designer tool configuration."""
    obo_token = _obo_token_from_request(fastapi_request)
    discovery_adapter = DesignerDiscoveryAdapter(
        cast(_DiscoveryServiceProto, DiscoveryService())
    )
    resources = await discovery_adapter.list_for_user(
        user_token=obo_token,
        kinds=_parse_resource_kinds(kinds),
        user_id=user.user_id,
    )
    return ResourcesResponse(resources=resources, total=len(resources))


@router.post(
    "/resources/sql-warehouses/{warehouse_id}/start",
    response_model=DiscoveredResource,
)
async def start_sql_warehouse(
    warehouse_id: str,
    _user: CurrentUser,
    fastapi_request: Request,
) -> DiscoveredResource:
    """Start a stopped SQL warehouse selected in Designer tool configuration."""
    discovery_adapter = DesignerDiscoveryAdapter(
        cast(_DiscoveryServiceProto, DiscoveryService())
    )
    try:
        return await discovery_adapter.start_sql_warehouse(
            user_token=_obo_token_from_request(fastapi_request),
            warehouse_id=warehouse_id,
        )
    except Exception as exc:  # noqa: BLE001 - surface Databricks SDK failures to the UI
        logger.warning(
            "DESIGNER_SQL_WAREHOUSE_START_FAILED",
            extra={"warehouse_id": warehouse_id, "error": repr(exc)},
        )
        raise HTTPException(
            status_code=502,
            detail={
                "error_kind": "sql_warehouse_start_failed",
                "message": "Could not start SQL warehouse.",
            },
        ) from exc


@router.get("/registry", response_model=RegistryResponse)
async def get_registry(
    user: CurrentUser,
    session: AsyncSession = Depends(get_db),
) -> RegistryResponse:
    _t0 = time.monotonic()
    result = RegistryResponse(
        node_types=node_types_payload(),
        agent_subtypes=AGENT_SUBTYPES,
        tool_kinds=await tool_kinds_payload_with_custom(session=session, user_id=user.user_id),
        model_tiers=model_tiers_payload(),
        query_modes=query_modes_payload(),
        research_depths=research_depths_payload(),
        source_kinds=source_kinds_payload(),
        version=REGISTRY_VERSION,
    )
    record_registry_fetch((time.monotonic() - _t0) * 1000)
    return result


def _tool_declarations_from_definition(
    definition: dict[str, Any],
    *,
    tool_names: list[str] | None = None,
) -> list[ToolDeclaration]:
    by_name: dict[str, dict[str, Any]] = {}
    ordered_raw: list[dict[str, Any]] = []
    for raw in definition.get("tools") or []:
        if not isinstance(raw, dict):
            continue
        raw_name = raw.get("name")
        if not isinstance(raw_name, str):
            continue
        by_name.setdefault(raw_name, raw)
        ordered_raw.append(raw)

    selected_raw: list[dict[str, Any]]
    if tool_names:
        selected_raw = [by_name[name] for name in tool_names if name in by_name]
    else:
        selected_raw = ordered_raw

    declarations: list[ToolDeclaration] = []
    for raw in selected_raw:
        try:
            declarations.append(ToolDeclaration(**raw))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=422,
                detail=f"Invalid tool declaration for catalog probe: {exc}",
            ) from exc
    return declarations


def _probe_config_from_app(*, persist: bool) -> ProbeConfig:
    cfg = get_app_config().agent_designer.probe
    return ProbeConfig(
        timeout_seconds=cfg.timeout_seconds,
        max_concurrent_probes=cfg.max_concurrent_probes,
        max_output_chars=cfg.max_output_chars,
        persist=persist,
    )


@router.post("/refresh-catalog", response_model=RefreshCatalogResponse)
async def refresh_catalog(
    req: RefreshCatalogRequest,
    user: CurrentUser,
    session: AsyncSession = Depends(get_db),
) -> RefreshCatalogResponse:
    """Refresh Designer materialized tool catalog prose without persisting it."""
    await _verify_agent_edit_permission(req.agent_id, user=user, session=session)
    try:
        load_workflow_from_dict(req.definition)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid workflow definition: {exc}") from exc
    refreshed = CatalogService().materialize_for_save(
        req.definition,
        force_regen=req.force_regen,
    )
    return RefreshCatalogResponse(definition=refreshed)


@router.post("/probe-tools", response_model=ProbeToolsResponse)
async def probe_tools(
    req: ProbeToolsRequest,
    user: CurrentUser,
    session: AsyncSession = Depends(get_db),
) -> ProbeToolsResponse:
    """Run SafeProbe-only samples for declared tools, isolated per tool."""
    await _verify_agent_edit_permission(req.agent_id, user=user, session=session)
    try:
        load_workflow_from_dict(req.definition)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid workflow definition: {exc}") from exc
    declarations = _tool_declarations_from_definition(
        req.definition,
        tool_names=req.tool_names,
    )
    samples = await ProbeOrchestrator.from_default_factories(
        config=_probe_config_from_app(persist=req.persist),
    ).probe(
        declarations,
        ctx=ToolContext(query=req.user_query or "", read_only=True),
        user_query=req.user_query,
    )

    definition = dict(req.definition)
    if req.persist:
        by_name = {
            decl.name: sample.model_dump(mode="json")
            for decl, sample in zip(declarations, samples, strict=False)
        }
        tools = []
        for raw in req.definition.get("tools") or []:
            raw_name = raw.get("name") if isinstance(raw, dict) else None
            if isinstance(raw, dict) and isinstance(raw_name, str) and raw_name in by_name:
                next_raw = dict(raw)
                next_raw["probe"] = by_name[raw_name]
                tools.append(next_raw)
            else:
                tools.append(raw)
        definition = {**req.definition, "tools": tools}
        definition = CatalogService().materialize_for_save(definition, force_regen=True)

    return ProbeToolsResponse(samples=samples, definition=definition, persist=req.persist)


# ---------- /import-yaml ----------


class ImportYamlResponse(BaseModel):
    definition: dict[str, Any]
    workflow_summary: WorkflowSummary


@router.post("/import-yaml", response_model=ImportYamlResponse)
async def import_yaml(request: Request, _user: CurrentUser) -> ImportYamlResponse:
    """Parse and validate a raw YAML workflow document.

    The endpoint:
    1. Enforces a size limit (default 256 KiB).
    2. Parses the body via ``yaml.safe_load`` (never ``yaml.load``).
    3. Checks the ``registry_version`` field against the current registry.
    4. Re-validates the AST via ``load_workflow_from_dict`` — the canonical
       framework validator.

    On success the validated definition dict and a workflow summary are
    returned; no data is persisted by this endpoint.
    """
    max_bytes: int = int(
        os.environ.get("AGENT_DESIGNER_YAML_MAX_BYTES", str(256 * 1024))
    )
    body = await request.body()
    if len(body) > max_bytes:
        record_yaml_import_outcome("too_large")
        raise HTTPException(
            status_code=413,
            detail={"error_kind": "too_large", "max_bytes": max_bytes},
        )
    try:
        definition = parse_and_validate_yaml(body)
    except YamlImportError as exc:
        record_yaml_import_outcome(exc.error_kind)  # type: ignore[arg-type]
        raise HTTPException(
            status_code=413 if exc.error_kind == "too_large" else 400,
            detail={
                "errors": [
                    {
                        "path": exc.path,
                        "kind": exc.error_kind,
                        "message": exc.message,
                    }
                ]
            },
        ) from exc
    record_yaml_import_outcome("success")
    summary = WorkflowSummary(
        node_count=_node_count(definition.get("root", {})),
        tool_count=len(definition.get("tools", []) or []),
        source_count=0,
    )
    return ImportYamlResponse(definition=definition, workflow_summary=summary)


# ---------- /chat ----------


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: str  # "user" | "assistant" | "tool"
    content: str = ""
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    messages: list[ChatMessage] = Field(min_length=1)
    current_ast: dict[str, Any] | None = None
    session_id: str | None = None
    assets: list[DesignerAsset] = Field(default_factory=list)


def _format_sse(event_type: str, payload: dict[str, Any]) -> str:
    """Serialise a single SSE frame."""
    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"


@router.post("/chat")
async def chat(
    request: ChatRequest,
    user: CurrentUser,
    fastapi_request: Request,
) -> StreamingResponse:
    """Stream LLM chat with AST mutation tool-calls.

    Stateless: each call carries its own messages + current_ast.  The
    session_id is for log correlation only — no server-side state is looked up.

    Size limits are enforced BEFORE opening the stream so clients receive a
    plain HTTP 413 (not an SSE error event) when the payload is too large.
    """
    from deep_research.agent_designer.llm_adapter import AppLLMAdapter
    from deep_research.agent_designer.orchestrator import (
        DesignerChatOrchestrator,
        RequestTooLargeError,
    )
    from deep_research.core.trace_provenance import set_trace_provenance

    # Resolve OBO token for per-user data source scoping.
    obo_token: str = (
        getattr(fastapi_request.state, "obo_token", None)
        or fastapi_request.headers.get("x-forwarded-access-token", "")
    )

    # Provenance — every designer-chat trace must self-identify so it lands
    # alongside main-chat + shell-app traces in the shared experiment. The
    # agent_v2_id may be unknown on the first turn (before propose_workflow
    # mints it); helper drops empty/None values gracefully and a later
    # re-tag inside the orchestrator (when an agent is created/updated)
    # fills it in. ``query_preview`` is the last user message for grep.
    _latest_user_message = next(
        (m.content for m in reversed(request.messages) if m.role == "user"),
        "",
    )
    set_trace_provenance(
        surface="designer-chat",
        user_id=user.user_id,
        session_id=request.session_id,
        query_preview=str(_latest_user_message)[:200],
    )

    # Build orchestrator dependencies from app-level singletons.
    app_llm = fastapi_request.app.state.llm_client
    llm_adapter = AppLLMAdapter(app_llm)

    # cast: DiscoveryService is structurally compatible with _DiscoveryServiceProto
    # at runtime; the protocol uses **kwargs which mypy cannot verify against the
    # concrete signature's named optional parameters.
    discovery_svc = cast(_DiscoveryServiceProto, DiscoveryService())
    discovery_adapter = DesignerDiscoveryAdapter(discovery_svc)

    orchestrator = DesignerChatOrchestrator(llm_adapter, discovery_adapter)

    # Pre-flight size check — must happen BEFORE StreamingResponse is returned
    # so the client receives HTTP 413, not an SSE error frame.
    messages_dicts = [m.model_dump(exclude_none=True) for m in request.messages]
    try:
        orchestrator.check_limits(messages_dicts, request.current_ast, request.assets)
    except RequestTooLargeError as exc:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)) from exc

    async def event_generator() -> AsyncIterator[str]:
        try:
            async for event in orchestrator.run_turn(
                messages=messages_dicts,
                current_ast=request.current_ast,
                session_id=request.session_id,
                user_token=obo_token,
                current_user_id=user.user_id,
                assets=request.assets,
            ):
                yield _format_sse(event.type, event.model_dump(exclude={"type"}))
        except Exception:
            # Never echo raw exception text to clients — it can leak internal
            # class names, paths, or upstream error detail (codex finding W6).
            # The real exception is captured in structured server logs.
            logger.exception(
                "DESIGNER_CHAT_STREAM_FAILED",
                extra={"session_id": request.session_id, "user_id": user.user_id},
            )
            yield _format_sse(
                "error",
                {
                    "error_kind": "agent_error",
                    "message": "The designer chat failed. See server logs for details.",
                },
            )
            yield _format_sse("done", {})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# Rebuild models that reference `Any` via forward-ref string annotations
# (required when `from __future__ import annotations` is active).
ValidateRequest.model_rebuild()
ValidateResponse.model_rebuild()
RegistryResponse.model_rebuild()
ResourcesResponse.model_rebuild()
ImportYamlResponse.model_rebuild()
ChatMessage.model_rebuild()
ChatRequest.model_rebuild()
CreateCustomToolRequest.model_rebuild()
UpdateCustomToolRequest.model_rebuild()
CustomToolResponse.model_rebuild()
CustomToolListResponse.model_rebuild()
