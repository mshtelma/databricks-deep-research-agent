"""API routes for AIditor integration in Deep Research Agent."""

import logging
import re
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
from fastapi import APIRouter, HTTPException, Request

from deep_research.middleware.auth import CurrentUser

from .config import aiditor_conf as conf
from .models import (
    ChatRequest,
    ChatResponse,
    DebugInfo,
    ExternalConnectionInfo,
    ExternalQueryRequest,
    ExternalQueryResponse,
    GenieQueryRequest,
    GenieQueryResponse,
    GenieSpaceInfo,
    KnowledgeAssistantInfo,
    KnowledgeAssistantQueryRequest,
    KnowledgeAssistantQueryResponse,
    MCPEndpoints,
    ModelInfo,
    ModelsResponse,
    ModelStatus,
    UsageInfo,
    VectorIndexInfo,
    VectorSearchRequest,
    VectorSearchResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# System prompt for LLM editing
SYSTEM_PROMPT = """You are an expert editor. You will receive a JSON payload containing:
1. "originalMarkdown": The original markdown document
2. "edits": An array of edit instructions

Each edit has:
- "type": REMOVE (delete this text), LESS (shorten/reduce), MORE (expand/elaborate), or CUSTOM (follow the instruction)
- "text": The text to modify
- "instruction": (optional) For CUSTOM type, the specific instruction

Your task:
1. Apply all edit instructions to the original markdown
2. Preserve all markdown formatting
3. Return ONLY the edited markdown document
4. Do not include explanations or commentary

Apply changes precisely as instructed. Maintain document coherence and flow."""


def _get_workspace_client(request: Request | None) -> WorkspaceClient:
    """Return a WorkspaceClient, preferring the one set by auth middleware.

    Priority:
    1. request.state.workspace_client — set by CurrentUser dependency / auth middleware
    2. Config-based fallback (host+token, profile, or default SDK auth)

    NOTE: The auth middleware (CurrentUser) must be added as a dependency on endpoints
    that call this function so that request.state.workspace_client is populated.
    """
    # Try the middleware-provided client first
    if request:
        ws_client = getattr(request.state, "workspace_client", None)
        if ws_client is not None:
            return ws_client  # type: ignore[return-value]

    # Fallback: create client from config/env (development without auth middleware)
    # TODO: Integrate with project's LLMClient for rate limiting and health tracking
    if conf.databricks_host and conf.databricks_token:
        return WorkspaceClient(host=conf.databricks_host, token=conf.databricks_token, auth_type="pat")
    if conf.databricks_profile:
        return WorkspaceClient(profile=conf.databricks_profile)
    return WorkspaceClient()


def _get_sp_client() -> WorkspaceClient:
    """Return a WorkspaceClient using the app's service principal identity."""
    if conf.databricks_profile:
        return WorkspaceClient(profile=conf.databricks_profile)
    if conf.databricks_host and conf.databricks_token:
        return WorkspaceClient(host=conf.databricks_host, token=conf.databricks_token, auth_type="pat")
    return WorkspaceClient()


def _extract_chat_content(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if choices:
        first = choices[0]
        message = getattr(first, "message", None)
        if message and getattr(message, "content", None):
            return message.content
        text = getattr(first, "text", None)
        if text:
            return text
    predictions = getattr(response, "predictions", None)
    if predictions:
        first = predictions[0]
        if isinstance(first, dict):
            return first.get("content") or first.get("text") or str(first)
        return str(first)
    return ""


# =============================================================================
# LLM Endpoints
# =============================================================================

_FOUNDATION_MODELS: list[str] = [
    "databricks-claude-sonnet-4",
    "databricks-claude-sonnet-4-5",
    "databricks-claude-3-7-sonnet",
    "databricks-claude-haiku-4-5",
    "databricks-claude-opus-4-1",
    "databricks-claude-opus-4-5",
    "databricks-claude-opus-4-6",
    "databricks-gpt-5",
    "databricks-gpt-5-1",
    "databricks-gpt-5-mini",
    "databricks-gpt-5-nano",
    "databricks-meta-llama-3-3-70b-instruct",
    "databricks-meta-llama-3-1-405b-instruct",
    "databricks-meta-llama-3-1-8b-instruct",
    "databricks-llama-4-maverick",
]


@router.get("/models", response_model=ModelsResponse, operation_id="aiditorListModels")
async def list_models(_request: Request, _user: CurrentUser) -> ModelsResponse:
    """List available LLM serving endpoints for AIditor."""
    models = [
        ModelInfo(name=name, display_name=name, status=ModelStatus.READY, task="llm/v1/chat")
        for name in _FOUNDATION_MODELS
    ]
    return ModelsResponse(models=models, default_model=conf.default_model)


@router.post("/chat", response_model=ChatResponse, operation_id="aiditorProcessEdits")
async def process_edits(request: ChatRequest, http_request: Request, _user: CurrentUser) -> ChatResponse:
    """Process markdown with edit instructions using LLM."""
    edits_payload = [
        {"type": edit.type.value, "text": edit.text, "instruction": edit.instruction}
        for edit in request.edits
    ]

    user_message = f"""Please apply the following edits to the markdown document:

Original Markdown:
```markdown
{request.original_markdown}
```

Edits to apply:
{edits_payload}

Return only the edited markdown document."""

    client = _get_workspace_client(http_request)
    response = client.serving_endpoints.query(
        request.model,
        messages=[
            ChatMessage(role=ChatMessageRole.SYSTEM, content=SYSTEM_PROMPT),
            ChatMessage(role=ChatMessageRole.USER, content=user_message),
        ],
        temperature=0.2,
        max_tokens=2048,
    )

    raw_content = _extract_chat_content(response)
    if not raw_content:
        raise HTTPException(status_code=502, detail="No content returned from model endpoint")

    processed_content = re.sub(r"^```(?:markdown)?\s*\n?", "", raw_content)
    processed_content = re.sub(r"\n?```\s*$", "", processed_content)

    usage = getattr(response, "usage", None)
    usage_info = None
    if usage:
        usage_info = UsageInfo(
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
            total_tokens=getattr(usage, "total_tokens", None),
        )

    debug_info = DebugInfo(system_prompt=SYSTEM_PROMPT, user_message=user_message, raw_response=raw_content)
    return ChatResponse(content=processed_content, model=request.model, usage=usage_info, debug=debug_info)


# =============================================================================
# MCP Endpoints
# =============================================================================

_DEFAULT_GENIE_SPACES: list[dict[str, str]] = []
_DEFAULT_VECTOR_INDEXES: list[dict[str, str]] = []
_DEFAULT_EXTERNAL_CONNECTIONS: list[dict[str, str]] = [
    {"name": "mcp-tavily-can", "type": "tavily"},
]
_DEFAULT_KNOWLEDGE_ASSISTANTS: list[dict[str, str]] = [
    {
        "tile_id": "112212ba",
        "name": "Gas Station Retail Operations KA",
        "endpoint_name": "ka-112212ba-endpoint",
    },
]


@router.get("/mcp/endpoints", response_model=MCPEndpoints, operation_id="aiditorListMCPEndpoints")
async def list_mcp_endpoints(request: Request) -> MCPEndpoints:
    """List available MCP endpoints (Genie Spaces, Vector Search, External, KA)."""
    default_genie = [
        GenieSpaceInfo(id=g["id"], name=g["name"], tables=[], status="active")
        for g in _DEFAULT_GENIE_SPACES
    ]
    default_vector = [
        VectorIndexInfo(name=v["name"], endpoint=v["endpoint"], num_docs=0)
        for v in _DEFAULT_VECTOR_INDEXES
    ]
    default_external = [
        ExternalConnectionInfo(name=e["name"], type=e["type"], status="active")
        for e in _DEFAULT_EXTERNAL_CONNECTIONS
    ]
    default_ka = [
        KnowledgeAssistantInfo(
            tile_id=ka["tile_id"], name=ka["name"],
            endpoint_name=ka.get("endpoint_name", ""), status="active",
        )
        for ka in _DEFAULT_KNOWLEDGE_ASSISTANTS
    ]

    if not conf.enable_auto_discovery:
        return MCPEndpoints(
            genie_spaces=default_genie, vector_indexes=default_vector,
            external_connections=default_external, knowledge_assistants=default_ka,
        )

    import asyncio

    default_genie_ids = {g["id"] for g in _DEFAULT_GENIE_SPACES}
    default_ka_ids = {ka["tile_id"] for ka in _DEFAULT_KNOWLEDGE_ASSISTANTS}

    try:
        ws_client = _get_workspace_client(request)
    except Exception as exc:
        logger.error("Failed to create workspace client: %s", exc)
        return MCPEndpoints(
            genie_spaces=default_genie, vector_indexes=default_vector,
            external_connections=default_external, knowledge_assistants=default_ka,
        )

    def _sync_genie() -> list:
        try:
            response = ws_client.genie.list_spaces()
            spaces = response.spaces or []
            return [
                GenieSpaceInfo(id=s.space_id, name=s.title, tables=[], status="active")
                for s in spaces if s.space_id not in default_genie_ids
            ]
        except Exception as exc:
            logger.warning("Genie discovery failed: %s", exc)
            return []

    def _sync_ka() -> list:
        try:
            endpoints = list(ws_client.serving_endpoints.list())
            ka_list = []
            for ep in endpoints:
                name = ep.name or ""
                if name.startswith("ka-") or "knowledge" in name.lower():
                    tile_id = name.replace("ka-", "").replace("-endpoint", "")
                    ready_status: Any = ep.state.ready if ep.state else None
                    if hasattr(ready_status, "value"):
                        ready_status = ready_status.value
                    status = "active" if ready_status == "READY" else "pending"
                    ka_list.append(
                        KnowledgeAssistantInfo(tile_id=tile_id, name=name, endpoint_name=name, status=status)
                    )
            return [ka for ka in ka_list if ka.tile_id not in default_ka_ids]
        except Exception as exc:
            logger.warning("KA discovery failed: %s", exc)
            return []

    async def _safe(label: str, fn, timeout: float = 10):
        try:
            return await asyncio.wait_for(asyncio.to_thread(fn), timeout=timeout)
        except TimeoutError:
            logger.warning("%s discovery timed out after %.0fs", label, timeout)
            return []
        except Exception as exc:
            logger.warning("%s discovery failed: %s", label, exc)
            return []

    extra_genie, extra_ka = await asyncio.gather(
        _safe("Genie", _sync_genie, timeout=10),
        _safe("KA", _sync_ka, timeout=10),
    )

    return MCPEndpoints(
        genie_spaces=default_genie + extra_genie,
        vector_indexes=default_vector,
        external_connections=default_external,
        knowledge_assistants=default_ka + extra_ka,
    )


@router.post("/mcp/genie", response_model=GenieQueryResponse, operation_id="aiditorQueryGenie")
async def query_genie(request: GenieQueryRequest, http_request: Request) -> GenieQueryResponse:
    """Query a Genie Space with natural language."""
    from .mcp.genie import GenieClient

    client = _get_sp_client()
    genie_client = GenieClient(client)
    response = await genie_client.query(request.space_id, request.query)
    if response.error:
        raise HTTPException(status_code=502, detail=response.error)
    return response


@router.post("/mcp/vector-search", response_model=VectorSearchResponse, operation_id="aiditorQueryVectorSearch")
async def query_vector_search(request: VectorSearchRequest, http_request: Request) -> VectorSearchResponse:
    """Query a Vector Search index."""
    from .mcp.vector import VectorSearchClient

    client = _get_sp_client()
    vector_client = VectorSearchClient(client)
    response = await vector_client.search(request.index_name, request.query, request.num_results)
    if response.error:
        raise HTTPException(status_code=502, detail=response.error)
    return response


@router.post("/mcp/external", response_model=ExternalQueryResponse, operation_id="aiditorQueryExternal")
async def query_external(request: ExternalQueryRequest, http_request: Request) -> ExternalQueryResponse:
    """Query an external MCP endpoint (e.g., Tavily)."""
    from .mcp.external import ExternalClient

    client = _get_sp_client()
    external_client = ExternalClient(client)
    response = await external_client.search(request.connection_name, request.query, request.max_results)
    if response.error:
        raise HTTPException(status_code=502, detail=response.error)
    return response


@router.post("/mcp/knowledge-assistant", response_model=KnowledgeAssistantQueryResponse, operation_id="aiditorQueryKA")
async def query_knowledge_assistant(
    request: KnowledgeAssistantQueryRequest, http_request: Request
) -> KnowledgeAssistantQueryResponse:
    """Query a Knowledge Assistant endpoint."""
    try:
        client = _get_sp_client()
    except Exception as exc:
        logger.error("Failed to create SP client for KA query: %s", exc)
        return KnowledgeAssistantQueryResponse(status="error", answer="", error=f"Failed to initialize client: {exc}")

    try:
        endpoint_name = f"ka-{request.tile_id}-endpoint"
        raw: dict = client.api_client.do(  # type: ignore[assignment]
            "POST",
            f"/serving-endpoints/{endpoint_name}/invocations",
            body={"input": [{"role": "user", "content": request.query}]},
        )
        data: dict = raw if isinstance(raw, dict) else {}

        answer = ""
        sources: list[str] = []
        output = data.get("output", [])
        for item in output:
            if isinstance(item, dict) and item.get("type") == "message" and item.get("role") == "assistant":
                content_items = item.get("content", [])
                for c in content_items:
                    if isinstance(c, dict) and c.get("type") == "output_text":
                        answer = c.get("text", "")

        return KnowledgeAssistantQueryResponse(status="success", answer=answer, sources=sources)

    except Exception as exc:
        logger.warning("KA query failed: %s", exc)
        return KnowledgeAssistantQueryResponse(status="error", answer="", error=str(exc))
