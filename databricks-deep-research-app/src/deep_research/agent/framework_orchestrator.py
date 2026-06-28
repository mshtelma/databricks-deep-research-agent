"""Framework-based orchestrator — thin wrapper around databricks-deep-research.

Replaces the 3769 LOC monolith orchestrator with a clean delegation to the
multi-agent framework.  The pipeline is:

    config_translator.translate(config) → WorkflowDefinition
    build_app_workflow_runner(...).stream(definition, ...) → yields StreamEvent
    DomainContextTracker.process_event(event) → list[AppSSEEvent]
    PersistenceDelta → DB writes

All app-specific concerns (persistence, SSE format, cancellation, error
handling) are handled here.  Tool context construction lives in
``workflow_runner_factory.py``.  The framework handles workflow execution.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
import traceback
from collections.abc import AsyncGenerator, Callable
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from databricks_deep_research.events.types import (
    AgentStreamChunkEvent,
    CoordinatorClassifiedEvent,
    ItemCompletedEvent,
    ReplanTriggeredEvent,
)
from databricks_deep_research.events.types import (
    WorkflowCompletedEvent as FwkWorkflowCompletedEvent,
)
from databricks_deep_research.memory import CHAT_MEMORY_APPENDIX_STATE_KEY
from databricks_deep_research.workflow.context import (
    ExecutionContext,
)
from databricks_deep_research.workflow.definition import (
    NodeType,
    WorkflowDefinition,
    WorkflowNode,
)
from databricks_deep_research.workflow.loader import load_workflow_from_dict
from databricks_deep_research.workflow.state import WorkflowState

from deep_research.agent.adapters.config_translator import translate
from deep_research.agent.adapters.domain_context import (
    AppSSEEvent,
    DomainContextTracker,
)
from deep_research.agent.adapters.llm_adapter import create_framework_llm_client
from deep_research.agent.adapters.tool_adapter import create_framework_tools
from deep_research.agent.chat_title import derive_chat_title_from_query
from deep_research.agent.followup import (
    LiveSearchUnavailable,
    TurnIntent,
    decide_turn_intent,
    stream_chat_about_results,
    stream_live_search_answer,
)
from deep_research.agent.tools.file_entities import GetFileEntitiesTool
from deep_research.agent.tools.list_files import ListAttachedFilesTool
from deep_research.agent.tools.read_file import ReadAttachedFileTool
from deep_research.agent.tools.search_chat_memory import SearchChatMemoryTool
from deep_research.agent.workflow_runner_factory import build_app_workflow_runner
from deep_research.core.tracing import safe_mlflow_run, safe_tool_span, safe_update_trace
from deep_research.plugins.base import ContextEnricher
from deep_research.schemas.streaming import (
    AgentCompletedEvent,
    AgentStartedEvent,
    PersistenceCompletedEvent,
    ResearchCompletedEvent,
    ResearchStartedEvent,
    StreamErrorEvent,
    StreamEvent,
    SynthesisProgressEvent,
    SynthesisStartedEvent,
)
from deep_research.services._impl_factory import make_chat_memory_service, make_file_upload_service
from deep_research.services._protocols import IChatMemoryService
from deep_research.services.chat_memory_service import APPENDIX_AGENT_TYPE
from deep_research.services.llm.client import LLMClient
from deep_research.services.llm.embedder import DEFAULT_EMBEDDING_ENDPOINT
from deep_research.services.research_event_buffer import EventBuffer
from deep_research.services.search.brave import BraveSearchClient

try:
    from mlflow.entities import SpanType as _SpanType
except ImportError:
    _SpanType = None  # type: ignore[misc, assignment]

if TYPE_CHECKING:
    from deep_research.agent.orchestration_config import OrchestrationConfig
    from deep_research.agent.tools.web_crawler import WebCrawler

try:
    from sqlalchemy.ext.asyncio import AsyncSession
except ImportError:
    AsyncSession = None  # type: ignore[misc, assignment]

try:
    from deep_research.plugins.manager import PluginManager
except ImportError:
    PluginManager = None  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)


class MissingOBOTokenError(RuntimeError):
    """Raised when a Databricks-bound workflow runs under the Apps runtime
    without an on-behalf-of user token.

    Surfaced to the caller as a streamed error (via the orchestrator's
    error handler) so the request fails fast with a clear message instead
    of silently running every Databricks tool as the app service principal
    and returning UC permission errors / empty results.
    """


# Regex matching [N] numeric citation markers produced by the framework synthesizer.
# Captures the integer N.  Matches [1], [12], [1][2], etc.
_NUMERIC_CITATION_RE = __import__("re").compile(r"\[(\d+)\]")


def _resolve_turn_intent(config: OrchestrationConfig) -> TurnIntent:
    """Resolve the requested per-turn routing for a custom-agent chat.

    Back-compat: the legacy ``query_mode == "simple"`` selection on a
    custom-agent chat historically meant "answer directly, don't run the full
    workflow" — which the custom-agent path silently ignored. Treat it as an
    explicit chat turn so that selection regains meaning. Otherwise honor the
    explicit ``turn_intent`` (defaulting to AUTO on any unknown value).
    """
    raw = (getattr(config, "turn_intent", "auto") or "auto").lower()
    # An explicit per-turn intent from the client (the chat UI's Auto/Chat/
    # Research control — now surfaced in every mode for agent chats, not just
    # Deep Research) wins over the legacy ``query_mode == "simple"`` → chat
    # shorthand. So Simple + an explicit "research" re-runs the agent instead of
    # being silently forced into a chat turn.
    if raw in ("chat", "research"):
        try:
            return TurnIntent(raw)
        except ValueError:
            pass
    # Back-compat: legacy clients that only sent ``query_mode`` used Simple on a
    # custom-agent chat to mean "answer from gathered data, don't re-run".
    if config.query_mode == "simple" and config.agent_id:
        return TurnIntent.CHAT
    try:
        return TurnIntent(raw)
    except ValueError:
        return TurnIntent.AUTO


def _definition_has_skills(definition: Any) -> bool:
    """True if any agent node in the workflow declares attached skills.

    Gates the (identity-lookup-touching) runtime skill-store build so skill-less
    runs stay byte-identical: no ``current_user.me()`` call, no store wired.
    """
    stack = [definition.root]
    while stack:
        node = stack.pop()
        cfg = getattr(node, "config", None)
        if isinstance(cfg, dict) and cfg.get("skills"):
            return True
        stack.extend(getattr(node, "children", None) or [])
    return False


def _merge_chat_attachments(
    definition: Any,
    enabled_skills: list[str] | None,
    enabled_mcp_servers: list[str] | None,
) -> None:
    """Merge chat-selected skills + MCP servers into *definition* in place (E1).

    Skills are appended to every agent config (top-level agent nodes plus
    plan_and_execute planner/evaluator/body agent dicts) so the runtime skill
    store + ``read_skill`` pick them up. MCP server names become Databricks
    UC-connection ``MCPServerConfig`` entries on ``definition.mcp_servers``
    (deduped). No-op when both lists are empty => byte-identical to today.
    """
    def _walk_agent_configs(value: Any) -> list[dict[str, Any]]:
        """Return mutable agent config dicts from raw configs or node-shaped dicts."""
        configs: list[dict[str, Any]] = []

        if isinstance(value, WorkflowNode):
            configs.extend(_walk_agent_configs(value.config))
            for child in value.children:
                configs.extend(_walk_agent_configs(child))
            return configs

        if not isinstance(value, dict):
            return configs

        # Raw AgentNodeConfig shape.
        if "subtype" in value:
            configs.append(value)

        # WorkflowNode dict shape: {"type": "agent", "config": {"subtype": ...}}.
        nested_config = value.get("config")
        if isinstance(nested_config, dict):
            configs.extend(_walk_agent_configs(nested_config))

            # Plan-and-execute config nested inside a node-shaped dict.
            for nested_key in ("planner", "evaluator", "body"):
                configs.extend(_walk_agent_configs(nested_config.get(nested_key)))

        # Plan-and-execute config raw shape.
        for nested_key in ("planner", "evaluator", "body"):
            configs.extend(_walk_agent_configs(value.get(nested_key)))

        for child in value.get("children") or []:
            configs.extend(_walk_agent_configs(child))

        return configs

    def _iter_definition_agent_configs() -> list[dict[str, Any]]:
        return _walk_agent_configs(definition.root)

    if enabled_skills:

        def _add_skills(agent_cfg: dict[str, Any]) -> None:
            merged = list(agent_cfg.get("skills") or [])
            for skill in enabled_skills:
                if skill and skill not in merged:
                    merged.append(skill)
            agent_cfg["skills"] = merged

        for agent_cfg in _iter_definition_agent_configs():
            _add_skills(agent_cfg)

    if enabled_mcp_servers:
        from databricks_deep_research import MCPServerConfig

        servers = list(definition.mcp_servers or [])
        seen = {getattr(s, "name", None) for s in servers}
        for name in enabled_mcp_servers:
            if name and name not in seen:
                servers.append(
                    MCPServerConfig(
                        name=name, client_kind="databricks", connection_name=name
                    )
                )
                seen.add(name)
        definition.mcp_servers = servers

        # Bind the chat-attached servers to the RESEARCHER agents so the runtime
        # MCP auto-attach (executor.maybe_attach_mcp) makes their tools callable.
        # Scope = researcher lanes only (the agents that gather evidence via
        # tools); synthesizer / reflector / coordinator stay clean. Designer-bound
        # agents already carry their own ``config.mcp_servers`` at author time.
        _attach_names = [n for n in enabled_mcp_servers if n]

        def _add_mcp(agent_cfg: dict[str, Any]) -> None:
            if str(agent_cfg.get("subtype", "")) != "researcher":
                return
            merged = list(agent_cfg.get("mcp_servers") or [])
            for mcp_name in _attach_names:
                if mcp_name not in merged:
                    merged.append(mcp_name)
            agent_cfg["mcp_servers"] = merged

        for agent_cfg in _iter_definition_agent_configs():
            _add_mcp(agent_cfg)


def _stash_mcp_tools_by_server(
    runner: Any,
    tool_resolver: Any,
    tools_by_server: dict[str, list[Any]],
) -> None:
    """Stash discovered MCP tools in every factory context used at execution.

    ``ToolResolver`` intentionally shallow-copies ``ToolFactoryContext.extras``
    at construction time to isolate resolver caches. MCP discovery happens
    after resolver construction, so writing only to ``runner.factory_context``
    leaves the resolver's copied context without ``_mcp_tools_by_server`` and
    makes researcher auto-attach see zero tools.
    """
    for owner in (runner, tool_resolver):
        factory_context = getattr(owner, "factory_context", None)
        extras = getattr(factory_context, "extras", None)
        if isinstance(extras, dict):
            extras["_mcp_tools_by_server"] = tools_by_server


async def _load_skill_folder_roots(
    db: AsyncSession | None, user_id: str | None
) -> list[str]:
    """Return the user's registered skill-folder roots (A3); fail-soft to []."""
    if not user_id:
        return []
    try:
        from deep_research.services.skill_folder_store import load_user_skill_roots

        if db is not None:
            return await load_user_skill_roots(db, user_id)
        from deep_research.db.session import get_session_maker

        session_maker = get_session_maker()
        async with session_maker() as session:
            return await load_user_skill_roots(session, user_id)
    except Exception:  # noqa: BLE001 — fail-soft; skills still work without extras
        logger.warning(
            "SKILL_FOLDER_ROOTS_UNAVAILABLE user=%s — continuing with built-in roots",
            user_id,
            exc_info=True,
        )
        return []


async def _maybe_prepend_cross_session_memory(
    conversation_history: list[dict[str, str]] | None,
    config: OrchestrationConfig,
    db: AsyncSession | None,
    user_id: str | None,
    chat_id: str | None,
) -> list[dict[str, str]] | None:
    """Prepend a spotlighted role=user cross-session-memory message when enabled.

    Flag-gated (``cross_session_memory_enabled``, default OFF) and fully
    fail-soft: returns ``conversation_history`` unchanged when the flag is off,
    no session is available, or anything goes wrong. The injected message rides
    the existing role=user channel (DeerFlow role-split — remembered facts are
    untrusted DATA). The read itself is bounded + guarded inside
    ``inject_cross_session_memory``; this wrapper additionally guards store
    construction so a config/import problem can never break the run.
    """
    try:
        from deep_research.core.config import get_settings as _get_settings_csm

        _settings = _get_settings_csm()
        # Per-run override (P2): config value wins when set; else inherit global.
        _csm_enabled = config.enable_cross_session_memory
        if _csm_enabled is None:
            _csm_enabled = _settings.cross_session_memory_enabled
        if not _csm_enabled or not user_id:
            return conversation_history
        # Default store is the legacy SQLAlchemy findings reader; without a
        # session we degrade to no-memory (cached-path cross-chat read is a
        # follow-up). Byte-identical to flag-off in that case.
        if db is None:
            return conversation_history

        from deep_research.agent.cross_session_memory import (
            ChatMemoryFindingsStore,
            inject_cross_session_memory,
        )

        exclude_chat_id = _to_uuid(chat_id) if chat_id is not None else None
        message = await inject_cross_session_memory(
            store=ChatMemoryFindingsStore(db),
            user_id=user_id,
            agent_id=config.agent_id,
            exclude_chat_id=exclude_chat_id,
            min_confidence=_settings.cross_session_memory_min_confidence,
            max_facts=_settings.cross_session_memory_max_facts,
            timeout_seconds=_settings.cross_session_memory_timeout_sec,
        )
    except Exception:
        # Fail-soft invariant: memory NEVER breaks a research run.
        logger.exception(
            "CROSS_SESSION_MEMORY_INJECT_GUARD_FAILED user_id=%s — continuing without memory",
            user_id,
        )
        return conversation_history

    if message is None:
        return conversation_history
    return [message, *(conversation_history or [])]


def _resolve_workflow(
    config: OrchestrationConfig,
    tool_names: list[str],
    plugin_manager: PluginManager | None,
) -> WorkflowDefinition:
    """Resolve workflow: plugin YAML when workflow_ref is set, else config_translator.

    When workflow_ref is None or empty, calls translate() (zero behavioral change).
    When set, iterates WorkflowProviderPlugin plugins for YAML content.
    If no plugin resolves the ref, raises ValueError (strict — no silent fallback).
    """
    ref = config.workflow_ref
    if not ref:
        return translate(config, available_tools=tool_names)

    from databricks_deep_research import load_workflow_from_string
    from databricks_deep_research.workflow.loader import (
        load_workflow_from_dict,
    )

    from deep_research.plugins.base import WorkflowProviderPlugin

    if plugin_manager is not None:
        for plugin in plugin_manager.get_plugins():
            if not isinstance(plugin, WorkflowProviderPlugin):
                continue

            # Scope 1: plugin lookup — failures are non-fatal (try next plugin)
            try:
                result = plugin.get_workflow_yaml(ref)
            except Exception:
                logger.exception(
                    "WORKFLOW_PLUGIN_LOOKUP_ERROR ref=%s plugin=%s",
                    ref,
                    getattr(plugin, "name", type(plugin).__name__),
                )
                continue  # try next plugin

            if result is None:
                continue  # this plugin doesn't own the ref

            logger.info(
                "WORKFLOW_PLUGIN_RESOLVED ref=%s plugin=%s type=%s",
                ref,
                getattr(plugin, "name", type(plugin).__name__),
                type(result).__name__,
            )

            # Scope 2: YAML/dict loading — failures ARE fatal (plugin claimed the ref)
            try:
                if isinstance(result, WorkflowDefinition):
                    defn = result
                elif isinstance(result, dict):
                    defn = load_workflow_from_dict(result)
                else:
                    defn = load_workflow_from_string(result)
            except Exception as exc:
                raise ValueError(
                    f"Plugin {getattr(plugin, 'name', type(plugin).__name__)!r} claimed "
                    f"ref={ref!r} but returned unparseable workflow: {exc}"
                ) from exc
            _filter_workflow_tools(defn, tool_names)
            return defn

    raise ValueError(
        f"workflow_ref={ref!r} set but no plugin resolved it. "
        f"Registered plugins: {[p.name for p in (plugin_manager.get_plugins() if plugin_manager else [])]}"
    )


async def _resolve_agent_v2_workflow(
    config: OrchestrationConfig,
    user_id: str | None,
    db: Any | None,
) -> WorkflowDefinition | None:
    """Load a selected Agent V2 workflow definition, if requested."""
    if not config.agent_id:
        return None
    if not user_id:
        raise ValueError("agent_id was provided but no user_id is available for visibility checks")

    try:
        agent_uuid = UUID(config.agent_id)
    except ValueError as exc:
        raise ValueError(f"Invalid agent_id {config.agent_id!r}") from exc

    from deep_research.services.agent_v2_service import AgentV2Service

    if db is not None:
        agent = await AgentV2Service(db).get_for_user(agent_uuid, user_id)
    else:
        from deep_research.db.session import get_session_maker

        session_maker = get_session_maker()
        async with session_maker() as session:
            agent = await AgentV2Service(session).get_for_user(agent_uuid, user_id)

    if agent is None:
        raise ValueError(f"Agent {config.agent_id!r} was not found or is not visible to the user")

    workflow_def = load_workflow_from_dict(agent.definition)
    logger.info(
        "FWK_AGENT_V2_WORKFLOW_RESOLVED agent_id=%s workflow_id=%s workflow_name=%s",
        config.agent_id,
        workflow_def.id,
        workflow_def.name,
    )
    return workflow_def


def _visit_workflow_agent_configs(
    workflow_def: WorkflowDefinition,
    visitor: Callable[[dict[str, Any]], None],
) -> None:
    """Visit agent configs, including agent configs nested inside raw config bodies."""

    def visit_raw(raw_node: dict[str, Any]) -> None:
        config_dict = raw_node.get("config")
        if isinstance(config_dict, dict):
            if raw_node.get("type") == NodeType.agent.value:
                visitor(config_dict)
            elif raw_node.get("type") == NodeType.plan_and_execute.value:
                for nested_key in ("planner", "evaluator"):
                    nested = config_dict.get(nested_key)
                    if isinstance(nested, dict):
                        visitor(nested)
                body = config_dict.get("body")
                if isinstance(body, dict):
                    visit_raw(body)
        for child in raw_node.get("children") or []:
            if isinstance(child, dict):
                visit_raw(child)

    def visit_node(node: Any) -> None:
        if node.type == NodeType.agent:
            visitor(node.config)
        elif node.type == NodeType.plan_and_execute:
            for nested_key in ("planner", "evaluator"):
                nested = node.config.get(nested_key)
                if isinstance(nested, dict):
                    visitor(nested)
            body = node.config.get("body")
            if isinstance(body, dict):
                visit_raw(body)
        for child in node.children:
            visit_node(child)

    visit_node(workflow_def.root)


def _apply_runtime_overlays_to_workflow(
    workflow_def: WorkflowDefinition,
    config: OrchestrationConfig,
) -> WorkflowDefinition:
    """Apply per-run chat toggles that are orthogonal to saved workflow shape."""
    if not config.verify_sources:
        return workflow_def

    def maybe_enable_reclaim(agent_config: dict[str, Any]) -> None:
        if agent_config.get("subtype") != "synthesizer":
            return
        output_schema = agent_config.get("output_schema")
        has_legacy_grounding = (
            isinstance(output_schema, dict)
            and output_schema.get("synthesis_mode") in {"reclaim", "interleaved"}
        )
        has_explicit_grounding = agent_config.get("grounding_mode") in {
            "none",
            "classical_lite",
            "reclaim",
        }
        if has_explicit_grounding or has_legacy_grounding:
            return
        schema = output_schema if isinstance(output_schema, dict) else {}
        schema.setdefault("synthesis_mode", "reclaim")
        schema.setdefault("enable_citation_verification", True)
        agent_config["output_schema"] = schema

    _visit_workflow_agent_configs(workflow_def, maybe_enable_reclaim)
    return workflow_def


def _tool_names_with_explicit_provider(workflow_def: WorkflowDefinition) -> set[str]:
    """Tool names whose YAML declaration sets an explicit ``config.provider``.

    Such tools must resolve via the factory chain (honoring the declared provider)
    rather than being shadowed by an auto-injected resolver override —
    ``ToolResolver.resolve()`` checks overrides BEFORE declarations, so without this
    a per-workflow ``provider: databricks`` on ``web_search`` would be silently
    ignored in favor of the app-default backend.
    """
    names: set[str] = set()
    for tool in workflow_def.tools or []:
        config = getattr(tool, "config", None)
        if isinstance(config, dict) and config.get("provider"):
            names.add(tool.name)
    return names


# Web-search tool kinds whose backend is provider-selectable. Mirrors
# ``ast_normalizer._WEB_PROVIDER_TOOL_KINDS`` / ``registry._WEB_PROVIDER_TOOL_KINDS``.
_WEB_PROVIDER_TOOL_KINDS: frozenset[str] = frozenset({"web_search", "web_research"})


def _fill_databricks_tool_defaults(
    workflow_def: WorkflowDefinition | None, search_cfg: Any
) -> None:
    """Fill app-config Databricks defaults onto per-tool databricks web tools.

    Runs on EVERY app execution (main chat + UI-saved + hand-written YAML), not
    only designer-normalized ASTs — a pure UI save bypasses ``normalize_ast``, so
    without this a ``provider: databricks`` web tool that omits ``model`` would
    reach the factory with no serving endpoint and raise. For a web tool whose
    effective provider is databricks (explicit ``config.provider == 'databricks'``
    or absent while the global default is databricks), the ``search.databricks``
    endpoint/tuning is filled in when absent. Idempotent on configs the designer
    normalizer already filled.
    """
    if workflow_def is None:
        return
    from deep_research.core.app_config import (
        DEFAULT_SEARCH_PROVIDER,
        fill_databricks_search_defaults,
        resolve_effective_provider,
    )

    global_provider = getattr(search_cfg, "provider", DEFAULT_SEARCH_PROVIDER)
    for tool in workflow_def.tools or []:
        if getattr(tool, "kind", None) not in _WEB_PROVIDER_TOOL_KINDS:
            continue
        config = getattr(tool, "config", None)
        if not isinstance(config, dict):
            continue
        effective = resolve_effective_provider(config.get("provider"), global_provider)
        if effective != "databricks":
            continue
        min_results = 0
        for count_key in ("total_results", "max_results"):
            raw = config.get(count_key)
            if isinstance(raw, int) and raw > min_results:
                min_results = raw
        if fill_databricks_search_defaults(
            config, search_cfg.databricks, min_results=min_results
        ):
            logger.info(
                "FWK_DBX_SEARCH_DEFAULTS tool=%s model=%s",
                getattr(tool, "name", "?"),
                config.get("model"),
            )
        # Tripwire: a model_family that contradicts the (now-resolved) endpoint
        # is a guaranteed 400 (e.g. openai family on a Gemini endpoint => OpenAI
        # Responses API on a Gemini serving endpoint). Save-time validation
        # blocks newly-authored ones; this catches legacy / hand-written-YAML /
        # API-imported configs. Shout loudly — built-in search would otherwise
        # silently return zero results for every query.
        model = config.get("model")
        family = config.get("model_family")
        if isinstance(model, str) and model and isinstance(family, str) and family:
            detected = search_cfg.databricks.family_for_endpoint(model)
            if detected is not None and detected != family:
                logger.error(
                    "DBX_SEARCH_FAMILY_ENDPOINT_MISMATCH tool=%s endpoint=%s "
                    "declared_family=%s detected_family=%s — built-in web search "
                    "will fail (wrong API for this endpoint); clear model_family "
                    "or choose a matching endpoint.",
                    getattr(tool, "name", "?"),
                    model,
                    family,
                    detected,
                )


async def _resolve_and_prepare_workflow_def(
    workflow_def: WorkflowDefinition | None,
    config: OrchestrationConfig,
    user_id: str | None,
    db: Any,
    tool_names: list[str],
    plugin_manager: Any,
    search_cfg: Any,
) -> WorkflowDefinition:
    """Resolve the workflow definition and prepare it for tool resolution.

    Single owner of the resolve → runtime-overlay → source-scope →
    databricks-fill sequence. The databricks web-search fill MUST stay coupled
    to resolution here: a designer/UI-saved (agent-v2) workflow whose web tool
    pins ``provider: databricks`` but omits ``model`` only becomes
    self-describing once resolved. Filling before the workflow is bound (while
    it is ``None``) is a silent no-op, and the tool then crashes construction
    with 'requires a serving endpoint'.
    """
    if workflow_def is None:
        workflow_def = (
            await _resolve_agent_v2_workflow(config, user_id, db)
            or _resolve_workflow(config, tool_names, plugin_manager)
        )
    workflow_def = _apply_runtime_overlays_to_workflow(workflow_def, config)
    workflow_def = _apply_source_scope_to_workflow_declarations(workflow_def, config)
    _fill_databricks_tool_defaults(workflow_def, search_cfg)
    return workflow_def


def _apply_source_scope_to_workflow_declarations(
    workflow_def: WorkflowDefinition,
    config: OrchestrationConfig,
) -> WorkflowDefinition:
    """Prevent saved workflow declarations from bypassing per-run source scope."""
    if config.source_scope not in {"enterprise_only", "web_only"}:
        return workflow_def

    web_kinds = {"web_search", "web_crawl", "brave_search"}
    enterprise_kinds = {
        "vector_search",
        "genie",
        "knowledge_assistant",
        "sql_analytics",
        "qa_assistant",
        "mcp",
        "mcp_server",
    }
    blocked_tool_names: set[str] = set()
    blocked_mcp_server_names: set[str] = set()
    kept_declarations = []
    for tool in workflow_def.tools:
        kind = getattr(tool.kind, "value", str(tool.kind))
        if config.source_scope == "enterprise_only" and kind in web_kinds:
            blocked_tool_names.add(tool.name)
            continue
        if config.source_scope == "web_only" and kind in enterprise_kinds:
            blocked_tool_names.add(tool.name)
            continue
        kept_declarations.append(tool)

    if config.source_scope == "web_only":
        mcp_servers = list(getattr(workflow_def, "mcp_servers", []) or [])
        blocked_mcp_server_names = {
            name
            for server in mcp_servers
            if (name := getattr(server, "name", None))
        }
        if blocked_mcp_server_names:
            workflow_def.mcp_servers = []

    if not blocked_tool_names and not blocked_mcp_server_names:
        return workflow_def

    workflow_def.tools = kept_declarations

    def filter_agent_tools(agent_config: dict[str, Any]) -> None:
        tools = agent_config.get("tools")
        if isinstance(tools, list):
            agent_config["tools"] = [
                tool
                for tool in tools
                if not (isinstance(tool, str) and tool in blocked_tool_names)
            ]
        if blocked_mcp_server_names:
            mcp_servers = agent_config.get("mcp_servers")
            if isinstance(mcp_servers, list):
                agent_config["mcp_servers"] = [
                    server
                    for server in mcp_servers
                    if not (
                        isinstance(server, str)
                        and server in blocked_mcp_server_names
                    )
                ]

    _visit_workflow_agent_configs(workflow_def, filter_agent_tools)
    logger.info(
        "FWK_WORKFLOW_SOURCE_SCOPE_APPLIED source_scope=%s blocked_tools=%s "
        "blocked_mcp_servers=%s",
        config.source_scope,
        sorted(blocked_tool_names),
        sorted(blocked_mcp_server_names),
    )
    return workflow_def


def _filter_workflow_tools(
    defn: WorkflowDefinition, available_tools: list[str]
) -> None:
    """Intersect agent node tool lists with runtime-available tools.

    Walks the workflow tree. For each agent node with a tools config,
    removes tool names that aren't in available_tools or declared in
    the workflow's tools section (resolvable via factory chain).
    """
    available = set(available_tools)
    # YAML-declared tools are resolvable via the factory chain at execution
    # time — don't strip them from agent nodes.
    declared_names = {t.name for t in defn.tools} if defn.tools else set()
    if declared_names:
        available |= declared_names
    logger.debug(
        "WORKFLOW_TOOLS_FILTER_START runtime_tools=%d declared_tools=%s "
        "total_available=%d",
        len(available_tools),
        sorted(declared_names),
        len(available),
    )
    _filter_node_tools(defn.root, available)


def _filter_node_tools(node: Any, available: set[str]) -> None:
    """Recursively filter tools on agent nodes."""
    from databricks_deep_research.workflow.definition import NodeType, WorkflowNode

    if node.type == NodeType.agent:
        tools = node.config.get("tools")
        if isinstance(tools, list):
            filtered = [t for t in tools if t in available]
            if len(filtered) != len(tools):
                removed = set(tools) - set(filtered)
                logger.info(
                    "WORKFLOW_TOOLS_FILTERED node=%s removed=%s",
                    node.id,
                    sorted(removed),
                )
            node.config["tools"] = filtered

    for child in node.children:
        _filter_node_tools(child, available)

    # Handle plan_and_execute body node
    if node.type == NodeType.plan_and_execute:
        body = node.config.get("body")
        if isinstance(body, WorkflowNode):
            tools = body.config.get("tools")
            if isinstance(tools, list):
                body.config["tools"] = [t for t in tools if t in available]


async def stream_workflow_via_framework(
    query: str,
    llm: LLMClient,
    brave_client: BraveSearchClient,
    crawler: WebCrawler,
    conversation_history: list[dict[str, str]] | None = None,
    session_id: UUID | None = None,
    user_id: str | None = None,
    chat_id: str | None = None,
    config: OrchestrationConfig | None = None,
    db: AsyncSession | None = None,
    plugin_manager: PluginManager | None = None,
    plugin_data: dict[str, Any] | None = None,  # noqa: ARG001
    storage_stack: Any = None,
    workflow_def: WorkflowDefinition | None = None,
    extra_state: dict[str, Any] | None = None,
) -> AsyncGenerator[StreamEvent | str, None]:
    """Stream a workflow via the multi-agent framework.

    Generalized entry point that backs both the main-chat research path and
    any caller that supplies a pre-built ``WorkflowDefinition`` directly
    (e.g. the designer-chat path).

    Args:
        query: User's research query.
        llm: App LLM client.
        brave_client: Brave search client.
        crawler: Web crawler.
        conversation_history: Previous messages.
        session_id: Optional session ID.
        user_id: Optional user ID.
        chat_id: Optional chat ID.
        config: Orchestration configuration.
        db: Optional database session.
        plugin_manager: Optional plugin manager.
        plugin_data: Optional plugin context data.
        workflow_def: When provided, skip agent_id/plugin lookup and use this
            definition directly.  When ``None``, falls through to existing
            lookup logic (preserves main-chat behaviour).
        extra_state: Optional dict of additional state keys to seed into
            ``wf_state`` after conversation_history seeding.  Caller-supplied
            values win over any defaults.

    Yields:
        StreamEvent objects and synthesis content chunks (strings).
    """
    from deep_research.agent.orchestration_config import (
        get_default_orchestration_config,
    )
    from deep_research.agent.utils.conversation import normalize_history_roles

    config = config or get_default_orchestration_config()
    # Translate the app's internal "agent" role to OpenAI's "assistant" at the
    # app→framework boundary (source-side prevention; the framework also
    # normalizes at message assembly). Covers main-chat and ChatAgent serving,
    # since both funnel through this entry point. Idempotent.
    conversation_history = normalize_history_roles(conversation_history)

    # Cross-session memory READ path (spec §4.1, flag-gated, default OFF).
    # Recall durable (user_id, agent_id)-keyed facts from prior sessions and
    # prepend them as a spotlighted role=user message so the agent sees earlier
    # corrections/preferences. Fail-soft + hard-timeout: the helper NEVER raises
    # and NEVER blocks — any backend error / slow read degrades to no-memory, so
    # flag-off (and any failure) is byte-identical to today. Runs before wf_state
    # is built so the message rides the existing role=user conversation channel.
    conversation_history = await _maybe_prepend_cross_session_memory(
        conversation_history, config, db, user_id, chat_id,
    )

    start_time = time.perf_counter()
    event_buffer: EventBuffer | None = None
    steps_executed = 0
    steps_skipped = 0
    plan_iterations = 0
    final_report: str | None = None
    structured_output: dict[str, Any] | None = None
    simple_response: str | None = None
    _synthesis_chunks: list[str] = []
    wf_state: WorkflowState | None = None

    # ------------------------------------------------------------------
    # 1. Session start (two-phase persistence)
    # ------------------------------------------------------------------
    try:
        if (
            (db is not None or storage_stack is not None)
            and config.research_session_id is not None
            and config.message_id is not None
            and chat_id is not None
            and user_id is not None
            and not config.session_pre_created
        ):
            from deep_research.agent.persistence import (
                persist_research_session_start_independent,
            )

            chat_id_uuid = _to_uuid(chat_id)
            try:
                await persist_research_session_start_independent(
                    chat_id=chat_id_uuid,
                    user_id=user_id,
                    user_query=query,
                    user_message_id=uuid4(),
                    agent_message_id=config.message_id,
                    research_session_id=config.research_session_id,
                    research_depth=config.research_depth,
                    query_mode=config.query_mode,
                    storage_stack=storage_stack,
                )
                event_buffer = EventBuffer(
                    config.research_session_id,
                    stack=storage_stack,
                )
                logger.info(
                    "FWK_SESSION_CREATED session_id=%s",
                    str(config.research_session_id)[:8],
                )
            except Exception as e:
                logger.warning(
                    "FWK_SESSION_START_FAILED error=%s",
                    str(e)[:200],
                )
        elif config.session_pre_created and config.research_session_id is not None:
            event_buffer = EventBuffer(
                    config.research_session_id,
                    stack=storage_stack,
                )

        # Emit research_started
        if config.message_id:
            started_evt = ResearchStartedEvent(
                message_id=str(config.message_id),
                research_session_id=(
                    str(config.research_session_id)
                    if config.research_session_id
                    else None
                ),
            )
            yield started_evt
            await _buffer_event(started_evt, event_buffer)

        # ------------------------------------------------------------------
        # 2-5. Build context, translate, execute, persist — under MLflow run
        # ------------------------------------------------------------------
        run_label = f"research_{str(session_id or config.research_session_id or '')[:8]}"
        with safe_mlflow_run(run_label):
            async with safe_tool_span(
                "framework_orchestration",
                _SpanType.CHAIN if _SpanType is not None else None,
                {"research.query": query[:200], "research.use_framework": True},
            ):
                # Trace metadata for MLflow correlation. The mlflow.trace.*
                # fields populate the UI's user/session columns; the dr.*
                # provenance tags below make the trace cross-surface
                # filterable (designer-chat / main-chat / shell-app).
                if user_id or chat_id:
                    trace_metadata: dict[str, str] = {}
                    if user_id:
                        trace_metadata["mlflow.trace.user"] = user_id
                    if chat_id:
                        trace_metadata["mlflow.trace.session"] = chat_id
                    if trace_metadata:
                        safe_update_trace(trace_metadata)
                from deep_research.core.trace_provenance import set_trace_provenance
                set_trace_provenance(
                    surface="main-chat",
                    user_id=user_id,
                    session_id=chat_id or session_id,
                    agent_v2_id=config.agent_id,
                    query_preview=query[:200] if query else "",
                )

                # -- 2. Build framework execution context --
                framework_llm = create_framework_llm_client(
                    llm,
                    embedding_model=DEFAULT_EMBEDDING_ENDPOINT,
                    model_overrides=config.model_overrides,
                )

                # Load file search tool (Step 3a)
                file_search_tool = await _load_file_search_tool(
                    config, db, user_id, chat_id, storage_stack=storage_stack,
                )

                # -- 2b. Hydrate chat-scoped memory + preprocess any new files --
                # Memory is durable state tied to the conversation, parallel
                # to ChatSourcePool. File chunks are the authoritative file
                # representation (raw storage is ephemeral tempdir). Any new
                # file is fed through a universal LLM extractor (no regex,
                # no per-format profile) — one cheap-tier call classifies the
                # file and produces typed entities + structured facts that
                # become step-0 KnowledgeFindings.
                chat_memory: IChatMemoryService | None = None
                if chat_id is not None and (db is not None or storage_stack is not None):
                    chat_uuid = UUID(str(chat_id))
                    from deep_research.core.config import get_settings

                    _settings = get_settings()
                    chat_memory = make_chat_memory_service(
                        _settings,
                        storage_stack,
                        session=db,
                        llm=framework_llm,
                    )
                    await chat_memory.hydrate(chat_uuid, user_id=user_id)
                    uploaded_file_ids = await _resolve_uploaded_file_ids(
                        config, db, user_id, chat_uuid, storage_stack=storage_stack,
                    )
                    if uploaded_file_ids:
                        from deep_research.core.config import get_settings as _gs
                        _fus = make_file_upload_service(_gs(), storage_stack, session=db)
                        file_service = _fus
                        await chat_memory.preprocess_new_files(
                            chat_uuid,
                            uploaded_file_ids,
                            file_service=file_service,
                            research_session_id=(
                                config.research_session_id
                                if hasattr(config, "research_session_id")
                                else None
                            ),
                        )
                    # Run ContextEnricher plugins (Architect lifecycle spec).
                    if plugin_manager:
                        from deep_research.agent.tools.base import (
                            ResearchContext as _EnricherCtx,
                        )

                        enricher_ctx = _EnricherCtx(
                            chat_id=chat_uuid,
                            user_id=user_id or "system",
                            research_type=config.research_depth or "medium",
                        )
                        for _plugin in plugin_manager.get_plugins():
                            if not isinstance(_plugin, ContextEnricher):
                                continue
                            plugin_label = getattr(
                                _plugin, "name", type(_plugin).__name__
                            )
                            try:
                                await asyncio.wait_for(
                                    _plugin.enrich_research_memory(
                                        chat_memory, enricher_ctx
                                    ),
                                    timeout=5.0,
                                )
                                logger.info(
                                    "CONTEXT_ENRICHER_DONE plugin=%s",
                                    plugin_label,
                                )
                            except TimeoutError:
                                logger.warning(
                                    "CONTEXT_ENRICHER_TIMEOUT plugin=%s",
                                    plugin_label,
                                )
                            except Exception:
                                logger.exception(
                                    "CONTEXT_ENRICHER_FAILED plugin=%s",
                                    plugin_label,
                                )

                # -- Follow-up chat gate (custom-agent chats with prior research) --
                # Route a new message in a custom-agent chat by intent: a
                # conversational follow-up answerable from data already gathered
                # is answered directly (no expensive re-run of the agent's
                # workflow). Runs BEFORE workflow resolution + the OBO fail-closed
                # check, so a chat turn needs neither the agent's Databricks tools
                # nor an OBO token. Flag-off / no agent_id / no prior research all
                # fall through unchanged to the workflow path below.
                from deep_research.core.config import (
                    get_settings as _get_settings_followup,
                )

                # Cheap guards first so non-agent / non-chat turns short-circuit
                # BEFORE constructing Settings() (which can raise without env).
                _followup_settings = (
                    _get_settings_followup()
                    if (config.agent_id and chat_id is not None)
                    else None
                )
                if (
                    _followup_settings is not None
                    and _followup_settings.followup_chat_gate_enabled
                ):
                    # ``_followup_settings is not None`` already implies
                    # ``chat_id is not None`` (see its guard above); bind a
                    # narrowed local so the followup stream helpers type-check.
                    assert chat_id is not None
                    _followup_chat_id: str = chat_id
                    _existing_for_gate = await _load_existing_sources(
                        storage_stack, db, chat_id,
                    )
                    if _existing_for_gate:
                        _followup_findings = ""
                        if chat_memory is not None:
                            _followup_findings = (
                                chat_memory.render_appendix_block(
                                    agent_type=APPENDIX_AGENT_TYPE,
                                )
                                or ""
                            )
                        _turn_decision = await decide_turn_intent(
                            query=query,
                            conversation_history=conversation_history,
                            prior_findings_summary=_followup_findings,
                            has_prior_research=True,
                            requested=_resolve_turn_intent(config),
                            llm=llm,
                            allow_live_search=(
                                config.allow_live_search
                                if config.allow_live_search is not None
                                else _followup_settings.followup_live_search_enabled
                            ),
                        )
                        logger.info(
                            "FWK_FOLLOWUP_GATE route=%s turn_intent=%s "
                            "query_mode=%s agent_id=%s reason=%s",
                            _turn_decision.route,
                            config.turn_intent,
                            config.query_mode,
                            config.agent_id,
                            _turn_decision.reasoning[:200],
                        )
                        if _turn_decision.route == "live_search":
                            # Bounded live-web-search escape hatch (spec §4.7).
                            # Build the provider-selected search client (reuses the
                            # configured backend + per-agent domain filter), run a
                            # small capped search, and answer from the fresh
                            # sources. On graceful fallback (no usable result) the
                            # code falls THROUGH to the normal research path below.
                            from deep_research.agent.adapters.tool_adapter import (
                                _build_web_search_client,
                            )
                            from deep_research.core.app_config import (
                                get_app_config as _get_app_cfg_ls,
                            )

                            _ls_search_cfg = _get_app_cfg_ls().search
                            _ls_client = _build_web_search_client(
                                search_provider=_ls_search_cfg.provider,
                                brave_client=brave_client,
                                domain_filter_config=config.domain_filter,
                                llm_client=framework_llm,
                                databricks_search_cfg=_ls_search_cfg.databricks,
                            )
                            _ls_chunks: list[str] = []
                            _ls_fell_back = False
                            async for _ls_evt in stream_live_search_answer(
                                query=query,
                                conversation_history=conversation_history,
                                chat_id=_followup_chat_id,
                                llm=llm,
                                web_search_client=_ls_client,
                                max_results=(
                                    _followup_settings.followup_live_search_max_results
                                ),
                                timeout_seconds=(
                                    _followup_settings.followup_live_search_timeout_sec
                                ),
                                prior_findings_summary=_followup_findings,
                            ):
                                if isinstance(_ls_evt, LiveSearchUnavailable):
                                    _ls_fell_back = True
                                    logger.info(
                                        "FWK_FOLLOWUP_LIVE_SEARCH_FALLBACK "
                                        "chat_id=%s reason=%s",
                                        chat_id,
                                        _ls_evt.reason,
                                    )
                                    break
                                if isinstance(_ls_evt, SynthesisProgressEvent):
                                    _ls_chunks.append(_ls_evt.content_chunk)
                                yield _ls_evt
                            if not _ls_fell_back:
                                _ls_answer = "".join(_ls_chunks)
                                await _persist_simple_response(
                                    config,
                                    db,
                                    chat_id,
                                    user_id,
                                    query,
                                    _ls_answer,
                                    event_buffer,
                                    storage_stack=storage_stack,
                                )
                                if config.message_id is not None:
                                    persistence_evt = PersistenceCompletedEvent(
                                        chat_id=str(chat_id),
                                        message_id=str(config.message_id),
                                        research_session_id=None,
                                        chat_title=derive_chat_title_from_query(
                                            query
                                        ),
                                        was_draft=config.is_draft,
                                        counts={"messages": 1},
                                    )
                                    yield persistence_evt
                                    await _buffer_event(persistence_evt, event_buffer)
                                    await _flush_event_buffer(event_buffer)
                                logger.info(
                                    "FWK_FOLLOWUP_LIVE_SEARCH_DONE chat_id=%s "
                                    "answer_len=%d",
                                    chat_id,
                                    len(_ls_answer),
                                )
                                return
                            # else: fall through to the normal research path.
                        if _turn_decision.route == "chat":
                            _followup_chunks: list[str] = []
                            async for _followup_evt in stream_chat_about_results(
                                query=query,
                                conversation_history=conversation_history,
                                chat_id=_followup_chat_id,
                                db=db,
                                llm=llm,
                                prior_findings_summary=_followup_findings,
                                storage_stack=storage_stack,
                            ):
                                if isinstance(_followup_evt, SynthesisProgressEvent):
                                    _followup_chunks.append(
                                        _followup_evt.content_chunk
                                    )
                                yield _followup_evt
                            _followup_answer = "".join(_followup_chunks)
                            await _persist_simple_response(
                                config,
                                db,
                                chat_id,
                                user_id,
                                query,
                                _followup_answer,
                                event_buffer,
                                storage_stack=storage_stack,
                            )
                            if config.message_id is not None:
                                persistence_evt = PersistenceCompletedEvent(
                                    chat_id=str(chat_id),
                                    message_id=str(config.message_id),
                                    research_session_id=None,
                                    chat_title=derive_chat_title_from_query(query),
                                    was_draft=config.is_draft,
                                    counts={"messages": 1},
                                )
                                yield persistence_evt
                                await _buffer_event(persistence_evt, event_buffer)
                                await _flush_event_buffer(event_buffer)
                            logger.info(
                                "FWK_FOLLOWUP_CHAT_DONE chat_id=%s answer_len=%d",
                                chat_id,
                                len(_followup_answer),
                            )
                            return

                logger.info(
                    "FWK_TOOL_CREATION domain_filter=%s domain_filter_type=%s",
                    config.domain_filter,
                    type(config.domain_filter).__name__ if config.domain_filter else "None",
                )

                from deep_research.core.app_config import (
                    get_app_config as _get_app_cfg,
                )

                _search_cfg = _get_app_cfg().search
                framework_tools = await create_framework_tools(
                    brave_client=brave_client,
                    crawler=crawler,
                    domain_filter_config=config.domain_filter,
                    enterprise_tools=await _load_enterprise_tools(
                        config, db, user_id, chat_id, storage_stack,
                    ),
                    user_token=config.user_token,
                    file_search_tool=file_search_tool,
                    chat_id=chat_id,
                    user_id=user_id,
                    # Central provider selection (brave default). The Databricks
                    # built-in-search adapter reuses the framework LLM client's
                    # serving-endpoints connection.
                    search_provider=_search_cfg.provider,
                    llm_client=framework_llm,
                    databricks_search_cfg=_search_cfg.databricks,
                )

                # Register chat-memory tools when memory has any content.
                # Silent no-op when nothing is attached — preserves baseline
                # tool list for workflows without files.
                if chat_memory is not None and not chat_memory.snapshot().empty:
                    from deep_research.core.config import get_settings as _gs2

                    _existing_tool_names = {t.definition.name for t in framework_tools}
                    _rfus = None
                    if db is not None or storage_stack is not None:
                        _rfus = make_file_upload_service(_gs2(), storage_stack, session=db)
                    # search_chat_memory exposed only under CHAT_MEMORY_UNIFIED
                    # (Phase 2c) — flag-off keeps the baseline tool list.
                    _chat_mem_tools: list[Any] = _build_chat_memory_tools(
                        chat_memory,
                        file_service=_rfus,
                        include_search=_gs2().chat_memory_unified,
                    )
                    for _cmt in _chat_mem_tools:
                        if _cmt.definition.name not in _existing_tool_names:
                            framework_tools.append(_cmt)
                            _existing_tool_names.add(_cmt.definition.name)
                            logger.info(
                                "CHAT_MEMORY_TOOL_REGISTERED tool=%s",
                                _cmt.definition.name,
                            )

                # Merge plugin-provided tools so YAML workflows can reference
                # them.  Plugin tools (e.g., sfdc_context) aren't created by
                # create_framework_tools but are needed by custom workflow steps.
                if plugin_manager:
                    from deep_research.agent.tools.base import (
                        ResearchContext as _ToolCtx,
                    )

                    _tool_ctx = _ToolCtx(
                        chat_id=session_id or uuid4(),
                        user_id=user_id or "system",
                        research_type=config.research_depth or "medium",
                    )
                    from deep_research.plugins.base import ToolProvider as _ToolProvider

                    _existing = {t.definition.name for t in framework_tools}
                    for _plugin in plugin_manager.get_plugins():
                        if not isinstance(_plugin, _ToolProvider):
                            continue
                        try:
                            for _tool in _plugin.get_tools(_tool_ctx):
                                if _tool.definition.name not in _existing:
                                    framework_tools.append(_tool)  # type: ignore[arg-type]
                                    _existing.add(_tool.definition.name)
                                    logger.info(
                                        "PLUGIN_TOOL_MERGED tool=%s plugin=%s",
                                        _tool.definition.name,
                                        getattr(
                                            _plugin, "name", type(_plugin).__name__
                                        ),
                                    )
                        except Exception:
                            logger.exception(
                                "PLUGIN_TOOL_MERGE_ERROR plugin=%s",
                                getattr(_plugin, "name", type(_plugin).__name__),
                            )

                context = ExecutionContext(
                    llm_client=framework_llm,
                    enterprise_tools=framework_tools,
                    model_overrides=config.model_overrides or {},
                    user_token=config.user_token,
                    user_id=user_id,
                    approval_broker=config.approval_broker,
                )

                # -- 3. Translate config to workflow definition --
                tool_names = [t.definition.name for t in framework_tools]

                # Validate: enterprise_only requires at least one non-web tool
                if config.source_scope == "enterprise_only":
                    web_names = {"web_search", "web_crawl"}
                    enterprise_names = [n for n in tool_names if n not in web_names]
                    has_mcp_attachments = bool(getattr(config, "enabled_mcp_servers", None))
                    if not enterprise_names and not has_mcp_attachments:
                        logger.error(
                            "FWK_NO_ENTERPRISE_TOOLS source_scope=enterprise_only "
                            "tool_names=%s — research will proceed with no enterprise data",
                            tool_names,
                        )

                # Resolve + prepare the workflow definition in one place so the
                # databricks web-search fill stays coupled to resolution — filling
                # before workflow_def is bound is a silent no-op (see the helper).
                workflow_def = await _resolve_and_prepare_workflow_def(
                    workflow_def,
                    config,
                    user_id,
                    db,
                    tool_names,
                    plugin_manager,
                    _search_cfg,
                )

                logger.info(
                    "FWK_WORKFLOW_TRANSLATED workflow_id=%s tool_names=%s",
                    workflow_def.id,
                    tool_names,
                )

                # Merge chat-attached skills + MCP servers (E1) into the workflow
                # BEFORE the OBO preflight, so a chat-attached Databricks MCP server
                # also fails closed without a token.
                _merge_chat_attachments(
                    workflow_def,
                    getattr(config, "enabled_skills", None),
                    getattr(config, "enabled_mcp_servers", None),
                )

                # Build the shared WorkflowRunner (single execution code path
                # for all app entry points — see workflow_runner_factory.py
                # for the project convention).
                # Fail closed (host policy): under the Databricks Apps runtime,
                # a workflow that declares Databricks-bound tools MUST run on
                # behalf of the user. Without an OBO token every Databricks
                # tool falls back to the app service principal and hits UC
                # permission errors, so reject the request with a clear error
                # rather than returning a silently-degraded report. Local dev
                # (no DATABRICKS_APP_NAME) and web-only workflows are exempt.
                from databricks_deep_research import workflow_requires_databricks
                from databricks_deep_research.tools.factories.builtin import (
                    BuiltinToolFactory,
                )
                from databricks_deep_research.tools.factories.databricks import (
                    DatabricksToolFactory,
                )
                from databricks_deep_research.tools.resolver import (
                    ToolResolver,
                )

                if (
                    os.environ.get("DATABRICKS_APP_NAME")
                    and (config is None or not config.user_token)
                    and workflow_def is not None
                    and workflow_requires_databricks(workflow_def)
                ):
                    raise MissingOBOTokenError(
                        "This workflow needs your Databricks identity to query "
                        "its data sources, but no user token "
                        "(x-forwarded-access-token) reached the server. Open the "
                        "app from the Databricks Apps page so your identity is "
                        "forwarded, or include your OBO token when calling the "
                        "API directly."
                    )

                _has_skills = _definition_has_skills(workflow_def) or bool(
                    getattr(config, "enabled_skills", None)
                )
                _ws_client = None
                if workflow_def.tools or workflow_def.mcp_servers or _has_skills:
                    try:
                        from deep_research.core.auth import get_workspace_client

                        _ws_client = get_workspace_client()
                        logger.info(
                            "FWK_WORKSPACE_CLIENT_OK host=%s",
                            getattr(_ws_client.config, "host", "unknown"),
                        )
                    except Exception as exc:
                        logger.warning(
                            "FWK_WORKSPACE_CLIENT_UNAVAILABLE reason=%s — "
                            "YAML-declared Databricks tools will not be available",
                            str(exc)[:200],
                        )

                # Runtime skill store (Feature 2.2) — built ONLY when an agent
                # declares skills, so skill-less runs avoid the identity lookup and
                # stay byte-identical. Spans workspace-FS (OBO) + bundled seeds;
                # Lakebase-at-runtime + per-user folders are threaded later (A3/C1).
                _skill_store = None
                _skill_scripts_enabled = False
                if _has_skills:
                    from deep_research.core.app_config import get_app_config
                    from deep_research.services.skill_runtime import (
                        build_runtime_skill_store,
                    )

                    # User-registered skill-folder roots (A3) extend the built-in
                    # workspace-FS roots. Fail-soft: a folder-store error must not
                    # break a run that merely declared skills.
                    _skill_roots = await _load_skill_folder_roots(db, user_id)

                    _skill_store = build_runtime_skill_store(
                        llm_client=framework_llm,
                        workspace_client=_ws_client,
                        user_token=config.user_token,
                        extra_roots=_skill_roots or None,
                    )
                    # Global half of the skill-script gate (A2). ANDed per-agent
                    # in the framework auto-attach with ``allow_skill_scripts``.
                    _skill_scripts_enabled = bool(
                        get_app_config().skills.allow_script_execution
                    )

                runner = build_app_workflow_runner(
                    llm_client=framework_llm,
                    workspace_client=_ws_client,
                    user_token=config.user_token,
                    skill_store=_skill_store,
                    skill_scripts_enabled=_skill_scripts_enabled,
                )

                # Build ToolResolver with YAML declarations + factories so
                # declared tools (vector_search, genie, etc.) can be created
                # on-demand by the factory chain. The resolver shares the
                # runner's factory_context — both must see the same
                # BRAVE_API_KEY / workspace_client / user_token wiring.
                tool_resolver = ToolResolver(
                    declarations=list(workflow_def.tools) if workflow_def.tools else None,
                    # No kind overlap: builtin handles web_search/web_crawl/file_search;
                    # Databricks handles vector_search/genie/knowledge_assistant.
                    factories=[BuiltinToolFactory(), DatabricksToolFactory()],
                    factory_context=runner.factory_context,
                )
                logger.info(
                    "FWK_TOOL_RESOLVER_READY declarations=%d "
                    "workspace_client=%s overrides=%d",
                    len(workflow_def.tools),
                    "present" if _ws_client else "MISSING",
                    len(framework_tools),
                )
                # Precedence guard: a per-workflow tool declaration that sets an
                # explicit ``config.provider`` (e.g. web_search with
                # provider: databricks) must win over the auto-injected backend.
                # Resolver overrides are checked BEFORE YAML declarations, so we
                # skip overriding those names and let the factory build them.
                _declared_provider_tools = _tool_names_with_explicit_provider(
                    workflow_def
                )
                for tool in framework_tools:
                    if tool.definition.name in _declared_provider_tools:
                        logger.info(
                            "FWK_OVERRIDE_SKIP tool=%s reason=workflow_declares_provider",
                            tool.definition.name,
                        )
                        continue
                    tool_resolver.override(tool.definition.name, tool)

                # MCP toolset injection (spec §4.3) — mirrors the per-agent
                # domain_filter runtime-override route above. Build one toolset
                # per configured server PER-REQUEST with OBO identity (never the
                # SP), discover its tools, and override the resolver for each so
                # they shadow any like-named declaration. Citeable MCP results
                # flow through admission to the pool (the §4.3 #11 fix lives in
                # the framework's _MCPTool source_kind). Absent mcp_servers =>
                # no-op (byte-identical default).
                _mcp_servers = list(getattr(workflow_def, "mcp_servers", []) or [])
                if _mcp_servers:
                    from deep_research.agent.adapters.mcp_adapter import (
                        build_mcp_toolsets,
                    )

                    # Build off the request event loop: MCP discovery calls the
                    # Databricks MCP client's synchronous ``list_tools()``, which
                    # bridges to an async ``_get_tools_async`` via its OWN event
                    # loop. Called inline here it collides with THIS running loop
                    # (RuntimeError → every server skipped, tools=0, with a
                    # "coroutine never awaited" warning). ``asyncio.to_thread``
                    # runs the whole build on a worker thread with no live loop.
                    _mcp_toolsets = await asyncio.to_thread(
                        build_mcp_toolsets,
                        _mcp_servers,
                        sp_client=_ws_client,
                        user_token=config.user_token,
                    )
                    # Map each built toolset back to its server name so the
                    # executor can attach a server's tools to the agents that
                    # bind it via ``config.mcp_servers`` (maybe_attach_mcp). The
                    # resolver override keeps the tools resolvable by name; the
                    # by-server map is what actually reaches the researcher agents.
                    _mcp_tools_by_server: dict[str, list[Any]] = {}
                    _mcp_injected = 0
                    for _toolset in _mcp_toolsets:
                        _server_tools = list(_toolset.tools)
                        _mcp_tools_by_server.setdefault(
                            _toolset.source_label, []
                        ).extend(_server_tools)
                        for _mcp_tool in _server_tools:
                            tool_resolver.override(
                                _mcp_tool.definition.name, _mcp_tool
                            )
                            _mcp_injected += 1
                    # Stash on both the runner context and the resolver's copied
                    # context. ToolResolver deliberately copies extras at
                    # construction time for cache isolation, so late MCP discovery
                    # must be written to the resolver context explicitly.
                    _stash_mcp_tools_by_server(runner, tool_resolver, _mcp_tools_by_server)
                    # skipped = servers that failed to build (SSRF / MCPConfigError
                    # / missing mcp packages) — surfaced so an attached-but-unused
                    # server is never silently dropped.
                    _mcp_skipped = len(_mcp_servers) - len(_mcp_tools_by_server)
                    logger.info(
                        "FWK_MCP_INJECTED servers=%d tools=%d skipped=%d obo=%s",
                        len(_mcp_servers),
                        _mcp_injected,
                        _mcp_skipped,
                        bool(config.user_token),
                    )

                # Pre-execution guard — fail before LLM tokens are spent.
                # Layer 3 of the layered tool-context validation: if a declared
                # tool's factory cannot construct it (e.g. ``schema_cache`` is
                # ``None`` because ``STORAGE_WAREHOUSE_ID`` was not propagated
                # to the deployed app), raise here with the per-tool error
                # list rather than letting the failure surface mid-stream as
                # a misleading ``WorkflowError: missing declared tools``.
                await tool_resolver.validate_all()

                tracker = DomainContextTracker()
                from deep_research.agent.promotion_capture import (
                    PromotionTraceCollector,
                )

                promotion_collector = PromotionTraceCollector(
                    run_id=str(config.research_session_id or ""),
                )

                wf_state = WorkflowState(query=query)
                if conversation_history:
                    wf_state.append("init", "conversation_history", conversation_history)
                    # Also set the typed field so the framework harness picks it
                    # up via AgentInput.conversation_history (US-06 W1 primitive).
                    # Both paths coexist for backward compatibility.
                    wf_state.conversation_history = list(conversation_history)

                if extra_state:
                    for key, value in extra_state.items():
                        wf_state.append("init", key, value)

                # Load existing sources for follow-up queries (Step 4)
                existing_sources = await _load_existing_sources(
                    storage_stack, db, chat_id,
                )
                if existing_sources:
                    # Durable history (feeds appendix/retrieval) — unchanged.
                    wf_state.append("init", "existing_sources", existing_sources)
                    # Read-path (Phase 2b): stage a bounded, canonical, citable
                    # seed under a DEDICATED key and flip the framework's
                    # run-start seed flag. Gated by CHAT_MEMORY_UNIFIED so
                    # flag-off behaviour is byte-identical.
                    from deep_research.core.config import get_settings as _gs_seed

                    if _gs_seed().chat_memory_unified:
                        _seed = _build_prior_source_seed(existing_sources, query, top_k=20)
                        if _seed:
                            wf_state.append("init", "prior_sources_for_seed", _seed)
                            wf_state.append("init", "seed_prior_sources", True)
                            logger.info(
                                "PRIOR_SOURCES_SEED_STAGED chat_id=%s seed=%d of %d",
                                chat_id,
                                len(_seed),
                                len(existing_sources),
                            )

                # Seed chat-memory appendix for universal system-prompt
                # injection (reserved key consumed by harness._build_input).
                # Empty when memory is empty → no-op, backward-compat
                # preserved (golden-file regression test covers this).
                if chat_memory is not None:
                    # Render for the union agent type so the single shared
                    # appendix key (injected into every node by the harness)
                    # carries findings + coverage + entities, not just the
                    # coordinator subset. See APPENDIX_AGENT_TYPE.
                    _appendix = chat_memory.render_appendix_block(agent_type=APPENDIX_AGENT_TYPE)
                    if _appendix:
                        wf_state.append(
                            "init",
                            CHAT_MEMORY_APPENDIX_STATE_KEY,
                            _appendix,
                        )
                        logger.info(
                            "CHAT_MEMORY_APPENDIX_SEEDED chat_id=%s chars=%d",
                            chat_id,
                            len(_appendix),
                        )

                research_timeout = config.research_timeout_seconds

                try:
                    async with asyncio.timeout(research_timeout):
                        async for fw_event in runner.stream(
                            workflow_def,
                            state=wf_state,
                            tool_resolver=tool_resolver,
                            context=context,
                            strict_tool_resolution=True,
                        ):
                            # Capture the run's observed behavior for promotion
                            # (spec 6.1). Fail-soft; never affects the stream.
                            promotion_collector.observe(fw_event)
                            # Detect simple query short-circuit (Step 2).
                            #
                            # Honor the coordinator's direct-answer short-circuit
                            # ONLY in Simple mode. Web Search / Deep Research run a
                            # real researcher + synthesizer that produce the
                            # authoritative report; a classifier ``direct_response``
                            # in those modes is often a premature decline ("I can't
                            # get real-time data, use <service>") and must NEVER
                            # pre-empt and discard the report the pipeline goes on to
                            # produce. (Diagnosed: backend produced a correct cited
                            # answer but the user only ever saw the classifier decline.)
                            if (
                                config.query_mode == "simple"
                                and isinstance(fw_event, CoordinatorClassifiedEvent)
                                and fw_event.is_simple
                                and fw_event.direct_response
                            ):
                                simple_response = fw_event.direct_response
                                # Yield the direct response as synthesis chunks
                                yield SynthesisStartedEvent(
                                    total_observations=0,
                                    total_sources=0,
                                )
                                yield AgentStartedEvent(
                                    agent="synthesizer", model_tier="simple",
                                )
                                yield SynthesisProgressEvent(
                                    content_chunk=simple_response,
                                )
                                yield AgentCompletedEvent(
                                    agent="synthesizer", duration_ms=0,
                                )
                                # Skip remaining workflow events for simple mode
                                continue

                            # Skip remaining events after a simple response was
                            # captured (only set above, i.e. only in Simple mode).
                            if simple_response is not None:
                                continue

                            # Map framework events to app SSE events
                            app_events = tracker.process_event(fw_event)

                            for app_evt in app_events:
                                sse_event = _to_sse_event(app_evt)
                                if sse_event:
                                    # DR_LEAK_TRACE sse_emit: capture each
                                    # SSE event sent to the client. Any
                                    # planning text first observable here
                                    # means it survived the entire pipeline.
                                    try:
                                        _evt_repr = repr(sse_event)[:300].replace("\n", "\\n")
                                        logger.info(
                                            "DR_LEAK_TRACE phase=sse_emit "
                                            "event_type=%s payload_head=%r",
                                            type(sse_event).__name__,
                                            _evt_repr,
                                        )
                                    except Exception as _exc:  # pragma: no cover
                                        logger.debug(
                                            "DR_LEAK_TRACE sse_emit skipped: %s", _exc
                                        )
                                    yield sse_event
                                    await _buffer_event(sse_event, event_buffer)

                            # Handle streaming chunks directly
                            if isinstance(fw_event, AgentStreamChunkEvent):
                                yield SynthesisProgressEvent(content_chunk=fw_event.chunk)
                                _synthesis_chunks.append(fw_event.chunk)

                            # Track progress via isinstance (not string matching)
                            if isinstance(fw_event, ItemCompletedEvent):
                                steps_executed += 1
                            elif isinstance(fw_event, ReplanTriggeredEvent):
                                plan_iterations += 1

                            # Capture final report directly from the authoritative
                            # source.  This must happen BEFORE the periodic-persist
                            # check below, which may call get_persistence_delta()
                            # and reset the delta (discarding final_report).
                            elif isinstance(fw_event, FwkWorkflowCompletedEvent):
                                if fw_event.final_report:
                                    final_report = fw_event.final_report
                                    logger.info(
                                        "FWK_FINAL_REPORT_CAPTURED len=%d",
                                        len(final_report),
                                    )
                                if fw_event.structured_output is not None:
                                    structured_output = fw_event.structured_output
                                    logger.info("FWK_STRUCTURED_OUTPUT_CAPTURED type=%s", type(structured_output).__name__)
                                    # Ensure final_report is valid JSON (not __repr__) when
                                    # structured output exists, so DB-persisted message.content
                                    # is parseable by parseStructuredOutput() on reload.
                                    final_report = json.dumps(structured_output, default=str)

                            # Periodic persistence
                            if tracker.should_persist():
                                delta = tracker.get_persistence_delta()
                                await _persist_delta(delta, config, db, chat_id, user_id)

                except TimeoutError:
                    logger.error(
                        "FWK_RESEARCH_TIMEOUT timeout_seconds=%d steps=%d",
                        research_timeout, steps_executed,
                    )
                    yield StreamErrorEvent(
                        error_code="RESEARCH_TIMEOUT",
                        error_message=(
                            f"Research timed out after {research_timeout}s. "
                            f"Completed {steps_executed} steps."
                        ),
                        recoverable=False,
                    )

                # Get final delta
                final_delta = tracker.get_persistence_delta()
                if final_delta.final_report:
                    final_report = final_delta.final_report

                # Safety net: if final_report is still None or truncated
                # (<=200 chars from AgentOutputEvent.output_preview), fall
                # back to accumulated streaming chunks.
                joined_chunks = "".join(_synthesis_chunks)
                if _synthesis_chunks and (
                    final_report is None
                    or (len(final_report) <= 200 and len(joined_chunks) > len(final_report))
                ):
                    final_report = joined_chunks

                # Plan v2.3 UX backstop (main-app surface). The framework's
                # ``WorkflowResult.output`` carries an equivalent backstop
                # for direct consumers (shell app, SDK, notebooks). We
                # duplicate the logic here because the main-app
                # orchestrator builds ``final_report`` from
                # ``tracker.get_persistence_delta()`` + ``_synthesis_chunks``
                # rather than from ``runner.last_result.output``; without
                # this duplication the main-app chat would still surface
                # an empty report when Stage 8 wiped every claim.
                _verif = getattr(final_delta, "verification_summary", None) or {}
                _verified_count = 0
                _total_claims = 0
                if isinstance(_verif, dict):
                    _verified_count = int(_verif.get("verified_claims", 0) or 0)
                    _total_claims = int(_verif.get("total_claims", 0) or 0)
                _final_report_len = len(final_report) if final_report else 0
                _chunks_len = len(joined_chunks)
                if (
                    _total_claims > 0
                    and _verified_count == 0
                    and _chunks_len > 200
                    and _final_report_len < _chunks_len // 2
                ):
                    logger.warning(
                        "FWK_VERIFICATION_BACKSTOP_TRIGGERED total_claims=%d "
                        "verified_claims=%d final_report_len=%d chunks_len=%d",
                        _total_claims,
                        _verified_count,
                        _final_report_len,
                        _chunks_len,
                    )
                    _banner = (
                        "> ⚠️ **Citations could not be verified.** "
                        "The framework's entailment checker did not ground "
                        f"any of the {_total_claims} claims this draft contains. "
                        "Numbers below come directly from the retrieved corpus "
                        "chunks; treat as a draft, not a final answer.\n\n"
                    )
                    final_report = _banner + joined_chunks

            # -- 5. Session completion --

            # Simple-mode auto-escalation to Web Search.
            # The Simple workflow is coordinator-only and writes a report only
            # when the coordinator returns a direct answer (is_simple=True). When
            # it declines — e.g. a current-events query it judges needs research —
            # ``simple_response`` is None and no node wrote a report, which used to
            # trip ``FWK_PERSISTENCE_GUARD_FAILED`` and hard-fail the run ("nothing
            # happens"). Instead, fall forward to a bounded live web-search answer
            # (the same helper as the follow-up live-search hatch) so Simple never
            # dead-ends. Truly-simple queries keep the fast direct path above.
            if (
                config.query_mode == "simple"
                and simple_response is None
                and not final_report
            ):
                logger.info(
                    "FWK_SIMPLE_ESCALATE_WEB chat_id=%s — coordinator declined a "
                    "direct answer; escalating to a bounded web search",
                    chat_id,
                )
                from deep_research.agent.adapters.tool_adapter import (
                    _build_web_search_client,
                )
                from deep_research.core.app_config import (
                    get_app_config as _get_app_cfg_esc,
                )

                _esc_max_results = 5
                _esc_timeout: float = 20.0
                try:
                    from deep_research.core.config import (
                        get_settings as _get_settings_esc,
                    )

                    _esc_settings = _get_settings_esc()
                    _esc_max_results = _esc_settings.followup_live_search_max_results
                    _esc_timeout = _esc_settings.followup_live_search_timeout_sec
                except Exception:
                    pass  # bounded literals are a safe fallback

                _esc_search_cfg = _get_app_cfg_esc().search
                _esc_client = _build_web_search_client(
                    search_provider=_esc_search_cfg.provider,
                    brave_client=brave_client,
                    domain_filter_config=config.domain_filter,
                    llm_client=framework_llm,
                    databricks_search_cfg=_esc_search_cfg.databricks,
                )
                _esc_chunks: list[str] = []
                _esc_unavailable = False
                async for _esc_evt in stream_live_search_answer(
                    query=query,
                    conversation_history=conversation_history,
                    chat_id=str(chat_id) if chat_id is not None else "",
                    llm=llm,
                    web_search_client=_esc_client,
                    max_results=_esc_max_results,
                    timeout_seconds=_esc_timeout,
                    prior_findings_summary="",
                ):
                    if isinstance(_esc_evt, LiveSearchUnavailable):
                        _esc_unavailable = True
                        logger.info(
                            "FWK_SIMPLE_ESCALATE_UNAVAILABLE chat_id=%s reason=%s",
                            chat_id,
                            _esc_evt.reason,
                        )
                        break
                    if isinstance(_esc_evt, SynthesisProgressEvent):
                        _esc_chunks.append(_esc_evt.content_chunk)
                    yield _esc_evt
                _esc_answer = "".join(_esc_chunks).strip()
                if not _esc_unavailable and _esc_answer:
                    # Persist via the existing simple-response path below.
                    simple_response = _esc_answer
                else:
                    # Never hard-fail: a short graceful note beats a dead run.
                    simple_response = (
                        "I couldn't pull live web results for this just now. "
                        "Try **Web Search** or **Deep Research** for a fuller, "
                        "sourced answer."
                    )

            # Simple mode persistence (Step 2)
            if simple_response is not None:
                final_report = simple_response
                await _persist_simple_response(
                    config, db, chat_id, user_id, query, simple_response,
                    event_buffer,
                    storage_stack=storage_stack,
                )
                # Yield persistence event for simple mode
                if (
                    config.message_id is not None
                    and chat_id is not None
                ):
                    chat_title = derive_chat_title_from_query(query)
                    persistence_evt = PersistenceCompletedEvent(
                        chat_id=str(chat_id),
                        message_id=str(config.message_id),
                        research_session_id=None,
                        chat_title=chat_title,
                        was_draft=config.is_draft,
                        counts={"messages": 1},
                    )
                    yield persistence_evt
                    await _buffer_event(persistence_evt, event_buffer)
                    await _flush_event_buffer(event_buffer)
            else:
                # Full research persistence
                if event_buffer:
                    try:
                        await event_buffer.flush()
                    except Exception as e:
                        logger.warning("FWK_EVENT_BUFFER_FLUSH_FAILED error=%s", str(e)[:200])

                # F-PERSIST-GUARDS: `db is not None` was a proxy for
                # "persistence is wired up", but the independent-session
                # helpers called by `_persist_completion` open their own
                # sessions via `get_session_maker()` internally. Dropping the
                # check lets SQL-Warehouse-only deploys (where the request
                # `db` is None) still persist research results instead of
                # silently logging `FWK_PERSISTENCE_GUARD_FAILED` and losing
                # the chat.
                if (
                    config.message_id is not None
                    and config.research_session_id is not None
                    and final_report
                    and chat_id is not None
                    and user_id is not None
                ):
                    chat_id_uuid = _to_uuid(chat_id)

                    pool_sources = _get_pool_sources(wf_state)
                    framework_claims, framework_summary = (
                        _extract_verification_from_framework_state(
                            wf_state,
                            pool_sources,
                        )
                    )

                    extracted_claims = framework_claims
                    effective_summary = framework_summary
                    if not extracted_claims and effective_summary is None and structured_output is None:
                        extracted_claims, effective_summary = (
                            _extract_verification_from_report(
                                final_report,
                                pool_sources,
                            )
                        )

                    if effective_summary is None and final_delta.verification_summary:
                        from deep_research.agent.state import VerificationSummaryInfo

                        td = final_delta.verification_summary
                        effective_summary = VerificationSummaryInfo(
                            total_claims=td.get("total_claims", 0),
                            supported_count=td.get("verified_claims", 0),
                            contradicted_count=td.get("removed_claims", 0),
                            unsupported_count=td.get("softened_claims", 0),
                            citation_corrections=td.get("corrected_citations", 0),
                        )

                    counts = await _persist_completion(
                        config, chat_id_uuid, user_id, query, final_report, event_buffer,
                        wf_state,
                        claims=extracted_claims,
                        verification_summary=effective_summary,
                        storage_stack=storage_stack,
                        promotion_trace=promotion_collector.build(query_shape=query),
                    )
                    if counts:
                        chat_title = derive_chat_title_from_query(query)
                        persistence_evt = PersistenceCompletedEvent(
                            chat_id=str(chat_id_uuid),
                            message_id=str(config.message_id),
                            research_session_id=str(config.research_session_id),
                            chat_title=chat_title,
                            was_draft=config.is_draft,
                            counts=counts,
                        )
                        yield persistence_evt
                        await _buffer_event(persistence_evt, event_buffer)
                        await _flush_event_buffer(event_buffer)
                else:
                    logger.warning(
                        "FWK_PERSISTENCE_GUARD_FAILED "
                        "db=%s message_id=%s session_id=%s "
                        "report_len=%d chat_id=%s user_id=%s",
                        db is not None,
                        config.message_id is not None,
                        config.research_session_id is not None,
                        len(final_report) if final_report else 0,
                        chat_id is not None,
                        user_id is not None,
                    )

    except Exception as e:
        tb = traceback.format_exc()
        logger.exception(
            "FWK_ORCHESTRATION_ERROR error_type=%s error=%s",
            type(e).__name__,
            str(e)[:200],
        )
        yield StreamErrorEvent(
            error_code="ORCHESTRATION_ERROR",
            error_message=str(e),
            recoverable=False,
            stack_trace=tb,
            error_type=type(e).__name__,
        )

        # Mark session as FAILED
        if (
            event_buffer is not None
            and config is not None
            and config.research_session_id is not None
            and config.message_id is not None
        ):
            with contextlib.suppress(Exception):
                await event_buffer.flush()
            try:
                from deep_research.agent.persistence import (
                    persist_research_session_failed_independent,
                )
                _chat_id_for_fail = _to_uuid(chat_id) if chat_id is not None else None
                await persist_research_session_failed_independent(
                    research_session_id=config.research_session_id,
                    agent_message_id=config.message_id,
                    error_message=str(e)[:500],
                    storage_stack=storage_stack,
                    chat_id=_chat_id_for_fail,
                )
            except Exception:
                # PR4 CRITICAL fix: do not silently drop the failure-marker.
                # Without this, the DB session row stays in 'running' forever
                # while the user sees a frontend error (silent data loss).
                logger.exception(
                    "FWK_FAILURE_PERSISTENCE_FAILED research_session_id=%s",
                    str(config.research_session_id)[:8],
                )

    # ------------------------------------------------------------------
    # 6. Final completion event (always emitted)
    # ------------------------------------------------------------------
    total_duration_ms = int((time.perf_counter() - start_time) * 1000)
    completed_evt = ResearchCompletedEvent(
        session_id=session_id or uuid4(),
        total_steps_executed=steps_executed,
        total_steps_skipped=steps_skipped,
        plan_iterations=plan_iterations,
        total_duration_ms=total_duration_ms,
        final_report=final_report,
        structured_output=structured_output,
    )
    yield completed_evt
    await _buffer_event(completed_evt, event_buffer)
    await _flush_event_buffer(event_buffer)


async def stream_research_via_framework(
    query: str,
    llm: LLMClient,
    brave_client: BraveSearchClient,
    crawler: WebCrawler,
    conversation_history: list[dict[str, str]] | None = None,
    session_id: UUID | None = None,
    user_id: str | None = None,
    chat_id: str | None = None,
    config: OrchestrationConfig | None = None,
    db: AsyncSession | None = None,
    plugin_manager: PluginManager | None = None,
    plugin_data: dict[str, Any] | None = None,
    storage_stack: Any = None,
) -> AsyncGenerator[StreamEvent | str, None]:
    """Back-compat alias for stream_workflow_via_framework. New callers should
    use stream_workflow_via_framework directly and (optionally) pass workflow_def."""
    async for event in stream_workflow_via_framework(
        query=query,
        llm=llm,
        brave_client=brave_client,
        crawler=crawler,
        conversation_history=conversation_history,
        session_id=session_id,
        user_id=user_id,
        chat_id=chat_id,
        config=config,
        db=db,
        plugin_manager=plugin_manager,
        plugin_data=plugin_data,
        storage_stack=storage_stack,
    ):
        yield event


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_uuid(value: str | UUID) -> UUID:
    """Convert a string to UUID. Pass-through if already a UUID."""
    return UUID(value) if isinstance(value, str) else value


def _safe_uuid(value: Any) -> UUID:
    """Convert a value to UUID, generating deterministic UUID for non-UUID strings.

    Replicates the behavior of ``_plan_id_to_uuid()`` in the legacy orchestrator.
    Uses uuid5 with NAMESPACE_DNS for deterministic mapping: same input string
    always produces the same UUID.
    """
    if isinstance(value, UUID):
        return value
    s = str(value)
    try:
        return UUID(s)
    except ValueError:
        from uuid import NAMESPACE_DNS, uuid5

        return uuid5(NAMESPACE_DNS, s)


def _to_sse_event(app_evt: AppSSEEvent) -> StreamEvent | None:
    """Convert an AppSSEEvent to the app's SSE StreamEvent type.

    Maps domain context tracker output to the existing app streaming
    event classes that the frontend expects.
    """
    evt_type = app_evt.event_type
    data = app_evt.data

    if evt_type == "coordinator_classified":
        return AgentStartedEvent(agent="coordinator", model_tier="simple")
    elif evt_type == "background_completed":
        return AgentCompletedEvent(agent="background_investigator", duration_ms=0)
    elif evt_type == "plan_created":
        from deep_research.schemas.streaming import PlanCreatedEvent

        # Coerce steps to PlanStepSummary-compatible dicts.
        # Framework steps may have extra fields (description, source_hints) —
        # Pydantic silently drops those (BaseSchema has no extra="forbid").
        # But if the LLM omits a required field, provide safe defaults.
        raw_steps = data.get("steps") or []
        safe_steps = [
            {
                "id": step.get("id", f"step-{i}"),
                "title": step.get("title", f"Step {i + 1}"),
                "step_type": step.get("step_type", "research"),
                "needs_search": step.get("needs_search", True),
            }
            for i, step in enumerate(raw_steps)
            if isinstance(step, dict)
        ]

        return PlanCreatedEvent(
            plan_id=_safe_uuid(data.get("plan_id") or uuid4()),
            title=data.get("title", ""),
            thought=data.get("thought", ""),
            steps=safe_steps,
            iteration=data.get("iteration", 1),
        )
    elif evt_type == "step_started":
        from deep_research.schemas.streaming import StepStartedEvent

        return StepStartedEvent(
            step_index=data.get("item_index", 0),
            step_id=str(data.get("item_index", 0)),
            step_title=data.get("item_summary", ""),
            step_type="research",
        )
    elif evt_type == "step_completed":
        from deep_research.schemas.streaming import StepCompletedEvent

        return StepCompletedEvent(
            step_index=data.get("item_index", 0),
            step_id=str(data.get("item_index", 0)),
            observation_summary="",
            sources_found=data.get("sources_found", 0),
        )
    elif evt_type == "reflection_decision":
        from deep_research.schemas.streaming import ReflectionDecisionEvent

        return ReflectionDecisionEvent(
            decision=data.get("decision", "continue"),
            reasoning=data.get("reasoning", ""),
        )
    elif evt_type == "synthesis_started":
        return SynthesisStartedEvent(
            total_observations=data.get("total_observations", 0),
            total_sources=data.get("total_sources", 0),
        )
    elif evt_type == "agent_output":
        # Agent output events are handled via final_report tracking
        return None
    elif evt_type == "workflow_completed":
        # Handled by the ResearchCompletedEvent at the end
        return None
    elif evt_type == "node_error":
        return StreamErrorEvent(
            error_code="NODE_ERROR",
            error_message=data.get("error", "Unknown error"),
            recoverable=data.get("will_retry", False),
        )
    elif evt_type == "research_progress":
        progress_type = data.get("progress_type")
        if progress_type == "tool_call":
            from deep_research.schemas.streaming import ToolCallEvent as AppToolCallEvent

            return AppToolCallEvent(
                tool_name=data.get("tool_name", ""),
                tool_args=data.get("tool_args", {}),
                call_number=data.get("call_number", 0),
            )
        elif progress_type == "tool_result":
            from deep_research.schemas.streaming import ToolResultEvent as AppToolResultEvent

            return AppToolResultEvent(
                tool_name=data.get("tool_name", ""),
                result_preview=data.get("result_summary", ""),
                sources_crawled=data.get("source_count", 0),
                sources_added=data.get("sources_added", 0),
            )
        elif progress_type == "claim_verified":
            from deep_research.schemas.streaming import ClaimVerifiedEvent as AppClaimVerified

            # Convert float confidence to string level
            _raw_conf = data.get("confidence", 0.5)
            if isinstance(_raw_conf, str):
                _conf_level = _raw_conf
            elif isinstance(_raw_conf, (int, float)):
                _conf_level = (
                    "high" if _raw_conf >= 0.8
                    else "low" if _raw_conf <= 0.4
                    else "medium"
                )
            else:
                _conf_level = "medium"

            return AppClaimVerified(
                claim_id=data.get("claim_id", uuid4()),
                claim_text=data.get("claim_text", ""),
                position_start=data.get("position_start", 0),
                position_end=data.get("position_end", 0),
                verdict=data.get("verdict", ""),
                confidence_level=_conf_level,
                evidence_preview=data.get("evidence_snippet", ""),
                # Numeric citation keys → frontend citationData mapping for live
                # (pre-persistence) marker coloring. Schema defaults handle absence.
                citation_key=data.get("citation_key"),
                citation_keys=data.get("citation_keys"),
            )
        elif progress_type == "verification_summary":
            from deep_research.schemas.streaming import (
                VerificationSummaryEvent as AppVerifSummary,
            )

            return AppVerifSummary(
                message_id=data.get("message_id", uuid4()),
                total_claims=data.get("total_claims", 0),
                supported=data.get("verified_claims", 0),
                partial=0,
                unsupported=data.get("softened_claims", 0),
                contradicted=data.get("removed_claims", 0),
                abstained_count=0,
                citation_corrections=data.get("corrected_citations", 0),
                warning=data.get("warning", False),
            )
        elif progress_type == "citation_corrected":
            from deep_research.schemas.streaming import (
                CitationCorrectedEvent as AppCitationCorrected,
            )

            return AppCitationCorrected(
                claim_id=data.get("claim_id", uuid4()),
                correction_type=data.get("action", "keep"),
            )
        # replan_triggered, evaluation_decision → informational, no frontend event
        return None
    # stream_chunk handled directly in the main loop
    return None


async def _buffer_event(
    event: StreamEvent | None,
    buffer: EventBuffer | None,
) -> None:
    """Add event to buffer for persistence, handling None gracefully."""
    if event and buffer:
        with contextlib.suppress(Exception):
            await buffer.add_event(event)


async def _flush_event_buffer(buffer: EventBuffer | None) -> None:
    """Flush buffered terminal events without failing the research response."""
    if not buffer:
        return
    try:
        await buffer.flush()
    except Exception as e:
        logger.warning("FWK_EVENT_BUFFER_FLUSH_FAILED error=%s", str(e)[:200])


async def _resolve_uploaded_file_ids(
    config: OrchestrationConfig,
    db: Any,
    user_id: str | None,
    chat_id: UUID | str | None,
    *,
    storage_stack: Any = None,
) -> list[UUID]:
    """Resolve the set of UploadedFile UUIDs visible to this turn.

    Mirrors the dual-path logic of ``_load_file_search_tool`` (explicit
    ``config.file_ids`` wins; otherwise auto-discover by chat session),
    but returns the raw UUID list instead of a constructed tool — so
    ``ChatMemoryService.preprocess_new_files`` can iterate over them.
    Returns [] if no files.
    """
    if config.file_ids:
        out: list[UUID] = []
        for raw in config.file_ids:
            try:
                out.append(UUID(str(raw)))
            except (TypeError, ValueError):
                logger.warning("FILE_ID_INVALID raw=%r", raw)
        return out

    if (db is None and storage_stack is None) or not user_id or not chat_id:
        return []

    try:
        from deep_research.core.config import get_settings as _gs3
        service = make_file_upload_service(_gs3(), storage_stack, session=db)
        session_uuid = chat_id if isinstance(chat_id, UUID) else UUID(str(chat_id))
        chat_files, _ = await service.get_session_files(
            user_id, session_uuid, limit=20,
        )
        return [f.id for f in chat_files if f.is_ready]
    except Exception as e:
        logger.warning(
            "FILE_IDS_AUTO_DISCOVERY_FAILED error=%s", str(e)[:200],
        )
        return []


async def _load_file_search_tool(
    config: OrchestrationConfig,
    db: Any,
    user_id: str | None,
    chat_id: str | None,
    *,
    storage_stack: Any = None,
) -> Any | None:
    """Load file search tool from explicit file_ids or auto-discovery.

    Returns an app-level FileSearchTool instance, or None.
    """
    if config.file_ids and (db is not None or storage_stack is not None) and user_id:
        try:
            from deep_research.agent.tools.file_search import create_file_search_tool

            return create_file_search_tool(
                session=db,
                owner_id=user_id,
                file_ids=config.file_ids,
            )
        except Exception as e:
            logger.warning(
                "FWK_FILE_SEARCH_TOOL_FAILED error=%s",
                str(e)[:200],
            )
            return None

    # Auto-discover files for the chat
    if (db is not None or storage_stack is not None) and user_id and chat_id:
        try:
            from uuid import UUID as _UUID

            from deep_research.core.config import get_settings as _gs4

            service = make_file_upload_service(_gs4(), storage_stack, session=db)
            chat_files, _ = await service.get_session_files(
                user_id, _UUID(chat_id), limit=20,
            )
            ready_files = [f for f in chat_files if f.is_ready]

            if ready_files:
                from deep_research.agent.tools.file_search import create_file_search_tool

                file_ids = [str(f.id) for f in ready_files]
                logger.info(
                    "FWK_FILE_IDS_AUTO_DISCOVERED file_count=%d chat_id=%s",
                    len(file_ids), chat_id,
                )
                return create_file_search_tool(
                    session=db,
                    owner_id=user_id,
                    file_ids=file_ids,
                )
        except Exception as e:
            logger.warning(
                "FWK_FILE_DISCOVERY_FAILED error=%s",
                str(e)[:200],
            )

    return None


def _build_chat_memory_tools(
    chat_memory: Any,
    *,
    file_service: Any = None,
    include_search: bool = False,
) -> list[Any]:
    """Build the chat-memory tool set exposed to agents.

    Always includes the attached-file tools (list/entities). Adds
    ``read_attached_file`` when a file service is available, and
    ``search_chat_memory`` (Phase 2c) when ``include_search`` is set (the
    orchestrator passes ``CHAT_MEMORY_UNIFIED``). Pure + side-effect-free so the
    registration policy is unit-testable without the full orchestrator.
    """
    tools: list[Any] = [
        ListAttachedFilesTool(chat_memory),
        GetFileEntitiesTool(chat_memory),
    ]
    if file_service is not None:
        tools.append(ReadAttachedFileTool(chat_memory, file_service))
    if include_search:
        tools.append(SearchChatMemoryTool(chat_memory))
    return tools


def _build_prior_source_seed(
    sources: list[dict[str, Any]],
    query: str,
    *,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """Bounded, canonical, citable, query-ranked seed for follow-up pool seeding.

    Distinct from ``existing_sources`` (full durable history). Per Codex review:
    - canonicalizes + dedups by canonical URL (Phase-1 ``canonicalize_url``) so
      tracking-param variants don't crowd the pool (§7);
    - requires an evidence body (``content``/``snippet``) — citable-snapshot
      guard (§2);
    - ADAPTIVE K: when any record overlaps the current query, keep only those
      (a weakly-related follow-up seeds fewer); else fall back to all citable
      records so the follow-up isn't starved; bounded by ``top_k``.

    Stamps ``evidence_quality="cached"`` on every record (2b-0 gate result): a
    bare record is normalized to ``evidence_quality=""`` and rejected by the
    synthesizer's ``source_is_substantive`` despite carrying text; ``"cached"``
    is in the substantive set and is semantically exact for a prior-turn source.
    The pool's hybrid index re-ranks downstream; this is a coarse pre-filter.
    """
    from deep_research.agent.url_canonical import canonicalize_url

    q_terms = {t for t in query.casefold().split() if len(t) > 2}

    def _overlap(s: dict[str, Any]) -> int:
        text = f"{s.get('title') or ''} {s.get('snippet') or ''}".casefold()
        return sum(1 for t in q_terms if t in text) if q_terms else 0

    seen: set[str] = set()
    scored: list[tuple[int, dict[str, Any]]] = []
    for s in sources:
        url = str(s.get("url") or "")
        if not url or not (s.get("content") or s.get("snippet")):
            continue
        canon = canonicalize_url(url)
        if canon in seen:
            continue
        seen.add(canon)
        scored.append((_overlap(s), {**s, "url": canon, "evidence_quality": "cached"}))

    positives = [item for score, item in scored if score > 0]
    candidates = positives if positives else [item for _score, item in scored]
    candidates.sort(key=_overlap, reverse=True)
    return candidates[:top_k]


async def _load_existing_sources(
    storage_stack: Any,
    db: Any,
    chat_id: str | None,
) -> list[dict[str, Any]]:
    """Load existing sources from prior messages for follow-up queries.

    F-SOURCES: prefer `storage_stack.cache` (ChatDocument.state.sources) when
    available — no DB round-trip, survives Lakebase-off deployments. Falls
    back to the legacy ORM when only `db` is provided.

    Returns a list of source dicts suitable for seeding into the framework's
    initial_state. Shape preserved from the legacy path:
    ``{"url", "title", "snippet", "content"}``.
    """
    if not chat_id:
        return []

    # Cached path: read from the hydrated ChatDocument.
    if storage_stack is not None:
        try:
            chat_id_uuid = _to_uuid(chat_id)
            doc = await storage_stack.cache.get(chat_id_uuid)
            items = list(doc.state.sources)
            if not items:
                return []

            # Sort by relevance_score desc (nulls last), matching the legacy
            # order_by. The new engine stores relevance_score / snippet /
            # content inside `Source.metadata`.
            def _score(src: Any) -> tuple[int, float]:
                meta = getattr(src, "metadata", None) or {}
                score = meta.get("relevance_score")
                if score is None:
                    return (1, 0.0)  # nulls last
                try:
                    return (0, -float(score))
                except (TypeError, ValueError):
                    return (1, 0.0)

            items.sort(key=_score)
            items = items[:100]

            result_list = []
            for src in items:
                meta = getattr(src, "metadata", None) or {}
                result_list.append({
                    "url": getattr(src, "url", None),
                    "title": getattr(src, "title", None),
                    "snippet": meta.get("snippet"),
                    "content": meta.get("content"),
                })

            logger.info(
                "FWK_EXISTING_SOURCES_LOADED chat_id=%s count=%d path=cached",
                str(chat_id_uuid)[:8], len(result_list),
            )
            return result_list
        except Exception as e:
            logger.warning(
                "FWK_EXISTING_SOURCES_CACHE_LOAD_FAILED error=%s",
                str(e)[:200],
            )
            # Fall through to legacy path if db is available.

    # Legacy path: ORM on public.sources.
    if db is None:
        return []

    try:
        from sqlalchemy import select

        from deep_research.models.source import Source

        chat_id_uuid = _to_uuid(chat_id)
        query = (
            select(Source)
            .where(Source.chat_id == chat_id_uuid)
            .order_by(Source.relevance_score.desc().nullslast())
            .limit(100)
        )
        result = await db.execute(query)
        sources = list(result.scalars().all())

        if not sources:
            return []

        out = []
        for src in sources:
            out.append({
                "url": src.url,
                "title": src.title,
                "snippet": getattr(src, "snippet", None),
                "content": getattr(src, "content", None),
            })

        logger.info(
            "FWK_EXISTING_SOURCES_LOADED chat_id=%s count=%d path=legacy",
            str(chat_id_uuid)[:8], len(out),
        )
        return out
    except Exception as e:
        logger.warning(
            "FWK_EXISTING_SOURCES_LOAD_FAILED error=%s",
            str(e)[:200],
        )
        return []


def _append_unique_tools(
    tools: list[Any],
    new_tools: list[Any],
) -> None:
    """Append tools while preserving uniqueness by definition name."""
    existing_names = {
        tool.definition.name
        for tool in tools
        if hasattr(tool, "definition")
    }
    for tool in new_tools:
        if not hasattr(tool, "definition"):
            continue
        tool_name = tool.definition.name
        if tool_name in existing_names:
            continue
        tools.append(tool)
        existing_names.add(tool_name)


def _is_enterprise_source_id(source_id: str) -> bool:
    """Return True when *source_id* identifies an enterprise source."""
    return source_id.startswith(("vs:", "genie:", "assistant:"))


def _source_id_from_user_source(source: Any) -> str | None:
    """Map a ``UserDataSource`` row to its discovery-style source ID."""
    source_type = getattr(source, "type", None)
    endpoint_identifier = getattr(source, "endpoint_identifier", "")
    config = getattr(source, "config", {}) or {}

    if source_type == "vector_search":
        index_name = config.get("index_name") or endpoint_identifier
        return f"vs:{index_name}" if index_name else None
    if source_type == "genie":
        space_id = config.get("space_id") or endpoint_identifier
        return f"genie:{space_id}" if space_id else None
    if source_type == "knowledge_assistant":
        endpoint_name = config.get("endpoint_name") or endpoint_identifier
        return f"assistant:{endpoint_name}" if endpoint_name else None
    return None


async def _load_enterprise_tools(
    config: OrchestrationConfig,
    db: Any,
    user_id: str | None,
    chat_id: str | None,  # noqa: ARG001
    storage_stack: Any | None = None,
) -> list[Any]:
    """Load enterprise tools from the app's tool factory.

    This stays in the app layer because enterprise tool loading depends
    on DB sessions, discovery cache, and user authentication.
    """
    tools: list[Any] = []

    if config.source_scope == "web_only":
        return tools

    disabled_source_ids = set(config.disabled_sources or [])
    has_explicit_source_selection = config.enabled_sources is not None
    selected_source_ids = [
        source_id
        for source_id in (config.enabled_sources or [])
        if _is_enterprise_source_id(source_id) and source_id not in disabled_source_ids
    ]
    remaining_source_ids = list(selected_source_ids)

    if has_explicit_source_selection and not selected_source_ids:
        logger.info(
            "FWK_ENTERPRISE_TOOLS_EMPTY_SELECTION source_scope=%s enabled_sources=%s "
            "disabled_sources=%s",
            config.source_scope,
            config.enabled_sources,
            config.disabled_sources,
        )
        return tools

    try:
        if (db is not None or storage_stack is not None) and user_id:
            if selected_source_ids:
                from deep_research.agent.tools.factory import create_tools_from_user_sources
                from deep_research.core.config import get_settings
                from deep_research.services._impl_factory import make_data_source_service

                service = make_data_source_service(
                    get_settings(), storage_stack, session=db,
                )
                sources, _ = await service.get_accessible_sources(user_id, only_valid=True)
                matched_sources = [
                    source for source in sources
                    if _source_id_from_user_source(source) in set(selected_source_ids)
                ]

                if matched_sources:
                    db_tools = await create_tools_from_user_sources(matched_sources)
                    _append_unique_tools(tools, db_tools)
                    if len(db_tools) == len(matched_sources):
                        matched_ids = {
                            source_id
                            for source_id in (
                                _source_id_from_user_source(source)
                                for source in matched_sources
                            )
                            if source_id is not None
                        }
                        remaining_source_ids = [
                            source_id for source_id in remaining_source_ids
                            if source_id not in matched_ids
                        ]
                    logger.info(
                        "FWK_ENTERPRISE_TOOLS_FROM_DB selected=%d loaded=%d remaining=%d",
                        len(selected_source_ids),
                        len(db_tools),
                        len(remaining_source_ids),
                    )
            else:
                from deep_research.agent.tools.factory import get_enabled_tools_for_user

                db_tools = await get_enabled_tools_for_user(
                    user_id=user_id,
                    user_token=config.user_token,
                    session=db,
                )
                _append_unique_tools(tools, db_tools)
                logger.info(
                    "FWK_ENTERPRISE_TOOLS_FROM_DB loaded=%d",
                    len(db_tools),
                )
    except Exception as e:
        logger.warning(
            "FWK_ENTERPRISE_TOOLS_LOAD_FAILED error=%s",
            str(e)[:200],
        )

    if remaining_source_ids and user_id:
        try:
            from deep_research.agent.tools.factory import create_tools_from_discovered_sources
            from deep_research.services.discovery_cache import get_discovery_cache

            cache = get_discovery_cache()
            cached_sources = await cache.get(user_id=user_id)

            if cached_sources:
                discovery_matched = [
                    source for source in cached_sources
                    if source.source_id in set(remaining_source_ids)
                ]
                if discovery_matched:
                    discovery_tools = await create_tools_from_discovered_sources(
                        discovery_matched
                    )
                    _append_unique_tools(tools, discovery_tools)
                    if len(discovery_tools) == len(discovery_matched):
                        matched_ids = {source.source_id for source in discovery_matched}
                        remaining_source_ids = [
                            source_id for source_id in remaining_source_ids
                            if source_id not in matched_ids
                        ]
                    logger.info(
                        "FWK_ENTERPRISE_TOOLS_FROM_DISCOVERY matched=%d loaded=%d remaining=%d",
                        len(discovery_matched),
                        len(discovery_tools),
                        len(remaining_source_ids),
                    )
        except Exception as e:
            logger.warning(
                "FWK_ENTERPRISE_DISCOVERY_LOAD_FAILED error=%s",
                str(e)[:200],
            )

    if remaining_source_ids:
        try:
            from deep_research.agent.tools.factory import create_tools_from_source_ids

            direct_tools = create_tools_from_source_ids(remaining_source_ids)
            _append_unique_tools(tools, direct_tools)
            logger.info(
                "FWK_ENTERPRISE_TOOLS_FROM_SOURCE_IDS requested=%d loaded=%d",
                len(remaining_source_ids),
                len(direct_tools),
            )
        except Exception as e:
            logger.warning(
                "FWK_ENTERPRISE_SOURCE_ID_LOAD_FAILED error=%s",
                str(e)[:200],
            )

    logger.info(
        "FWK_ENTERPRISE_TOOLS_RESOLVED source_scope=%s selected=%d loaded=%d tool_names=%s",
        config.source_scope,
        len(selected_source_ids),
        len(tools),
        [tool.definition.name for tool in tools if hasattr(tool, "definition")],
    )

    if (
        config.source_scope == "enterprise_only"
        and not tools
        and not getattr(config, "enabled_mcp_servers", None)
    ):
        logger.error(
            "FWK_ENTERPRISE_TOOLS_REQUIRED_BUT_EMPTY source_scope=%s "
            "enabled_sources=%s — enterprise_only mode has no tools",
            config.source_scope,
            config.enabled_sources,
        )

    return tools


async def _persist_simple_response(
    config: OrchestrationConfig,
    _db: Any,
    chat_id: str | None,
    user_id: str | None,
    query: str,
    response: str,
    _event_buffer: EventBuffer | None,
    *,
    storage_stack: Any = None,
) -> None:
    """Persist a simple query response (no research session).

    F-PERSIST-GUARDS: the `db is None` check was a proxy for "persistence is
    wired up", but the `persist_simple_message_*_independent` helpers open
    their own sessions via `get_session_maker()` — no `db` needed.
    When storage_stack is provided and impl==cached, skips session_maker.
    """
    if (
        config.message_id is None
        or chat_id is None
        or user_id is None
    ):
        logger.warning("FWK_SIMPLE_PERSISTENCE_SKIPPED reason=missing_params")
        return

    from deep_research.agent.persistence import (
        persist_research_session_completed_independent,
        persist_simple_message_independent,
        persist_simple_message_update_independent,
    )

    chat_id_uuid = _to_uuid(chat_id)

    try:
        if config.session_pre_created:
            await asyncio.shield(
                persist_simple_message_update_independent(
                    message_id=config.message_id,
                    content=response,
                    storage_stack=storage_stack,
                    chat_id=chat_id_uuid,
                )
            )
        else:
            await asyncio.shield(
                persist_simple_message_independent(
                    chat_id=chat_id_uuid,
                    user_id=user_id,
                    user_query=query,
                    message_id=config.message_id,
                    content=response,
                    storage_stack=storage_stack,
                )
            )

        logger.info(
            "FWK_SIMPLE_MODE_PERSISTED message_id=%s content_len=%d",
            str(config.message_id)[:8], len(response),
        )
    except asyncio.CancelledError:
        logger.warning("FWK_SIMPLE_PERSISTENCE_CANCELLED")
    except Exception as e:
        logger.warning(
            "FWK_SIMPLE_PERSISTENCE_FAILED error=%s",
            str(e)[:200],
        )

    # Transition the pre-created research session to COMPLETED. JobManager
    # creates the session at submit time (status=in_progress) for *every* job;
    # without this, a simple-classified query persists its answer but leaves the
    # session in_progress, so JobManager's end-of-stream check force-fails it
    # ("persistence_transition_missing") and the UI shows "research failed".
    # Kept in its own try so a status-transition failure never masks the
    # already-persisted answer message above.
    if config.research_session_id is not None:
        try:
            await asyncio.shield(
                persist_research_session_completed_independent(
                    research_session_id=config.research_session_id,
                    storage_stack=storage_stack,
                    chat_id=chat_id_uuid,
                )
            )
            logger.info(
                "FWK_SIMPLE_SESSION_COMPLETED research_session_id=%s",
                str(config.research_session_id)[:8],
            )
        except asyncio.CancelledError:
            logger.warning("FWK_SIMPLE_SESSION_COMPLETE_CANCELLED")
        except Exception as e:
            logger.warning(
                "FWK_SIMPLE_SESSION_COMPLETE_FAILED error=%s",
                str(e)[:200],
            )


async def _persist_delta(
    delta: Any,
    config: OrchestrationConfig,  # noqa: ARG001
    db: Any,  # noqa: ARG001
    chat_id: str | None,  # noqa: ARG001
    user_id: str | None,  # noqa: ARG001
) -> None:
    """Persist incremental state changes from DomainContextTracker.

    Called periodically during workflow execution.
    """
    if not delta._dirty:
        return

    # TODO(framework): Implement incremental persistence.
    # Currently all data is persisted at session completion only.
    # If the process crashes mid-research, accumulated sources/observations
    # from the delta are lost.
    logger.warning(
        "FWK_INCREMENTAL_PERSIST_NOT_IMPLEMENTED sources=%d observations=%d "
        "step_updates=%d (data will be persisted at session completion only)",
        len(delta.new_sources),
        len(delta.new_observations),
        len(delta.step_updates),
    )


async def _persist_completion(
    config: OrchestrationConfig,
    chat_id_uuid: UUID,
    user_id: str,
    query: str,
    final_report: str,
    event_buffer: EventBuffer | None,
    wf_state: WorkflowState | None = None,
    *,
    claims: list[Any] | None = None,
    verification_summary: Any | None = None,
    storage_stack: Any = None,
    promotion_trace: dict[str, Any] | None = None,
) -> dict[str, int] | None:
    """Persist final research session completion.

    Uses asyncio.shield to survive client disconnection.
    When storage_stack is provided and impl==cached, persistence helpers
    skip session_maker entirely.
    """
    try:
        if event_buffer is not None:
            # Two-phase: session was created at START
            from deep_research.agent.persistence import (
                persist_research_session_complete_update_independent,
            )

            # Build a minimal state-like object for the persistence function
            state_proxy = _build_state_proxy(
                config, final_report, wf_state,
                claims=claims, verification_summary=verification_summary,
                promotion_trace=promotion_trace,
            )

            assert config.research_session_id is not None
            assert config.message_id is not None
            counts = await asyncio.shield(
                persist_research_session_complete_update_independent(
                    chat_id=chat_id_uuid,
                    research_session_id=config.research_session_id,
                    agent_message_id=config.message_id,
                    state=state_proxy,
                    storage_stack=storage_stack,
                )
            )
            logger.info(
                "FWK_SESSION_COMPLETED research_session_id=%s report_len=%d claims=%d",
                str(config.research_session_id)[:8],
                len(final_report),
                len(claims or []),
            )
            return counts
        else:
            # Fallback: single-phase persistence
            from deep_research.agent.persistence import (
                persist_complete_research_independent,
            )

            state_proxy = _build_state_proxy(
                config, final_report, wf_state,
                claims=claims, verification_summary=verification_summary,
                promotion_trace=promotion_trace,
            )

            assert config.message_id is not None
            assert config.research_session_id is not None
            counts = await asyncio.shield(
                persist_complete_research_independent(
                    chat_id=chat_id_uuid,
                    user_id=user_id,
                    user_query=query,
                    message_id=config.message_id,
                    research_session_id=config.research_session_id,
                    research_depth=config.research_depth,
                    state=state_proxy,
                    storage_stack=storage_stack,
                )
            )
            return counts
    except asyncio.CancelledError:
        logger.warning("FWK_PERSISTENCE_CANCELLED")
        return None
    except Exception as e:
        logger.warning("FWK_PERSISTENCE_FAILED error=%s", str(e)[:200])
        # Mark session as FAILED
        if config.research_session_id and config.message_id:
            try:
                from deep_research.agent.persistence import (
                    persist_research_session_failed_independent,
                )
                await persist_research_session_failed_independent(
                    research_session_id=config.research_session_id,
                    agent_message_id=config.message_id,
                    error_message=str(e)[:500],
                    storage_stack=storage_stack,
                    chat_id=chat_id_uuid,
                )
            except Exception:
                # PR4 CRITICAL fix: do not silently drop the failure-marker.
                # Without this, the DB row stays in 'running' forever after
                # the completion-persist failure (silent data loss).
                logger.exception(
                    "FWK_FAILURE_PERSISTENCE_FAILED research_session_id=%s",
                    str(config.research_session_id)[:8],
                )
        return None


def _get_pool_sources(wf_state: WorkflowState | None) -> list[Any]:
    """Extract source objects from the workflow state's sources pool.

    Returns the pool items in insertion order (matching the [N] indices
    used by the synthesizer).
    """
    if wf_state is None:
        return []
    sources_pool = wf_state.pools.get("sources") if wf_state.pools else None
    if sources_pool is None or sources_pool.count() == 0:
        return []
    return list(sources_pool.get_recent(sources_pool.count()))


def _adapt_framework_evidence(fw_evidence: Any) -> Any:
    """Convert framework :class:`Evidence` into the app's :class:`EvidenceInfo`."""
    from deep_research.agent.state import EvidenceInfo

    if fw_evidence is None:
        return None
    return EvidenceInfo(
        source_url=fw_evidence.source_url,
        quote_text=fw_evidence.quote_text,
        start_offset=fw_evidence.start_offset,
        end_offset=fw_evidence.end_offset,
        section_heading=fw_evidence.section_heading,
        relevance_score=fw_evidence.relevance_score,
        has_numeric_content=fw_evidence.has_numeric_content,
    )


def _adapt_framework_claim(fw_claim: Any) -> Any:
    """Convert framework :class:`Claim` into the app's :class:`ClaimInfo`."""
    from deep_research.agent.state import ClaimInfo

    return ClaimInfo(
        claim_text=fw_claim.claim_text,
        claim_type=fw_claim.claim_type,
        position_start=fw_claim.position_start,
        position_end=fw_claim.position_end,
        evidence=_adapt_framework_evidence(fw_claim.evidence),
        confidence_level=fw_claim.confidence_level,
        verification_verdict=fw_claim.verification_verdict,
        verification_reasoning=fw_claim.verification_reasoning,
        abstained=fw_claim.abstained,
        citation_key=fw_claim.citation_key,
        citation_keys=fw_claim.citation_keys,
        from_free_block=fw_claim.from_free_block,
    )


def _adapt_framework_summary(fw_summary: Any) -> Any:
    """Convert framework :class:`SummaryInfo` into app's :class:`VerificationSummaryInfo`."""
    from deep_research.agent.state import VerificationSummaryInfo

    if fw_summary is None:
        return None
    return VerificationSummaryInfo(
        total_claims=fw_summary.total_claims,
        supported_count=fw_summary.supported_count,
        partial_count=fw_summary.partial_count,
        unsupported_count=fw_summary.unsupported_count,
        contradicted_count=fw_summary.contradicted_count,
        abstained_count=fw_summary.abstained_count,
        unsupported_rate=fw_summary.unsupported_rate,
        contradicted_rate=fw_summary.contradicted_rate,
        warning=fw_summary.warning,
        citation_corrections=fw_summary.citation_corrections,
        claim_revisions=fw_summary.claim_revisions,
        atomic_facts_total=fw_summary.atomic_facts_total,
        atomic_facts_verified=fw_summary.atomic_facts_verified,
        atomic_facts_softened=fw_summary.atomic_facts_softened,
        claims_fully_verified=fw_summary.claims_fully_verified,
        claims_partially_softened=fw_summary.claims_partially_softened,
        claims_fully_softened=fw_summary.claims_fully_softened,
        external_searches=fw_summary.external_searches,
        new_sources_added=fw_summary.new_sources_added,
    )


def _extract_verification_from_framework_state(
    wf_state: WorkflowState | None,
    sources: list[Any],
) -> tuple[list[Any], Any]:
    """Backward-compat shim — delegates to the framework's extraction utility.

    Original ~116-LoC implementation relocated to
    :mod:`databricks_deep_research.citation.extraction`. Field names match
    1:1; we only adapt the wrapper types back to the app's dataclasses.
    """
    from databricks_deep_research.citation.extraction import (
        extract_verification as _fw_extract,
    )

    summary = _fw_extract(wf_state, sources)
    claims = [_adapt_framework_claim(c) for c in summary.claims]
    return claims, _adapt_framework_summary(summary.summary)


def _extract_verification_from_report(
    final_report: str,
    sources: list[Any],
) -> tuple[list[Any], Any]:
    """Backward-compat shim — delegates to the framework's report extractor.

    Original ~136-LoC implementation relocated to
    :mod:`databricks_deep_research.citation.extraction`. Field names match
    1:1; we only adapt the wrapper types back to the app's dataclasses.
    """
    from databricks_deep_research.citation.extraction import (
        extract_verification_from_report as _fw_extract_report,
    )

    summary = _fw_extract_report(final_report, sources)
    claims = [_adapt_framework_claim(c) for c in summary.claims]
    return claims, _adapt_framework_summary(summary.summary)


def _build_state_proxy(
    config: OrchestrationConfig,
    final_report: str,
    wf_state: WorkflowState | None = None,
    *,
    claims: list[Any] | None = None,
    verification_summary: Any | None = None,
    promotion_trace: dict[str, Any] | None = None,
) -> Any:
    """Build a minimal object that satisfies the persistence layer's state interface.

    The persistence functions expect a ResearchState-like object with
    specific attributes.  Rather than importing and constructing a full
    ResearchState, we create a lightweight proxy.

    IMPORTANT: This proxy must include ALL attributes accessed by:
      - persist_research_session_complete_update_independent() (persistence.py:690-718)
      - persist_complete_research() (persistence.py:367-451)
      - persist_research_data() (persistence.py:89-158)
    If you add a new state.* access in persistence, add it here too.
    See TestBuildStateProxy for the contract test.
    """
    from types import SimpleNamespace

    from deep_research.agent.state import SourceInfo as AppSourceInfo

    sources: list[AppSourceInfo] = []
    if wf_state is not None:
        sources_pool = wf_state.pools.get("sources") if wf_state.pools else None
        if sources_pool is not None and sources_pool.count() > 0:
            for raw_source in sources_pool.get_recent(sources_pool.count()):
                if isinstance(raw_source, dict):
                    url = raw_source.get("url")
                    if not url:
                        continue
                    content = raw_source.get("content")
                    snippet = raw_source.get("snippet") or (
                        content[:500] if isinstance(content, str) else None
                    )
                    sources.append(AppSourceInfo(
                        url=str(url),
                        title=raw_source.get("title") or raw_source.get("filename"),
                        snippet=snippet,
                        content=content if isinstance(content, str) else None,
                        relevance_score=raw_source.get("relevance_score"),
                        source_type=(
                            raw_source.get("source_type")
                            or raw_source.get("type")
                            or "web"
                        ),
                    ))
                    continue

                url = getattr(raw_source, "url", None)
                if not url:
                    continue
                content = getattr(raw_source, "content", None)
                snippet = getattr(raw_source, "snippet", None) or (
                    content[:500] if isinstance(content, str) else None
                )
                sources.append(AppSourceInfo(
                    url=str(url),
                    title=getattr(raw_source, "title", None),
                    snippet=snippet,
                    content=content if isinstance(content, str) else None,
                    relevance_score=getattr(raw_source, "relevance_score", None),
                    source_type=(
                        getattr(raw_source, "source_type", None)
                        or getattr(raw_source, "type", None)
                        or "web"
                    ),
                ))

    return SimpleNamespace(
        final_report=final_report,
        final_report_structured=None,
        sources=sources,
        all_observations=[],
        current_plan=None,
        plan_iterations=0,
        query=config.system_instructions or "",
        session_id=config.research_session_id,
        output_format=config.output_format,
        output_schema=config.output_schema,
        enable_citation_verification=config.verify_sources,
        synthesis_mode=config.synthesis_mode,
        # Persistence-required attributes (previously missing → AttributeError)
        reflection_history=[],       # L431, L711: [r.to_dict() for r in ...]
        current_step_index=0,        # L432, L715: integer column
        query_mode=config.query_mode,  # L427: string column (Path 2 only)
        claims=claims or [],         # L112, L120, L253: iteration + len
        verification_summary=verification_summary,   # L249, L266: .to_dict() if truthy
        promotion_trace=promotion_trace,   # spec 6.1: value-free run trace
    )


__all__ = [
    "stream_research_via_framework",
    "_safe_uuid",
    "_to_uuid",
    "_to_sse_event",
    "_build_state_proxy",
]
