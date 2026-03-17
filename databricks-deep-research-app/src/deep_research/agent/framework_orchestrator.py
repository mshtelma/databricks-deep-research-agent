"""Framework-based orchestrator — thin wrapper around databricks-deep-research.

Replaces the 3769 LOC monolith orchestrator with a clean delegation to the
multi-agent framework.  The pipeline is:

    config_translator.translate(config) → WorkflowDefinition
    WorkflowExecutor(definition, llm_client, ...).execute(state) → yields StreamEvent
    DomainContextTracker.process_event(event) → list[AppSSEEvent]
    PersistenceDelta → DB writes

All app-specific concerns (persistence, SSE format, cancellation, error
handling) are handled here.  The framework handles workflow execution.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from databricks_deep_research.events.types import (
    AgentStreamChunkEvent,
    CoordinatorClassifiedEvent,
    ItemCompletedEvent,
    ReplanTriggeredEvent,
    WorkflowCompletedEvent as FwkWorkflowCompletedEvent,
)
from databricks_deep_research.workflow.context import ExecutionContext
from databricks_deep_research.workflow.definition import WorkflowDefinition
from databricks_deep_research.workflow.executor import WorkflowExecutor
from databricks_deep_research.workflow.state import WorkflowState

from deep_research.agent.adapters.config_translator import translate
from deep_research.agent.adapters.domain_context import (
    AppSSEEvent,
    DomainContextTracker,
)
from deep_research.agent.adapters.llm_adapter import create_framework_llm_client
from deep_research.agent.adapters.tool_adapter import create_framework_tools
from deep_research.core.tracing import safe_mlflow_run, safe_tool_span, safe_update_trace
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
from deep_research.services.llm.client import LLMClient
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

# Regex matching [N] numeric citation markers produced by the framework synthesizer.
# Captures the integer N.  Matches [1], [12], [1][2], etc.
_NUMERIC_CITATION_RE = __import__("re").compile(r"\[(\d+)\]")


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
    from databricks_deep_research.workflow.loader import load_workflow_from_dict

    from deep_research.plugins.base import WorkflowProviderPlugin

    if plugin_manager is not None:
        for plugin in plugin_manager.get_plugins():
            if not isinstance(plugin, WorkflowProviderPlugin):
                continue
            try:
                result = plugin.get_workflow_yaml(ref)
                if result is not None:
                    logger.info(
                        "WORKFLOW_PLUGIN_RESOLVED ref=%s plugin=%s type=%s",
                        ref,
                        getattr(plugin, "name", type(plugin).__name__),
                        type(result).__name__,
                    )
                    # Accept str (YAML), dict, or WorkflowDefinition
                    if isinstance(result, WorkflowDefinition):
                        defn = result
                    elif isinstance(result, dict):
                        defn = load_workflow_from_dict(result)
                    else:
                        defn = load_workflow_from_string(result)
                    _filter_workflow_tools(defn, tool_names)
                    return defn
            except Exception:
                logger.exception(
                    "WORKFLOW_PLUGIN_ERROR ref=%s plugin=%s",
                    ref,
                    getattr(plugin, "name", type(plugin).__name__),
                )

    raise ValueError(
        f"workflow_ref={ref!r} set but no plugin resolved it. "
        f"Registered plugins: {[p.name for p in (plugin_manager.get_plugins() if plugin_manager else [])]}"
    )


def _filter_workflow_tools(
    defn: WorkflowDefinition, available_tools: list[str]
) -> None:
    """Intersect agent node tool lists with runtime-available tools.

    Walks the workflow tree. For each agent node with a tools config,
    removes tool names that aren't in available_tools.
    """
    available = set(available_tools)
    _filter_node_tools(defn.root, available)


def _filter_node_tools(node: Any, available: set[str]) -> None:
    """Recursively filter tools on agent nodes."""
    if node.type.value == "agent":
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
    if node.type.value == "plan_and_execute":
        body = node.config.get("body")
        if hasattr(body, "config"):
            tools = body.config.get("tools")
            if isinstance(tools, list):
                body.config["tools"] = [t for t in tools if t in available]


async def stream_research_via_framework(
    query: str,
    llm: LLMClient,
    brave_client: BraveSearchClient,
    crawler: "WebCrawler",
    conversation_history: list[dict[str, str]] | None = None,
    session_id: UUID | None = None,
    user_id: str | None = None,
    chat_id: str | None = None,
    config: "OrchestrationConfig | None" = None,
    db: "AsyncSession | None" = None,
    plugin_manager: "PluginManager | None" = None,
    plugin_data: dict[str, Any] | None = None,
) -> AsyncGenerator[StreamEvent | str, None]:
    """Stream research via the multi-agent framework.

    Drop-in replacement for ``stream_research()`` with identical external
    interface.  Internally delegates to the framework executor.

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

    Yields:
        StreamEvent objects and synthesis content chunks (strings).
    """
    from deep_research.agent.orchestration_config import (
        get_default_orchestration_config,
    )

    config = config or get_default_orchestration_config()
    start_time = time.perf_counter()
    event_buffer: EventBuffer | None = None
    steps_executed = 0
    steps_skipped = 0
    plan_iterations = 0
    final_report: str | None = None
    structured_output: dict | None = None
    simple_response: str | None = None
    _synthesis_chunks: list[str] = []
    wf_state: WorkflowState | None = None

    # ------------------------------------------------------------------
    # 1. Session start (two-phase persistence)
    # ------------------------------------------------------------------
    try:
        if (
            db is not None
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
                )
                event_buffer = EventBuffer(config.research_session_id)
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
            event_buffer = EventBuffer(config.research_session_id)

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
                _SpanType.CHAIN if _SpanType else None,
                {"research.query": query[:200], "research.use_framework": True},
            ):
                # Trace metadata for MLflow correlation
                if user_id or chat_id:
                    trace_metadata: dict[str, str] = {}
                    if user_id:
                        trace_metadata["mlflow.trace.user"] = user_id
                    if chat_id:
                        trace_metadata["mlflow.trace.session"] = chat_id
                    if trace_metadata:
                        safe_update_trace(trace_metadata)

                # -- 2. Build framework execution context --
                framework_llm = create_framework_llm_client(
                    llm,
                    model_overrides=config.model_overrides,
                )

                # Load file search tool (Step 3a)
                file_search_tool = await _load_file_search_tool(
                    config, db, user_id, chat_id,
                )

                logger.info(
                    "FWK_TOOL_CREATION domain_filter=%s domain_filter_type=%s",
                    config.domain_filter,
                    type(config.domain_filter).__name__ if config.domain_filter else "None",
                )

                framework_tools = await create_framework_tools(
                    brave_client=brave_client,
                    crawler=crawler,
                    domain_filter_config=config.domain_filter,
                    enterprise_tools=await _load_enterprise_tools(
                        config, db, user_id, chat_id
                    ),
                    user_token=config.user_token,
                    file_search_tool=file_search_tool,
                    chat_id=chat_id,
                    user_id=user_id,
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
                    _existing = {t.definition.name for t in framework_tools}
                    for _plugin in plugin_manager.get_plugins():
                        if not hasattr(_plugin, "get_tools"):
                            continue
                        try:
                            for _tool in _plugin.get_tools(_tool_ctx):
                                if _tool.definition.name not in _existing:
                                    framework_tools.append(_tool)
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

                # Build ToolResolver with all tools registered as overrides.
                # This replaces the old ToolRegistry approach — the resolver
                # handles both name-string refs (new) and dict refs (legacy).
                from databricks_deep_research.tools.resolver import ToolResolver

                tool_resolver = ToolResolver()
                for tool in framework_tools:
                    tool_resolver.override(tool.definition.name, tool)

                context = ExecutionContext(
                    llm_client=framework_llm,
                    enterprise_tools=framework_tools,
                    model_overrides=config.model_overrides or {},
                    user_token=config.user_token,
                )

                # -- 3. Translate config to workflow definition --
                tool_names = [t.definition.name for t in framework_tools]

                # Validate: enterprise_only requires at least one non-web tool
                if config.source_scope == "enterprise_only":
                    web_names = {"web_search", "web_crawl"}
                    enterprise_names = [n for n in tool_names if n not in web_names]
                    if not enterprise_names:
                        logger.error(
                            "FWK_NO_ENTERPRISE_TOOLS source_scope=enterprise_only "
                            "tool_names=%s — research will proceed with no enterprise data",
                            tool_names,
                        )

                workflow_def = _resolve_workflow(config, tool_names, plugin_manager)

                logger.info(
                    "FWK_WORKFLOW_TRANSLATED workflow_id=%s tool_names=%s",
                    workflow_def.id,
                    tool_names,
                )

                # -- 4. Execute workflow and stream events --
                executor = WorkflowExecutor(
                    workflow_def,
                    framework_llm,
                    tool_resolver=tool_resolver,
                    context=context,
                )
                tracker = DomainContextTracker()

                wf_state = WorkflowState(query=query)
                if conversation_history:
                    wf_state.append("init", "conversation_history", conversation_history)

                # Load existing sources for follow-up queries (Step 4)
                existing_sources = await _load_existing_sources(db, chat_id)
                if existing_sources:
                    wf_state.append("init", "existing_sources", existing_sources)

                research_timeout = config.research_timeout_seconds

                try:
                    async with asyncio.timeout(research_timeout):
                        async for fw_event in executor.execute(wf_state):
                            # Detect simple query short-circuit (Step 2)
                            if isinstance(fw_event, CoordinatorClassifiedEvent) and fw_event.is_simple and fw_event.direct_response:
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

                            # Skip remaining events after simple response detected
                            if simple_response is not None:
                                continue

                            # Map framework events to app SSE events
                            app_events = tracker.process_event(fw_event)

                            for app_evt in app_events:
                                sse_event = _to_sse_event(app_evt)
                                if sse_event:
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

            # -- 5. Session completion --

            # Simple mode persistence (Step 2)
            if simple_response is not None:
                final_report = simple_response
                await _persist_simple_response(
                    config, db, chat_id, user_id, query, simple_response,
                    event_buffer,
                )
                # Yield persistence event for simple mode
                if (
                    config.message_id is not None
                    and chat_id is not None
                ):
                    chat_title = query[:47] + "..." if len(query) > 50 else query
                    yield PersistenceCompletedEvent(
                        chat_id=str(chat_id),
                        message_id=str(config.message_id),
                        research_session_id=None,
                        chat_title=chat_title,
                        was_draft=config.is_draft,
                        counts={"messages": 1},
                    )
            else:
                # Full research persistence
                if event_buffer:
                    try:
                        await event_buffer.flush()
                    except Exception as e:
                        logger.warning("FWK_EVENT_BUFFER_FLUSH_FAILED error=%s", str(e)[:200])

                if (
                    db is not None
                    and config.message_id is not None
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
                    )
                    if counts:
                        chat_title = query[:47] + "..." if len(query) > 50 else query
                        yield PersistenceCompletedEvent(
                            chat_id=str(chat_id_uuid),
                            message_id=str(config.message_id),
                            research_session_id=str(config.research_session_id),
                            chat_title=chat_title,
                            was_draft=config.is_draft,
                            counts=counts,
                        )
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
            try:
                await event_buffer.flush()
            except Exception:
                pass
            try:
                from deep_research.agent.persistence import (
                    persist_research_session_failed_independent,
                )
                await persist_research_session_failed_independent(
                    research_session_id=config.research_session_id,
                    agent_message_id=config.message_id,
                    error_message=str(e)[:500],
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 6. Final completion event (always emitted)
    # ------------------------------------------------------------------
    total_duration_ms = int((time.perf_counter() - start_time) * 1000)
    yield ResearchCompletedEvent(
        session_id=session_id or uuid4(),
        total_steps_executed=steps_executed,
        total_steps_skipped=steps_skipped,
        plan_iterations=plan_iterations,
        total_duration_ms=total_duration_ms,
        final_report=final_report,
        structured_output=structured_output,
    )


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
                tool_args={},
                call_number=0,
            )
        elif progress_type == "tool_result":
            from deep_research.schemas.streaming import ToolResultEvent as AppToolResultEvent

            return AppToolResultEvent(
                tool_name=data.get("tool_name", ""),
                result_preview=data.get("result_summary", ""),
                sources_crawled=data.get("source_count", 0),
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
        try:
            await buffer.add_event(event)
        except Exception:
            pass


async def _load_file_search_tool(
    config: "OrchestrationConfig",
    db: Any,
    user_id: str | None,
    chat_id: str | None,
) -> Any | None:
    """Load file search tool from explicit file_ids or auto-discovery.

    Returns an app-level FileSearchTool instance, or None.
    """
    if config.file_ids and db is not None and user_id:
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
    if db is not None and user_id and chat_id:
        try:
            from uuid import UUID as _UUID

            from deep_research.services.file_upload_service import FileUploadService

            service = FileUploadService(db)
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


async def _load_existing_sources(
    db: Any,
    chat_id: str | None,
) -> list[dict[str, Any]]:
    """Load existing sources from prior messages for follow-up queries.

    Returns a list of source dicts suitable for seeding into the framework's
    initial_state.
    """
    if db is None or not chat_id:
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

        result = []
        for src in sources:
            result.append({
                "url": src.url,
                "title": src.title,
                "snippet": getattr(src, "snippet", None),
                "content": getattr(src, "content", None),
            })

        logger.info(
            "FWK_EXISTING_SOURCES_LOADED chat_id=%s count=%d",
            str(chat_id_uuid)[:8], len(result),
        )
        return result
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
    config: "OrchestrationConfig",
    db: Any,
    user_id: str | None,
    chat_id: str | None,
) -> list[Any]:
    """Load enterprise tools from the app's tool factory.

    This stays in the app layer because enterprise tool loading depends
    on DB sessions, discovery cache, and user authentication.
    """
    tools: list[Any] = []

    if config.source_scope == "web_only":
        return tools

    disabled_source_ids = set(config.disabled_sources or [])
    selected_source_ids = [
        source_id
        for source_id in (config.enabled_sources or [])
        if _is_enterprise_source_id(source_id) and source_id not in disabled_source_ids
    ]
    remaining_source_ids = list(selected_source_ids)

    try:
        if db is not None and user_id:
            if selected_source_ids:
                from deep_research.agent.tools.factory import create_tools_from_user_sources
                from deep_research.services.data_source_service import DataSourceService

                service = DataSourceService(db)
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
                matched_sources = [
                    source for source in cached_sources
                    if source.source_id in set(remaining_source_ids)
                ]
                if matched_sources:
                    discovery_tools = await create_tools_from_discovered_sources(
                        matched_sources
                    )
                    _append_unique_tools(tools, discovery_tools)
                    if len(discovery_tools) == len(matched_sources):
                        matched_ids = {source.source_id for source in matched_sources}
                        remaining_source_ids = [
                            source_id for source_id in remaining_source_ids
                            if source_id not in matched_ids
                        ]
                    logger.info(
                        "FWK_ENTERPRISE_TOOLS_FROM_DISCOVERY matched=%d loaded=%d remaining=%d",
                        len(matched_sources),
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

    if config.source_scope == "enterprise_only" and not tools:
        logger.error(
            "FWK_ENTERPRISE_TOOLS_REQUIRED_BUT_EMPTY source_scope=%s "
            "enabled_sources=%s — enterprise_only mode has no tools",
            config.source_scope,
            config.enabled_sources,
        )

    return tools


async def _persist_simple_response(
    config: "OrchestrationConfig",
    db: Any,
    chat_id: str | None,
    user_id: str | None,
    query: str,
    response: str,
    event_buffer: EventBuffer | None,
) -> None:
    """Persist a simple query response (no research session)."""
    if (
        db is None
        or config.message_id is None
        or chat_id is None
        or user_id is None
    ):
        logger.warning("FWK_SIMPLE_PERSISTENCE_SKIPPED reason=missing_params")
        return

    from deep_research.agent.persistence import (
        persist_simple_message_independent,
        persist_simple_message_update_independent,
    )

    try:
        chat_id_uuid = _to_uuid(chat_id)

        if config.session_pre_created:
            await asyncio.shield(
                persist_simple_message_update_independent(
                    message_id=config.message_id,
                    content=response,
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


async def _persist_delta(
    delta: Any,
    config: "OrchestrationConfig",
    db: Any,
    chat_id: str | None,
    user_id: str | None,
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
    config: "OrchestrationConfig",
    chat_id_uuid: UUID,
    user_id: str,
    query: str,
    final_report: str,
    event_buffer: EventBuffer | None,
    wf_state: WorkflowState | None = None,
    *,
    claims: list[Any] | None = None,
    verification_summary: Any | None = None,
) -> dict[str, int] | None:
    """Persist final research session completion.

    Uses asyncio.shield to survive client disconnection.
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
            )

            counts = await asyncio.shield(
                persist_research_session_complete_update_independent(
                    chat_id=chat_id_uuid,
                    research_session_id=config.research_session_id,
                    agent_message_id=config.message_id,
                    state=state_proxy,
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
            )

            counts = await asyncio.shield(
                persist_complete_research_independent(
                    chat_id=chat_id_uuid,
                    user_id=user_id,
                    user_query=query,
                    message_id=config.message_id,
                    research_session_id=config.research_session_id,
                    research_depth=config.research_depth,
                    state=state_proxy,
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
                )
            except Exception:
                pass
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


def _extract_verification_from_framework_state(
    wf_state: WorkflowState | None,
    sources: list[Any],
) -> tuple[list[Any], Any]:
    """Prefer framework-native claims/summary over markdown re-parsing."""
    from deep_research.agent.state import ClaimInfo, EvidenceInfo, VerificationSummaryInfo

    if wf_state is None:
        return [], None

    raw_claims = wf_state.get("claims") if hasattr(wf_state, "get") else None
    raw_summary = (
        wf_state.get("verification_summary") if hasattr(wf_state, "get") else None
    )

    if not raw_claims and not raw_summary:
        return [], None

    claims: list[ClaimInfo] = []
    if isinstance(raw_claims, list):
        for raw_claim in raw_claims:
            if not isinstance(raw_claim, dict):
                continue

            evidence_raw = raw_claim.get("evidence")
            evidence: EvidenceInfo | None = None
            if isinstance(evidence_raw, dict) and evidence_raw.get("source_url"):
                evidence = EvidenceInfo(
                    source_url=str(evidence_raw.get("source_url", "") or ""),
                    quote_text=str(evidence_raw.get("quote_text", "") or ""),
                    start_offset=evidence_raw.get("start_offset"),
                    end_offset=evidence_raw.get("end_offset"),
                    section_heading=evidence_raw.get("section_heading"),
                    relevance_score=evidence_raw.get("relevance_score"),
                    has_numeric_content=bool(
                        evidence_raw.get("has_numeric_content", False)
                    ),
                )
            else:
                citation_keys = raw_claim.get("citation_keys") or []
                citation_key = (
                    raw_claim.get("citation_key")
                    or (citation_keys[0] if citation_keys else None)
                )
                if isinstance(citation_key, str) and citation_key.isdigit():
                    source_index = int(citation_key)
                    if 0 <= source_index < len(sources):
                        source = sources[source_index]
                        if isinstance(source, dict):
                            evidence = EvidenceInfo(
                                source_url=str(source.get("url", "") or ""),
                                quote_text=str(source.get("snippet", "") or ""),
                            )
                        else:
                            evidence = EvidenceInfo(
                                source_url=str(getattr(source, "url", "") or ""),
                                quote_text=str(getattr(source, "snippet", "") or ""),
                            )

            claims.append(
                ClaimInfo(
                    claim_text=str(raw_claim.get("claim_text", "") or ""),
                    claim_type=str(raw_claim.get("claim_type", "general") or "general"),
                    position_start=int(raw_claim.get("position_start", 0) or 0),
                    position_end=int(raw_claim.get("position_end", 0) or 0),
                    evidence=evidence,
                    confidence_level=raw_claim.get("confidence_level"),
                    verification_verdict=raw_claim.get("verification_verdict"),
                    verification_reasoning=raw_claim.get("verification_reasoning"),
                    abstained=bool(raw_claim.get("abstained", False)),
                    citation_key=raw_claim.get("citation_key"),
                    citation_keys=raw_claim.get("citation_keys"),
                    from_free_block=bool(raw_claim.get("from_free_block", False)),
                )
            )

    summary = None
    if isinstance(raw_summary, dict):
        summary = VerificationSummaryInfo(
            total_claims=int(raw_summary.get("total_claims", 0) or 0),
            supported_count=int(
                raw_summary.get(
                    "verified_claims",
                    raw_summary.get("supported_count", 0),
                )
                or 0
            ),
            partial_count=int(raw_summary.get("partial_count", 0) or 0),
            unsupported_count=int(
                raw_summary.get(
                    "softened_claims",
                    raw_summary.get("unsupported_count", 0),
                )
                or 0
            ),
            contradicted_count=int(
                raw_summary.get(
                    "removed_claims",
                    raw_summary.get("contradicted_count", 0),
                )
                or 0
            ),
            abstained_count=int(raw_summary.get("abstained_count", 0) or 0),
            unsupported_rate=float(raw_summary.get("unsupported_rate", 0.0) or 0.0),
            contradicted_rate=float(raw_summary.get("contradicted_rate", 0.0) or 0.0),
            warning=bool(raw_summary.get("warning", False)),
            citation_corrections=int(
                raw_summary.get(
                    "corrected_citations",
                    raw_summary.get("citation_corrections", 0),
                )
                or 0
            ),
        )

    return claims, summary


def _extract_verification_from_report(
    final_report: str,
    sources: list[Any],
) -> tuple[list[Any], Any]:
    """Extract claims and verification summary from a report with [N] markers.

    Parses the final report for ``[N]`` numeric citation markers produced by
    the framework synthesizer.  Each sentence containing markers becomes a
    claim linked to the corresponding source(s) in the pool.

    Args:
        final_report: The synthesizer's final report text.
        sources: Ordered list of source objects (AppSourceInfo) from the pool.

    Returns:
        Tuple of (claims_list, verification_summary) where claims are
        ``ClaimInfo`` objects and summary is ``VerificationSummaryInfo``.
    """
    from deep_research.agent.state import ClaimInfo, EvidenceInfo, VerificationSummaryInfo

    if not final_report or not sources:
        return [], None

    # Split into sentences (rough but sufficient for claim extraction).
    # We split on sentence-ending punctuation followed by whitespace,
    # preserving markdown structure.
    import re

    # Split on ". ", "! ", "? " or newlines, keeping delimiters with the sentence
    sentences = re.split(r"(?<=[.!?])\s+|\n+", final_report)

    # Pre-scan: detect indexing convention.
    # If any [0] marker exists → 0-indexed (evidence pool style).
    # If smallest marker is >= 1 → 1-indexed (synthesizer prompt style).
    all_marker_indices: set[int] = set()
    for s in sentences:
        s = s.strip()
        if s:
            all_marker_indices.update(int(m) for m in _NUMERIC_CITATION_RE.findall(s))
    index_offset = 0 if 0 in all_marker_indices else 1

    logger.info(
        "FWK_VERIFICATION_INDEX_DETECT markers=%s offset=%d sources=%d",
        sorted(all_marker_indices)[:10],
        index_offset,
        len(sources),
    )

    claims: list[ClaimInfo] = []
    position = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            position += 1
            continue

        # Find all [N] markers in this sentence
        markers = _NUMERIC_CITATION_RE.findall(sentence)
        if not markers:
            position += len(sentence) + 1
            continue

        # Clean the sentence text (remove [N] markers for claim_text)
        claim_text = _NUMERIC_CITATION_RE.sub("", sentence).strip()
        if len(claim_text) < 10:  # Skip trivially short "claims"
            position += len(sentence) + 1
            continue

        # Map marker indices to sources
        cited_indices = sorted(set(int(m) for m in markers))

        # Build evidence from the first valid cited source
        evidence: EvidenceInfo | None = None
        # Citation keys MUST be the numeric index strings ("1", "2", etc.)
        # because the frontend markdown parser extracts [N] and uses
        # the number as the citationKey for lookup in citationData map.
        citation_keys: list[str] = [str(idx) for idx in cited_indices]
        for idx in cited_indices:
            pool_idx = idx - index_offset
            if 0 <= pool_idx < len(sources):
                source = sources[pool_idx]
                # Pool items can be dicts or objects (AppSourceInfo)
                if isinstance(source, dict):
                    source_url = source.get("url", "") or ""
                    source_snippet = source.get("snippet", "") or ""
                else:
                    source_url = getattr(source, "url", "") or ""
                    source_snippet = getattr(source, "snippet", "") or ""
                if evidence is None and source_url:
                    evidence = EvidenceInfo(
                        source_url=source_url,
                        quote_text=source_snippet[:300] if source_snippet else "",
                    )
                logger.debug(
                    "FWK_CLAIM_EXTRACTED key=%s pool_idx=%d evidence_url=%s",
                    str(idx),
                    pool_idx,
                    source_url[:60] if source_url else "none",
                )

        # Detect numeric claims (contains numbers like $35.1B, 1.2%, etc.)
        has_numbers = bool(re.search(r"\$[\d,.]+|\d+\.\d+%|\d{2,}", claim_text))

        claim = ClaimInfo(
            claim_text=claim_text,
            claim_type="numeric" if has_numbers else "general",
            position_start=position,
            position_end=position + len(sentence),
            evidence=evidence,
            confidence_level="high",
            verification_verdict="supported",
            verification_reasoning="Cited by synthesizer with source reference",
            abstained=False,
            citation_key=citation_keys[0] if citation_keys else None,
            citation_keys=citation_keys if len(citation_keys) > 1 else None,
        )
        claims.append(claim)
        position += len(sentence) + 1

    # Build verification summary
    summary = VerificationSummaryInfo(
        total_claims=len(claims),
        supported_count=len(claims),
        partial_count=0,
        unsupported_count=0,
        contradicted_count=0,
        abstained_count=0,
        unsupported_rate=0.0,
        contradicted_rate=0.0,
        warning=False,
        citation_corrections=0,
    )

    logger.info(
        "FWK_VERIFICATION_EXTRACTED claims=%d sources_cited=%d",
        len(claims),
        len({c.evidence.source_url for c in claims if c.evidence}),
    )

    return claims, summary if claims else None


def _build_state_proxy(
    config: "OrchestrationConfig",
    final_report: str,
    wf_state: WorkflowState | None = None,
    *,
    claims: list[Any] | None = None,
    verification_summary: Any | None = None,
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
    )


__all__ = [
    "stream_research_via_framework",
    "_safe_uuid",
    "_to_uuid",
    "_to_sse_event",
    "_build_state_proxy",
]
