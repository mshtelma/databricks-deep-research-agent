"""Workflow executor — tree walker that handles all 8 node types.

The executor walks the workflow tree depth-first, yielding StreamEvent
objects as an async generator.  Each node type has a dedicated handler.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, TypeAdapter

from databricks_deep_research.agents.config import (
    AgentNodeConfig,
    ConditionalNodeConfig,
    LoopNodeConfig,
    PlanAndExecuteNodeConfig,
    SubworkflowNodeConfig,
    ToolNodeConfig,
)
from databricks_deep_research.agents.execution.output_normalizer import (
    source_is_substantive,
)
from databricks_deep_research.agents.harness import execute_agent
from databricks_deep_research.errors import (
    NodeBudgetExceededError,
    WorkflowCancelledError,
    WorkflowConditionEvaluationError,
    WorkflowError,
    WorkflowExecutionError,
)
from databricks_deep_research.events.status_contract import make_status_kwargs
from databricks_deep_research.events.types import (
    AgentOutputEvent,
    BranchSelectedEvent,
    EvaluationDecisionEvent,
    ItemCompletedEvent,
    ItemsExtractedEvent,
    ItemStartedEvent,
    LoopExitEvent,
    LoopIterationEvent,
    NodeBudgetExceededEvent,
    NodeCompletedEvent,
    NodeErrorEvent,
    NodeSkippedEvent,
    NodeStartedEvent,
    PlanAndExecuteExitEvent,
    ReplanTriggeredEvent,
    StreamEvent,
    ToolCallEvent,
    ToolResultEvent,
    WorkflowCompletedEvent,
    WorkflowFailedEvent,
    WorkflowStartedEvent,
)
from databricks_deep_research.llm.client import FrameworkLLMClient
from databricks_deep_research.pools.pool_state import PoolConfig, PoolState
from databricks_deep_research.tools.factories.builtin import BuiltinToolFactory
from databricks_deep_research.tools.factories.databricks import DatabricksToolFactory
from databricks_deep_research.tools.factories.decorated import DecoratedToolFactory
from databricks_deep_research.tools.factory import ToolFactory, ToolFactoryContext
from databricks_deep_research.tools.protocol import (
    ResearchTool,
    TableRegistry,
    ToolContext,
    UrlRegistry,
)
from databricks_deep_research.tools.registry import ToolRegistry
from databricks_deep_research.tools.resolver import ToolResolver
from databricks_deep_research.tracing import get_current_span, trace_span
from databricks_deep_research.workflow.conditions import (
    CompositeCondition,
    Condition,
    ConditionEvaluationError,
    LLMCondition,
    StateCondition,
    evaluate_condition_strict,
    summarize_condition,
)
from databricks_deep_research.workflow.context import ExecutionContext
from databricks_deep_research.workflow.definition import NodeType, WorkflowDefinition, WorkflowNode
from databricks_deep_research.workflow.runtime.context import (
    PlanExecuteRunnerDeps,
    PlanExecuteRuntimeContext,
)
from databricks_deep_research.workflow.runtime.plan_execute_context import (
    build_evaluator_runtime_context as _build_evaluator_runtime_context_impl,
)
from databricks_deep_research.workflow.runtime.plan_execute_context import (
    build_planner_runtime_context as _build_planner_runtime_context_impl,
)
from databricks_deep_research.workflow.runtime.plan_execute_context import (
    format_available_sources as _format_available_sources_impl,
)
from databricks_deep_research.workflow.runtime.plan_execute_context import (
    format_completed_steps as _format_completed_steps_impl,
)
from databricks_deep_research.workflow.runtime.plan_execute_context import (
    format_reflector_feedback as _format_reflector_feedback_impl,
)
from databricks_deep_research.workflow.runtime.plan_execute_contracts import (
    NormalizedPlanContract,
)
from databricks_deep_research.workflow.runtime.plan_execute_contracts import (
    extract_raw_plan_contract as _extract_raw_plan_contract_impl,
)
from databricks_deep_research.workflow.runtime.plan_execute_contracts import (
    finalize_plan_contract as _finalize_plan_contract_impl,
)
from databricks_deep_research.workflow.runtime.plan_execute_contracts import (
    normalize_executable_plan_contract as _normalize_executable_plan_contract_impl,
)
from databricks_deep_research.workflow.runtime.plan_execute_execution import (
    append_completed_step as _append_completed_step_impl,
)
from databricks_deep_research.workflow.runtime.plan_execute_execution import (
    append_replan_feedback as _append_replan_feedback_impl,
)
from databricks_deep_research.workflow.runtime.plan_execute_execution import (
    build_available_source_catalog as _build_available_source_catalog_impl,
)
from databricks_deep_research.workflow.runtime.plan_execute_execution import (
    extract_decision as _extract_decision_impl,
)
from databricks_deep_research.workflow.runtime.plan_execute_execution import (
    extract_evidence_sufficiency as _extract_evidence_sufficiency_impl,
)
from databricks_deep_research.workflow.runtime.plan_execute_execution import (
    extract_failure_mode as _extract_failure_mode_impl,
)
from databricks_deep_research.workflow.runtime.plan_execute_execution import (
    extract_reasoning as _extract_reasoning_impl,
)
from databricks_deep_research.workflow.runtime.plan_execute_execution import (
    normalize_evaluation_decision as _normalize_evaluation_decision_impl,
)
from databricks_deep_research.workflow.runtime.plan_execute_execution import (
    populate_synthesis_state as _populate_synthesis_state_impl,
)
from databricks_deep_research.workflow.runtime.plan_execute_formatting import (
    extract_step_title as _extract_step_title_impl,
)
from databricks_deep_research.workflow.runtime.plan_execute_formatting import (
    format_all_observations as _format_all_observations_impl,
)
from databricks_deep_research.workflow.runtime.plan_execute_formatting import (
    format_plan_for_reflector as _format_plan_for_reflector_impl,
)
from databricks_deep_research.workflow.runtime.plan_execute_formatting import (
    format_source_quality as _format_source_quality_impl,
)
from databricks_deep_research.workflow.runtime.plan_execute_formatting import (
    format_source_topics as _format_source_topics_impl,
)
from databricks_deep_research.workflow.runtime.plan_execute_formatting import (
    obs_to_text as _obs_to_text_impl,
)
from databricks_deep_research.workflow.runtime.plan_execute_recovery import (
    coerce_discovered_sources as _coerce_discovered_sources_impl,
)
from databricks_deep_research.workflow.runtime.plan_execute_recovery import (
    hydrate_pools_from_discovered_sources as _hydrate_pools_from_discovered_sources_impl,
)
from databricks_deep_research.workflow.runtime.plan_execute_runner import run_plan_execute
from databricks_deep_research.workflow.runtime.plan_execute_types import (
    AvailableSourceDescriptor,
    PlanCycleContext,
    ReplanFeedbackEntry,
)
from databricks_deep_research.workflow.runtime_core import TypedRuntimeStateStore
from databricks_deep_research.workflow.runtime_core.api import WorkflowRunRequest, WorkflowRunResult
from databricks_deep_research.workflow.state import WorkflowState

RuntimeCondition = StateCondition | LLMCondition | CompositeCondition


_CONDITION_ADAPTER: TypeAdapter[RuntimeCondition] = TypeAdapter(Condition)


def _deserialize_condition(cond: Any) -> RuntimeCondition:
    """Convert a dict/model into a condition for evaluation."""
    if isinstance(cond, (StateCondition, LLMCondition, CompositeCondition)):
        return cond
    if hasattr(cond, "model_dump"):
        cond = cond.model_dump(mode="json")
    if isinstance(cond, dict):
        return _CONDITION_ADAPTER.validate_python(cond)
    raise TypeError(f"Cannot deserialize condition: {cond!r}")


def _state_to_eval_dict(state: WorkflowState) -> dict[str, Any]:
    """Build a flat dict of latest state values for condition evaluation.

    ``evaluate_condition_strict`` uses dot-path resolution which expects
    dict-style access.  WorkflowState stores data in an append-only log,
    so we materialise a snapshot of latest values here.
    """
    return {key: state.log[idx].value for key, idx in state._latest_index.items()}

logger = logging.getLogger(__name__)

_TOOL_KIND_ENDPOINT_KEY: dict[str, str] = {
    "vector_search": "index_name",
    "genie_space": "space_id",
    "sql_warehouse": "endpoint_name",
}


def _vector_index_metadata(
    tools: list[ResearchTool],
    declarations: list[Any],
) -> dict[str, Any]:
    """Build a compute-safe snapshot of configured vector-search indexes."""
    by_name = {getattr(decl, "name", ""): decl for decl in declarations}
    snapshot: dict[str, Any] = {}
    for tool in tools:
        definition = tool.definition
        decl = by_name.get(definition.name)
        if getattr(decl, "kind", None) == "vector_search":
            config = getattr(decl, "config", {}) or {}
            entry = {
                key: config[key]
                for key in (
                    "index_name",
                    "columns",
                    "num_results",
                    "query_type",
                    "filters_json",
                    "exclude_chunk_types",
                )
                if key in config
            }
            if entry:
                snapshot[definition.name] = entry
            continue

        if str(getattr(definition, "source_kind", "")) != "vector_index":
            continue

        index_name = getattr(tool, "_index_name", None)
        if isinstance(index_name, str) and index_name:
            snapshot[definition.name] = {
                "index_name": index_name,
                "columns": getattr(tool, "_columns", None),
                "num_results": getattr(tool, "_num_results", None),
                "query_type": getattr(tool, "_query_type", None),
            }
    return snapshot



def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _event_detail(event: StreamEvent) -> str:
    """Extract a compact log-friendly summary from a stream event."""
    if isinstance(event, ItemStartedEvent):
        return f"item_index={event.item_index} total={event.total_items} summary={event.item_summary[:200]!r}"
    if isinstance(event, ItemCompletedEvent):
        return f"item_index={event.item_index} items_processed={event.items_processed}"
    if isinstance(event, ItemsExtractedEvent):
        return f"total_items={event.total_items} cycle={event.cycle}"
    if isinstance(event, EvaluationDecisionEvent):
        return f"decision={event.decision} reasoning={event.reasoning[:500]!r}"
    if isinstance(event, PlanAndExecuteExitEvent):
        return (
            f"reason={event.reason} items={event.total_items_processed}/{event.total_planned} "
            f"replans={event.replan_cycles}"
        )
    if isinstance(event, ToolCallEvent):
        args_display = {k: str(v)[:200] for k, v in event.arguments.items()}
        return f"tool={event.tool_name} args={args_display}"
    if isinstance(event, ToolResultEvent):
        return (
            f"tool={event.tool_name} result_len={len(event.result_summary)} "
            f"accepted={event.accepted_source_count} "
            f"raw={event.raw_source_count}"
        )
    if isinstance(event, NodeStartedEvent):
        return f"type={event.node_type} label={event.label!r}"
    if isinstance(event, NodeCompletedEvent):
        return f"duration_ms={event.duration_ms:.0f}"
    if isinstance(event, NodeErrorEvent):
        return f"error={event.error_message[:300]!r} retry={event.will_retry}"
    if isinstance(event, NodeSkippedEvent):
        return f"reason={event.reason[:200]!r}"
    if isinstance(event, WorkflowStartedEvent):
        return f"workflow={event.workflow_name!r}"
    if isinstance(event, WorkflowCompletedEvent):
        return f"duration_ms={event.duration_ms:.0f} sources={event.total_sources}"
    if isinstance(event, LoopIterationEvent):
        return f"iteration={event.iteration}/{event.max_iterations}"
    if isinstance(event, LoopExitEvent):
        return f"reason={event.reason} iterations={event.total_iterations}"
    if isinstance(event, BranchSelectedEvent):
        return f"branch={event.branch_index} {event.condition_summary}"
    if isinstance(event, ReplanTriggeredEvent):
        return f"cycle={event.cycle} remaining={event.items_remaining}"
    if isinstance(event, AgentOutputEvent):
        return f"key={event.output_key} preview={event.output_preview[:200]!r}"

    return str(event)[:150]


def _normalize_plan_contract(plan_data: Any, items_path: str) -> dict[str, Any]:
    has_enough_context = False
    title = ""
    thought = ""
    if isinstance(plan_data, dict):
        has_enough_context = bool(plan_data.get("has_enough_context", False))
        title = str(plan_data.get("title", "") or "")
        thought = str(plan_data.get("thought", "") or "")
    else:
        has_enough_context = bool(getattr(plan_data, "has_enough_context", False))
        title = str(getattr(plan_data, "title", "") or "")
        thought = str(getattr(plan_data, "thought", "") or "")
    return {
        "items": _extract_items(plan_data, items_path),
        "has_enough_context": has_enough_context,
        "title": title,
        "thought": thought,
    }


def _extract_raw_plan_contract(plan_data: Any, items_path: str) -> NormalizedPlanContract:
    return _extract_raw_plan_contract_impl(plan_data, items_path)


def _finalize_plan_contract(contract: NormalizedPlanContract, plan_data: Any) -> NormalizedPlanContract:
    return _finalize_plan_contract_impl(contract, plan_data)


def _normalize_executable_plan_contract(plan_data: Any, items_path: str) -> dict[str, Any]:
    return _normalize_executable_plan_contract_impl(plan_data, items_path)


def _format_completed_steps(completed_steps: list[str]) -> str:
    return _format_completed_steps_impl(completed_steps)


def _format_reflector_feedback(entries: list[ReplanFeedbackEntry]) -> str:
    return _format_reflector_feedback_impl(entries)


def _format_available_sources(sources: list[AvailableSourceDescriptor]) -> str:
    return _format_available_sources_impl(sources)


class WorkflowExecutor:
    """Executes a WorkflowDefinition by walking its tree depth-first.

    Yields StreamEvent objects for each significant action.
    """

    def __init__(
        self,
        definition: WorkflowDefinition,
        llm_client: FrameworkLLMClient,
        *,
        tool_resolver: ToolResolver | None = None,
        tool_registry: ToolRegistry | None = None,  # DEPRECATED — use tool_resolver
        tool_factories: list[ToolFactory] | None = None,
        factory_context: ToolFactoryContext | None = None,
        strict_tool_resolution: bool = False,
        enterprise_tools: list[ResearchTool] | None = None,
        url_registry: UrlRegistry | None = None,
        table_registry: TableRegistry | None = None,
        context: ExecutionContext | None = None,
    ) -> None:
        self._defn = definition
        self._llm = llm_client
        self._url_registry = url_registry or UrlRegistry()
        self._table_registry = table_registry or TableRegistry()
        self._context = context
        self._total_tokens: int = 0
        # Recursion depth of the current executor when invoked as a child of a
        # subworkflow node. 0 for a top-level workflow; the parent sets the
        # child's value to parent_depth + 1 in ``_exec_subworkflow``.
        self._subworkflow_depth: int = 0
        self._strict_tool_resolution = strict_tool_resolution
        self._workflow_total_sources_raw = 0
        self._workflow_total_sources_accepted = 0
        self._workflow_blocked_steps = 0
        self._workflow_missing_declared_tools = 0
        self._workflow_plan_exit_reasons: list[str] = []
        self._workflow_total_steps_executed = 0

        registry = tool_registry or ToolRegistry()
        resolved_factories: list[ToolFactory] = list(
            tool_factories
            or [
                BuiltinToolFactory(),
                DatabricksToolFactory(),
                # Fail-closed by default: default-chain hosts may execute
                # STORED workflow definitions, and a decorated import runs
                # arbitrary code. Hosts whose YAML is import-time-authored can
                # pass their own DecoratedToolFactory(allowed_import_prefixes=None).
                DecoratedToolFactory(allowed_import_prefixes=()),
            ]
        )
        resolved_factory_context = factory_context or ToolFactoryContext()

        # Build resolver: prefer explicit resolver, else wrap registry
        if tool_resolver is not None:
            self._resolver = tool_resolver
        else:
            self._resolver = ToolResolver(
                declarations=list(definition.tools) if definition.tools else None,
                factories=resolved_factories,
                factory_context=resolved_factory_context,
                legacy_registry=registry,
            )

        # Register enterprise tools as overrides on the resolver
        if enterprise_tools:
            for t in enterprise_tools:
                self._resolver.override(t.definition.name, t)

        # Keep a registry reference for _exec_tool (legacy tool nodes)
        self._tool_registry = registry
        if enterprise_tools:
            for t in enterprise_tools:
                self._tool_registry.register_external(t.definition.name, t)

        # Initialize pools through registry (adds BM25+vector search layer)
        from databricks_deep_research.pools.pool_registry import PoolRegistry
        self._registry = PoolRegistry(llm_client=llm_client)
        self._registry.initialize_from_configs(definition.pools)
        self._pools: dict[str, PoolState] = self._registry.all_pools()

    def _inject_compute_callables(
        self,
        tools: list[ResearchTool],
        tool_declarations: list[Any],
    ) -> None:
        """Expose provider-backed callables inside each Python compute tool."""
        from databricks_deep_research.tools.builtins.compute import PythonComputeTool
        from databricks_deep_research.tools.builtins.text_table import (
            ComputeCallableProvider,
            TableBindingRegistry,
            inject_table_callables,
        )

        compute_tools = [
            tool for tool in tools if isinstance(tool, PythonComputeTool)
        ]
        if not compute_tools:
            return

        providers = [
            tool
            for tool in tools
            if not isinstance(tool, PythonComputeTool)
            and isinstance(tool, ComputeCallableProvider)
        ]
        if not providers:
            return

        factory_context = getattr(self._resolver, "factory_context", None)
        registry = getattr(factory_context, "table_registry", None)
        if registry is not None and not isinstance(registry, TableBindingRegistry):
            registry = None
        vector_indexes = _vector_index_metadata(tools, tool_declarations)

        for compute in compute_tools:
            injected = inject_table_callables(
                compute=compute,
                providers=providers,
                registry=registry,
                vector_indexes=vector_indexes,
            )
            logger.info(
                "COMPUTE_CALLABLES_INJECTED compute=%s callables=%s "
                "bindings=%s vector_indexes=%s",
                compute.definition.name,
                injected,
                registry is not None,
                sorted(vector_indexes),
            )

    def _emit(self, event: StreamEvent) -> StreamEvent:
        """Log and return an event."""
        logger.info(
            "EVENT %s node=%s %s",
            type(event).__name__,
            event.node_id,
            _event_detail(event),
        )
        return event

    async def execute(
        self,
        state: WorkflowState,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Execute the workflow and yield events."""
        import time

        start = time.monotonic()

        state.runtime_store = TypedRuntimeStateStore(
            query=state.query or "",
            workflow_id=self._defn.id,
            workflow_name=self._defn.name,
        )

        async with trace_span(
            f"workflow.{self._defn.name}",
            span_type="CHAIN",
            attributes={
                "workflow.name": self._defn.name,
                "workflow.id": self._defn.id,
                "query": (state.query or "")[:200],
                "workflow.query": (state.query or "")[:500],
            },
        ) as wf_span:
            yield self._emit(WorkflowStartedEvent(
                node_id=self._defn.root.id,
                timestamp=_now(),
                workflow_id=self._defn.id,
                workflow_name=self._defn.name,
            ))

            # Attach pools to state for cross-node access
            state.pools = self._pools

            try:
                async for event in self._exec_node(self._defn.root, state):
                    yield event
            except WorkflowCancelledError:
                logger.info("WORKFLOW_CANCELLED workflow_id=%s", self._defn.id)
                elapsed_ms = (time.monotonic() - start) * 1000
                if state.runtime_store is not None:
                    state.runtime_store.set_workflow_cancelled(duration_ms=elapsed_ms)
                if wf_span:
                    wf_span.set_attributes({
                        "workflow.terminal_status": "cancelled",
                        "workflow.duration_ms": elapsed_ms,
                    })
            except Exception as exc:
                logger.exception("WORKFLOW_ERROR workflow_id=%s", self._defn.id)
                elapsed_ms = (time.monotonic() - start) * 1000
                if state.runtime_store is not None:
                    state.runtime_store.set_workflow_failed(
                        duration_ms=elapsed_ms,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                if wf_span:
                    wf_span.set_attributes({
                        "workflow.terminal_status": "failed",
                        "workflow.duration_ms": elapsed_ms,
                        "workflow.error_type": type(exc).__name__,
                    })
                raise
            finally:
                try:
                    await self._close_sandbox_session()
                except Exception:  # pragma: no cover — cleanup must never mask the run
                    logger.debug("SANDBOX_SESSION_CLOSE_FAILED", exc_info=True)

            elapsed_ms = (time.monotonic() - start) * 1000
            sources_pool = self._pools.get("sources", PoolState(PoolConfig(name="_")))
            total_sources = sum(
                1 for source in sources_pool.snapshot() if source_is_substantive(source)
            )

            if wf_span:
                wf_span.set_attributes({
                    "workflow.terminal_status": "completed",
                    "workflow.duration_ms": elapsed_ms,
                    "workflow.total_sources": total_sources,
                    "workflow.total_sources_raw": self._workflow_total_sources_raw,
                    "workflow.total_sources_accepted": self._workflow_total_sources_accepted,
                    "workflow.blocked_steps": self._workflow_blocked_steps,
                    "workflow.missing_declared_tools": self._workflow_missing_declared_tools,
                    "workflow.total_steps_executed": self._workflow_total_steps_executed,
                })

            if state.runtime_store is not None:
                state.runtime_store.set_workflow_completed(
                    duration_ms=elapsed_ms,
                    total_tokens=self._total_tokens,
                    total_sources=total_sources,
                    total_steps_executed=self._workflow_total_steps_executed,
                    blocked_steps=self._workflow_blocked_steps,
                    missing_declared_tools=self._workflow_missing_declared_tools,
                )

            report_value = state.get("report") or state.get("output")
            structured = None
            if isinstance(report_value, BaseModel):
                structured = report_value.model_dump(mode="json")
                final_report_text = report_value.model_dump_json()
            elif isinstance(report_value, dict) and report_value.get("output_type"):
                # A structured-output deliverable (e.g. a plugin assembler node) writes
                # a dict carrying ``output_type``. Serialize it as JSON — NOT ``str()``
                # (a Python repr with single quotes) — so the persisted message.content
                # is parseable by the frontend's structured-output renderer instead of
                # degrading to raw text. Mirrors the ``structured`` capture just above.
                structured = report_value
                final_report_text = json.dumps(report_value, default=str)
            else:
                final_report_text = str(report_value or "")

            yield self._emit(WorkflowCompletedEvent(
                node_id=self._defn.root.id,
                timestamp=_now(),
                workflow_id=self._defn.id,
                duration_ms=elapsed_ms,
                total_tokens=self._total_tokens,
                final_report=final_report_text,
                structured_output=structured,
                total_sources=total_sources,
                total_steps_executed=self._workflow_total_steps_executed,
            ))

    # -- Node dispatch -------------------------------------------------------

    async def _exec_node(
        self, node: WorkflowNode, state: WorkflowState
    ) -> AsyncGenerator[StreamEvent, None]:
        """Dispatch to the correct handler based on node type."""
        import time

        # Cancellation check
        if state.is_cancelled:
            raise WorkflowCancelledError()

        start = time.monotonic()

        if state.runtime_store is not None:
            state.runtime_store.start_node(node_id=node.id, node_type=node.type.value, label=node.label or node.id)

        async with trace_span(
            f"node.{node.type.value}.{node.label or node.id}",
            span_type="CHAIN",
            attributes={"node.id": node.id, "node.type": node.type.value},
        ) as node_span:
            budget_seconds = float(node.budget_seconds or 0)
            yield self._emit(NodeStartedEvent(
                node_id=node.id, timestamp=_now(),
                node_type=node.type.value, label=node.label,
            ))

            try:
                node_handler = self._handlers.get(node.type)
                if node_handler is None:
                    raise WorkflowError(f"Unknown node type: {node.type}")

                async for event in node_handler(self, node, state):
                    yield event

                elapsed_ms = (time.monotonic() - start) * 1000
                if budget_seconds > 0 and elapsed_ms > budget_seconds * 1000:
                    raise NodeBudgetExceededError(node.id, budget_seconds, elapsed_ms)

            except WorkflowCancelledError:
                raise
            except Exception as exc:
                # Error handling per ErrorConfig
                error_cfg = node.error_handling
                retry_handler = self._handlers.get(node.type)
                if isinstance(exc, NodeBudgetExceededError):
                    yield self._emit(NodeBudgetExceededEvent(
                        node_id=node.id,
                        timestamp=_now(),
                        budget_seconds=exc.budget_seconds,
                        elapsed_ms=exc.elapsed_ms,
                    ))
                if error_cfg and error_cfg.on_error == "skip":
                    yield self._emit(NodeSkippedEvent(
                        node_id=node.id, timestamp=_now(),
                        reason=str(exc),
                    ))
                    if state.runtime_store is not None:
                        state.runtime_store.fail_node(node_id=node.id, duration_ms=0.0)
                    return
                elif error_cfg and error_cfg.on_error == "retry" and retry_handler is not None:
                    for attempt in range(error_cfg.max_retries):
                        yield self._emit(NodeErrorEvent(
                            node_id=node.id, timestamp=_now(),
                            error_message=str(exc),
                            will_retry=True,
                            retry_attempt=attempt + 1,
                            # Not terminal: the node is about to retry.
                            **make_status_kwargs("running"),
                        ))
                        await asyncio.sleep(error_cfg.retry_delay_seconds * (2 ** attempt))
                        try:
                            async for event in retry_handler(self, node, state):
                                yield event
                            break  # Success
                        except Exception as retry_exc:
                            exc = retry_exc
                    else:
                        # All retries exhausted
                        yield self._emit(NodeErrorEvent(
                            node_id=node.id, timestamp=_now(),
                            error_message=f"All {error_cfg.max_retries} retries exhausted: {exc}",
                            **make_status_kwargs("failed"),
                        ))
                        raise
                    return
                else:
                    yield self._emit(NodeErrorEvent(
                        node_id=node.id, timestamp=_now(),
                        error_message=str(exc),
                        **make_status_kwargs("failed"),
                    ))
                    if state.runtime_store is not None:
                        state.runtime_store.fail_node(node_id=node.id, duration_ms=0.0)
                    raise

            elapsed_ms = (time.monotonic() - start) * 1000
            if node_span:
                node_span.set_attributes({"node.duration_ms": elapsed_ms})

            if state.runtime_store is not None:
                state.runtime_store.complete_node(node_id=node.id, duration_ms=elapsed_ms)

            yield self._emit(NodeCompletedEvent(
                node_id=node.id, timestamp=_now(),
                duration_ms=elapsed_ms,
                **make_status_kwargs("completed"),
            ))

    def _record_step_completed(self) -> None:
        self._workflow_total_steps_executed += 1

    # -- Node type handlers --------------------------------------------------

    async def _exec_sequence(
        self, node: WorkflowNode, state: WorkflowState
    ) -> AsyncGenerator[StreamEvent, None]:
        """Execute children sequentially."""
        for child in node.children:
            async for event in self._exec_node(child, state):
                yield event

    async def _exec_parallel(
        self, node: WorkflowNode, state: WorkflowState
    ) -> AsyncGenerator[StreamEvent, None]:
        """Execute children concurrently via merged event queue."""
        merged: asyncio.Queue[tuple[str, StreamEvent | None]] = asyncio.Queue()
        errors: list[Exception] = []

        async def _run_child(child: WorkflowNode) -> None:
            try:
                async for event in self._exec_node(child, state):
                    await merged.put((child.id, event))
            except Exception as exc:
                errors.append(exc)
            finally:
                await merged.put((child.id, None))  # sentinel

        tasks = [asyncio.create_task(_run_child(c)) for c in node.children]
        active = {c.id for c in node.children}

        while active:
            child_id, event = await merged.get()  # efficient blocking wait
            if event is None:
                active.discard(child_id)
            else:
                yield event

        await asyncio.gather(*tasks, return_exceptions=True)
        if errors:
            raise errors[0]

    async def _exec_loop(
        self, node: WorkflowNode, state: WorkflowState
    ) -> AsyncGenerator[StreamEvent, None]:
        """Execute children in a loop with condition checking."""
        config = LoopNodeConfig(**node.config)
        iteration = 0

        while iteration < config.max_iterations:
            iteration += 1
            yield self._emit(LoopIterationEvent(
                node_id=node.id, timestamp=_now(),
                iteration=iteration, max_iterations=config.max_iterations,
            ))

            # Execute body
            for child in node.children:
                async for event in self._exec_node(child, state):
                    yield event

            # Check exit condition (only after min_iterations)
            if iteration >= config.min_iterations:
                try:
                    cond = _deserialize_condition(config.until)
                    should_exit = evaluate_condition_strict(cond, _state_to_eval_dict(state))
                    if should_exit:
                        yield self._emit(LoopExitEvent(
                            node_id=node.id, timestamp=_now(),
                            reason="condition_met",
                            total_iterations=iteration,
                        ))
                        return
                except Exception as exc:
                    raise WorkflowConditionEvaluationError(
                        f"Workflow loop condition evaluation failed at node {node.id!r}: "
                        f"{exc}. This workflow should not have passed validation."
                    ) from exc

        yield self._emit(LoopExitEvent(
            node_id=node.id, timestamp=_now(),
            reason="max_iterations",
            total_iterations=iteration,
        ))

    async def _exec_conditional(
        self, node: WorkflowNode, state: WorkflowState
    ) -> AsyncGenerator[StreamEvent, None]:
        """Evaluate conditions and execute the matching branch."""
        config = ConditionalNodeConfig(**node.config)
        selected_idx = config.default_branch
        condition_summary = (
            f"Default branch {selected_idx} selected: no conditions matched"
        )

        eval_dict = _state_to_eval_dict(state)
        for i, branch in enumerate(config.conditions):
            child_index = getattr(branch, "child_index", i)
            branch_condition: Any = getattr(branch, "condition", branch)
            cond = _deserialize_condition(branch_condition)
            try:
                if evaluate_condition_strict(cond, eval_dict):
                    selected_idx = child_index
                    condition_summary = (
                        f"Branch {selected_idx} selected: {summarize_condition(cond)}"
                    )
                    break
            except (ConditionEvaluationError, TypeError, ValueError) as exc:
                raise WorkflowConditionEvaluationError(
                    f"Workflow condition evaluation failed at node {node.id!r}, "
                    f"condition[{i}] {summarize_condition(cond)!r}: {exc}. "
                    "This workflow should not have passed validation. Remove the router "
                    "or declare a typed upstream discriminator."
                ) from exc

        yield self._emit(BranchSelectedEvent(
            node_id=node.id, timestamp=_now(),
            branch_index=selected_idx,
            condition_summary=condition_summary,
        ))

        if 0 <= selected_idx < len(node.children):
            async for event in self._exec_node(node.children[selected_idx], state):
                yield event

    async def _exec_agent(
        self, node: WorkflowNode, state: WorkflowState
    ) -> AsyncGenerator[StreamEvent, None]:
        """Execute an agent node via the harness."""
        config = AgentNodeConfig(**node.config)

        # Resolve tools for this agent
        tools: list[ResearchTool] = []
        tool_declarations: list[Any] = []
        errors: list[str] = []
        for ref in config.tools:
            try:
                tool = await self._resolver.resolve(ref)
                tools.append(tool)
                if isinstance(ref, str):
                    decl = self._resolver.get_declaration(ref)
                    if decl is not None:
                        tool_declarations.append(decl)
            except ValueError as exc:
                ref_name = ref if isinstance(ref, str) else ref.get("name", str(ref))
                errors.append(str(ref_name))
                logger.warning("TOOL_NOT_FOUND ref=%s error=%s", ref, exc)

        # Auto-attach read_skill when this agent declares skills (Feature 2.2).
        # Built from the wired _skill_store and appended to the resolved tools so
        # the ReAct loop (which maps tools by name from this list) can call it.
        # run_skill_script is auto-attached only when BOTH the per-agent
        # ``allow_skill_scripts`` and the global kill-switch are on (A2).
        from databricks_deep_research.agents.skill_attach import (
            maybe_attach_read_skill,
            maybe_attach_run_skill_script,
        )

        _factory_ctx = getattr(self._resolver, "factory_context", None)
        maybe_attach_read_skill(tools, config.skills, _factory_ctx)
        maybe_attach_run_skill_script(
            tools, config.skills, config.allow_skill_scripts, _factory_ctx
        )

        # Auto-attach discovered MCP tools for the servers this agent binds via
        # ``config.mcp_servers`` (Feature 4.3). The host stashes a
        # ``{server: [tools]}`` map in the factory context; without this step the
        # injected tools stay orphaned in the resolver and no agent ever calls
        # them. No-op when the agent binds no servers / no map is wired.
        from databricks_deep_research.agents.mcp_attach import maybe_attach_mcp

        maybe_attach_mcp(tools, config.mcp_servers, _factory_ctx)

        resolved_names = [t.definition.name for t in tools]
        logger.info(
            "AGENT_TOOLS_RESOLVED node=%s config_tool_refs=%d "
            "resolved_tools=%d tool_names=%s max_tool_calls=%s",
            node.id,
            len(config.tools),
            len(tools),
            resolved_names,
            config.max_tool_calls,
        )

        if errors:
            self._workflow_missing_declared_tools += len(errors)
            logger.warning(
                "AGENT_TOOLS_MISSING node=%s missing=%s",
                node.id,
                errors,
            )
            if self._strict_tool_resolution:
                raise WorkflowError(
                    f"Node {node.id!r} is missing declared tools: {errors}"
                )

        # Researcher zero-tools guard. A researcher that writes to the
        # sources pool MUST have at least one runtime tool that produces
        # evidence — otherwise the LLM emits planning text only and the
        # synthesizer fail-closes with "Insufficient Evidence". Cross-phase
        # invariant: a source-emitting researcher must have resolvable
        # evidence tools, or fail before LLM spend. Only enforced under
        # strict_tool_resolution to stay opt-in.
        if self._strict_tool_resolution:
            subtype = (config.subtype or "").casefold()
            if subtype == "researcher" and not tools:
                writes_sources = any(
                    (pw.pool or "").casefold() == "sources"
                    for pw in (config.pool_writes or [])
                )
                if writes_sources:
                    logger.warning(
                        "AGENT_ZERO_TOOLS_RESEARCHER node=%s subtype=%s "
                        "pool_writes=%s — refusing to start under strict mode",
                        node.id,
                        subtype,
                        [pw.pool for pw in (config.pool_writes or [])],
                    )
                    raise WorkflowError(
                        f"Node {node.id!r} is a researcher subtype that writes "
                        "to the 'sources' pool but has zero bound runtime tools. "
                        "It cannot produce evidence; refusing to start (see "
                        "strict_tool_resolution invariant)."
                    )

        if tool_declarations:
            from databricks_deep_research.tools.catalog_service import (
                CATALOG_DECLARATIONS_EXTRA,
                declarations_to_jsonable,
            )

            config.extras[CATALOG_DECLARATIONS_EXTRA] = declarations_to_jsonable(
                tool_declarations
            )

        if tools:
            from databricks_deep_research.tools.builtins.text_table import (
                TableBindingRegistry,
                render_table_bindings_prompt,
            )

            factory_context = getattr(self._resolver, "factory_context", None)
            registry = getattr(factory_context, "table_registry", None)
            has_table_tool = any(
                str(getattr(tool.definition, "source_kind", "")) == "text_table"
                for tool in tools
            )
            if has_table_tool and isinstance(registry, TableBindingRegistry):
                table_prompt = render_table_bindings_prompt(registry)
                if table_prompt and table_prompt not in config.system_prompt:
                    separator = "\n\n" if config.system_prompt else ""
                    config = config.model_copy(
                        update={
                            "system_prompt": (
                                f"{config.system_prompt}{separator}{table_prompt}"
                            )
                        }
                    )

        # Attach tool resolution details to the parent node span
        node_span = get_current_span()
        if node_span:
            node_span.set_attributes({
                "node.resolved_tools": str(resolved_names),
                "node.missing_tools": str(errors) if errors else "[]",
                "node.config_tool_refs": len(config.tools),
                "node.max_tool_calls": config.max_tool_calls,
            })

        # Add pool tools if configured (with registry for hybrid search)
        if config.pool_tools:
            from databricks_deep_research.pools.pool_tools import create_pool_tools
            for pool_name in config.pool_tools:
                if pool_name in self._pools:
                    pool_tools = create_pool_tools(
                        pool_name, self._pools[pool_name],
                        registry=self._registry,
                    )
                    tools.extend(pool_tools)

        self._inject_compute_callables(tools, tool_declarations)

        # GOVERNED spawn_agent runner (spec §3.3). Built ONLY when this node's
        # config opts in (code/hybrid action_mode + a non-empty declared set + a
        # positive budget); otherwise ``None`` so ``execute_agent`` is byte-
        # identical and no spawn closure is ever injected. The runner runs a
        # DECLARED inline subworkflow in an ISOLATED scratchpad scope (fresh
        # compute namespace + private VFS) seeded with the spawn prompt as its
        # query — so a spawned child cannot reach the parent Cell's compute
        # namespace / VFS / variables; OBO identity is preserved (governed tools
        # only). The same depth guard bounds spawn recursion.
        spawn_runner: Callable[..., Awaitable[Any]] | None = None
        spawn_enabled = (
            config.action_mode in ("code", "hybrid")
            and bool(config.spawnable_subagents)
            and config.spawn_budget > 0
        )
        if spawn_enabled:
            spawn_depth = self._subworkflow_depth + 1
            max_spawn_depth = SubworkflowNodeConfig.model_fields[
                "max_subworkflow_depth"
            ].default

            async def _spawn_runner(
                *, name: str, prompt: str, inline: dict[str, Any]
            ) -> Any:
                # Depth guard — bounds a spawned child spawning further (spec
                # §3.3 security invariant: spawn recursion is finite).
                if spawn_depth > max_spawn_depth:
                    raise WorkflowError(
                        f"max_subworkflow_depth ({max_spawn_depth}) exceeded by "
                        f"spawn_agent('{name}') at node '{node.id}'"
                    )
                result_sink: dict[str, Any] = {}
                # Drive to completion. v1 spawns are synchronous/sequential; we
                # consume the child's events to drive it (parallel fan-out +
                # event re-emission are deferred — see config.max_concurrent_spawns).
                async for _event in self._run_inline_subworkflow(
                    inline,
                    query=prompt,
                    depth=spawn_depth,
                    parent_state=state,
                    pool_mode="isolate",
                    result_sink=result_sink,
                ):
                    pass
                return result_sink.get("primary")

            spawn_runner = _spawn_runner

        output = await execute_agent(
            node_id=node.id,
            config=config,
            state=state,
            llm_client=self._llm,
            tools=tools,
            pools=self._pools,
            url_registry=self._url_registry,
            table_registry=self._table_registry,
            tool_call_cache=self._context.tool_call_cache if self._context else None,
            execution_context=self._context,
            spawn_runner=spawn_runner,
        )

        # Track token usage
        agent_tokens = int(output.token_usage.get("total_tokens") or sum(v for k, v in output.token_usage.items() if k != "total_tokens"))
        self._total_tokens += agent_tokens
        logger.info(
            "AGENT_TOKENS node=%s tokens=%d cumulative=%d usage=%s",
            node.id, agent_tokens, self._total_tokens, output.token_usage,
        )
        if node_span:
            node_span.set_attributes({
                "node.agent_tokens": agent_tokens,
                "node.cumulative_tokens": self._total_tokens,
            })

        tool_result_events = [
            event for event in output.events if isinstance(event, ToolResultEvent)
        ]
        self._workflow_total_sources_raw += sum(
            int(getattr(event, "raw_source_count", 0)) for event in tool_result_events
        )
        self._workflow_total_sources_accepted += sum(
            int(getattr(event, "accepted_source_count", getattr(event, "source_count", 0)))
            for event in tool_result_events
        )

        # Yield all events from harness
        for event in output.events:
            yield event

    async def _exec_tool(
        self, node: WorkflowNode, state: WorkflowState
    ) -> AsyncGenerator[StreamEvent, None]:
        """Execute a direct tool call (no LLM).

        Resolution goes resolver-first (overrides -> cache -> declarations ->
        the resolver's own legacy fallback) so declared, factory-built, and
        per-request override tools (e.g. MCP) are reachable from DAG steps.
        The executor's legacy registry remains an explicit fallback because
        host-injected resolvers may carry no legacy registry of their own.
        """
        config = ToolNodeConfig(**node.config)

        from databricks_deep_research.tools.protocol import ToolRef

        ref = config.ref
        try:
            tool = await self._resolver.resolve({"type": ref.type, "name": ref.name})
        except ValueError as resolve_exc:
            try:
                tool = self._tool_registry.resolve(ToolRef(type=ref.type, name=ref.name))
            except Exception:
                if ref.type == "mcp":
                    raise WorkflowError(
                        f"tool node '{node.id}': MCP tool {ref.name!r} was not "
                        "discovered at runtime — its server may have been "
                        "skipped (build failure) or is not configured for this "
                        "request"
                    ) from resolve_exc
                raise resolve_exc from None

        # Literals first; input_mapping keys cannot collide (config validator).
        args: dict[str, Any] = dict(config.input_literals)
        for arg_name, state_key in config.input_mapping.items():
            args[arg_name] = state.get(state_key)

        # Event payloads carry truncated argument text: mapped state values can
        # be entire documents, and events are streamed/persisted.
        event_args: dict[str, Any] = {}
        for arg_name, value in args.items():
            if value is None or isinstance(value, (int, float, bool)):
                event_args[arg_name] = value
            else:
                text = value if isinstance(value, str) else repr(value)
                event_args[arg_name] = text[:500]
        yield self._emit(ToolCallEvent(
            node_id=node.id,
            timestamp=_now(),
            tool_name=ref.name,
            arguments=event_args,
        ))

        async with trace_span(
            f"tool.{ref.name}",
            span_type="TOOL",
            attributes={"tool.name": ref.name},
        ) as tool_span:
            validated = tool.validate_arguments(args)
            factory_ctx = self._resolver.factory_context
            extras: dict[str, Any] = {**factory_ctx.extras}
            user_token = state.user_token or factory_ctx.user_token
            if user_token:
                # get_user_token(extras) consumers (text_table tools) read this
                # key; the factory context carries the token only as a field.
                extras["user_token"] = user_token
            ctx = ToolContext(
                query=state.query,
                url_registry=self._url_registry,
                table_registry=self._table_registry,
                extras=extras,
            )
            result = await tool.execute(validated, ctx)

            if tool_span:
                tool_span.set_attributes({"tool.result_len": len(result.content)})

        # Pool admission: non-builtin sources (function:// / mcp:// artifacts)
        # flow through the same evidence gate as agent-invoked tools, so
        # DAG-step results are citeable in synthesis.
        raw_source_count = len(result.sources)
        accepted_count = 0
        rejected_count = 0
        if result.sources and tool.definition.source_kind != "builtin":
            from databricks_deep_research.agents.source_aware import admit_tool_result

            admitted = admit_tool_result(
                tool.definition,
                result,
                current_step=None,
                root_query=state.query,
            )
            accepted_count = admitted.accepted_count
            rejected_count = len(admitted.rejected_sources)
            sources_pool = self._pools.get("sources")
            if sources_pool is not None:
                added = sum(
                    1 for source in admitted.accepted_sources if sources_pool.add(source)
                )
                logger.info(
                    "TOOL_NODE_POOL_ADMIT node=%s tool=%s raw=%d accepted=%d "
                    "added=%d rejected=%d",
                    node.id, ref.name, raw_source_count, accepted_count,
                    added, rejected_count,
                )
            else:
                logger.info(
                    "TOOL_NODE_POOL_ADMIT_SKIPPED node=%s tool=%s reason=no_sources_pool",
                    node.id, ref.name,
                )

        _content_str = result.content if isinstance(result.content, str) else str(result.content)
        yield self._emit(ToolResultEvent(
            node_id=node.id,
            timestamp=_now(),
            tool_name=ref.name,
            result_summary=_content_str[:400],
            source_count=accepted_count,
            raw_source_count=raw_source_count,
            accepted_source_count=accepted_count,
            rejected_source_count=rejected_count,
            tool_success=result.success,
            tool_error=result.error or "",
        ))

        # DR_LEAK_TRACE state_write: capture tool-node state writes too.
        try:
            logger.info(
                "DR_LEAK_TRACE phase=state_write origin=tool "
                "node=%s tool=%s output_key=%s value_len=%d value_head=%r",
                node.id,
                ref.name,
                config.output_key,
                len(_content_str),
                _content_str[:300].replace("\n", "\\n"),
            )
        except Exception as _exc:  # pragma: no cover — diagnostic only
            logger.debug("DR_LEAK_TRACE state_write (tool) skipped: %s", _exc)
        state.append(node.id, config.output_key, result.content)

        # Table interop: a structured table in the result becomes addressable
        # by table_* tools and compute callables via the run's TableRegistry.
        table_index: int | None = None
        table_json = result.data.get("table_json")
        if isinstance(table_json, dict) and "headers" in table_json and "rows" in table_json:
            try:
                table_index = self._table_registry.register(
                    table_json,
                    source_kind=str(tool.definition.source_kind),
                    source_label=f"function://{ref.name}",
                )
            except ValueError as exc:
                logger.warning(
                    "TOOL_NODE_TABLE_REGISTER_FAILED node=%s tool=%s err=%s",
                    node.id, ref.name, exc,
                )

        if config.output_data_key:
            data_payload: dict[str, Any] = dict(result.data)
            data_payload["success"] = result.success
            if result.error:
                data_payload["error"] = result.error
            if table_index is not None:
                data_payload["table_index"] = table_index
            state.append(node.id, config.output_data_key, data_payload)

        if config.bind_namespace:
            from databricks_deep_research.tools.compute_session import get_compute_tool

            compute = get_compute_tool(factory_ctx.extras)
            if compute is not None:
                bind_value = result.data.get("result", result.content)
                compute.inject_variable(config.bind_namespace, bind_value)
            else:
                logger.info(
                    "TOOL_NODE_BIND_NAMESPACE_SKIPPED node=%s var=%s reason=no_compute_tool",
                    node.id,
                    config.bind_namespace,
                )

        if config.enforce_output_schema and config.output_schema:
            payload: Any = result.data if result.data else result.content
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except (TypeError, ValueError):
                    payload = None
            required = config.output_schema.get("required", [])
            missing = [
                key for key in required
                if not (isinstance(payload, dict) and key in payload)
            ]
            if missing:
                raise WorkflowError(
                    f"tool node '{node.id}' ({ref.name}): output missing "
                    f"required key(s) {missing} per output_schema"
                )

        if config.fail_on_error and not result.success:
            raise WorkflowError(
                f"tool node '{node.id}' ({ref.name}) failed: "
                f"{result.error or _content_str[:200]}"
            )

    async def _close_sandbox_session(self, resolver: ToolResolver | None = None) -> None:
        """Close the run's persistent sandbox REPL, if one was created."""
        from databricks_deep_research.tools.code_executor import SandboxSessionHolder

        target = resolver or self._resolver
        holder = target.factory_context.extras.get("_sandbox_session")
        if isinstance(holder, SandboxSessionHolder):
            await holder.aclose()

    def _build_isolated_child_resolver(
        self, child_defn: WorkflowDefinition
    ) -> ToolResolver:
        """Build a private resolver for an ``isolate`` subworkflow child.

        Gives the child its own scratchpad: a FRESH compute namespace and a
        PRIVATE VFS, so child scratchpad writes never leak to the parent and the
        parent's scratchpad state is hidden from the child.

        How isolation is achieved (two extras keys diverge; identity preserved):

        * ``_framework_vfs`` is OVERRIDDEN with a fresh in-memory filesystem so
          the VFS tools (``ls``/``read_file``/``write_file``/…) read the child's
          private store rather than the parent's.
        * ``_resolver_cache`` is EXCLUDED from the copied extras. The
          :class:`ToolResolver` constructor installs a fresh ``_resolver_cache``,
          so the child resolves ``compute`` to a NEW :class:`PythonComputeTool`
          with an empty namespace (and ``compute_namespace`` finds that fresh
          sibling via the fresh cache). Carrying the parent's cache over would
          re-share the parent's compute singleton and defeat isolation.

        Every OTHER :class:`ToolFactoryContext` field is preserved verbatim by
        ``dataclasses.replace`` — crucially the identity fields
        (``workspace_client``, ``user_token``, ``serving_client_provider``) and
        host-injected extras like ``_skill_store`` — so the isolated child runs
        as the SAME principal (OBO) as the parent; only its scratchpad differs.
        """
        from databricks_deep_research.api.vfs.in_memory import InMemoryBackend

        parent_ctx = self._resolver.factory_context
        # Copy the parent extras MINUS the per-resolver cache (the ToolResolver
        # ctor installs a fresh one), then point the VFS at a private backend.
        child_extras: dict[str, Any] = {
            key: value
            for key, value in parent_ctx.extras.items()
            # _sandbox_session diverges like _resolver_cache: the isolated child
            # gets its own persistent sandbox REPL (fresh scratchpad).
            if key not in ("_resolver_cache", "_sandbox_session")
        }
        child_extras["_framework_vfs"] = InMemoryBackend()
        child_factory_ctx = dataclasses.replace(parent_ctx, extras=child_extras)
        child_resolver = ToolResolver(
            declarations=list(child_defn.tools) if child_defn.tools else None,
            factories=self._resolver._factories,
            factory_context=child_factory_ctx,
            legacy_registry=self._resolver._legacy,
        )
        # Carry runtime overrides (enterprise + per-request MCP tools) so tool
        # nodes inside isolate children still resolve. Sharing the instances is
        # identity-preserving (same OBO principal) per the contract above —
        # only the scratchpad diverges.
        for override_name, override_tool in self._resolver._overrides.items():
            child_resolver.override(override_name, override_tool)
        return child_resolver

    async def _run_inline_subworkflow(
        self,
        inline: dict[str, Any],
        *,
        query: str,
        depth: int,
        parent_state: WorkflowState,
        pool_mode: str = "inherit",
        seed_inputs: dict[str, Any] | None = None,
        result_sink: dict[str, Any] | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Run a DECLARED inline subworkflow as a child, yielding its events.

        The single shared spine behind BOTH ``_exec_subworkflow`` (the
        subworkflow node) and the GOVERNED ``spawn_agent`` runner (spec §3.3), so
        the two cannot drift. It validates ``inline``, builds the child resolver
        per ``pool_mode`` (``isolate`` => fresh compute namespace + private VFS;
        identity/OBO preserved), constructs a fresh child :class:`WorkflowState`
        seeded with ``query`` + ``seed_inputs``, wires pools per ``pool_mode``,
        drives the child root directly (never ``execute()``), folds ``merge``
        pools back, and stashes ``{"primary": <value>, "child_state": <state>}``
        into ``result_sink`` (when provided) for the caller to read AFTER the
        generator is fully consumed.

        ``depth`` is the child's ``_subworkflow_depth`` — callers pass
        ``self._subworkflow_depth + 1`` AFTER applying their own recursion guard,
        so the bound is enforced identically on both paths.
        """
        child_defn = WorkflowDefinition.model_validate(inline)

        # Scratchpad (compute namespace + VFS) scope follows ``pool_mode``:
        #   * inherit / merge — share the parent resolver, so the child sees the
        #     SAME cached ``compute`` singleton (one namespace) and the SAME
        #     ``_framework_vfs``. Producer→consumer by handle, no re-serialisation.
        #   * isolate — give the child its OWN resolver with a fresh compute
        #     namespace and a private VFS, so child scratchpad writes never leak to
        #     the parent and parent scratchpad state is hidden from the child. Only
        #     ``output_mapping`` / the primary ``output_key`` (and ``submit()``)
        #     surface results back across the boundary.
        if pool_mode == "isolate":
            child_resolver = self._build_isolated_child_resolver(child_defn)
        else:
            child_resolver = self._resolver

        # Child executor shares parent tool/identity context (isolate swaps in a
        # private resolver above; URL/table registries + ExecutionContext, hence
        # OBO identity, always carry over).
        child_exec = WorkflowExecutor(
            child_defn,
            self._llm,
            tool_resolver=child_resolver,
            url_registry=self._url_registry,
            table_registry=self._table_registry,
            context=self._context,
        )
        child_exec._subworkflow_depth = depth

        # Fresh child state carrying the parent's caller identity/config. A clean
        # log/index keeps the child's dataflow scoped (no wholesale log copy).
        child_state = WorkflowState(
            query=query,
            model_overrides=dict(parent_state.model_overrides),
            enterprise_tools=list(parent_state.enterprise_tools),
            user_token=parent_state.user_token,
            domain_filter=parent_state.domain_filter,
            conversation_history=list(parent_state.conversation_history),
            is_cancelled=parent_state.is_cancelled,
        )

        # Seed declared inputs (subworkflow node: input_mapping + params; spawn:
        # nothing — the prompt rides on ``query``).
        for child_key, value in (seed_inputs or {}).items():
            child_state.append(child_defn.id, child_key, value)

        # Pool wiring per pool_mode (the compute scratchpad + VFS were already
        # scoped per pool_mode above via the child resolver choice).
        if pool_mode == "inherit":
            # Bind the child to the parent's pools so child nodes read and write
            # the SAME pools the parent sees. Agent nodes write via the harness'
            # ``pools`` argument, which the child ``_exec_agent`` sources from
            # ``child_exec._pools`` (NOT ``child_state.pools``) — so we must point
            # BOTH the child executor's pools and the child state's pools at the
            # parent pools for inheritance to actually take effect.
            child_exec._pools = parent_state.pools
            child_state.pools = parent_state.pools
        else:
            # ``isolate`` and ``merge`` both run against the child's own fresh
            # pools (built from child_defn.pools); ``merge`` reconciles after.
            child_state.pools = child_exec._pools

        # Drive the child's root directly, propagating parent cancellation in.
        try:
            async for event in child_exec._exec_node(child_defn.root, child_state):
                if parent_state.is_cancelled:
                    child_state.is_cancelled = True
                yield event
        finally:
            if pool_mode == "isolate":
                # The isolated child owns a private sandbox REPL (its holder was
                # excluded from the copied extras); close it with the child.
                try:
                    await self._close_sandbox_session(child_resolver)
                except Exception:  # pragma: no cover — cleanup must never mask the run
                    logger.debug("CHILD_SANDBOX_CLOSE_FAILED", exc_info=True)

        # ``merge``: fold child pool contents back into the parent pools.
        if pool_mode == "merge":
            for pool_name, child_pool in child_state.pools.items():
                parent_pool = parent_state.pools.get(pool_name)
                if parent_pool is None:
                    continue
                for item in child_pool.snapshot():
                    parent_pool.add(item)

        # Compute the child's primary result: the last declared output key of the
        # child definition (``output_keys`` always has at least the default
        # "output"), falling back to the last log entry.
        primary_value: Any = None
        if child_defn.output_keys:
            primary_value = child_state.get(child_defn.output_keys[-1])
        if primary_value is None and child_state.log:
            primary_value = child_state.log[-1].value

        if result_sink is not None:
            result_sink["primary"] = primary_value
            result_sink["child_state"] = child_state

    async def _exec_subworkflow(
        self, node: WorkflowNode, state: WorkflowState
    ) -> AsyncGenerator[StreamEvent, None]:
        """Run a nested workflow as a child of this node.

        The parent ``_exec_node`` that dispatched here already supplies this
        node's ``NodeStarted``/``NodeCompleted`` events and full
        ``error_handling`` — so this handler simply does its work and raises on
        failure. Child node errors are handled by the *child's* ``_exec_node``
        per each child's own ``error_handling`` policy.

        The child runs against a SEPARATE :class:`WorkflowState` and a fresh
        child :class:`WorkflowExecutor`. For ``pool_mode`` inherit/merge it shares
        the parent's tool resolver (hence the parent's compute namespace + VFS);
        for ``isolate`` it gets a private resolver with a fresh compute namespace
        and a private VFS. URL/table registries and the :class:`ExecutionContext`
        (so OBO identity and per-agent tool overrides) always carry over. We drive
        the child's root directly (``_exec_node``), never ``execute()`` — that
        would reset ``runtime_store`` and emit a nested ``WorkflowStartedEvent``.
        """
        config = SubworkflowNodeConfig(**node.config)

        # Recursion guard.
        next_depth = self._subworkflow_depth + 1
        if next_depth > config.max_subworkflow_depth:
            raise WorkflowError(
                f"max_subworkflow_depth ({config.max_subworkflow_depth}) "
                f"exceeded at node '{node.id}'"
            )

        # Resolve the child definition. ``inline`` is the primary path (written
        # by api/compile.py). No named-subworkflow registry exists, so a bare
        # ``ref`` is not resolvable.
        if config.inline is None:
            raise WorkflowError(
                f"subworkflow ref '{config.ref}' is not resolvable; "
                f"provide an inline definition"
            )

        # Seed declared inputs, then params (params override/augment). Built here
        # so the shared spine stays node-agnostic.
        seed_inputs: dict[str, Any] = {}
        for child_key, parent_key in config.input_mapping.items():
            seed_inputs[child_key] = state.get(parent_key)
        for param_key, param_value in config.params.items():
            seed_inputs[param_key] = param_value

        result_sink: dict[str, Any] = {}
        async for event in self._run_inline_subworkflow(
            config.inline,
            query=state.query,
            depth=next_depth,
            parent_state=state,
            pool_mode=config.pool_mode,
            seed_inputs=seed_inputs,
            result_sink=result_sink,
        ):
            yield event

        child_state: WorkflowState = result_sink["child_state"]

        # Map outputs back to the PARENT state.
        for parent_key, child_key in config.output_mapping.items():
            state.append(node.id, parent_key, child_state.get(child_key))

        # Always expose the child's primary result under ``output_key``.
        state.append(node.id, config.output_key, result_sink["primary"])

    async def _exec_plan_and_execute(
        self, node: WorkflowNode, state: WorkflowState
    ) -> AsyncGenerator[StreamEvent, None]:
        config = PlanAndExecuteNodeConfig(**node.config)
        runtime = PlanExecuteRuntimeContext(
            node=node,
            config=config,
            state=state,
            pools=self._pools,
            llm=self._llm,
            definition=self._defn,
            resolver=self._resolver,
            execution_context=self._context,
        )
        deps = PlanExecuteRunnerDeps(
            emit=self._emit,
            exec_node=self._exec_node,
            execute_agent=execute_agent,
            now=_now,
            logger=logger,
            record_step_completed=self._record_step_completed,
        )
        async for event in run_plan_execute(runtime, deps):
            self._context = runtime.execution_context
            yield event


    # Handler dispatch table
    _handlers: dict[NodeType, Any] = {
        NodeType.sequence: _exec_sequence,
        NodeType.parallel: _exec_parallel,
        NodeType.loop: _exec_loop,
        NodeType.conditional: _exec_conditional,
        NodeType.agent: _exec_agent,
        NodeType.tool: _exec_tool,
        NodeType.subworkflow: _exec_subworkflow,
        NodeType.plan_and_execute: _exec_plan_and_execute,
    }


# ---------------------------------------------------------------------------
# Compatibility exports for existing tests/callers
# ---------------------------------------------------------------------------


_coerce_discovered_sources = _coerce_discovered_sources_impl
_hydrate_pools_from_discovered_sources = _hydrate_pools_from_discovered_sources_impl
_obs_to_text = _obs_to_text_impl
_extract_step_title = _extract_step_title_impl
_format_plan_for_reflector = _format_plan_for_reflector_impl
_format_source_topics = _format_source_topics_impl
_format_all_observations = _format_all_observations_impl
_format_source_quality = _format_source_quality_impl
_populate_synthesis_state = _populate_synthesis_state_impl
_extract_decision = _extract_decision_impl
_extract_reasoning = _extract_reasoning_impl
_extract_evidence_sufficiency = _extract_evidence_sufficiency_impl
_extract_failure_mode = _extract_failure_mode_impl
_normalize_evaluation_decision = _normalize_evaluation_decision_impl
_append_completed_step = _append_completed_step_impl


def _append_replan_feedback(
    cycle_ctx: PlanCycleContext,
    *,
    reason: str,
    cycle: int,
    message: str,
    step_title: str = "",
) -> None:
    _append_replan_feedback_impl(cycle_ctx, reason=reason, cycle=cycle, message=message, step_title=step_title)


async def _build_available_source_catalog(
    definition: WorkflowDefinition,
    resolver: ToolResolver,
    body: dict[str, Any],
) -> list[AvailableSourceDescriptor]:
    return await _build_available_source_catalog_impl(definition, resolver, body)


def _build_planner_runtime_context(
    *,
    config: PlanAndExecuteNodeConfig,
    state: WorkflowState,
    pools: dict[str, Any],
    cycle_ctx: PlanCycleContext,
) -> dict[str, Any]:
    return _build_planner_runtime_context_impl(config=config, state=state, pools=pools, cycle_ctx=cycle_ctx)


def _build_evaluator_runtime_context(
    *,
    config: PlanAndExecuteNodeConfig,
    state: WorkflowState,
    pools: dict[str, Any],
    items: list[Any],
    current_idx: int,
    current_item: Any,
    cycle: int,
    total_items_processed: int,
    replan_cycles: int = 0,
) -> dict[str, Any]:
    context = _build_evaluator_runtime_context_impl(
        config=config,
        state=state,
        pools=pools,
        items=items,
        current_idx=current_idx,
        current_item=current_item,
        cycle=cycle,
        total_items_processed=total_items_processed,
        replan_cycles=replan_cycles,
    )
    context.setdefault("current_step", str(current_idx + 1))
    context.setdefault("replan_budget", (
        f"{config.max_replan_cycles - replan_cycles} of "
        f"{config.max_replan_cycles} remaining"
    ))
    return context





def _extract_items(plan_data: Any, items_path: str) -> list[Any]:
    current = plan_data
    if isinstance(current, str):
        try:
            current = json.loads(current)
        except (json.JSONDecodeError, ValueError):
            if "```json" in current:
                try:
                    start = current.index("```json") + 7
                    end = current.index("```", start)
                    current = json.loads(current[start:end].strip())
                except (json.JSONDecodeError, ValueError, IndexError):
                    pass
    for part in items_path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            return []
        if current is None:
            return []
    if isinstance(current, list):
        return current
    return [current]


async def run_workflow(
    definition: WorkflowDefinition,
    llm_client: FrameworkLLMClient,
    *,
    initial_state: dict[str, Any] | None = None,
    enterprise_tools: list[ResearchTool] | None = None,
    tool_registry: ToolRegistry | None = None,
    tool_factories: list[ToolFactory] | None = None,
    factory_context: ToolFactoryContext | None = None,
    strict_tool_resolution: bool = False,
) -> tuple[WorkflowState, list[StreamEvent]]:
    """Run a workflow to completion and return final state + events.

    Compatibility helper retained while public APIs migrate to typed results.
    """
    state = WorkflowState(query=initial_state.get("query", "") if initial_state else "")
    if initial_state:
        for k, v in initial_state.items():
            if k != "query":
                state.append("init", k, v)

    executor = WorkflowExecutor(
        definition,
        llm_client,
        tool_registry=tool_registry,
        tool_factories=tool_factories,
        factory_context=factory_context,
        strict_tool_resolution=strict_tool_resolution,
        enterprise_tools=enterprise_tools,
    )

    events: list[StreamEvent] = []
    try:
        async for event in executor.execute(state):
            events.append(event)
    except Exception as exc:
        events.append(
            WorkflowFailedEvent(
                node_id=definition.root.id,
                timestamp=_now(),
                workflow_id=definition.id,
                duration_ms=0.0,
                error_type=type(exc).__name__,
                error_message=str(exc),
                total_sources=0,
                total_steps_executed=executor._workflow_total_steps_executed,
            )
        )
        raise WorkflowExecutionError(
            f"Workflow execution failed: {exc}",
            state=state,
            events=events,
            cause=exc,
        ) from exc

    return state, events



async def run_workflow_typed(request: WorkflowRunRequest, llm_client: FrameworkLLMClient) -> WorkflowRunResult:
    state = WorkflowState(query=request.query or request.inputs.get("query", ""))
    if request.inputs:
        for k, v in request.inputs.items():
            if k != "query":
                state.append("init", k, v)

    executor = WorkflowExecutor(
        request.definition,
        llm_client,
        tool_registry=request.tool_registry,
        tool_factories=request.tool_factories,
        factory_context=request.factory_context,
        strict_tool_resolution=request.strict_tool_resolution,
        enterprise_tools=request.enterprise_tools,
    )

    events: list[StreamEvent] = []
    async for event in executor.execute(state):
        events.append(event)
    runtime = state.runtime_state()
    if runtime is None:
        raise WorkflowExecutionError(
            "Typed runtime state was not initialized",
            state=state,
            events=events,
            cause=RuntimeError("Typed runtime state was not initialized"),
        )
    return WorkflowRunResult(runtime_state=runtime, events=events)


async def run_workflow_from_yaml(
    yaml_path: str,
    llm_client: FrameworkLLMClient,
    *,
    initial_state: dict[str, Any] | None = None,
    enterprise_tools: list[ResearchTool] | None = None,
    tool_registry: ToolRegistry | None = None,
    tool_factories: list[ToolFactory] | None = None,
    factory_context: ToolFactoryContext | None = None,
    strict_tool_resolution: bool = False,
) -> tuple[WorkflowState, list[StreamEvent]]:
    from databricks_deep_research.workflow.loader import load_workflow

    definition = load_workflow(yaml_path)
    return await run_workflow(
        definition,
        llm_client,
        initial_state=initial_state,
        enterprise_tools=enterprise_tools,
        tool_registry=tool_registry,
        tool_factories=tool_factories,
        factory_context=factory_context,
        strict_tool_resolution=strict_tool_resolution,
    )
