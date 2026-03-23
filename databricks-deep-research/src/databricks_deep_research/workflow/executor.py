"""Workflow executor — tree walker that handles all 8 node types.

The executor walks the workflow tree depth-first, yielding StreamEvent
objects as an async generator.  Each node type has a dedicated handler.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from databricks_deep_research.agents.config import (
    AgentNodeConfig,
    ConditionalNodeConfig,
    LoopNodeConfig,
    PlanAndExecuteNodeConfig,
    ToolNodeConfig,
)
from databricks_deep_research.agents.harness import execute_agent
from databricks_deep_research.errors import (
    NodeBudgetExceededError,
    WorkflowCancelledError,
    WorkflowError,
    WorkflowExecutionError,
)
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
from databricks_deep_research.tools.factory import ToolFactory, ToolFactoryContext
from databricks_deep_research.tools.protocol import ResearchTool, ToolContext, UrlRegistry
from databricks_deep_research.tools.registry import ToolRegistry
from databricks_deep_research.tools.resolver import ToolResolver
from databricks_deep_research.tracing import trace_span
from databricks_deep_research.workflow.conditions import (
    StateCondition,
    evaluate_state_condition,
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


def _deserialize_condition(cond: Any) -> StateCondition:
    """Convert a dict/model into a StateCondition for evaluation."""
    if isinstance(cond, StateCondition):
        return cond
    if isinstance(cond, dict):
        return StateCondition(**{k: v for k, v in cond.items() if k != "type"})
    raise TypeError(f"Cannot deserialize condition: {cond!r}")


def _state_to_eval_dict(state: WorkflowState) -> dict[str, Any]:
    """Build a flat dict of latest state values for condition evaluation.

    ``evaluate_state_condition`` uses ``resolve_dot_path`` which expects
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
        context: ExecutionContext | None = None,
    ) -> None:
        self._defn = definition
        self._llm = llm_client
        self._url_registry = url_registry or UrlRegistry()
        self._context = context
        self._total_tokens: int = 0
        self._strict_tool_resolution = strict_tool_resolution
        self._workflow_total_sources_raw = 0
        self._workflow_total_sources_accepted = 0
        self._workflow_blocked_steps = 0
        self._workflow_missing_declared_tools = 0
        self._workflow_plan_exit_reasons: list[str] = []
        self._workflow_total_steps_executed = 0

        registry = tool_registry or ToolRegistry()
        resolved_factories: list[ToolFactory] = list(tool_factories or [BuiltinToolFactory(), DatabricksToolFactory()])
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

            elapsed_ms = (time.monotonic() - start) * 1000
            total_sources = self._pools.get(
                "sources", PoolState(PoolConfig(name="_"))
            ).count()

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
            elif isinstance(report_value, dict) and report_value.get("output_type"):
                structured = report_value

            yield self._emit(WorkflowCompletedEvent(
                node_id=self._defn.root.id,
                timestamp=_now(),
                workflow_id=self._defn.id,
                duration_ms=elapsed_ms,
                total_tokens=self._total_tokens,
                final_report=report_value.model_dump_json() if isinstance(report_value, BaseModel) else str(report_value or ""),
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
                        ))
                        raise
                    return
                else:
                    yield self._emit(NodeErrorEvent(
                        node_id=node.id, timestamp=_now(),
                        error_message=str(exc),
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
                    should_exit = evaluate_state_condition(cond, _state_to_eval_dict(state))
                    if should_exit:
                        yield self._emit(LoopExitEvent(
                            node_id=node.id, timestamp=_now(),
                            reason="condition_met",
                            total_iterations=iteration,
                        ))
                        return
                except Exception:
                    yield self._emit(LoopExitEvent(
                        node_id=node.id, timestamp=_now(),
                        reason="parse_failure",
                        total_iterations=iteration,
                    ))
                    return

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

        eval_dict = _state_to_eval_dict(state)
        for i, cond_dict in enumerate(config.conditions):
            try:
                cond = _deserialize_condition(cond_dict)
                if evaluate_state_condition(cond, eval_dict):
                    selected_idx = i
                    break
            except Exception:
                continue

        yield self._emit(BranchSelectedEvent(
            node_id=node.id, timestamp=_now(),
            branch_index=selected_idx,
            condition_summary=f"Branch {selected_idx} selected",
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
        errors: list[str] = []
        for ref in config.tools:
            try:
                tool = await self._resolver.resolve(ref)
                tools.append(tool)
            except ValueError as exc:
                ref_name = ref if isinstance(ref, str) else ref.get("name", str(ref))
                errors.append(str(ref_name))
                logger.warning("TOOL_NOT_FOUND ref=%s error=%s", ref, exc)

        logger.info(
            "AGENT_TOOLS_RESOLVED node=%s config_tool_refs=%d "
            "resolved_tools=%d tool_names=%s max_tool_calls=%d",
            node.id,
            len(config.tools),
            len(tools),
            [t.definition.name for t in tools],
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

        output = await execute_agent(
            node_id=node.id,
            config=config,
            state=state,
            llm_client=self._llm,
            tools=tools,
            pools=self._pools,
            url_registry=self._url_registry,
            tool_call_cache=self._context.tool_call_cache if self._context else None,
        )

        # Track token usage
        agent_tokens = int(output.token_usage.get("total_tokens") or sum(v for k, v in output.token_usage.items() if k != "total_tokens"))
        self._total_tokens += agent_tokens
        logger.info(
            "AGENT_TOKENS node=%s tokens=%d cumulative=%d usage=%s",
            node.id, agent_tokens, self._total_tokens, output.token_usage,
        )

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
        """Execute a direct tool call (no LLM)."""
        config = ToolNodeConfig(**node.config)

        from databricks_deep_research.tools.protocol import ToolRef
        ref_dict = config.ref
        ref = ToolRef(type=ref_dict.get("type", "builtin"), name=ref_dict.get("name", ""))
        tool = self._tool_registry.resolve(ref)

        # Map inputs from state
        args: dict[str, Any] = {}
        for arg_name, state_key in config.input_mapping.items():
            args[arg_name] = state.get(state_key)

        async with trace_span(
            f"tool.{ref.name}",
            span_type="TOOL",
            attributes={"tool.name": ref.name},
        ) as tool_span:
            validated = tool.validate_arguments(args)
            ctx = ToolContext(query=state.query, url_registry=self._url_registry)
            result = await tool.execute(validated, ctx)

            if tool_span:
                tool_span.set_attributes({"tool.result_len": len(result.content)})

        state.append(node.id, config.output_key, result.content)
        return
        yield  # pragma: no cover — make this an async generator

    async def _exec_subworkflow(
        self, _node: WorkflowNode, _state: WorkflowState
    ) -> AsyncGenerator[StreamEvent, None]:
        """Execute a subworkflow (deferred to P2)."""
        raise NotImplementedError("Subworkflow execution is deferred to P2")
        yield  # Make this an async generator  # noqa: B033

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
