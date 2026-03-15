from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from databricks_deep_research.agents.config import AgentNodeConfig
from databricks_deep_research.errors import PlanningContractError
from databricks_deep_research.events.types import (
    EvaluationDecisionEvent,
    ItemCompletedEvent,
    ItemsExtractedEvent,
    ItemStartedEvent,
    PlanAndExecuteExitEvent,
    ReflectionDecisionEvent,
    ReplanTriggeredEvent,
    StreamEvent,
)
from databricks_deep_research.tracing import trace_span
from databricks_deep_research.workflow.context import ExecutionContext
from databricks_deep_research.workflow.definition import WorkflowNode
from databricks_deep_research.workflow.runtime.context import (
    PlanExecuteRunnerDeps,
    PlanExecuteRuntimeContext,
)
from databricks_deep_research.workflow.runtime.plan_execute_context import (
    build_evaluator_runtime_context,
    build_planner_runtime_context,
)
from databricks_deep_research.workflow.runtime.plan_execute_contracts import (
    normalize_executable_plan_contract,
)
from databricks_deep_research.workflow.runtime.plan_execute_execution import (
    append_completed_step,
    append_replan_feedback,
    build_available_source_catalog,
    extract_decision,
    extract_reasoning,
    extract_evidence_sufficiency,
    extract_failure_mode,
    normalize_evaluation_decision,
    populate_synthesis_state,
    summarize_item_health,
)
from databricks_deep_research.workflow.runtime.plan_execute_recovery import (
    coerce_discovered_sources,
    hydrate_pools_from_discovered_sources,
)
from databricks_deep_research.workflow.runtime.plan_execute_types import PlanCycleContext


async def run_plan_execute(
    runtime: PlanExecuteRuntimeContext,
    deps: PlanExecuteRunnerDeps,
) -> AsyncGenerator[StreamEvent, None]:
    node = runtime.node
    config = runtime.config
    state = runtime.state
    pools = runtime.pools
    total_items_processed = runtime.total_items_processed
    replan_cycles = runtime.replan_cycles
    cycle_ctx = PlanCycleContext()
    cycle_ctx.available_sources = await build_available_source_catalog(
        runtime.definition,
        runtime.resolver,
        config.body,
    )

    from databricks_deep_research.agents.react_loop import ToolCallCache as _ToolCallCache

    if runtime.execution_context is None:
        runtime.execution_context = ExecutionContext(llm_client=runtime.llm)
    if runtime.execution_context.tool_call_cache is None:
        runtime.execution_context.tool_call_cache = _ToolCallCache()
        deps.logger.info("SHARED_CACHE_CREATED node=%s", node.id)

    deps.logger.info(
        "P_AND_E_START node=%s max_iterations=%d min_iterations=%d max_replan_cycles=%d complete_on_exhaustion=%s",
        node.id,
        config.max_iterations,
        config.min_iterations,
        config.max_replan_cycles,
        config.complete_on_exhaustion,
    )

    state.append(node.id, "min_steps", str(config.min_iterations))
    state.append(node.id, "max_steps", str(config.max_iterations))
    if config.planner_guidance:
        state.append(node.id, "step_prompt_guidance", config.planner_guidance)
    for key, value in config.synthesis_metadata.items():
        state.append(node.id, key, value)

    items: list[Any] = []
    for cycle in range(config.max_replan_cycles + 1):
        cycle_ctx.cycle = cycle
        deps.logger.info(
            "P_AND_E_CYCLE_START node=%s cycle=%d total_items_processed=%d max_iterations=%d",
            node.id,
            cycle,
            total_items_processed,
            config.max_iterations,
        )
        planner_runtime_context = build_planner_runtime_context(
            config=config,
            state=state,
            pools=pools,
            cycle_ctx=cycle_ctx,
        )
        async with trace_span(f"plan_cycle_{cycle}", span_type="CHAIN") as plan_span:
            planner_output = await deps.execute_agent(
                node_id=f"{node.id}_planner_c{cycle}",
                config=AgentNodeConfig(**config.planner),
                state=state,
                llm_client=runtime.llm,
                tools=[],
                pools=pools,
                runtime_context=planner_runtime_context,
            )
            for event in planner_output.events:
                yield event

        plan_data = planner_output.content
        plan_contract = normalize_executable_plan_contract(plan_data, config.items_path)
        items = list(plan_contract["items"])
        has_enough_context = bool(plan_contract["has_enough_context"])
        if state.runtime_store is not None:
            state.runtime_store.begin_plan_cycle(
                cycle=cycle,
                title=str(plan_contract.get("title", "") or ""),
                thought=str(plan_contract.get("thought", "") or ""),
                has_enough_context=has_enough_context,
                steps=[item for item in items if isinstance(item, dict)],
            )

        deps.logger.debug(
            "P_AND_E_PLAN_OUTPUT node=%s cycle=%d plan_type=%s preview=%s",
            node.id,
            cycle,
            type(plan_data).__name__,
            str(plan_data)[:400],
        )
        if plan_span:
            plan_span.set_attributes(
                {
                    "plan.items_count": len(items),
                    "plan.has_enough_context": has_enough_context,
                    "plan.available_sources_count": len(cycle_ctx.available_sources),
                    "plan.repair_mode": str(plan_contract.get("repair_mode", "")),
                    "plan.raw_item_count": int(plan_contract.get("raw_item_count", len(items))),
                    "plan.normalized_item_count": int(plan_contract.get("normalized_item_count", len(items))),
                }
            )

        if items and has_enough_context:
            deps.logger.warning(
                "P_AND_E_CONTRADICTORY_PLAN node=%s cycle=%d reason=has_enough_context_with_steps",
                node.id,
                cycle,
            )
            has_enough_context = False

        if not items:
            deps.logger.warning(
                "P_AND_E_ZERO_ITEMS node=%s cycle=%d items_path=%s plan_type=%s plan_preview=%s",
                node.id,
                cycle,
                config.items_path,
                type(plan_data).__name__,
                str(plan_data)[:300],
            )

        deps.logger.info(
            "P_AND_E_ITEMS_EXTRACTED node=%s cycle=%d total_items=%d items=%s",
            node.id,
            cycle,
            len(items),
            [str(item)[:100] for item in items],
        )
        yield deps.emit(
            ItemsExtractedEvent(
                node_id=node.id,
                timestamp=deps.now(),
                total_items=len(items),
                items_path=config.items_path,
                cycle=cycle,
            )
        )

        if not items:
            discovered_sources = coerce_discovered_sources(state)
            pools_have_content = any(pool.count() > 0 for pool in pools.values()) if pools else False
            planner_sufficient_context = has_enough_context
            if not planner_sufficient_context and discovered_sources:
                can_recover_now = config.min_iterations == 0 or cycle >= config.max_replan_cycles
                if can_recover_now:
                    hydrated = hydrate_pools_from_discovered_sources(pools, discovered_sources)
                    pools_have_content = any(pool.count() > 0 for pool in pools.values()) if pools else False
                    planner_sufficient_context = hydrated or pools_have_content
            planner_reason = "planner_sufficient_context" if planner_sufficient_context else "empty_plan"
            if planner_sufficient_context and (
                total_items_processed >= config.min_iterations or pools_have_content
            ):
                populate_synthesis_state(
                    node.id,
                    state,
                    pools,
                    total_items_processed,
                    replan_cycles,
                )
                if state.runtime_store is not None:
                    state.runtime_store.finalize_plan_cycle(cycle=cycle, exit_reason=planner_reason)
                yield deps.emit(
                    PlanAndExecuteExitEvent(
                        node_id=node.id,
                        timestamp=deps.now(),
                        reason=planner_reason,
                        total_items_processed=total_items_processed,
                        total_planned=len(items),
                        replan_cycles=replan_cycles,
                    )
                )
                return
            if cycle < config.max_replan_cycles:
                replan_cycles += 1
                append_replan_feedback(
                    cycle_ctx,
                    reason="empty_plan",
                    cycle=cycle,
                    message="Planner returned zero executable steps.",
                )
                if state.runtime_store is not None:
                    state.runtime_store.finalize_plan_cycle(cycle=cycle, exit_reason="empty_plan")
                yield deps.emit(
                    ReplanTriggeredEvent(
                        node_id=node.id,
                        timestamp=deps.now(),
                        reason="empty_plan",
                        cycle=cycle + 1,
                        items_remaining=0,
                    )
                )
                continue
            raise PlanningContractError("empty_plan", f"Node {node.id} produced zero executable steps")

        for idx, item in enumerate(items):
            if total_items_processed >= config.max_iterations:
                break

            state.append(node.id, config.item_state_key, item)
            yield deps.emit(
                ItemStartedEvent(
                    node_id=node.id,
                    timestamp=deps.now(),
                    item_index=idx,
                    item_summary=str(item)[:200],
                    total_items=len(items),
                )
            )

            item_events: list[StreamEvent] = []
            sources_before = pools.get("sources").count() if "sources" in pools else 0
            async with trace_span(
                f"item_{idx}",
                span_type="CHAIN",
                attributes={"item.index": idx, "item.summary": str(item)[:200]},
            ):
                if config.body:
                    body_node = WorkflowNode(**config.body)
                    async for event in deps.exec_node(body_node, state):
                        item_events.append(event)
                        yield event
            sources_after = pools.get("sources").count() if "sources" in pools else 0
            item_health = summarize_item_health(
                item,
                item_events,
                sources_before=sources_before,
                sources_after=sources_after,
            )
            if item_health["blocked"]:
                state.append(node.id, "last_blocked_step", item_health)
                if isinstance(item, dict) and state.runtime_store is not None:
                    state.runtime_store.mark_step_blocked(
                        step_id=str(item.get("id", f"step-{idx}")),
                        reason=str(item_health.get("reason") or "blocked_step"),
                    )

            total_items_processed += 1
            append_completed_step(cycle_ctx, item)
            deps.record_step_completed()
            if isinstance(item, dict) and state.runtime_store is not None:
                state.runtime_store.mark_step_completed(
                    step_id=str(item.get("id", f"step-{idx}"))
                )
            yield deps.emit(
                ItemCompletedEvent(
                    node_id=node.id,
                    timestamp=deps.now(),
                    item_index=idx,
                    items_processed=total_items_processed,
                )
            )

            if config.evaluator:
                evaluator_runtime_context = build_evaluator_runtime_context(
                    config=config,
                    state=state,
                    pools=pools,
                    items=items,
                    current_idx=idx,
                    current_item=item,
                    cycle=cycle,
                    total_items_processed=total_items_processed,
                    replan_cycles=replan_cycles,
                )
                evaluator_output = await deps.execute_agent(
                    node_id=f"{node.id}_eval_i{idx}",
                    config=AgentNodeConfig(**config.evaluator),
                    state=state,
                    llm_client=runtime.llm,
                    tools=[],
                    pools=pools,
                    runtime_context=evaluator_runtime_context,
                )
                for event in evaluator_output.events:
                    if not isinstance(event, ReflectionDecisionEvent):
                        yield event
                decision = normalize_evaluation_decision(
                    extract_decision(evaluator_output.content)
                )
                reasoning = extract_reasoning(evaluator_output.content)
                evidence_sufficiency = extract_evidence_sufficiency(evaluator_output.content)
                failure_mode = extract_failure_mode(evaluator_output.content)
                if item_health["blocked"] and decision == "continue":
                    decision = "replan"
                    blocked_reason = str(item_health["reason"])
                    reasoning = (
                        f"{reasoning}\n\nBlocked step detected: {blocked_reason}"
                        if reasoning
                        else f"Blocked step detected: {blocked_reason}"
                    )
                yield deps.emit(
                    EvaluationDecisionEvent(
                        node_id=node.id,
                        timestamp=deps.now(),
                        decision=decision,
                        reasoning=reasoning,
                        items_processed=total_items_processed,
                        evidence_sufficiency=evidence_sufficiency,
                        failure_mode=failure_mode,
                    )
                )
                if decision == "complete" and total_items_processed < config.min_iterations:
                    decision = "continue"
                if decision == "complete":
                    completion_mode = "degraded" if evidence_sufficiency in {"partial", "insufficient"} else "normal"
                    exit_reason = "insufficient_evidence_exhausted" if completion_mode == "degraded" else "evaluator_complete"
                    populate_synthesis_state(
                        node.id,
                        state,
                        pools,
                        total_items_processed,
                        replan_cycles,
                        completion_mode=completion_mode,
                        evidence_sufficiency=evidence_sufficiency,
                        failure_mode=failure_mode,
                    )
                    yield deps.emit(
                        PlanAndExecuteExitEvent(
                            node_id=node.id,
                            timestamp=deps.now(),
                            reason=exit_reason,
                            total_items_processed=total_items_processed,
                            replan_cycles=replan_cycles,
                            total_planned=len(items),
                            completion_mode=completion_mode,
                            evidence_sufficiency=evidence_sufficiency,
                            failure_mode=failure_mode,
                        )
                    )
                    return
                if decision == "replan":
                    if replan_cycles < config.max_replan_cycles:
                        replan_cycles += 1
                        append_replan_feedback(
                            cycle_ctx,
                            reason="blocked_step"
                            if item_health["blocked"]
                            else "evaluator_replan",
                            cycle=cycle,
                            message=reasoning,
                            step_title=str(item),
                        )
                        yield deps.emit(
                            ReplanTriggeredEvent(
                                node_id=node.id,
                                timestamp=deps.now(),
                                cycle=replan_cycles,
                                reason="blocked_step"
                                if item_health["blocked"]
                                else "evaluator_replan",
                                items_remaining=len(items) - idx - 1,
                            )
                        )
                        break
                    populate_synthesis_state(
                        node.id,
                        state,
                        pools,
                        total_items_processed,
                        replan_cycles,
                    )
                    yield deps.emit(
                        PlanAndExecuteExitEvent(
                            node_id=node.id,
                            timestamp=deps.now(),
                            reason="items_exhausted",
                            total_items_processed=total_items_processed,
                            replan_cycles=replan_cycles,
                            total_planned=len(items),
                        )
                    )
                    return
            elif item_health["blocked"]:
                if replan_cycles < config.max_replan_cycles:
                    replan_cycles += 1
                    yield deps.emit(
                        ReplanTriggeredEvent(
                            node_id=node.id,
                            timestamp=deps.now(),
                            cycle=replan_cycles,
                            reason="blocked_step",
                            items_remaining=len(items) - idx - 1,
                        )
                    )
                    break
                populate_synthesis_state(
                    node.id,
                    state,
                    pools,
                    total_items_processed,
                    replan_cycles,
                )
                yield deps.emit(
                    PlanAndExecuteExitEvent(
                        node_id=node.id,
                        timestamp=deps.now(),
                        reason="blocked_step",
                        total_items_processed=total_items_processed,
                        replan_cycles=replan_cycles,
                        total_planned=len(items),
                    )
                )
                return
        else:
            if total_items_processed < config.min_iterations:
                if replan_cycles < config.max_replan_cycles:
                    replan_cycles += 1
                    append_replan_feedback(
                        cycle_ctx,
                        reason="min_iterations_unmet",
                        cycle=cycle,
                        message="All planned steps exhausted before reaching minimum iterations.",
                    )
                    yield deps.emit(
                        ReplanTriggeredEvent(
                            node_id=node.id,
                            timestamp=deps.now(),
                            reason="min_iterations_unmet",
                            cycle=replan_cycles,
                            items_remaining=0,
                        )
                    )
                    continue
                populate_synthesis_state(
                    node.id,
                    state,
                    pools,
                    total_items_processed,
                    replan_cycles,
                )
                yield deps.emit(
                    PlanAndExecuteExitEvent(
                        node_id=node.id,
                        timestamp=deps.now(),
                        reason="min_iterations_unmet",
                        total_items_processed=total_items_processed,
                        replan_cycles=replan_cycles,
                        total_planned=len(items),
                    )
                )
                return
            if config.complete_on_exhaustion or not config.evaluator:
                populate_synthesis_state(
                    node.id,
                    state,
                    pools,
                    total_items_processed,
                    replan_cycles,
                )
                yield deps.emit(
                    PlanAndExecuteExitEvent(
                        node_id=node.id,
                        timestamp=deps.now(),
                        reason="items_exhausted",
                        total_items_processed=total_items_processed,
                        replan_cycles=replan_cycles,
                        total_planned=len(items),
                    )
                )
                return
            continue
        continue

    populate_synthesis_state(
        node.id,
        state,
        pools,
        total_items_processed,
        replan_cycles,
    )
    yield deps.emit(
        PlanAndExecuteExitEvent(
            node_id=node.id,
            timestamp=deps.now(),
            reason="items_exhausted" if total_items_processed > 0 else "max_replan_cycles",
            total_items_processed=total_items_processed,
            replan_cycles=replan_cycles,
            total_planned=len(items) if items else 0,
        )
    )
