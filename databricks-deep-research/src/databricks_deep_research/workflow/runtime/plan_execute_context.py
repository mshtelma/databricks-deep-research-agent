from __future__ import annotations

from databricks_deep_research.workflow.runtime_core.selectors import (
    select_all_observations_text,
    select_current_plan_title,
    select_latest_observation_text,
    select_sources_count,
)
from typing import Any

from databricks_deep_research.agents.config import PlanAndExecuteNodeConfig
from databricks_deep_research.workflow.runtime.planner_context import (
    build_planner_runtime_context as _build_planner_runtime_context_impl,
    format_available_sources as _format_available_sources_impl,
    format_completed_steps as _format_completed_steps_impl,
    format_reflector_feedback as _format_reflector_feedback_impl,
)
from databricks_deep_research.workflow.runtime.plan_execute_formatting import (
    extract_step_title,
    format_all_observations,
    format_plan_for_reflector,
    format_source_quality,
    format_source_topics,
)
from databricks_deep_research.workflow.runtime.plan_execute_types import AvailableSourceDescriptor, PlanCycleContext, ReplanFeedbackEntry
from databricks_deep_research.workflow.state import WorkflowState


def format_completed_steps(completed_steps: list[str]) -> str:
    return _format_completed_steps_impl(completed_steps)


def format_reflector_feedback(entries: list[ReplanFeedbackEntry]) -> str:
    return _format_reflector_feedback_impl(entries)


def format_available_sources(sources: list[AvailableSourceDescriptor]) -> str:
    return _format_available_sources_impl(sources)


def build_planner_runtime_context(*, config: PlanAndExecuteNodeConfig, state: WorkflowState, pools: dict[str, Any], cycle_ctx: PlanCycleContext) -> dict[str, Any]:
    return _build_planner_runtime_context_impl(config=config, state=state, pools=pools, cycle_ctx=cycle_ctx, format_all_observations=format_all_observations)


def build_evaluator_runtime_context(*, config: PlanAndExecuteNodeConfig, state: WorkflowState, pools: dict[str, Any], items: list[Any], current_idx: int, current_item: Any, cycle: int, total_items_processed: int, replan_cycles: int) -> dict[str, Any]:
    remaining = items[current_idx + 1:]
    remaining_text = "\n".join(f"  Step {current_idx + j + 2}/{len(items)}: {extract_step_title(step)}" for j, step in enumerate(remaining)) or "(none — all steps completed)"
    sources_pool = pools.get("sources")
    return {
        "remaining_steps": remaining_text,
        "total_steps": str(len(items)),
        "steps_completed": str(total_items_processed),
        "min_steps": str(config.min_iterations),
        "step_title": extract_step_title(current_item),
        "plan_summary": format_plan_for_reflector(select_current_plan_title(state), items, current_idx),
        "iteration": str(cycle),
        "observation": select_latest_observation_text(state),
        "sources_count": str(select_sources_count(state, pools)),
        "source_topics": format_source_topics(pools),
        "source_quality": format_source_quality(pools),
        "all_observations": select_all_observations_text(state) or format_all_observations(pools),
        "replan_cycles": str(replan_cycles),
    }
