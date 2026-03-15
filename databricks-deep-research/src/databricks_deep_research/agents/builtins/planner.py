"""Planner builtin — research plan generation.

Generates structured plans with ordered steps, supporting depth-aware
step counts and source-aware planning.  Emits ``PlanCreatedEvent``.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from databricks_deep_research.agents.builtins.registry import register_builtin
from databricks_deep_research.agents.config import AgentNodeConfig
from databricks_deep_research.agents.output_models import PlanOutput
from databricks_deep_research.events.types import (
    PlanCreatedEvent,
    StreamEvent,
)
from databricks_deep_research.workflow.runtime.plan_execute_contracts import normalize_executable_plan_contract
from databricks_deep_research.workflow.state import WorkflowState

logger = logging.getLogger(__name__)




def _post_process(
    node_id: str,
    output: Any,
    _config: AgentNodeConfig,
    _state: WorkflowState,
) -> list[StreamEvent]:
    """Emit PlanCreatedEvent from planner output."""
    if not isinstance(output, (PlanOutput, dict)):
        return []

    plan_data = output.model_dump(mode="json") if isinstance(output, PlanOutput) else output
    normalized = normalize_executable_plan_contract(plan_data, "steps")
    title = str(normalized.get("title") or (output.title if isinstance(output, PlanOutput) else output.get("title", "Research Plan")))
    thought = str(normalized.get("thought") or (output.thought if isinstance(output, PlanOutput) else output.get("thought", "")))
    steps = list(normalized.get("items") or [])
    iteration = output.iteration if isinstance(output, PlanOutput) else output.get("iteration", 1)
    has_enough = bool(normalized.get("has_enough_context", output.has_enough_context if isinstance(output, PlanOutput) else output.get("has_enough_context", False)))

    logger.info(
        "PLAN_CREATED title=%s iteration=%s steps=%d has_enough_context=%s",
        title,
        iteration,
        len(steps),
        has_enough,
    )
    logger.debug(
        "PLAN_CREATED_DETAIL title=%s thought=%s step_titles=%s",
        title,
        thought[:200],
        [
            step.get("title", "") if isinstance(step, dict) else str(step)[:120]
            for step in steps
        ],
    )

    return [
        PlanCreatedEvent(
            node_id=node_id,
            timestamp=datetime.now(tz=UTC).isoformat(),
            plan_id=f"{node_id}_iter{iteration}",
            title=title,
            thought=thought,
            steps=steps,
            iteration=iteration,
            has_enough_context=has_enough,
        )
    ]


def _enrich_config(
    config: AgentNodeConfig,
    _state: WorkflowState,
    runtime_context: dict[str, Any] | None = None,
) -> AgentNodeConfig:
    """Fill in planner defaults if not specified."""
    updates: dict[str, Any] = {}
    context = runtime_context or {}
    has_available_sources = bool(context.get("available_sources"))
    has_data_landscape = bool(
        context.get("data_landscape")
        or context.get("discovered_sources")
        or _state.get("data_landscape")
        or _state.get("discovered_sources")
    )

    if not config.system_prompt:
        from databricks_deep_research.agents.prompts.planner import (
            PLANNER_SYSTEM_PROMPT,
            SOURCE_AWARE_PLANNER_SYSTEM_PROMPT,
        )
        updates["system_prompt"] = (
            SOURCE_AWARE_PLANNER_SYSTEM_PROMPT
            if has_available_sources or has_data_landscape
            else PLANNER_SYSTEM_PROMPT
        )

    if not config.user_prompt_template:
        from databricks_deep_research.agents.prompts.planner import (
            PLANNER_USER_PROMPT,
            SOURCE_AWARE_PLANNER_USER_PROMPT,
        )
        updates["user_prompt_template"] = (
            SOURCE_AWARE_PLANNER_USER_PROMPT
            if has_available_sources or has_data_landscape
            else PLANNER_USER_PROMPT
        )

    if config.output_model is None:
        updates["output_model"] = PlanOutput

    # Ensure JSON parsing on the fallback path (when structured output
    # parsing fails and the harness falls through to _parse_output).
    if config.output_format == "text":
        updates["output_format"] = "json"

    if updates:
        return config.model_copy(update=updates)
    return config


register_builtin(
    "planner",
    post_process=_post_process,
    enrich_config=_enrich_config,
    output_model=PlanOutput,
)

__all__: list[str] = []
