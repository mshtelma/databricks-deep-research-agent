from __future__ import annotations

from databricks_deep_research.workflow.runtime_core.selectors import select_data_landscape, select_discovered_sources
from typing import Any


def format_completed_steps(completed_steps: list[str]) -> str:
    if not completed_steps:
        return "(none completed yet)"
    return "\n".join(f"- Step {idx + 1}: {title}" for idx, title in enumerate(completed_steps))


def format_reflector_feedback(entries: list[Any]) -> str:
    if not entries:
        return "(none)"
    recent = entries[-3:]
    lines: list[str] = []
    for entry in recent:
        prefix = f"Cycle {entry.cycle + 1} [{entry.reason}]"
        if entry.step_title:
            prefix += f" step={entry.step_title}"
        lines.append(f"- {prefix}: {entry.message}")
    rendered = "\n".join(lines)
    return rendered if len(rendered) <= 1200 else rendered[:1200] + "\n...(truncated)"


def format_available_sources(sources: list[Any]) -> str:
    if not sources:
        return "(no explicit source catalog available)"
    lines: list[str] = []
    for source in sources:
        summary = source.description or "No description provided."
        detail = f"{source.source_name} [{source.tool_kind}/{source.source_kind}]"
        if source.endpoint:
            detail += f" endpoint={source.endpoint}"
        lines.append(f"- {detail}: {summary}")
    return "\n".join(lines)


def build_planner_runtime_context(
    *,
    config: Any,
    state: Any,
    pools: dict[str, Any],
    cycle_ctx: Any,
    format_all_observations: Any,
) -> dict[str, Any]:
    return {
        "min_steps": str(config.min_iterations),
        "max_steps": str(config.max_iterations),
        "step_prompt_guidance": config.planner_guidance,
        "iteration": str(cycle_ctx.cycle + 1),
        "completed_steps": format_completed_steps(cycle_ctx.completed_steps),
        "all_observations": format_all_observations(pools),
        "reflector_feedback": format_reflector_feedback(cycle_ctx.feedback_history),
        "available_sources": format_available_sources(cycle_ctx.available_sources),
        "data_landscape": select_data_landscape(state),
        "discovered_sources": select_discovered_sources(state),
    }
