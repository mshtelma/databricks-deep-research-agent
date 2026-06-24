from __future__ import annotations

from typing import Any

from databricks_deep_research.workflow.runtime_core.selectors import (
    select_background_summary,
    select_data_landscape,
    select_discovered_sources,
)


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


def format_knowledge_gaps(entries: list[Any]) -> str:
    """Render reflector-emitted knowledge gaps from the most recent replan entries.

    Reads ``knowledge_gaps`` off the recent :class:`ReplanFeedbackEntry`
    records (the same history :func:`format_reflector_feedback` summarizes),
    deduplicates while preserving order, and returns ``""`` when there are no
    gaps so the planner prompt's slot collapses to nothing (== today).
    """
    seen: set[str] = set()
    gaps: list[str] = []
    for entry in entries[-3:]:
        for gap in getattr(entry, "knowledge_gaps", None) or []:
            text = str(gap).strip()
            if text and text not in seen:
                seen.add(text)
                gaps.append(text)
    if not gaps:
        return ""
    rendered = "\n".join(f"- {gap}" for gap in gaps)
    return rendered if len(rendered) <= 1200 else rendered[:1200] + "\n...(truncated)"


def _select_query_decomposition(state: Any) -> list[str]:
    """Sub-questions the background investigator decomposed the query into.

    Sourced from the same ``runtime.capabilities.background`` payload the other
    background selectors read (with a plain-state fallback), so it is empty
    whenever the background node did not run or produced none.
    """
    from databricks_deep_research.workflow.runtime_core.selectors import get_runtime

    runtime = get_runtime(state)
    background = getattr(runtime.capabilities, "background", None) if runtime else None
    if isinstance(background, dict):
        value = background.get("query_decomposition")
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
    fallback = state.get("query_decomposition") if hasattr(state, "get") else None
    if isinstance(fallback, list):
        return [str(item).strip() for item in fallback if str(item).strip()]
    return []


def format_background(state: Any) -> str:
    """Render the pre-search background summary + query decomposition.

    Fills the EXISTING ``{background}`` slot in both planner prompts. Returns
    ``""`` when the always-on background node produced nothing, so an empty
    background renders exactly as it does today.
    """
    summary = select_background_summary(state).strip()
    decomposition = _select_query_decomposition(state)
    sections: list[str] = []
    if summary:
        sections.append(summary)
    if decomposition:
        bullets = "\n".join(f"- {sub}" for sub in decomposition[:10])
        sections.append("Query decomposition (sub-questions identified):\n" + bullets)
    return "\n\n".join(sections)


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
        "knowledge_gaps": format_knowledge_gaps(cycle_ctx.feedback_history),
        "available_sources": format_available_sources(cycle_ctx.available_sources),
        "data_landscape": select_data_landscape(state),
        "discovered_sources": select_discovered_sources(state),
        "background": format_background(state),
    }
