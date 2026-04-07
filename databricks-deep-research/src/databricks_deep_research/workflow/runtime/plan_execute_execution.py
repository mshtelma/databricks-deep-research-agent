from __future__ import annotations

from typing import Any

from databricks_deep_research.agents.config import (
    AgentNodeConfig,
    PlanAndExecuteNodeConfig,
    ToolNodeConfig,
)
from databricks_deep_research.events.types import StreamEvent, ToolResultEvent
from databricks_deep_research.tools.protocol import tool_kind_to_source_kind
from databricks_deep_research.tools.resolver import ToolResolver
from databricks_deep_research.workflow.definition import NodeType, WorkflowDefinition, WorkflowNode
from databricks_deep_research.workflow.runtime.plan_execute_formatting import (
    extract_step_title,
    obs_to_text,
)
from databricks_deep_research.workflow.runtime.plan_execute_types import (
    AvailableSourceDescriptor,
    PlanCycleContext,
    ReplanFeedbackEntry,
)
from databricks_deep_research.workflow.state import WorkflowState

_TOOL_KIND_ENDPOINT_KEY: dict[str, str] = {
    "vector_search": "index_name",
    "genie_space": "space_id",
    "sql_warehouse": "endpoint_name",
}


def append_completed_step(cycle_ctx: PlanCycleContext, item: Any) -> None:
    title = extract_step_title(item)
    if title and title not in cycle_ctx.completed_steps:
        cycle_ctx.completed_steps.append(title)


def append_replan_feedback(cycle_ctx: PlanCycleContext, *, reason: str, cycle: int, message: str, step_title: str = "") -> None:
    cycle_ctx.feedback_history.append(ReplanFeedbackEntry(reason=reason, cycle=cycle, message=message.strip() or reason, step_title=step_title.strip()))


def extract_decision(eval_output: Any) -> str:
    if isinstance(eval_output, dict):
        return str(eval_output.get("decision", "continue"))
    if hasattr(eval_output, "decision"):
        return str(eval_output.decision)
    return "continue"


def extract_reasoning(eval_output: Any) -> str:
    if isinstance(eval_output, dict):
        return str(eval_output.get("reasoning", ""))
    if hasattr(eval_output, "reasoning"):
        return str(eval_output.reasoning)
    return ""




def extract_evidence_sufficiency(eval_output: Any) -> str | None:
    if isinstance(eval_output, dict):
        value = eval_output.get("evidence_sufficiency")
        return str(value) if value else None
    if hasattr(eval_output, "evidence_sufficiency"):
        value = eval_output.evidence_sufficiency
        return str(value) if value else None
    return None


def extract_failure_mode(eval_output: Any) -> str | None:
    if isinstance(eval_output, dict):
        value = eval_output.get("failure_mode")
        return str(value) if value else None
    if hasattr(eval_output, "failure_mode"):
        value = eval_output.failure_mode
        return str(value) if value else None
    return None

def normalize_evaluation_decision(decision: Any) -> str:
    normalized = str(decision or "continue").strip().lower()
    if normalized == "adjust":
        return "replan"
    return normalized if normalized in {"continue", "replan", "complete"} else "continue"


def item_requires_evidence(item: Any) -> bool:
    if isinstance(item, dict):
        needs_search = item.get("needs_search")
        if isinstance(needs_search, bool):
            return needs_search
        if str(item.get("step_type", "") or "").lower() == "analysis":
            return False
    return True


def summarize_item_health(item: Any, item_events: list[StreamEvent], *, sources_before: int, sources_after: int) -> dict[str, Any]:
    tool_results = [event for event in item_events if isinstance(event, ToolResultEvent)]
    tool_calls = sum(1 for event in item_events if getattr(event, "event_type", "") == "tool_call")
    successful_results = [event for event in tool_results if event.tool_success]
    accepted_sources = sum(int(getattr(event, "accepted_source_count", getattr(event, "source_count", 0))) for event in tool_results)
    raw_sources = sum(int(getattr(event, "raw_source_count", 0)) for event in tool_results)
    rejected_sources = sum(int(getattr(event, "rejected_source_count", 0)) for event in tool_results)
    source_delta = max(0, sources_after - sources_before)
    requires_evidence = item_requires_evidence(item)
    blocked = False
    reason = ""
    if requires_evidence:
        if tool_calls == 0:
            blocked = True
            reason = "no tool calls were executed"
        elif not successful_results:
            blocked = True
            reason = "all tool calls failed"
        elif accepted_sources == 0 and source_delta == 0:
            blocked = True
            reason = "no accepted evidence was admitted"
    return {
        "blocked": blocked,
        "reason": reason,
        "tool_calls": tool_calls,
        "tool_results": len(tool_results),
        "successful_tool_results": len(successful_results),
        "raw_sources": raw_sources,
        "accepted_sources": accepted_sources,
        "rejected_sources": rejected_sources,
        "source_pool_delta": source_delta,
        "requires_evidence": requires_evidence,
    }


def populate_synthesis_state(node_id: str, state: WorkflowState, pools: dict[str, Any], total_items_processed: int, replan_cycles: int, *, completion_mode: str = "normal", evidence_sufficiency: str | None = None, failure_mode: str | None = None) -> None:
    sid = f"{node_id}_synthesis"
    state.append(sid, "steps_executed", str(total_items_processed))
    state.append(sid, "plan_iterations", str(replan_cycles + 1))
    sources_pool = pools.get("sources")
    state.append(sid, "sources_count", str(sources_pool.count() if sources_pool else 0))
    state.append(sid, "completion_mode", completion_mode)
    if evidence_sufficiency is not None:
        state.append(sid, "evidence_sufficiency", evidence_sufficiency)
    if failure_mode is not None:
        state.append(sid, "failure_mode", failure_mode)
    obs_pool = pools.get("observations")
    if obs_pool and obs_pool.count() > 0:
        items = obs_pool.get_recent(min(obs_pool.count(), 30))
        state.append(sid, "all_observations", "\n".join(f"- {obs_to_text(item)[:300]}" for item in items))
    if sources_pool and sources_pool.count() > 0:
        items = sources_pool.get_recent(min(sources_pool.count(), 50))
        lines = [f"- [{item.get('title', 'Source')}]({item.get('url', '')})" if isinstance(item, dict) else f"- {str(item)[:200]}" for item in items]
        state.append(sid, "sources_list", "\n".join(lines))
    discovery_pool = pools.get("discovery_sources")
    if discovery_pool and discovery_pool.count() > 0:
        discovery_items = discovery_pool.get_recent(min(discovery_pool.count(), 20))
        should_use_fallback = sources_pool is None or sources_pool.count() == 0
        if not should_use_fallback and sources_pool is not None:
            from databricks_deep_research.agents.source_aware import sources_match_query
            should_use_fallback = not sources_match_query(state.query, sources_pool.get_recent(min(sources_pool.count(), 20)))
        if should_use_fallback:
            lines = [f"- [{item.get('title', 'Source')}]({item.get('url', '')})" if isinstance(item, dict) else f"- {str(item)[:200]}" for item in discovery_items]
            state.append(sid, "fallback_discovery_sources", "\n".join(lines))


def tool_ref_name(ref: str | dict[str, Any]) -> str:
    return ref if isinstance(ref, str) else str(ref.get("name", "") or "")


def collect_body_tool_refs(raw_node: dict[str, Any] | None) -> list[str | dict[str, Any]]:
    if not raw_node:
        return []
    node = WorkflowNode(**raw_node)
    refs: list[str | dict[str, Any]] = []
    if node.type == NodeType.agent:
        refs.extend(AgentNodeConfig(**node.config).tools)
    elif node.type == NodeType.tool:
        refs.append(ToolNodeConfig(**node.config).ref)
    elif node.type == NodeType.plan_and_execute:
        refs.extend(collect_body_tool_refs(PlanAndExecuteNodeConfig(**node.config).body))
    for child in node.children:
        refs.extend(collect_body_tool_refs(child.model_dump(mode="json")))
    return refs


def is_evidence_tool_kind(tool_kind: str, source_kind: str) -> bool:
    return tool_kind not in {"web_crawl"} and source_kind not in {"builtin", "web_crawl"}


async def build_available_source_catalog(definition: WorkflowDefinition, resolver: ToolResolver, body: dict[str, Any]) -> list[AvailableSourceDescriptor]:
    refs = collect_body_tool_refs(body)
    if not refs:
        return []
    decl_by_name = {tool.name: tool for tool in definition.tools}
    source_by_name = {source.name: source for source in definition.sources}
    seen: set[str] = set()
    catalog: list[AvailableSourceDescriptor] = []
    for ref in refs:
        name = tool_ref_name(ref)
        if not name or name in seen:
            continue
        seen.add(name)
        decl = decl_by_name.get(name)
        source = source_by_name.get(name)
        tool_kind = decl.kind if decl else name
        source_kind = source.kind if source else tool_kind_to_source_kind(tool_kind)
        if not is_evidence_tool_kind(tool_kind, source_kind):
            continue
        description = source.description if source and source.description else decl.description if decl else ""
        endpoint = str(source.endpoint) if source and getattr(source, "endpoint", None) else str(decl.config.get(_TOOL_KIND_ENDPOINT_KEY.get(decl.kind, ""), "") or "") if decl else ""
        if not decl and not source:
            try:
                tool = await resolver.resolve(ref)
            except ValueError:
                continue
            metadata = tool.definition.metadata or {}
            tool_kind = str(metadata.get("tool_kind") or tool.definition.name)
            source_kind = str(tool.definition.source_kind or "builtin")
            if not is_evidence_tool_kind(tool_kind, source_kind):
                continue
            description = tool.definition.description
            endpoint = str(metadata.get("index_name") or metadata.get("space_id") or metadata.get("endpoint_name") or "")
        catalog.append(AvailableSourceDescriptor(source_name=name, tool_kind=tool_kind, source_kind=source_kind, description=description, endpoint=endpoint))
    return catalog
