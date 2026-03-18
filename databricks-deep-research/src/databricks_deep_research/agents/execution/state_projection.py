from __future__ import annotations

from typing import Any

from databricks_deep_research.agents.config import AgentNodeConfig
from databricks_deep_research.agents.execution.output_normalizer import NormalizedResearchOutput
from databricks_deep_research.workflow.state import WorkflowState


def project_research_state(
    node_id: str,
    config: AgentNodeConfig,
    state: WorkflowState,
    normalized_research_output: NormalizedResearchOutput,
) -> tuple[str, dict[str, Any]]:
    structured_findings = {
        config.output_key: normalized_research_output.findings_text,
        "findings": normalized_research_output.findings_text,
        "observation": normalized_research_output.observation_text,
        "search_queries": normalized_research_output.search_queries,
        "key_points": normalized_research_output.key_points,
        "sources": normalized_research_output.sources,
        "research_status": normalized_research_output.research_status,
        "blocking_reason": normalized_research_output.blocking_reason,
        "repair_mode": normalized_research_output.repair_mode,
    }
    state.append(node_id, f"{config.output_key}_structured", structured_findings)
    state.append(node_id, "research_status", normalized_research_output.research_status)
    if normalized_research_output.blocking_reason is not None:
        state.append(node_id, "blocking_reason", normalized_research_output.blocking_reason)
    if normalized_research_output.repair_mode is not None:
        state.append(node_id, "repair_mode", normalized_research_output.repair_mode)
    if normalized_research_output.search_queries:
        state.append(node_id, "search_queries", normalized_research_output.search_queries)
    if normalized_research_output.key_points:
        state.append(node_id, "key_points", normalized_research_output.key_points)
    return normalized_research_output.state_text, structured_findings
