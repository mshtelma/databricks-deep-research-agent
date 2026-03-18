"""Background investigator builtin — quick context gathering before planning.

Performs initial query decomposition and data landscape assessment.
Emits ``BackgroundCompletedEvent`` with discovery results.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from databricks_deep_research.agents.builtins.registry import register_builtin
from databricks_deep_research.agents.config import AgentNodeConfig
from databricks_deep_research.agents.output_models import BackgroundOutput
from databricks_deep_research.events.types import (
    BackgroundCompletedEvent,
    StreamEvent,
)
from databricks_deep_research.workflow.state import WorkflowState

logger = logging.getLogger(__name__)


def _post_process(
    node_id: str,
    output: Any,
    _config: AgentNodeConfig,
    _state: WorkflowState,
) -> list[StreamEvent]:
    """Emit BackgroundCompletedEvent from background output."""
    if isinstance(output, BackgroundOutput):
        landscape = output.data_landscape
        summary = output.summary
        decomposition = output.query_decomposition
        discovered_sources = output.discovered_sources
    elif isinstance(output, dict):
        landscape = output.get("data_landscape", {})
        summary = output.get("summary", "")
        decomposition = output.get("query_decomposition", [])
        discovered_sources = output.get("discovered_sources", [])
    else:
        landscape = {}
        summary = str(output)[:200] if output else ""
        decomposition = []
        discovered_sources = []

    if _state.runtime_store is not None:
        _state.runtime_store.set_background(
            summary=summary,
            data_landscape=landscape if isinstance(landscape, dict) else {},
            query_decomposition=decomposition if isinstance(decomposition, list) else [],
            discovered_sources=discovered_sources if isinstance(discovered_sources, list) else [],
        )
    else:
        _state.append(node_id, "background_summary", summary)
        _state.append(node_id, "data_landscape", landscape if isinstance(landscape, dict) else {})
        _state.append(node_id, "query_decomposition", decomposition)
        _state.append(node_id, "discovered_sources", discovered_sources if isinstance(discovered_sources, list) else [])

    return [
        BackgroundCompletedEvent(
            node_id=node_id,
            timestamp=datetime.now(tz=UTC).isoformat(),
            sources_discovered=len(landscape.get("sources", [])) if isinstance(landscape, dict) else 0,
            data_landscape_summary=summary,
            data_landscape=landscape if isinstance(landscape, dict) else {},
            query_decomposition=decomposition,
        )
    ]


def _enrich_config(
    config: AgentNodeConfig,
    _state: WorkflowState,
    _runtime_context: dict[str, Any] | None = None,
) -> AgentNodeConfig:
    """Fill in background defaults if not specified."""
    updates: dict[str, Any] = {}

    if not config.system_prompt:
        from databricks_deep_research.agents.prompts.background import (
            BACKGROUND_SYSTEM_PROMPT,
        )
        updates["system_prompt"] = BACKGROUND_SYSTEM_PROMPT

    if not config.user_prompt_template:
        from databricks_deep_research.agents.prompts.background import (
            BACKGROUND_USER_PROMPT,
        )
        updates["user_prompt_template"] = BACKGROUND_USER_PROMPT

    if config.output_model is None:
        updates["output_model"] = BackgroundOutput

    if config.output_format == "text":
        updates["output_format"] = "json"

    if updates:
        return config.model_copy(update=updates)
    return config


register_builtin(
    "background",
    post_process=_post_process,
    enrich_config=_enrich_config,
    output_model=BackgroundOutput,
)

__all__: list[str] = []
