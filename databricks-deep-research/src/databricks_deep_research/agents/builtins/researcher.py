"""Researcher builtin — ReAct-mode research step execution.

Uses LLM-controlled tool calls (web search, web crawl, pool search) to
investigate a research step.  Source tracking, observation synthesis,
and token budget awareness are handled by the generic ReAct loop in
``react_loop.py``.

The researcher builtin enriches the config with default prompts and
max_tool_calls, and emits no additional domain events beyond what the
ReAct loop already produces (``ToolCallEvent``, ``ToolResultEvent``).
"""

from __future__ import annotations

import logging
from typing import Any

from databricks_deep_research.agents.builtins.registry import register_builtin
from databricks_deep_research.agents.config import AgentNodeConfig
from databricks_deep_research.agents.output_models import ResearcherOutput
from databricks_deep_research.events.types import StreamEvent
from databricks_deep_research.workflow.state import WorkflowState

logger = logging.getLogger(__name__)

# Default max tool calls for researcher (can be overridden in YAML)
DEFAULT_MAX_TOOL_CALLS = 15


def _post_process(
    _node_id: str,
    _output: Any,
    _config: AgentNodeConfig,
    _state: WorkflowState,
) -> list[StreamEvent]:
    """Researcher emits no additional domain events (ReAct loop handles tool events)."""
    return []


def _enrich_config(
    config: AgentNodeConfig,
    _state: WorkflowState,
    _runtime_context: dict[str, Any] | None = None,
) -> AgentNodeConfig:
    """Fill in researcher defaults if not specified."""
    updates: dict[str, Any] = {}

    if not config.system_prompt:
        from databricks_deep_research.agents.prompts.researcher import (
            RESEARCHER_SYSTEM_PROMPT,
        )
        updates["system_prompt"] = RESEARCHER_SYSTEM_PROMPT

    if not config.user_prompt_template:
        from databricks_deep_research.agents.prompts.researcher import (
            RESEARCHER_USER_PROMPT,
        )
        updates["user_prompt_template"] = RESEARCHER_USER_PROMPT

    # Researcher should have tools enabled by default
    if config.max_tool_calls is None:
        updates["max_tool_calls"] = DEFAULT_MAX_TOOL_CALLS

    if config.output_model is None:
        updates["output_model"] = ResearcherOutput

    if config.output_format in ("text", "markdown"):
        updates["output_format"] = "json"

    if updates:
        return config.model_copy(update=updates)
    return config


register_builtin(
    "researcher",
    post_process=_post_process,
    enrich_config=_enrich_config,
)

__all__: list[str] = []
