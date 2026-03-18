"""Coordinator builtin — query classification and routing.

Classifies incoming queries by complexity, detects simple queries that
can be answered directly, and recommends research depth.  Emits
``CoordinatorClassifiedEvent`` with full classification metadata.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from databricks_deep_research.agents.builtins.registry import register_builtin
from databricks_deep_research.agents.config import AgentNodeConfig
from databricks_deep_research.agents.output_models import CoordinatorOutput
from databricks_deep_research.events.types import (
    CoordinatorClassifiedEvent,
    StreamEvent,
)
from databricks_deep_research.workflow.state import WorkflowState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Post-process hook
# ---------------------------------------------------------------------------


def _post_process(
    node_id: str,
    output: Any,
    _config: AgentNodeConfig,
    _state: WorkflowState,
) -> list[StreamEvent]:
    """Emit CoordinatorClassifiedEvent from coordinator output."""
    events: list[StreamEvent] = []

    # Extract fields — works whether output is CoordinatorOutput, dict, or raw
    if isinstance(output, CoordinatorOutput):
        complexity = output.complexity
        recommended_depth = output.recommended_depth
        is_simple = output.is_simple
        direct_response = output.direct_response
        follow_up_type = output.follow_up_type
        reasoning = ""
    elif isinstance(output, dict):
        complexity = output.get("complexity", "moderate")
        recommended_depth = output.get("recommended_depth", "standard")
        is_simple = output.get("is_simple", False)
        direct_response = output.get("direct_response")
        follow_up_type = output.get("follow_up_type")
        reasoning = output.get("reasoning", "")
    else:
        logger.warning("COORDINATOR_UNEXPECTED_OUTPUT type=%s", type(output).__name__)
        return events

    events.append(
        CoordinatorClassifiedEvent(
            node_id=node_id,
            timestamp=datetime.now(tz=UTC).isoformat(),
            complexity=complexity,
            recommended_depth=recommended_depth,
            is_simple=is_simple,
            direct_response=direct_response,
            follow_up_type=follow_up_type,
            reasoning=reasoning,
        )
    )

    logger.info(
        "COORDINATOR_CLASSIFIED complexity=%s is_simple=%s depth=%s",
        complexity,
        is_simple,
        recommended_depth,
    )
    logger.debug(
        "COORDINATOR_CLASSIFIED_DETAIL follow_up_type=%s direct_response=%s reasoning=%s",
        follow_up_type,
        str(direct_response)[:200] if direct_response is not None else None,
        reasoning[:200],
    )

    return events


# ---------------------------------------------------------------------------
# Config enrichment
# ---------------------------------------------------------------------------


def _enrich_config(
    config: AgentNodeConfig,
    _state: WorkflowState,
    _runtime_context: dict[str, Any] | None = None,
) -> AgentNodeConfig:
    """Fill in coordinator defaults if not specified in YAML."""
    updates: dict[str, Any] = {}

    if not config.system_prompt:
        from databricks_deep_research.agents.prompts.coordinator import (
            COORDINATOR_SYSTEM_PROMPT,
        )
        updates["system_prompt"] = COORDINATOR_SYSTEM_PROMPT

    if not config.user_prompt_template:
        from databricks_deep_research.agents.prompts.coordinator import (
            COORDINATOR_USER_PROMPT,
        )
        updates["user_prompt_template"] = COORDINATOR_USER_PROMPT

    if config.output_model is None:
        updates["output_model"] = CoordinatorOutput

    if updates:
        return config.model_copy(update=updates)
    return config


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_builtin(
    "coordinator",
    post_process=_post_process,
    enrich_config=_enrich_config,
    output_model=CoordinatorOutput,
)

__all__: list[str] = []
