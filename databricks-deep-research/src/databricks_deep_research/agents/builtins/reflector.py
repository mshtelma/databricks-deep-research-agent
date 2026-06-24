"""Reflector builtin — coverage analysis and step-by-step decisions.

Evaluates research progress after each step and decides whether to
CONTINUE, ADJUST, or COMPLETE.  Uses pool injection to review accumulated
observations.  Emits ``ReflectionDecisionEvent``.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from databricks_deep_research.agents.builtins.registry import register_builtin
from databricks_deep_research.agents.config import AgentNodeConfig
from databricks_deep_research.agents.output_models import ReflectionOutput
from databricks_deep_research.events.types import (
    ReflectionDecisionEvent,
    StreamEvent,
)
from databricks_deep_research.workflow.state import WorkflowState

logger = logging.getLogger(__name__)


def _coerce_optional_str(value: Any) -> str | None:
    """Coerce an LLM-produced field to ``str | None`` for the event schema.

    ``ReflectionDecisionEvent`` types ``decision``/``reasoning``/
    ``evidence_sufficiency``/``failure_mode`` as strings, but a stochastic LLM
    occasionally returns a structured object (e.g. ``{"completeness": 8, ...}``)
    under one of those keys. Strings pass through untouched; ``None`` stays
    ``None``; anything else is compactly stringified (capped) so a malformed
    value degrades gracefully instead of raising a ``ValidationError`` that would
    fail the whole already-completed research run. Mirrors
    ``extract_evidence_sufficiency`` in the plan-and-execute runtime, which
    already stringifies non-string output.
    """
    if value is None or isinstance(value, str):
        return value
    text = str(value).strip()
    return text[:500] or None


def _post_process(
    node_id: str,
    output: Any,
    _config: AgentNodeConfig,
    _state: WorkflowState,
) -> list[StreamEvent]:
    """Emit ReflectionDecisionEvent from reflector output."""
    if isinstance(output, ReflectionOutput):
        decision = output.decision
        reasoning = output.reasoning
    elif isinstance(output, dict):
        decision = output.get("decision", "continue")
        reasoning = output.get("reasoning", "")
    else:
        return []

    logger.info(
        "REFLECTOR_DECISION decision=%s evidence_sufficiency=%s failure_mode=%s",
        decision,
        getattr(output, "evidence_sufficiency", None) if not isinstance(output, dict) else output.get("evidence_sufficiency"),
        getattr(output, "failure_mode", None) if not isinstance(output, dict) else output.get("failure_mode"),
    )

    evidence_sufficiency = _coerce_optional_str(
        getattr(output, "evidence_sufficiency", None)
        if not isinstance(output, dict)
        else output.get("evidence_sufficiency")
    )
    failure_mode = _coerce_optional_str(
        getattr(output, "failure_mode", None)
        if not isinstance(output, dict)
        else output.get("failure_mode")
    )

    return [
        ReflectionDecisionEvent(
            node_id=node_id,
            timestamp=datetime.now(tz=UTC).isoformat(),
            decision=_coerce_optional_str(decision) or "continue",
            reasoning=_coerce_optional_str(reasoning) or "",
            evidence_sufficiency=evidence_sufficiency,
            failure_mode=failure_mode,
        )
    ]




def _enrich_config(
    config: AgentNodeConfig,
    _state: WorkflowState,
    _runtime_context: dict[str, Any] | None = None,
) -> AgentNodeConfig:
    """Fill in reflector defaults if not specified."""
    updates: dict[str, Any] = {}

    if not config.system_prompt:
        from databricks_deep_research.agents.prompts.reflector import (
            REFLECTOR_SYSTEM_PROMPT,
        )
        updates["system_prompt"] = REFLECTOR_SYSTEM_PROMPT

    if not config.user_prompt_template:
        from databricks_deep_research.agents.prompts.reflector import (
            REFLECTOR_USER_PROMPT,
        )
        updates["user_prompt_template"] = REFLECTOR_USER_PROMPT

    if config.output_model is None:
        updates["output_model"] = ReflectionOutput

    if updates:
        return config.model_copy(update=updates)
    return config


register_builtin(
    "reflector",
    post_process=_post_process,
    enrich_config=_enrich_config,
    output_model=ReflectionOutput,
)

__all__: list[str] = []
