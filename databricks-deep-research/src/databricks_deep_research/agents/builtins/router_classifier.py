"""Router-classifier builtin subtype — typed discriminator for conditional routers.

A conditional/router node branches on a STATE field via a ``StateCondition``
(e.g. ``routing.route == "X"``). The static condition-contract validator requires
that field to be a typed enum in the producer's ``output_schema``; at runtime the
field must actually be emitted or the conditional executor hard-raises. But
structured LLM output is driven by ``config.output_model`` (a Pydantic class set
by the subtype) — NOT by the AST's ``output_schema`` JSON. This subtype closes the
gap: its ``enrich_config`` synthesizes a bounded ``RouterDecision`` model
(``route: Literal[*cases]`` + ``reason: str``) FROM the node's declared
``output_schema`` and sets it as ``output_model``, so the classifier is FORCED to
emit exactly one of the declared route cases. Bounded to enum+scalar by design
(NOT a general JSON-schema→model compiler).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import create_model

from databricks_deep_research.agents.builtins.registry import register_builtin
from databricks_deep_research.agents.config import AgentNodeConfig
from databricks_deep_research.workflow.state import WorkflowState


def _route_cases(config: AgentNodeConfig) -> tuple[str, ...] | None:
    """Extract the declared ``route`` enum from output_schema, or None."""
    schema = config.output_schema or {}
    if not isinstance(schema, dict):
        return None
    props = schema.get("properties")
    if not isinstance(props, dict):
        return None
    route_spec = props.get("route")
    if not isinstance(route_spec, dict):
        return None
    enum = route_spec.get("enum")
    if not isinstance(enum, list) or len(enum) < 2:
        return None
    return tuple(str(case) for case in enum)


def _enrich_config(
    config: AgentNodeConfig,
    _state: WorkflowState,
    _runtime_context: dict[str, Any] | None,
) -> AgentNodeConfig:
    """Synthesize a forced ``RouterDecision`` output_model from output_schema.

    No-op when an output_model is already set or the schema declares no usable
    ``route`` enum (the workflow then fails the router probe/validation rather
    than silently emitting untyped text at runtime).
    """
    if config.output_model is not None:
        return config
    cases = _route_cases(config)
    if cases is None:
        return config
    model = create_model(
        "RouterDecision",
        route=(Literal[cases], ...),
        reason=(str, ""),
    )
    return config.model_copy(update={"output_model": model})


register_builtin(
    "router_classifier",
    enrich_config=_enrich_config,
    default_system_prompt="",
    default_user_prompt="{query}",
    output_model=None,
)

__all__: list[str] = []
