"""FW-2 — router_classifier builtin subtype.

The subtype's ``enrich_config`` synthesizes a bounded ``RouterDecision`` model
(``route: Literal[*cases]`` + ``reason: str``) FROM the node's declared
``output_schema`` and sets it as ``output_model`` — so the classifier is FORCED to
emit one of the declared route cases at runtime (the AST ``output_schema`` JSON
alone never reaches the LLM call). This is the framework half of the router
topology's runtime-correctness contract.
"""

from __future__ import annotations

import pytest

import databricks_deep_research.agents.builtins  # noqa: F401  (triggers registration)
from databricks_deep_research.agents.builtins import get_builtin
from databricks_deep_research.agents.builtins.router_classifier import _enrich_config
from databricks_deep_research.agents.config import AgentNodeConfig

_ROUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "route": {"enum": ["pricing", "performance", "migration"]},
        "reason": {"type": "string"},
    },
    "required": ["route"],
}


def test_router_classifier_is_registered() -> None:
    builtin = get_builtin("router_classifier")
    assert builtin is not None
    assert builtin.enrich_config is not None


def test_enrich_synthesizes_forced_route_model() -> None:
    cfg = AgentNodeConfig(
        subtype="router_classifier", output_key="routing", output_schema=_ROUTE_SCHEMA
    )
    enriched = _enrich_config(cfg, None, {})
    assert enriched.output_model is not None
    model = enriched.output_model
    # the synthesized model accepts a declared case and rejects anything else
    assert model(route="performance").route == "performance"
    with pytest.raises(Exception):
        model(route="not-a-declared-case")


def test_enrich_is_noop_without_route_enum() -> None:
    # no route enum -> no output_model (the workflow then fails the router probe
    # rather than silently emitting untyped text)
    assert _enrich_config(
        AgentNodeConfig(subtype="router_classifier", output_key="routing"),
        None,
        {},
    ).output_model is None
    # a route without a usable enum is also a no-op
    assert _enrich_config(
        AgentNodeConfig(
            subtype="router_classifier",
            output_key="routing",
            output_schema={"type": "object", "properties": {"route": {"type": "string"}}},
        ),
        None,
        {},
    ).output_model is None


def test_enrich_preserves_explicit_output_model() -> None:
    from pydantic import BaseModel

    class Explicit(BaseModel):
        x: int

    cfg = AgentNodeConfig(
        subtype="router_classifier",
        output_key="routing",
        output_schema=_ROUTE_SCHEMA,
        output_model=Explicit,
    )
    assert _enrich_config(cfg, None, {}).output_model is Explicit
