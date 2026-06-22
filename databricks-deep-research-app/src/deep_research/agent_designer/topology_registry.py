"""Phase 3 — TopologySpec registry.

Single source of truth for the two genuinely PER-topology behaviors:

* ``build`` — name -> builder function (consumed by the builder dispatch in
  ``workflow_builder.build_web_research_workflow``).
* ``structural_family`` — name -> the structural shape the behavioral probe
  compares against (consumed by ``probe._structural_family``).

It deliberately does NOT own selection precedence or blueprint lane-resolution.
``task_signature.select_topology`` (independence-first ordering) and
``blueprint._resolve_lane_descriptions`` are CROSS-topology policies: the lane
count derives from the signature, not the topology, and a per-spec ``selects()``
predicate cannot express selection precedence without re-introducing ordering
bugs. Keeping those centralized — and the registry narrow — is intentional, not
an oversight.

Adding a topology in a later phase is one registry entry plus its builder; the
enum-parity test (``set(TOPOLOGY_REGISTRY) == set(TOPOLOGIES) ==
get_args(TopologyKind)``) keeps the registry, the ``TopologyName``/``TopologyKind``
Literals, and the dispatch from drifting apart.

Hard cap on scope: a flat dict of frozen specs with three fields each. No plugin
discovery, no inheritance, no runtime registration.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from deep_research.agent_designer.task_signature import TopologyName

if TYPE_CHECKING:  # avoid an import cycle at module load (workflow_builder is heavy)
    from deep_research.agent_designer.designer_types import WorkflowDesignBrief


@dataclass(frozen=True)
class TopologyBuildContext:
    """Every input any topology builder might need.

    Unifies the divergent builder signatures behind one call: each builder
    adapter (see ``_build_registry``) picks the subset of fields it uses, so the
    dispatch can stay topology-agnostic.
    """

    intent: str
    name: str
    brief: WorkflowDesignBrief
    assets: list[dict[str, Any]] | None = None
    ambiguity_axes: list[str] | None = None
    # best_of_n coordination params (ignored by the other topologies' adapters).
    candidate_count: int | None = None
    judge_tier: str | None = None
    # iterative_refinement coordination params (ignored by the others' adapters).
    refine_participants: int | None = None
    refine_max_iterations: int | None = None
    proposer_families: list[str] | None = None
    # router coordination params (ignored by the others' adapters).
    router_cases: list[str] | None = None


@dataclass(frozen=True)
class TopologySpec:
    name: TopologyName
    structural_family: str
    build: Callable[[TopologyBuildContext], dict[str, Any]]


def _build_registry() -> dict[str, TopologySpec]:
    """Construct the registry. Builders are imported lazily so importing this
    module never triggers an import cycle through the heavy ``workflow_builder``.
    """
    from deep_research.agent_designer.workflow_builder import (
        _build_best_of_n_workflow,
        _build_iterative_refinement_workflow,
        _build_parallel_lanes_workflow,
        _build_plan_and_execute_workflow,
        _build_router_workflow,
        _build_single_agent_workflow,
    )

    return {
        "single_agent": TopologySpec(
            name="single_agent",
            structural_family="single_agent",
            build=lambda c: _build_single_agent_workflow(c.intent, c.name, c.brief),
        ),
        "parallel_lanes": TopologySpec(
            name="parallel_lanes",
            structural_family="parallel_lanes",
            build=lambda c: _build_parallel_lanes_workflow(
                c.intent,
                c.name,
                c.brief,
                assets=c.assets,
                ambiguity_axes=c.ambiguity_axes,
            ),
        ),
        "plan_and_execute": TopologySpec(
            name="plan_and_execute",
            structural_family="plan_and_execute",
            build=lambda c: _build_plan_and_execute_workflow(
                c.intent,
                c.name,
                c.brief,
                assets=c.assets,
                ambiguity_axes=c.ambiguity_axes,
            ),
        ),
        "best_of_n": TopologySpec(
            name="best_of_n",
            # best_of_n is a parallel fan-out, so it shares parallel_lanes'
            # structural family for the probe's topology-match check; its
            # specific shape is verified by the probe's best_of_n invariants.
            structural_family="parallel_lanes",
            build=lambda c: _build_best_of_n_workflow(
                c.intent,
                c.name,
                c.brief,
                assets=c.assets,
                ambiguity_axes=c.ambiguity_axes,
                candidate_count=c.candidate_count,
                judge_tier=c.judge_tier,
            ),
        ),
        "iterative_refinement": TopologySpec(
            name="iterative_refinement",
            # A coordinator → evidence-parallel → loop → finalizer sequence; the
            # probe's topology-match check keys off the evidence parallel, so it
            # shares the parallel_lanes structural family. Its loop-specific shape
            # is verified by the probe's iterative_refinement invariants.
            structural_family="parallel_lanes",
            build=lambda c: _build_iterative_refinement_workflow(
                c.intent,
                c.name,
                c.brief,
                assets=c.assets,
                ambiguity_axes=c.ambiguity_axes,
                participants=c.refine_participants,
                max_iterations=c.refine_max_iterations,
                proposer_families=c.proposer_families,
            ),
        ),
        "router": TopologySpec(
            name="router",
            # A classifier -> conditional(branches) sequence: its own structural
            # family (the probe detects a root-level conditional). Branch-specific
            # shape is verified by the probe's router invariants.
            structural_family="router",
            build=lambda c: _build_router_workflow(
                c.intent,
                c.name,
                c.brief,
                assets=c.assets,
                ambiguity_axes=c.ambiguity_axes,
                router_cases=c.router_cases,
            ),
        ),
    }


_REGISTRY: dict[str, TopologySpec] | None = None


def topology_registry() -> dict[str, TopologySpec]:
    """Return the (lazily built, cached) topology registry."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return _REGISTRY


def get_topology_spec(name: str) -> TopologySpec:
    """Return the spec for ``name`` or raise on an unknown topology.

    This is the single fail-closed point for the builder dispatch — there is no
    silent default to ``parallel_lanes``.
    """
    spec = topology_registry().get(name)
    if spec is None:
        raise ValueError(
            f"unknown topology {name!r}; expected one of {tuple(topology_registry())}"
        )
    return spec


def structural_family(name: str) -> str:
    """Structural family for the probe's topology-match comparison.

    Unknown names map to themselves (so an out-of-registry value simply fails the
    probe's equality check rather than raising inside the probe).
    """
    spec = topology_registry().get(name)
    return spec.structural_family if spec is not None else name
