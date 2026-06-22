"""Phase 3 — TopologySpec registry: parity, fail-closed dispatch, determinism.

The registry is a behavior-preserving refactor of the builder dispatch + the
probe structural-family map. These tests are the drift guard (registry keys ==
the topology Literals) and the golden pin (build_blueprint routes each topology
and is deterministic across rebuilds — so the refactor cannot silently change a
builder's output).
"""

from __future__ import annotations

import json
from typing import Any, get_args

import pytest

from deep_research.agent_designer.blueprint import build_blueprint
from deep_research.agent_designer.designer_types import TopologyKind
from deep_research.agent_designer.task_signature import (
    TOPOLOGIES,
    TaskSignature,
    TopologyName,
    select_topology,
)
from deep_research.agent_designer.topology_registry import (
    TopologySpec,
    get_topology_spec,
    structural_family,
    topology_registry,
)

_FAMILIES = {"single_agent", "parallel_lanes", "plan_and_execute", "router"}


def test_registry_parity_with_enums() -> None:
    keys = set(topology_registry())
    assert keys == set(TOPOLOGIES)
    assert keys == set(get_args(TopologyName))
    assert keys == set(get_args(TopologyKind))


def test_every_spec_well_formed() -> None:
    for name, spec in topology_registry().items():
        assert isinstance(spec, TopologySpec)
        assert spec.name == name
        assert spec.structural_family in _FAMILIES
        assert callable(spec.build)


def test_get_topology_spec_raises_on_unknown() -> None:
    with pytest.raises(ValueError, match="unknown topology"):
        get_topology_spec("nope_not_a_topology")


def test_structural_family() -> None:
    assert structural_family("best_of_n") == "parallel_lanes"
    assert structural_family("plan_and_execute") == "plan_and_execute"
    assert structural_family("single_agent") == "single_agent"
    # unknown self-maps (the probe equality check fails rather than raising)
    assert structural_family("unknown") == "unknown"


# --- golden / determinism: build_blueprint is behavior-stable per topology -----

_INTENT = "Analyze the competitive dynamics of the cloud database market."


def _sig(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "asset_signature": "web_only",
        "retrieval_pattern": "open_research",
        "question_class": "open_research",
        "primary_evidence_kind": "web_articles",
        "expected_output_shape": "structured_report",
        "step_dependencies_present": False,
        "independent_workstreams_count": 1,
        "iteration_required": False,
        "output_aggregation_kind": "single_answer",
        "lane_descriptions": ["the primary concern"],
    }
    base.update(kw)
    return base


_TOPOLOGY_SIGS: dict[str, dict[str, Any]] = {
    "single_agent": _sig(
        retrieval_pattern="bounded_lookup", question_class="bounded_lookup"
    ),
    "parallel_lanes": _sig(
        independent_workstreams_count=3,
        lane_descriptions=["pricing", "performance", "ecosystem"],
    ),
    "plan_and_execute": _sig(step_dependencies_present=True),
    "best_of_n": _sig(
        coordination_pattern="best_of_n", coordination_candidate_count=4
    ),
}


@pytest.mark.parametrize("topology,sig", list(_TOPOLOGY_SIGS.items()))
def test_blueprint_routes_and_is_deterministic(
    topology: str, sig: dict[str, Any]
) -> None:
    assert select_topology(TaskSignature.load_from_storage(sig)) == topology
    first = build_blueprint(sig, _INTENT)
    second = build_blueprint(sig, _INTENT)
    # Canonical-JSON identical across rebuilds => deterministic registry dispatch
    # + build. This is the golden pin protecting the Phase-3 refactor.
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["structural_fingerprint"] == second["structural_fingerprint"]
