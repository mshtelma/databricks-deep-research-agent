"""Plan v2.1 M9 — probe-gate enforcement.

Codex CRITICAL-4: ``behavioral_probe.passed`` was previously only read
advisorily by the critic and was NOT a blocking signal in the
gate_router (which only branched on ``gate_result.status``). Deleting
``detect_topology_mismatch`` without first wiring the probe into the
gate would leave a hole.

This test asserts the wiring: the designer_workflow.yaml ``gate_router``
conditional has a probe-fail condition that routes to a dedicated
copy_probe_to_feedback child, evaluated BEFORE the gate-pass condition
(fail-first ordering).
"""
from __future__ import annotations

from pathlib import Path

import yaml

_YAML_PATH = (
    Path(__file__).parents[3]
    / "src"
    / "deep_research"
    / "agent_designer"
    / "designer_workflow.yaml"
)


def _load_designer_yaml() -> dict[str, object]:
    """Load the raw YAML (not the framework's load_workflow) so we can
    inspect the conditional structure directly."""
    with _YAML_PATH.open() as f:
        return yaml.safe_load(f)


def _find_node_by_id(tree: object, node_id: str) -> dict[str, object] | None:
    """Depth-first search for a node with the given id."""
    if isinstance(tree, dict):
        if tree.get("id") == node_id:
            return tree
        for value in tree.values():
            found = _find_node_by_id(value, node_id)
            if found is not None:
                return found
    elif isinstance(tree, list):
        for item in tree:
            found = _find_node_by_id(item, node_id)
            if found is not None:
                return found
    return None


def test_gate_router_node_exists() -> None:
    raw = _load_designer_yaml()
    gate_router = _find_node_by_id(raw, "gate_router")
    assert gate_router is not None
    assert gate_router["type"] == "conditional"


def test_gate_router_has_three_children_for_two_conditions() -> None:
    """The framework requires len(children) == len(conditions) + 1.

    With two conditions (probe-fail-first + gate-pass), we need three
    children: copy_gate_to_feedback (default), critic (gate-pass),
    copy_probe_to_feedback (probe-fail).
    """
    raw = _load_designer_yaml()
    gate_router = _find_node_by_id(raw, "gate_router")
    assert gate_router is not None
    config = gate_router["config"]  # type: ignore[index]
    children = gate_router["children"]  # type: ignore[index]
    conditions = config["conditions"]  # type: ignore[index]
    assert len(children) == len(conditions) + 1, (
        f"Framework contract: conditional must have one more child than "
        f"conditions. Got {len(children)} children and {len(conditions)} "
        f"conditions."
    )


def test_gate_router_probe_fail_condition_first() -> None:
    """M9 fail-first ordering: probe_result.passed == false must be
    evaluated BEFORE gate_result.status == 'pass'. If gate-pass were
    evaluated first, a workflow with a passing gate but a failing
    probe would route to the critic instead of the FAIL branch."""
    raw = _load_designer_yaml()
    gate_router = _find_node_by_id(raw, "gate_router")
    assert gate_router is not None
    conditions = gate_router["config"]["conditions"]  # type: ignore[index]
    assert len(conditions) >= 2, "Need both probe-fail and gate-pass conditions"

    first_cond = conditions[0]["condition"]
    assert first_cond["key"] == "probe_result.passed", (
        "Plan v2.1 M9: probe-fail condition must be evaluated first "
        "(fail-first ordering)."
    )
    assert first_cond["operator"] == "eq"
    assert first_cond["value"] is False


def test_gate_router_probe_fail_routes_to_dedicated_child() -> None:
    """The probe-fail condition must route to copy_probe_to_feedback
    (child_index 2), not back to copy_gate_to_feedback (which would
    leak gate info as probe feedback)."""
    raw = _load_designer_yaml()
    gate_router = _find_node_by_id(raw, "gate_router")
    assert gate_router is not None
    conditions = gate_router["config"]["conditions"]  # type: ignore[index]
    probe_cond = conditions[0]
    assert probe_cond["child_index"] == 2

    # And child_index 2 must be the copy_probe_to_feedback node.
    children = gate_router["children"]  # type: ignore[index]
    child_2 = children[2]
    assert child_2["id"] == "copy_probe_to_feedback"
    assert child_2["type"] == "tool"


def test_gate_router_gate_pass_condition_unchanged() -> None:
    """The gate-pass condition (route to critic) must still exist and
    route to child_index 1 — preserving the previous behavior."""
    raw = _load_designer_yaml()
    gate_router = _find_node_by_id(raw, "gate_router")
    assert gate_router is not None
    conditions = gate_router["config"]["conditions"]  # type: ignore[index]
    gate_pass_cond = conditions[1]["condition"]
    assert gate_pass_cond["key"] == "gate_result.status"
    assert gate_pass_cond["operator"] == "eq"
    assert gate_pass_cond["value"] == "pass"
    assert conditions[1]["child_index"] == 1


def test_gate_router_default_branch_unchanged() -> None:
    """When no condition matches (gate failed AND probe passed), the
    default branch (0 = copy_gate_to_feedback) still fires. Preserves
    legacy behavior for gate-failure routing."""
    raw = _load_designer_yaml()
    gate_router = _find_node_by_id(raw, "gate_router")
    assert gate_router is not None
    assert gate_router["config"]["default_branch"] == 0  # type: ignore[index]


def test_copy_probe_to_feedback_uses_extract_critic_approved_shim() -> None:
    """The probe-fail child uses extract_critic_approved as a shim with
    input_mapping critic_verdict=probe_result, mirroring how
    copy_gate_to_feedback handles gate failures. This ensures probe
    failures reach the architect as critic_feedback on next iteration."""
    raw = _load_designer_yaml()
    node = _find_node_by_id(raw, "copy_probe_to_feedback")
    assert node is not None
    assert node["type"] == "tool"
    config = node["config"]  # type: ignore[index]
    assert config["ref"]["name"] == "extract_critic_approved"
    assert config["input_mapping"]["critic_verdict"] == "probe_result"
    assert config["output_key"] == "critic_approved"


def test_behavioral_probe_runs_before_gate_router() -> None:
    """Sanity: probe must execute before the gate_router can read its
    result. Verify the sibling ordering inside designer_body."""
    raw = _load_designer_yaml()
    designer_body = _find_node_by_id(raw, "designer_body")
    assert designer_body is not None
    children_ids = [c.get("id") for c in designer_body["children"]]  # type: ignore[index]
    probe_idx = children_ids.index("behavioral_probe")
    router_idx = children_ids.index("gate_router")
    assert probe_idx < router_idx, (
        "behavioral_probe must run before gate_router reads its result"
    )
