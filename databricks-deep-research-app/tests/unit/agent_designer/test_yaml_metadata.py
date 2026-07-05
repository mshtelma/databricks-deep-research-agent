"""Whitelisted designer-metadata carriage across the YAML import boundary.

Exercises ``yaml_metadata.carry_designer_metadata`` against the healed loader
dump exactly the way ``parse_and_validate_yaml`` invokes it: ``source`` is the
exported document (a ``build_blueprint`` AST), ``definition`` is
``load_workflow_from_dict(source).model_dump()``. Covers the plan's eight test
groups: clean carry, per-key invalid shapes, signature consistency mismatch,
lane-key pruning, placeholder stale-id pruning, the loader-healing fingerprint
divergence, exporter↔validator parity for the contract summary, and the
absence rule (never synthesize).
"""

from __future__ import annotations

from typing import Any

import pytest
from databricks_deep_research import load_workflow_from_dict

from deep_research.agent_designer.blueprint import (
    build_blueprint,
    compute_lane_key,
    compute_structural_fingerprint,
)
from deep_research.agent_designer.designer_types import (
    PromptObligationContract,
    ResolvedToolContract,
    ResourceContract,
)
from deep_research.agent_designer.tool_contract import (
    sanitized_resolved_tool_contract_summary,
)
from deep_research.agent_designer.yaml_metadata import (
    DESIGNER_METADATA_KEYS,
    ImportMetadataWarning,
    ResolvedToolContractSummaryV1,
    carry_designer_metadata,
)
from deep_research.surface import scaffold_surface_from_workflow

_METADATA_KEYS = tuple(spec.key for spec in DESIGNER_METADATA_KEYS)


def _parallel_sig() -> dict[str, Any]:
    return {
        "asset_signature": "web_only",
        "retrieval_pattern": "independent_lanes",
        "question_class": "open_research",
        "primary_evidence_kind": "web_articles",
        "expected_output_shape": "structured_report",
        "independent_workstreams_count": 3,
        "step_dependencies_present": False,
        "iteration_required": False,
        "output_aggregation_kind": "cross_concern_synthesis",
        "lane_descriptions": ["alpha", "beta", "gamma"],
    }


def _single_agent_sig() -> dict[str, Any]:
    return {
        "asset_signature": "web_only",
        "retrieval_pattern": "bounded_lookup",
        "question_class": "bounded_lookup",
        "primary_evidence_kind": "web_articles",
        "expected_output_shape": "paragraph",
        "independent_workstreams_count": 1,
        "step_dependencies_present": False,
        "iteration_required": False,
        "output_aggregation_kind": "single_answer",
        "lane_descriptions": ["the one concern"],
    }


def _stamped_source() -> dict[str, Any]:
    """A build_blueprint AST — the exact document serialize_to_yaml would emit.

    ``surface`` is not stamped by the blueprint (its writers are the surface
    authoring tools), so the fixture attaches the deterministic scaffold to
    cover every ``DESIGNER_METADATA_KEYS`` entry the way an exported
    UI-carrying agent would.
    """
    source = build_blueprint(_parallel_sig(), "compare three separate things", [])
    source["surface"] = scaffold_surface_from_workflow(source)
    return source


def _healed_dump(source: dict[str, Any]) -> dict[str, Any]:
    """The framework projection parse_and_validate_yaml rebuilds from."""
    dumped = load_workflow_from_dict(source).model_dump()
    assert isinstance(dumped, dict)
    return dumped


def _carry(source: dict[str, Any]) -> tuple[dict[str, Any], list[ImportMetadataWarning]]:
    definition = _healed_dump(source)
    warnings = carry_designer_metadata(source, definition)
    return definition, warnings


# ---------------------------------------------------------------------------
# 1. Clean carry
# ---------------------------------------------------------------------------


def test_clean_carry_preserves_all_keys_without_warnings() -> None:
    source = _stamped_source()
    for key in _METADATA_KEYS:
        assert key in source, f"fixture must stamp {key}"
    definition, warnings = _carry(source)
    assert warnings == []
    assert definition["designer_signature"] == source["designer_signature"]
    assert definition["lane_keys"] == source["lane_keys"]
    assert definition["evidence_policy"] == source["evidence_policy"]
    assert definition["required_prompt_terms"] == source["required_prompt_terms"]
    assert (
        definition["resolved_tool_contract_summary"]
        == source["resolved_tool_contract_summary"]
    )
    assert (
        definition["placeholder_pending_nodes"] == source["placeholder_pending_nodes"]
    )
    # Surface carries verbatim (byte-lossless round trip).
    assert definition["surface"] == source["surface"]
    # Structure is unchanged by import for a well-formed blueprint AST, so the
    # recomputed fingerprint equals the stamped one (clean round trip).
    assert definition["structural_fingerprint"] == source["structural_fingerprint"]
    assert definition["structural_fingerprint"] == compute_structural_fingerprint(
        definition
    )


# ---------------------------------------------------------------------------
# 2. Per-key invalid shapes -> dropped (or recomputed) + exactly one warning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "bad_value", "expected_action"),
    [
        ("designer_signature", "not-a-dict", "dropped"),
        ("designer_signature", {"asset_signature": "nonsense"}, "dropped"),
        ("lane_keys", ["not", "a", "dict"], "dropped"),
        ("evidence_policy", "carrier_pigeon", "dropped"),
        ("required_prompt_terms", "not-a-list", "dropped"),
        ("resolved_tool_contract_summary", {"schema": "wrong.v9"}, "dropped"),
        ("placeholder_pending_nodes", "not-a-list", "dropped"),
        ("surface", "not-a-dict", "dropped"),
        (
            "surface",
            {
                "version": 1,
                "components": [
                    {
                        "id": "root",
                        "component": "NoSuchComponent",
                        "props": {},
                        "children": [],
                    }
                ],
                "data_model": {},
                "bindings": [],
            },
            "dropped",
        ),
        ("structural_fingerprint", 12345, "recomputed"),
    ],
)
def test_invalid_shape_per_key(key: str, bad_value: Any, expected_action: str) -> None:
    source = _stamped_source()
    source[key] = bad_value
    definition, warnings = _carry(source)
    mine = [w for w in warnings if w.key == key]
    assert len(mine) == 1
    assert mine[0].code == "invalid_shape"
    assert mine[0].action == expected_action
    if expected_action == "dropped":
        assert key not in definition
    else:  # structural_fingerprint: derived state is recomputed, never dropped
        assert definition[key] == compute_structural_fingerprint(definition)
    # No collateral damage to the other keys.
    for other in _METADATA_KEYS:
        if other != key:
            assert other in definition


# ---------------------------------------------------------------------------
# 3. designer_signature consistency cross-check
# ---------------------------------------------------------------------------


def test_signature_topology_mismatch_drops_with_warning() -> None:
    # A single_agent document claiming a 3-lane parallel signature.
    source = build_blueprint(_single_agent_sig(), "one bounded question", [])
    source["designer_signature"] = _parallel_sig()
    definition, warnings = _carry(source)
    assert "designer_signature" not in definition
    mine = [w for w in warnings if w.key == "designer_signature"]
    assert len(mine) == 1
    assert mine[0].code == "consistency_mismatch"
    assert mine[0].action == "dropped"


def test_signature_check_skipped_when_topology_unknown() -> None:
    # A shape the topology walker cannot classify: carry proceeds (probe parity).
    definition: dict[str, Any] = {
        "root": {
            "id": "main",
            "type": "sequence",
            "label": "main",
            "config": {},
            "children": [
                {
                    "id": "t1",
                    "type": "tool",
                    "label": "tool",
                    "config": {"ref": "noop"},
                    "children": [],
                }
            ],
        }
    }
    warnings = carry_designer_metadata(
        {"designer_signature": _parallel_sig()}, definition
    )
    assert warnings == []
    assert definition["designer_signature"] == _parallel_sig()


# ---------------------------------------------------------------------------
# 4. lane_keys entry pruning
# ---------------------------------------------------------------------------


def test_lane_keys_mismatched_entries_pruned() -> None:
    source = _stamped_source()
    good_key = compute_lane_key("alpha")
    source["lane_keys"] = {good_key: "alpha", "forged_key_123": "beta"}
    definition, warnings = _carry(source)
    assert definition["lane_keys"] == {good_key: "alpha"}
    mine = [w for w in warnings if w.key == "lane_keys"]
    assert len(mine) == 1
    assert mine[0].code == "consistency_mismatch"
    assert mine[0].action == "pruned"
    assert mine[0].detail == ["forged_key_123"]


def test_lane_keys_all_invalid_drops_key() -> None:
    source = _stamped_source()
    source["lane_keys"] = {"forged_a": "alpha", "forged_b": "beta"}
    definition, warnings = _carry(source)
    assert "lane_keys" not in definition
    mine = [w for w in warnings if w.key == "lane_keys"]
    assert len(mine) == 1
    assert mine[0].action == "dropped"
    assert sorted(mine[0].detail) == ["forged_a", "forged_b"]


# ---------------------------------------------------------------------------
# 5. placeholder_pending_nodes stale-id pruning
# ---------------------------------------------------------------------------


def test_placeholder_stale_ids_pruned() -> None:
    source = _stamped_source()
    pending = list(source["placeholder_pending_nodes"])
    assert pending, "parallel blueprint stamps a non-empty placeholder list"
    source["placeholder_pending_nodes"] = [pending[0], "ghost-node-id"]
    definition, warnings = _carry(source)
    assert definition["placeholder_pending_nodes"] == [pending[0]]
    mine = [w for w in warnings if w.key == "placeholder_pending_nodes"]
    assert len(mine) == 1
    assert mine[0].code == "stale_entries_pruned"
    assert mine[0].detail == ["ghost-node-id"]


def test_placeholder_all_stale_drops_key() -> None:
    source = _stamped_source()
    source["placeholder_pending_nodes"] = ["ghost-1", "ghost-2"]
    definition, warnings = _carry(source)
    assert "placeholder_pending_nodes" not in definition
    mine = [w for w in warnings if w.key == "placeholder_pending_nodes"]
    assert len(mine) == 1
    assert mine[0].action == "dropped"


# ---------------------------------------------------------------------------
# 6. Structural drift -> fingerprint recomputed + divergence warning
# ---------------------------------------------------------------------------


def test_structural_drift_recomputes_fingerprint() -> None:
    source = _stamped_source()
    # Simulate a hand-edited export: rename a node id AFTER the fingerprint was
    # stamped. Node ids are part of the fingerprint's canonical projection, so
    # the carried value describes a structure that no longer exists — the
    # import must recompute rather than trust the document.
    coordinator = source["root"]["children"][0]
    assert coordinator["id"] == "coordinator"
    coordinator["id"] = "coordinator-renamed"
    definition, warnings = _carry(source)
    mine = [w for w in warnings if w.key == "structural_fingerprint"]
    assert len(mine) == 1
    assert mine[0].code == "recomputed_divergent"
    assert mine[0].action == "recomputed"
    assert definition["structural_fingerprint"] != source["structural_fingerprint"]
    assert definition["structural_fingerprint"] == compute_structural_fingerprint(
        definition
    )


def test_tool_healing_alone_keeps_fingerprint_stable() -> None:
    # Dropping the web tool declaration forces the loader to synthesize it, but
    # the fingerprint's tools projection is {name, kind} only — the healed
    # declaration matches, so the round trip stays warning-free. Pins the
    # (deliberate) insensitivity so a projection change surfaces here.
    source = _stamped_source()
    assert source["tools"], "blueprint declares web tools"
    source["tools"] = []
    definition, warnings = _carry(source)
    assert [w for w in warnings if w.key == "structural_fingerprint"] == []
    assert definition["structural_fingerprint"] == source["structural_fingerprint"]


# ---------------------------------------------------------------------------
# 7. Exporter <-> validator parity for the contract summary
# ---------------------------------------------------------------------------


def test_summary_validator_accepts_sanitizer_output() -> None:
    contract = ResolvedToolContract(
        evidence_policy="corpus_only",
        resources=[
            ResourceContract(
                kind="vector_index",
                identity="cat.schema.idx",
                usage="required",
                access_status="verified",
                capabilities=["semantic_search"],
                domain_terms=["earnings"],
            )
        ],
        required_capabilities=["semantic_search"],
        ready_tool_kinds=["vector_search"],
        prompt_obligations=PromptObligationContract(
            required_terms=["earnings"],
            synthesis_obligations=["cite by chunk_id"],
            planner_obligations=["search the corpus first"],
            forbidden_tool_kinds=["web_search"],
        ),
    )
    populated = sanitized_resolved_tool_contract_summary(contract)
    ResolvedToolContractSummaryV1.model_validate(populated)
    stub = sanitized_resolved_tool_contract_summary(None)
    assert stub == {"schema": "resolved_tool_contract.v1", "available": False}
    ResolvedToolContractSummaryV1.model_validate(stub)


# ---------------------------------------------------------------------------
# 8. Absence rule — never synthesize
# ---------------------------------------------------------------------------


def test_raw_framework_document_gets_no_metadata() -> None:
    source = _stamped_source()
    for key in _METADATA_KEYS:
        source.pop(key, None)
    definition, warnings = _carry(source)
    assert warnings == []
    for key in _METADATA_KEYS:
        assert key not in definition


# ---------------------------------------------------------------------------
# 9. Carried metadata passes the CRUD write gate
# ---------------------------------------------------------------------------


def test_carried_metadata_passes_create_write_gate() -> None:
    """A metadata-carrying imported definition must remain creatable.

    Pins the contract documented in ``semantic_validation_errors``: the CRUD
    write gate is limited to STRUCTURAL invariants, so carried designer
    metadata (including a non-empty ``placeholder_pending_nodes`` and the
    ``required_prompt_terms`` list) never blocks agent creation. Guards the
    fail-open-on-metadata principle end to end.
    """
    from deep_research.agent_designer.edit_planning import stored_signature
    from deep_research.schemas.agent_v2 import CreateAgentV2Request

    source = _stamped_source()
    definition, warnings = _carry(source)
    assert warnings == []
    request = CreateAgentV2Request(name="imported fixture", definition=definition)
    # The validated request keeps the carried metadata intact (no projection),
    # so the edit lane's persisted-signature path stays sound after import.
    assert stored_signature(request.definition) == source["designer_signature"]
    assert request.definition["placeholder_pending_nodes"] == source[
        "placeholder_pending_nodes"
    ]
