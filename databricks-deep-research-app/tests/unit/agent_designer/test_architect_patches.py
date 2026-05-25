"""Plan v2.1 PR-3 — architect patch-only contract tests.

Tests cover three layers of the patch-only contract:

1. ``mutations.update_block`` — config-level allow-list. When
   ``DESIGNER_DETERMINISTIC_BLUEPRINT`` is ON, ``update_block`` rejects
   patches with structural keys (``body``, ``evaluator``, ``children``,
   ``subtype``, ``type``, ``pools``, ``node_id``) and unknown keys
   outside the prompt-only allow-list (``system_prompt``,
   ``user_prompt_template``, ``model_tier``, ``error_handling``,
   ``max_tool_calls``).

2. ``framework_tools._apply_architect_patches`` — patch merging into
   the immutable blueprint, with allow-list enforcement per patch and
   lane_key / subtype / node_id resolution.

3. ``ParseArchitectAstTool`` in patch-mode — reads the architect's
   final message as a patch JSON, merges it into ``state.initial_blueprint``,
   and validates the structural fingerprint hasn't drifted.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from databricks_deep_research.tools.protocol import ToolContext

from deep_research.agent_designer.blueprint import (
    DESIGNER_DETERMINISTIC_BLUEPRINT_ENV,
    build_blueprint,
    compute_structural_fingerprint,
)
from deep_research.agent_designer.framework_tools import (
    ParseArchitectAstTool,
    _apply_architect_patches,
    _flatten_node_index,
)
from deep_research.agent_designer.mutations import (
    BlockMutationError,
    update_block,
)


def _ctx() -> ToolContext:
    return ToolContext()  # type: ignore[call-arg]


def _investment_signature() -> dict[str, Any]:
    return {
        "asset_signature": "web_only",
        "retrieval_pattern": "independent_lanes",
        "question_class": "open_research",
        "primary_evidence_kind": "web_articles",
        "expected_output_shape": "structured_report",
        "independent_workstreams_count": 6,
        "lane_descriptions": [
            "fundamentals",
            "valuation",
            "risk",
            "market trends",
            "earnings",
            "competitors",
        ],
    }


def _build_test_blueprint() -> dict[str, Any]:
    return build_blueprint(_investment_signature(), "Investment analysis", [])


# ---------------------------------------------------------------------------
# update_block — config-level allow-list (flag-gated)
# ---------------------------------------------------------------------------


def test_update_block_legacy_path_accepts_subtype_patch_flag_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flag OFF: legacy semantics preserved — subtype patch accepted.

    This guards the PR-2-flag-OFF rollout invariant: existing tests
    that patch structural fields via update_block keep working until
    PR-3's flag flip activates the strictness.
    """
    monkeypatch.setenv(DESIGNER_DETERMINISTIC_BLUEPRINT_ENV, "0")
    ast = _build_test_blueprint()
    # The blueprint has a coordinator at root.children.0. Patch its subtype.
    new_ast = update_block(
        ast,
        "root.children.0",
        {"config": {"subtype": "synthesizer"}},
    )
    assert new_ast["root"]["children"][0]["config"]["subtype"] == "synthesizer"


def test_update_block_rejects_subtype_patch_when_flag_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flag ON: structural keys in config dict are rejected with
    BlockMutationError."""
    monkeypatch.setenv(DESIGNER_DETERMINISTIC_BLUEPRINT_ENV, "1")
    ast = _build_test_blueprint()
    with pytest.raises(BlockMutationError, match="subtype"):
        update_block(
            ast,
            "root.children.0",
            {"config": {"subtype": "synthesizer"}},
        )


@pytest.mark.parametrize(
    "forbidden_key", ["body", "evaluator", "children", "subtype", "type", "pools", "node_id"]
)
def test_update_block_rejects_each_forbidden_key_flag_on(
    monkeypatch: pytest.MonkeyPatch, forbidden_key: str
) -> None:
    """Every structural key in the forbidden list is rejected by name."""
    monkeypatch.setenv(DESIGNER_DETERMINISTIC_BLUEPRINT_ENV, "1")
    ast = _build_test_blueprint()
    with pytest.raises(BlockMutationError, match=forbidden_key):
        update_block(
            ast,
            "root.children.0",
            {"config": {forbidden_key: "anything"}},
        )


def test_update_block_rejects_unknown_keys_flag_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keys outside both allow-list and forbidden-list are also rejected.

    The contract is allow-listed, not deny-listed: anything not
    explicitly permitted is rejected so future architect bugs can't
    smuggle in new structural fields.
    """
    monkeypatch.setenv(DESIGNER_DETERMINISTIC_BLUEPRINT_ENV, "1")
    ast = _build_test_blueprint()
    with pytest.raises(BlockMutationError, match="custom_field"):
        update_block(
            ast,
            "root.children.0",
            {"config": {"custom_field": "anything"}},
        )


def test_update_block_accepts_allow_listed_config_keys_flag_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All prompt-only allow-list keys are accepted under flag ON."""
    monkeypatch.setenv(DESIGNER_DETERMINISTIC_BLUEPRINT_ENV, "1")
    ast = _build_test_blueprint()
    new_ast = update_block(
        ast,
        "root.children.0",
        {
            "config": {
                "system_prompt": "Customized coordinator",
                "user_prompt_template": "Updated user template",
                "model_tier": "complex",
                "error_handling": "abort",
                "max_tool_calls": 10,
            }
        },
    )
    cfg = new_ast["root"]["children"][0]["config"]
    assert cfg["system_prompt"] == "Customized coordinator"
    assert cfg["model_tier"] == "complex"


# ---------------------------------------------------------------------------
# _apply_architect_patches — patch merging
# ---------------------------------------------------------------------------


def test_apply_patches_no_changes_returns_identical_fingerprint() -> None:
    """An empty patch dict leaves the blueprint structurally identical."""
    blueprint = _build_test_blueprint()
    merged, errors = _apply_architect_patches(blueprint, {})
    assert errors == []
    assert compute_structural_fingerprint(merged) == compute_structural_fingerprint(blueprint)


def test_apply_patches_by_lane_key_merges_prompt_only() -> None:
    """A valid prompt-only patch keyed by lane_key merges into config."""
    blueprint = _build_test_blueprint()
    lane_keys = blueprint["lane_keys"]
    first_key = next(iter(lane_keys.keys()))
    patches = {
        first_key: {
            "system_prompt": "Customized fundamentals researcher",
            "user_prompt_template": "Investigate fundamentals: {{ query }}",
        }
    }
    merged, errors = _apply_architect_patches(blueprint, patches)
    assert errors == []
    # Find the matching lane researcher node and verify the patch landed
    index = _flatten_node_index(merged)
    lane_node = index.get(first_key)
    assert lane_node is not None
    assert lane_node["config"]["system_prompt"] == "Customized fundamentals researcher"
    # Fingerprint unchanged: prompts are NOT in the structural projection
    assert compute_structural_fingerprint(merged) == compute_structural_fingerprint(blueprint)


def test_apply_patches_by_subtype_targets_singleton_role() -> None:
    """Patching by subtype (e.g., 'synthesizer') matches the singleton role."""
    blueprint = _build_test_blueprint()
    patches = {
        "synthesizer": {"system_prompt": "Customized synthesizer"},
    }
    merged, errors = _apply_architect_patches(blueprint, patches)
    assert errors == []
    index = _flatten_node_index(merged)
    synth = index.get("synthesizer")
    assert synth is not None
    assert synth["config"]["system_prompt"] == "Customized synthesizer"


def test_apply_patches_rejects_structural_keys() -> None:
    """Patches containing structural keys are rejected (errors emitted)."""
    blueprint = _build_test_blueprint()
    first_key = next(iter(blueprint["lane_keys"].keys()))
    patches = {first_key: {"subtype": "planner"}}
    _, errors = _apply_architect_patches(blueprint, patches)
    assert errors
    assert any("structural_drift_detected" in e for e in errors)


def test_apply_patches_rejects_unknown_target() -> None:
    """Patch keys not matching any node/subtype/lane_key are rejected."""
    blueprint = _build_test_blueprint()
    patches = {
        "no_such_node": {"system_prompt": "ignored"},
    }
    _, errors = _apply_architect_patches(blueprint, patches)
    assert errors
    assert any("no matching node" in e for e in errors)


def test_apply_patches_rejects_unknown_config_keys() -> None:
    """Keys outside the architect allow-list are rejected."""
    blueprint = _build_test_blueprint()
    first_key = next(iter(blueprint["lane_keys"].keys()))
    patches = {first_key: {"some_random_field": "x"}}
    _, errors = _apply_architect_patches(blueprint, patches)
    assert errors
    assert any("unknown" in e.lower() or "allow-list" in e for e in errors)


def test_apply_patches_non_dict_patch_rejected() -> None:
    """A patch value that isn't a dict is rejected with a clear error."""
    blueprint = _build_test_blueprint()
    first_key = next(iter(blueprint["lane_keys"].keys()))
    patches = {first_key: "this is a string not a dict"}  # type: ignore[dict-item]
    _, errors = _apply_architect_patches(blueprint, patches)
    assert errors
    assert any("not a dict" in e for e in errors)


def test_apply_patches_does_not_mutate_original_blueprint() -> None:
    """Patch application returns a deep copy; the original is unchanged."""
    blueprint = _build_test_blueprint()
    first_key = next(iter(blueprint["lane_keys"].keys()))
    snapshot_fp = compute_structural_fingerprint(blueprint)
    patches = {first_key: {"system_prompt": "won't leak back"}}
    _apply_architect_patches(blueprint, patches)
    # Original blueprint untouched
    assert compute_structural_fingerprint(blueprint) == snapshot_fp


# ---------------------------------------------------------------------------
# _flatten_node_index — addressing
# ---------------------------------------------------------------------------


def test_flatten_index_finds_lane_researchers_by_lane_key() -> None:
    blueprint = _build_test_blueprint()
    index = _flatten_node_index(blueprint)
    for lane_key in blueprint["lane_keys"]:
        assert lane_key in index, f"missing lane_key {lane_key} in index"


def test_flatten_index_finds_synthesizer_by_subtype() -> None:
    blueprint = _build_test_blueprint()
    index = _flatten_node_index(blueprint)
    assert "synthesizer" in index


def test_flatten_index_finds_node_by_literal_id() -> None:
    """The index also resolves nodes by their literal ``id`` field, so
    the architect can target lane researchers by ``lane_N-researcher``
    if it prefers that addressing.
    """
    blueprint = _build_test_blueprint()
    index = _flatten_node_index(blueprint)
    assert "lane_1-researcher" in index


# ---------------------------------------------------------------------------
# ParseArchitectAstTool patch mode (integration)
# ---------------------------------------------------------------------------


def test_parse_architect_ast_patch_mode_happy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end patch mode: flag ON, valid patches → merged AST returned."""
    monkeypatch.setenv(DESIGNER_DETERMINISTIC_BLUEPRINT_ENV, "1")
    blueprint = _build_test_blueprint()
    expected_fp = blueprint["structural_fingerprint"]
    first_key = next(iter(blueprint["lane_keys"].keys()))
    patches_json = json.dumps(
        {
            "node_patches": {
                first_key: {
                    "system_prompt": "patched fundamentals researcher",
                }
            }
        }
    )
    raw = f"Here is my patch:\n```json\n{patches_json}\n```\n"

    tool = ParseArchitectAstTool(
        blueprint_getter=lambda: blueprint,
        fingerprint_getter=lambda: expected_fp,
    )
    result = asyncio.run(tool.execute({"raw_message": raw}, _ctx()))
    assert result.success is not False
    data = result.data or {}
    assert data["parse_ok"] is True
    assert data["parse_mode"] == "patches"
    assert data["structural_fingerprint"] == expected_fp


def test_parse_architect_ast_patch_mode_rejects_structural_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plan v2.1 M2: an architect that emits a structural key in a patch
    cannot land it — patch_errors carries structural_drift_detected and
    state.current_ast reverts to the immutable blueprint.
    """
    monkeypatch.setenv(DESIGNER_DETERMINISTIC_BLUEPRINT_ENV, "1")
    blueprint = _build_test_blueprint()
    first_key = next(iter(blueprint["lane_keys"].keys()))
    patches_json = json.dumps(
        {"node_patches": {first_key: {"subtype": "planner"}}}
    )
    raw = f"```json\n{patches_json}\n```"

    tool = ParseArchitectAstTool(
        blueprint_getter=lambda: blueprint,
        fingerprint_getter=lambda: blueprint["structural_fingerprint"],
    )
    result = asyncio.run(tool.execute({"raw_message": raw}, _ctx()))
    data = result.data or {}
    assert data["parse_ok"] is False
    assert "structural_drift_detected" in (data.get("patch_errors") or [""])[0] or \
        "structural_drift_detected" in str(data.get("error", ""))
    # Reverted to immutable blueprint
    assert data["current_ast"] is blueprint or data["current_ast"] == blueprint


def test_parse_architect_ast_patch_mode_revert_on_unknown_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(DESIGNER_DETERMINISTIC_BLUEPRINT_ENV, "1")
    blueprint = _build_test_blueprint()
    patches_json = json.dumps(
        {
            "node_patches": {
                "no_such_node": {"system_prompt": "x"},
            }
        }
    )
    raw = f"```json\n{patches_json}\n```"

    tool = ParseArchitectAstTool(
        blueprint_getter=lambda: blueprint,
        fingerprint_getter=lambda: blueprint["structural_fingerprint"],
    )
    result = asyncio.run(tool.execute({"raw_message": raw}, _ctx()))
    data = result.data or {}
    assert data["parse_ok"] is False
    assert "no matching node" in str(data.get("error") or "")


def test_parse_architect_ast_falls_back_to_legacy_when_flag_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flag OFF: legacy AST-extraction path runs unchanged."""
    monkeypatch.setenv(DESIGNER_DETERMINISTIC_BLUEPRINT_ENV, "0")
    # A minimal valid AST in fenced JSON form — legacy parse should
    # succeed and never touch the blueprint path.
    minimal_ast = {
        "id": "wf",
        "name": "test",
        "root": {
            "id": "root",
            "type": "agent",
            "label": "agent",
            "config": {"subtype": "coordinator"},
        },
    }
    ast_json = json.dumps(minimal_ast)
    raw = f"```json\n{ast_json}\n```"

    blueprint_called = False

    def _blueprint_getter() -> Any:
        nonlocal blueprint_called
        blueprint_called = True
        return None

    tool = ParseArchitectAstTool(
        blueprint_getter=_blueprint_getter,
    )
    result = asyncio.run(tool.execute({"raw_message": raw}, _ctx()))
    data = result.data or {}
    # Legacy path returns parse_ok with current_ast keyed (no parse_mode field).
    assert data.get("parse_ok") is True
    assert "parse_mode" not in data
    # The blueprint_getter is never called in flag-OFF mode (path short-circuits
    # in _patch_mode_result before reading blueprint).
    assert blueprint_called is False


def test_parse_architect_ast_no_blueprint_falls_back_to_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flag ON but blueprint missing (build_blueprint didn't run): the
    tool falls back to the legacy path rather than crashing.
    """
    monkeypatch.setenv(DESIGNER_DETERMINISTIC_BLUEPRINT_ENV, "1")
    minimal_ast = {
        "id": "wf",
        "name": "test",
        "root": {
            "id": "root",
            "type": "agent",
            "label": "agent",
            "config": {"subtype": "coordinator"},
        },
    }
    raw = f"```json\n{json.dumps(minimal_ast)}\n```"
    tool = ParseArchitectAstTool(blueprint_getter=lambda: None)
    result = asyncio.run(tool.execute({"raw_message": raw}, _ctx()))
    data = result.data or {}
    # Legacy path took over.
    assert data.get("parse_ok") is True
    assert "parse_mode" not in data


def test_parse_architect_ast_blueprint_as_json_string_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """State backends may serialize the blueprint as a JSON string."""
    monkeypatch.setenv(DESIGNER_DETERMINISTIC_BLUEPRINT_ENV, "1")
    blueprint = _build_test_blueprint()
    blueprint_str = json.dumps(blueprint)
    first_key = next(iter(blueprint["lane_keys"].keys()))
    patches_json = json.dumps(
        {"node_patches": {first_key: {"system_prompt": "x"}}}
    )
    raw = f"```json\n{patches_json}\n```"

    tool = ParseArchitectAstTool(
        blueprint_getter=lambda: blueprint_str,
        fingerprint_getter=lambda: blueprint["structural_fingerprint"],
    )
    result = asyncio.run(tool.execute({"raw_message": raw}, _ctx()))
    data = result.data or {}
    assert data["parse_ok"] is True
    assert data["parse_mode"] == "patches"


def test_parse_architect_ast_rejects_unknown_top_level_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plan v2.1 generic-robustness — ``tool_bindings`` (or any other key
    not in ``_TOP_LEVEL_PATCH_ALLOW_LIST``) at the top of the patch
    document is rejected explicitly. Historically the parser silently
    dropped these, which let architects believe their tool changes had
    landed."""
    monkeypatch.setenv(DESIGNER_DETERMINISTIC_BLUEPRINT_ENV, "1")
    blueprint = _build_test_blueprint()
    first_key = next(iter(blueprint["lane_keys"].keys()))
    payload = {
        "node_patches": {
            first_key: {"system_prompt": "specialized prompt"},
        },
        "tool_bindings": {first_key: ["vector_search"]},
    }
    raw = f"```json\n{json.dumps(payload)}\n```"

    tool = ParseArchitectAstTool(
        blueprint_getter=lambda: blueprint,
        fingerprint_getter=lambda: blueprint["structural_fingerprint"],
    )
    result = asyncio.run(tool.execute({"raw_message": raw}, _ctx()))
    data = result.data or {}
    assert data["parse_ok"] is False
    error = str(data.get("error") or "")
    assert "tool_bindings" in error, (
        f"error must name the rejected unknown key; got: {error}"
    )
    assert "request_signature_revision" in error, (
        "rejection must guide the architect at the correct fix"
    )


def test_parse_architect_ast_accepts_node_patches_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sanity: with ONLY ``node_patches`` at the top level, parsing succeeds."""
    monkeypatch.setenv(DESIGNER_DETERMINISTIC_BLUEPRINT_ENV, "1")
    blueprint = _build_test_blueprint()
    first_key = next(iter(blueprint["lane_keys"].keys()))
    payload = {
        "node_patches": {
            first_key: {"system_prompt": "specialized prompt"},
        }
    }
    raw = f"```json\n{json.dumps(payload)}\n```"

    tool = ParseArchitectAstTool(
        blueprint_getter=lambda: blueprint,
        fingerprint_getter=lambda: blueprint["structural_fingerprint"],
    )
    result = asyncio.run(tool.execute({"raw_message": raw}, _ctx()))
    data = result.data or {}
    assert data["parse_ok"] is True
    assert data["parse_mode"] == "patches"
