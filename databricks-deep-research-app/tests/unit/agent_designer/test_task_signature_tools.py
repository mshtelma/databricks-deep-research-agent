"""PR3-B Layer 1 — EmitTaskSignatureTool + SelectTopologyTool tests."""
from __future__ import annotations

import json

import pytest
from databricks_deep_research.tools.protocol import ToolContext

from deep_research.agent_designer.framework_tools import (
    EmitTaskSignatureTool,
    SelectTopologyTool,
    builtin_designer_tools,
)


@pytest.fixture
def ctx() -> ToolContext:
    return ToolContext(query="")


def _valid_signature_payload() -> dict[str, object]:
    """Fresh-emission payload that satisfies the strict
    ``TaskSignature.from_classifier_emission`` contract.

    Includes all five Plan v2.1 structural axes plus the legacy fields.
    ``EmitTaskSignatureTool`` rejects payloads missing any structural axis.
    """
    return {
        "asset_signature": "corpus_only",
        "retrieval_pattern": "pipelined_retrieve_read_compute",
        "question_class": "numeric_aggregation",
        "question_ambiguity": ["period_basis"],
        "primary_evidence_kind": "structured_tables",
        "expected_output_shape": "single_number",
        "step_dependencies_present": True,
        "independent_workstreams_count": 1,
        "iteration_required": False,
        "output_aggregation_kind": "single_answer",
        "lane_descriptions": ["retrieve-read-compute pipeline"],
    }


async def test_emit_task_signature_writes_state_and_returns_payload(
    ctx: ToolContext,
) -> None:
    written: list[object] = []
    tool = EmitTaskSignatureTool(state_setter=written.append)
    result = await tool.execute(_valid_signature_payload(), ctx)
    body = json.loads(result.content)
    assert body["asset_signature"] == "corpus_only"
    assert body["retrieval_pattern"] == "pipelined_retrieve_read_compute"
    assert body["question_ambiguity"] == ["period_basis"]
    # state setter received the same payload:
    assert written and written[0] == body


async def test_emit_task_signature_rejects_invalid_payload(ctx: ToolContext) -> None:
    tool = EmitTaskSignatureTool()
    bad = _valid_signature_payload()
    bad["asset_signature"] = "not_a_real_signature"
    result = await tool.execute(bad, ctx)
    body = json.loads(result.content)
    assert body.get("ok") is False or "error" in body


async def test_select_topology_uses_argument_signature(ctx: ToolContext) -> None:
    tool = SelectTopologyTool()
    result = await tool.execute({"signature": _valid_signature_payload()}, ctx)
    body = json.loads(result.content)
    assert body["topology"] == "plan_and_execute"
    assert body["signature"]["retrieval_pattern"] == "pipelined_retrieve_read_compute"


def _legacy_storage_payload() -> dict[str, object]:
    """Legacy 7-field payload with structural axes at their storage
    defaults (empty lane_descriptions, count=1, deps=False, iter=False).

    Used to exercise the ``select_topology`` retrieval_pattern fallback
    path. ``SelectTopologyTool.execute`` runs ``load_from_storage`` so
    this payload parses; the strict ``EmitTaskSignatureTool`` would
    reject it (missing structural axes).
    """
    return {
        "asset_signature": "corpus_only",
        "retrieval_pattern": "pipelined_retrieve_read_compute",
        "question_class": "numeric_aggregation",
        "question_ambiguity": ["period_basis"],
        "primary_evidence_kind": "structured_tables",
        "expected_output_shape": "single_number",
    }


async def test_select_topology_reads_state_when_no_argument(ctx: ToolContext) -> None:
    payload = _legacy_storage_payload()
    payload["retrieval_pattern"] = "independent_lanes"
    tool = SelectTopologyTool(state_getter=lambda: payload)
    result = await tool.execute({}, ctx)
    body = json.loads(result.content)
    assert body["topology"] == "parallel_lanes"


async def test_select_topology_errors_with_no_signature(ctx: ToolContext) -> None:
    tool = SelectTopologyTool()
    result = await tool.execute({}, ctx)
    body = json.loads(result.content)
    assert body.get("ok") is False or "error" in body


async def test_select_topology_writes_topology_via_state_setter(
    ctx: ToolContext,
) -> None:
    written: list[object] = []
    tool = SelectTopologyTool(state_setter=written.append)
    payload = _legacy_storage_payload()
    payload["retrieval_pattern"] = "bounded_lookup"
    payload["question_class"] = "bounded_lookup"
    await tool.execute({"signature": payload}, ctx)
    assert written == ["single_agent"]


def test_builtin_designer_tools_registers_layer1_tools() -> None:
    names = {t.definition.name for t in builtin_designer_tools()}
    assert "emit_task_signature" in names
    assert "select_topology" in names


# ---------------------------------------------------------------------------
# Fix H — regression tests for the strict-emission contract.
#
# These guard the contract changes that fix the live investment_research
# scaffold-and-run failure: the classifier's tool schema must expose the
# Plan v2.1 structural axes; the strict path must reject partial payloads;
# the lenient path must keep legacy storage parsing.
# ---------------------------------------------------------------------------


def test_emit_task_signature_tool_schema_matches_model() -> None:
    """Auto-generated tool schema covers every TaskSignature field and
    marks the five structural axes as required."""
    from deep_research.agent_designer.task_signature import TaskSignature

    tool = EmitTaskSignatureTool()
    params = tool.definition.parameters
    props = params.get("properties") or {}
    required = set(params.get("required") or [])

    # Every model field appears in the tool schema (defends against
    # silent schema/model drift).
    for name in TaskSignature.model_fields:
        assert name in props, f"{name!r} missing from tool schema"

    # Structural axes are required by the strict-emission contract.
    for axis in (
        "step_dependencies_present",
        "independent_workstreams_count",
        "iteration_required",
        "output_aggregation_kind",
        "lane_descriptions",
    ):
        assert axis in required, f"{axis!r} not in required set"

    # axis_reasoning is optional (default=None) so the LLM may omit it.
    assert "axis_reasoning" not in required

    # No anyOf survives post-processing — Databricks tool APIs reject anyOf
    # in parameter schemas.
    for name, prop in props.items():
        assert "anyOf" not in prop, f"{name!r} retained anyOf {prop!r}"


async def test_emit_task_signature_rejects_missing_structural_axes(
    ctx: ToolContext,
) -> None:
    """Partial classifier emission (legacy 7-field payload) is rejected."""
    tool = EmitTaskSignatureTool()
    legacy = {
        "asset_signature": "corpus_only",
        "retrieval_pattern": "pipelined_retrieve_read_compute",
        "question_class": "numeric_aggregation",
        "primary_evidence_kind": "structured_tables",
        "expected_output_shape": "single_number",
    }
    result = await tool.execute(legacy, ctx)
    body = json.loads(result.content)
    assert "error" in body or body.get("ok") is False
    assert "Field required" in str(body) or "missing" in str(body).lower()


async def test_emit_task_signature_rejects_lane_count_mismatch(
    ctx: ToolContext,
) -> None:
    """Cross-field rule: len(lane_descriptions) == max(count, 1)."""
    tool = EmitTaskSignatureTool()
    bad = _valid_signature_payload()
    bad["independent_workstreams_count"] = 6
    bad["lane_descriptions"] = ["only one"]  # 1 != 6
    result = await tool.execute(bad, ctx)
    body = json.loads(result.content)
    assert "error" in body or body.get("ok") is False
    assert "lane_descriptions" in str(body)


def test_load_from_storage_fills_legacy_defaults() -> None:
    """Lenient path pre-fills the five structural-axis defaults for
    payloads that predate the v2.1 extension."""
    from deep_research.agent_designer.task_signature import TaskSignature

    sig = TaskSignature.load_from_storage({
        "asset_signature": "corpus_only",
        "retrieval_pattern": "pipelined_retrieve_read_compute",
        "question_class": "numeric_aggregation",
        "primary_evidence_kind": "structured_tables",
        "expected_output_shape": "single_number",
    })
    assert sig.step_dependencies_present is False
    assert sig.independent_workstreams_count == 1
    assert sig.iteration_required is False
    assert sig.output_aggregation_kind == "single_answer"
    assert sig.lane_descriptions == []
    assert sig.axis_reasoning is None


def test_parse_architect_ast_uses_blueprint_when_available() -> None:
    """Fix D — ParseArchitectAstTool must receive the blueprint+fingerprint
    getters from the builtin registry so patch mode actually fires."""
    from deep_research.agent_designer.framework_tools import (
        ParseArchitectAstTool,
        builtin_designer_tools,
    )

    blueprint = {"id": "x", "name": "y", "root": {"type": "sequence"}, "lane_keys": {}}
    fingerprint = "deadbeef"

    tools = builtin_designer_tools(
        blueprint_getter=lambda: blueprint,
        fingerprint_getter=lambda: fingerprint,
    )
    parse = next(t for t in tools if t.definition.name == "parse_architect_ast")
    assert isinstance(parse, ParseArchitectAstTool)
    # The closure references the wired getters — patch mode reads from them.
    assert parse._blueprint_getter() is blueprint  # type: ignore[union-attr]
    assert parse._fingerprint_getter() == fingerprint  # type: ignore[union-attr]


def test_architect_prompt_omits_propose_workflow_imperatives() -> None:
    """Fix E — the architect's compiled prompts must not instruct the LLM
    to call ``propose_workflow`` or emit a full AST. Those instructions
    contradicted the PR-3 patch-only contract and caused the architect
    to thrash against REACT_TOOL_RESTRICTED in the failing scaffold-and-run."""
    import re
    from pathlib import Path

    import yaml

    yaml_path = (
        Path(__file__).parent.parent.parent.parent
        / "src/deep_research/agent_designer/designer_workflow.yaml"
    )
    raw = yaml.safe_load(yaml_path.read_text())

    def find_node(node: object, target: str) -> object:
        if isinstance(node, dict):
            if node.get("id") == target:
                return node
            for v in node.values():
                r = find_node(v, target)
                if r is not None:
                    return r
        elif isinstance(node, list):
            for it in node:
                r = find_node(it, target)
                if r is not None:
                    return r
        return None

    architect = find_node(raw, "architect")
    assert isinstance(architect, dict)
    cfg = architect["config"]
    text = (cfg.get("system_prompt") or "") + "\n" + (cfg.get("user_prompt_template") or "")
    forbidden = [
        r"Call the propose_workflow",
        r"call ``propose_workflow``",
        r"FORWARD the task_signature",
        r"Produce the complete revised WorkflowDefinition AST",
    ]
    for pat in forbidden:
        assert not re.search(pat, text), f"architect prompt still contains: {pat!r}"
